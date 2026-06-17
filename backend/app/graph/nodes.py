import re
import uuid
from datetime import datetime
from time import perf_counter
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

from app.config import settings
from app.graph.prompts import ANSWER_SYSTEM, DIRECT_ANSWER_SYSTEM, GREETING_SYSTEM, RETRIEVE_HINT, TOP_K
from app.graph.state import AgentState, ExecutionTrace, RetrievedChunk
from app.observability import record_answer_validation, record_retrieval_metrics
from app.rag.hybrid_retriever import hybrid_retrieve
from app.rag.kb_match import is_chunk_relevant_to_question
from app.rag.retrieval_optimizer import (
    compress_lines,
    expected_answer_aspects,
    extract_relevant_facts,
    infer_answer_type,
    truncate_by_budget,
    validate_answer_grounding,
)
from app.task_store import append_task_event, get_task

_GREETING_RE = re.compile(
    r"^(你好|您好|嗨|哈喽|hello|hi|hey|早上好|下午好|晚上好|在吗|同学你好)[\s!！?？。.~、，]*$",
    re.IGNORECASE,
)


def is_greeting(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 24:
        return False
    return bool(_GREETING_RE.match(t))


class _FallbackChatModel:
    def __init__(self, model: str, api_key: str, api_base: str, temperature: float) -> None:
        self.model = model
        self.temperature = temperature

    def _extract_question(self, text: str) -> str:
        match = re.search(r"用户问题：(.+)", text)
        return match.group(1).strip() if match else text.strip()

    def _extract_facts(self, text: str) -> list[str]:
        if "可用事实：" not in text:
            return []
        block = text.split("可用事实：", 1)[1]
        for marker in ("图谱补充关系：", "补充资料摘录：", "请仅依据可用事实组织回答"):
            if marker in block:
                block = block.split(marker, 1)[0]
        facts: list[str] = []
        for line in block.splitlines():
            cleaned = re.sub(r"^\s*\d+[.、]\s*", "", line).strip()
            if cleaned:
                facts.append(cleaned)
        return facts[:4]

    def invoke(self, messages: list[Any]) -> AIMessage:
        last_message = messages[-1] if messages else HumanMessage(content="")
        text = getattr(last_message, "content", str(last_message))
        question = self._extract_question(text)
        if is_greeting(question):
            return AIMessage(content="你好，我在。当前环境未连接外部模型服务，但我仍可以基于本地知识库继续协助你。")
        facts = self._extract_facts(text)
        if facts:
            lines = "\n".join(f"{idx}. {fact}" for idx, fact in enumerate(facts, start=1))
            return AIMessage(content=f"关于“{question}”，根据当前本地资料可先得到：\n{lines}")
        return AIMessage(content=f"当前环境未连接外部模型服务，我先基于本地规则给出简要答复：{question}")


def _resolve_model_name(answer_type: str | None = None, *, greeting: bool = False) -> str:
    if greeting and settings.openai_model_greeting:
        return settings.openai_model_greeting
    mapping = {
        "definition": settings.openai_model_definition,
        "process": settings.openai_model_process,
        "comparison": settings.openai_model_comparison,
        "analysis": settings.openai_model_analysis,
        "advice": settings.openai_model_advice,
        "fact": settings.openai_model_fact,
    }
    candidate = str(mapping.get(str(answer_type or "").strip()) or "").strip()
    return candidate or settings.openai_model


def get_llm(answer_type: str | None = None, *, greeting: bool = False) -> Any:
    model_name = _resolve_model_name(answer_type, greeting=greeting)
    if ChatOpenAI is not None:
        return ChatOpenAI(
            model=model_name,
            openai_api_key=settings.openai_api_key or "dummy",
            openai_api_base=settings.openai_api_base,
            temperature=0.3,
        )
    return _FallbackChatModel(
        model=model_name,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        temperature=0.3,
    )


def _doc_to_chunk(doc: Document, index: int) -> RetrievedChunk:
    meta = doc.metadata or {}
    return RetrievedChunk(
        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{meta.get('source', '')}-{index}-{doc.page_content[:80]}")),
        content=doc.page_content,
        source=meta.get("source", "unknown"),
        file_type=meta.get("file_type", "unknown"),
        score=meta.get("score"),
        subject=meta.get("subject"),
        chapter=meta.get("chapter"),
        retrieval_mode=meta.get("retrieval_mode"),
        concepts=list(meta.get("concepts") or []),
        rank_score=meta.get("rank_score"),
        coverage_score=meta.get("coverage_score"),
    )


def _question_from_state(state: AgentState) -> str:
    question = state.get("plan_question") or state.get("question") or ""
    if not question and state.get("messages"):
        last = state["messages"][-1]
        question = getattr(last, "content", str(last))
    return question


def _original_question(state: AgentState) -> str:
    return state.get("question") or _question_from_state(state)


def _dedup_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        values.append(text)
    return values


def _default_validation(question: str = "", answer_type: str | None = None) -> dict:
    resolved_type = answer_type or infer_answer_type(question)
    return {
        "grounded": False,
        "grounding_score": 0.0,
        "reference_overlap": 0.0,
        "question_overlap": 0.0,
        "citation_coverage": 0.0,
        "supported_claims": 0,
        "unsupported_claims": 0,
        "weak_sentences": [],
        "answer_type": resolved_type,
        "aspect_coverage": 0.0,
        "missing_aspects": expected_answer_aspects(question, resolved_type),
        "fact_coverage": 0.0,
        "used_facts": 0,
    }


def _task_id(state: AgentState) -> str:
    return str(state.get("task_id") or state.get("thread_id") or "").strip()


def _task_timed_out(state: AgentState) -> bool:
    task_id = _task_id(state)
    if not task_id:
        return False
    task = get_task(task_id)
    if not task:
        return False
    created_at = str(task.get("created_at") or "").strip()
    if not created_at:
        return False
    try:
        started = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    elapsed = (datetime.utcnow() - started.replace(tzinfo=None)).total_seconds()
    return elapsed > settings.graph_task_timeout_seconds


def _task_cancelled(state: AgentState) -> bool:
    task_id = _task_id(state)
    if not task_id:
        return False
    task = get_task(task_id)
    if not task:
        return False
    return str(task.get("status") or "").strip().lower() == "cancelled"


def _cancelled_result(node_name: str, step: int) -> dict[str, Any]:
    return {
        "task_status": "cancelled",
        "task_error_code": "TASK_CANCELLED",
        "critic_decision": "end",
        "critic_reason": "任务已取消",
        "critic_reason_code": "TASK_CANCELLED",
        "trace_message": f"任务已取消，停止执行节点：{node_name}",
        "trace_data": {"step": step},
    }


def _run_node(node_name: str, state: AgentState, fn):
    started = perf_counter()
    task_id = _task_id(state)
    step = int(state.get("loop_step") or 0)
    if _task_cancelled(state):
        result = _cancelled_result(node_name, step)
        trace = ExecutionTrace(
            node=node_name,
            status="cancelled",
            message=result["trace_message"],
            step=step,
            elapsed_ms=0,
            data=dict(result.get("trace_data") or {}),
        )
        if task_id:
            append_task_event(task_id, "node_cancelled", message=trace["message"], node=node_name, status="cancelled", data=trace["data"])
        result["execution_trace"] = [trace]
        result.pop("trace_message", None)
        result.pop("trace_data", None)
        return result
    if task_id:
        append_task_event(task_id, "node_started", message=f"节点开始执行：{node_name}", node=node_name, status=state.get("task_status") or "running", data={"step": step})
    try:
        result = fn()
        elapsed_ms = int((perf_counter() - started) * 1000)
        if _task_cancelled(state):
            result = {**dict(result), **_cancelled_result(node_name, int(result.get("loop_step") or step))}
        trace = ExecutionTrace(
            node=node_name,
            status=str(result.get("task_status") or state.get("task_status") or "running"),
            message=str(result.get("trace_message") or f"节点执行完成：{node_name}"),
            step=int(result.get("loop_step") or step),
            elapsed_ms=elapsed_ms,
            data=dict(result.get("trace_data") or {}),
        )
        if task_id:
            event_type = "node_cancelled" if trace["status"] == "cancelled" else "node_completed"
            append_task_event(
                task_id,
                event_type,
                message=trace["message"],
                node=node_name,
                status=trace["status"],
                data=trace["data"],
            )
        result["execution_trace"] = [trace]
        result.pop("trace_message", None)
        result.pop("trace_data", None)
        return result
    except Exception as exc:
        elapsed_ms = int((perf_counter() - started) * 1000)
        if task_id:
            append_task_event(
                task_id,
                "node_failed",
                message=f"节点执行失败：{node_name}",
                node=node_name,
                status="failed",
                data={"error": str(exc), "step": step, "elapsed_ms": elapsed_ms},
            )
        raise


def _selected_primary_and_support(chunks: list[dict], selected_ids: list[str]) -> tuple[dict | None, list[dict]]:
    primary = None
    if selected_ids:
        sid = selected_ids[0]
        primary = next((chunk for chunk in chunks if chunk.get("id") == sid), None)
    if primary is None and chunks:
        primary = next((chunk for chunk in chunks if chunk.get("file_type") != "graph"), chunks[0])
    support: list[dict] = []
    for chunk in chunks:
        if not primary or chunk.get("id") == primary.get("id"):
            continue
        support.append(chunk)
        if len(support) >= 3:
            break
    return primary, support


def _format_chunk_context(chunk: dict, *, label: str) -> str:
    concepts = "、".join(chunk.get("concepts") or [])
    header = f"[{label}][来源: {chunk['source']} | 类型: {chunk['file_type']}]"
    if chunk.get("subject"):
        header += f"[学科: {chunk['subject']}]"
    if chunk.get("chapter"):
        header += f"[章节: {chunk['chapter']}]"
    if concepts:
        header += f"[概念: {concepts}]"
    return f"{header}\n{truncate_by_budget(chunk['content'], settings.retrieval_chunk_budget_tokens)}"


def _build_query_rewrites(
    question: str,
    answer_type: str,
    aspects: list[str],
    retry_strategy: str,
    matched_concepts: list[str],
    related_concepts: list[str],
) -> list[str]:
    suffix_map = {
        "definition": "定义 核心要点 例子",
        "process": "步骤 条件 注意事项",
        "comparison": "对比 区别 结论",
        "analysis": "原因 依据 结论",
        "advice": "建议 步骤 注意事项",
        "fact": "关键事实 直接答案",
    }
    base_queries = [question]
    suffix = suffix_map.get(answer_type, "关键事实")
    if suffix:
        base_queries.append(f"{question} {suffix}")
    if aspects:
        base_queries.append(f"{question} {' '.join(aspects[:3])}")
    if matched_concepts:
        base_queries.append(f"{question} {' '.join(matched_concepts[:3])}")
    if related_concepts:
        base_queries.append(f"{question} {' '.join(related_concepts[:3])}")
    if retry_strategy == "widen_retrieval":
        base_queries.append(f"{question} {' '.join(_dedup_strings([*matched_concepts[:3], *related_concepts[:3], *aspects[:2]]))}")
    elif retry_strategy == "focus_coverage":
        base_queries.append(f"{question} {' '.join(aspects[:3])} 关键条件 关键结论")
    elif retry_strategy == "query_rewrite":
        base_queries.append(f"{suffix} {question}")
    return _dedup_strings(base_queries)[:4]


def _selection_score(chunk: dict) -> float:
    rank_score = float(chunk.get("rank_score") or 0.0)
    coverage_score = float(chunk.get("coverage_score") or 0.0)
    semantic_score = 0.0
    score = chunk.get("score")
    if isinstance(score, (int, float)):
        semantic_score = max(0.0, 1.2 - float(score))
    return round(rank_score + coverage_score + semantic_score * 0.2, 4)


def _choose_evidence(chunks: list[dict], answer_type: str) -> tuple[list[str], bool, float]:
    candidates = [chunk for chunk in chunks if chunk.get("file_type") != "graph"] or list(chunks)
    if not candidates:
        return [], False, 0.0
    candidates = sorted(candidates, key=_selection_score, reverse=True)
    top = candidates[0]
    top_score = _selection_score(top)
    next_score = _selection_score(candidates[1]) if len(candidates) > 1 else 0.0
    gap = round(max(0.0, top_score - next_score), 4)
    threshold = settings.graph_auto_select_complex_gap if answer_type in {"process", "comparison", "analysis"} else settings.graph_auto_select_min_score_gap
    low_top_confidence = top_score < 0.35 and len(candidates) > 1
    requires_human = len(candidates) > 1 and (gap < threshold or low_top_confidence)
    confidence = round(min(1.0, max(0.05, 0.52 + gap)), 4)
    return [str(top.get("id") or "")], requires_human, confidence


def _answer_style_instruction(answer_type: str, aspects: list[str]) -> str:
    if answer_type == "definition":
        return f"先解释定义，再覆盖 {'、'.join(aspects)}。"
    if answer_type == "process":
        return f"按顺序写步骤，并补充 {'、'.join(aspects[1:])}。"
    if answer_type == "comparison":
        return f"按对比维度组织，并明确 {'、'.join(aspects[1:])}。"
    if answer_type == "analysis":
        return f"先给结论，再说明 {'、'.join(aspects[1:])}。"
    if answer_type == "advice":
        return f"给出可执行建议，并覆盖 {'、'.join(aspects[1:])}。"
    return f"直接回答问题，并补充 {'、'.join(aspects)}。"


def _build_answer_prompt(question: str, answer_type: str, aspects: list[str], facts: list[str], primary: dict, support_chunks: list[dict], graph_context: str) -> str:
    evidence_lines = []
    if primary:
        evidence_lines.append(f"主证据来源：{primary.get('source')}")
    if support_chunks:
        evidence_lines.append(f"辅助证据来源：{'、'.join(str(chunk.get('source') or '') for chunk in support_chunks if chunk.get('source'))}")
    evidence_block = "\n".join(evidence_lines)
    fact_block = "\n".join(f"{idx}. {fact}" for idx, fact in enumerate(facts, start=1)) if facts else "1. 资料中缺少足够可复述事实，请谨慎作答。"
    prompt = (
        f"用户问题：{question}\n"
        f"问题类型：{answer_type}\n"
        f"必须覆盖：{'、'.join(aspects) if aspects else '直接回答'}\n"
        f"作答要求：{_answer_style_instruction(answer_type, aspects)}\n"
    )
    if evidence_block:
        prompt += f"{evidence_block}\n"
    prompt += f"\n可用事实：\n{fact_block}"
    if graph_context and primary.get("file_type") != "graph":
        prompt += f"\n\n图谱补充关系：\n{graph_context}"
    prompt += "\n\n请仅依据可用事实组织回答，不要扩展没有证据支持的新结论。"
    return prompt


def greeting_node(state: AgentState) -> dict:
    def _work() -> dict:
        question = _original_question(state)
        answer_type = infer_answer_type(question)
        model_name = _resolve_model_name(answer_type, greeting=True)
        llm = get_llm(answer_type, greeting=True)
        response = llm.invoke([
            SystemMessage(content=GREETING_SYSTEM),
            HumanMessage(content=question or "你好"),
        ])
        answer = response.content if hasattr(response, "content") else str(response)
        return {
            "question": question,
            "answer_type": answer_type,
            "must_cover_aspects": expected_answer_aspects(question, answer_type),
            "final_answer": answer,
            "answer_mode": "greeting",
            "kb_hit": False,
            "graph_context": "",
            "graph_matched_concepts": [],
            "graph_related_concepts": [],
            "retrieval_summary": {
                "query_expansions": [],
                "route_subjects": [],
                "route_type": "simple",
                "answer_type": answer_type,
                "router_features": [answer_type],
                "graph_documents": 0,
                "vector_candidates": 0,
                "lexical_candidates": 0,
                "final_candidates": 0,
                "max_per_source": settings.retrieval_max_per_source,
                "vector_k": settings.retrieval_vector_k,
                "lexical_k": settings.retrieval_lexical_k,
                "final_k": settings.retrieval_final_k,
                "rerank_window": settings.retrieval_rerank_window,
                "chunk_budget_tokens": settings.retrieval_chunk_budget_tokens,
                "graph_budget_tokens": settings.retrieval_graph_budget_tokens,
                "cache_hit": False,
                "cache_similarity": 0.0,
                "cache_policy": "fast",
                "cache_risk": "low",
                "retry_count": 0,
                "retry_strategy": "none",
                "score_profile": {},
                "planner_queries": [question],
                "selected_by": "none",
                "selection_confidence": 0.0,
                "evidence_sources": [],
            },
            "answer_validation": _default_validation(question, answer_type),
            "messages": [AIMessage(content=answer)],
            "critic_decision": "end",
            "critic_reason": "greeting",
            "critic_reason_code": "NO_RETRY_GREETING",
            "retry_strategy": "none",
            "task_status": "completed",
            "task_error_code": "",
            "trace_message": "寒暄节点已直接生成回复",
            "trace_data": {"answer_mode": "greeting", "answer_type": answer_type, "model": model_name},
        }

    return _run_node("greeting", state, _work)


def planner_node(state: AgentState) -> dict:
    def _work() -> dict:
        original_question = _original_question(state)
        loop_step = int(state.get("loop_step") or 0) + 1
        max_steps = max(1, int(state.get("max_steps") or settings.graph_max_steps))
        retry_strategy = str(state.get("retry_strategy") or "none")
        matched_concepts = list(state.get("graph_matched_concepts") or [])
        related_concepts = list(state.get("graph_related_concepts") or [])
        answer_type = str(state.get("answer_type") or infer_answer_type(original_question))
        aspects = _dedup_strings(list(state.get("must_cover_aspects") or expected_answer_aspects(original_question, answer_type)))
        query_rewrites = _build_query_rewrites(original_question, answer_type, aspects, retry_strategy, matched_concepts, related_concepts)
        plan_question = query_rewrites[0] if query_rewrites else original_question
        status = "timeout" if _task_timed_out(state) else "running"
        return {
            "question": original_question,
            "plan_question": plan_question,
            "query_rewrites": query_rewrites,
            "answer_type": answer_type,
            "must_cover_aspects": aspects,
            "loop_step": loop_step,
            "max_steps": max_steps,
            "critic_decision": "",
            "task_status": status,
            "task_error_code": "",
            "trace_message": "完成结构化规划",
            "trace_data": {
                "loop_step": loop_step,
                "max_steps": max_steps,
                "plan_question": plan_question,
                "retry_strategy": retry_strategy,
                "answer_type": answer_type,
                "must_cover_aspects": aspects,
                "query_rewrites": query_rewrites,
            },
        }

    return _run_node("planner", state, _work)


def retrieve_node(state: AgentState) -> dict:
    def _work() -> dict:
        query_rewrites = list(state.get("query_rewrites") or [])
        query_for_retrieve = query_rewrites[0] if query_rewrites else _question_from_state(state)
        original_question = _original_question(state)
        retry_count = int(state.get("retry_count") or 0)
        retry_strategy = str(state.get("retry_strategy") or "none")
        answer_type = str(state.get("answer_type") or infer_answer_type(original_question))
        aspects = _dedup_strings(list(state.get("must_cover_aspects") or expected_answer_aspects(original_question, answer_type)))
        documents, graph_result = hybrid_retrieve(query_for_retrieve, retry_count=retry_count, retry_strategy=retry_strategy)
        matched_concepts = [item.get("name", "") for item in graph_result.get("matched_concepts", []) if item.get("name")]
        related_concepts = [item for item in graph_result.get("related_concepts", []) if item]
        retrieval_summary = dict(graph_result.get("retrieval_summary") or {})
        retrieval_summary["chunk_budget_tokens"] = settings.retrieval_chunk_budget_tokens
        retrieval_summary["graph_budget_tokens"] = settings.retrieval_graph_budget_tokens
        retrieval_summary["planner_queries"] = query_rewrites or [query_for_retrieve]
        retrieval_summary["answer_type"] = answer_type

        relevant_docs: list[Document] = []
        for doc in documents:
            meta = dict(doc.metadata or {})
            distance = meta.get("score")
            if is_chunk_relevant_to_question(
                original_question,
                doc.page_content,
                float(distance) if isinstance(distance, (int, float)) else None,
                metadata=meta,
                matched_concepts=matched_concepts,
            ):
                page_content = doc.page_content
                if meta.get("file_type") == "graph":
                    page_content = compress_lines(page_content, settings.retrieval_graph_budget_tokens)
                else:
                    page_content = truncate_by_budget(page_content, settings.retrieval_chunk_budget_tokens)
                relevant_docs.append(Document(page_content=page_content, metadata=meta))
                if len(relevant_docs) >= max(TOP_K + 1, 4):
                    break

        record_retrieval_metrics(retrieval_summary)

        if relevant_docs:
            chunks = [_doc_to_chunk(d, i) for i, d in enumerate(relevant_docs)]
            graph_context = next((d.page_content for d in relevant_docs if d.metadata.get("file_type") == "graph"), "")
            selected_ids, requires_human, confidence = _choose_evidence(chunks, answer_type)
            selected_by = "human" if requires_human else "auto"
            retrieval_summary["selected_by"] = selected_by
            retrieval_summary["selection_confidence"] = confidence
            retrieval_summary["evidence_sources"] = [str(chunk.get("source") or "") for chunk in chunks if str(chunk.get("source") or "")]
            if requires_human:
                message = (
                    f"知识库已匹配 {min(len(chunks), TOP_K)} 条相关资料。"
                    f" 当前问题类型为 {answer_type}，前两条证据分差较小，请单选 1 条后继续生成解答。"
                )
            else:
                source = next((chunk.get("source") for chunk in chunks if chunk.get("id") in selected_ids), "")
                message = f"已自动选定高置信证据{f'：{source}' if source else ''}，继续生成解答。"
            return {
                "question": original_question,
                "plan_question": query_for_retrieve,
                "query_rewrites": query_rewrites or [query_for_retrieve],
                "answer_type": answer_type,
                "must_cover_aspects": aspects,
                "kb_hit": True,
                "answer_mode": "graph_kb" if graph_context else "kb",
                "retrieved_chunks": chunks,
                "selected_chunk_ids": selected_ids,
                "requires_human_selection": requires_human,
                "selection_confidence": confidence,
                "selected_by": selected_by,
                "graph_context": graph_context,
                "graph_matched_concepts": matched_concepts,
                "graph_related_concepts": related_concepts,
                "retrieval_summary": retrieval_summary,
                "answer_validation": _default_validation(original_question, answer_type),
                "messages": [AIMessage(content=message if requires_human else RETRIEVE_HINT.replace("请**单选**其中一条，", "已自动选定高置信证据，"))],
                "task_status": "awaiting_input" if requires_human else "running",
                "task_error_code": "",
                "trace_message": "检索命中知识库候选",
                "trace_data": {
                    "kb_hit": True,
                    "candidates": len(chunks),
                    "route_type": retrieval_summary.get("route_type"),
                    "retry_strategy": retrieval_summary.get("retry_strategy"),
                    "selected_by": selected_by,
                    "selection_confidence": confidence,
                },
            }

        return {
            "question": original_question,
            "plan_question": query_for_retrieve,
            "query_rewrites": query_rewrites or [query_for_retrieve],
            "answer_type": answer_type,
            "must_cover_aspects": aspects,
            "kb_hit": False,
            "answer_mode": "llm",
            "retrieved_chunks": [],
            "selected_chunk_ids": [],
            "requires_human_selection": False,
            "selection_confidence": 0.0,
            "selected_by": "none",
            "graph_context": "",
            "graph_matched_concepts": matched_concepts,
            "graph_related_concepts": related_concepts,
            "retrieval_summary": retrieval_summary,
            "answer_validation": _default_validation(original_question, answer_type),
            "messages": [AIMessage(content="知识库中未找到与您问题直接相关的资料，将为您生成专业解答。")],
            "task_status": "running",
            "task_error_code": "RETRIEVAL_EMPTY",
            "trace_message": "知识库未命中，转入 LLM 直答",
            "trace_data": {
                "kb_hit": False,
                "route_type": retrieval_summary.get("route_type"),
                "retry_strategy": retrieval_summary.get("retry_strategy"),
                "answer_type": answer_type,
            },
        }

    return _run_node("retrieve", state, _work)


def human_select_node(state: AgentState) -> dict:
    chunks = (state.get("retrieved_chunks") or [])[:TOP_K]
    payload = {
        "question": state.get("question", ""),
        "chunks": chunks,
        "instruction": f"请从下列 {len(chunks)} 条知识库资料中单选 1 条。",
        "selection_mode": "single",
        "kb_hit": True,
    }
    task_id = _task_id(state)
    if task_id:
        append_task_event(task_id, "awaiting_input", message="等待用户选择知识片段", node="human_select", status="awaiting_input", data={"choices": len(chunks)})
    from langgraph.types import interrupt

    selection = interrupt(payload)
    selected_ids = selection.get("selected_chunk_ids", []) if isinstance(selection, dict) else []
    if isinstance(selected_ids, list) and len(selected_ids) > 1:
        selected_ids = selected_ids[:1]
    if task_id:
        append_task_event(task_id, "input_received", message="已收到用户选择的知识片段", node="human_select", status="running", data={"selected_chunk_ids": selected_ids})
    retrieval_summary = dict(state.get("retrieval_summary") or {})
    retrieval_summary["selected_by"] = "human"
    return {
        "selected_chunk_ids": selected_ids,
        "selected_by": "human",
        "requires_human_selection": False,
        "retrieval_summary": retrieval_summary,
        "task_status": "running",
        "execution_trace": [
            {
                "node": "human_select",
                "status": "running",
                "message": "已完成人工选择",
                "step": int(state.get("loop_step") or 0),
                "elapsed_ms": 0,
                "data": {"selected_chunk_ids": selected_ids},
            }
        ],
    }


def generate_answer_llm_node(state: AgentState) -> dict:
    def _work() -> dict:
        question = _original_question(state)
        answer_type = str(state.get("answer_type") or infer_answer_type(question))
        aspects = _dedup_strings(list(state.get("must_cover_aspects") or expected_answer_aspects(question, answer_type)))
        model_name = _resolve_model_name(answer_type)
        llm = get_llm(answer_type)
        prompt = f"用户问题：{question}\n问题类型：{answer_type}\n请覆盖：{'、'.join(aspects)}"
        response = llm.invoke([
            SystemMessage(content=DIRECT_ANSWER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        answer = response.content if hasattr(response, "content") else str(response)
        validation = _default_validation(question, answer_type)
        validation["missing_aspects"] = [aspect for aspect in aspects if aspect not in answer]
        record_answer_validation(validation)
        return {
            "final_answer": answer,
            "answer_mode": "llm",
            "kb_hit": False,
            "answer_type": answer_type,
            "must_cover_aspects": aspects,
            "answer_validation": validation,
            "messages": [AIMessage(content=answer)],
            "task_status": "running",
            "task_error_code": "RETRIEVAL_EMPTY",
            "trace_message": "已完成 LLM 直答",
            "trace_data": {"answer_mode": "llm", "answer_length": len(answer), "answer_type": answer_type, "model": model_name},
        }

    return _run_node("generate_llm", state, _work)


def generate_answer_node(state: AgentState) -> dict:
    def _work() -> dict:
        question = _original_question(state)
        answer_type = str(state.get("answer_type") or infer_answer_type(question))
        aspects = _dedup_strings(list(state.get("must_cover_aspects") or expected_answer_aspects(question, answer_type)))
        chunks = state.get("retrieved_chunks", [])
        selected_ids = state.get("selected_chunk_ids") or []
        primary, support_chunks = _selected_primary_and_support(chunks, selected_ids)
        if not primary:
            return generate_answer_llm_node(state)

        graph_context = compress_lines(state.get("graph_context") or "", settings.retrieval_graph_budget_tokens)
        primary_text = truncate_by_budget(primary["content"], settings.retrieval_chunk_budget_tokens)
        support_texts = [_format_chunk_context(chunk, label="辅助参考") for chunk in support_chunks]
        references = [primary_text, *(chunk.get("content") or "" for chunk in support_chunks), graph_context]
        facts = extract_relevant_facts(question, references, aspects, limit=settings.answer_fact_limit)
        if not facts:
            facts = [primary_text]
        user = _build_answer_prompt(question, answer_type, aspects, facts, {**primary, "content": primary_text}, support_chunks, graph_context)
        if support_texts:
            user += "\n\n补充资料摘录：\n" + "\n\n".join(support_texts[:2])

        model_name = _resolve_model_name(answer_type)
        llm = get_llm(answer_type)
        response = llm.invoke([SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=user)])
        answer = response.content if hasattr(response, "content") else str(response)
        concepts = list(primary.get("concepts") or [])
        for chunk in support_chunks:
            concepts.extend(chunk.get("concepts") or [])
        validation = validate_answer_grounding(
            question,
            answer,
            references,
            concepts,
            answer_type=answer_type,
            expected_aspects=aspects,
            facts=facts,
        )
        record_answer_validation(validation)
        return {
            "final_answer": answer,
            "answer_mode": "graph_kb" if graph_context else "kb",
            "kb_hit": True,
            "answer_type": answer_type,
            "must_cover_aspects": aspects,
            "evidence_facts": facts,
            "answer_validation": validation,
            "messages": [AIMessage(content=answer)],
            "task_status": "running",
            "task_error_code": "",
            "trace_message": "已基于知识库事实生成回答",
            "trace_data": {
                "answer_mode": "graph_kb" if graph_context else "kb",
                "grounding_score": validation.get("grounding_score", 0.0),
                "selected_source": primary.get("source"),
                "supporting_sources": [chunk.get("source") for chunk in support_chunks],
                "fact_count": len(facts),
                "answer_type": answer_type,
                "model": model_name,
            },
        }

    return _run_node("generate_answer", state, _work)


def critic_node(state: AgentState) -> dict:
    def _work() -> dict:
        step = int(state.get("loop_step") or 1)
        max_steps = max(1, int(state.get("max_steps") or settings.graph_max_steps))
        answer = str(state.get("final_answer") or "").strip()
        validation = dict(state.get("answer_validation") or {})
        grounded = bool(validation.get("grounded"))
        kb_hit = bool(state.get("kb_hit"))
        reference_overlap = float(validation.get("reference_overlap") or 0.0)
        question_overlap = float(validation.get("question_overlap") or 0.0)
        citation_coverage = float(validation.get("citation_coverage") or 0.0)
        unsupported_claims = int(validation.get("unsupported_claims") or 0)
        aspect_coverage = float(validation.get("aspect_coverage") or 0.0)
        fact_coverage = float(validation.get("fact_coverage") or 0.0)
        missing_aspects = [str(item).strip() for item in validation.get("missing_aspects") or [] if str(item).strip()]
        answer_type = str(state.get("answer_type") or validation.get("answer_type") or infer_answer_type(_original_question(state)))
        if _task_cancelled(state):
            return {
                "critic_decision": "end",
                "critic_reason": "任务已取消",
                "critic_reason_code": "TASK_CANCELLED",
                "retry_strategy": "none",
                "task_status": "cancelled",
                "task_error_code": "TASK_CANCELLED",
                "trace_message": "检测到任务取消，停止循环",
                "trace_data": {"step": step, "max_steps": max_steps},
            }
        if _task_timed_out(state):
            return {
                "critic_decision": "end",
                "critic_reason": "任务执行超时，终止后续循环",
                "critic_reason_code": "TASK_TIMEOUT",
                "retry_strategy": "none",
                "task_status": "timeout",
                "task_error_code": "TASK_TIMEOUT",
                "trace_message": "达到任务超时阈值，停止循环",
                "trace_data": {"step": step, "max_steps": max_steps},
            }
        if state.get("answer_mode") == "greeting":
            return {
                "critic_decision": "end",
                "critic_reason": "寒暄类回答无需循环",
                "critic_reason_code": "NO_RETRY_GREETING",
                "retry_strategy": "none",
                "task_status": "completed",
                "task_error_code": "",
                "trace_message": "寒暄回复无需进一步校验",
                "trace_data": {"step": step},
            }
        if answer and grounded and unsupported_claims <= 1 and (aspect_coverage >= 0.5 or answer_type == "fact"):
            return {
                "critic_decision": "end",
                "critic_reason": "回答 grounded 且覆盖核心维度，直接结束",
                "critic_reason_code": "GROUNDING_PASSED",
                "retry_strategy": "none",
                "task_status": "completed",
                "task_error_code": "",
                "trace_message": "回答已通过 critic 校验",
                "trace_data": {
                    "step": step,
                    "grounding_score": validation.get("grounding_score", 0.0),
                    "citation_coverage": citation_coverage,
                    "aspect_coverage": aspect_coverage,
                    "fact_coverage": fact_coverage,
                },
            }

        reason_code = "ANSWER_LOW_CONFIDENCE"
        reason = "上一轮回答 grounding 偏低，请扩大召回范围并优先选择高覆盖资料"
        retry_strategy = "focus_coverage"
        error_code = "ANSWER_LOW_CONFIDENCE"

        if not answer:
            reason_code = "ANSWER_EMPTY"
            reason = "上一轮未生成有效答案，请改写检索表达并补充检索线索"
            retry_strategy = "query_rewrite"
            error_code = "ANSWER_EMPTY"
        elif missing_aspects and aspect_coverage < 0.67:
            missing_text = "、".join(missing_aspects[:3])
            reason_code = "ANSWER_MISSING_ASPECTS"
            reason = f"上一轮回答缺少关键覆盖维度：{missing_text}，请优先检索并补全这些部分"
            retry_strategy = "focus_coverage"
            error_code = "ANSWER_MISSING_ASPECTS"
        elif not kb_hit:
            reason_code = "RETRIEVAL_EMPTY"
            reason = "上一轮知识库未命中，请放宽召回范围并优先扩展别名与图谱概念"
            retry_strategy = "widen_retrieval"
            error_code = "RETRIEVAL_EMPTY"
        elif citation_coverage < 0.34 or reference_overlap < 0.16 or fact_coverage < 0.2:
            reason_code = "GROUNDING_WEAK_EVIDENCE"
            reason = "上一轮回答引用证据不足，请扩大召回范围并优先保留覆盖问题核心的资料"
            retry_strategy = "widen_retrieval"
            error_code = "GROUNDING_WEAK_EVIDENCE"
        elif question_overlap < 0.16:
            reason_code = "ANSWER_OFF_TOPIC"
            reason = "上一轮回答偏离问题主线，请优先围绕问题关键词和目标维度生成"
            retry_strategy = "focus_coverage"
            error_code = "ANSWER_OFF_TOPIC"
        elif unsupported_claims >= 2:
            reason_code = "ANSWER_UNSUPPORTED"
            reason = "上一轮回答包含多条缺少依据的表述，请缩小发挥范围并严格贴合资料"
            retry_strategy = "focus_coverage"
            error_code = "ANSWER_UNSUPPORTED"

        if step < max_steps:
            return {
                "critic_decision": "retry",
                "critic_reason": reason,
                "critic_reason_code": reason_code,
                "retry_strategy": retry_strategy,
                "retry_count": int(state.get("retry_count") or 0) + 1,
                "must_cover_aspects": _dedup_strings([*(state.get("must_cover_aspects") or []), *missing_aspects]),
                "task_status": "retrying",
                "task_error_code": error_code,
                "trace_message": "critic 要求进入下一轮重试",
                "trace_data": {
                    "step": step,
                    "max_steps": max_steps,
                    "reason": reason,
                    "reason_code": reason_code,
                    "grounding_score": validation.get("grounding_score", 0.0),
                    "aspect_coverage": aspect_coverage,
                    "missing_aspects": missing_aspects,
                },
            }
        status = "completed" if answer else "failed"
        reason = "达到最大步数，返回当前最佳答案" if answer else "达到最大步数仍未生成有效答案"
        return {
            "critic_decision": "end",
            "critic_reason": reason,
            "critic_reason_code": reason_code,
            "retry_strategy": "none",
            "task_status": status,
            "task_error_code": "" if answer else error_code,
            "trace_message": "critic 结束循环",
            "trace_data": {
                "step": step,
                "max_steps": max_steps,
                "grounding_score": validation.get("grounding_score", 0.0),
                "has_answer": bool(answer),
                "reason_code": reason_code,
                "aspect_coverage": aspect_coverage,
                "missing_aspects": missing_aspects,
            },
        }

    return _run_node("critic", state, _work)
