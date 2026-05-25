import re
import uuid
from datetime import datetime
from time import perf_counter

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.graph.prompts import ANSWER_SYSTEM, DIRECT_ANSWER_SYSTEM, GREETING_SYSTEM, RETRIEVE_HINT, TOP_K
from app.graph.state import AgentState, ExecutionTrace, RetrievedChunk
from app.observability import record_answer_validation, record_retrieval_metrics
from app.rag.hybrid_retriever import hybrid_retrieve
from app.rag.kb_match import is_chunk_relevant_to_question
from app.rag.retrieval_optimizer import compress_lines, truncate_by_budget, validate_answer_grounding
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


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key or "dummy",
        openai_api_base=settings.openai_api_base,
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
    )


def _question_from_state(state: AgentState) -> str:
    question = state.get("plan_question") or state.get("question") or ""
    if not question and state.get("messages"):
        last = state["messages"][-1]
        question = getattr(last, "content", str(last))
    return question


def _original_question(state: AgentState) -> str:
    return state.get("question") or _question_from_state(state)


def _default_validation() -> dict:
    return {
        "grounded": False,
        "grounding_score": 0.0,
        "reference_overlap": 0.0,
        "question_overlap": 0.0,
        "citation_coverage": 0.0,
        "supported_claims": 0,
        "unsupported_claims": 0,
        "weak_sentences": [],
    }


def _graph_message(graph_result: dict) -> str:
    matched = [item.get("name", "") for item in graph_result.get("matched_concepts", []) if item.get("name")]
    related = [item for item in graph_result.get("related_concepts", []) if item]
    summary = dict(graph_result.get("retrieval_summary") or {})
    if not matched:
        return RETRIEVE_HINT
    text = f"图谱已识别概念：{'、'.join(matched)}。"
    if related:
        text += f" 关联概念：{'、'.join(related[:4])}。"
    if summary.get("query_expansions"):
        text += f" 查询扩展：{'、'.join(summary['query_expansions'][:4])}。"
    if summary.get("cache_hit"):
        text += " 命中语义缓存。"
    if summary.get("route_type"):
        text += f" 当前路由策略：{summary['route_type']}。"
    if summary.get("lexical_candidates"):
        text += f" 当前采用混合召回，保留 {summary.get('final_candidates', 0)} 条候选。"
    text += " 请单选 1 条资料，我将结合图谱关系严格生成解答。"
    return text


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


def _run_node(node_name: str, state: AgentState, fn):
    started = perf_counter()
    task_id = _task_id(state)
    step = int(state.get("loop_step") or 0)
    if task_id:
        append_task_event(task_id, "node_started", message=f"节点开始执行：{node_name}", node=node_name, status=state.get("task_status") or "running", data={"step": step})
    try:
        result = fn()
        elapsed_ms = int((perf_counter() - started) * 1000)
        trace = ExecutionTrace(
            node=node_name,
            status=str(result.get("task_status") or state.get("task_status") or "running"),
            message=str(result.get("trace_message") or f"节点执行完成：{node_name}"),
            step=int(result.get("loop_step") or step),
            elapsed_ms=elapsed_ms,
            data=dict(result.get("trace_data") or {}),
        )
        if task_id:
            append_task_event(
                task_id,
                "node_completed",
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
        primary = chunks[0]
    support: list[dict] = []
    for chunk in chunks:
        if not primary or chunk.get("id") == primary.get("id"):
            continue
        support.append(chunk)
        if len(support) >= 2:
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


def greeting_node(state: AgentState) -> dict:
    def _work() -> dict:
        question = _original_question(state)
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=GREETING_SYSTEM),
            HumanMessage(content=question or "你好"),
        ])
        answer = response.content if hasattr(response, "content") else str(response)
        return {
            "question": question,
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
                "retry_count": 0,
                "retry_strategy": "none",
                "score_profile": {},
            },
            "answer_validation": _default_validation(),
            "messages": [AIMessage(content=answer)],
            "critic_decision": "end",
            "critic_reason": "greeting",
            "critic_reason_code": "NO_RETRY_GREETING",
            "retry_strategy": "none",
            "task_status": "completed",
            "task_error_code": "",
            "trace_message": "寒暄节点已直接生成回复",
            "trace_data": {"answer_mode": "greeting"},
        }

    return _run_node("greeting", state, _work)


def planner_node(state: AgentState) -> dict:
    def _work() -> dict:
        original_question = _original_question(state)
        loop_step = int(state.get("loop_step") or 0) + 1
        max_steps = max(1, int(state.get("max_steps") or settings.graph_max_steps))
        critic_reason = str(state.get("critic_reason") or "").strip()
        retry_strategy = str(state.get("retry_strategy") or "none")
        reason_code = str(state.get("critic_reason_code") or "").strip()
        plan_question = original_question
        if loop_step > 1 and critic_reason:
            if retry_strategy == "widen_retrieval":
                plan_question = f"{original_question}\n补充检索要求：扩大召回范围，补充别名、章节词和关联概念。失败原因：{critic_reason}"
            elif retry_strategy == "focus_coverage":
                plan_question = f"{original_question}\n补充检索要求：优先覆盖定义、条件、步骤和关键概念。失败原因：{critic_reason}"
            elif retry_strategy == "query_rewrite":
                plan_question = f"{original_question}\n补充检索要求：改写检索表达并保留核心术语。失败原因：{critic_reason}"
            else:
                plan_question = f"{original_question}\n补充检索要求：{critic_reason}"
        status = "timeout" if _task_timed_out(state) else "running"
        return {
            "question": original_question,
            "plan_question": plan_question,
            "loop_step": loop_step,
            "max_steps": max_steps,
            "critic_decision": "",
            "task_status": status,
            "task_error_code": "",
            "trace_message": "完成本轮规划",
            "trace_data": {
                "loop_step": loop_step,
                "max_steps": max_steps,
                "plan_question": plan_question,
                "retry_strategy": retry_strategy,
                "reason_code": reason_code,
            },
        }

    return _run_node("planner", state, _work)


def retrieve_node(state: AgentState) -> dict:
    def _work() -> dict:
        query_for_retrieve = _question_from_state(state)
        original_question = _original_question(state)
        retry_count = int(state.get("retry_count") or 0)
        retry_strategy = str(state.get("retry_strategy") or "none")
        documents, graph_result = hybrid_retrieve(query_for_retrieve, retry_count=retry_count, retry_strategy=retry_strategy)
        matched_concepts = [item.get("name", "") for item in graph_result.get("matched_concepts", []) if item.get("name")]
        related_concepts = [item for item in graph_result.get("related_concepts", []) if item]
        retrieval_summary = dict(graph_result.get("retrieval_summary") or {})
        retrieval_summary["chunk_budget_tokens"] = settings.retrieval_chunk_budget_tokens
        retrieval_summary["graph_budget_tokens"] = settings.retrieval_graph_budget_tokens

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
                if len(relevant_docs) >= TOP_K:
                    break

        record_retrieval_metrics(retrieval_summary)

        if relevant_docs:
            chunks = [_doc_to_chunk(d, i) for i, d in enumerate(relevant_docs)]
            graph_context = next((d.page_content for d in relevant_docs if d.metadata.get("file_type") == "graph"), "")
            return {
                "question": original_question,
                "kb_hit": True,
                "answer_mode": "graph_kb" if graph_context else "kb",
                "retrieved_chunks": chunks,
                "graph_context": graph_context,
                "graph_matched_concepts": matched_concepts,
                "graph_related_concepts": related_concepts,
                "retrieval_summary": retrieval_summary,
                "answer_validation": _default_validation(),
                "messages": [AIMessage(content=_graph_message(graph_result))],
                "task_status": "awaiting_input",
                "task_error_code": "",
                "trace_message": "检索命中知识库候选",
                "trace_data": {
                    "kb_hit": True,
                    "candidates": len(chunks),
                    "route_type": retrieval_summary.get("route_type"),
                    "retry_strategy": retrieval_summary.get("retry_strategy"),
                },
            }

        return {
            "question": original_question,
            "kb_hit": False,
            "answer_mode": "llm",
            "retrieved_chunks": [],
            "graph_context": "",
            "graph_matched_concepts": matched_concepts,
            "graph_related_concepts": related_concepts,
            "retrieval_summary": retrieval_summary,
            "answer_validation": _default_validation(),
            "messages": [AIMessage(content="知识库中未找到与您问题直接相关的资料，将为您生成专业解答。")],
            "task_status": "running",
            "task_error_code": "RETRIEVAL_EMPTY",
            "trace_message": "知识库未命中，转入 LLM 直答",
            "trace_data": {
                "kb_hit": False,
                "route_type": retrieval_summary.get("route_type"),
                "retry_strategy": retrieval_summary.get("retry_strategy"),
            },
        }

    return _run_node("retrieve", state, _work)


def generate_answer_llm_node(state: AgentState) -> dict:
    def _work() -> dict:
        question = _original_question(state)
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=DIRECT_ANSWER_SYSTEM),
            HumanMessage(content=question),
        ])
        answer = response.content if hasattr(response, "content") else str(response)
        validation = _default_validation()
        record_answer_validation(validation)
        return {
            "final_answer": answer,
            "answer_mode": "llm",
            "kb_hit": False,
            "answer_validation": validation,
            "messages": [AIMessage(content=answer)],
            "task_status": "running",
            "task_error_code": "RETRIEVAL_EMPTY",
            "trace_message": "已完成 LLM 直答",
            "trace_data": {"answer_mode": "llm", "answer_length": len(answer)},
        }

    return _run_node("generate_llm", state, _work)


def generate_answer_node(state: AgentState) -> dict:
    def _work() -> dict:
        question = _original_question(state)
        chunks = state.get("retrieved_chunks", [])
        selected_ids = state.get("selected_chunk_ids") or []
        primary, support_chunks = _selected_primary_and_support(chunks, selected_ids)
        if not primary:
            return generate_answer_llm_node(state)

        graph_context = compress_lines(state.get("graph_context") or "", settings.retrieval_graph_budget_tokens)
        primary_text = truncate_by_budget(primary["content"], settings.retrieval_chunk_budget_tokens)
        support_texts = [_format_chunk_context(chunk, label="辅助参考") for chunk in support_chunks]
        user = (
            f"用户问题：{question}\n\n"
            f"主参考资料：\n{_format_chunk_context({**primary, 'content': primary_text}, label='主参考')}"
        )
        if support_texts:
            user += "\n\n辅助参考资料：\n" + "\n\n".join(support_texts)
        if graph_context and primary.get("file_type") != "graph":
            user += f"\n\n图谱补充关系：\n{graph_context}"
        user += "\n\n请基于上述资料回答，优先覆盖定义、关键条件、步骤和结论，不要脱离证据扩展。"

        llm = get_llm()
        response = llm.invoke([SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=user)])
        answer = response.content if hasattr(response, "content") else str(response)
        references = [primary_text, *(chunk.get("content") or "" for chunk in support_chunks), graph_context]
        concepts = list(primary.get("concepts") or [])
        for chunk in support_chunks:
            concepts.extend(chunk.get("concepts") or [])
        validation = validate_answer_grounding(question, answer, references, concepts)
        record_answer_validation(validation)
        return {
            "final_answer": answer,
            "answer_mode": "graph_kb" if graph_context else "kb",
            "kb_hit": True,
            "answer_validation": validation,
            "messages": [AIMessage(content=answer)],
            "task_status": "running",
            "task_error_code": "",
            "trace_message": "已基于知识库生成回答",
            "trace_data": {
                "answer_mode": "graph_kb" if graph_context else "kb",
                "grounding_score": validation.get("grounding_score", 0.0),
                "selected_source": primary.get("source"),
                "supporting_sources": [chunk.get("source") for chunk in support_chunks],
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
        if answer and grounded and unsupported_claims <= 1:
            return {
                "critic_decision": "end",
                "critic_reason": "回答 grounded，直接结束",
                "critic_reason_code": "GROUNDING_PASSED",
                "retry_strategy": "none",
                "task_status": "completed",
                "task_error_code": "",
                "trace_message": "回答已通过 critic 校验",
                "trace_data": {
                    "step": step,
                    "grounding_score": validation.get("grounding_score", 0.0),
                    "citation_coverage": citation_coverage,
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
        elif not kb_hit:
            reason_code = "RETRIEVAL_EMPTY"
            reason = "上一轮知识库未命中，请放宽召回范围并优先扩展别名与图谱概念"
            retry_strategy = "widen_retrieval"
            error_code = "RETRIEVAL_EMPTY"
        elif citation_coverage < 0.34 or reference_overlap < 0.16:
            reason_code = "GROUNDING_WEAK_EVIDENCE"
            reason = "上一轮回答引用证据不足，请扩大召回范围并优先保留覆盖问题核心的资料"
            retry_strategy = "widen_retrieval"
            error_code = "GROUNDING_WEAK_EVIDENCE"
        elif question_overlap < 0.16:
            reason_code = "ANSWER_OFF_TOPIC"
            reason = "上一轮回答偏离问题主线，请优先围绕问题关键词和步骤生成"
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
                "task_status": "retrying",
                "task_error_code": error_code,
                "trace_message": "critic 要求进入下一轮重试",
                "trace_data": {
                    "step": step,
                    "max_steps": max_steps,
                    "reason": reason,
                    "reason_code": reason_code,
                    "grounding_score": validation.get("grounding_score", 0.0),
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
            },
        }

    return _run_node("critic", state, _work)
