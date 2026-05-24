import re
import uuid

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.graph.prompts import ANSWER_SYSTEM, DIRECT_ANSWER_SYSTEM, GREETING_SYSTEM, RETRIEVE_HINT, TOP_K
from app.graph.state import AgentState, RetrievedChunk
from app.observability import record_answer_validation, record_retrieval_metrics
from app.rag.hybrid_retriever import hybrid_retrieve
from app.rag.kb_match import is_chunk_relevant_to_question
from app.rag.retrieval_optimizer import compress_lines, truncate_by_budget, validate_answer_grounding

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
    question = state.get("question") or ""
    if not question and state.get("messages"):
        last = state["messages"][-1]
        question = getattr(last, "content", str(last))
    return question


def _default_validation() -> dict:
    return {
        "grounded": False,
        "grounding_score": 0.0,
        "reference_overlap": 0.0,
        "question_overlap": 0.0,
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


def greeting_node(state: AgentState) -> dict:
    question = _question_from_state(state)
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=GREETING_SYSTEM),
            HumanMessage(content=question or "你好"),
        ]
    )
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
        },
        "answer_validation": _default_validation(),
        "messages": [AIMessage(content=answer)],
    }


def retrieve_node(state: AgentState) -> dict:
    question = _question_from_state(state)
    documents, graph_result = hybrid_retrieve(question)
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
            question,
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
            "question": question,
            "kb_hit": True,
            "answer_mode": "graph_kb" if graph_context else "kb",
            "retrieved_chunks": chunks,
            "graph_context": graph_context,
            "graph_matched_concepts": matched_concepts,
            "graph_related_concepts": related_concepts,
            "retrieval_summary": retrieval_summary,
            "answer_validation": _default_validation(),
            "messages": [AIMessage(content=_graph_message(graph_result))],
        }

    return {
        "question": question,
        "kb_hit": False,
        "answer_mode": "llm",
        "retrieved_chunks": [],
        "graph_context": "",
        "graph_matched_concepts": matched_concepts,
        "graph_related_concepts": related_concepts,
        "retrieval_summary": retrieval_summary,
        "answer_validation": _default_validation(),
        "messages": [AIMessage(content="知识库中未找到与您问题直接相关的资料，将为您生成专业解答。")],
    }


def generate_answer_llm_node(state: AgentState) -> dict:
    question = state.get("question", "") or _question_from_state(state)
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=DIRECT_ANSWER_SYSTEM),
            HumanMessage(content=question),
        ]
    )
    answer = response.content if hasattr(response, "content") else str(response)
    validation = _default_validation()
    record_answer_validation(validation)
    return {
        "final_answer": answer,
        "answer_mode": "llm",
        "kb_hit": False,
        "answer_validation": validation,
        "messages": [AIMessage(content=answer)],
    }


def generate_answer_node(state: AgentState) -> dict:
    question = state.get("question", "")
    chunks = state.get("retrieved_chunks", [])
    selected_ids = state.get("selected_chunk_ids") or []

    if selected_ids:
        sid = selected_ids[0]
        selected = [c for c in chunks if c["id"] == sid]
    else:
        selected = chunks[:1]

    if not selected:
        return generate_answer_llm_node(state)

    chunk = selected[0]
    graph_context = compress_lines(state.get("graph_context") or "", settings.retrieval_graph_budget_tokens)
    concepts = "、".join(chunk.get("concepts") or [])
    context_header = f"[来源: {chunk['source']} | 类型: {chunk['file_type']}]"
    if chunk.get("subject"):
        context_header += f"[学科: {chunk['subject']}]"
    if chunk.get("chapter"):
        context_header += f"[章节: {chunk['chapter']}]"
    if concepts:
        context_header += f"[概念: {concepts}]"
    chunk_text = truncate_by_budget(chunk["content"], settings.retrieval_chunk_budget_tokens)
    user = f"用户问题：{question}\n\n知识库参考资料（仅此一段）：\n{context_header}\n{chunk_text}"
    if graph_context and chunk.get("file_type") != "graph":
        user += f"\n\n图谱补充关系：\n{graph_context}"
    user += "\n\n请给出具体、可直接用于学习的回答。"
    llm = get_llm()
    response = llm.invoke([SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=user)])
    answer = response.content if hasattr(response, "content") else str(response)
    validation = validate_answer_grounding(question, answer, [chunk_text, graph_context], chunk.get("concepts") or [])
    record_answer_validation(validation)
    return {
        "final_answer": answer,
        "answer_mode": "graph_kb" if graph_context else "kb",
        "kb_hit": True,
        "answer_validation": validation,
        "messages": [AIMessage(content=answer)],
    }
