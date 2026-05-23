import re
import uuid

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.graph.prompts import (
    ANSWER_SYSTEM,
    DIRECT_ANSWER_SYSTEM,
    GREETING_SYSTEM,
    RETRIEVE_HINT,
    TOP_K,
)
from app.graph.state import AgentState, RetrievedChunk
from app.rag.vector_store import similarity_search_with_scores
from app.rag.kb_match import is_chunk_relevant_to_question

_GREETING_RE = re.compile(
    r"^(你好|您好|嗨|哈喽|hello|hi|hey|早上好|下午好|晚上好|在吗|同学你好)[\s!！?？。.~、，]*$",
    re.IGNORECASE,
)

FETCH_K = 15


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
    )


def _question_from_state(state: AgentState) -> str:
    question = state.get("question") or ""
    if not question and state.get("messages"):
        last = state["messages"][-1]
        question = getattr(last, "content", str(last))
    return question


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
        "messages": [AIMessage(content=answer)],
    }


def retrieve_node(state: AgentState) -> dict:
    """检索知识库；仅保留与问题真正相关的片段。"""
    question = _question_from_state(state)
    pairs = similarity_search_with_scores(question, k=FETCH_K)
    relevant_docs: list[Document] = []
    for doc, distance in pairs:
        if is_chunk_relevant_to_question(question, doc.page_content, float(distance)):
            meta = dict(doc.metadata or {})
            meta["score"] = round(float(distance), 4)
            relevant_docs.append(Document(page_content=doc.page_content, metadata=meta))
            if len(relevant_docs) >= TOP_K:
                break

    if relevant_docs:
        chunks = [_doc_to_chunk(d, i) for i, d in enumerate(relevant_docs)]
        return {
            "question": question,
            "kb_hit": True,
            "answer_mode": "kb",
            "retrieved_chunks": chunks,
            "messages": [AIMessage(content=RETRIEVE_HINT)],
        }

    return {
        "question": question,
        "kb_hit": False,
        "answer_mode": "llm",
        "retrieved_chunks": [],
        "messages": [
            AIMessage(content="知识库中未找到与您问题直接相关的资料，将为您生成专业解答。")
        ],
    }


def generate_answer_llm_node(state: AgentState) -> dict:
    """知识库无匹配：大模型根据用户问题直接作答。"""
    question = state.get("question", "") or _question_from_state(state)
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=DIRECT_ANSWER_SYSTEM),
            HumanMessage(content=question),
        ]
    )
    answer = response.content if hasattr(response, "content") else str(response)
    return {
        "final_answer": answer,
        "answer_mode": "llm",
        "kb_hit": False,
        "messages": [AIMessage(content=answer)],
    }


def generate_answer_node(state: AgentState) -> dict:
    """知识库有匹配且用户已选资料：严格依据所选片段作答。"""
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
    context = f"[来源: {chunk['source']} | 类型: {chunk['file_type']}]\n{chunk['content']}"
    user = (
        f"用户问题：{question}\n\n"
        f"知识库参考资料（仅此一段）：\n{context}\n\n"
        "请给出具体、可直接用于学习的回答。"
    )
    llm = get_llm()
    response = llm.invoke([SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=user)])
    answer = response.content if hasattr(response, "content") else str(response)
    return {
        "final_answer": answer,
        "answer_mode": "kb",
        "kb_hit": True,
        "messages": [AIMessage(content=answer)],
    }
