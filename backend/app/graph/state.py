from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class RetrievedChunk(TypedDict):
    id: str
    content: str
    source: str
    file_type: str
    score: float | None


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    retrieved_chunks: list[RetrievedChunk]
    selected_chunk_ids: list[str]
    final_answer: str
    thread_id: str
    kb_hit: bool
    answer_mode: str
