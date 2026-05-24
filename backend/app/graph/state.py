from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class RetrievedChunk(TypedDict):
    id: str
    content: str
    source: str
    file_type: str
    score: float | None
    subject: str | None
    chapter: str | None
    retrieval_mode: str | None
    concepts: list[str]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    retrieved_chunks: list[RetrievedChunk]
    selected_chunk_ids: list[str]
    final_answer: str
    thread_id: str
    kb_hit: bool
    answer_mode: str
    graph_context: str
    graph_matched_concepts: list[str]
    graph_related_concepts: list[str]
