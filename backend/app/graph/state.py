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
    rank_score: float | None


class RetrievalSummary(TypedDict):
    query_expansions: list[str]
    route_subjects: list[str]
    route_type: str
    graph_documents: int
    vector_candidates: int
    lexical_candidates: int
    final_candidates: int
    max_per_source: int
    vector_k: int
    lexical_k: int
    final_k: int
    rerank_window: int
    chunk_budget_tokens: int
    graph_budget_tokens: int
    cache_hit: bool
    cache_similarity: float


class AnswerValidation(TypedDict):
    grounded: bool
    grounding_score: float
    reference_overlap: float
    question_overlap: float


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
    retrieval_summary: RetrievalSummary
    answer_validation: AnswerValidation
