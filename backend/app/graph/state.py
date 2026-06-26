import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class RetrievedChunk(TypedDict, total=False):
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
    coverage_score: float | None


class RetrievalSummary(TypedDict, total=False):
    query_expansions: list[str]
    route_subjects: list[str]
    route_type: str
    answer_type: str
    router_features: list[str]
    intent_profile: dict[str, object]
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
    cache_policy: str
    cache_risk: str
    retry_count: int
    retry_strategy: str
    score_profile: dict[str, float]
    planner_queries: list[str]
    selected_by: str
    selection_confidence: float
    evidence_sources: list[str]


class AnswerValidation(TypedDict, total=False):
    grounded: bool
    grounding_score: float
    reference_overlap: float
    question_overlap: float
    citation_coverage: float
    supported_claims: int
    unsupported_claims: int
    weak_sentences: list[str]
    answer_type: str
    aspect_coverage: float
    missing_aspects: list[str]
    fact_coverage: float
    used_facts: int


class ExecutionTrace(TypedDict):
    node: str
    status: str
    message: str
    step: int
    elapsed_ms: int
    data: dict


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    execution_trace: Annotated[list[ExecutionTrace], operator.add]
    question: str
    plan_question: str
    query_rewrites: list[str]
    answer_type: str
    must_cover_aspects: list[str]
    retrieved_chunks: list[RetrievedChunk]
    selected_chunk_ids: list[str]
    requires_human_selection: bool
    selection_confidence: float
    selected_by: str
    evidence_facts: list[str]
    final_answer: str
    thread_id: str
    task_id: str
    kb_hit: bool
    answer_mode: str
    graph_context: str
    graph_matched_concepts: list[str]
    graph_related_concepts: list[str]
    retrieval_summary: RetrievalSummary
    answer_validation: AnswerValidation
    loop_step: int
    max_steps: int
    critic_decision: str
    critic_reason: str
    critic_reason_code: str
    retry_strategy: str
    retry_count: int
    task_status: str
    task_error_code: str
