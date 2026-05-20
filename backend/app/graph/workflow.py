from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.graph.nodes import (
    generate_answer_llm_node,
    generate_answer_node,
    greeting_node,
    is_greeting,
    retrieve_node,
)
from app.graph.prompts import TOP_K
from app.graph.state import AgentState


def _question_from_state(state: AgentState) -> str:
    q = state.get("question") or ""
    if not q and state.get("messages"):
        last = state["messages"][-1]
        q = getattr(last, "content", str(last))
    return q


def human_select_node(state: AgentState) -> dict:
    chunks = (state.get("retrieved_chunks") or [])[:TOP_K]
    payload = {
        "question": state.get("question", ""),
        "chunks": chunks,
        "instruction": f"请从下列 {len(chunks)} 条知识库资料中单选 1 条。",
        "selection_mode": "single",
        "kb_hit": True,
    }
    selection = interrupt(payload)
    selected_ids = selection.get("selected_chunk_ids", []) if isinstance(selection, dict) else []
    if isinstance(selected_ids, list) and len(selected_ids) > 1:
        selected_ids = selected_ids[:1]
    return {"selected_chunk_ids": selected_ids}


def route_after_retrieve(state: AgentState) -> str:
    if state.get("kb_hit"):
        return "human_select"
    return "generate_llm"


def build_graph():
    builder = StateGraph(AgentState)

    def entry_router(state: AgentState) -> str:
        if is_greeting(_question_from_state(state)):
            return "greeting"
        return "retrieve"

    builder.add_node("greeting", greeting_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("human_select", human_select_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("generate_llm", generate_answer_llm_node)

    builder.add_conditional_edges(
        START,
        entry_router,
        {"greeting": "greeting", "retrieve": "retrieve"},
    )
    builder.add_edge("greeting", END)
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"human_select": "human_select", "generate_llm": "generate_llm"},
    )
    builder.add_edge("human_select", "generate_answer")
    builder.add_edge("generate_answer", END)
    builder.add_edge("generate_llm", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
