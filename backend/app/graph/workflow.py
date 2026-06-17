from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    critic_node,
    generate_answer_llm_node,
    generate_answer_node,
    greeting_node,
    human_select_node,
    is_greeting,
    planner_node,
    retrieve_node,
)
from app.graph.state import AgentState


def _question_from_state(state: AgentState) -> str:
    q = state.get("question") or ""
    if not q and state.get("messages"):
        last = state["messages"][-1]
        q = getattr(last, "content", str(last))
    return q


def route_after_retrieve(state: AgentState) -> str:
    if state.get("kb_hit"):
        if state.get("requires_human_selection"):
            return "human_select"
        return "generate_answer"
    return "generate_llm"


def route_after_critic(state: AgentState) -> str:
    if str(state.get("critic_decision") or "") == "retry":
        return "planner"
    return "end"


def _merge_state(state: AgentState, update: dict) -> AgentState:
    merged = dict(state)
    for key, value in update.items():
        if key == "messages":
            merged[key] = [*(state.get("messages") or []), *(value or [])]
            continue
        if key == "execution_trace":
            merged[key] = [*(state.get("execution_trace") or []), *(value or [])]
            continue
        merged[key] = value
    return merged


def _should_stop(state: AgentState) -> bool:
    return str(state.get("task_status") or "").strip().lower() in {"cancelled", "timeout", "failed"}


def _run_from_node(state: AgentState, current: str) -> tuple[AgentState, bool]:
    active = current
    merged_state = dict(state)
    while True:
        if active == "greeting":
            merged_state = _merge_state(merged_state, greeting_node(merged_state))
            return merged_state, False
        if active == "planner":
            merged_state = _merge_state(merged_state, planner_node(merged_state))
            if _should_stop(merged_state):
                return merged_state, False
            active = "retrieve"
            continue
        if active == "retrieve":
            merged_state = _merge_state(merged_state, retrieve_node(merged_state))
            if _should_stop(merged_state):
                return merged_state, False
            next_node = route_after_retrieve(merged_state)
            if next_node == "human_select":
                return merged_state, True
            active = "generate_answer" if next_node == "generate_answer" else "generate_llm"
            continue
        if active == "generate_answer":
            merged_state = _merge_state(merged_state, generate_answer_node(merged_state))
            if _should_stop(merged_state):
                return merged_state, False
            active = "critic"
            continue
        if active == "generate_llm":
            merged_state = _merge_state(merged_state, generate_answer_llm_node(merged_state))
            if _should_stop(merged_state):
                return merged_state, False
            active = "critic"
            continue
        if active == "critic":
            merged_state = _merge_state(merged_state, critic_node(merged_state))
            if _should_stop(merged_state):
                return merged_state, False
            next_node = route_after_critic(merged_state)
            if next_node == "end":
                return merged_state, False
            active = "planner"
            continue
        return merged_state, False


def run_agent_state(state: AgentState) -> tuple[AgentState, bool]:
    entry = "greeting" if is_greeting(_question_from_state(state)) else "planner"
    return _run_from_node(state, entry)


def resume_agent_state(state: AgentState) -> tuple[AgentState, bool]:
    entry = "generate_answer" if state.get("kb_hit") else "generate_llm"
    return _run_from_node(state, entry)


def build_graph():
    builder = StateGraph(AgentState)

    def entry_router(state: AgentState) -> str:
        if is_greeting(_question_from_state(state)):
            return "greeting"
        return "planner"

    builder.add_node("greeting", greeting_node)
    builder.add_node("planner", planner_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("human_select", human_select_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("generate_llm", generate_answer_llm_node)
    builder.add_node("critic", critic_node)

    builder.add_conditional_edges(
        START,
        entry_router,
        {"greeting": "greeting", "planner": "planner"},
    )
    builder.add_edge("greeting", END)
    builder.add_edge("planner", "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"human_select": "human_select", "generate_answer": "generate_answer", "generate_llm": "generate_llm"},
    )
    builder.add_edge("human_select", "generate_answer")
    builder.add_edge("generate_answer", "critic")
    builder.add_edge("generate_llm", "critic")
    builder.add_conditional_edges("critic", route_after_critic, {"planner": "planner", "end": END})

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
