from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.graph.nodes import (
    critic_node,
    generate_answer_llm_node,
    generate_answer_node,
    greeting_node,
    is_greeting,
    planner_node,
    retrieve_node,
)
from app.graph.prompts import TOP_K
from app.graph.state import AgentState
from app.task_store import append_task_event


def _question_from_state(state: AgentState) -> str:
    q = state.get("question") or ""
    if not q and state.get("messages"):
        last = state["messages"][-1]
        q = getattr(last, "content", str(last))
    return q


def _task_id_from_state(state: AgentState) -> str:
    return str(state.get("task_id") or state.get("thread_id") or "").strip()


def human_select_node(state: AgentState) -> dict:
    chunks = (state.get("retrieved_chunks") or [])[:TOP_K]
    payload = {
        "question": state.get("question", ""),
        "chunks": chunks,
        "instruction": f"请从下列 {len(chunks)} 条知识库资料中单选 1 条。",
        "selection_mode": "single",
        "kb_hit": True,
    }
    task_id = _task_id_from_state(state)
    if task_id:
        append_task_event(task_id, "awaiting_input", message="等待用户选择知识片段", node="human_select", status="awaiting_input", data={"choices": len(chunks)})
    selection = interrupt(payload)
    selected_ids = selection.get("selected_chunk_ids", []) if isinstance(selection, dict) else []
    if isinstance(selected_ids, list) and len(selected_ids) > 1:
        selected_ids = selected_ids[:1]
    if task_id:
        append_task_event(task_id, "input_received", message="已收到用户选择的知识片段", node="human_select", status="running", data={"selected_chunk_ids": selected_ids})
    return {
        "selected_chunk_ids": selected_ids,
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


def route_after_retrieve(state: AgentState) -> str:
    if state.get("kb_hit"):
        return "human_select"
    return "generate_llm"


def route_after_critic(state: AgentState) -> str:
    if str(state.get("critic_decision") or "") == "retry":
        return "planner"
    return "end"


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
        {"human_select": "human_select", "generate_llm": "generate_llm"},
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
