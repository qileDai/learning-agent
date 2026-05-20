import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.graph.prompts import TOP_K
from app.graph.workflow import get_graph
from app.rag.ingest import ingest_knowledge_base
from app.scheduler.daily_push import generate_daily_plan, get_push_history
from app.scheduler.daily_schedule import get_today_schedule

router = APIRouter()


class ChatStartRequest(BaseModel):
    question: str
    thread_id: str | None = None


class ChatResumeRequest(BaseModel):
    thread_id: str
    selected_chunk_ids: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    reset: bool = False


@router.get("/health")
def health():
    return {"status": "ok", "service": "education-agent"}


@router.post("/ingest")
def ingest(req: IngestRequest):
    return ingest_knowledge_base(reset=req.reset)


def _graph_state_dict(result: Any, snapshot: Any) -> dict:
    """合并 checkpoint 状态与 invoke 返回值（兼容 dict / GraphOutput）。"""
    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    if hasattr(result, "value"):
        extra = result.value if isinstance(result.value, dict) else {}
    elif isinstance(result, dict):
        extra = {k: v for k, v in result.items() if k != "__interrupt__"}
    else:
        extra = {}
    return {**values, **extra}


def _is_paused(snapshot: Any, result: Any) -> bool:
    if snapshot.next:
        return True
    if getattr(snapshot, "interrupts", ()):
        return True
    if hasattr(result, "interrupts") and result.interrupts:
        return True
    if isinstance(result, dict) and result.get("__interrupt__"):
        return True
    return False


def _chunks_from_interrupt(snapshot: Any) -> list | None:
    if snapshot.tasks:
        for task in snapshot.tasks:
            if task.interrupts:
                val = task.interrupts[0].value
                if isinstance(val, dict) and "chunks" in val:
                    return val["chunks"]
    return None


@router.post("/chat/start")
def chat_start(req: ChatStartRequest):
    graph = get_graph()
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial = {
        "question": req.question,
        "messages": [],
        "thread_id": thread_id,
        "kb_hit": False,
        "answer_mode": "",
        "retrieved_chunks": [],
        "selected_chunk_ids": [],
        "final_answer": "",
    }
    result = graph.invoke(initial, config)
    snapshot = graph.get_state(config)
    state = _graph_state_dict(result, snapshot)

    if _is_paused(snapshot, result):
        chunks = (_chunks_from_interrupt(snapshot) or state.get("retrieved_chunks") or [])[:TOP_K]
        return {
            "thread_id": thread_id,
            "status": "awaiting_selection",
            "mode": "kb",
            "kb_hit": True,
            "retrieved_chunks": chunks,
            "selection_mode": "single",
            "message": f"知识库已匹配 {len(chunks)} 条相关资料，请单选 1 条后生成解答。",
        }
    answer = state.get("final_answer") or ""
    mode = state.get("answer_mode") or "llm"
    return {
        "thread_id": thread_id,
        "status": "completed",
        "mode": mode,
        "kb_hit": bool(state.get("kb_hit")),
        "answer": answer,
        "message": (
            None
            if mode == "kb"
            else "知识库中无直接相关条目，以下由 AI 根据您的问题生成"
        ),
    }


@router.post("/chat/resume")
def chat_resume(req: ChatResumeRequest):
    graph = get_graph()
    config = {"configurable": {"thread_id": req.thread_id}}
    snapshot = graph.get_state(config)
    if not _is_paused(snapshot, None):
        raise HTTPException(400, "No pending interrupt for this thread")
    chunk_ids = req.selected_chunk_ids[:1] if req.selected_chunk_ids else []
    if not chunk_ids:
        raise HTTPException(400, "请单选一条知识片段")
    result = graph.invoke(
        Command(resume={"selected_chunk_ids": chunk_ids}),
        config,
    )
    after = graph.get_state(config)
    state = _graph_state_dict(result, after)
    answer = state.get("final_answer") or ""
    messages = state.get("messages") or []
    return {
        "thread_id": req.thread_id,
        "status": "completed",
        "mode": "kb",
        "kb_hit": True,
        "answer": answer,
        "message": None if answer else "未生成回答",
        "messages": [m.content for m in messages if hasattr(m, "content")],
    }


@router.get("/chat/state/{thread_id}")
def chat_state(thread_id: str):
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    values = state.values or {}
    return {
        "thread_id": thread_id,
        "next": state.next,
        "retrieved_chunks": values.get("retrieved_chunks", []),
        "final_answer": values.get("final_answer"),
    }


@router.get("/daily-push/latest")
def daily_push_latest():
    return get_today_schedule()


@router.get("/daily-schedule")
def daily_schedule():
    """今日系统默认任务表，含 upcoming / active / completed 与亮灯任务 id。"""
    return get_today_schedule()


@router.get("/daily-push/history")
def daily_push_history():
    return {"items": get_push_history()}


@router.post("/daily-push/generate")
def daily_push_generate():
    return generate_daily_plan()
