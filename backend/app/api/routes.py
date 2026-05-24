import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.graph.prompts import TOP_K
from app.graph.workflow import get_graph
from app.media.service import create_image_job, create_video_job, get_media_job
from app.rag.evaluation import evaluate_answer, evaluate_retrieval
from app.rag.graph_store import graph_overview, search_graph
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


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    style: str = Field(default="教育海报", min_length=1, max_length=80)
    aspect_ratio: str = Field(default="16:9", min_length=3, max_length=20)


class VideoGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1400)
    mode: str = Field(default="text-to-video", min_length=1, max_length=40)
    duration_seconds: int = Field(default=6, ge=3, le=20)
    source_image_url: str | None = Field(default=None, max_length=2000)


class RetrievalEvalCase(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    expected_sources: list[str] = Field(default_factory=list)
    expected_terms: list[str] = Field(default_factory=list)
    gold_answer: str | None = Field(default=None, max_length=4000)


class RetrievalEvalRequest(BaseModel):
    cases: list[RetrievalEvalCase] = Field(default_factory=list, min_length=1, max_length=100)
    top_k: int = Field(default=3, ge=1, le=10)


class AnswerEvalRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    answer: str = Field(min_length=1, max_length=6000)
    references: list[str] = Field(default_factory=list, max_length=20)
    concepts: list[str] = Field(default_factory=list, max_length=20)


@router.get("/health")
def health():
    return {"status": "ok", "service": "education-agent"}


@router.post("/ingest")
def ingest(req: IngestRequest):
    return ingest_knowledge_base(reset=req.reset)


@router.post("/media/image/generate")
async def generate_image(req: ImageGenerationRequest):
    return await create_image_job(req.prompt, req.style, req.aspect_ratio)


@router.post("/media/video/generate")
async def generate_video(req: VideoGenerationRequest):
    return await create_video_job(req.prompt, req.mode, req.duration_seconds, req.source_image_url)


@router.get("/media/jobs/{job_id}")
async def media_job(job_id: str):
    job = await get_media_job(job_id)
    if not job:
        raise HTTPException(404, "Media job not found")
    return job


@router.get("/graph/overview")
def graph_overview_api(limit: int = Query(default=12, ge=1, le=50)):
    return graph_overview(limit=limit)


@router.get("/graph/search")
def graph_search_api(question: str = Query(..., min_length=1), limit_sources: int = Query(default=4, ge=1, le=20)):
    return search_graph(question=question, limit_sources=limit_sources)


@router.post("/eval/retrieval")
def retrieval_eval_api(req: RetrievalEvalRequest):
    return evaluate_retrieval([case.model_dump() for case in req.cases], top_k=req.top_k)


@router.post("/eval/answer")
def answer_eval_api(req: AnswerEvalRequest):
    return evaluate_answer(req.model_dump())


def _graph_state_dict(result: Any, snapshot: Any) -> dict:
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


def _graph_summary(state: dict) -> dict:
    return {
        "matched_concepts": state.get("graph_matched_concepts") or [],
        "related_concepts": state.get("graph_related_concepts") or [],
        "has_graph_context": bool(state.get("graph_context")),
    }


def _default_retrieval_summary() -> dict:
    return {
        "query_expansions": [],
        "route_subjects": [],
        "route_type": "simple",
        "graph_documents": 0,
        "vector_candidates": 0,
        "lexical_candidates": 0,
        "final_candidates": 0,
        "max_per_source": 0,
        "vector_k": 0,
        "lexical_k": 0,
        "final_k": 0,
        "rerank_window": 0,
        "chunk_budget_tokens": 0,
        "graph_budget_tokens": 0,
        "cache_hit": False,
        "cache_similarity": 0.0,
    }


def _default_answer_validation() -> dict:
    return {
        "grounded": False,
        "grounding_score": 0.0,
        "reference_overlap": 0.0,
        "question_overlap": 0.0,
    }


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
        "graph_context": "",
        "graph_matched_concepts": [],
        "graph_related_concepts": [],
        "retrieval_summary": _default_retrieval_summary(),
        "answer_validation": _default_answer_validation(),
    }
    result = graph.invoke(initial, config)
    snapshot = graph.get_state(config)
    state = _graph_state_dict(result, snapshot)

    if _is_paused(snapshot, result):
        chunks = (_chunks_from_interrupt(snapshot) or state.get("retrieved_chunks") or [])[:TOP_K]
        return {
            "thread_id": thread_id,
            "status": "awaiting_selection",
            "mode": state.get("answer_mode") or "kb",
            "kb_hit": True,
            "retrieved_chunks": chunks,
            "selection_mode": "single",
            "graph_summary": _graph_summary(state),
            "retrieval_summary": state.get("retrieval_summary") or {},
            "answer_validation": state.get("answer_validation") or _default_answer_validation(),
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
        "graph_summary": _graph_summary(state),
        "retrieval_summary": state.get("retrieval_summary") or {},
        "answer_validation": state.get("answer_validation") or _default_answer_validation(),
        "message": None if mode in {"kb", "graph_kb"} else "知识库中无直接相关条目，以下由 AI 根据您的问题生成",
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
    result = graph.invoke(Command(resume={"selected_chunk_ids": chunk_ids}), config)
    after = graph.get_state(config)
    state = _graph_state_dict(result, after)
    answer = state.get("final_answer") or ""
    messages = state.get("messages") or []
    return {
        "thread_id": req.thread_id,
        "status": "completed",
        "mode": state.get("answer_mode") or "kb",
        "kb_hit": True,
        "answer": answer,
        "graph_summary": _graph_summary(state),
        "retrieval_summary": state.get("retrieval_summary") or {},
        "answer_validation": state.get("answer_validation") or _default_answer_validation(),
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
        "graph_summary": _graph_summary(values),
        "retrieval_summary": values.get("retrieval_summary", {}),
        "answer_validation": values.get("answer_validation", _default_answer_validation()),
    }


@router.get("/daily-push/latest")
def daily_push_latest():
    return get_today_schedule()


@router.get("/daily-schedule")
def daily_schedule():
    return get_today_schedule()


@router.get("/daily-push/history")
def daily_push_history():
    return {"items": get_push_history()}


@router.post("/daily-push/generate")
def daily_push_generate():
    return generate_daily_plan()
