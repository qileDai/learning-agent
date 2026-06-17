import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import settings
from app.graph.prompts import TOP_K
from app.graph.workflow import resume_agent_state, run_agent_state
from app.media.service import create_image_job, create_video_job, get_media_job
from app.rag.evaluation import evaluate_answer, evaluate_retrieval, export_failed_cases
from app.rag.graph_store import graph_overview, search_graph
from app.rag.ingest import ingest_knowledge_base
from app.scheduler.daily_push import generate_daily_plan, get_push_history
from app.scheduler.daily_schedule import get_today_schedule
from app.stock_service import get_daily_stock_picks
from app.task_store import (
    awaiting_input_task,
    cancel_task,
    complete_task,
    fail_task,
    get_task,
    get_task_events,
    list_tasks,
    mark_task_status,
    start_task,
    timeout_task,
)

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


@router.get("/stocks/daily-picks")
def daily_stock_picks(limit: int = Query(default=10, ge=1, le=10)):
    try:
        return get_daily_stock_picks(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"股票实时数据暂不可用：{exc}") from exc


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


@router.get("/eval/failure-samples")
def failure_samples_api(limit: int = Query(default=50, ge=1, le=500), write_file: bool = Query(default=True)):
    return export_failed_cases(limit=limit, write_file=write_file)


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
        "answer_type": "fact",
        "router_features": [],
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
        "cache_policy": "fast",
        "cache_risk": "low",
        "retry_count": 0,
        "retry_strategy": "none",
        "score_profile": {},
        "planner_queries": [],
        "selected_by": "pending",
        "selection_confidence": 0.0,
        "evidence_sources": [],
    }


def _default_answer_validation() -> dict:
    return {
        "grounded": False,
        "grounding_score": 0.0,
        "reference_overlap": 0.0,
        "question_overlap": 0.0,
        "citation_coverage": 0.0,
        "supported_claims": 0,
        "unsupported_claims": 0,
        "weak_sentences": [],
        "answer_type": "fact",
        "aspect_coverage": 0.0,
        "missing_aspects": [],
        "fact_coverage": 0.0,
        "used_facts": 0,
    }


def _initial_chat_state(question: str, thread_id: str, task_id: str) -> dict[str, Any]:
    return {
        "question": question,
        "plan_question": question,
        "query_rewrites": [question],
        "answer_type": "fact",
        "must_cover_aspects": [],
        "messages": [],
        "execution_trace": [],
        "thread_id": thread_id,
        "task_id": task_id,
        "kb_hit": False,
        "answer_mode": "",
        "retrieved_chunks": [],
        "selected_chunk_ids": [],
        "requires_human_selection": False,
        "selection_confidence": 0.0,
        "selected_by": "pending",
        "evidence_facts": [],
        "final_answer": "",
        "graph_context": "",
        "graph_matched_concepts": [],
        "graph_related_concepts": [],
        "retrieval_summary": _default_retrieval_summary(),
        "answer_validation": _default_answer_validation(),
        "loop_step": 0,
        "max_steps": settings.graph_max_steps,
        "critic_decision": "",
        "critic_reason": "",
        "critic_reason_code": "",
        "retry_strategy": "none",
        "retry_count": 0,
        "task_status": "running",
        "task_error_code": "",
    }


def _stored_chat_state(task: dict[str, Any] | None) -> dict[str, Any] | None:
    if not task:
        return None
    task_id = str(task.get("task_id") or "").strip()
    thread_id = str(task.get("thread_id") or task_id).strip()
    payload = dict(task.get("payload") or {})
    question = str(payload.get("question") or "").strip()
    if not task_id or not question:
        return None
    result = dict(task.get("result") or {})
    state = _initial_chat_state(question, thread_id, task_id)
    state.update(
        {
            "plan_question": result.get("plan_question") or question,
            "query_rewrites": result.get("query_rewrites") or [question],
            "answer_type": result.get("answer_type") or "fact",
            "must_cover_aspects": result.get("must_cover_aspects") or [],
            "retrieved_chunks": result.get("retrieved_chunks") or [],
            "selected_chunk_ids": result.get("selected_chunk_ids") or [],
            "requires_human_selection": str(task.get("status") or "") == "awaiting_input",
            "selection_confidence": float(result.get("selection_confidence") or 0.0),
            "selected_by": result.get("selected_by") or "pending",
            "evidence_facts": result.get("evidence_facts") or [],
            "final_answer": result.get("final_answer") or "",
            "kb_hit": bool(result.get("kb_hit")),
            "answer_mode": result.get("answer_mode") or "",
            "graph_context": result.get("graph_context") or "",
            "graph_matched_concepts": result.get("graph_matched_concepts") or [],
            "graph_related_concepts": result.get("graph_related_concepts") or [],
            "retrieval_summary": result.get("retrieval_summary") or _default_retrieval_summary(),
            "answer_validation": result.get("answer_validation") or _default_answer_validation(),
            "execution_trace": result.get("execution_trace") or [],
            "loop_step": int(result.get("loop_step") or 0),
            "max_steps": int(task.get("max_steps") or settings.graph_max_steps),
            "critic_reason": result.get("critic_reason") or "",
            "critic_reason_code": result.get("critic_reason_code") or "",
            "retry_strategy": result.get("retry_strategy") or "none",
            "retry_count": int(result.get("retry_count") or 0),
            "task_status": str(task.get("status") or "running"),
            "task_error_code": str(task.get("error_code") or result.get("warning_code") or ""),
        }
    )
    return state


def _graph_state_for_thread(thread_id: str) -> dict[str, Any] | None:
    task = get_task(thread_id)
    state = _stored_chat_state(task)
    if state is None:
        return None
    return {
        "task_id": thread_id,
        "thread_id": thread_id,
        "next": ["human_select"] if state.get("requires_human_selection") else [],
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "selected_chunk_ids": state.get("selected_chunk_ids", []),
        "final_answer": state.get("final_answer"),
        "graph_summary": _graph_summary(state),
        "retrieval_summary": state.get("retrieval_summary", _default_retrieval_summary()),
        "answer_validation": state.get("answer_validation", _default_answer_validation()),
        "execution_trace": state.get("execution_trace", []),
    }


def _build_task_detail(task: dict[str, Any]) -> dict[str, Any]:
    task_id = str(task.get("task_id") or "")
    thread_id = str(task.get("thread_id") or task_id)
    graph_state = _graph_state_for_thread(thread_id) if str(task.get("kind") or "") == "chat" else None
    return {
        "task": task,
        "events": get_task_events(task_id, limit=200),
        "graph_state": graph_state,
    }


def _sync_chat_task(task_id: str, state: dict, *, paused: bool, message: str | None = None) -> None:
    result = {
        "question": state.get("question") or "",
        "plan_question": state.get("plan_question") or state.get("question") or "",
        "query_rewrites": state.get("query_rewrites") or [],
        "answer_type": state.get("answer_type") or "fact",
        "must_cover_aspects": state.get("must_cover_aspects") or [],
        "answer_mode": state.get("answer_mode") or "llm",
        "kb_hit": bool(state.get("kb_hit")),
        "final_answer": state.get("final_answer") or "",
        "retrieved_chunks": state.get("retrieved_chunks") or [],
        "selected_chunk_ids": state.get("selected_chunk_ids") or [],
        "selection_confidence": float(state.get("selection_confidence") or 0.0),
        "selected_by": state.get("selected_by") or "pending",
        "evidence_facts": state.get("evidence_facts") or [],
        "graph_context": state.get("graph_context") or "",
        "graph_matched_concepts": state.get("graph_matched_concepts") or [],
        "graph_related_concepts": state.get("graph_related_concepts") or [],
        "retrieval_summary": state.get("retrieval_summary") or {},
        "answer_validation": state.get("answer_validation") or _default_answer_validation(),
        "execution_trace": state.get("execution_trace") or [],
        "loop_step": int(state.get("loop_step") or 0),
        "retry_count": int(state.get("retry_count") or 0),
        "critic_reason": state.get("critic_reason") or "",
        "critic_reason_code": state.get("critic_reason_code") or "",
        "retry_strategy": state.get("retry_strategy") or "none",
    }
    status = str(state.get("task_status") or "running")
    error_code = str(state.get("task_error_code") or "").strip() or None
    error_message = str(state.get("critic_reason") or "").strip() or None
    if paused:
        awaiting_input_task(task_id, message=message or "等待用户选择知识片段", current_step=result["loop_step"], result=result)
        return
    if status == "timeout":
        timeout_task(task_id, error_message=error_message or "任务超时", current_step=result["loop_step"])
        return
    if status == "failed":
        fail_task(task_id, error_code=error_code or "GRAPH_EXECUTION_FAILED", error_message=error_message or "图执行失败", current_step=result["loop_step"])
        return
    updated_result = dict(result)
    if error_code:
        updated_result["warning_code"] = error_code
        updated_result["warning_message"] = error_message
    complete_task(task_id, result=updated_result, current_step=result["loop_step"])


@router.post("/chat/start")
def chat_start(req: ChatStartRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    task_id = thread_id
    start_task(
        task_id,
        kind="chat",
        title="知识库问答任务",
        payload={"question": req.question},
        thread_id=thread_id,
        max_steps=settings.graph_max_steps,
    )
    initial = _initial_chat_state(req.question, thread_id, task_id)
    try:
        state, paused = run_agent_state(initial)
        if paused:
            chunks = (state.get("retrieved_chunks") or [])[:TOP_K]
            _sync_chat_task(task_id, state, paused=True, message=f"知识库已匹配 {len(chunks)} 条相关资料，请单选 1 条后生成解答。")
            return {
                "task_id": task_id,
                "thread_id": thread_id,
                "status": "awaiting_selection",
                "mode": state.get("answer_mode") or "kb",
                "kb_hit": True,
                "retrieved_chunks": chunks,
                "selection_mode": "single",
                "graph_summary": _graph_summary(state),
                "retrieval_summary": state.get("retrieval_summary") or _default_retrieval_summary(),
                "answer_validation": state.get("answer_validation") or _default_answer_validation(),
                "execution_trace": state.get("execution_trace") or [],
                "message": f"知识库已匹配 {len(chunks)} 条相关资料，请单选 1 条后生成解答。",
            }
        _sync_chat_task(task_id, state, paused=False)
        answer = state.get("final_answer") or ""
        mode = state.get("answer_mode") or "llm"
        return {
            "task_id": task_id,
            "thread_id": thread_id,
            "status": "completed",
            "mode": mode,
            "kb_hit": bool(state.get("kb_hit")),
            "answer": answer,
            "graph_summary": _graph_summary(state),
            "retrieval_summary": state.get("retrieval_summary") or _default_retrieval_summary(),
            "answer_validation": state.get("answer_validation") or _default_answer_validation(),
            "execution_trace": state.get("execution_trace") or [],
            "message": None if mode in {"kb", "graph_kb", "greeting"} else "知识库中无直接相关条目，以下由 AI 根据您的问题生成",
        }
    except Exception as exc:
        fail_task(task_id, error_code="CHAT_START_FAILED", error_message=str(exc))
        raise HTTPException(500, f"chat start failed: {exc}") from exc


@router.post("/chat/resume")
def chat_resume(req: ChatResumeRequest):
    task = get_task(req.thread_id)
    if not task:
        raise HTTPException(404, "Task not found")
    state = _stored_chat_state(task)
    if state is None or not state.get("requires_human_selection"):
        raise HTTPException(400, "No pending interrupt for this thread")
    chunk_ids = req.selected_chunk_ids[:1] if req.selected_chunk_ids else []
    if not chunk_ids:
        raise HTTPException(400, "请单选一条知识片段")
    try:
        state["selected_chunk_ids"] = chunk_ids
        state["selected_by"] = "human"
        state["requires_human_selection"] = False
        retrieval_summary = dict(state.get("retrieval_summary") or {})
        retrieval_summary["selected_by"] = "human"
        state["retrieval_summary"] = retrieval_summary
        mark_task_status(task["task_id"], status="running", message="继续执行知识库问答任务")
        state, paused = resume_agent_state(state)
        if paused:
            chunks = (state.get("retrieved_chunks") or [])[:TOP_K]
            _sync_chat_task(task["task_id"], state, paused=True, message=f"知识库已匹配 {len(chunks)} 条相关资料，请单选 1 条后生成解答。")
            return {
                "task_id": task["task_id"],
                "thread_id": req.thread_id,
                "status": "awaiting_selection",
                "mode": state.get("answer_mode") or "kb",
                "kb_hit": bool(state.get("kb_hit")),
                "retrieved_chunks": chunks,
                "selection_mode": "single",
                "graph_summary": _graph_summary(state),
                "retrieval_summary": state.get("retrieval_summary") or _default_retrieval_summary(),
                "answer_validation": state.get("answer_validation") or _default_answer_validation(),
                "execution_trace": state.get("execution_trace") or [],
                "message": f"知识库已匹配 {len(chunks)} 条相关资料，请单选 1 条后生成解答。",
            }
        _sync_chat_task(task["task_id"], state, paused=False)
        answer = state.get("final_answer") or ""
        return {
            "task_id": task["task_id"],
            "thread_id": req.thread_id,
            "status": "completed",
            "mode": state.get("answer_mode") or "kb",
            "kb_hit": bool(state.get("kb_hit")),
            "answer": answer,
            "graph_summary": _graph_summary(state),
            "retrieval_summary": state.get("retrieval_summary") or _default_retrieval_summary(),
            "answer_validation": state.get("answer_validation") or _default_answer_validation(),
            "execution_trace": state.get("execution_trace") or [],
            "message": None if answer else "未生成回答",
        }
    except Exception as exc:
        fail_task(task["task_id"], error_code="CHAT_RESUME_FAILED", error_message=str(exc))
        raise HTTPException(500, f"chat resume failed: {exc}") from exc


@router.get("/chat/state/{thread_id}")
def chat_state(thread_id: str):
    state = _graph_state_for_thread(thread_id)
    task = get_task(thread_id)
    if state is None:
        return {
            "task_id": thread_id,
            "thread_id": thread_id,
            "next": [],
            "retrieved_chunks": [],
            "final_answer": None,
            "graph_summary": _graph_summary({}),
            "retrieval_summary": _default_retrieval_summary(),
            "answer_validation": _default_answer_validation(),
            "execution_trace": [],
            "task": task,
        }
    return {**state, "task": task}


@router.get("/tasks")
def task_list(kind: str | None = None, status: str | None = None, limit: int = Query(default=50, ge=1, le=200)):
    return {"items": list_tasks(kind=kind, status=status, limit=limit)}


@router.get("/tasks/{task_id}")
def task_detail(task_id: str):
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return {"task": task}


@router.get("/tasks/{task_id}/detail")
def task_detail_full(task_id: str):
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return _build_task_detail(task)


@router.get("/tasks/{task_id}/events")
def task_events(task_id: str, limit: int = Query(default=100, ge=1, le=500)):
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return {"task_id": task_id, "events": get_task_events(task_id, limit=limit)}


@router.post("/tasks/{task_id}/cancel")
def task_cancel(task_id: str):
    task = cancel_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return {"task": task}


@router.post("/tasks/{task_id}/retry")
async def task_retry(task_id: str):
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    kind = str(task.get("kind") or "")
    payload = dict(task.get("payload") or {})
    if kind == "chat":
        question = str(payload.get("question") or "").strip()
        if not question:
            raise HTTPException(400, "Chat task missing question payload")
        return chat_start(ChatStartRequest(question=question))
    if kind == "media":
        media_kind = str(payload.get("kind") or "image")
        if media_kind == "image":
            return await create_image_job(
                str(payload.get("prompt") or ""),
                str(payload.get("style") or "教育海报"),
                str(payload.get("aspect_ratio") or "16:9"),
            )
        return await create_video_job(
            str(payload.get("prompt") or ""),
            str(payload.get("mode") or "text-to-video"),
            int(payload.get("duration_seconds") or 6),
            payload.get("source_image_url"),
        )
    if kind == "scheduler":
        return generate_daily_plan(force=True)
    raise HTTPException(400, f"Task kind not retryable: {kind}")


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
    return generate_daily_plan(force=True)
