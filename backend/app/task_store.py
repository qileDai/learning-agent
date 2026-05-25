from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import settings

_TASK_LOCK = threading.RLock()
_ALLOWED_STATUS = {
    "pending",
    "running",
    "awaiting_input",
    "retrying",
    "completed",
    "failed",
    "timeout",
    "cancelled",
}


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _task_file() -> Path:
    return Path(settings.task_store_file)


def _empty_store() -> dict[str, Any]:
    return {"tasks": {}, "events": {}}


def _load_store() -> dict[str, Any]:
    path = _task_file()
    if not path.exists():
        return _empty_store()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_store()
    if not isinstance(data, dict):
        return _empty_store()
    tasks = data.get("tasks") if isinstance(data.get("tasks"), dict) else {}
    events = data.get("events") if isinstance(data.get("events"), dict) else {}
    return {"tasks": tasks, "events": events}


def _save_store(store: dict[str, Any]) -> None:
    path = _task_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_status(status: str | None) -> str:
    value = str(status or "pending").strip().lower()
    return value if value in _ALLOWED_STATUS else "pending"


def upsert_task(
    task_id: str,
    *,
    kind: str,
    title: str,
    status: str = "pending",
    payload: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    thread_id: str | None = None,
    source_id: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    retry_count: int | None = None,
    max_steps: int | None = None,
    current_step: int | None = None,
    owner: str | None = None,
) -> dict[str, Any]:
    with _TASK_LOCK:
        store = _load_store()
        tasks = store["tasks"]
        now = _now_iso()
        task = dict(tasks.get(task_id) or {})
        if not task:
            task = {
                "task_id": task_id,
                "kind": kind,
                "title": title,
                "status": _normalize_status(status),
                "payload": payload or {},
                "result": result or {},
                "thread_id": thread_id,
                "source_id": source_id,
                "owner": owner,
                "retry_count": retry_count or 0,
                "max_steps": max_steps or 0,
                "current_step": current_step or 0,
                "error_code": error_code,
                "error_message": error_message,
                "created_at": now,
                "updated_at": now,
            }
        else:
            task["kind"] = kind or task.get("kind")
            task["title"] = title or task.get("title")
            task["status"] = _normalize_status(status or task.get("status"))
            if payload is not None:
                task["payload"] = payload
            if result is not None:
                task["result"] = result
            if thread_id is not None:
                task["thread_id"] = thread_id
            if source_id is not None:
                task["source_id"] = source_id
            if owner is not None:
                task["owner"] = owner
            if retry_count is not None:
                task["retry_count"] = retry_count
            if max_steps is not None:
                task["max_steps"] = max_steps
            if current_step is not None:
                task["current_step"] = current_step
            task["error_code"] = error_code
            task["error_message"] = error_message
            task["updated_at"] = now
        tasks[task_id] = task
        _save_store(store)
        return task


def append_task_event(
    task_id: str,
    event_type: str,
    *,
    message: str,
    data: dict[str, Any] | None = None,
    node: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    with _TASK_LOCK:
        store = _load_store()
        events = store["events"]
        task_events = list(events.get(task_id) or [])
        event = {
            "event_id": uuid.uuid4().hex,
            "task_id": task_id,
            "type": str(event_type or "info"),
            "node": node,
            "status": _normalize_status(status) if status else None,
            "message": message,
            "data": data or {},
            "created_at": _now_iso(),
        }
        task_events.append(event)
        task_events = task_events[-int(settings.task_event_limit) :]
        events[task_id] = task_events
        task = dict(store["tasks"].get(task_id) or {})
        if task:
            task["updated_at"] = event["created_at"]
            if status:
                task["status"] = _normalize_status(status)
            store["tasks"][task_id] = task
        _save_store(store)
        return event


def get_task(task_id: str) -> dict[str, Any] | None:
    with _TASK_LOCK:
        return dict(_load_store()["tasks"].get(task_id) or {}) or None


def get_task_events(task_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    with _TASK_LOCK:
        events = list(_load_store()["events"].get(task_id) or [])
    if limit is not None and limit > 0:
        return events[-limit:]
    return events


def list_tasks(kind: str | None = None, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    with _TASK_LOCK:
        items = list(_load_store()["tasks"].values())
    if kind:
        items = [item for item in items if str(item.get("kind") or "") == kind]
    if status:
        target = _normalize_status(status)
        items = [item for item in items if str(item.get("status") or "") == target]
    items.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return items[: max(1, limit)]


def start_task(task_id: str, *, kind: str, title: str, payload: dict[str, Any] | None = None, thread_id: str | None = None, source_id: str | None = None, max_steps: int | None = None) -> dict[str, Any]:
    task = upsert_task(
        task_id,
        kind=kind,
        title=title,
        status="running",
        payload=payload or {},
        thread_id=thread_id,
        source_id=source_id,
        max_steps=max_steps,
        current_step=0,
        error_code=None,
        error_message=None,
    )
    append_task_event(task_id, "task_started", message=f"任务已启动：{title}", status="running")
    return task


def mark_task_status(task_id: str, *, status: str, message: str, result: dict[str, Any] | None = None, current_step: int | None = None, retry_count: int | None = None, error_code: str | None = None, error_message: str | None = None) -> dict[str, Any] | None:
    task = get_task(task_id)
    if not task:
        return None
    updated = upsert_task(
        task_id,
        kind=str(task.get("kind") or "generic"),
        title=str(task.get("title") or task_id),
        status=status,
        payload=dict(task.get("payload") or {}),
        result=result if result is not None else dict(task.get("result") or {}),
        thread_id=task.get("thread_id"),
        source_id=task.get("source_id"),
        retry_count=retry_count if retry_count is not None else int(task.get("retry_count") or 0),
        max_steps=int(task.get("max_steps") or 0),
        current_step=current_step if current_step is not None else int(task.get("current_step") or 0),
        error_code=error_code,
        error_message=error_message,
        owner=task.get("owner"),
    )
    append_task_event(task_id, f"task_{_normalize_status(status)}", message=message, status=status, data=result or {})
    return updated


def complete_task(task_id: str, *, result: dict[str, Any] | None = None, current_step: int | None = None) -> dict[str, Any] | None:
    return mark_task_status(task_id, status="completed", message="任务执行完成", result=result, current_step=current_step)


def fail_task(task_id: str, *, error_code: str, error_message: str, current_step: int | None = None) -> dict[str, Any] | None:
    return mark_task_status(
        task_id,
        status="failed",
        message=f"任务执行失败：{error_message}",
        current_step=current_step,
        error_code=error_code,
        error_message=error_message,
    )


def timeout_task(task_id: str, *, error_message: str, current_step: int | None = None) -> dict[str, Any] | None:
    return mark_task_status(
        task_id,
        status="timeout",
        message=f"任务执行超时：{error_message}",
        current_step=current_step,
        error_code="TASK_TIMEOUT",
        error_message=error_message,
    )


def awaiting_input_task(task_id: str, *, message: str, current_step: int | None = None, result: dict[str, Any] | None = None) -> dict[str, Any] | None:
    return mark_task_status(task_id, status="awaiting_input", message=message, result=result, current_step=current_step)


def retry_task(task_id: str, *, message: str, current_step: int | None = None) -> dict[str, Any] | None:
    task = get_task(task_id)
    if not task:
        return None
    retry_count = int(task.get("retry_count") or 0) + 1
    return mark_task_status(task_id, status="retrying", message=message, current_step=current_step, retry_count=retry_count)


def cancel_task(task_id: str, *, message: str = "任务已取消") -> dict[str, Any] | None:
    return mark_task_status(task_id, status="cancelled", message=message)
