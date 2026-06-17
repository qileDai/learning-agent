from __future__ import annotations

import json
import sqlite3
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
_STORE_READY = False


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _normalize_status(status: str | None) -> str:
    value = str(status or "pending").strip().lower()
    return value if value in _ALLOWED_STATUS else "pending"


def _use_sqlite() -> bool:
    return str(settings.task_store_backend or "sqlite").strip().lower() == "sqlite"


def _task_file() -> Path:
    return Path(settings.task_store_file)


def _db_file() -> Path:
    return Path(settings.task_store_db_file)


def _empty_store() -> dict[str, Any]:
    return {"tasks": {}, "events": {}}


def _load_json_store() -> dict[str, Any]:
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


def _save_json_store(store: dict[str, Any]) -> None:
    path = _task_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def _decode_json_blob(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        value = json.loads(text)
    except Exception:
        return default
    return value if isinstance(value, type(default)) else default


def _connect() -> sqlite3.Connection:
    path = _db_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def _ensure_sqlite_store() -> None:
    global _STORE_READY
    if _STORE_READY:
        return
    with _TASK_LOCK:
        if _STORE_READY:
            return
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    thread_id TEXT,
                    source_id TEXT,
                    tenant_id TEXT,
                    owner TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_steps INTEGER NOT NULL DEFAULT 0,
                    current_step INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_events (
                    event_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    node TEXT,
                    status TEXT,
                    message TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            _ensure_column(conn, "tasks", "tenant_id", "tenant_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_kind_status_updated ON tasks(kind, status, updated_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_tenant_status_updated ON tasks(tenant_id, status, updated_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_events_task_created ON task_events(task_id, created_at)")
            count = conn.execute("SELECT COUNT(1) FROM tasks").fetchone()[0]
            if count == 0 and settings.task_store_migrate_legacy_json:
                store = _load_json_store()
                for task in store["tasks"].values():
                    payload = dict(task.get("payload") or {})
                    result = dict(task.get("result") or {})
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO tasks (
                            task_id, kind, title, status, payload_json, result_json, thread_id, source_id, tenant_id, owner,
                            retry_count, max_steps, current_step, error_code, error_message, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(task.get("task_id") or ""),
                            str(task.get("kind") or "generic"),
                            str(task.get("title") or task.get("task_id") or "task"),
                            _normalize_status(task.get("status")),
                            json.dumps(payload, ensure_ascii=False),
                            json.dumps(result, ensure_ascii=False),
                            task.get("thread_id"),
                            task.get("source_id"),
                            task.get("tenant_id") or payload.get("tenant_id") or settings.default_tenant_id,
                            task.get("owner") or payload.get("submitted_by"),
                            int(task.get("retry_count") or 0),
                            int(task.get("max_steps") or 0),
                            int(task.get("current_step") or 0),
                            task.get("error_code"),
                            task.get("error_message"),
                            str(task.get("created_at") or _now_iso()),
                            str(task.get("updated_at") or task.get("created_at") or _now_iso()),
                        ),
                    )
                for task_id, events in store["events"].items():
                    for event in events or []:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO task_events (
                                event_id, task_id, type, node, status, message, data_json, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(event.get("event_id") or uuid.uuid4().hex),
                                str(event.get("task_id") or task_id),
                                str(event.get("type") or "info"),
                                event.get("node"),
                                _normalize_status(event.get("status")) if event.get("status") else None,
                                str(event.get("message") or ""),
                                json.dumps(event.get("data") or {}, ensure_ascii=False),
                                str(event.get("created_at") or _now_iso()),
                            ),
                        )
                conn.commit()
        _STORE_READY = True


def _task_from_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    row_keys = set(row.keys())
    return {
        "task_id": row["task_id"],
        "kind": row["kind"],
        "title": row["title"],
        "status": row["status"],
        "payload": _decode_json_blob(row["payload_json"], {}),
        "result": _decode_json_blob(row["result_json"], {}),
        "thread_id": row["thread_id"],
        "source_id": row["source_id"],
        "tenant_id": row["tenant_id"] if "tenant_id" in row_keys else settings.default_tenant_id,
        "owner": row["owner"],
        "retry_count": int(row["retry_count"] or 0),
        "max_steps": int(row["max_steps"] or 0),
        "current_step": int(row["current_step"] or 0),
        "error_code": row["error_code"],
        "error_message": row["error_message"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "event_id": row["event_id"],
        "task_id": row["task_id"],
        "type": row["type"],
        "node": row["node"],
        "status": row["status"],
        "message": row["message"],
        "data": _decode_json_blob(row["data_json"], {}),
        "created_at": row["created_at"],
    }


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
    tenant_id: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    retry_count: int | None = None,
    max_steps: int | None = None,
    current_step: int | None = None,
    owner: str | None = None,
) -> dict[str, Any]:
    with _TASK_LOCK:
        resolved_tenant = str(tenant_id or settings.default_tenant_id).strip() or settings.default_tenant_id
        if _use_sqlite():
            _ensure_sqlite_store()
            now = _now_iso()
            with _connect() as conn:
                existing = _task_from_row(conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone())
                if not existing:
                    task = {
                        "task_id": task_id,
                        "kind": kind,
                        "title": title,
                        "status": _normalize_status(status),
                        "payload": payload or {},
                        "result": result or {},
                        "thread_id": thread_id,
                        "source_id": source_id,
                        "tenant_id": resolved_tenant,
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
                    task = dict(existing)
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
                    if tenant_id is not None:
                        task["tenant_id"] = resolved_tenant
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
                conn.execute(
                    """
                    INSERT OR REPLACE INTO tasks (
                        task_id, kind, title, status, payload_json, result_json, thread_id, source_id, tenant_id, owner,
                        retry_count, max_steps, current_step, error_code, error_message, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task["task_id"],
                        task["kind"],
                        task["title"],
                        task["status"],
                        json.dumps(task.get("payload") or {}, ensure_ascii=False),
                        json.dumps(task.get("result") or {}, ensure_ascii=False),
                        task.get("thread_id"),
                        task.get("source_id"),
                        task.get("tenant_id") or resolved_tenant,
                        task.get("owner"),
                        int(task.get("retry_count") or 0),
                        int(task.get("max_steps") or 0),
                        int(task.get("current_step") or 0),
                        task.get("error_code"),
                        task.get("error_message"),
                        task.get("created_at") or now,
                        task.get("updated_at") or now,
                    ),
                )
                conn.commit()
                return task
        store = _load_json_store()
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
                "tenant_id": resolved_tenant,
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
            if tenant_id is not None:
                task["tenant_id"] = resolved_tenant
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
        _save_json_store(store)
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
        if _use_sqlite():
            _ensure_sqlite_store()
            with _connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO task_events (event_id, task_id, type, node, status, message, data_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event["event_id"],
                        task_id,
                        event["type"],
                        event["node"],
                        event["status"],
                        event["message"],
                        json.dumps(event["data"], ensure_ascii=False),
                        event["created_at"],
                    ),
                )
                overflow = conn.execute(
                    "SELECT event_id FROM task_events WHERE task_id = ? ORDER BY created_at DESC LIMIT -1 OFFSET ?",
                    (task_id, int(settings.task_event_limit)),
                ).fetchall()
                if overflow:
                    conn.executemany("DELETE FROM task_events WHERE event_id = ?", [(row["event_id"],) for row in overflow])
                if status:
                    conn.execute(
                        "UPDATE tasks SET status = ?, updated_at = ? WHERE task_id = ?",
                        (_normalize_status(status), event["created_at"], task_id),
                    )
                else:
                    conn.execute("UPDATE tasks SET updated_at = ? WHERE task_id = ?", (event["created_at"], task_id))
                conn.commit()
            return event
        store = _load_json_store()
        events = store["events"]
        task_events = list(events.get(task_id) or [])
        task_events.append(event)
        task_events = task_events[-int(settings.task_event_limit) :]
        events[task_id] = task_events
        task = dict(store["tasks"].get(task_id) or {})
        if task:
            task["updated_at"] = event["created_at"]
            if status:
                task["status"] = _normalize_status(status)
            store["tasks"][task_id] = task
        _save_json_store(store)
        return event


def get_task(task_id: str) -> dict[str, Any] | None:
    with _TASK_LOCK:
        if _use_sqlite():
            _ensure_sqlite_store()
            with _connect() as conn:
                return _task_from_row(conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone())
        return dict(_load_json_store()["tasks"].get(task_id) or {}) or None


def get_task_events(task_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    with _TASK_LOCK:
        if _use_sqlite():
            _ensure_sqlite_store()
            query = "SELECT * FROM task_events WHERE task_id = ? ORDER BY created_at ASC"
            params: list[Any] = [task_id]
            if limit is not None and limit > 0:
                query = "SELECT * FROM task_events WHERE task_id = ? ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
            with _connect() as conn:
                rows = conn.execute(query, tuple(params)).fetchall()
            items = [_event_from_row(row) for row in rows]
            return list(reversed(items)) if limit is not None and limit > 0 else items
        events = list(_load_json_store()["events"].get(task_id) or [])
    if limit is not None and limit > 0:
        return events[-limit:]
    return events


def list_tasks(
    kind: str | None = None,
    status: str | None = None,
    limit: int = 50,
    tenant_id: str | None = None,
    owner: str | None = None,
) -> list[dict[str, Any]]:
    with _TASK_LOCK:
        if _use_sqlite():
            _ensure_sqlite_store()
            clauses: list[str] = []
            params: list[Any] = []
            if kind:
                clauses.append("kind = ?")
                params.append(kind)
            if status:
                clauses.append("status = ?")
                params.append(_normalize_status(status))
            if tenant_id:
                clauses.append("tenant_id = ?")
                params.append(tenant_id)
            if owner:
                clauses.append("owner = ?")
                params.append(owner)
            where_clause = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            params.append(max(1, limit))
            query = f"SELECT * FROM tasks{where_clause} ORDER BY updated_at DESC LIMIT ?"
            with _connect() as conn:
                rows = conn.execute(query, tuple(params)).fetchall()
            return [_task_from_row(row) for row in rows if _task_from_row(row)]
        items = list(_load_json_store()["tasks"].values())
    if kind:
        items = [item for item in items if str(item.get("kind") or "") == kind]
    if status:
        target = _normalize_status(status)
        items = [item for item in items if str(item.get("status") or "") == target]
    if tenant_id:
        items = [item for item in items if str(item.get("tenant_id") or settings.default_tenant_id) == tenant_id]
    if owner:
        items = [item for item in items if str(item.get("owner") or "") == owner]
    items.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return items[: max(1, limit)]


def start_task(
    task_id: str,
    *,
    kind: str,
    title: str,
    payload: dict[str, Any] | None = None,
    thread_id: str | None = None,
    source_id: str | None = None,
    max_steps: int | None = None,
    tenant_id: str | None = None,
    owner: str | None = None,
) -> dict[str, Any]:
    task = upsert_task(
        task_id,
        kind=kind,
        title=title,
        status="running",
        payload=payload or {},
        thread_id=thread_id,
        source_id=source_id,
        tenant_id=tenant_id,
        owner=owner,
        max_steps=max_steps,
        current_step=0,
        error_code=None,
        error_message=None,
    )
    append_task_event(task_id, "task_started", message=f"任务已启动：{title}", status="running")
    return task


def mark_task_status(
    task_id: str,
    *,
    status: str,
    message: str,
    result: dict[str, Any] | None = None,
    current_step: int | None = None,
    retry_count: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any] | None:
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
        tenant_id=task.get("tenant_id") or settings.default_tenant_id,
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


def retry_task(task_id: str, *, message: str, current_step: int | None = None, error_code: str | None = None, error_message: str | None = None) -> dict[str, Any] | None:
    task = get_task(task_id)
    if not task:
        return None
    retry_count = int(task.get("retry_count") or 0) + 1
    return mark_task_status(
        task_id,
        status="retrying",
        message=message,
        current_step=current_step,
        retry_count=retry_count,
        error_code=error_code,
        error_message=error_message,
    )


def cancel_task(task_id: str, *, message: str = "任务已取消") -> dict[str, Any] | None:
    return mark_task_status(task_id, status="cancelled", message=message, error_code="TASK_CANCELLED", error_message=message)
