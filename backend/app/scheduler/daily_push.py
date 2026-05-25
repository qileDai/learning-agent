import json
import uuid
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import BACKEND_ROOT, settings
from app.task_store import complete_task, fail_task, start_task

PUSH_STORE = BACKEND_ROOT / "data" / "daily_push.json"
_scheduler: BackgroundScheduler | None = None


def _load_history() -> list[dict]:
    if PUSH_STORE.exists():
        return json.loads(PUSH_STORE.read_text(encoding="utf-8"))
    return []


def _save_history(items: list[dict]) -> None:
    PUSH_STORE.parent.mkdir(parents=True, exist_ok=True)
    PUSH_STORE.write_text(json.dumps(items[-30:], ensure_ascii=False, indent=2), encoding="utf-8")


def generate_daily_plan(force: bool = False) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    task_id = f"scheduler:{today}:{uuid.uuid4().hex[:8]}" if force else f"scheduler:{today}"
    start_task(task_id, kind="scheduler", title=f"每日学习推送 {today}", payload={"date": today, "force": force})
    try:
        llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key or "dummy",
            openai_api_base=settings.openai_api_base,
            temperature=0.5,
        )
        system = (
            "你是学习规划助手。根据通用 K12 与大学基础课程学习规律，"
            "生成今日 AI 学习工作安排（中文），包含：晨间复习、上午重点、下午练习、晚间复盘。"
            "输出 JSON：title, summary, tasks(数组，每项含 time, subject, action, duration_minutes)"
        )
        user = f"请为 {today} 生成一份教育类每日学习推送安排。"
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        text = resp.content if hasattr(resp, "content") else str(resp)
        item = {
            "task_id": task_id,
            "date": today,
            "created_at": datetime.now().isoformat(),
            "content": text,
            "title": f"今日学习安排 · {today}",
        }
        history = _load_history()
        history = [h for h in history if h.get("date") != today or force]
        history.append(item)
        _save_history(history)
        complete_task(task_id, result=item)
        return item
    except Exception as exc:
        fail_task(task_id, error_code="SCHEDULER_GENERATE_FAILED", error_message=str(exc))
        raise


def get_latest_push() -> dict | None:
    from app.scheduler.daily_schedule import get_today_schedule

    return get_today_schedule()


def get_push_history(limit: int = 7) -> list[dict]:
    return _load_history()[-limit:]


def start_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        return
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        generate_daily_plan,
        "cron",
        hour=settings.daily_push_cron_hour,
        minute=settings.daily_push_cron_minute,
        id="daily_push",
    )
    _scheduler.start()


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
