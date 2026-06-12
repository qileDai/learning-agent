import json
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except Exception:
    BackgroundScheduler = None

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

from app.config import BACKEND_ROOT, settings
from app.task_store import complete_task, fail_task, start_task

PUSH_STORE = BACKEND_ROOT / "data" / "daily_push.json"
_scheduler: Any = None


class _FallbackChatModel:
    def __init__(self, model: str, api_key: str, api_base: str, temperature: float) -> None:
        self.model = model
        self.temperature = temperature

    def invoke(self, messages: list[Any]) -> SimpleNamespace:
        today = datetime.now().strftime("%Y-%m-%d")
        content = json.dumps(
            {
                "title": f"今日学习安排 · {today}",
                "summary": "当前环境未连接外部模型服务，已返回本地默认学习计划。",
                "tasks": [
                    {"time": "07:30", "subject": "晨间复习", "action": "回顾昨天错题与重点概念", "duration_minutes": 25},
                    {"time": "09:00", "subject": "上午重点", "action": "学习一个核心知识点并整理笔记", "duration_minutes": 60},
                    {"time": "15:00", "subject": "下午练习", "action": "完成配套练习并记录不会的题目", "duration_minutes": 45},
                    {"time": "20:00", "subject": "晚间复盘", "action": "总结今日收获并规划明日重点", "duration_minutes": 20},
                ],
            },
            ensure_ascii=False,
        )
        return SimpleNamespace(content=content)


def _get_llm() -> Any:
    if ChatOpenAI is not None:
        return ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key or "dummy",
            openai_api_base=settings.openai_api_base,
            temperature=0.5,
        )
    return _FallbackChatModel(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
        temperature=0.5,
    )


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
        llm = _get_llm()
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
    if _scheduler is not None or BackgroundScheduler is None:
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
