"""系统默认今日任务时间表（教育辅导场景），按当前时间计算亮灯状态。"""
from datetime import date, datetime, time

# 默认一日工作安排（可按需扩展）
DEFAULT_SCHEDULE: list[dict] = [
    {
        "id": "t08_00",
        "time": "08:00",
        "title": "晨间准备",
        "action": "梳理当日学生名单、辅导重点与待回复咨询",
    },
    {
        "id": "t08_30",
        "time": "08:30",
        "title": "学生一对一辅导",
        "action": "完成预约学生的定点辅导与学情记录",
    },
    {
        "id": "t10_00",
        "time": "10:00",
        "title": "回答学生咨询",
        "action": "在线答疑：学习问题、作业思路、心理困扰初筛",
    },
    {
        "id": "t11_30",
        "time": "11:30",
        "title": "作业批改与反馈",
        "action": "批改作业、标注薄弱知识点并推送巩固建议",
    },
    {
        "id": "t14_00",
        "time": "14:00",
        "title": "知识点精讲",
        "action": "针对班级共性难点做微课或讲义补充",
    },
    {
        "id": "t15_30",
        "time": "15:30",
        "title": "心理辅导跟进",
        "action": "跟进个案访谈记录，必要时建议转介",
    },
    {
        "id": "t17_00",
        "time": "17:00",
        "title": "知识库维护",
        "action": "更新教辅资料入库，检查智能体检索效果",
    },
    {
        "id": "t18_30",
        "time": "18:30",
        "title": "当日教学复盘",
        "action": "汇总今日辅导与咨询，撰写简报",
    },
    {
        "id": "t20_00",
        "time": "20:00",
        "title": "明日课程准备",
        "action": "备课、打印学案、预设明日答疑要点",
    },
]


def _parse_hhmm(value: str) -> time:
    h, m = value.split(":")
    return time(int(h), int(m))


def _task_window(tasks: list[dict], index: int, today: date) -> tuple[datetime, datetime]:
    start_t = _parse_hhmm(tasks[index]["time"])
    start = datetime.combine(today, start_t)
    if index + 1 < len(tasks):
        end_t = _parse_hhmm(tasks[index + 1]["time"])
        end = datetime.combine(today, end_t)
    else:
        end = datetime.combine(today, time(23, 59, 59))
    return start, end


def get_today_schedule(now: datetime | None = None) -> dict:
    now = now or datetime.now()
    today = now.date()
    tasks_out: list[dict] = []
    active_task_id: str | None = None

    for i, task in enumerate(DEFAULT_SCHEDULE):
        start, end = _task_window(DEFAULT_SCHEDULE, i, today)
        if now < start:
            status = "upcoming"
        elif start <= now < end:
            status = "active"
            active_task_id = task["id"]
        else:
            status = "completed"

        tasks_out.append(
            {
                **task,
                "status": status,
                "start_at": start.isoformat(timespec="minutes"),
                "end_at": end.isoformat(timespec="minutes"),
            }
        )

    return {
        "date": today.isoformat(),
        "title": f"今日任务安排 · {today.isoformat()}",
        "source": "system_default",
        "server_time": now.isoformat(timespec="seconds"),
        "active_task_id": active_task_id,
        "tasks": tasks_out,
    }


def ensure_daily_schedule_on_startup() -> dict:
    """服务启动时确保返回当日默认安排（无需 LLM）。"""
    return get_today_schedule()
