import { useCallback, useEffect, useRef, useState } from "react";
import { getDailySchedule, type DailySchedule } from "./api";

export default function DailySchedulePanel() {
  const [schedule, setSchedule] = useState<DailySchedule | null>(null);
  const prevActiveRef = useRef<string | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await getDailySchedule();
      setSchedule(data);

      if (
        data.active_task_id &&
        data.active_task_id !== prevActiveRef.current &&
        typeof Notification !== "undefined" &&
        Notification.permission === "granted"
      ) {
        const task = data.tasks.find((t) => t.id === data.active_task_id);
        if (task) {
          new Notification("任务提醒", {
            body: `${task.time} ${task.title}：${task.action}`,
          });
        }
      }
      prevActiveRef.current = data.active_task_id;
    } catch {
      /* backend offline */
    }
  }, []);

  useEffect(() => {
    load();
    const timer = setInterval(load, 30_000);
    return () => clearInterval(timer);
  }, [load]);

  const requestNotify = () => {
    if (typeof Notification !== "undefined") {
      Notification.requestPermission();
    }
  };

  if (!schedule) {
    return <p className="hint">加载今日安排…</p>;
  }

  const now = new Date(schedule.server_time);

  return (
    <div className="schedule-panel">
      <p className="schedule-now">
        当前 {now.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}
      </p>
      <ul className="schedule-list">
        {schedule.tasks.map((task) => (
          <li
            key={task.id}
            className={`schedule-item schedule-item--${task.status}`}
          >
            <span
              className={`schedule-led schedule-led--${task.status}`}
              title={
                task.status === "active"
                  ? "进行中 — 请处理本项任务"
                  : task.status === "completed"
                    ? "已完成"
                    : "未到时间"
              }
            />
            <div className="schedule-body">
              <div className="schedule-head">
                <span className="schedule-time">{task.time}</span>
                <span className="schedule-title">{task.title}</span>
              </div>
              <p className="schedule-action">{task.action}</p>
            </div>
          </li>
        ))}
      </ul>
      <button type="button" className="btn-link" onClick={requestNotify}>
        开启浏览器到点提醒
      </button>
    </div>
  );
}
