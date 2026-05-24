import { useState } from "react";
import {
  chatResume,
  chatStart,
  ingestKnowledge,
  type GraphSummary,
  type RetrievedChunk,
} from "./api";
import DailySchedulePanel from "./DailySchedule";
import "./App.css";

type Message = { role: "user" | "assistant"; content: string };

export default function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [chunks, setChunks] = useState<RetrievedChunk[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [phase, setPhase] = useState<"idle" | "select" | "loading" | "done">("idle");
  const [error, setError] = useState<string | null>(null);
  const [ingestStatus, setIngestStatus] = useState<string | null>(null);
  const [graphSummary, setGraphSummary] = useState<GraphSummary | null>(null);

  const formatGraphSummary = (summary?: GraphSummary | null) => {
    if (!summary?.matched_concepts?.length) return "";
    const matched = `图谱命中：${summary.matched_concepts.join("、")}`;
    const related = summary.related_concepts.length
      ? `；关联概念：${summary.related_concepts.slice(0, 4).join("、")}`
      : "";
    return `${matched}${related}`;
  };

  const handleIngest = async () => {
    setIngestStatus("正在入库…");
    try {
      const res = await ingestKnowledge(true);
      const hasXinli = res.xinli_included || res.sources?.some((s: string) => s.includes("xinli"));
      const graphPart = res.graph
        ? `；图谱 ${res.graph.concepts ?? 0} 概念 / ${res.graph.relations ?? 0} 关系`
        : "";
      setIngestStatus(
        `完成：${res.chunks} 块，${res.files} 文件${graphPart}${hasXinli ? "（含 xinli 心理文档）" : ""}`
      );
    } catch (e) {
      setIngestStatus(`失败：${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const handleAsk = async () => {
    const q = question.trim();
    if (!q || phase === "loading") return;
    setError(null);
    setPhase("loading");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setQuestion("");
    try {
      const res = await chatStart(q);
      setThreadId(res.thread_id);
      setGraphSummary(res.graph_summary ?? null);
      const graphHint = formatGraphSummary(res.graph_summary);
      const retrievedChunks = res.retrieved_chunks;
      if (res.status === "awaiting_selection" && res.kb_hit && retrievedChunks?.length) {
        setChunks(retrievedChunks);
        setSelectedId(retrievedChunks[0]?.id ?? null);
        setPhase("select");
        setMessages((m) => [
          ...m,
          {
            role: "assistant",
            content:
              (res.message ?? `知识库已匹配 ${retrievedChunks.length} 条资料，请单选 1 条。`) +
              (graphHint ? `\n\n${graphHint}` : ""),
          },
        ]);
      } else if (res.status === "completed" && res.answer) {
        setPhase("done");
        const note = res.mode === "llm" && res.message ? `${res.message}\n\n` : "";
        const graphPrefix = graphHint ? `${graphHint}\n\n` : "";
        setMessages((m) => [...m, { role: "assistant", content: graphPrefix + note + res.answer }]);
      } else if (res.answer) {
        setPhase("done");
        const graphPrefix = graphHint ? `${graphHint}\n\n` : "";
        setMessages((m) => [...m, { role: "assistant", content: graphPrefix + res.answer }]);
      } else {
        setPhase("idle");
        setMessages((m) => [
          ...m,
          {
            role: "assistant",
            content: res.message ?? "服务端未返回有效内容，请检查 API 与知识库配置。",
          },
        ]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("idle");
    }
  };

  const handleConfirmSelection = async () => {
    if (!threadId || !selectedId) return;
    setPhase("loading");
    setError(null);
    try {
      const res = await chatResume(threadId, [selectedId]);
      setPhase("done");
      setChunks([]);
      setGraphSummary(res.graph_summary ?? null);
      const graphHint = formatGraphSummary(res.graph_summary);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `${graphHint ? `${graphHint}\n\n` : ""}${res.answer || res.message || "未生成回答"}`,
        },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("select");
    }
  };

  return (
    <div className="layout">
      <aside className="sidebar">
        <header className="brand">
          <span className="logo">📚</span>
          <div>
            <h1>教育智能体</h1>
          </div>
        </header>

        <section className="panel">
          <h2>知识库</h2>
          <p className="hint">支持 PDF · Word · Markdown</p>
          <button type="button" className="btn secondary" onClick={handleIngest}>
            重建向量索引
          </button>
          {ingestStatus && <p className="status">{ingestStatus}</p>}
          {graphSummary?.matched_concepts?.length ? (
            <p className="status">{formatGraphSummary(graphSummary)}</p>
          ) : null}
        </section>

        <section className="panel push-panel">
          <h2>今日任务安排</h2>
          <p className="hint">系统默认时段 · 到点亮灯提示</p>
          <DailySchedulePanel />
        </section>
      </aside>

      <main className="chat">
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">
              <p>向智能体提问，例如：</p>
              <ul>
                <li>Python 列表推导式怎么写？</li>
                <li>牛顿第二定律公式是什么？</li>
                <li>古诗词中「月」意象常表示什么？</li>
              </ul>
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`bubble ${msg.role}`}>
              {msg.content}
            </div>
          ))}
        </div>

        {phase === "select" && chunks.length > 0 && (
          <section className="chunk-picker">
            <h3>单选一条参考资料</h3>
            <div className="chunk-list">
              {chunks.map((c) => (
                <label key={c.id} className={`chunk-item ${selectedId === c.id ? "selected" : ""}`}>
                  <input
                    type="radio"
                    name="kb-chunk"
                    checked={selectedId === c.id}
                    onChange={() => setSelectedId(c.id)}
                  />
                  <span className="meta">
                    {c.source} · {c.file_type}
                    {c.subject ? ` · ${c.subject}` : ""}
                    {c.chapter ? ` · ${c.chapter}` : ""}
                    {c.retrieval_mode ? ` · ${c.retrieval_mode}` : ""}
                  </span>
                  {c.concepts?.length ? <span className="meta">概念：{c.concepts.join("、")}</span> : null}
                  <p>
                    {c.content.slice(0, 200)}
                    {c.content.length > 200 ? "…" : ""}
                  </p>
                </label>
              ))}
            </div>
            <button
              type="button"
              className="btn primary"
              onClick={handleConfirmSelection}
              disabled={!selectedId}
            >
              基于所选资料生成解答
            </button>
          </section>
        )}

        {error && <p className="error">{error}</p>}

        <footer className="composer">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="输入学习问题…"
            rows={2}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleAsk();
              }
            }}
            disabled={phase === "loading" || phase === "select"}
          />
          <button
            type="button"
            className="btn primary"
            onClick={handleAsk}
            disabled={phase === "loading" || phase === "select" || !question.trim()}
          >
            {phase === "loading" ? "处理中…" : "提问"}
          </button>
        </footer>
      </main>
    </div>
  );
}
