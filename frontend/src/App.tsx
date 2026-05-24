import { useEffect, useState } from "react";
import {
  chatResume,
  chatStart,
  generateImage,
  generateVideo,
  getMediaJob,
  ingestKnowledge,
  type GraphSummary,
  type MediaJob,
  type RetrievedChunk,
} from "./api";
import DailySchedulePanel from "./DailySchedule";
import "./App.css";

type Message = { role: "user" | "assistant"; content: string };

type VideoMode = "text-to-video" | "image-to-video";

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

  const [mediaPrompt, setMediaPrompt] = useState("");
  const [imageStyle, setImageStyle] = useState("课堂活动宣传海报");
  const [aspectRatio, setAspectRatio] = useState("16:9");
  const [videoMode, setVideoMode] = useState<VideoMode>("text-to-video");
  const [durationSeconds, setDurationSeconds] = useState(6);
  const [sourceImageUrl, setSourceImageUrl] = useState("");
  const [mediaJob, setMediaJob] = useState<MediaJob | null>(null);
  const [mediaError, setMediaError] = useState<string | null>(null);
  const [mediaLoading, setMediaLoading] = useState(false);

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

  const handleGenerateImage = async () => {
    const prompt = mediaPrompt.trim();
    if (!prompt || mediaLoading) return;
    setMediaLoading(true);
    setMediaError(null);
    try {
      const job = await generateImage(prompt, imageStyle.trim() || "教育海报", aspectRatio);
      setMediaJob(job);
      if (job.image_url) setSourceImageUrl(job.image_url);
    } catch (e) {
      setMediaError(e instanceof Error ? e.message : String(e));
    } finally {
      setMediaLoading(false);
    }
  };

  const handleGenerateVideo = async () => {
    const prompt = mediaPrompt.trim();
    if (!prompt || mediaLoading) return;
    setMediaLoading(true);
    setMediaError(null);
    try {
      const job = await generateVideo(
        prompt,
        videoMode,
        durationSeconds,
        videoMode === "image-to-video" ? sourceImageUrl.trim() : undefined
      );
      setMediaJob(job);
    } catch (e) {
      setMediaError(e instanceof Error ? e.message : String(e));
    } finally {
      setMediaLoading(false);
    }
  };

  useEffect(() => {
    if (!mediaJob || mediaJob.status !== "processing") return;
    const timer = window.setInterval(async () => {
      try {
        const next = await getMediaJob(mediaJob.job_id);
        setMediaJob(next);
      } catch (e) {
        setMediaError(e instanceof Error ? e.message : String(e));
      }
    }, 3000);
    return () => window.clearInterval(timer);
  }, [mediaJob]);

  return (
    <div className="layout">
      <aside className="sidebar">
        <header className="brand">
          <span className="logo">📚</span>
          <div>
            <h1>教育智能体</h1>
          </div>
        </header>

        <section className="panel media-pitch">
          <h2>AI 素材生成</h2>
          <p className="hint">一句话就能生成课堂宣传图、教学短视频，再也不愁没有宣传素材。</p>
          <ul className="feature-list">
            <li>文本生成图片：适合课堂活动海报、招生宣传图、课程封面</li>
            <li>文本生成视频：适合教学短视频、活动预告、课程导入片头</li>
            <li>图片转视频：适合把海报、封面图继续扩展成动态短片</li>
          </ul>
        </section>

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
        <section className="media-studio">
          <div className="media-studio__header">
            <div>
              <h2>AI 图片 / 视频工作台</h2>
              <p>老师只要描述想表达什么，就能快速生成宣传图或教学视频素材。</p>
            </div>
            {mediaJob?.message ? <span className="media-badge">{mediaJob.message}</span> : null}
          </div>

          <div className="media-grid">
            <div className="media-form card">
              <label className="field">
                <span>素材文案</span>
                <textarea
                  value={mediaPrompt}
                  onChange={(e) => setMediaPrompt(e.target.value)}
                  placeholder="例如：为小学历史公开课生成一张课堂活动宣传图，突出中国古代文明、互动闯关、金色国风海报感。"
                  rows={4}
                />
              </label>

              <div className="field-row">
                <label className="field">
                  <span>图片风格</span>
                  <input value={imageStyle} onChange={(e) => setImageStyle(e.target.value)} placeholder="课堂活动宣传海报" />
                </label>
                <label className="field">
                  <span>图片比例</span>
                  <select value={aspectRatio} onChange={(e) => setAspectRatio(e.target.value)}>
                    <option value="16:9">16:9</option>
                    <option value="4:3">4:3</option>
                    <option value="1:1">1:1</option>
                    <option value="9:16">9:16</option>
                  </select>
                </label>
              </div>

              <div className="field-row">
                <label className="field">
                  <span>视频模式</span>
                  <select value={videoMode} onChange={(e) => setVideoMode(e.target.value as VideoMode)}>
                    <option value="text-to-video">文本生成视频</option>
                    <option value="image-to-video">图片转视频</option>
                  </select>
                </label>
                <label className="field">
                  <span>视频时长</span>
                  <select
                    value={durationSeconds}
                    onChange={(e) => setDurationSeconds(Number(e.target.value))}
                  >
                    <option value={4}>4 秒</option>
                    <option value={6}>6 秒</option>
                    <option value={8}>8 秒</option>
                    <option value={10}>10 秒</option>
                  </select>
                </label>
              </div>

              {videoMode === "image-to-video" && (
                <label className="field">
                  <span>源图片地址</span>
                  <input
                    value={sourceImageUrl}
                    onChange={(e) => setSourceImageUrl(e.target.value)}
                    placeholder="先生成图片，或粘贴已有图片 URL"
                  />
                </label>
              )}

              <div className="media-actions">
                <button
                  type="button"
                  className="btn primary"
                  onClick={handleGenerateImage}
                  disabled={mediaLoading || !mediaPrompt.trim()}
                >
                  {mediaLoading ? "生成中…" : "生成图片"}
                </button>
                <button
                  type="button"
                  className="btn secondary media-action-secondary"
                  onClick={handleGenerateVideo}
                  disabled={
                    mediaLoading ||
                    !mediaPrompt.trim() ||
                    (videoMode === "image-to-video" && !sourceImageUrl.trim())
                  }
                >
                  生成视频
                </button>
              </div>

              <div className="scene-list">
                <button
                  type="button"
                  className="scene-chip"
                  onClick={() => {
                    setMediaPrompt("为七年级历史公开课设计一张课堂活动宣传图，突出古代中国文明、团队闯关、金色国风风格。");
                    setImageStyle("国风课堂海报");
                    setAspectRatio("16:9");
                  }}
                >
                  课堂活动宣传图
                </button>
                <button
                  type="button"
                  className="scene-chip"
                  onClick={() => {
                    setMediaPrompt("生成一条 6 秒教学短视频，展示老师讲解牛顿第二定律、学生互动实验、结尾出现课堂口号。");
                    setVideoMode("text-to-video");
                    setDurationSeconds(6);
                  }}
                >
                  教学短视频
                </button>
              </div>
            </div>

            <div className="media-preview card">
              <div className="preview-head">
                <h3>生成结果</h3>
                {mediaJob ? (
                  <span className="preview-meta">
                    {mediaJob.provider}
                    {mediaJob.provider_model ? ` · ${mediaJob.provider_model}` : ""}
                  </span>
                ) : null}
              </div>

              {!mediaJob && <p className="placeholder">输入一句话，立即生成宣传图或视频素材。</p>}

              {mediaJob?.image_url && (
                <div className="media-block">
                  <img className="media-image" src={mediaJob.image_url} alt={mediaJob.prompt} />
                  <a href={mediaJob.image_url} target="_blank" rel="noreferrer">
                    打开图片
                  </a>
                </div>
              )}

              {mediaJob?.preview_url && !mediaJob.image_url && (
                <div className="media-block">
                  <img className="media-image" src={mediaJob.preview_url} alt={mediaJob.prompt} />
                </div>
              )}

              {mediaJob?.video_url && (
                <div className="media-block">
                  <video className="media-video" src={mediaJob.video_url} controls playsInline />
                  <a href={mediaJob.video_url} target="_blank" rel="noreferrer">
                    打开视频
                  </a>
                </div>
              )}

              {mediaJob?.storyboard && (
                <div className="storyboard">
                  <h4>视频分镜</h4>
                  <pre>{mediaJob.storyboard}</pre>
                </div>
              )}

              {mediaJob?.status === "processing" && <p className="status">素材任务处理中，系统会自动轮询结果。</p>}
              {mediaError && <p className="error media-error">{mediaError}</p>}
            </div>
          </div>
        </section>

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
