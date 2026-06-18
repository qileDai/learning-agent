import { useCallback, useEffect, useMemo, useState } from "react";
import {
  chatResume,
  chatStart,
  generateImage,
  generateVideo,
  getChatState,
  getChatTaskDetail,
  getChatTasks,
  getDailyStockPicks,
  getFailureSamples,
  getMediaJob,
  getPendingChatTasks,
  ingestKnowledge,
  runRetrievalEval,
  type AnswerValidation,
  type ChatStateResponse,
  type ChatTask,
  type ChatTaskDetailResponse,
  type DailyStockResponse,
  type ExecutionTrace,
  type FailureSamplesResponse,
  type GraphSummary,
  type MediaJob,
  type RetrievalEvalCase,
  type RetrievalEvalResponse,
  type RetrievalSummary,
  type RetrievedChunk,
  type TaskEvent,
} from "./api";
import DailySchedulePanel from "./DailySchedule";
import "./App.css";

type Message = { role: "user" | "assistant"; content: string };

type VideoMode = "text-to-video" | "image-to-video";
type WorkspaceView = "education" | "stocks";

const TRACE_NODE_LABEL: Record<string, string> = {
  greeting: "寒暄识别",
  planner: "问题规划",
  retrieve: "知识检索",
  human_select: "人工选择",
  generate_answer: "基于知识生成",
  generate_llm: "大模型直答",
  critic: "结果评估",
};

const TRACE_STATUS_LABEL: Record<string, string> = {
  running: "执行中",
  completed: "已完成",
  awaiting_input: "等待选择",
  failed: "失败",
  timeout: "超时",
  cancelled: "已取消",
};

const TRACE_DATA_LABEL: Record<string, string> = {
  loop_step: "轮次",
  max_steps: "最大轮次",
  plan_question: "规划问题",
  kb_hit: "命中知识库",
  candidates: "候选数",
  route_type: "路由策略",
  selected_chunk_ids: "已选片段",
  answer_mode: "回答模式",
  grounding_score: "可信分",
  retry_count: "重试次数",
  retry_strategy: "重试策略",
  reason_code: "原因编码",
  citation_coverage: "证据覆盖",
  selected_source: "主证据",
  supporting_sources: "辅助证据",
};

export default function App() {
  const [activeView, setActiveView] = useState<WorkspaceView>("education");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [chunks, setChunks] = useState<RetrievedChunk[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [phase, setPhase] = useState<"idle" | "select" | "loading" | "done">("idle");
  const [error, setError] = useState<string | null>(null);
  const [ingestStatus, setIngestStatus] = useState<string | null>(null);
  const [graphSummary, setGraphSummary] = useState<GraphSummary | null>(null);
  const [retrievalSummary, setRetrievalSummary] = useState<RetrievalSummary | null>(null);
  const [answerValidation, setAnswerValidation] = useState<AnswerValidation | null>(null);
  const [executionTrace, setExecutionTrace] = useState<ExecutionTrace[]>([]);
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const [pendingChats, setPendingChats] = useState<ChatTask[]>([]);
  const [pendingLoading, setPendingLoading] = useState(false);
  const [taskHistory, setTaskHistory] = useState<ChatTask[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [resumeLoadingId, setResumeLoadingId] = useState<string | null>(null);
  const [inspectedTaskDetail, setInspectedTaskDetail] = useState<ChatTaskDetailResponse | null>(null);
  const [detailLoadingId, setDetailLoadingId] = useState<string | null>(null);
  const [failureSamples, setFailureSamples] = useState<FailureSamplesResponse | null>(null);
  const [failureLoading, setFailureLoading] = useState(false);
  const [failureEval, setFailureEval] = useState<RetrievalEvalResponse | null>(null);
  const [failureEvalLoading, setFailureEvalLoading] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);

  const [mediaPrompt, setMediaPrompt] = useState("");
  const [imageStyle, setImageStyle] = useState("课堂活动宣传海报");
  const [aspectRatio, setAspectRatio] = useState("16:9");
  const [videoMode, setVideoMode] = useState<VideoMode>("text-to-video");
  const [durationSeconds, setDurationSeconds] = useState(6);
  const [sourceImageUrl, setSourceImageUrl] = useState("");
  const [mediaJob, setMediaJob] = useState<MediaJob | null>(null);
  const [mediaError, setMediaError] = useState<string | null>(null);
  const [mediaLoading, setMediaLoading] = useState(false);
  const [stockBoard, setStockBoard] = useState<DailyStockResponse | null>(null);
  const [stockLoading, setStockLoading] = useState(false);
  const [stockError, setStockError] = useState<string | null>(null);

  const formatGraphSummary = (summary?: GraphSummary | null) => {
    if (!summary?.matched_concepts?.length) return "";
    const matched = `图谱命中：${summary.matched_concepts.join("、")}`;
    const related = summary.related_concepts.length
      ? `；关联概念：${summary.related_concepts.slice(0, 4).join("、")}`
      : "";
    return `${matched}${related}`;
  };

  const formatTaskTime = (value?: string) => {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const formatTaskStatus = (status?: string | null) => {
    if (!status) return "未知";
    return (
      {
        running: "运行中",
        awaiting_input: "待选择",
        retrying: "重试中",
        completed: "已完成",
        failed: "失败",
        timeout: "超时",
        cancelled: "已取消",
      }[status] ?? status
    );
  };

  const formatPercent = (value?: number | null) => {
    if (typeof value !== "number" || Number.isNaN(value)) return "--";
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatDecimal = (value?: number | null) => {
    if (typeof value !== "number" || Number.isNaN(value)) return "--";
    return value.toFixed(3);
  };

  const formatStockScore = (value?: number | null) => {
    if (typeof value !== "number" || Number.isNaN(value)) return "--";
    return value.toFixed(1);
  };

  const formatStockPercent = (value?: number | null) => {
    if (typeof value !== "number" || Number.isNaN(value)) return "--";
    return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
  };

  const formatStockPrice = (value?: number | null) => {
    if (typeof value !== "number" || Number.isNaN(value)) return "--";
    return `¥${value.toFixed(2)}`;
  };

  const formatStockMoney = (value?: number | null) => {
    if (typeof value !== "number" || Number.isNaN(value)) return "--";
    const abs = Math.abs(value);
    if (abs >= 100000000) return `${value >= 0 ? "+" : "-"}${(abs / 100000000).toFixed(2)}亿`;
    if (abs >= 10000) return `${value >= 0 ? "+" : "-"}${(abs / 10000).toFixed(2)}万`;
    return `${value >= 0 ? "+" : ""}${value.toFixed(0)}`;
  };

  const formatStockCap = (value?: number | null) => {
    if (typeof value !== "number" || Number.isNaN(value)) return "--";
    return `${(value / 100000000).toFixed(1)}亿`;
  };

  const formatStockCount = (value?: number | null) => {
    if (typeof value !== "number" || Number.isNaN(value)) return "--";
    if (Math.abs(value) >= 100000000) return `${(value / 100000000).toFixed(2)}亿`;
    if (Math.abs(value) >= 10000) return `${(value / 10000).toFixed(2)}万`;
    return value.toFixed(0);
  };

  const formatBoolean = (value: boolean) => (value ? "是" : "否");

  const formatJoined = (items?: Array<string | null | undefined>, empty = "--") => {
    const values = (items ?? []).map((item) => String(item || "").trim()).filter(Boolean);
    return values.length ? values.join("、") : empty;
  };

  const buildFailureEvalCases = (samples?: FailureSamplesResponse | null): RetrievalEvalCase[] => {
    return (samples?.items ?? [])
      .map((item) => ({
        question: item.question,
        expected_sources: item.retrieval_summary?.evidence_sources ?? [],
        expected_terms: item.missing_aspects ?? [],
        gold_answer: item.answer?.trim() ? item.answer : undefined,
      }))
      .filter((item) => item.question && ((item.expected_sources?.length ?? 0) > 0 || (item.expected_terms?.length ?? 0) > 0 || item.gold_answer))
      .slice(0, 8);
  };

  const normalizeTraceValue = (value: unknown): string => {
    if (value == null) return "";
    if (Array.isArray(value)) return value.map((item) => normalizeTraceValue(item)).filter(Boolean).join("、");
    if (typeof value === "boolean") return formatBoolean(value);
    if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  };

  const formatTraceDetails = (trace: ExecutionTrace) => {
    const entries = Object.entries(trace.data || {}).filter(([, value]) => {
      if (value == null || value === "") return false;
      if (Array.isArray(value) && value.length === 0) return false;
      return true;
    });
    if (!entries.length) return "";
    return entries
      .slice(0, 4)
      .map(([key, value]) => `${TRACE_DATA_LABEL[key] ?? key}：${normalizeTraceValue(value)}`)
      .join(" · ");
  };

  const applyDiagnostics = useCallback(
    (payload: {
      graphSummary?: GraphSummary | null;
      retrievalSummary?: RetrievalSummary | null;
      answerValidation?: AnswerValidation | null;
      executionTrace?: ExecutionTrace[] | null;
    }) => {
      setGraphSummary(payload.graphSummary ?? null);
      setRetrievalSummary(payload.retrievalSummary ?? null);
      setAnswerValidation(payload.answerValidation ?? null);
      setExecutionTrace(payload.executionTrace ?? []);
    },
    []
  );

  const clearDiagnostics = useCallback(() => {
    setGraphSummary(null);
    setRetrievalSummary(null);
    setAnswerValidation(null);
    setExecutionTrace([]);
    setDiagnosticsOpen(false);
  }, []);

  const refreshPendingChats = useCallback(async () => {
    setPendingLoading(true);
    try {
      const res = await getPendingChatTasks();
      setPendingChats(res.items ?? []);
    } catch {
      setPendingChats([]);
    } finally {
      setPendingLoading(false);
    }
  }, []);

  const refreshTaskHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const res = await getChatTasks({ limit: 12 });
      setTaskHistory((res.items ?? []).filter((task) => task.status !== "awaiting_input"));
    } catch {
      setTaskHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const refreshTaskLists = useCallback(async () => {
    await Promise.all([refreshPendingChats(), refreshTaskHistory()]);
  }, [refreshPendingChats, refreshTaskHistory]);

  const refreshFailureCases = useCallback(async () => {
    setFailureLoading(true);
    setEvalError(null);
    try {
      const res = await getFailureSamples(12, true);
      setFailureSamples(res);
      return res;
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setEvalError(message);
      return null;
    } finally {
      setFailureLoading(false);
    }
  }, []);

  const handleRunFailureEval = useCallback(async () => {
    setFailureEvalLoading(true);
    setEvalError(null);
    try {
      const samples = failureSamples ?? (await refreshFailureCases());
      const cases = buildFailureEvalCases(samples);
      if (!cases.length) {
        throw new Error("当前没有可回流的失败样本，先产生失败样本或刷新列表后再试。");
      }
      const result = await runRetrievalEval(cases, 3);
      setFailureEval(result);
    } catch (e) {
      setEvalError(e instanceof Error ? e.message : String(e));
    } finally {
      setFailureEvalLoading(false);
    }
  }, [failureSamples, refreshFailureCases]);

  useEffect(() => {
    void refreshTaskLists();
  }, [refreshTaskLists]);

  useEffect(() => {
    if (activeView !== "education" || failureSamples || failureLoading) return;
    void refreshFailureCases();
  }, [activeView, failureLoading, failureSamples, refreshFailureCases]);

  const refreshStockBoard = useCallback(async () => {
    setStockLoading(true);
    setStockError(null);
    try {
      const res = await getDailyStockPicks(10);
      setStockBoard(res);
    } catch (e) {
      setStockError(e instanceof Error ? e.message : String(e));
    } finally {
      setStockLoading(false);
    }
  }, []);

  useEffect(() => {
    if (activeView !== "stocks" || stockBoard || stockLoading) return;
    void refreshStockBoard();
  }, [activeView, refreshStockBoard, stockBoard, stockLoading]);

  const buildResumeMessage = (state: ChatStateResponse) => {
    const graphHint = formatGraphSummary(state.graph_summary ?? null);
    const base = state.task?.result?.message || "已恢复中断任务，请单选 1 条资料后继续生成解答。";
    return graphHint ? `${base}\n\n${graphHint}` : base;
  };

  const restorePendingTask = (state: ChatStateResponse) => {
    const restoredThreadId = state.thread_id || state.task_id;
    const restoredQuestion = state.task?.payload?.question || "已恢复中断任务";
    const restoredChunks = state.retrieved_chunks ?? [];
    setThreadId(restoredThreadId);
    setInspectedTaskDetail(null);
    applyDiagnostics({
      graphSummary: state.graph_summary ?? null,
      retrievalSummary: state.retrieval_summary ?? state.task?.result?.retrieval_summary ?? null,
      answerValidation: state.answer_validation ?? state.task?.result?.answer_validation ?? null,
      executionTrace: state.execution_trace ?? [],
    });
    setChunks(restoredChunks);
    setSelectedId(restoredChunks[0]?.id ?? null);
    setQuestion("");
    setPhase("select");
    setMessages([
      { role: "user", content: restoredQuestion },
      { role: "assistant", content: buildResumeMessage(state) },
    ]);
  };

  const handleOpenTaskDetail = async (task: ChatTask) => {
    setDetailLoadingId(task.task_id);
    setError(null);
    try {
      const detail = await getChatTaskDetail(task.task_id);
      setInspectedTaskDetail(detail);
      setDiagnosticsOpen(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDetailLoadingId(null);
    }
  };

  const handleResumePending = async (task: ChatTask) => {
    const resumeThreadId = task.thread_id || task.task_id;
    if (!resumeThreadId) return;
    setResumeLoadingId(task.task_id);
    setError(null);
    try {
      const state = await getChatState(resumeThreadId);
      if (!state.retrieved_chunks?.length) {
        throw new Error("该中断任务未找到可选知识片段，请重新提问。 ");
      }
      restorePendingTask(state);
    } catch (e) {
      setError(e instanceof Error ? e.message.trim() : String(e));
      await refreshTaskLists();
    } finally {
      setResumeLoadingId(null);
    }
  };

  const handlePauseSelection = async () => {
    if (!threadId) return;
    setPhase("idle");
    setChunks([]);
    setSelectedId(null);
    setMessages((m) => [
      ...m,
      {
        role: "assistant",
        content: "当前问答已暂存，可在左侧“待继续问答”中随时恢复。",
      },
    ]);
    await refreshTaskLists();
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
    setInspectedTaskDetail(null);
    clearDiagnostics();
    setPhase("loading");
    setMessages((m) => [...m, { role: "user", content: q }]);
    setQuestion("");
    try {
      const res = await chatStart(q);
      setThreadId(res.thread_id);
      applyDiagnostics({
        graphSummary: res.graph_summary ?? null,
        retrievalSummary: res.retrieval_summary ?? null,
        answerValidation: res.answer_validation ?? null,
        executionTrace: res.execution_trace ?? [],
      });
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
        await refreshTaskLists();
      } else if (res.status === "completed" && res.answer) {
        setPhase("done");
        setChunks([]);
        setSelectedId(null);
        const note = res.mode === "llm" && res.message ? `${res.message}\n\n` : "";
        const graphPrefix = graphHint ? `${graphHint}\n\n` : "";
        setMessages((m) => [...m, { role: "assistant", content: graphPrefix + note + res.answer }]);
        await refreshTaskLists();
      } else if (res.answer) {
        setPhase("done");
        setChunks([]);
        setSelectedId(null);
        const graphPrefix = graphHint ? `${graphHint}\n\n` : "";
        setMessages((m) => [...m, { role: "assistant", content: graphPrefix + res.answer }]);
        await refreshTaskLists();
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
    setInspectedTaskDetail(null);
    try {
      const res = await chatResume(threadId, [selectedId]);
      setPhase("done");
      setChunks([]);
      setSelectedId(null);
      applyDiagnostics({
        graphSummary: res.graph_summary ?? null,
        retrievalSummary: res.retrieval_summary ?? null,
        answerValidation: res.answer_validation ?? null,
        executionTrace: res.execution_trace ?? [],
      });
      const graphHint = formatGraphSummary(res.graph_summary);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `${graphHint ? `${graphHint}\n\n` : ""}${res.answer || res.message || "未生成回答"}`,
        },
      ]);
      await refreshTaskLists();
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

  const detailGraphState = inspectedTaskDetail?.graph_state;
  const detailTask = inspectedTaskDetail?.task ?? null;
  const displayedGraphSummary = detailGraphState?.graph_summary ?? graphSummary;
  const displayedRetrievalSummary =
    detailGraphState?.retrieval_summary ?? detailTask?.result?.retrieval_summary ?? retrievalSummary;
  const displayedAnswerValidation =
    detailGraphState?.answer_validation ?? detailTask?.result?.answer_validation ?? answerValidation;
  const displayedExecutionTrace =
    detailGraphState?.execution_trace ?? detailTask?.result?.execution_trace ?? executionTrace;
  const displayedFinalAnswer =
    detailGraphState?.final_answer ?? detailTask?.result?.final_answer ?? null;
  const displayedEvents: TaskEvent[] = inspectedTaskDetail?.events ?? [];

  const hasDiagnostics = Boolean(
    inspectedTaskDetail || displayedRetrievalSummary || displayedAnswerValidation || displayedExecutionTrace.length > 0
  );

  const traceItems = useMemo(
    () =>
      displayedExecutionTrace.map((trace, index) => ({
        key: `${trace.node}-${trace.step}-${index}`,
        nodeLabel: TRACE_NODE_LABEL[trace.node] ?? trace.node,
        statusLabel: TRACE_STATUS_LABEL[trace.status] ?? trace.status,
        details: formatTraceDetails(trace),
      })),
    [displayedExecutionTrace]
  );

  const failureReasonEntries = useMemo(() => Object.entries(failureSamples?.by_reason ?? {}).slice(0, 4), [failureSamples]);
  const failurePreviewItems = failureSamples?.items.slice(0, 3) ?? [];
  const failureEvalPreview = failureEval?.items.slice(0, 4) ?? [];
  const isSelecting = phase === "select" && chunks.length > 0;
  const workspacePhaseLabel =
    phase === "loading" ? "处理中" : phase === "select" ? "待选证据" : phase === "done" ? "已完成" : "待提问";

  return (
    <div className="layout">
      <aside className="sidebar">
        <header className="brand">
          <span className="logo">📚</span>
          <div>
            <h1>智能工作台</h1>
            <p>教育问答 · 素材生成 · 股票机会</p>
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

        <section className="panel pending-panel">
          <h2>待继续问答</h2>
          <p className="hint">支持前端中断后继续，恢复后可直接续选知识片段。</p>
          {phase === "select" && threadId ? (
            <button type="button" className="btn secondary pending-action" onClick={handlePauseSelection}>
              暂存当前问答
            </button>
          ) : null}
          {pendingLoading ? <p className="pending-empty">正在加载待处理任务…</p> : null}
          {!pendingLoading && pendingChats.length === 0 ? <p className="pending-empty">暂无待继续任务</p> : null}
          <div className="pending-list">
            {pendingChats.map((task) => {
              const resumeThreadId = task.thread_id || task.task_id;
              const questionText = task.payload?.question || task.title;
              const isActive = threadId === resumeThreadId && phase === "select";
              const isLoading = resumeLoadingId === task.task_id;
              return (
                <button
                  key={task.task_id}
                  type="button"
                  className={`pending-item${isActive ? " pending-item--active" : ""}`}
                  onClick={() => void handleResumePending(task)}
                  disabled={isLoading}
                >
                  <span className="pending-title">{questionText}</span>
                  <span className="pending-meta">
                    {isLoading ? "恢复中…" : `状态：待选择 · ${formatTaskTime(task.updated_at || task.created_at)}`}
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        <section className="panel pending-panel">
          <h2>最近问答</h2>
          <p className="hint">查看历史任务详情、评分结果和执行轨迹。</p>
          {historyLoading ? <p className="pending-empty">正在加载历史任务…</p> : null}
          {!historyLoading && taskHistory.length === 0 ? <p className="pending-empty">暂无历史任务</p> : null}
          <div className="pending-list">
            {taskHistory.map((task) => {
              const isLoading = detailLoadingId === task.task_id;
              const isInspecting = inspectedTaskDetail?.task.task_id === task.task_id;
              const questionText = task.payload?.question || task.title;
              return (
                <button
                  key={task.task_id}
                  type="button"
                  className={`pending-item${isInspecting ? " pending-item--active" : ""}`}
                  onClick={() => void handleOpenTaskDetail(task)}
                  disabled={isLoading}
                >
                  <span className="pending-title">{questionText}</span>
                  <span className="pending-meta">
                    {isLoading
                      ? "加载详情中…"
                      : `状态：${formatTaskStatus(task.status)} · ${formatTaskTime(task.updated_at || task.created_at)}`}
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        <section className="panel eval-panel">
          <h2>评测面板</h2>
          <p className="hint">把失败问答回流成检索评测样本，快速查看命中率、MRR 和 grounded 表现。</p>
          <div className="eval-actions">
            <button type="button" className="btn secondary" onClick={() => void refreshFailureCases()} disabled={failureLoading}>
              {failureLoading ? "刷新中…" : "刷新失败样本"}
            </button>
            <button type="button" className="btn secondary" onClick={() => void handleRunFailureEval()} disabled={failureEvalLoading || failureLoading}>
              {failureEvalLoading ? "评测中…" : "回流评测"}
            </button>
          </div>
          {evalError ? <p className="error eval-error">{evalError}</p> : null}
          {failureSamples ? (
            <>
              <div className="eval-summary-grid">
                <div className="metric-item">
                  <span>失败样本</span>
                  <strong>{failureSamples.total}</strong>
                </div>
                <div className="metric-item">
                  <span>导出文件</span>
                  <strong>{failureSamples.file ? "已写入" : "未写入"}</strong>
                </div>
              </div>
              {failureReasonEntries.length ? (
                <div className="diagnostics-tags">
                  {failureReasonEntries.map(([reason, count]) => (
                    <span key={reason} className="diagnostics-tag">{reason} · {count}</span>
                  ))}
                </div>
              ) : null}
              {failurePreviewItems.length ? (
                <div className="eval-list">
                  {failurePreviewItems.map((item) => (
                    <div key={item.task_id} className="eval-item">
                      <strong>{item.question}</strong>
                      <span className="pending-meta">
                        {item.reason_code} · 缺失维度：{formatJoined(item.missing_aspects, "无")}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="pending-empty">暂无失败样本</p>
              )}
            </>
          ) : null}
          {failureEval ? (
            <>
              <div className="eval-summary-grid">
                <div className="metric-item">
                  <span>Hit@K</span>
                  <strong>{formatPercent(failureEval.hit_at_k)}</strong>
                </div>
                <div className="metric-item">
                  <span>MRR</span>
                  <strong>{formatDecimal(failureEval.mrr)}</strong>
                </div>
                <div className="metric-item">
                  <span>Grounding 均值</span>
                  <strong>{formatDecimal(failureEval.avg_grounding_score)}</strong>
                </div>
                <div className="metric-item">
                  <span>评测样本</span>
                  <strong>{failureEval.total}</strong>
                </div>
              </div>
              {failureEvalPreview.length ? (
                <div className="eval-list">
                  {failureEvalPreview.map((item) => (
                    <div key={`${item.index}-${item.question}`} className="eval-item">
                      <strong>{item.hit ? "命中" : "未命中"} · {item.question}</strong>
                      <span className="pending-meta">
                        检索来源：{formatJoined(item.retrieved_sources, "无")} · 维度：{formatJoined(item.expected_aspects, "无")}
                      </span>
                    </div>
                  ))}
                </div>
              ) : null}
            </>
          ) : null}
        </section>

        <section className="panel push-panel">
          <h2>今日任务安排</h2>
          <p className="hint">系统默认时段 · 到点亮灯提示</p>
          <DailySchedulePanel />
        </section>
      </aside>

      <main className="chat">
        <section className="workspace-header">
          <div className="workspace-header__top">
            <div>
              <h2>{activeView === "education" ? "教育智能体工作台" : "股票潜力榜"}</h2>
              <p>
                {activeView === "education"
                  ? "把问答、检索诊断、任务恢复和素材生产收拢到同一个工作区，减少来回切换。"
                  : "接入实时 A 股行情后生成最有潜力的 10 只股票，便于快速浏览机会与风险。"}
              </p>
            </div>
            <div className="workspace-tabs">
              <button
                type="button"
                className={`workspace-tab${activeView === "education" ? " workspace-tab--active" : ""}`}
                onClick={() => setActiveView("education")}
              >
                教育智能体
              </button>
              <button
                type="button"
                className={`workspace-tab${activeView === "stocks" ? " workspace-tab--active" : ""}`}
                onClick={() => setActiveView("stocks")}
              >
                股票机会
              </button>
            </div>
          </div>

          {activeView === "education" ? (
            <div className="workspace-overview">
              <article className="workspace-overview-card">
                <span>当前状态</span>
                <strong>{workspacePhaseLabel}</strong>
                <p>{isSelecting ? "已命中知识片段，等待人工单选后继续生成。" : "支持直接问答、暂存恢复和过程回看。"}</p>
              </article>
              <article className="workspace-overview-card">
                <span>待继续问答</span>
                <strong>{pendingChats.length}</strong>
                <p>中断问答会自动暂存，可随时恢复到选证步骤。</p>
              </article>
              <article className="workspace-overview-card">
                <span>最近任务</span>
                <strong>{taskHistory.length}</strong>
                <p>支持查看任务详情、检索评分和执行轨迹。</p>
              </article>
              <article className="workspace-overview-card">
                <span>失败样本</span>
                <strong>{failureSamples?.total ?? 0}</strong>
                <p>可直接回流成检索评测样本，观察命中率和 grounding 表现。</p>
              </article>
            </div>
          ) : null}
        </section>

        {activeView === "education" ? (
          <section className="education-shell">
            <div className="education-main">
              <section className="conversation-shell">
                <div className="conversation-shell__header">
                  <div>
                    <h3>学习问答区</h3>
                    <p>优先检索知识库并结合图谱生成答案，必要时进入人工选证流程。</p>
                  </div>
                  <div className="conversation-shell__chips">
                    <span className="conversation-chip">流程：规划 → 检索 → 回答 → 评审</span>
                    <span className={`conversation-chip${isSelecting ? " conversation-chip--accent" : ""}`}>状态：{workspacePhaseLabel}</span>
                  </div>
                </div>

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

                {isSelecting && (
                  <section className="chunk-picker">
                    <div className="chunk-picker__header">
                      <div>
                        <h3>单选一条参考资料</h3>
                        <p className="chunk-picker__hint">当前会话已进入人工中断状态，可暂存后稍后继续。</p>
                      </div>
                      <button type="button" className="btn secondary chunk-picker__pause" onClick={handlePauseSelection}>
                        稍后继续
                      </button>
                    </div>
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
                          <span className="meta">
                            排序分：{formatDecimal(c.rank_score)} · 覆盖分：{formatDecimal(c.coverage_score ?? null)}
                          </span>
                          {c.concepts?.length ? <span className="meta">概念：{c.concepts.join("、")}</span> : null}
                          <p>
                            {c.content.slice(0, 200)}
                            {c.content.length > 200 ? "…" : ""}
                          </p>
                        </label>
                      ))}
                    </div>
                    <div className="chunk-picker__actions">
                      <button
                        type="button"
                        className="btn primary"
                        onClick={handleConfirmSelection}
                        disabled={!selectedId}
                      >
                        基于所选资料生成解答
                      </button>
                    </div>
                  </section>
                )}

                {error && <p className="error">{error}</p>}

                <footer className="composer">
                  <div className="composer__body">
                    <textarea
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      placeholder="输入学习问题…"
                      rows={2}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          void handleAsk();
                        }
                      }}
                      disabled={phase === "loading"}
                    />
                    <span className="composer__hint">Enter 发送，Shift + Enter 换行</span>
                  </div>
                  <button
                    type="button"
                    className="btn primary"
                    onClick={() => void handleAsk()}
                    disabled={phase === "loading" || !question.trim()}
                  >
                    {phase === "loading" ? "处理中…" : "提问"}
                  </button>
                </footer>
              </section>
            </div>

            <aside className="education-aside">
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
            </aside>
          </section>
        ) : (
          <section className="stock-page">
            <div className="stock-hero card">
              <div>
                <h3>{stockBoard?.title || "今日潜力股 Top 10"}</h3>
                <p className="placeholder">{stockBoard?.summary || "按实时行情、量能、估值、行业热度和风险惩罚做综合打分。"}</p>
              </div>
              <div className="stock-hero__actions">
                <span className="stock-date">{stockBoard ? `交易日：${stockBoard.trading_day} · 数据源：${stockBoard.data_source}` : "等待生成榜单"}</span>
                <button type="button" className="btn secondary stock-refresh" onClick={() => void refreshStockBoard()} disabled={stockLoading}>
                  {stockLoading ? "分析中…" : "刷新榜单"}
                </button>
              </div>
            </div>

            {stockError ? <p className="error">{stockError}</p> : null}

            {stockBoard ? (
              <>
                <div className="stock-overview">
                  <article className="card stock-metric-card">
                    <span>市场温度</span>
                    <strong>{formatStockScore(stockBoard.market_view.market_temperature)}</strong>
                    <p>{stockBoard.market_view.style}</p>
                  </article>
                  <article className="card stock-metric-card">
                    <span>上涨家数 / 候选池</span>
                    <strong>{stockBoard.market_view.up_count} / {stockBoard.market_view.candidate_size}</strong>
                    <p>全市场样本 {stockBoard.market_view.universe_size} 只，上涨占比 {stockBoard.market_view.rising_ratio}%</p>
                  </article>
                  <article className="card stock-metric-card">
                    <span>热点方向</span>
                    <strong>{stockBoard.market_view.hot_sectors.join(" / ") || "--"}</strong>
                    <p>{stockBoard.market_view.summary}</p>
                  </article>
                </div>

                <div className="stock-grid">
                  <article className="card stock-methodology">
                    <h3>分析框架</h3>
                    <ul className="stock-list">
                      {stockBoard.methodology.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                    <p className="placeholder stock-disclaimer">{stockBoard.disclaimer}</p>
                  </article>

                  <article className="card stock-ranking">
                    <h3>今日最有潜力的 10 只股票</h3>
                    <div className="stock-ranking__list">
                      {stockBoard.picks.map((pick) => (
                        <div key={pick.symbol} className="stock-row">
                          <div className="stock-row__main">
                            <div className="stock-row__title">
                              <span className="stock-rank">#{pick.rank}</span>
                              <div>
                                <strong>{pick.name}</strong>
                                <p>{pick.symbol} · {pick.sector} · {pick.board}</p>
                              </div>
                            </div>
                            <div className="stock-score-group">
                              <span className="stock-style">{pick.style}</span>
                              <strong className="stock-score">{formatStockScore(pick.score)}</strong>
                            </div>
                          </div>
                          <div className="stock-tags">
                            <span>现价 {formatStockPrice(pick.latest_price)}</span>
                            <span>涨跌 {formatStockPercent(pick.change_percent)}</span>
                            <span>换手 {formatStockPercent(pick.turnover_rate)}</span>
                            <span>振幅 {formatStockPercent(pick.amplitude)}</span>
                            <span>PE {formatStockScore(pick.pe_dynamic)}</span>
                            <span>PB {formatStockScore(pick.pb_ratio)}</span>
                            <span>成交额 {formatStockMoney(pick.amount)}</span>
                          </div>
                          <div className="stock-tags">
                            <span>趋势 {formatStockScore(pick.metrics.trend)}</span>
                            <span>质量 {formatStockScore(pick.metrics.quality)}</span>
                            <span>量能 {formatStockScore(pick.metrics.volume)}</span>
                            <span>估值 {formatStockScore(pick.metrics.valuation)}</span>
                            <span>催化 {formatStockScore(pick.metrics.catalyst)}</span>
                            <span>风险 {formatStockScore(pick.metrics.risk)}</span>
                          </div>
                          <div className="stock-detail-grid">
                            <div>
                              <h4>实时面板</h4>
                              <ul className="stock-list">
                                <li>昨收：{formatStockPrice(pick.previous_close)}</li>
                                <li>今开：{formatStockPrice(pick.open_price)}</li>
                                <li>最高：{formatStockPrice(pick.high_price)}</li>
                                <li>最低：{formatStockPrice(pick.low_price)}</li>
                                <li>成交量：{formatStockCount(pick.volume)}</li>
                              </ul>
                            </div>
                            <div>
                              <h4>入选理由</h4>
                              <ul className="stock-list">
                                {pick.reasons.map((reason) => (
                                  <li key={reason}>{reason}</li>
                                ))}
                              </ul>
                            </div>
                            <div>
                              <h4>风险提示</h4>
                              <ul className="stock-list stock-list--risk">
                                {pick.risk_flags.map((risk) => (
                                  <li key={risk}>{risk}</li>
                                ))}
                              </ul>
                            </div>
                            <div>
                              <h4>综合判断</h4>
                              <ul className="stock-list">
                                <li>综合机会分：{formatStockScore(pick.score)}</li>
                                <li>模型置信度：{formatStockScore(pick.metrics.confidence)}</li>
                                <li>当前风格：{pick.style}</li>
                                <li>日涨跌额：{formatStockPrice(pick.change_amount)}</li>
                                <li>总市值：{formatStockCap(pick.total_market_cap)}</li>
                                <li>流通市值：{formatStockCap(pick.circulating_market_cap)}</li>
                              </ul>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </article>
                </div>
              </>
            ) : (
              <div className="card stock-empty">
                <p className="placeholder">{stockLoading ? "系统正在分析今日最有潜力的股票，请稍候…" : "点击刷新按钮生成今日榜单。"}</p>
              </div>
            )}
          </section>
        )}
      </main>

      {activeView === "education" && hasDiagnostics ? (
        <button
          type="button"
          className="diagnostics-trigger"
          onClick={() => setDiagnosticsOpen(true)}
        >
          查看运行详情
        </button>
      ) : null}

      {activeView === "education" && hasDiagnostics && diagnosticsOpen ? (
        <div className="diagnostics-drawer__backdrop" onClick={() => setDiagnosticsOpen(false)}>
          <aside
            className="diagnostics-drawer"
            onClick={(e) => e.stopPropagation()}
            aria-label="运行详情"
          >
            <div className="diagnostics__header diagnostics__header--drawer">
              <div>
                <h3>运行详情</h3>
                <p>把评分结果和执行规划放在独立抽屉里，避免干扰主聊天界面。</p>
              </div>
              <button
                type="button"
                className="diagnostics-close"
                onClick={() => setDiagnosticsOpen(false)}
              >
                关闭
              </button>
            </div>

            <div className="diagnostics__grid diagnostics__grid--drawer">
              {detailTask ? (
                <article className="diagnostics-card">
                  <h4>任务摘要</h4>
                  <div className="metrics-grid">
                    <div className="metric-item">
                      <span>任务状态</span>
                      <strong>{formatTaskStatus(detailTask.status)}</strong>
                    </div>
                    <div className="metric-item">
                      <span>当前轮次</span>
                      <strong>
                        {detailTask.current_step ?? detailTask.result?.loop_step ?? 0}/{detailTask.max_steps ?? "--"}
                      </strong>
                    </div>
                    <div className="metric-item">
                      <span>错误编码</span>
                      <strong>{detailTask.error_code || detailTask.result?.warning_code || "--"}</strong>
                    </div>
                    <div className="metric-item">
                      <span>更新时间</span>
                      <strong>{formatTaskTime(detailTask.updated_at || detailTask.created_at) || "--"}</strong>
                    </div>
                  </div>
                  <p className="diagnostics-text">问题：{detailTask.payload?.question || detailTask.title}</p>
                  {detailTask.error_message || detailTask.result?.warning_message ? (
                    <p className="diagnostics-text">说明：{detailTask.error_message || detailTask.result?.warning_message}</p>
                  ) : null}
                  {displayedFinalAnswer ? <p className="diagnostics-text">答案摘要：{displayedFinalAnswer.slice(0, 180)}</p> : null}
                </article>
              ) : null}

              <article className="diagnostics-card">
                <h4>评分结果</h4>
                <div className="score-grid">
                  <div className="score-item">
                    <span className="score-label">回答可信</span>
                    <strong className={`score-value${displayedAnswerValidation?.grounded ? " is-good" : ""}`}>
                      {displayedAnswerValidation ? formatBoolean(displayedAnswerValidation.grounded) : "--"}
                    </strong>
                  </div>
                  <div className="score-item">
                    <span className="score-label">答案类型</span>
                    <strong className="score-value">{displayedAnswerValidation?.answer_type || "--"}</strong>
                  </div>
                  <div className="score-item">
                    <span className="score-label">Grounding Score</span>
                    <strong className="score-value">{formatDecimal(displayedAnswerValidation?.grounding_score)}</strong>
                  </div>
                  <div className="score-item">
                    <span className="score-label">参考重叠度</span>
                    <strong className="score-value">{formatPercent(displayedAnswerValidation?.reference_overlap)}</strong>
                  </div>
                  <div className="score-item">
                    <span className="score-label">问题覆盖度</span>
                    <strong className="score-value">{formatPercent(displayedAnswerValidation?.question_overlap)}</strong>
                  </div>
                  <div className="score-item">
                    <span className="score-label">证据覆盖</span>
                    <strong className="score-value">{formatPercent(displayedAnswerValidation?.citation_coverage)}</strong>
                  </div>
                  <div className="score-item">
                    <span className="score-label">维度覆盖</span>
                    <strong className="score-value">{formatPercent(displayedAnswerValidation?.aspect_coverage)}</strong>
                  </div>
                  <div className="score-item">
                    <span className="score-label">事实覆盖</span>
                    <strong className="score-value">{formatPercent(displayedAnswerValidation?.fact_coverage)}</strong>
                  </div>
                  <div className="score-item">
                    <span className="score-label">使用事实数</span>
                    <strong className="score-value">{displayedAnswerValidation?.used_facts ?? "--"}</strong>
                  </div>
                  <div className="score-item">
                    <span className="score-label">支持 / 弱证据</span>
                    <strong className="score-value">
                      {(displayedAnswerValidation?.supported_claims ?? 0)}/{displayedAnswerValidation?.unsupported_claims ?? 0}
                    </strong>
                  </div>
                </div>
                {displayedAnswerValidation?.missing_aspects?.length ? (
                  <div className="diagnostics-tags">
                    {displayedAnswerValidation.missing_aspects.map((aspect) => (
                      <span key={aspect} className="diagnostics-tag diagnostics-tag--warn">缺失：{aspect}</span>
                    ))}
                  </div>
                ) : null}
                {displayedAnswerValidation?.weak_sentences?.length ? (
                  <p className="diagnostics-text">弱证据语句：{displayedAnswerValidation.weak_sentences.join("；")}</p>
                ) : null}
              </article>

              <article className="diagnostics-card">
                <h4>检索策略</h4>
                <div className="metrics-grid">
                  <div className="metric-item">
                    <span>路由策略</span>
                    <strong>{displayedRetrievalSummary?.route_type || "--"}</strong>
                  </div>
                  <div className="metric-item">
                    <span>答案类型</span>
                    <strong>{displayedRetrievalSummary?.answer_type || "--"}</strong>
                  </div>
                  <div className="metric-item">
                    <span>重试策略</span>
                    <strong>{displayedRetrievalSummary?.retry_strategy || "--"}</strong>
                  </div>
                  <div className="metric-item">
                    <span>缓存策略</span>
                    <strong>{displayedRetrievalSummary?.cache_policy || "--"}</strong>
                  </div>
                  <div className="metric-item">
                    <span>缓存风险</span>
                    <strong>{displayedRetrievalSummary?.cache_risk || "--"}</strong>
                  </div>
                  <div className="metric-item">
                    <span>命中缓存</span>
                    <strong>{typeof displayedRetrievalSummary?.cache_hit === "boolean" ? formatBoolean(displayedRetrievalSummary.cache_hit) : "--"}</strong>
                  </div>
                  <div className="metric-item">
                    <span>选证方式</span>
                    <strong>{displayedRetrievalSummary?.selected_by || "--"}</strong>
                  </div>
                  <div className="metric-item">
                    <span>选证置信度</span>
                    <strong>{formatPercent(displayedRetrievalSummary?.selection_confidence)}</strong>
                  </div>
                  <div className="metric-item">
                    <span>图谱文档</span>
                    <strong>{displayedRetrievalSummary?.graph_documents ?? "--"}</strong>
                  </div>
                  <div className="metric-item">
                    <span>向量候选</span>
                    <strong>{displayedRetrievalSummary?.vector_candidates ?? "--"}</strong>
                  </div>
                  <div className="metric-item">
                    <span>关键词候选</span>
                    <strong>{displayedRetrievalSummary?.lexical_candidates ?? "--"}</strong>
                  </div>
                  <div className="metric-item">
                    <span>最终候选</span>
                    <strong>{displayedRetrievalSummary?.final_candidates ?? "--"}</strong>
                  </div>
                </div>
                {displayedRetrievalSummary?.query_expansions?.length ? (
                  <p className="diagnostics-text">查询扩展：{displayedRetrievalSummary.query_expansions.join("、")}</p>
                ) : null}
                {displayedRetrievalSummary?.route_subjects?.length ? (
                  <p className="diagnostics-text">学科路由：{displayedRetrievalSummary.route_subjects.join("、")}</p>
                ) : null}
                {displayedRetrievalSummary?.planner_queries?.length ? (
                  <p className="diagnostics-text">规划查询：{displayedRetrievalSummary.planner_queries.join(" ｜ ")}</p>
                ) : null}
                {displayedRetrievalSummary?.router_features?.length ? (
                  <div className="diagnostics-tags">
                    {displayedRetrievalSummary.router_features.map((feature) => (
                      <span key={feature} className="diagnostics-tag">{feature}</span>
                    ))}
                  </div>
                ) : null}
                {displayedRetrievalSummary?.evidence_sources?.length ? (
                  <p className="diagnostics-text">证据来源：{displayedRetrievalSummary.evidence_sources.join("、")}</p>
                ) : null}
                {displayedGraphSummary?.matched_concepts?.length ? (
                  <p className="diagnostics-text">图谱命中：{displayedGraphSummary.matched_concepts.join("、")}</p>
                ) : null}
              </article>

              {displayedEvents.length ? (
                <article className="diagnostics-card diagnostics-card--trace">
                  <div className="trace-head">
                    <h4>任务事件</h4>
                    <span>{displayedEvents.length} 条事件</span>
                  </div>
                  <div className="trace-list">
                    {displayedEvents.slice(-8).reverse().map((event) => (
                      <div key={event.event_id} className="trace-item">
                        <div className="trace-item__top">
                          <span className="trace-step">{event.node || event.type}</span>
                          <span className={`trace-status trace-status--${event.status || "running"}`}>
                            {formatTaskStatus(event.status)}
                          </span>
                        </div>
                        <div className="trace-node">{event.message}</div>
                        <span className="trace-time">{formatTaskTime(event.created_at)}</span>
                      </div>
                    ))}
                  </div>
                </article>
              ) : null}
            </div>

            <article className="diagnostics-card diagnostics-card--trace">
              <div className="trace-head">
                <h4>执行规划步骤</h4>
                <span>{displayedExecutionTrace.length ? `${displayedExecutionTrace.length} 个节点` : "暂无执行轨迹"}</span>
              </div>
              {traceItems.length ? (
                <div className="trace-list">
                  {traceItems.map((item, index) => {
                    const trace = displayedExecutionTrace[index];
                    return (
                      <div key={item.key} className="trace-item">
                        <div className="trace-item__top">
                          <span className="trace-step">Step {trace.step || index + 1}</span>
                          <span className={`trace-status trace-status--${trace.status}`}>{item.statusLabel}</span>
                        </div>
                        <div className="trace-node">{item.nodeLabel}</div>
                        <p className="trace-message">{trace.message}</p>
                        {item.details ? <p className="trace-details">{item.details}</p> : null}
                        <span className="trace-time">耗时 {trace.elapsed_ms} ms</span>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="diagnostics-empty">当前调用还没有返回执行轨迹。</p>
              )}
            </article>
          </aside>
        </div>
      ) : null}
    </div>
  );
}
