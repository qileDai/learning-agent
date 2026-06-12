const API = "/api";

export interface RetrievedChunk {
  id: string;
  content: string;
  source: string;
  file_type: string;
  score?: number | null;
  subject?: string | null;
  chapter?: string | null;
  retrieval_mode?: string | null;
  concepts?: string[];
  rank_score?: number | null;
}

export interface GraphSummary {
  matched_concepts: string[];
  related_concepts: string[];
  has_graph_context: boolean;
}

export interface RetrievalSummary {
  query_expansions: string[];
  route_subjects: string[];
  route_type?: string;
  graph_documents: number;
  vector_candidates: number;
  lexical_candidates: number;
  final_candidates: number;
  max_per_source: number;
  vector_k?: number;
  lexical_k?: number;
  final_k?: number;
  rerank_window?: number;
  chunk_budget_tokens: number;
  graph_budget_tokens: number;
  cache_hit?: boolean;
  cache_similarity?: number;
  retry_count?: number;
  retry_strategy?: string;
  score_profile?: Record<string, number>;
}

export interface AnswerValidation {
  grounded: boolean;
  grounding_score: number;
  reference_overlap: number;
  question_overlap: number;
  citation_coverage?: number;
  supported_claims?: number;
  unsupported_claims?: number;
  weak_sentences?: string[];
}

export interface ExecutionTrace {
  node: string;
  status: string;
  message: string;
  step: number;
  elapsed_ms: number;
  data: Record<string, unknown>;
}

export interface ChatStartResponse {
  task_id: string;
  thread_id: string;
  status: "awaiting_selection" | "completed";
  mode?: "kb" | "graph_kb" | "llm" | "greeting";
  kb_hit?: boolean;
  retrieved_chunks?: RetrievedChunk[];
  answer?: string;
  message?: string | null;
  selection_mode?: "single";
  graph_summary?: GraphSummary;
  retrieval_summary?: RetrievalSummary;
  answer_validation?: AnswerValidation;
  execution_trace?: ExecutionTrace[];
}

export interface ChatResumeResponse {
  task_id: string;
  thread_id: string;
  status: "completed";
  mode?: "kb" | "graph_kb" | "llm" | "greeting";
  kb_hit?: boolean;
  answer?: string;
  message?: string | null;
  graph_summary?: GraphSummary;
  retrieval_summary?: RetrievalSummary;
  answer_validation?: AnswerValidation;
  execution_trace?: ExecutionTrace[];
}

export interface ChatTaskResult {
  answer_mode?: string;
  kb_hit?: boolean;
  final_answer?: string;
  loop_step?: number;
  retry_count?: number;
  message?: string;
  warning_code?: string;
  warning_message?: string;
  critic_reason?: string;
  critic_reason_code?: string;
  retry_strategy?: string;
  retrieved_chunks?: RetrievedChunk[];
  selected_chunk_ids?: string[];
  retrieval_summary?: RetrievalSummary;
  answer_validation?: AnswerValidation;
  execution_trace?: ExecutionTrace[];
}

export interface ChatTask {
  task_id: string;
  thread_id?: string | null;
  title: string;
  kind?: string;
  status: "running" | "awaiting_input" | "retrying" | "completed" | "failed" | "timeout" | "cancelled";
  payload?: {
    question?: string;
  };
  result?: ChatTaskResult;
  error_code?: string | null;
  error_message?: string | null;
  retry_count?: number;
  current_step?: number;
  max_steps?: number;
  updated_at?: string;
  created_at?: string;
}

export interface ChatStateResponse {
  task_id: string;
  thread_id: string;
  next: string[];
  retrieved_chunks: RetrievedChunk[];
  selected_chunk_ids?: string[];
  final_answer?: string;
  graph_summary?: GraphSummary;
  retrieval_summary?: RetrievalSummary;
  answer_validation?: AnswerValidation;
  execution_trace?: ExecutionTrace[];
  task?: ChatTask | null;
}

export interface TaskEvent {
  event_id: string;
  task_id: string;
  type: string;
  node?: string | null;
  status?: string | null;
  message: string;
  data: Record<string, unknown>;
  created_at: string;
}

export interface ChatTaskDetailResponse {
  task: ChatTask;
  events: TaskEvent[];
  graph_state?: ChatStateResponse | null;
}

export interface TaskListResponse {
  items: ChatTask[];
}

export interface DailyPush {
  date: string;
  title: string;
  content: string;
  created_at: string;
}

export type TaskStatus = "upcoming" | "active" | "completed";

export interface ScheduleTask {
  id: string;
  time: string;
  title: string;
  action: string;
  status: TaskStatus;
  start_at: string;
  end_at: string;
}

export interface DailySchedule {
  date: string;
  title: string;
  source: string;
  server_time: string;
  active_task_id: string | null;
  tasks: ScheduleTask[];
}

export interface MediaJob {
  job_id: string;
  kind: "image" | "video";
  provider: string;
  status: "completed" | "processing" | "failed" | "mock_ready";
  prompt: string;
  message?: string;
  preview_url?: string | null;
  image_url?: string | null;
  poster_url?: string | null;
  video_url?: string | null;
  storyboard?: string | null;
  storyboard_url?: string | null;
  style?: string;
  aspect_ratio?: string;
  mode?: string;
  duration_seconds?: number;
  source_image_url?: string | null;
  provider_model?: string;
  created_at: string;
  updated_at?: string;
}

export interface StockMetrics {
  trend: number;
  quality: number;
  volume: number;
  valuation: number;
  catalyst: number;
  risk: number;
  confidence: number;
  turnover_rate: number;
}

export interface StockPick {
  rank: number;
  symbol: string;
  name: string;
  sector: string;
  board: string;
  score: number;
  style: string;
  latest_price: number;
  change_percent: number;
  change_amount: number;
  turnover_rate: number;
  amplitude: number;
  pe_dynamic: number | null;
  pb_ratio: number | null;
  total_market_cap: number;
  circulating_market_cap: number;
  amount: number;
  volume: number;
  open_price: number | null;
  high_price: number | null;
  low_price: number | null;
  previous_close: number | null;
  reasons: string[];
  risk_flags: string[];
  metrics: StockMetrics;
}

export interface StockMarketView {
  trading_day: string;
  market_temperature: number;
  average_score: number;
  style: string;
  hot_sectors: string[];
  summary: string;
  up_count: number;
  down_count: number;
  rising_ratio: number;
  universe_size: number;
  candidate_size: number;
}

export interface DailyStockResponse {
  generated_at: string;
  trading_day: string;
  title: string;
  summary: string;
  methodology: string[];
  market_view: StockMarketView;
  picks: StockPick[];
  disclaimer: string;
  data_source: string;
}

export async function getDailySchedule(): Promise<DailySchedule> {
  const r = await fetch(`${API}/daily-schedule`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function healthCheck() {
  const r = await fetch(`${API}/health`);
  return r.json();
}

export async function ingestKnowledge(reset = false) {
  const r = await fetch(`${API}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reset }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function chatStart(question: string, threadId?: string): Promise<ChatStartResponse> {
  const r = await fetch(`${API}/chat/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, thread_id: threadId }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function chatResume(threadId: string, selectedChunkIds: string[]): Promise<ChatResumeResponse> {
  const r = await fetch(`${API}/chat/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, selected_chunk_ids: selectedChunkIds }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getChatState(threadId: string): Promise<ChatStateResponse> {
  const r = await fetch(`${API}/chat/state/${threadId}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getChatTasks(options?: { status?: string; limit?: number }): Promise<TaskListResponse> {
  const params = new URLSearchParams({ kind: "chat", limit: String(options?.limit ?? 20) });
  if (options?.status) params.set("status", options.status);
  const r = await fetch(`${API}/tasks?${params.toString()}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getPendingChatTasks(limit = 20): Promise<TaskListResponse> {
  return getChatTasks({ status: "awaiting_input", limit });
}

export async function getChatTaskDetail(taskId: string): Promise<ChatTaskDetailResponse> {
  const r = await fetch(`${API}/tasks/${taskId}/detail`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getDailyStockPicks(limit = 10): Promise<DailyStockResponse> {
  const r = await fetch(`${API}/stocks/daily-picks?limit=${limit}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function generateImage(prompt: string, style: string, aspectRatio: string): Promise<MediaJob> {
  const r = await fetch(`${API}/media/image/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, style, aspect_ratio: aspectRatio }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function generateVideo(
  prompt: string,
  mode: "text-to-video" | "image-to-video",
  durationSeconds: number,
  sourceImageUrl?: string
): Promise<MediaJob> {
  const r = await fetch(`${API}/media/video/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      mode,
      duration_seconds: durationSeconds,
      source_image_url: sourceImageUrl || null,
    }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getMediaJob(jobId: string): Promise<MediaJob> {
  const r = await fetch(`${API}/media/jobs/${jobId}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function getLatestPush(): Promise<DailyPush | { status: string }> {
  const r = await fetch(`${API}/daily-push/latest`);
  return r.json();
}

export async function generateDailyPush(): Promise<DailyPush> {
  const r = await fetch(`${API}/daily-push/generate`, { method: "POST" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
