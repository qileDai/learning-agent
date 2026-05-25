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
  chunk_budget_tokens: number;
  graph_budget_tokens: number;
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
}

export interface ChatTaskResult {
  answer_mode?: string;
  kb_hit?: boolean;
  final_answer?: string;
  loop_step?: number;
  retry_count?: number;
  message?: string;
}

export interface ChatTask {
  task_id: string;
  thread_id?: string | null;
  title: string;
  status: "running" | "awaiting_input" | "retrying" | "completed" | "failed" | "timeout" | "cancelled";
  payload?: {
    question?: string;
  };
  result?: ChatTaskResult;
  updated_at?: string;
  created_at?: string;
}

export interface ChatStateResponse {
  task_id: string;
  thread_id: string;
  next: string[];
  retrieved_chunks: RetrievedChunk[];
  final_answer?: string;
  graph_summary?: GraphSummary;
  retrieval_summary?: RetrievalSummary;
  task?: ChatTask | null;
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

export async function getPendingChatTasks(limit = 20): Promise<TaskListResponse> {
  const params = new URLSearchParams({ kind: "chat", status: "awaiting_input", limit: String(limit) });
  const r = await fetch(`${API}/tasks?${params.toString()}`);
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
