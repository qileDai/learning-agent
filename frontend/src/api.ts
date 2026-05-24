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

export interface ChatStartResponse {
  thread_id: string;
  status: "awaiting_selection" | "completed";
  mode?: "kb" | "graph_kb" | "llm" | "greeting";
  kb_hit?: boolean;
  retrieved_chunks?: RetrievedChunk[];
  answer?: string;
  message?: string | null;
  selection_mode?: "single";
  graph_summary?: GraphSummary;
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

export async function chatResume(threadId: string, selectedChunkIds: string[]) {
  const r = await fetch(`${API}/chat/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, selected_chunk_ids: selectedChunkIds }),
  });
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
