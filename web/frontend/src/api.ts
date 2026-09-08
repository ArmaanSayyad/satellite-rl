import type { ReplayIndex, SimulationResult } from "./types";

const API_BASE = import.meta.env.VITE_API_ORIGIN ?? "";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok)
    throw new Error(
      `Request failed (${response.status}). This artifact or service may not be available in this installation.`,
    );
  return response.json();
}
export function loadIndex(): Promise<ReplayIndex> {
  return request("/replays/index.json");
}
export async function loadReplay(path: string): Promise<SimulationResult> {
  const replay = await request<SimulationResult>(
    path.startsWith("/api/") ? `${API_BASE}${path}` : path,
  );
  if (
    !replay ||
    !Array.isArray(replay.dense_frames) ||
    !replay.dense_frames.length ||
    !Array.isArray(replay.keyframes) ||
    !replay.keyframes.length ||
    !Array.isArray(replay.decisions) ||
    !replay.result ||
    !replay.constants ||
    !replay.scenario
  )
    throw new Error(
      "The replay artifact is incomplete or has an unsupported schema.",
    );
  return replay;
}
export interface Job {
  id: string;
  status: string;
  progress?: number;
  result?: SimulationResult;
  error?: unknown;
  artifact_url?: string;
  cached?: boolean;
}
export function createJob(
  scenarioId: string,
  policyId: string,
  seed: number,
  variant: string,
  radiusScale: number,
  scheduleMode = "controlled",
): Promise<Job> {
  return request(`${API_BASE}/api/v2/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scenario_id: scenarioId,
      policy_id: policyId,
      seed,
      variant,
      radius_scale: radiusScale,
      schedule_mode: scheduleMode,
    }),
  });
}
export function pollJob(id: string): Promise<Job> {
  return request(`${API_BASE}/api/v2/jobs/${encodeURIComponent(id)}`);
}
export function cancelJob(id: string): Promise<Job> {
  return request(`${API_BASE}/api/v2/jobs/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}
