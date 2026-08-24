import type { SimulationResult } from "./types";

const API_BASE = "http://127.0.0.1:8000";

export async function runSimulation(): Promise<SimulationResult> {
  const res = await fetch(`${API_BASE}/api/simulate`, { method: "POST" });
  if (!res.ok) {
    throw new Error(`Simulation request failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}
