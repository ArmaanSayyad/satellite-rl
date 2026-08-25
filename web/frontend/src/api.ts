import type { Scenario, SimulationResult } from "./types";

const API_BASE = "http://127.0.0.1:8000";

export async function listScenarios(): Promise<Scenario[]> {
  const res = await fetch(`${API_BASE}/api/scenarios`);
  if (!res.ok) {
    throw new Error(`Scenario list request failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function runSimulation(seed?: number): Promise<SimulationResult> {
  const url = new URL(`${API_BASE}/api/simulate`);
  if (seed != null) url.searchParams.set("seed", String(seed));
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) {
    throw new Error(`Simulation request failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}
