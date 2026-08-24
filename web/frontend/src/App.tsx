import { useState } from "react";
import "./App.css";
import { runSimulation } from "./api";
import type { SimulationResult } from "./types";
import { useSimulationPlayback } from "./useSimulationPlayback";
import OrbitalContextView from "./components/OrbitalContextView";
import EncounterView from "./components/EncounterView";
import Dashboard from "./components/Dashboard";
import MissionTimeline from "./components/MissionTimeline";

type Status = "idle" | "loading" | "ready" | "error";

export default function App() {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const playback = useSimulationPlayback(result);

  async function handleBegin() {
    setStatus("loading");
    setError(null);
    try {
      const data = await runSimulation();
      setResult(data);
      setStatus("ready");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Satellite Collision Avoidance</h1>
        <p className="subtitle">
          A reinforcement-learning policy deciding, in real time, whether to burn fuel to dodge a real
          historical close approach — physics simulated with Basilisk, not scripted.
        </p>
      </header>

      <div className="controls">
        <button className="begin-button" onClick={handleBegin} disabled={status === "loading"}>
          {status === "loading" ? "Running Simulation…" : status === "ready" ? "Run Another Simulation" : "Begin Simulation"}
        </button>
        {status === "error" && <div className="error-banner">Simulation failed: {error}</div>}
      </div>

      {result && (
        <>
          <div className="scenario-banner">
            <span>
              Real historical event · miss distance <strong>{result.scenario.miss_distance_m.toFixed(0)} m</strong> · closing
              speed <strong>{(result.scenario.relative_speed_ms / 1000).toFixed(2)} km/s</strong>
            </span>
            {playback.finished && (
              <span className={`outcome ${result.result.collision_occurred ? "bad" : "good"}`}>
                {result.result.collision_occurred
                  ? "COLLISION OCCURRED"
                  : `COLLISION AVOIDED — final Pc ${result.result.pc_final.toExponential(2)}, fuel used ${result.result.total_fuel_used_ms.toFixed(2)} m/s`}
              </span>
            )}
          </div>

          <MissionTimeline result={result} playback={playback} />
          <Dashboard result={result} playback={playback} />

          <div className="views">
            <OrbitalContextView
              frame={playback.frame}
              earthRadiusM={result.constants.earth_radius_m}
              denseFrames={result.dense_frames}
            />
            <EncounterView
              frame={playback.frame}
              denseFrames={result.dense_frames}
              combinedRadiusM={result.scenario.combined_radius_m}
            />
          </div>
        </>
      )}

      {!result && status !== "loading" && (
        <div className="empty-state">
          Click <strong>Begin Simulation</strong> to fly a real, historically-dangerous conjunction and watch the
          trained policy decide whether and when to maneuver.
        </div>
      )}
    </div>
  );
}
