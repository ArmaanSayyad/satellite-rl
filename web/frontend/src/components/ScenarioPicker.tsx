import { useEffect, useState } from "react";
import { listScenarios } from "../api";
import type { Scenario } from "../types";
import Tooltip from "./Tooltip";

interface Props {
  selectedSeed: number | null;
  onSelect: (seed: number | null) => void;
  disabled: boolean;
}

export default function ScenarioPicker({ selectedSeed, onSelect, disabled }: Props) {
  const [scenarios, setScenarios] = useState<Scenario[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listScenarios()
      .then(setScenarios)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) return null; // non-critical -- "Begin Simulation" still works with a random pick
  if (!scenarios) return <div className="scenario-picker scenario-picker-loading">Loading real events…</div>;

  return (
    <div className="scenario-picker">
      <span className="scenario-picker-label">Real event:</span>
      <div className="scenario-chip-row" role="group" aria-label="Choose a real conjunction event">
        <button
          className={`scenario-chip ${selectedSeed === null ? "active" : ""}`}
          onClick={() => onSelect(null)}
          disabled={disabled}
        >
          🎲 Random
        </button>
        {scenarios.map((s) => (
          <Tooltip
            key={s.seed}
            content={
              <>
                Miss distance {s.miss_distance_m.toFixed(0)} m · closing speed{" "}
                {(s.relative_speed_ms / 1000).toFixed(2)} km/s · native Pc {s.native_pc.toExponential(2)}
              </>
            }
          >
            <button
              className={`scenario-chip ${selectedSeed === s.seed ? "active" : ""}`}
              onClick={() => onSelect(s.seed)}
              disabled={disabled}
            >
              {s.miss_distance_m.toFixed(0)} m
            </button>
          </Tooltip>
        ))}
      </div>
    </div>
  );
}
