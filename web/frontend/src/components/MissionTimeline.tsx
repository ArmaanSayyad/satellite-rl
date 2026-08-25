import type { MouseEvent } from "react";
import type { SimulationResult } from "../types";
import type { PlaybackState } from "../useSimulationPlayback";
import { explainDecision } from "../decisionExplain";
import Tooltip from "./Tooltip";

interface Props {
  result: SimulationResult;
  playback: PlaybackState;
}

export default function MissionTimeline({ result, playback }: Props) {
  const { missionDurationS, progress, seek } = playback;

  function handleTrackClick(e: MouseEvent<HTMLDivElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    seek(frac * missionDurationS);
  }

  return (
    <div className="timeline">
      <div className="timeline-track" onClick={handleTrackClick} title="Click to jump to this point in the mission">
        <div className="timeline-fill" style={{ width: `${progress * 100}%` }} />
        {result.decisions.map((d, i) => {
          const pct = missionDurationS > 0 ? (d.t_s / missionDurationS) * 100 : 0;
          const fired = d.t_s <= playback.missionTimeS;
          const maneuvered = d.action_magnitude_ms > 1e-3;
          const explanation = explainDecision(d, result.keyframes[i] ?? null, result.constants.pc_threshold);
          return (
            <span
              key={i}
              className={`timeline-marker ${fired ? "fired" : ""} ${maneuvered ? "maneuvered" : "coasted"}`}
              style={{ left: `${pct}%` }}
              onClick={(e) => {
                e.stopPropagation();
                seek(d.t_s);
              }}
            >
              <Tooltip
                content={
                  <>
                    t = {(d.t_s / 3600).toFixed(2)}h — {explanation}
                  </>
                }
              >
                <span className="timeline-marker-hit" />
              </Tooltip>
            </span>
          );
        })}
      </div>
    </div>
  );
}
