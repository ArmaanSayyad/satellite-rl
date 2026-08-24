import type { SimulationResult } from "../types";
import type { PlaybackState } from "../useSimulationPlayback";

interface Props {
  result: SimulationResult;
  playback: PlaybackState;
}

export default function MissionTimeline({ result, playback }: Props) {
  const { missionDurationS, progress } = playback;
  return (
    <div className="timeline">
      <div className="timeline-track">
        <div className="timeline-fill" style={{ width: `${progress * 100}%` }} />
        {result.decisions.map((d, i) => {
          const pct = missionDurationS > 0 ? (d.t_s / missionDurationS) * 100 : 0;
          const fired = d.t_s <= playback.missionTimeS;
          const maneuvered = d.action_magnitude_ms > 1e-3;
          return (
            <div
              key={i}
              className={`timeline-marker ${fired ? "fired" : ""} ${maneuvered ? "maneuvered" : "coasted"}`}
              style={{ left: `${pct}%` }}
              title={`t=${(d.t_s / 3600).toFixed(2)}h · Δv=${d.action_magnitude_ms.toFixed(2)} m/s`}
            />
          );
        })}
      </div>
    </div>
  );
}
