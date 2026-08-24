import type { PlaybackState } from "../useSimulationPlayback";
import type { SimulationResult } from "../types";
import { gravityAccelMs2, separationM, speedMs } from "../physics";

interface Props {
  result: SimulationResult;
  playback: PlaybackState;
}

function fmt(n: number, digits = 2): string {
  if (n === 0) return "0";
  if (Math.abs(n) < 1e-3 || Math.abs(n) >= 1e6) return n.toExponential(digits);
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function Stat({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">
        {value}
        {unit && <span className="stat-unit"> {unit}</span>}
      </div>
    </div>
  );
}

export default function Dashboard({ result, playback }: Props) {
  const { frame, currentKeyframe, missionTimeS, missionDurationS, maneuversSoFar } = playback;
  const { earth_mu_m3s2 } = result.constants;

  const speed = frame ? speedMs(frame.ego_v) : 0;
  const gravity = frame ? gravityAccelMs2(frame.ego_r, earth_mu_m3s2) : 0;
  const separation = frame ? separationM(frame.ego_r, frame.sec_r) : 0;
  const fuel = currentKeyframe?.fuel_ms ?? result.keyframes[0]?.fuel_ms ?? 0;
  const pc = currentKeyframe?.pc_estimate;

  return (
    <div className="dashboard">
      <div className="dashboard-row">
        <Stat label="Mission Time" value={`${(missionTimeS / 3600).toFixed(2)} / ${(missionDurationS / 3600).toFixed(2)}`} unit="hr" />
        <Stat label="Orbital Speed" value={fmt(speed, 1)} unit="m/s" />
        <Stat label="Gravitational Accel." value={fmt(gravity, 3)} unit="m/s²" />
        <Stat label="Separation" value={fmt(separation, 1)} unit="m" />
      </div>
      <div className="dashboard-row">
        <Stat label="Fuel Remaining (Δv)" value={fmt(fuel, 2)} unit="m/s" />
        <Stat label="Maneuvers Fired" value={`${maneuversSoFar} / ${result.decisions.length}`} />
        <Stat label="Est. Collision Probability" value={pc != null ? fmt(pc, 3) : "—"} />
        <Stat label="Danger Threshold" value={fmt(result.constants.pc_threshold, 0)} />
      </div>
    </div>
  );
}
