import type { PlaybackState } from "../useSimulationPlayback";
import type { SimulationResult } from "../types";
import { gravityAccelMs2, separationM, speedMs } from "../physics";
import Info from "./Info";

interface Props {
  result: SimulationResult;
  playback: PlaybackState;
}

function fmt(n: number, digits = 2): string {
  if (n === 0) return "0";
  if (Math.abs(n) < 1e-3 || Math.abs(n) >= 1e6) return n.toExponential(digits);
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function Stat({ label, value, unit, info }: { label: string; value: string; unit?: string; info?: string }) {
  return (
    <div className="stat">
      <div className="stat-label">
        {label}
        {info && <Info text={info} />}
      </div>
      <div className="stat-value">
        {value}
        {unit && <span className="stat-unit"> {unit}</span>}
      </div>
    </div>
  );
}

export default function Dashboard({ result, playback }: Props) {
  const { frame, currentKeyframe, missionTimeS, missionDurationS, maneuversSoFar } = playback;
  const { earth_mu_m3s2, pc_threshold } = result.constants;

  const speed = frame ? speedMs(frame.ego_v) : 0;
  const gravity = frame ? gravityAccelMs2(frame.ego_r, earth_mu_m3s2) : 0;
  const separation = frame ? separationM(frame.ego_r, frame.sec_r) : 0;
  const fuel = currentKeyframe?.fuel_ms ?? result.keyframes[0]?.fuel_ms ?? 0;
  const pc = currentKeyframe?.pc_estimate;

  const trainedNeutralized = result.result.pc_final <= pc_threshold;
  const baselineNeutralized = result.baseline.pc_final <= pc_threshold;

  return (
    <div className="dashboard">
      <div className="dashboard-row">
        <Stat
          label="Mission Time"
          value={`${(missionTimeS / 3600).toFixed(2)} / ${(missionDurationS / 3600).toFixed(2)}`}
          unit="hr"
        />
        <Stat label="Orbital Speed" value={fmt(speed, 1)} unit="m/s" />
        <Stat label="Gravitational Accel." value={fmt(gravity, 3)} unit="m/s²" />
        <Stat label="Separation" value={fmt(separation, 1)} unit="m" />
      </div>
      <div className="dashboard-row">
        <Stat
          label="Fuel Remaining"
          info="Δv (delta-v): total velocity change the satellite's remaining fuel can still provide. The standard way to budget maneuver fuel, independent of the specific engine."
          value={fmt(fuel, 2)}
          unit="m/s Δv"
        />
        <Stat label="Maneuvers Fired" value={`${maneuversSoFar} / ${result.decisions.length}`} />
        <Stat
          label="Est. Collision Probability"
          info="Pc: the policy's own real-time collision-probability estimate, recomputed from the live simulated state at this instant -- not a value looked up once and held fixed."
          value={pc != null ? fmt(pc, 3) : "—"}
        />
        <Stat
          label="Danger Threshold"
          info="If estimated Pc exceeds this value, the encounter counts as genuinely dangerous -- the same order-of-magnitude threshold real conjunction-assessment operations commonly use to decide whether a maneuver is warranted."
          value={fmt(pc_threshold, 0)}
        />
      </div>

      <div className="baseline-compare">
        <div className="baseline-compare-title">
          What if nothing maneuvered?
          <Info text="The identical real event, re-simulated with the satellite never burning fuel -- the actual unmitigated risk this encounter posed, for comparison against the trained policy's outcome." />
        </div>
        <div className="baseline-compare-row">
          <span className={`baseline-pill ${trainedNeutralized ? "good" : "bad"}`}>
            Trained policy — Pc {fmt(result.result.pc_final, 2)} · {fmt(result.result.total_fuel_used_ms, 2)} m/s
            fuel · {trainedNeutralized ? "risk neutralized" : "still over threshold"}
          </span>
          <span className={`baseline-pill ${baselineNeutralized ? "good" : "bad"}`}>
            No maneuver — Pc {fmt(result.baseline.pc_final, 2)} · 0 m/s fuel ·{" "}
            {baselineNeutralized ? "would have been fine anyway" : "would have exceeded threshold"}
          </span>
        </div>
      </div>
    </div>
  );
}
