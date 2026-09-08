import type { Encounter, SimulationResult } from "../types";

import { policyName, ellipseGeometry, probability } from "../evidence";

export function EncounterPlane({
  encounter,
  radius,
  radiusFactor,
  setRadiusFactor,
  sensitivity,
  mode = "observed",
}: {
  encounter?: Encounter;
  radius: number;
  radiusFactor: number;
  setRadiusFactor: (n: number) => void;
  sensitivity?: Record<string, number>;
  mode?: "observed" | "truth";
}) {
  if (!encounter)
    return (
      <section className="instrument encounter">
        <div className="section-label">02 / Encounter plane</div>
        <h3>Geometry not recorded</h3>
        <p>
          This replay does not include projected covariance. No illustrative
          ellipse is substituted.
        </p>
      </section>
    );
  const { major, minor, angle } = ellipseGeometry(encounter);
  const [mx, my] = encounter.miss_vector_m;
  const extent = Math.max(
    Math.abs(mx) + 3 * major,
    Math.abs(my) + 3 * major,
    radius * 3,
    1,
  );
  const scale = 145 / extent;
  return (
    <section className="instrument encounter">
      <div className="instrument-heading">
        <span className="section-label">02 / Encounter plane</span>
        <span className="micro">
          {mode === "truth" ? "Terminal truth" : "Predicted at TCA"}
        </span>
      </div>
      <svg
        viewBox="0 0 400 350"
        role="img"
        aria-label={`Encounter plane: miss vector ${mx.toFixed(1)}, ${my.toFixed(1)} meters; covariance standard deviations ${major.toFixed(1)} and ${minor.toFixed(1)} meters.`}
      >
        <defs>
          <pattern
            id="plane-grid"
            width="35"
            height="35"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 35 0 L 0 0 0 35"
              fill="none"
              stroke="#243332"
              strokeWidth=".6"
            />
          </pattern>
        </defs>
        <rect x="25" y="15" width="350" height="315" fill="url(#plane-grid)" />
        <path d="M200 15V330 M25 175H375" stroke="#61706a" strokeWidth=".6" />
        <g
          transform={`translate(${200 + mx * scale} ${175 - my * scale}) rotate(${(-angle * 180) / Math.PI})`}
        >
          {[3, 2, 1].map((n) => (
            <ellipse
              key={n}
              rx={major * scale * n}
              ry={minor * scale * n}
              fill={n === 1 ? "#8bc9b31a" : "none"}
              stroke="#88bea6"
              strokeOpacity={1 / n}
              strokeDasharray={n === 3 ? "3 4" : undefined}
            />
          ))}
        </g>
        <path
          d={`M200 175L${200 + mx * scale} ${175 - my * scale}`}
          stroke="#e8b881"
          strokeWidth="1.4"
        />
        <circle
          cx="200"
          cy="175"
          r={radius * radiusFactor * scale}
          fill="#e88c7133"
          stroke="#ed927b"
        />
        <path d="M195 175h10 M200 170v10" stroke="#f4ece0" />
        <circle
          cx={200 + mx * scale}
          cy={175 - my * scale}
          r="3"
          fill="#e8b881"
        />
        <text x="28" y="345" fill="#a3afa5" fontSize="10">
          ± {extent.toFixed(0)} m · equal axis scale
        </text>
        <text x="275" y="345" fill="#a3afa5" fontSize="10">
          1σ / 2σ / 3σ
        </text>
      </svg>
      <div className="plane-legend">
        <span>
          <i className="dot mint" />
          Covariance
        </span>
        <span>
          <i className="dot amber" />
          Miss vector
        </span>
        <span>
          <i className="dot coral" />
          Hard-body radius
        </span>
      </div>
      <label className="sensitivity">
        Radius sensitivity{" "}
        <output>
          {radiusFactor.toFixed(1)}× · {(radius * radiusFactor).toFixed(1)} m
        </output>
        <input
          aria-label="Hard-body radius sensitivity"
          type="range"
          min="0"
          max="2"
          step="1"
          value={[0.5, 1, 2].indexOf(radiusFactor)}
          onChange={(e) => setRadiusFactor([0.5, 1, 2][Number(e.target.value)])}
        />
      </label>
      {sensitivity ? (
        <p className="fine">
          Final Pc (Foster), radius {radiusFactor}×:{" "}
          <strong>{probability(sensitivity[String(radiusFactor)])}</strong>.
          Precomputed at terminal truth, same trajectory. RCS-derived radius is
          uncertain.
        </p>
      ) : (
        <p className="fine">
          RCS-derived radius is a proxy. This control changes the displayed
          collision region only; recorded Pc is unchanged.
        </p>
      )}
    </section>
  );
}

export function RiskChart({
  runs,
  time,
  threshold,
}: {
  runs: SimulationResult[];
  time: number;
  threshold: number;
}) {
  const duration = Math.max(
    1,
    ...runs.flatMap((r) => r.keyframes.map((k) => k.t_s)),
  );
  const y = (pc: number) =>
    20 +
    (Math.min(12, Math.max(0, -Math.log10(Math.max(pc, 1e-12)))) / 12) * 145;
  const colors = ["#a3d7bd", "#e6b77e", "#a9b6e6", "#dc8e7c", "#7bc3d5"];
  return (
    <section className="instrument chart">
      <div className="instrument-heading">
        <span className="section-label">03 / Risk through the encounter</span>
        <span className="micro">Pc · log scale</span>
      </div>
      <svg
        viewBox="0 0 660 205"
        role="img"
        aria-label="Recorded collision probability at successive decision updates, logarithmic scale"
      >
        {[0, -4, -8, -12].map((v) => (
          <g key={v}>
            <line
              x1="50"
              x2="635"
              y1={y(10 ** v)}
              y2={y(10 ** v)}
              stroke="#263430"
            />
            <text x="4" y={y(10 ** v) + 4} fill="#a5b0a9" fontSize="10">
              10^{v}
            </text>
          </g>
        ))}
        <line
          x1="50"
          x2="635"
          y1={y(threshold)}
          y2={y(threshold)}
          stroke="#df987a"
          strokeDasharray="4 5"
        />
        <text x="510" y={y(threshold) - 7} fill="#df987a" fontSize="10">
          Decision threshold
        </text>
        {runs.map((run, i) => {
          const points = run.keyframes
            .filter((k) => k.pc_estimate != null)
            .map(
              (k) => `${50 + (k.t_s / duration) * 585},${y(k.pc_estimate!)}`,
            );
          return (
            <polyline
              key={run.id ?? i}
              points={points.join(" ")}
              fill="none"
              stroke={colors[i % colors.length]}
              strokeWidth="1.8"
            />
          );
        })}
        <line
          x1={50 + (time / duration) * 585}
          x2={50 + (time / duration) * 585}
          y1="15"
          y2="175"
          stroke="#eee6d2"
          opacity=".6"
        />
        <text x="50" y="195" fill="#a5b0a9" fontSize="10">
          First warning
        </text>
        <text x="606" y="195" fill="#a5b0a9" fontSize="10">
          TCA
        </text>
      </svg>
      <div className="plot-legend">
        {runs.map((r, i) => (
          <span key={r.id ?? i}>
            <i
              className="dot"
              style={{ background: colors[i % colors.length] }}
            />
            {policyName(r.policy_id ?? "v1")}
          </span>
        ))}
      </div>
      <p className="fine">
        Lines connect recorded estimates; the final point is terminal scoring.
        Earlier inputs differ by policy observation model, not continuous
        measurements. Values below 10⁻¹² are clipped for display.
      </p>
    </section>
  );
}

export function UncertaintyChart({ run }: { run: SimulationResult }) {
  const points = run.keyframes
    .filter((k) => k.encounter)
    .map((k) => ({ time: k.t_s, ...ellipseGeometry(k.encounter!) }));
  if (!points.length) return null;
  const max = Math.max(1, ...points.map((p) => p.major));
  const duration = points.at(-1)?.time || 1;
  return (
    <section className="instrument uncertainty-chart">
      <div className="instrument-heading">
        <span className="section-label">
          Uncertainty / successive CDM updates
        </span>
        <span className="micro">Principal standard deviations · meters</span>
      </div>
      <svg
        viewBox="0 0 900 125"
        role="img"
        aria-label="Major and minor covariance standard deviations through the episode"
      >
        <line x1="50" y1="95" x2="875" y2="95" stroke="#344639" />
        {(["major", "minor"] as const).map((axis, i) => (
          <polyline
            key={axis}
            fill="none"
            stroke={i ? "#e6b77e" : "#a3d7bd"}
            strokeWidth="1.5"
            points={points
              .map(
                (p) =>
                  `${50 + (p.time / duration) * 825},${95 - (p[axis] / max) * 70}`,
              )
              .join(" ")}
          />
        ))}
        <text x="0" y="28" fill="#a0afa5" fontSize="9">
          {max.toFixed(0)} m
        </text>
        <text x="0" y="99" fill="#a0afa5" fontSize="9">
          0 m
        </text>
        <text x="50" y="119" fill="#a0afa5" fontSize="9">
          First warning
        </text>
        <text x="850" y="119" fill="#a0afa5" fontSize="9">
          TCA
        </text>
      </svg>
      <div className="plot-legend">
        <span>
          <i className="dot mint" />
          Major axis · 1σ
        </span>
        <span>
          <i className="dot amber" />
          Minor axis · 1σ
        </span>
      </div>
      <p className="fine">
        Flat lines mean this replay used fixed covariance. They must not be
        interpreted as evolving-uncertainty training evidence.
      </p>
    </section>
  );
}

export function FuelChart({ runs }: { runs: SimulationResult[] }) {
  const maxFuel = Math.max(
    0.01,
    ...runs.map((r) => r.result.total_fuel_used_ms),
  );
  return (
    <section className="instrument chart">
      <div className="instrument-heading">
        <span className="section-label">04 / The cost of avoidance</span>
        <span className="micro">Lower & left is better</span>
      </div>
      <svg
        viewBox="0 0 430 205"
        role="img"
        aria-label="Fuel consumption versus final collision probability, one point per available policy"
      >
        {[0, 1, 2, 3].map((n) => (
          <line
            key={n}
            x1="45"
            x2="405"
            y1={25 + n * 46}
            y2={25 + n * 46}
            stroke="#263430"
          />
        ))}
        {runs.map((r, i) => {
          const x = 45 + (r.result.total_fuel_used_ms / maxFuel) * 310;
          const y =
            25 +
            (Math.min(12, -Math.log10(Math.max(r.result.pc_final, 1e-12))) /
              12) *
              138;
          return (
            <g key={r.id ?? i}>
              <circle
                cx={x}
                cy={y}
                r="4"
                fill={i === 0 ? "#a3d7bd" : "#e6b77e"}
              />
              <text
                x={Math.min(x + 8, 290)}
                y={y - 8}
                fill="#d6dfd5"
                fontSize="10"
              >
                {policyName(r.policy_id ?? "v1")}
              </text>
            </g>
          );
        })}
        <text x="45" y="195" fill="#a5b0a9" fontSize="10">
          0 m/s Δv
        </text>
        <text x="330" y="195" fill="#a5b0a9" fontSize="10">
          {maxFuel.toFixed(2)} m/s
        </text>
        <text x="0" y="30" fill="#a5b0a9" fontSize="10">
          10⁰
        </text>
        <text x="0" y="166" fill="#a5b0a9" fontSize="10">
          10⁻¹²
        </text>
      </svg>
      <p className="fine">
        A per-event comparison, not a population Pareto frontier. Zero Pc
        reflects numerical output, not proof of zero risk.
      </p>
    </section>
  );
}
