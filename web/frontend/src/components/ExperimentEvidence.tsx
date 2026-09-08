import { useEffect, useState } from "react";

interface EvidenceRun {
  id: string;
  label: string;
  steps: number;
  elapsed_seconds: number;
  checkpoint_sha256?: string;
  config: Record<string, unknown>;
  learning_curve: { step: number; reward: number }[];
  evaluations: {
    label: string;
    partition: string;
    summary: Record<
      string,
      {
        n_episodes: number;
        n_high_risk: number;
        n_source_events?: number;
        n_safe?: number;
        mitigation_rate: number | null;
        mitigation_ci95?: [number, number] | null;
        false_positive_rate: number | null;
        false_positive_ci95?: [number, number] | null;
        mean_fuel_ms: number;
      }
    >;
  }[];
}
interface Evidence {
  status: string;
  runs: EvidenceRun[];
  limitations: string[];
}
const percent = (n: number | null | undefined) =>
  n == null ? "Not estimable" : `${(n * 100).toFixed(0)}%`;

export default function ExperimentEvidence() {
  const [data, setData] = useState<Evidence | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  useEffect(() => {
    let cancelled = false;
    fetch("/replays/evaluation.json")
      .then((r) => {
        if (!r.ok) throw new Error("Evidence unavailable");
        return r.json();
      })
      .then((value) => {
        if (!cancelled) setData(value);
      })
      .catch(() => {
        if (!cancelled) setUnavailable(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);
  if (!data)
    return (
      <section className="instrument experiment-evidence">
        <span className="section-label">Post-training laboratory</span>
        <p>
          {unavailable
            ? "No measured training report is installed. Replay outcomes alone do not establish a trained policy's generalization."
            : "Loading measured training evidence…"}
        </p>
      </section>
    );
  return (
    <section className="experiment-evidence instrument">
      <div className="instrument-heading">
        <span className="section-label">
          Post-training laboratory / actual runs
        </span>
        <a className="micro" href="/replays/evaluation.json" download>
          ↓ Evaluation JSON
        </a>
      </div>
      <h2>{data.status.replaceAll("-", " ")}</h2>
      <p className="fine">
        Observed percentages describe these runs only. Sensor seeds are repeated
        measurements within source families, not independent historical events.
        Missing confidence intervals are not evidence of certainty.
      </p>
      <div className="experiment-runs">
        {data.runs.map((run) => (
          <article key={run.id}>
            <div className="experiment-title">
              <h3>{run.label}</h3>
              <span className="micro">
                {run.steps.toLocaleString()} STEPS ·{" "}
                {(run.elapsed_seconds / 60).toFixed(1)} MIN
              </span>
            </div>
            {run.learning_curve?.length > 1 ? (
              <svg
                viewBox="0 0 600 120"
                role="img"
                aria-label={`Recorded learning curve for ${run.label}`}
              >
                <polyline
                  fill="none"
                  stroke="#77d5f0"
                  strokeWidth="1.5"
                  points={run.learning_curve
                    .map((p, i) => {
                      const values = run.learning_curve.map((v) => v.reward);
                      const min = Math.min(...values);
                      const span = Math.max(...values) - min || 1;
                      return `${20 + (i / (run.learning_curve.length - 1)) * 560},${100 - ((p.reward - min) / span) * 80}`;
                    })
                    .join(" ")}
                />
                <text x="20" y="117" fill="#9cacc2" fontSize="9">
                  Step {run.learning_curve[0].step} →{" "}
                  {run.learning_curve.at(-1)?.step} · reward range{" "}
                  {Math.min(...run.learning_curve.map((p) => p.reward)).toFixed(
                    3,
                  )}{" "}
                  to{" "}
                  {Math.max(...run.learning_curve.map((p) => p.reward)).toFixed(
                    3,
                  )}
                </text>
              </svg>
            ) : (
              <p className="fine">
                No multi-point learning curve recorded for this proof run.
              </p>
            )}
            {run.evaluations.map((evaluation, i) => (
              <div key={i} className="evaluation-table">
                <div className="section-label">
                  {evaluation.label} / {evaluation.partition}
                </div>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Policy</th>
                        <th>Episodes</th>
                        <th>Families</th>
                        <th>High risk</th>
                        <th>Safe</th>
                        <th>Mitigated</th>
                        <th>False burns</th>
                        <th>Mean Δv</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(evaluation.summary).map(([id, row]) => (
                        <tr key={id}>
                          <td>{id.replaceAll("_", " ")}</td>
                          <td>{row.n_episodes}</td>
                          <td>{row.n_source_events ?? "—"}</td>
                          <td>{row.n_high_risk}</td>
                          <td>{row.n_safe ?? "—"}</td>
                          <td>
                            {percent(row.mitigation_rate)}
                            <small>
                              {row.mitigation_ci95
                                ? `95% CI ${row.mitigation_ci95.map(percent).join("–")}`
                                : "95% CI unavailable"}
                            </small>
                          </td>
                          <td>
                            {percent(row.false_positive_rate)}
                            <small>
                              {row.false_positive_ci95
                                ? `95% CI ${row.false_positive_ci95.map(percent).join("–")}`
                                : "95% CI unavailable"}
                            </small>
                          </td>
                          <td>{row.mean_fuel_ms?.toFixed(3) ?? "—"} m/s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
            <details>
              <summary>Run configuration & checkpoint</summary>
              <pre>
                {JSON.stringify(
                  { checkpoint_sha256: run.checkpoint_sha256, ...run.config },
                  null,
                  2,
                )}
              </pre>
            </details>
          </article>
        ))}
      </div>
      <div className="research-limitations">
        <span className="section-label">Limits of the evidence</span>
        {data.limitations.map((s, i) => (
          <p key={i}>{s}</p>
        ))}
      </div>
    </section>
  );
}
