import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import "./App.css";
import { cancelJob, createJob, loadIndex, loadReplay, pollJob } from "./api";
import type { Job } from "./api";
import type { ReplayEntry, SimulationResult } from "./types";
import { useSimulationPlayback } from "./useSimulationPlayback";
import { explainDecision } from "./decisionExplain";
import {
  EncounterPlane,
  FuelChart,
  RiskChart,
  UncertaintyChart,
} from "./components/ResearchPlots";
import { policyName, probability } from "./evidence";
import ExperimentEvidence from "./components/ExperimentEvidence";
const OrbitalContextView = lazy(
  () => import("./components/OrbitalContextView"),
);

type Page = "mission" | "briefing" | "evidence";
const lessons = [
  [
    "A close approach is not a collision.",
    "A conjunction is a predicted close approach between two orbiting objects. Here, historical ESA geometry anchors a new simulated encounter. The orbital tracks are simulator output, not a reconstruction of a historical satellite's flight.",
  ],
  [
    "Every warning has an uncertainty.",
    "A conjunction data message (CDM) refines a predicted encounter. The ellipse represents position uncertainty in the plane perpendicular to relative velocity. The miss vector locates its center relative to the protected satellite.",
  ],
  [
    "Probability is geometry, not distance alone.",
    "Pc integrates the uncertain relative position over the combined hard-body collision region. A small miss distance does not by itself imply high risk. Covariance shape, orientation, and the uncertain object sizes all matter.",
  ],
  [
    "Small decisions accumulate.",
    "Δv is a change in velocity, measured in meters per second. An early, small maneuver can change the future encounter. Every burn consumes a finite budget; a useful policy must also know when to coast.",
  ],
  [
    "A successful dodge is only half the test.",
    "The original PPO policy often makes similar small maneuvers across different risks. That can mitigate a dangerous encounter without demonstrating reactive judgment. Compare safe-event fuel use, matched counterfactuals, and held-out event families.",
  ],
];

function download(result: SimulationResult) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = `${result.id ?? "apsis-replay"}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

export default function App() {
  const initial = useMemo(
    () => new URLSearchParams(window.location.search),
    [],
  );
  const [entries, setEntries] = useState<ReplayEntry[]>([]);
  const [eventId, setEventId] = useState(initial.get("event") ?? "8767");
  const [policyId, setPolicyId] = useState(initial.get("policy") ?? "v1");
  const [runs, setRuns] = useState<SimulationResult[]>([]);
  const [page, setPage] = useState<Page>("mission");
  const [expert, setExpert] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [lesson, setLesson] = useState(0);
  const [radiusFactor, setRadiusFactor] = useState(1);
  const [geometryMode, setGeometryMode] = useState<"observed" | "truth">(
    "observed",
  );
  const [overlay, setOverlay] = useState(true);
  const [job, setJob] = useState<Job | null>(null);
  const [live, setLive] = useState(false);
  const [variant, setVariant] = useState("historical_geometry");
  const [liveSeed, setLiveSeed] = useState(0);
  const [liveRadius, setLiveRadius] = useState(1);
  const [scheduleMode, setScheduleMode] = useState("controlled");
  const result = runs.find((r) => r.policy_id === policyId) ?? runs[0] ?? null;
  const playback = useSimulationPlayback(result);
  const eventEntries = entries.filter((e) => String(e.event_id) === eventId);
  const events = [
    ...new Map(entries.map((e) => [String(e.event_id), e])).values(),
  ];
  const activeEntry =
    eventEntries.find((e) => e.policy_id === result?.policy_id) ??
    eventEntries[0];

  useEffect(() => {
    let cancelled = false;
    loadIndex()
      .then((index) => {
        if (cancelled) return;
        if (!index.replays.length)
          throw new Error(
            "No verified replays are installed. Generate replay artifacts to populate the laboratory.",
          );
        setEntries(index.replays);
        setEventId((current) =>
          index.replays.some((e) => String(e.event_id) === current)
            ? current
            : String(index.replays[0].event_id),
        );
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e.message);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);
  useEffect(() => {
    if (!entries.length || !eventId) return;
    let cancelled = false;
    // This effect synchronizes a selected historical event with external replay files.
    // eslint-disable-next-line react/set-state-in-effect
    setLoading(true);
    setError("");
    setLive(false);
    Promise.all(
      entries
        .filter((e) => String(e.event_id) === eventId)
        .map(async (e) => ({
          ...(await loadReplay(e.path)),
          id: e.id,
          policy_id: e.policy_id,
        })),
    )
      .then((data) => {
        if (cancelled) return;
        setRuns(data);
        setPolicyId((current) =>
          data.some((r) => r.policy_id === current)
            ? current
            : (data[0]?.policy_id ?? ""),
        );
        setLoading(false);
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e.message);
          setRuns([]);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [eventId, entries]);
  useEffect(() => {
    if (!job || !["pending", "queued", "running"].includes(job.status)) return;
    const timer = window.setTimeout(() => {
      pollJob(job.id)
        .then((next) => {
          setJob(next);
          if (["completed", "succeeded"].includes(next.status)) {
            if (next.result) {
              setRuns([next.result]);
              setLive(!next.cached);
            } else if (next.artifact_url)
              loadReplay(next.artifact_url)
                .then((r) => {
                  setRuns([r]);
                  setLive(!next.cached);
                })
                .catch((e) => setError(e.message));
          }
          if (next.status === "failed")
            setError(`Simulation failed: ${JSON.stringify(next.error)}`);
        })
        .catch((e) => setError(e.message));
    }, 1000);
    return () => clearTimeout(timer);
  }, [job]);

  const currentDecision = result
    ? Math.max(
        0,
        result.decisions.findLastIndex((d) => d.t_s <= playback.missionTimeS),
      )
    : 0;
  const decision = result?.decisions[currentDecision];
  const terminalEncounter = result?.keyframes.at(-1)?.truth_encounter;
  const encounter =
    geometryMode === "truth"
      ? terminalEncounter
      : (playback.currentKeyframe?.encounter ?? decision?.encounter);
  const busy = !!job && ["pending", "queued", "running"].includes(job.status);
  const share = async () => {
    if (live || !entries.some((e) => e.id === result?.id)) {
      setNotice(
        "This local run is not in the public replay catalog. Download its artifact to share the evidence.",
      );
      return;
    }
    const url = new URL(window.location.href);
    url.search = new URLSearchParams({
      event: eventId,
      policy: result?.policy_id ?? policyId,
    }).toString();
    window.history.replaceState(null, "", url);
    try {
      await navigator.clipboard.writeText(url.href);
      setNotice("Deterministic replay link copied.");
    } catch {
      setNotice("Replay URL is ready in your address bar.");
    }
  };
  async function runLive() {
    setError("");
    try {
      const next = await createJob(
        `kelvins-${eventId.replace(/^kelvins-/, "")}`,
        result?.policy_id ?? policyId,
        liveSeed,
        variant,
        liveRadius,
        scheduleMode,
      );
      setJob(next);
      if (next.status === "completed") {
        const artifact =
          next.result ??
          (next.artifact_url ? await loadReplay(next.artifact_url) : null);
        if (artifact) {
          setRuns([artifact]);
          setPolicyId(artifact.policy_id ?? policyId);
          setLive(!next.cached);
          setNotice(
            next.cached
              ? "Loaded a deterministic cached simulation; no new physics job was executed."
              : "New physics simulation completed.",
          );
        }
      }
    } catch (e) {
      setError(
        `Live physics service unavailable. Verified replays remain available. ${e instanceof Error ? e.message : e}`,
      );
    }
  }

  return (
    <div className="app">
      <a className="skip-link" href="#main">
        Skip to mission
      </a>
      <header className="masthead">
        <a href="/" className="brand" aria-label="Apsis home">
          <svg viewBox="0 0 40 40" aria-hidden="true">
            <ellipse
              cx="20"
              cy="20"
              rx="16"
              ry="8"
              transform="rotate(-40 20 20)"
            />
            <circle cx="20" cy="20" r="3" />
            <circle cx="32" cy="10" r="2" />
          </svg>
          <span>
            APSIS<small>CONJUNCTION RESEARCH INSTRUMENT</small>
          </span>
        </a>
        <nav aria-label="Main navigation">
          {(
            [
              ["mission", "Mission control"],
              ["briefing", "The briefing"],
              ["evidence", "Research & evidence"],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              onClick={() => setPage(id)}
              aria-current={page === id ? "page" : undefined}
            >
              {label}
            </button>
          ))}
        </nav>
        <button
          className="density"
          aria-pressed={expert}
          onClick={() => setExpert(!expert)}
        >
          {expert ? "Expert detail" : "Essential detail"}
          <span>↗</span>
        </button>
      </header>
      <main id="main">
        <div className="page-heading">
          <div>
            <div className="eyebrow">
              SATELLITE COLLISION AVOIDANCE / OPEN RESEARCH
            </div>
            <h1>
              {page === "mission" ? (
                <>
                  Room to <em>maneuver.</em>
                </>
              ) : page === "briefing" ? (
                <>
                  The space <em>between.</em>
                </>
              ) : (
                <>
                  Show the <em>evidence.</em>
                </>
              )}
            </h1>
            <p>
              {page === "mission"
                ? "One encounter. Several policies. Every decision has a cost."
                : page === "briefing"
                  ? "A guided encounter with the geometry of uncertainty."
                  : "A successful maneuver is an outcome. Reactive behavior is a research question."}
            </p>
          </div>
          <div className="mode-stamp">
            <i className="status-dot" />
            <span>
              {live ? "LIVE RUN COMPLETE" : "REPLAY LABORATORY"}
              <small>
                {live
                  ? "New simulator output"
                  : "Recorded physics · reproducible evidence"}
              </small>
            </span>
          </div>
        </div>
        {error && (
          <div className="error-banner" role="alert">
            {error}
            <button onClick={() => window.location.reload()}>
              Reload artifacts
            </button>
          </div>
        )}
        {notice && (
          <div className="notice" role="status">
            {notice}
            <button onClick={() => setNotice("")}>Dismiss</button>
          </div>
        )}
        {page === "briefing" && (
          <section className="briefing instrument">
            <div className="briefing-index">
              FIELD NOTES / 0{lesson + 1}
              <div className="lesson-steps">
                {lessons.map((_, i) => (
                  <button
                    key={i}
                    aria-label={`Briefing chapter ${i + 1}`}
                    aria-current={lesson === i ? "step" : undefined}
                    onClick={() => setLesson(i)}
                  >
                    {i + 1}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <h2>{lessons[lesson][0]}</h2>
              <p>{lessons[lesson][1]}</p>
              <button
                className="primary"
                onClick={() =>
                  lesson < lessons.length - 1
                    ? setLesson(lesson + 1)
                    : setPage("mission")
                }
              >
                {lesson < lessons.length - 1
                  ? "Next field note →"
                  : "Explore the encounter →"}
              </button>
            </div>
          </section>
        )}
        {page === "evidence" && (
          <section className="evidence-grid">
            <article className="instrument">
              <span className="section-label">Research question</span>
              <h2>
                Does the policy react
                <br />
                to risk?
              </h2>
              <p>
                The v1 checkpoint demonstrated inexpensive mitigation relative
                to crude baselines, but near-constant burns are not evidence of
                risk-conditioned decisions.
              </p>
              <p>
                Use held-out source events, safe-event false positives, fuel
                regret, and matched risk/covariance counterfactuals. Reward
                alone does not pass this test.
              </p>
              <a href="/docs/EXPERIMENT_PROTOCOL.md">
                Read the experiment protocol ↗
              </a>
            </article>
            <article className="instrument">
              <span className="section-label">
                What this installation contains
              </span>
              <h2>
                {new Set(entries.map((e) => e.policy_id)).size} policies ·{" "}
                {events.length} events
              </h2>
              <p>
                These counts describe installed replay artifacts, not an
                independent test set or a generalization result.
              </p>
              <div className="model-list">
                {[...new Set(entries.map((e) => e.policy_id))].map((id) => (
                  <div key={id}>
                    <span>{policyName(id)}</span>
                    <span className="micro">
                      {id.includes("v2") ? "EXPERIMENTAL" : "RECORDED"}
                    </span>
                  </div>
                ))}
              </div>
              <p className="fine">
                A planner is a numerical comparison policy, not a globally
                optimal or operational flight solution.
              </p>
            </article>
            <article className="instrument provenance">
              <span className="section-label">
                Data lineage / selected replay
              </span>
              <dl>
                <dt>Source event</dt>
                <dd>
                  ESA Kelvins · {result?.provenance?.source_event_id ?? eventId}
                </dd>
                <dt>Geometry type</dt>
                <dd>
                  {result?.provenance?.kind?.replaceAll("_", " ") ??
                    "Not recorded"}
                </dd>
                <dt>Simulator</dt>
                <dd>{result?.provenance?.simulator ?? "Not recorded"}</dd>
                <dt>Generated</dt>
                <dd>{result?.provenance?.generated_at ?? "Not recorded"}</dd>
                <dt>Seed</dt>
                <dd>{result?.seed ?? "—"}</dd>
                <dt>Checkpoint SHA-256</dt>
                <dd className="hash">
                  {result?.provenance?.checkpoint_sha256 ??
                    "Not applicable / not recorded"}
                </dd>
              </dl>
              <p>{result?.provenance?.fidelity_note}</p>
              {result?.provenance?.verification && (
                <details>
                  <summary>Replay verification measurements</summary>
                  <pre>
                    {JSON.stringify(result.provenance.verification, null, 2)}
                  </pre>
                </details>
              )}
              <p className="fine">
                ESA data attribution and licensing remain separate from the code
                license. Historical geometry is used to construct simulator
                scenarios; it is not an operational risk assessment.
              </p>
            </article>
          </section>
        )}
        {page === "evidence" && <ExperimentEvidence />}
        <div className="workspace">
          <aside className="scenario-rail">
            <div className="rail-title">
              <span className="section-label">Encounter catalog</span>
              <span className="micro">
                {String(events.length).padStart(2, "0")}
              </span>
            </div>
            <p className="fine">
              Historical geometry.
              <br />
              Simulated decisions.
            </p>
            <div className="event-list">
              {events.map((e, i) => (
                <button
                  className={`event-card ${String(e.event_id) === eventId ? "selected" : ""}`}
                  key={String(e.event_id)}
                  disabled={busy}
                  onClick={() => setEventId(String(e.event_id))}
                  aria-pressed={String(e.event_id) === eventId}
                >
                  <span className="event-number">
                    {String(i + 1).padStart(2, "0")} / KELVINS
                  </span>
                  <strong>{e.title || `Event ${e.event_id}`}</strong>
                  <span className="micro">SOURCE {e.event_id}</span>
                  <span className="event-arrow">↗</span>
                </button>
              ))}
            </div>
            <div className="rail-note">
              <span className="status-dot" />
              <p>
                Every plotted result comes from an installed run artifact. No
                policy outcome is invented for presentation.
              </p>
            </div>
          </aside>
          <div className="mission-body" aria-busy={loading}>
            {loading && (
              <div className="loading instrument" role="status">
                <div className="loading-orbit" />
                <h2>Acquiring encounter data</h2>
                <p>Loading recorded trajectories and decision evidence.</p>
              </div>
            )}
            {!loading && !result && (
              <div className="empty-state instrument">
                <h2>The laboratory is ready for evidence.</h2>
                <p>
                  No replay could be loaded. Install or generate verified
                  artifacts under <code>public/replays</code>, then reload.
                </p>
              </div>
            )}
            {result && !loading && (
              <>
                <div className="mission-topbar">
                  <div
                    className="policy-switch"
                    role="group"
                    aria-label="Replay policy"
                  >
                    {runs.map((e) => (
                      <button
                        key={e.id}
                        disabled={busy}
                        aria-pressed={result.policy_id === e.policy_id}
                        onClick={() => setPolicyId(e.policy_id ?? "")}
                      >
                        {policyName(e.policy_id ?? "")}
                      </button>
                    ))}
                  </div>
                  {job?.status === "completed" && (
                    <button
                      className="text-button"
                      onClick={() => {
                        setJob(null);
                        setEntries((current) => [...current]);
                      }}
                    >
                      Return to verified comparisons
                    </button>
                  )}
                  <button
                    onClick={share}
                    className="text-button"
                    disabled={!!job || !entries.some((e) => e.id === result.id)}
                    aria-label={
                      job ? "Download live artifact to share" : "Share replay"
                    }
                  >
                    {job ? "Share via downloaded artifact" : "Share replay ↗"}
                  </button>
                  <button
                    onClick={() => download(result)}
                    className="text-button"
                  >
                    ↓ Artifact
                  </button>
                </div>
                <div className="run-context micro">
                  {result.provenance?.kind?.replaceAll("_", " ") ??
                    "Geometry lineage not recorded"}{" "}
                  ·{" "}
                  {result.provenance?.schedule_kind?.replaceAll("_", " ") ??
                    "Schedule not recorded"}{" "}
                  · {result.decisions.length} decisions
                </div>
                <div className="geometry-toolbar">
                  <div role="group" aria-label="Encounter geometry source">
                    <button
                      aria-pressed={geometryMode === "observed"}
                      onClick={() => setGeometryMode("observed")}
                    >
                      Observed prediction
                    </button>
                    <button
                      aria-pressed={geometryMode === "truth"}
                      disabled={!terminalEncounter}
                      onClick={() => setGeometryMode("truth")}
                    >
                      Terminal simulator truth
                    </button>
                  </div>
                  <button
                    aria-pressed={overlay}
                    onClick={() => setOverlay(!overlay)}
                  >
                    {overlay ? "Hide" : "Show"} no-maneuver orbit
                  </button>
                </div>
                <div className="visual-grid">
                  <section className="orbital-instrument">
                    <div className="orbital-overlay">
                      <span className="section-label">
                        01 / Orbital context · ECI
                      </span>
                      <h2>Event {eventId}</h2>
                      <span className="record-label">
                        {live
                          ? "NEW LIVE RUN"
                          : activeEntry?.verified
                            ? "VERIFIED REPLAY"
                            : "RECORDED REPLAY"}
                      </span>
                    </div>
                    <Suspense
                      fallback={
                        <div className="loading">
                          Loading orbital instrument…
                        </div>
                      }
                    >
                      <OrbitalContextView
                        frame={playback.frame}
                        missionTimeS={playback.missionTimeS}
                        earthRadiusM={result.constants.earth_radius_m}
                        denseFrames={result.dense_frames}
                        comparisonFrames={
                          overlay
                            ? runs.find(
                                (r) =>
                                  r.policy_id === "never_maneuver" &&
                                  r.policy_id !== result.policy_id,
                              )?.dense_frames
                            : undefined
                        }
                        maneuversSoFar={playback.maneuversSoFar}
                      />
                    </Suspense>
                    <div className="orbital-footer">
                      <span>
                        <i className="dot mint" />
                        Protected satellite
                      </span>
                      <span>
                        <i className="dot amber" />
                        Conjunction partner
                      </span>
                      <small>
                        Markers enlarged · Earth lighting illustrative
                      </small>
                    </div>
                  </section>
                  <EncounterPlane
                    mode={geometryMode}
                    sensitivity={result.result.radius_sensitivity_foster}
                    encounter={encounter}
                    radius={result.scenario.combined_radius_m}
                    radiusFactor={radiusFactor}
                    setRadiusFactor={setRadiusFactor}
                  />
                </div>
                <section
                  className="transport instrument"
                  aria-label="Synchronized replay controls"
                >
                  <div className="transport-main">
                    <button
                      className="play-button"
                      onClick={
                        playback.finished
                          ? playback.restart
                          : playback.togglePlay
                      }
                    >
                      {playback.finished
                        ? "↻ Replay"
                        : playback.playing
                          ? "Ⅱ Pause"
                          : "▷ Play"}
                    </button>
                    <button
                      aria-label="Previous decision"
                      onClick={() => playback.stepToKeyframe(-1)}
                    >
                      ←
                    </button>
                    <button
                      aria-label="Next decision"
                      onClick={() => playback.stepToKeyframe(1)}
                    >
                      →
                    </button>
                    <span className="mission-clock">
                      T −{" "}
                      {(
                        (playback.missionDurationS - playback.missionTimeS) /
                        3600
                      ).toFixed(2)}
                      <small> HOURS TO TCA</small>
                    </span>
                    <label className="speed-label">
                      Speed
                      <select
                        aria-label="Playback speed"
                        value={playback.speedFactor}
                        onChange={(e) =>
                          playback.setSpeedFactor(Number(e.target.value))
                        }
                      >
                        {[0.25, 0.5, 1, 2, 4].map((n) => (
                          <option key={n} value={n}>
                            {n}×
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <input
                    className="mission-seek"
                    type="range"
                    aria-label="Mission time"
                    min="0"
                    max={playback.missionDurationS}
                    step="1"
                    value={playback.missionTimeS}
                    onChange={(e) => {
                      playback.pause();
                      playback.seek(Number(e.target.value));
                    }}
                  />
                  <div className="timeline-labels">
                    <span>FIRST WARNING</span>
                    <span>
                      {result.decisions.length} DECISION UPDATES · LINKED
                      TIMELINE
                    </span>
                    <span>TCA</span>
                  </div>
                </section>
                <div className="metric-strip">
                  <div>
                    <span className="section-label">
                      Historical miss distance
                    </span>
                    <strong>
                      {result.scenario.miss_distance_m.toFixed(0)}
                      <small>m</small>
                    </strong>
                  </div>
                  <div>
                    <span className="section-label">
                      {playback.finished
                        ? "Terminal scored Pc"
                        : "Observed Pc / current update"}
                    </span>
                    <strong>
                      {probability(playback.currentKeyframe?.pc_estimate)}
                    </strong>
                  </div>
                  <div>
                    <span className="section-label">Final simulated Pc</span>
                    <strong
                      className={
                        result.result.pc_final <= result.constants.pc_threshold
                          ? "mint-text"
                          : "coral-text"
                      }
                    >
                      {probability(result.result.pc_final)}
                    </strong>
                  </div>
                  <div>
                    <span className="section-label">Total maneuver cost</span>
                    <strong>
                      {result.result.total_fuel_used_ms.toFixed(3)}
                      <small>m/s Δv</small>
                    </strong>
                  </div>
                </div>
                <div className="charts-grid">
                  <RiskChart
                    runs={runs}
                    time={playback.missionTimeS}
                    threshold={result.constants.pc_threshold}
                  />
                  <FuelChart runs={runs} />
                </div>
                <UncertaintyChart run={result} />
                <div className="comparison-table instrument">
                  <div className="section-label">
                    Identical source event / whole-episode outcomes
                  </div>
                  <p className="fine">
                    Historical ESA-reported Pc:{" "}
                    {probability(result.scenario.esa_reported_pc)}. Recomputed
                    source-geometry Pc:{" "}
                    {probability(result.scenario.source_native_pc)}. Recomputed
                    simulator outcomes use generated reference orbits and each
                    policy’s recorded trajectory.
                  </p>
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Policy</th>
                          <th>Final simulated Pc</th>
                          <th>Total Δv</th>
                          <th>Maneuvers</th>
                          <th>Below threshold</th>
                        </tr>
                      </thead>
                      <tbody>
                        {runs.map((run) => (
                          <tr key={run.id}>
                            <td>{policyName(run.policy_id ?? "")}</td>
                            <td>{probability(run.result.pc_final)}</td>
                            <td>
                              {run.result.total_fuel_used_ms.toFixed(3)} m/s
                            </td>
                            <td>{run.result.maneuver_count}</td>
                            <td>
                              {run.result.pc_final <= run.constants.pc_threshold
                                ? "Yes"
                                : "No"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="fine">
                    Single-event comparisons do not estimate population success.
                    A completed pass is not proof of no physical collision.
                  </p>
                </div>
                <section className="decision-section instrument">
                  <div className="instrument-heading">
                    <span className="section-label">
                      05 / Decision evidence
                    </span>
                    <span className="micro">OBSERVATION → RECORDED ACTION</span>
                  </div>
                  <div className="decision-buttons">
                    {result.decisions.map((d, i) => (
                      <button
                        key={i}
                        aria-pressed={currentDecision === i}
                        onClick={() => {
                          playback.pause();
                          playback.seek(d.t_s);
                        }}
                      >
                        CDM {String(i + 1).padStart(2, "0")}
                        <small>
                          {d.action_magnitude_ms > 0.001 ? "MANEUVER" : "COAST"}
                        </small>
                      </button>
                    ))}
                  </div>
                  {decision && (
                    <div className="decision-detail">
                      <div>
                        <h3>
                          {decision.action_magnitude_ms > 0.001
                            ? `${decision.action_magnitude_ms.toFixed(3)} m/s recorded burn`
                            : "No measurable burn"}
                        </h3>
                        <p>
                          {explainDecision(
                            decision,
                            result.keyframes[currentDecision] ?? null,
                            result.constants.pc_threshold,
                          )}
                        </p>
                        <p className="fine">
                          The comparison policies are whole-episode
                          counterfactuals. They do not establish the causal
                          effect of this individual action.
                        </p>
                      </div>
                      <dl>
                        <dt>Action / RTN m/s</dt>
                        <dd>
                          {decision.action_dv_ms
                            .map((n) => n.toFixed(4))
                            .join(" / ")}
                        </dd>
                        <dt>Fuel before → after</dt>
                        <dd>
                          {decision.fuel_before_ms.toFixed(3)} →{" "}
                          {decision.fuel_after_ms.toFixed(3)} m/s
                        </dd>
                        <dt>Observation model</dt>
                        <dd>
                          {decision.observation?.version ??
                            "v1 · simulator state"}
                        </dd>
                      </dl>
                    </div>
                  )}
                  {expert && decision?.observation && (
                    <details>
                      <summary>Inspect policy input vector</summary>
                      <pre>{JSON.stringify(decision.observation, null, 2)}</pre>
                    </details>
                  )}
                </section>
                <section className="live-panel instrument">
                  <div>
                    <span className="section-label">Run your own physics</span>
                    <h3>From replay to experiment.</h3>
                    <p>
                      Execute this event with the selected policy on a local
                      Basilisk installation.
                    </p>
                    <div className="live-settings">
                      <label>
                        CDM schedule
                        <select
                          aria-label="Live CDM schedule"
                          value={scheduleMode}
                          onChange={(e) => setScheduleMode(e.target.value)}
                        >
                          <option value="controlled">
                            Controlled comparison
                          </option>
                          <option value="source">Full source schedule</option>
                        </select>
                      </label>
                      <label>
                        Geometry
                        <select
                          aria-label="Live geometry type"
                          value={variant}
                          onChange={(e) => setVariant(e.target.value)}
                        >
                          <option value="historical_geometry">
                            Historical geometry
                          </option>
                          <option value="posterior_variant">
                            Posterior-resampled variant
                          </option>
                        </select>
                      </label>
                      <label>
                        Seed
                        <input
                          aria-label="Simulation seed"
                          type="number"
                          min="0"
                          max="1000000"
                          value={liveSeed}
                          onChange={(e) =>
                            setLiveSeed(
                              Math.max(
                                0,
                                Math.min(
                                  1000000,
                                  Math.floor(Number(e.target.value)),
                                ),
                              ),
                            )
                          }
                        />
                      </label>
                      <label>
                        Radius scale
                        <select
                          aria-label="Live radius scale"
                          value={liveRadius}
                          onChange={(e) =>
                            setLiveRadius(Number(e.target.value))
                          }
                        >
                          {[0.5, 1, 2].map((n) => (
                            <option key={n} value={n}>
                              {n}×
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                  </div>
                  <div>
                    {busy ? (
                      <>
                        <span role="status">
                          {job?.status}{" "}
                          {job?.progress != null
                            ? `${Math.round(job.progress * 100)}%`
                            : ""}
                        </span>
                        <button
                          onClick={() =>
                            job &&
                            cancelJob(job.id)
                              .then(setJob)
                              .catch((e) => setError(e.message))
                          }
                        >
                          Cancel job
                        </button>
                      </>
                    ) : (
                      <button className="primary" onClick={runLive}>
                        Run live simulation ↗
                      </button>
                    )}
                    <p className="fine">
                      Requires the optional physics service.
                    </p>
                  </div>
                </section>
              </>
            )}
          </div>
        </div>
      </main>
      <footer>
        <span>APSIS / AN OPEN SATELLITE-RL LABORATORY</span>
        <p>Research software. Not for operational collision avoidance.</p>
        <a href="https://github.com/ArmaanSayyad/satellite-rl">
          Source & methodology ↗
        </a>
      </footer>
    </div>
  );
}
