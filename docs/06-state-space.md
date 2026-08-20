# 06 — State (Observation) Space Design

Builds on `02-bsk_rl-architecture.md` §4 (`SatProperties`/`RelativeProperties`,
the `fn=` hook for injecting externally-computed values) and
`03-scenario-design.md` (the CDM-sequence structure, evolving covariance).

## A key epistemic design decision: the agent observes *estimates*, not ground truth

Basilisk propagates the **true** relative state of ego and secondary object.
A real conjunction-assessment system never knows this true state — it knows
an *estimate* (from ground-based tracking/orbit determination) plus a
covariance describing how wrong that estimate might be. If we gave the
agent the true relative state directly, we'd be solving an easier, less
realistic problem (and an easier RL problem, ironically less interesting
pedagogically).

**Decision**: at each CDM-update step, the observed relative position/
velocity = true relative position/velocity (from Basilisk) **+ noise drawn
from the current covariance** (`N(0, Σ_t)`), and the covariance `Σ_t` itself
(shrinking per `03-scenario-design.md`) is given to the agent as an
explicit feature — it's the agent's belief about its own uncertainty, not
something it has to infer. This makes v1 a genuinely partially-observable
problem: the agent must reason about risk under its own estimate
uncertainty, and — because covariance shrinks over the CDM sequence — there
is a real, learnable value to waiting for a better estimate before
committing fuel to a maneuver.

## Observation vector (per decision step)

All continuous features are normalized via `bsk_rl.obs.SatProperties`'
built-in `norm` support, using scales derived from the Kelvins-fitted
distributions (`03-scenario-design.md`) so features stay in a
comparable range for the policy network — exact norm constants are an
implementation detail tuned during Phase 4 (`13-roadmap.md`), not fixed
here.

| Feature | Source | Notes |
|---|---|---|
| Relative position (RTN/Hill frame), 3 components | `RelativeProperties` + noise per above | Estimated, not true, state |
| Relative velocity (RTN/Hill frame), 3 components | `RelativeProperties` + noise | Same |
| Combined covariance, principal-axis std devs (σx, σz) + orientation angle in encounter plane | our Pc module (`04-collision-probability.md`), injected via `SatProperties(fn=...)` | Summary of the full covariance, not the raw 6×6 — keeps the observation compact and matches what the Pc computation itself uses internally |
| Current computed Pc | our Pc module, `SatProperties(fn=compute_pc)` | The single most decision-relevant scalar |
| Time to TCA | `bsk_rl.obs.Time` or custom `fn` | Remaining decision window |
| Time to next scheduled CDM update | custom `fn` | Lets the agent reason about "how much better will my information get before I must decide again" |
| Own remaining Δv budget (`dv_available` fraction) | `SatProperties(prop="dv_available", norm=dv_available_init)` (per `02-bsk_rl-architecture.md` §2) | Fuel state |
| Combined hard-body radius (HBR) | scenario metadata, static per-episode `fn` | Constant within an episode, but varies across episodes (different object sizes) — needs to be observed since it's not otherwise inferable |
| Own semi-major axis, eccentricity | `SatProperties` orbital-element props (per `02` §2) | Context for how costly/effective a given Δv is in this orbit regime; low-dimensional, cheap to include, aids generalization across scenarios at different altitudes |
| Number of CDM updates seen so far this episode | custom `fn` / step counter | Lets the policy learn schedule-position-dependent behavior (e.g., be more willing to wait early, more willing to act as TCA nears) without having to infer it indirectly from time-to-TCA alone |

## What's deliberately excluded from v1's observation

- **Raw full covariance matrix** — the diagonalized 2-parameter summary
  (σx, σz, orientation) carries the same risk-relevant information the Pc
  computation itself uses; the full matrix adds dimensionality without
  adding decision-relevant signal for this task.
- **Background object state** — v1 has no background field
  (`03-scenario-design.md`); this becomes relevant only once v1.1's
  populated background field is added.
- **Attitude state** — v1 treats maneuvers as impulsive Δv (per
  `02-bsk_rl-architecture.md` §2), so attitude isn't part of the decision.

## Partial-observability caveat to track

Because the agent's observed relative state includes injected noise, two
episodes with identical *true* underlying scenario parameters can look
different to the agent — this is intentional (matches reality) but means
evaluation needs enough episodes per scenario type to distinguish policy
quality from observation-noise variance. Flagging this now so it's not
mistaken for training instability later — see `11-evaluation.md`.
