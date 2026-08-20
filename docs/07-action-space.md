# 07 — Action Space Design

Builds on `02-bsk_rl-architecture.md` §3 (`ImpulsiveThrustHill`, the
`ContinuousAction` constraint of exactly one continuous action per
satellite) and `03-scenario-design.md`'s decision cadence (one decision per
CDM-update step, not a fixed fine timestep).

## Decision: pure continuous action, no separate discrete "burn/wait" gate

`bsk_rl`'s `ContinuousAction` framework allows exactly one continuous
action spec per satellite (`02-bsk_rl-architecture.md` §3). Rather than
building a custom hybrid discrete(burn/wait)+continuous(Δv vector) action
space — which would require a non-standard policy architecture (most RL
libraries, including Stable-Baselines3's default PPO, assume a single
action distribution family, not a mix) — v1 uses **`ImpulsiveThrustHill`
directly**: the action is a 4-vector `[dv_R, dv_T, dv_N, coast_duration]`
in the secondary object's Hill/RTN frame (Hill frame chosen over pure
inertial per `02`'s note that conjunction geometry is naturally reasoned
about relative to the encounter, not in absolute inertial coordinates).

"Wait" is not a separate action — it's what the policy learns to output
when Δv ≈ 0. This is simpler to implement and train, and is not a
meaningful loss of realism: a real operator's "do nothing this cycle"
decision *is* equivalent to choosing zero burn.

## Handling the near-zero-action problem

A pure continuous policy exploring with Gaussian noise around a
near-zero mean will still occasionally emit tiny nonzero Δv values that
aren't meaningful "maneuvers" — just exploration jitter. Two mitigations,
both applied:

1. **Fuel-cost reward term** (`08-reward-function.md`) naturally penalizes
   any nonzero Δv, giving the policy gradient pressure to converge to
   exactly zero when zero is optimal — this is the primary mechanism and
   should dominate.
2. **An implementation-level deadzone**: Δv magnitudes below a small
   threshold (e.g. 1 mm/s — far below any operationally meaningful
   avoidance burn, which are typically cm/s–m/s scale) are treated as
   exactly zero for fuel-accounting and "maneuver count" metrics, purely to
   keep evaluation metrics (`11-evaluation.md`) from being noisy artifacts
   of policy exploration variance. This does **not** change the reward
   computation during training (the true, un-deadzoned Δv is what's
   charged against fuel during training) — it only affects how we *report*
   "did the agent maneuver" for analysis/baseline-comparison purposes.

## Action bounds

`ImpulsiveThrustHill(max_dv=..., max_drift_duration=...)` constructor
arguments bound the action space (per `02`). v1 sets:
- `max_dv`: informed by realistic collision-avoidance maneuver magnitudes
  (typically cm/s to a few m/s for LEO conjunction avoidance — this is a
  commonly cited operational range, not independently re-derived from a
  primary source in this research pass; worth confirming against a
  primary operational reference before publishing final bounds). Set
  generously above that range so the policy is never bound-constrained
  during learning, then check post-training whether learned actions stay
  well within the realistic range (a sanity check on the policy's
  behavior, not just a hyperparameter).
- `max_drift_duration`: bounded by the time until the next scheduled CDM
  update (an action's coast shouldn't span past the point where the agent
  gets new information and a new decision opportunity) — computed
  dynamically per step, not a single fixed constant.

## Why not act at a fixed fine timestep instead

An alternative design would have the agent act every N minutes/hours
throughout the encounter window, rather than once per CDM update. Rejected
for v1 because: (a) it doesn't match the real decision cadence (operators
act on new information, i.e. new CDMs, not on a clock), (b) it produces
much longer episodes (days of fine timesteps vs. ~6–12 CDM updates) for no
additional decision-relevant information between updates — the true state
evolves continuously but the agent's *actionable information* only changes
at CDM updates in our model, so intermediate steps would just be wasted
compute with an unchanged observation. If a future version models
continuous re-tracking (updates arriving at irregular, agent-uninformed
times) this assumption would need revisiting — noted for `13-roadmap.md`.
