# 10 — RL Algorithm & Training Infrastructure

## Algorithm: PPO

PPO is the natural choice here: on-policy, handles continuous action
spaces natively via a Gaussian policy head (fits `07-action-space.md`'s
pure-continuous `ImpulsiveThrustHill` design directly, no custom
hybrid-action architecture needed), and is comparatively stable/forgiving
of reward-scale and hyperparameter mistakes relative to off-policy
alternatives (SAC, TD3) — a meaningful advantage for a project whose
purpose includes *learning* simulation+RL, where debugging a subtly
miscalibrated off-policy algorithm on top of a novel custom environment
would confound two hard problems (environment correctness vs. algorithm
tuning) at once.

## Library: Stable-Baselines3 for v1, note RLlib as bsk_rl's own path

Two real options, both legitimate:

- **Stable-Baselines3 (SB3)** — v1 choice. Directly consumes standard
  Gymnasium `Env`, which is exactly what `bsk_rl.gym.SatelliteTasking`
  provides for a single satellite (per `02-bsk_rl-architecture.md` §1).
  Simple, well-documented, minimal infrastructure overhead — appropriate
  for a single-agent v1 problem and better suited to the project's stated
  learning goal (understand the RL loop clearly, not fight distributed-
  training infrastructure).
- **RLlib** — `bsk_rl`'s own docs ship PPO training recipes for RLlib
  specifically (per `02` §7), including multi-agent recipes matching
  `ConstellationTasking`'s PettingZoo `ParallelEnv` interface (per `02`
  §6). **This becomes the natural choice if/when the v2 multi-satellite
  constellation-scheduling stretch goal (`01-problem-scope.md`) is
  pursued** — noted here so the v1→v2 transition isn't a surprise
  framework migration, but deferred until v1 is validated.

## Handling the noisy/partial observation (`06-state-space.md`)

No special algorithmic handling needed — PPO with a standard MLP policy
over the flat observation vector (which already includes the covariance
summary and Pc as explicit features, per `06`) treats the noise as part of
the environment's observation function; the agent doesn't need an
explicit belief-state/filtering module because the covariance is *given*
to it directly rather than something it must infer. (If a future version
removed the explicit covariance feature and forced the agent to infer
uncertainty from repeated noisy observations, a recurrent policy — LSTM/
GRU — or an explicit Bayesian filter would become necessary. Not needed
for v1, flagged for completeness.)

## Hyperparameters — Phase 6 actuals, and a corrected prediction

v1 started from SB3's PPO defaults (`n_steps=2048`, `batch_size=64`,
`gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `learning_rate=3e-4`).
Phase 6 (`21-training-results.md`) ran real training and settled on:
`n_steps=64`, `batch_size=32`, `gamma=0.95`, `gae_lambda=0.9`,
`ent_coef=0.01` (SB3 default is 0.0), `clip_range`/`learning_rate`
unchanged.

- `gamma=0.95`: as predicted below — short episodes (4-6 decision steps
  for the fast curriculum stages actually used for training, see
  `21`) don't need 0.99's long-horizon discounting.
- `ent_coef=0.01`: as predicted below — added explicit exploration
  pressure so the policy doesn't collapse to always-near-zero-action.
- **`n_steps=64`, a correction, not a confirmation**: this doc originally
  predicted `n_steps` would need tuning *upward* from 2048 (more steps
  per update, for a stable batch of complete short episodes). Real
  measured environment throughput (`17`/`20`: ~8-9 steps/sec for the fast
  curriculum stages, only ~1.3 steps/sec for stage 3) made that
  direction infeasible in practice: at 2048 steps/update, even the fast
  stages need ~17 minutes of wall-clock time *before a single gradient
  update*, so a realistic training budget would only afford a couple of
  updates total — nowhere near enough for real learning. Going *smaller*
  (64) instead trades noisier per-update gradient estimates for enough
  actual updates (~78 for a 5,000-timestep run) to show real training
  signal within a feasible wall-clock budget. The original prediction
  wasn't wrong about the underlying tension (short episodes vs. batch
  stability) — it just didn't yet have real throughput numbers to weigh
  that against the compute-budget constraint, which turned out to dominate.

## Compute budget

No GPU required (per `02-bsk_rl-architecture.md` §8, Basilisk is CPU-only;
SB3's PPO for a small MLP policy on a low-dimensional observation is also
comfortably CPU-trainable). Main compute cost is **environment step
throughput** — each step runs a real Basilisk propagation, which is slower
than a toy Gym environment's `step()`. Parallelizing environment rollouts
(SB3's `SubprocVecEnv` across multiple CPU cores) is the relevant scaling
lever, not GPU acquisition.

**Real measured throughput** (Phase 6, single process, no parallelization):
~8-9 steps/sec on curriculum stages 1/2 with a short fixed schedule,
~1.3 steps/sec on stage 3 (real per-event schedules, up to 20+ decision
points per episode). This is why Phase 6's actual training run used stage
2 with a short schedule, not the "full v1" stage 3 environment — stage 3
at this single-process throughput would need many hours for even a modest
timestep budget. `SubprocVecEnv` parallelization was not attempted in
Phase 6 (flagged as the natural next lever if more compute becomes
available or stage-3 training is pursued later) — see `21` for what was
actually run and why.
