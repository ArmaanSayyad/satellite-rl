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

## Hyperparameters — starting point, not final

v1 starts from SB3's PPO defaults (well-tuned for continuous-control tasks
generally: `n_steps=2048`, `batch_size=64`, `gamma=0.99`, `gae_lambda=0.95`,
`clip_range=0.2`, `learning_rate=3e-4`) and adjusts based on observed
training behavior rather than pre-guessing task-specific values:
- `gamma`: given short episodes (6–12 decision steps per `09-episode-
  design.md`), the effective horizon is short — 0.99 may be higher than
  needed; worth testing 0.95–0.99 range once training is running.
- `n_steps`/rollout length: needs enough episodes per update given the
  short-episode structure; likely needs tuning upward from the default to
  get a stable enough batch of complete episodes per PPO update.
- Entropy coefficient: given the near-zero-action "wait" behavior
  discussed in `07-action-space.md`, some exploration pressure is needed
  early in training to discover that *acting* is sometimes necessary
  (a policy that collapses to always-zero-action early would look
  falsely converged) — worth explicit attention, not just left at default,
  during Phase 6 (`13-roadmap.md`).

These are documented here as the starting hypothesis; actual tuned values
get recorded in a training-log doc once Phase 6 runs, not retrofitted into
this design doc.

## Compute budget

No GPU required (per `02-bsk_rl-architecture.md` §8, Basilisk is CPU-only;
SB3's PPO for a small MLP policy on a low-dimensional observation is also
comfortably CPU-trainable). Main compute cost is **environment step
throughput** — each step runs a real Basilisk propagation, which is slower
than a toy Gym environment's `step()`. Parallelizing environment rollouts
(SB3's `SubprocVecEnv` across multiple CPU cores) is the relevant scaling
lever, not GPU acquisition — worth benchmarking single-env step time early
(Phase 4, `13-roadmap.md`) to estimate realistic wall-clock training time
before committing to a specific total-timestep budget.
