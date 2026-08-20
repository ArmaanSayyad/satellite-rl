# 21 — Phase 6 Training Results

## What was actually run

Stage 2 curriculum environment (`sample_geometry=True, evolve_uncertainty=
False`), short fixed schedule `(0.2, 0.1, 0.05, 0.01, 0.0)` days-before-TCA
(4 decision steps/episode), `targeting_seed=0`. Not stage 3: `20-curriculum-
stage-3.md` measured stage 3 at ~1.3 steps/sec (real per-event schedules,
20+ decision points), which at single-process throughput would need many
hours for even a modest timestep budget (`10-rl-algorithm.md`'s compute-
budget section). Stage 2 was the only stage a single training run could
cover in a reasonable wall-clock time this phase.

PPO hyperparameters, chosen for short-episode/small-budget training per
`10`'s corrected reasoning: `n_steps=64`, `batch_size=32`, `gamma=0.95`,
`gae_lambda=0.9`, `ent_coef=0.01`, `clip_range=0.2`, `learning_rate=3e-4`.

`total_timesteps=5000`, `seed=0`. Wall clock: ~53 minutes (measured
throughput ~1.6 steps/sec effective, including PPO update overhead — lower
than stage 2's raw ~8-9 steps/sec env-only benchmark from `17`/`20`, since
that benchmark didn't include PPO's own update cost). 1264 episodes
completed (5-step episodes: 4 real decisions + terminal bookkeeping),
780 gradient updates.

## Training curve

From `runs/ppo_stage2_run1_monitor.csv` (SB3 `Monitor` wrapper, per-episode
reward):

- First 10 episodes mean reward: **-0.2596**
- Last 10 episodes mean reward: **-0.2435**
- SB3's own rolling log: `ep_rew_mean` went from ~-0.262 (early) to -0.242
  (final logged window), reward std from ~1.0 down to 0.655.

This is a real, if modest, upward trend — visible in
`runs/ppo_stage2_run1_monitor.png` as a noisy but genuinely rising 20-
episode rolling mean. It is not a strong or fully-converged result: 5,000
timesteps / 780 updates is a small budget (chosen for wall-clock
feasibility on a single process, not because it was expected to be
sufficient for convergence), and the shrinking std is at least partly
consistent with an entropy-collapse-toward-fixed-action failure mode
(see below), not purely "the policy got better at discriminating states."

## Held-out evaluation vs. baselines

`compare_policy_to_baselines()`, `targeting_seed=999` (different from
training's `targeting_seed=0` — real Kelvins events not seen during
training). Run at both `n_episodes=20` and `n_episodes=50` to check the
result wasn't a small-sample fluke; results agree closely:

**n=20:**

| policy | reward | fuel (m/s) | maneuvers | pc_final |
|---|---|---|---|---|
| trained_policy | -0.2039 ± 0.0000 | 0.388 | 4.00 | 1.34e-10 |
| never_maneuver | -0.0000 ± 0.0000 | 0.000 | 0.00 | 5.01e-10 |
| always_max_thrust | -0.6000 ± 0.0000 | 40.000 | 4.00 | 0.00e+00 |
| random_policy | -0.5474 ± 0.0364 | 34.742 | 4.00 | 2.71e-107 |

**n=50 (confirms n=20, not a fluke):**

| policy | reward | fuel (m/s) | maneuvers | pc_final |
|---|---|---|---|---|
| trained_policy | -0.2039 ± 0.0000 | 0.388 | 4.00 | 1.02e-10 |
| never_maneuver | -0.0000 ± 0.0000 | 0.000 | 0.00 | 4.00e-10 |
| always_max_thrust | -0.6000 ± 0.0000 | 40.000 | 4.00 | 1.33e-17 |
| random_policy | -0.5654 ± 0.0258 | 36.542 | 4.00 | 1.33e-17 |

## Honest interpretation

Two findings worth stating plainly, neither of which is the "training
worked, ship it" story:

**1. The trained policy is worse than doing nothing, on this held-out
set.** `never_maneuver` gets reward ≈ 0 (no fuel cost, and final Pc is
already far below `pc_threshold=1e-4` without any intervention).
`trained_policy` still burns fuel every episode (0.388 m/s total, spread
over all 4 steps — small, but nonzero on every single decision) for no
measurable risk-reduction benefit versus not acting at all. It clearly
beats `always_max_thrust` and `random_policy` (both far more expensive,
both terminate with Pc pinned to the simulator's numerical floor from
over-maneuvering), so training clearly did *something* — the policy learned
to use dramatically less fuel than undirected/adversarial baselines. It
has not yet learned the specific behavior that would beat
`never_maneuver` here: recognizing "predicted risk is negligible, the
reward-optimal action is exactly zero."

**2. Reward std = 0.0000 across 20-50 varied real events is a real
finding, not a display artifact.** It was checked at two sample sizes
specifically because an exactly-zero std looked suspicious. It holds at
both n=20 and n=50, and `pc_final` differs slightly between the two runs
(1.34e-10 vs 1.02e-10) confirming the underlying episodes are genuinely
different draws, not a caching bug. The remaining explanation is that
`model.predict(obs, deterministic=True)` is producing a nearly
state-independent action — i.e., PPO converged toward a fixed small
"nudge" action regardless of the input observation, rather than a
policy that discriminates between geometries. This is consistent with:
the shrinking-but-nonzero std in the training log (entropy dropping
toward a narrow, roughly stationary action distribution) and with 5,000
timesteps/780 updates being a small budget for a stochastic-gradient
method to learn fine state-dependent structure. It is *not* purely an
artifact of an easy held-out set: `never_maneuver`'s own reward std of
0.0000 is separately explained (below) and doesn't imply the trained
policy's action is state-insensitive — that's a distinct, additional
observation about `trained_policy` specifically.

**Why `never_maneuver`'s std is also ~0**: this held-out sample is
dominated by low-risk real events, consistent with Phase 1's dataset
finding (`14-pc-validation-results.md`) that a large majority of real
Kelvins CDMs sit at or near the risk floor. With `pc_final` far below
`pc_threshold` in essentially every sampled episode, the risk term in the
reward (`08-reward-function.md`'s deadzone-scaled penalty) is a small
near-constant contribution for a policy that never acts, so reward is
consistently ≈ 0 regardless of which specific event was drawn. This is a
real property of the evaluation *sample*, not a bug — but it also means
this quick check under-tests the higher-risk tail of the distribution,
where a good policy should differ most sharply from `never_maneuver`.
That's a genuine limitation of this phase's evaluation, not fixed here:
Phase 7 (`11-evaluation.md`) is where a properly stratified (by native
risk level) held-out evaluation belongs.

## What this does and doesn't show

Does show: the training loop, reward signal, and PPO integration are all
wired correctly and produce a real training effect — fuel usage collapsed
from random-baseline levels (~35 m/s) to near-minimal (0.39 m/s) within
5,000 timesteps, and the reward curve trends upward, not flat or
degenerate.

Does not show: a policy that is actually good at this task yet. On a
held-out sample dominated by low-native-risk events, it's currently worse
than the trivial baseline because it hasn't learned to gate its (already
small) action on predicted risk — it acts on essentially every step
regardless of whether risk is negligible. This is a legitimate outcome of
a small-budget training run, reported honestly rather than glossed over,
per the project's standing practice (`11`'s "honest negative results"
section; the corrections in `16`/`17`/`19`).

## Follow-up (not done in this phase, flagged for Phase 7+)

- Longer training run / more timesteps, to see if the state-independence
  in §2 resolves with more gradient updates rather than being a stable
  local optimum.
- A held-out evaluation set stratified by native (never-maneuver) Pc, so
  high-risk episodes — where a good policy should visibly separate from
  `never_maneuver` — aren't diluted by the dataset's real-world skew
  toward low-risk events.
- `SubprocVecEnv` parallelization (`10`'s compute-budget section) to make
  a larger-timestep run wall-clock feasible, and to eventually make stage
  3 training feasible at all.
