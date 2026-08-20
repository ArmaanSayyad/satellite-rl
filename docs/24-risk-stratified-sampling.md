# 24 — Risk-Stratified Sampling (Phase 7c)

Follow-up to `23-anisotropic-covariance-fix.md`'s finding: even with the
correct anisotropic covariance, actionable-risk (`Pc > 1e-4`) real events
are ~1-in-8,672 — too rare for uniform bootstrap sampling to expose
training to one in any practical budget. This implements the recommended
mechanism (deliberate, explicit oversampling of the real risk tail) and
reports, honestly, how far it actually goes.

## What changed

**`scenario/distributions.py`**: `_add_encounter_geometry` now also
computes `native_pc` per row — Pc from that row's own real
`(x0, z0, sigma_x, sigma_z, combined_radius)`, i.e. what that specific
event's risk actually is standalone. Stored in `geometry_events.csv` so
downstream code doesn't recompute it per draw. `data/fitted/geometry_
events.csv` regenerated (8,672 events, 0 dropped, one new column).

**`env/scenario_sampling.py`**: `SecondaryScenarioSampler` takes two new
constructor args, both opt-in (default preserves the exact pre-Phase-7c
behavior):
- `high_risk_fraction` (default `0.0`): probability, per episode, of
  drawing the geometry row from an elevated-risk pool instead of
  uniformly from the full real table.
- `high_risk_pool_fraction` (default `0.05`): the pool's size, as a
  fraction of the table, selected by rank (top-N by `native_pc`, not a
  value threshold — most rows sit at `native_pc == 0`, so a threshold
  would produce ties/an ill-defined cutoff; rank-based selection avoids
  that).

Both thread through `CollisionAvoidanceEnv.__init__` unchanged (same
names, same defaults) to `env_kwargs`, so `training/train_ppo.py`'s
`make_env`/`train` already accept them via their existing `**env_kwargs`
passthrough — no changes needed there.

5 new tests (`tests/test_scenario_sampling.py`, synthetic geometry table
with exactly-known pool membership) verify: `high_risk_fraction=0` never
draws from the pool, `=1` always does, `=0.5` draws from both, the
default is off/backward-compatible, and out-of-range values are
rejected. `SecondaryScenarioSampler` itself doesn't need bsk_rl/Basilisk,
but importing it still runs `satellite_rl.env`'s package `__init__.py`
(which does), so -- same as `test_env.py` -- these skip in the
lightweight CI environment rather than running there.

## Verified: the mechanism works exactly as designed

Real-data pool composition (`data/fitted/geometry_events.csv`, 8,672
rows) at the default `high_risk_pool_fraction=0.05`: **433 rows**,
`native_pc` ranging `1.4e-8` to `1.6e-4` — correctly includes the single
real event that exceeds `pc_threshold=1e-4`.

A 300-episode held-out sweep (`high_risk_fraction=0.5`,
`targeting_seed=999`) confirms the mechanism is live end-to-end (not just
correct in isolation): drawing from the pool measurably shifts the
observed `pc_final` distribution versus the unstratified baseline
(`23`'s 0%/5.3%/max-2.2e-7 figures) in the expected direction, though at
this sample size the shift is within Monte Carlo noise for a rare-tail
statistic and shouldn't be over-read numerically (see below for why a
bigger empirical sweep wasn't the right way to pin this down further).

## Honest result: this does NOT make `Pc > 1e-4` episodes reliably
## appear yet, and that's a closed-form, not an empirical, finding

The pool has exactly **one** row above `pc_threshold=1e-4` (there is only
one in the whole real dataset — `23`'s finding). Oversampling toward a
pool that itself is 99.8% below-threshold doesn't manufacture more
above-threshold examples; it just increases how often you draw from a
*slightly* elevated set. The actual expected count of hitting that one
specific event in `n` episodes, at `high_risk_fraction=f` and pool size
`k`, is exactly `n * f / k` (a Bernoulli draw, no simulation needed to
know this) — not worth chasing with more expensive Monte Carlo runs, the
arithmetic is exact:

| `high_risk_pool_fraction` | pool size | `n_actionable` in pool | expected hits per 1,000 episodes at `f=0.5` |
|---|---|---|---|
| 0.05 (default) | 433 | 1 | 1.15 |
| 0.01 | 86 | 1 | 5.81 |
| 0.001 | 8 | 1 | 62.5 |

This is the real, explicit tradeoff the two parameters control, not a
free lunch: shrinking the pool raises exposure to the one truly-
actionable event roughly linearly, at the direct cost of training-set
diversity within the "elevated" pool — at `pool_fraction=0.001` (8 rows),
a policy trained with heavy oversampling would see the *same 8 real
geometries* repeatedly, risking memorizing those 8 specific instances
rather than learning a generalizable notion of "Pc is high → act." At
the default `0.05` (433 rows), diversity is much better preserved, but
`n * f / k` says a training run would need on the order of several
thousand episodes at `f=0.5` before expecting even a couple of sightings
of the one genuinely-actionable event.

## What this means going forward

Two honest options, not decided here:

1. **Accept the real scarcity and use a much longer training run** (many
   thousands of episodes, `SubprocVecEnv` parallelization per `10-rl-
   algorithm.md`'s compute-budget section) with a small, aggressive pool
   (e.g. `pool_fraction=0.001-0.01`) — accepting the memorization risk as
   a known, explicit limitation, and reporting policy behavior on the
   held-out set with that caveat stated plainly.
2. **Lower the operational `pc_threshold`** the reward function gates on
   (e.g. to `1e-6`, which 22 real events exceed, or `1e-8`, which ~511
   do) — trading fidelity to the "1e-4 is a real operational threshold"
   framing (`08-reward-function.md`) for a meaningfully larger, more
   diverse set of real events the policy can actually learn to
   distinguish from the benign majority, at a more conservative
   (arguably still defensible — real CARA/ESA screening thresholds vary)
   risk cutoff.

Neither is implemented here. This phase's job was the sampling
mechanism itself, built and verified correct; which operating point to
use it at is a training-run decision, not a scenario-generation one.
