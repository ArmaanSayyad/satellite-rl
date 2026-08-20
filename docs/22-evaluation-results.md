# 22 — Phase 7 Evaluation Results

## What was run

`satellite_rl.training.full_evaluation.run_full_evaluation()` against the
Phase 6 checkpoint (`runs/ppo_stage2_run1`), on the stage-2 environment
(`schedule_days_before_tca=(0.2, 0.1, 0.05, 0.01, 0.0)`), held-out
`targeting_seed=999` (not used during training), `n_episodes=60`
(seeds 100-159, matched across all policies for a fair paired
comparison), `pc_threshold=1e-4`, `max_dv_ms=10.0`.

Five policies: `never_maneuver`, `always_max_thrust`, `threshold_heuristic`
(new this phase — fixed-magnitude, fixed-radial-direction burn gated on
observed Pc, `training/baselines.py`), `trained_policy`, `random_policy`.
Plus the hindsight oracle (new this phase — bisection search over burn
magnitude, radial direction, first decision step only; see
`baselines.py`'s docstring for exactly what it does and doesn't search
over).

## Headline finding: the held-out set has zero high-risk episodes

Stratifying by native risk (each scenario's `never_maneuver` final Pc) was
supposed to separate a "low-risk, correctly do nothing" subset from a
"high-risk, should act" subset (the Phase 6 follow-up item this was meant
to address). It didn't get the chance to: **all 60 held-out episodes have
native Pc below `pc_threshold=1e-4`** (`n_high_risk=0`, `n_low_risk=60`).
A broader sweep (never-maneuver only, n=300, same held-out
`targeting_seed=999`) confirms this isn't a small-sample fluke: max
observed Pc was `4.5e-9`, six orders of magnitude below threshold, and
65% of episodes had Pc exactly 0. **The same check against the *training*
seed (`targeting_seed=0`, n=150) gives the same result**: 0/150 exceed
`1e-4`, max `9.3e-9`.

Direct consequence: `threshold_heuristic` and the hindsight oracle are
**trivial** on this evaluation — the heuristic never crosses its trigger
condition (its metrics are bit-for-bit identical to `never_maneuver`:
reward, fuel, Pc all match), and the oracle's answer is "0 fuel needed"
for every one of the 60 scenarios (`_oracle.mean_fuel = 0.0`,
`n_infeasible_within_max_dv = 0`). Baselines 3 and 4 from `11-evaluation.
md` were implemented correctly (they behave exactly as designed — gate on
Pc, or minimize fuel for a target Pc), but this evaluation set never
exercises the interesting regime either one exists to handle.

## Root cause, investigated rather than assumed

Two candidate explanations, both checked directly against the real data
rather than guessed:

**1. The real Kelvins geometry is genuinely mostly-benign.** Recomputing
Pc directly from `data/fitted/geometry_events.csv` (8,672 real
final-CDM events, isotropic sigma = geometric mean of `sigma_x`/
`sigma_z`, same simplification the env uses) on a 500-row sample: 0%
exceed `1e-4`, max `8.5e-6`. Median `miss_distance` is **14,458 m**
against a median `combined_radius` of **~1.1 m** — most real conjunctions,
even at their final/most-confident CDM, simply aren't close. This matches
`14-pc-validation-results.md`'s finding that 68.9% of real events sit at
ESA's own reported risk floor — not a new finding, a consistent one.

**2. The env's isotropic-sigma simplification materially amplifies the
suppression, on top of that.** `docs/06-state-space.md` frames the
isotropic simplification ("uses sigma_m as an isotropic covariance...
looks the same under any encounter-plane projection") as roughly neutral.
Checked directly: the real projected covariance is **not** close to
isotropic. `sigma_z/sigma_x` (the two encounter-plane principal std
devs, both present in `geometry_events.csv`) has median **5.76x**, and up
to **4,248x**, eccentricity. Recomputing Pc for the same 500-row sample
using each row's *actual* anisotropic `(sigma_x, sigma_z)` pair with the
miss vector placed along the tighter axis (the most favorable-for-risk
alignment, an upper bound rather than the row's real alignment, which
`geometry_events.csv` doesn't retain — see below) shifts the exceedance
rate from **0% to 3.2%**. Small in absolute terms, but a >0 result where
the isotropic version gives exactly zero — i.e., real, non-floor risk
exists in this data at a rate the isotropic simplification doesn't
surface at all in a 500-row sample.

**Compounding factor**: `SecondaryScenarioSampler` also re-randomizes the
encounter's orientation angle (`orientation_angle_rad`, uniform 0-2π)
independently of the real event's own geometry — `_add_encounter_geometry`
(`scenario/distributions.py`) computes the real projected miss vector's
principal-frame position (`x0`, `z0`) via `project_to_encounter_plane`,
but `geometry_events.csv` only keeps `sigma_x`/`sigma_z`, not `x0`/`z0`
(see `pc/geometry.py`'s `EncounterGeometry2D`) — so even if the isotropic
simplification were fixed, the *specific* real alignment between a given
event's miss vector and its own covariance ellipse is already discarded
before the geometry table is written, and can't be recovered downstream.

**Conclusion**: this isn't a bug in the sense of "wrong code" — every
piece (`compute_pc`, `project_to_encounter_plane`, the sampler) does what
it's documented to do. It's a **scenario-generation fidelity gap**: the
combination of (a) dropping `x0`/`z0` when building `geometry_events.csv`
and (b) collapsing the two retained principal sigmas to one isotropic
value at sample time means curriculum stage 2 (and stage 3, which reuses
the same isotropic `sigma_at_fraction` mechanism) **cannot currently
generate a genuinely high-risk episode from real data**, independent of
how much of the real distribution's risk the underlying events actually
contain.

## Reframing the Phase 6 result

`21-training-results.md` attributed the trained policy's failure to gate
on Pc to a small training budget (5,000 timesteps). That's still true as
far as it goes, but this phase's finding gives a more direct explanation:
**the training run (`targeting_seed=0`) also never produced a high-risk
episode** (confirmed above, 0/150). A policy can't learn "act only when
risk is high, otherwise don't" from a training distribution that never
contains a high-risk example — there's no gradient signal pointing that
direction at all. The near state-independent "small nudge every step"
behavior found in Phase 6 is consistent with a policy that received
reward feedback almost entirely from the fuel/disruption terms (which
*do* vary meaningfully with actions) and essentially none from the risk
term (which was ~constant-near-zero regardless of action, for nearly
every training episode) — it optimized the part of the reward surface
that actually had signal.

## Metric suite (as designed, on the data actually available)

All 60 episodes, `n_high_risk=0` so no stratified breakdown is
informative here (the `pc_final_high_risk_subset` field is `null` for
every policy — reported as such, not silently omitted):

| policy | reward | fuel (m/s) | maneuvers | first maneuver step | pc_final mean / p99 | regret vs. oracle |
|---|---|---|---|---|---|---|
| never_maneuver | -0.0000 ± 0.0000 | 0.000 | 0.00 | never | 3.4e-10 / 4.3e-9 | 0.000 |
| threshold_heuristic | -0.0000 ± 0.0000 | 0.000 | 0.00 | never | 3.4e-10 / 4.3e-9 | 0.000 |
| trained_policy | -0.2039 ± 0.0000 | 0.388 | 4.00 | step 1 | 8.5e-11 / 1.5e-9 | +0.388 |
| always_max_thrust | -0.6000 ± 0.0000 | 40.000 | 4.00 | step 1 | 1.1e-17 / 2.7e-16 | +40.000 |
| random_policy | -0.5467 ± 0.0365 | 34.666 | 4.00 | step 1 | 9.3e-18 / 2.3e-16 | +34.666 |
| **hindsight oracle** | — | **0.000** | — | — | (target: ≤1e-4) | — |

(`threshold_heuristic` is identical to `never_maneuver` to full displayed
precision, as expected given §"Headline finding" above.)

Two things this table does show honestly, despite the trivial-oracle
caveat:

- **Every policy that acts at all (trained, always-max, random) acts on
  literally every episode, on the very first decision step**
  (`mean_first_maneuver_step = 1.0`, `n_acted_episodes = 60/60`) — none
  of them show any state-dependent timing behavior (the "does it wait for
  better information" question `11-evaluation.md` asks about). For
  `always_max_thrust`/`random_policy` this is fully expected (neither
  reads Pc). For `trained_policy`, this reconfirms Phase 6's finding
  (near-zero-variance, near-state-independent action) with a cleaner
  signal: since this evaluation set never contains a genuinely risky
  scenario, *always* acting the same way regardless of state is,
  descriptively, the same failure mode Phase 6 found, not a new one.
- **Regret vs. oracle is exactly the fuel used**, for every acting
  policy, since the oracle's answer is always 0 — i.e. on this
  particular held-out sample, *any* nonzero fuel spent is pure waste
  relative to the true optimum. This is a real, correctly-computed
  number, but it's only informative about "how much do non-risk-aware
  policies overspend on uniformly-benign data," not about how close a
  policy gets to optimal *when there's a real decision to make* — the
  question the oracle exists to answer, and which this evaluation set
  can't currently test.

## Real-data validation sanity check (per `11-evaluation.md`)

Of the three things `11`'s "Real-data validation" section asks a sanity
check to verify: "doesn't waste fuel on consistently-low-risk real
events" is satisfied trivially by `never_maneuver`/`threshold_heuristic`
but **not** by `trained_policy` (see above — it acts on all 60, all of
which are low-risk); "does act on real events flagged higher-risk" is
**untestable with the current sampler**, since no high-risk episode was
generated to check against; "degrades gracefully on out-of-distribution
statistics" wasn't specifically stressed-tested this phase (no crashes/
NaNs were observed across ~700+ episode runs total this phase, which is
weak evidence in favor, not a targeted check). Reporting this as a
partially-completed, not a passed, sanity check.

## What Phase 7 does and doesn't establish

Does establish: the baseline suite (threshold heuristic, hindsight
oracle) is implemented correctly and behaves exactly as designed; the
full metric/regret/timing methodology works end-to-end; and — the most
consequential finding — **the curriculum stage-2/3 scenario generator's
isotropic-covariance simplification, combined with discarding the real
per-event miss-vector/covariance alignment when building
`geometry_events.csv`, prevents it from ever generating a genuinely
high-risk episode from real data**, which likely explains Phase 6's
training result better than "budget was too small" alone.

Does not establish: whether the learned policy (or the threshold
heuristic) is actually good at the core task (act when risky, don't
otherwise) — that question remains untested, not because the evaluation
methodology is wrong, but because the environment it would be tested
against can't currently produce the scenarios that would test it.

## Recommended next step (not started this phase)

Fix the scenario generator before spending more compute on training or
evaluation: preserve each real event's actual encounter-plane miss offset
(`x0`, `z0`, already computed by `project_to_encounter_plane` and
already discarded — `pc/geometry.py`'s `EncounterGeometry2D` has them,
`scenario/distributions.py`'s `_add_encounter_geometry` just doesn't keep
them) and/or replace the isotropic `sigma_at_fraction` observation/reward
computation with the real anisotropic `(sigma_x, sigma_z)` pair. This is
a scenario-generation and observation-fidelity change, not a training-
budget or algorithm change — retraining on the current sampler, however
long, cannot produce a risk-gating policy if the training distribution
itself contains no risk to gate on.
