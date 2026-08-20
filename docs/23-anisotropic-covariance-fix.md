# 23 — Anisotropic Covariance Fix (Phase 7b)

Follow-up to `22-evaluation-results.md`'s finding: curriculum stage 2/3
never produced a genuinely high-risk episode from real data. This
implements the recommended fix (preserve each real event's actual
miss-vector/covariance alignment, use anisotropic covariance instead of
an isotropic geometric mean) and reports what it did and didn't change.

## What changed

**`scenario/distributions.py`**: `_add_encounter_geometry` now also
records `alignment_angle_rad = atan2(z0, x0)` -- the real miss vector's
angle within the event's own principal-axis (tight-x/loose-z) frame,
computed by `project_to_encounter_plane` but previously discarded before
being written to `geometry_events.csv`. `sample_scenario_geometry_
bootstrap` carries it through. `data/fitted/geometry_events.csv`
regenerated (8,672 events, 0 dropped -- same event set as before, one
new column).

**`env/scenario_sampling.py`**: `SecondaryScenarioSampler` now passes
this real angle as `orientation_angle_rad` to `solve_secondary_initial_
state_robust`, instead of `self.rng.uniform(0, 2*pi)`. This works because
`targeting.py`'s solver places the miss vector within `encounter_plane_
basis(v_rel)` -- the exact same basis convention `project_to_encounter_
plane` used to compute the angle in the first place (both in `pc/
geometry.py`). Reusing it reproduces a specific real event's actual
miss-vector-vs-covariance-ellipse relationship; it does not (and cannot)
reproduce the event's real absolute 3D orientation, which was never
physically meaningful to preserve here since the simulated relative-
velocity direction is independently sampled. `current_sigma`/`sigma_at_
fraction` (isotropic, geometric-mean) replaced with `current_sigma_xz`/
`sigma_xz_at_fraction` (anisotropic pair, each axis interpolated
independently in log space for stage 3's evolving case).

**`env/observations.py`**: `_compute_pc` now builds `cov_combined` as
`basis @ diag(sigma_x**2, sigma_z**2) @ basis.T`, with `basis =
encounter_plane_basis(v_rel)` computed from the CURRENT (predicted, not
stored) relative velocity at Pc-evaluation time -- matching real
operational practice (the encounter-plane basis is always derived from
the current best-estimate relative velocity, not fixed at initial
scenario setup). The embedded 3x3 matrix is rank-2 (zero variance along
`v_rel`); confirmed this is fine because `compute_pc`/`project_to_
encounter_plane` only ever use the 2D projection, never require the full
3x3 to be positive-definite.

**`env/collision_avoidance_env.py`**: `_pc_sigma` (scalar) split into
`_pc_sigma_x`/`_pc_sigma_z`. Fixed-scenario mode (curriculum stage 1)
sets both equal to `sigma_m` -- reduces to the isotropic case exactly
(`diag(s,s)` embedded via an orthonormal basis is `s*I`), so no separate
code path was needed for that mode; it was never claiming real-data
grounding in the first place.

11 tests updated/added (`test_distributions.py`, `test_env.py`); full
suite re-run, 102 passed.

## Empirical result: the fix is real, but not sufficient by itself

**Isotropic-vs-anisotropic ceiling, same 300-episode held-out sweep
(`targeting_seed=999`)**:

| | pre-fix (isotropic) | post-fix (anisotropic, real alignment) |
|---|---|---|
| max Pc observed | 4.5e-9 | 2.2e-7 (~50x higher) |
| fraction > 1e-8 | 0% | 5.3% |
| fraction > 1e-4 | 0% | 0% |

The fix does what it was designed to do -- it's no longer discarding real
risk-relevant information, and the ceiling of what the sampler can
produce moved by a real, measured ~50x. It does **not** make `pc_final >
1e-4` episodes appear in a 300-episode sample, or even a 60-episode one
(Phase 7's evaluation set).

**Ground truth, computed directly from the full real table** (all 8,672
events, no orbital simulation needed -- just `compute_pc` on each row's
own real `(miss_distance, sigma_x, sigma_z, alignment_angle_rad,
combined_radius)`, using the TRUE alignment rather than the earlier
worst-case-alignment upper-bound estimate from `22`): **1 of 8,672 real
final-CDM events (0.012%) has Pc > 1e-4.** Max Pc across the whole real
table is `1.6e-4`.

This is the real, honest answer to "why didn't the fix produce high-risk
training episodes": at a **1-in-8,672 real base rate**, uniform bootstrap
resampling (`SecondaryScenarioSampler` draws one real row uniformly,
with replacement, per episode) has an expected count of `n/8672`
high-risk draws in `n` episodes -- about 0.035 for the 300-episode sweep,
0.007 for Phase 7's 60-episode evaluation set. Seeing zero in either is
exactly what this real rate predicts, not evidence of a remaining bug.
The isotropic simplification was suppressing risk on top of this
already-low real rate (confirmed: it made the *ceiling* undetectable at
these sample sizes, 4.5e-9 vs. the true-alignment ceiling of 1.6e-4
system-wide), but the isotropic bug was never the dominant reason
high-risk episodes are rare -- real actionable-risk conjunctions
genuinely are rare in this dataset, consistent with `14-pc-validation-
results.md`'s 68.9%-floored finding, just far more extreme at the
`pc_threshold=1e-4` cutoff specifically than that floor statistic alone
would suggest.

## What this means for training/eval going forward

Uniform bootstrap sampling over real events, however anisotropically
correct, cannot make a training run see enough high-risk episodes to
learn risk-gating in any practical timestep budget -- at a 1-in-8,672
rate, even a 50,000-timestep run (orders of magnitude more than Phase 6's
5,000) would expect single digits of high-risk episodes at best,
depending on episode length. This is now a **known, quantified, correctly
attributed limitation**, not an open question: the scenario generator
needs deliberate **stratified/oversampled risk exposure** (e.g., draw
some fixed fraction of episodes from the real table's high-risk tail
specifically, rather than uniformly from all 8,672 rows) for training to
have a realistic chance of learning the risk-gating behavior at all. Not
implemented this phase -- flagged as the next concrete step, and a
larger design decision than this fix (how much to oversample, whether to
weight by risk magnitude, whether it distorts the "grounded in real
distribution" claim elsewhere in this project) worth deciding explicitly
rather than defaulting into.

## Two other design-vs-implementation gaps noticed while doing this fix

Not fixed here (out of this fix's scope), but worth recording rather than
staying silent, since `06-state-space.md`'s original design explicitly
called for both:

- **No observation noise is actually injected.** `06` specifies the
  observed relative state = true state + noise drawn from the current
  covariance, explicitly to make this "a genuinely partially-observable
  problem." The actual implementation (`observations.py`) passes
  Basilisk's true relative state straight through, unperturbed. This also
  means `11-evaluation.md`'s "N repeated episodes per scenario type, to
  distinguish policy quality from observation-noise variance" doesn't
  apply as written -- a fixed scenario replayed twice currently gives an
  identical outcome both times (there is no noise to average over).
- **σx/σz/orientation aren't separate observation features.** `06` lists
  them explicitly as observation components ("Combined covariance,
  principal-axis std devs (σx, σz) + orientation angle in encounter
  plane"). The actual observation vector (`satellites.py`'s
  `EgoSatellite.observation_spec`) only exposes the resulting scalar Pc,
  not the covariance shape/alignment that produced it -- this fix's
  anisotropic covariance is used internally for a more accurate Pc, but
  the agent still can't see *why* a given Pc is high or low (e.g.,
  distinguish "close miss, tight uncertainty" from "far miss, huge
  along-track smear" if they happened to produce similar Pc).

Neither gap is new -- both trace back to Phase 4/5 implementation
decisions that simplified `06`'s original design without `06` itself
being corrected to say so. Flagging now since this fix's own docstrings
reference the real σx/σz distinction directly, making the omission easy
to notice.
