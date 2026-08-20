# 20 — Curriculum Stage 3: CDM-Sequence Uncertainty Evolution (Phase 5d)

Implements `03-scenario-design.md`'s curriculum stage 3 — the actual v1
target environment: real, irregular per-event decision schedules (not
Phase 5c's fixed 4-6-point schedule) and covariance that genuinely shrinks
across the episode (not stage 2's constant sigma). This is what makes the
wait-vs-act tradeoff real rather than a proxy for it.

## What changed

`CollisionAvoidanceEnv(sample_geometry=True, evolve_uncertainty=True, ...)`:
every reset samples, together as one consistent draw, a real event's
geometry (stage 2, unchanged), a real event's CDM-timing schedule
(bootstrapped from the 5,000-event `schedule_library.json` built in
Phase 2), and a real event's (first-CDM, last-CDM) covariance pair
(bootstrapped from a **new** `covariance_evolution_events.csv`, extracted
in this phase — Phase 2 had only saved the fitted, KS-rejected lognormal
for the shrink *ratio*, not the raw per-event pairs needed for genuine
bootstrap sampling, following the same pattern `15-distribution-fitting-
results.md` established for geometry).

Sigma is interpolated **geometrically** (log-linear) between the sampled
event's real first/last combined-sigma magnitude, based on the fraction
of elapsed episode time — not linearly, since real covariance shrinks
multiplicatively (median ~8.36x per Phase 2), not additively. Updated
every `step()`, not just at `reset()`, mirroring the existing
`_time_to_tca_s` mutable-state pattern from Phase 4/5c.

## Two real bugs found and fixed before this shipped

### 1. Negative real timestamps broke schedule ordering

Some real Kelvins CDMs are reported *after* TCA (`time_to_tca` as low as
-0.15 days, confirmed in `14-pc-validation-results.md`). The first version
of `_clean_schedule()` didn't filter these before appending a `0.0`
endpoint, producing schedules like `[0.381, -0.016, 0.0]` — not properly
descending, since `-0.016 < 0.0`. Caught by directly inspecting a live
episode's `schedule_s`, not just trusting env_checker (which doesn't check
this specific invariant). Fixed by dropping negative entries before
processing, with a fallback (`[86400.0, 0.0]`, one default 1-day step) for
the rare case where an event's entire schedule was at/after TCA.

### 2. `env_checker`'s determinism check went from a warning to a hard failure — and that was the right prompt to actually fix it

Phase 5c's `env_checker` run only produced a soft warning about
`reset(seed=X)` not being reproducible, which got documented as an
accepted, deliberate trade-off (curriculum sampling needs variety, so
exact reproducibility seemed like it had to be sacrificed). Stage 3's much
larger schedule-length variation turned that same underlying issue into a
hard `AssertionError` from `check_step_determinism` (which explicitly
calls `reset(seed=X)` twice and asserts exact equality).

That failure was the right prompt to reconsider the original trade-off,
not route around it: **the two needs were never actually in tension.**
`CollisionAvoidanceEnv.reset()` now reseeds the sampler's RNG whenever an
explicit `seed` is passed (making `reset(seed=X)` genuinely reproducible,
as Gym expects), while `reset()`/`reset(seed=None)` — what a real training
loop actually calls every episode — still lets the RNG advance, giving
fresh real events each episode exactly as before. Both stage 2 and stage
3 now pass `env_checker` with no warnings at all, not just no errors.
`19-curriculum-stage-2.md` updated to correct its earlier "deliberate
deviation" framing rather than leave it standing now that it's actually
resolved.

## What's deliberately still simplified

Sigma remains **isotropic** even though real per-event sigma_x/sigma_z
are anisotropic and now available at both first and last CDM — the
isotropic simplification from `06-state-space.md`/`19` continues here
(full anisotropic evolution would need a well-defined, consistently
time-varying encounter-plane frame to project onto, a separate piece of
work not attempted in this pass). The interpolation model is a simple
geometric blend based on elapsed-time fraction, not a fit to how real
per-event covariance actually evolves *between* first and last CDM
(likely nonlinear/non-monotonic in reality per `03-scenario-design.md`'s
own note) — a reasonable v1 simplification, not claimed to be more
realistic than it is.

## Validated

16/16 `test_env.py` tests pass (6 new for stage 3, all previous stage
1/2 tests still passing unmodified). Confirmed directly, not just via
`env_checker`: schedules are always properly descending, non-negative,
and end at exactly 0.0, across repeated resets with genuinely different
real events (schedule lengths observed ranging from 2 to 22+ points,
consistent with Phase 2's 1-23 CDM/event range); sigma's median shrink
ratio across 10 sampled episodes is meaningfully greater than 1.0 (real
per-event variance means this isn't asserted as a strict per-episode
invariant, matching the same honesty principle used throughout this
project for other statistically-noisy real-data properties). `env_checker`
passes with zero warnings in both stage 2 and stage 3 modes.

**Real cost worth flagging**: the stage-3 test suite took ~11 minutes
locally (real multi-day schedules with up to 20+ decision points mean
many more Basilisk propagation steps per test episode than stages 1/2's
short fixed schedules). Not a CI concern (these tests skip entirely in
CI, which lacks the Basilisk stack), but a real consideration for Phase 6
training throughput — a training run using `evolve_uncertainty=True`
will be substantially slower per episode than stages 1/2, something
`10-rl-algorithm.md`'s compute-budget planning should account for
directly rather than extrapolate naively from the ~8.5 steps/sec figure
measured in Phase 4 on short fixed schedules.
