# 26 — Precise Targeting (Phase 7e)

Follow-up to `25-augmentation-and-threshold-findings.md`: even with the
elevated-risk pool correctly identifying 7 real actionable events and
augmentation generating real variety around them, the actual simulated
outcome for these small-miss-distance (tens-of-meters) scenarios didn't
match what the risk math predicted. This phase root-causes and fixes
that gap — which turned out to be five separate bugs, not one, found
across an initial investigation and three parallel follow-up threads.

## The original problem

The J2-only targeting solver (`scenario/targeting.py`) that places the
secondary satellite is only accurate to ~100-200m at the short (~5hr)
lead times curriculum stage 2 uses, regardless of target size —
invisible at the km-scale miss distances every earlier phase validated
against, but larger than the tens-of-meters miss distances real
high-risk events need. Requesting "38m apart" produced an actual
simulated encounter more like 100-200m apart, unrecognizable as the
intended near-miss.

## The fix that seemed to work in isolation: `correct_targeting_geometry`

A Newton/fixed-point correction loop, using Basilisk itself (already a
project dependency, and literally the ground-truth model being matched)
rather than a new external library — researched Orekit, NASA's GMAT, and
TU Delft's Tudat first; all three would still leave *some* residual
mismatch with Basilisk since they're independent physics models, and
GMAT specifically is architecturally unsuited to being called
programmatically thousands of times. The loop: propagate the current
candidate state under real Basilisk dynamics, measure the achieved miss
vector, request a correction equal to the target minus the observed
bias, repeat. Initial validation (15 test cases, 2-642m miss distances,
0.2-3 day lead times) converged to sub-meter accuracy in 1-8 iterations.

Along the way, two more bugs surfaced and were fixed before the
remaining gap was even visible:

- **Missing Sun gravity**: the correction's Basilisk probe
  (`_fly_passive_pair`) only included Earth; bsk_rl's real `WorldModel`
  also includes the Sun as a third-body perturbation. Fixed to match
  exactly (gravity bodies, ephemeris path, `zeroBase` casing).
- **Unrealistic-orbit validation gap**: rejecting only hyperbolic
  orbits (eccentricity ≥ 1) wasn't enough — a real event (relative
  speed 14,919 m/s, well above typical LEO orbital speed) produced a
  *bound* orbit (e=0.982) with a perfectly fine periapsis but an
  apoapsis near GEO altitude, just as numerically fragile as a
  hyperbolic orbit for the same reason. Added
  `osculating_apoapsis_altitude_m` + a 2,000km ceiling (the common LEO
  definition), checked only once the orbit is confirmed bound.
- **Severe performance bug**: the retry loop's real per-attempt cost was
  measured at ~22.5s for a 3-day lead time — not the ~10ms originally
  (wrongly) assumed, an unverified guess rather than a measurement. Fixed
  by restructuring the loop: the ego's forward propagation doesn't
  depend on the sampled direction, so it's computed once instead of once
  per attempt; periapsis/eccentricity/apoapsis are checked against the
  already-computed (free) TCA state instead of the expensive backward-
  propagated t0 state, so the genuinely expensive step only runs for
  candidates that already pass. Result: 20-60x faster, `max_attempts`
  raised from 50 to 200,000 with real headroom (a full 20-case sweep
  across the entire realistic parameter range that previously failed
  even at 3,000 attempts on 2 cases now completes in 13.3s total).

All of this was real, verified progress — 61 tests passing — but an
end-to-end check (comparing the sampler's own `native_pc` estimate
against the actually-simulated `pc_final`) still showed most cases
wildly diverging, sometimes by 50+ orders of magnitude, for reasons that
weren't yet understood.

## Three threads run in parallel to decide the next move

Rather than keep debugging alone, three investigations ran in parallel,
matching the actual decision the project faced (keep fixing vs. work
around it vs. reconsider the target):

**Thread A — root-cause debugging.** See below; this is what actually
resolved the problem.

**Thread B — quantify a miss-distance floor as a fallback.** Result:
**not a viable fix even setting aside effort.** A 150-episode sweep found
the divergence doesn't cleanly track miss distance — cases at
200-5000m (assumed "safe") still showed catastrophic mismatches (e.g. a
999m case: predicted `1.0e-8`, actual `5.8e-126`) while some smaller
cases agreed fine. Separately, the 7 known-actionable real events are
concentrated exactly where any safe floor would exclude them (miss
distances 38-1102m) — a floor tight enough to be reliable would have
sacrificed most of the value it exists to protect, on top of not
obviously working.

**Thread C — reconsider the threshold.** Result: **no tension, and no
change warranted** (detailed in `25`'s Finding 4) — real risk isn't
confined to small miss distances in this dataset, so this couldn't have
been a real alternative to fixing the underlying bug regardless.

With B ruled out as unreliable and C as unnecessary, Thread A's
root-cause fix became the only path that could preserve the full value
of the elevated-risk pool.

## Thread A: three more bugs, the real root cause

An end-to-end check comparing `env.satellites[0].dynamics.r_BN_N` (the
ego's real simulated position) against `_fly_passive_pair`'s prediction
for the exact same fixed, deterministic initial state — no secondary,
no targeting, no correction loop involved — found a stable ~2.5km
divergence, confirmed not a numerical artifact (identical to the
millimeter at 20x finer integration resolution). Three genuinely
separate bugs were behind it and the much larger secondary-position
error that followed:

**Bug 1 — wrong epoch reaching the physics engine.**
`CollisionAvoidanceEnv` never passed `world_args` to
`GeneralSatelliteTasking.__init__`. bsk_rl's `reset()` reads `utc_init`
exclusively from `world_args` — any `utc_init` sitting in per-satellite
`sat_args` (where this project had always set it) is silently ignored.
The real environment had been running on bsk_rl's own default epoch
(2000/06/23) instead of the intended 2018-09-29, for the entire project.
Confirmed directly: `env.simulator.world.gravFactory.spiceObject.
UTCCalInit` read back the wrong date, and the resulting Sun-position
vectors were 94.19° apart between the two epochs. Both satellites
experienced the same wrong epoch, so relative geometry was internally
self-consistent (this never broke anything on its own) — but it meant
any *external* probe using the correct epoch, like the new correction
loop, was comparing against a different Sun position than the real
simulation actually used. This alone accounted for the full 2540m
ego-only divergence (2540.153m → 2.586e-08m after the fix). Fixed: pass
`world_args={"utc_init": UTC_INIT}` explicitly.

**Bug 2 — a 2D blind spot in the correction's own error metric.**
`correct_targeting_geometry` projected the achieved 3D separation onto
the encounter-plane basis implied by the *target* relative velocity,
computed once up front. The achieved velocity at the real TCA rotates
from that target velocity over the propagation (J2 + solar
perturbation) — a concrete case showed 3153.8m of true 3D separation
error collapsing to just 0.31m of apparent 2D error, because nearly all
of the true error happened to lie along the direction the stale,
frozen basis couldn't see. The correction loop could converge cleanly
by its own measure while the real 3D state remained badly wrong. Fixed
by rewriting the whole loop to compare and correct raw 3D vectors — no
projection, no blind spot. (`_fly_passive_pair` now also returns
velocities, needed for this.)

**Bug 3 — the deep one: `deepcopy` silently orphaning the scenario
sampler.** `GeneralSatelliteTasking.__init__` does `self.satellites =
deepcopy(satellites)`. The secondary satellite's `sat_args["rN"]`/
`["vN"]` are bound methods on a `SecondaryScenarioSampler` instance;
`deepcopy` recursively clones the object those methods are bound to,
producing an independent, permanently-orphaned copy that the satellite's
actual `rN()`/`vN()` calls use from that point on — completely
disconnected from the `SecondaryScenarioSampler` instance `env.reset()`
mutates (`.generation`, `.rng`). That orphaned clone's `generation`
never advances (nothing external ever touches it again), so its own
caching logic short-circuits on every subsequent call: **the secondary
satellite's actual simulated position had been frozen to whatever the
very first sample happened to be, for the entire lifetime of any given
environment instance, since curriculum stage 2 was built (Phase 5c) —
for the entire project, until this was found.**

Confirmed via direct `id()`/generation inspection before implementing
anything (not assumed from the theory alone): the sampler instance
actually driving the satellite's `rN`/`vN` had a different `id()` than
`env._sampler`; across 3 resets with different seeds, `env._sampler.
generation` correctly advanced 1→2→3, while the orphaned clone's
generation stayed frozen at 0, returning byte-identical output every
time regardless of seed. This is exactly why the bug went undetected for
so long: `sigma_x`/`sigma_z`/`combined_radius`/schedule are read
directly off `env._sampler` (not through the satellite's `sat_args`), so
those *did* correctly vary every episode — episodes looked diverse
(varying risk metadata, varying reward-relevant reads) while the actual
simulated geometry was silently constant underneath. Fixed:
`SecondaryScenarioSampler.__deepcopy__` returns `self` (with
`memo[id(self)] = self`) — the standard idiom for an object meant to be
shared, live, singleton-like state rather than a value object safe to
clone.

Are Bugs 1-2 and Bug 3 the same root cause wearing different faces? No —
genuinely separate. 1-2 govern whether the ego's own propagation and the
correction's internal convergence check are trustworthy at all. Bug 3
governs whether the corrected state, even once trustworthy, ever reaches
the satellite that actually gets simulated. Fixing 1-2 alone still left
`pc_final` off from `native_pc` by many orders of magnitude, because the
correctly-computed correction was never reaching the real satellite in
the first place.

## Validation

The single real event that exceeds `pc_threshold=1e-4` (event 8767,
38m miss distance) now matches expected risk almost exactly:
`native_pc=1.6158e-4`, actual simulated `pc_final=1.6201e-4` — ratio
**1.0027**. A second real event (47m): ratio **1.0690**. A broader
20-episode sweep across varied real scenarios: ratio range 0.9826–1.0847,
median 1.0002 — every single episode within 2x of the sampler's own
estimate, most within a few percent.

All 127 tests in the full suite pass (61 targeting/tca_refinement tests
plus the rest of the project). Two pre-existing `test_tca_refinement.py`
failures surfaced by Bug 2's fix were diagnosed and fixed, not papered
over: one was a stale call site still unpacking `_fly_passive_pair`'s
old 3-tuple return; the other was a test tolerance that had only been
passing because the buggy 2D metric was hiding most of the true error —
under the honest 3D metric the same case converges geometrically (never
diverging, ~1.69x error reduction per iteration) but needs up to 19
iterations for a 3-day lead time rather than 8. `max_iters` raised to 24
with margin; the actual production regime (0.2-day lead time) still
converges in 1-3 iterations regardless of the cap.

## What this means

The elevated-risk pool (`25`) and the precise-targeting correction
(this doc) now compose correctly: real high-risk events, augmented for
diversity, ranked by the more authoritative of two risk estimates, get
simulated with the miss distance they were actually meant to have. The
scenario generator can now produce genuinely high-risk (`Pc > 1e-4`)
training episodes from real data — the capability the last several
phases were building toward.

## Correction: `max_attempts=200,000` (above) was revised down to 20,000

Found on the first real training attempt after this phase, not caught
by any test: a real event with `relative_speed_ms=15,850` (nearly 2x
circular LEO speed) crashed training when `solve_secondary_initial_
state_robust` exhausted its full attempt budget and raised. This isn't
a retry-budget problem — confirmed analytically, not just empirically:
even the most favorable possible relative-velocity direction for that
event implies an apoapsis around 3,344km, still past the 2,000km
ceiling, so **no** direction can satisfy every orbit-realism check at
once. About 3% of the real event table (262/8,672, by the same
analytic check) is similarly infeasible under our LEO-only conjunction
model — real data outside what this project represents, not a bug.

Two fixes: `SecondaryScenarioSampler` now catches this specific
`RuntimeError` and resamples a different row (bounded at 5 attempts)
instead of letting it crash the caller — matching the same "if this
specific draw doesn't work, try another" pattern the retry loop itself
already uses one level down. And `max_attempts` was lowered from
200,000 to 20,000: exhausting the full 200,000 on a genuinely
infeasible row measured ~29s, while the hardest *confirmed-feasible*
case succeeds comfortably within 3,000 — 20,000 keeps a real ~7x margin
over that while failing ~10x faster (~3s) when a row truly can't be
placed, so a resample doesn't cost 29 seconds every time it's needed.
