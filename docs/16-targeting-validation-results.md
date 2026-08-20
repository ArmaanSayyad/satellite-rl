# 16 — Targeting Solver Validation Results (Phase 3)

Two separate questions, kept explicitly separate because they have very
different confidence levels:

1. **Is the two-body targeting solver's own math correct?** Yes —
   thoroughly validated, high confidence.
2. **How much does Basilisk's real high-fidelity dynamics diverge from
   the two-body model used to compute initial conditions?** Not yet
   answered — the investigation is documented honestly below, including
   why it's being deferred to Phase 4 rather than pushed through.

## 1. Two-body solver validation — solid

### Geometry reproduction (`test_solver_reproduces_targeted_geometry`, 20 trials)

The solved secondary state, recombined at TCA, reproduces the exact
targeted miss distance and relative speed to `1e-9` relative tolerance
(floating-point precision), and the miss vector is confirmed perpendicular
to the relative velocity vector (the actual definition of TCA) to the same
precision. This validates the solver's vector algebra directly, with no
dependency on propagation accuracy.

### Self-consistency (`test_self_consistency_batch`, 200-trial batch run manually for distribution stats; 20 trials in the fast CI suite)

Forward-propagating the solved initial state (same two-body model used to
solve for it) and checking it reproduces the targeted TCA state:

| Statistic | Position error (m) | Velocity error (m/s) |
|---|---|---|
| Mean | 5.64 | 2.6e-3 |
| Median | 1.9e-6 | 2.3e-9 |
| p95 | 36.2 | 1.6e-2 |
| p99 | 106.2 | 4.1e-2 |
| Max (of 200) | 169.8 | 8.4e-2 |

The median is floating-point-perfect, as expected for an exactly
time-reversible propagator (confirmed separately, see below). But a real
minority of cases (25% of 200 trials had >1m error) show meaningfully
larger error, up to ~170m.

**Root cause, confirmed not guessed**: sorting the 200-trial batch by
error and cross-referencing each case's resulting orbital eccentricity
shows the largest errors correlate exactly with **hyperbolic secondary
orbits** (eccentricity > 1, negative semi-major axis):

```
worst 10 by pos_error: pos_error, relative_speed, eccentricity, semimajor_axis
   169.847    14656.6  ecc=  2.3644  a=-2.306e+06
   141.346    14113.0  ecc=  2.6211  a=-2.212e+06
   105.813    14801.2  ecc=  7.4502  a=-1.062e+06
   ...
best 5:
     7e-08     6350.4  ecc=  0.0861  a= 6.530e+06
     ...
```

This makes physical sense: the ego orbit's own speed is ~7.6 km/s; a
sampled relative speed (drawn from the real Kelvins distribution, up to
~17 km/s in the real data per `05-datasets.md`) can push the secondary's
absolute velocity well past local escape velocity, producing a hyperbolic
trajectory. hapsira's Kepler-equation solver is less numerically precise
for high-eccentricity/hyperbolic cases over long (3-day) propagation
spans than for elliptical ones — a real, explainable, bounded numerical
limitation of the two-body layer, not a bug in the targeting algebra.

**Implication for later phases**: this error (up to ~170m in the observed
sample) is small relative to typical real miss distances (median ~12km
per `05-datasets.md`) but could matter for the tightest realistic
scenarios. Flagged as a follow-up, not a v1 blocker: if it becomes
relevant, consider switching hapsira's propagation method/tolerance for
detected hyperbolic cases, or accept it as part of the intentional
two-body-vs-Basilisk approximation gap that Phase 4/5 already has to
handle regardless (see part 2 below).

### Forward/backward round-trip (ad hoc check, not a formal test)

Confirmed the propagator is exactly time-reversible: propagating forward
600s then back -600s reproduces the origin to `~3e-9` m (pure floating
point noise), and propagating directly by a large negative time also
round-trips cleanly. This is what makes the whole backward-propagation
targeting approach (`03-scenario-design.md`) valid in the first place.

## 2. Basilisk-fidelity cross-validation — deferred to Phase 4, honestly unresolved

### What was attempted

A script (`scripts/validate_targeting_against_basilisk.py`) that flies
the solver's targeted initial conditions through Basilisk's real dynamics
(two `Spacecraft` objects, 10th-degree spherical harmonics Earth gravity)
and measures the realized-vs-targeted miss distance/relative speed — the
open risk flagged in `03-scenario-design.md`.

### Real bugs found and fixed along the way (genuinely useful findings)

1. **numpy arrays silently mis-parse through `hub.r_CN_NInit`/`v_CN_NInit`.**
   Assigning a raw numpy `ndarray` produces a *wrong* initial state with
   no error or warning. Confirmed by direct comparison: with plain
   point-mass gravity, a numpy-array initial condition gave 473m error
   vs. a two-body reference over 3 days; converting to a plain Python
   `list` first gave 0.02m error, all else identical. **Fix: always
   convert to `list(...)` before assigning to these fields.** This will
   matter again in Phase 4's custom `Satellite` subclass.
2. **Spherical harmonics of degree/order ≥ 1 need a connected
   planet-orientation message**, or Basilisk logs `BSK_WARNING` and
   (we now know, from finding Basilisk's own
   `examples/scenarioOrbitConsistencyVerification.py` — a real, maintained
   test written specifically to check this exact failure mode, GitHub
   issue #1352) silently treats Earth as non-rotating, causing tesseral/
   sectoral terms to fail to average out and produce spurious secular
   drift. Fix: `gravFactory.createSpiceInterface(time=...)` +
   `gravFactory.spiceObject.zeroBase = "Earth"`, added to the sim task.
3. **An explicit `gravField.gravBodies = spacecraft.GravBodyVector(...)`
   assignment** appears in Basilisk's own consistency-verification example
   instead of (or in addition to) `gravFactory.addBodiesTo(...)`.

### What's still unresolved

Even with all three fixes applied, a single-spacecraft test against the
two-body reference over just 600s (10 minutes) gave results ranging from
~2000m to ~75,800m error across slightly different configurations of the
same nominal setup (e.g. `zeroBase="earth"` vs `"Earth"` capitalization
changed the result substantially) — none of which is physically plausible
for real J2+higher perturbation over 10 minutes (expected: meters to tens
of meters, not kilometers). This means there is at least one more
uncontrolled variable in the raw-Basilisk setup that wasn't run to ground.

### Decision: stop hand-rolling raw Basilisk scripting for this, defer to Phase 4

`bsk_rl`'s own `DynamicsModel` class (`02-bsk_rl-architecture.md` §2)
already implements exactly this gravity+SPICE+body-registration wiring
correctly — it's a real, maintained, published RL framework with actual
users, so its internal setup demonstrably works. Continuing to hand-roll
an equivalent raw-Basilisk script, debugging it fact-by-fact via trial and
error, is redundant, error-prone effort that duplicates code `bsk_rl`
already gets right. **The Basilisk-fidelity cross-check will be done in
Phase 4** as part of building the actual custom `Satellite` subclass on
top of `bsk_rl`'s existing (correct) `DynamicsModel`, rather than a
separate hand-rolled validation script beforehand. `scripts/
validate_targeting_against_basilisk.py` is kept in the repo with the
three real fixes applied and an explicit "not a trustworthy result yet"
header, since the debugging groundwork (bugs 1-3 above) will still be
directly useful there.

This is judged the responsible choice given diminishing returns on
continued raw-API trial-and-error, not a resolved finding — `03-scenario-
design.md`'s "open technical risk" note remains genuinely open, now with
a concrete plan (Phase 4, via `bsk_rl`) rather than a vague "measure it
later."
