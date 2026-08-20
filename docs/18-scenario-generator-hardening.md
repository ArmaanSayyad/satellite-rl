# 18 — Scenario Generator Hardening (Phase 5)

Addresses the two items carried over from Phase 4
(`17-env-implementation-notes.md` part 3): the scenario generator's
missing orbit-sanity check, and the TCA-timing-sensitivity issue at high
relative speed. Written incrementally as each piece lands.

## 1. Orbit-sanity check — done

**Problem**: `solve_secondary_initial_state`'s relative-velocity
direction is sampled freely (uniform on the unit sphere, per
`03-scenario-design.md` — there's no real-world-derived distribution for
it). For some sampled directions, especially combined with large
relative speeds and multi-day lead times, the resulting secondary orbit
has a periapsis below a physically sane altitude. This wasn't caught
until the *environment* (Phase 4) hit `bsk_rl`'s own `altitude_valid`
aliveness check failing mid-episode — a crash, not a graceful rejection.

**Fix**: `osculating_periapsis_altitude_m(r, v)` — a direct vis-viva +
eccentricity-vector computation of the two-body osculating orbit's
periapsis altitude, valid for both elliptical and hyperbolic orbits
(periapsis `r_p = a(1-e)` is well-defined in both cases). Periapsis is an
intrinsic property of the orbit (conserved under two-body dynamics), so
it can be evaluated from any single point on the trajectory — no full
propagation needed, making this a cheap gate.

Folded into `solve_secondary_initial_state_robust`'s existing retry loop
(previously only retrying on Cowell integration failure, per
`17-env-implementation-notes.md`): now also rejects and resamples when
periapsis altitude falls below `min_altitude_m` (default `200e3`,
matching `bsk_rl`'s own `min_orbital_radius` default margin).

**Validated**: the exact parameters that crashed Phase 4's environment
(`miss_distance_m=500, relative_speed_ms=8000`, 5-day lead time) now
reliably produce a scenario with periapsis altitude ~493km, well clear of
the 200km floor — confirmed both interactively and in
`tests/test_targeting.py::test_robust_solver_rejects_low_periapsis_scenario`.
The periapsis formula itself is checked against two independent
references (not just self-consistency): a circular orbit, where periapsis
= apoapsis = the orbit's own constant radius by definition, and a
hand-constructed eccentric orbit checked against the closed-form vis-viva
periapsis speed.

**Retry-rate data — corrected after a closer look.** An initial 100-trial
batch (`max_attempts=20`) across the full realistic parameter range (miss
distance 20m–50km, relative speed 100m/s–15km/s per `05-datasets.md`,
lead time 0.5–7 days) succeeded 100/100, which read as "the retry
mechanism handles a rare edge case." Running the fast test suite exposed
that this was too optimistic: with the *original* default `max_attempts=10`,
one specific case (`miss_distance≈31.3km, relative_speed≈13.5km/s`)
failed outright after exhausting all 10 attempts.

Characterizing it directly (200 draws at that exact parameter point):
**only 45/200 (22.5%) of randomly-sampled relative-velocity directions
produced a periapsis-safe orbit** — a ~77.5% per-attempt rejection rate,
not a rare miss. A broader sweep of relative speed (500–15,000 m/s, fixed
miss distance/orientation, 60 draws each) confirmed this is typical, not
an outlier case:

| relative speed (m/s) | valid fraction |
|---|---|
| 500 | 42% |
| 2,000 | 28% |
| 5,000 | 22% |
| 8,000 | 18% |
| 12,000 | 20% |
| 15,000 | 20% |

**Corrected understanding**: periapsis rejection is a *common* outcome of
sampling an unconstrained relative-velocity direction (roughly 60–80% of
draws get rejected across the realistic speed range), not a rare failure
mode alongside the separate ~4% Cowell-integration-failure rate (which is
a different, much rarer issue — see `17-env-implementation-notes.md`).
The earlier 100/100 clean batch was consistent with this (at even the
worst ~18% per-attempt success rate, 20 attempts still succeeds ~98.8% of
the time per scenario) but didn't reveal how close to the edge it was
running. **Fix applied**: raised `max_attempts` default from 10 to 50,
which gives >99.99% cumulative success at the worst observed per-attempt
rate. `docs/03-scenario-design.md` and the function's own docstring
updated to state the real rate rather than the earlier optimistic one.

## 2. TCA-timing-sensitivity — real finding, but Phase 4's diagnosis was wrong

### What Phase 4 concluded (`17-env-implementation-notes.md` part 3)

Comparing one relative_speed=200 m/s case against one relative_speed=20
m/s case, the realized-vs-targeted miss distance error was much larger at
the higher speed. Phase 4 concluded this was a **timing** effect: a small
residual dynamics offset (Basilisk's true closest-approach time differing
slightly from the targeting solver's precomputed nominal instant),
amplified by relative speed (`position_error ≈ timing_offset_s ×
relative_speed`). That led to `refine_tca()` (this section): find the
TRUE local-minimum-separation time via a real Basilisk propagation from
t0, rather than trusting the nominal instant.

### What `refine_tca()` actually implements

`scenario/tca_refinement.py`: propagates both objects under full
Basilisk dynamics (10th-degree spherical harmonics + SPICE, matching
`bsk_rl`'s own `DynamicsModel` — reusing the exact setup validated in
Phase 4/`17`) from t0 to just past the nominal TCA, records the full
trajectory at 5–10s resolution, and locates the true minimum-separation
time via quadratic interpolation through the bracketing samples (no
second, finer simulation pass needed — benchmarked at ~1.5–3s wall clock
for a realistic 3-day, two-satellite scenario, a one-time cost per
scenario).

### Testing it broadly overturned the Phase 4 diagnosis

Running `refine_tca()` across a wider range of parameters than the single
before/after comparison Phase 4 used:

| relative speed | duration | nominal separation | refined separation | timing offset | target |
|---|---|---|---|---|---|
| 200 m/s | 0.2d | 2989m | 2356m | +4.6s | 300m |
| 2000 m/s | 0.2d | 901m | 901m | -0.25s | 300m |
| 8000 m/s | 0.2d | 272m | 272m | -0.01s | 300m |
| 15000 m/s | 0.2d | 238m | 238m | -0.00s | 300m |

At fixed duration, **higher relative speed correlates with a *smaller*
timing offset and a *smaller* realized-vs-target gap** — the opposite of
Phase 4's conclusion. And in most rows here, refinement changes nothing
(the nominal instant was already the true minimum) while the
realized-vs-target gap is still large or still small independent of that.
**The dominant driver is not a timing effect amplified by relative
speed.** Phase 4's conclusion was drawn from too small a sample (one
comparison) and doesn't hold up under broader testing — recorded here
plainly rather than left standing.

### Corrected understanding

The realized-vs-targeted miss distance gap is a **real, scenario-
dependent residual** from the J2-vs-full-Basilisk model gap (J3+,
tesseral/sectoral harmonics not modeled by the targeting solver's
propagator — see part 1 of `17-env-implementation-notes.md`). Its size
appears to depend on the specific sampled geometry (how the two orbits'
orientation interacts with Earth's higher-order gravity field) in ways
not explained by relative speed or duration alone — a genuine positional/
geometric divergence, not primarily a "right trajectory, sampled at the
wrong instant" timing problem. Timing refinement (`refine_tca()`) is real
and correctly implemented — it finds the objectively true closest-
approach time and distance, which is strictly at least as accurate as
assuming the nominal instant — but it does not reliably close the
dominant gap.

### What this means for the project, practically

Two honest options, and the pragmatic one is adopted for now:

1. **Full closed-loop re-targeting** (differential correction: fly the
   initial guess through real Basilisk dynamics, measure the error,
   adjust the secondary's initial condition, repeat until converged).
   This is the principled fix and how real orbit-determination software
   handles exactly this class of problem — but it's a substantially
   bigger feature (needs a sensitivity/Jacobian estimate, likely via
   finite differences, each requiring its own Basilisk propagation; not
   attempted here, flagged as a legitimate future improvement if tighter
   fidelity becomes necessary for training results).
2. **Use the realized (refined) values as the scenario's actual ground
   truth**, rather than insisting the environment match the originally-
   sampled target exactly. `refine_tca()`'s output (`refined_tca_s`,
   `min_separation_m`) is genuinely correct information about what the
   scenario actually is once flown through real dynamics — adopted here.
   `miss_distance_m`/`relative_speed_ms` passed to the targeting solver
   should be understood as *requested* values that ground the sampling
   distribution in real Kelvins statistics (per `03-scenario-design.md`),
   not as guarantees about the exact realized encounter.

`refine_tca()` is not yet wired into `CollisionAvoidanceEnv` — that
integration (updating the environment's schedule/reward reference to use
the refined TCA and realized separation) is deferred to the curriculum-
stage work (sampled geometry, evolving uncertainty) later in Phase 5,
where the environment is being modified anyway.
