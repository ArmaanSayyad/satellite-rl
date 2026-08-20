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
