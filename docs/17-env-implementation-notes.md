# 17 — Phase 4 Environment Implementation Notes

Covers three things: (1) resolving the Basilisk-fidelity question carried
over from Phase 3, which turned out to have a different answer than
Phase 3 assumed, (2) the `bsk_rl` API details needed to actually build
the environment, verified directly against the installed source (not
secondhand summaries), and (3) a real, still-open limitation found while
validating the full environment end-to-end: high-relative-speed
encounters need a TCA-refinement step this phase doesn't implement.

## 1. Resolving the Phase 3 "carried to Phase 4" item — with a correction

**Phase 3's framing was wrong.** `docs/16-targeting-validation-results.md`
treated the ~2000-3000km divergence between two-body-propagated and
Basilisk-propagated trajectories as an unresolved bug (something in the
raw-Basilisk setup not fully debugged). Building the real environment on
`bsk_rl` this phase — which handles Basilisk's gravity/SPICE wiring
correctly internally, sidestepping all three raw-API bugs Phase 3 found —
**reproduced the same ~2960km divergence anyway**, over the same 3-day,
500km/51.6°-inclination example orbit. That ruled out "bad raw Basilisk
setup" as the explanation.

**The real explanation: J2 nodal (RAAN) precession, and it's large.**
Checking orbital elements before/after the 3-day `bsk_rl` propagation:

```
initial: a=6878.14km e=0.00000 i=51.600deg Omega=0.000deg
final:   a=6866.28km e=0.00081 i=51.557deg Omega=345.690deg
```

`a`, `e`, `i` are essentially unchanged; `Ω` (RAAN) drifted by
`345.69° - 360° = -14.31°` over 3 days. This is exactly J2's known
secular signature (RAAN/argument-of-perigee drift, semi-major
axis/eccentricity/inclination roughly constant), and it matches a direct
hand calculation from the standard J2 secular RAAN-drift formula
(`dΩ/dt ≈ -1.5 · n · J2 · (Re/p)² · cos(i)`), which predicts **-14.26°**
for this exact orbit — a near-exact match. A 14° rotation of the orbital
plane at ~6878km radius corresponds to a multi-thousand-km position
difference from a naively non-precessing two-body prediction, fully
explaining the magnitude Phase 3 judged "not physically plausible." That
judgment was the error, not the Basilisk setup — J2 is a large secular
effect, not a small correction, and `docs/16` under-weighted this.

**Fix**: `scenario/targeting.py`'s propagator was upgraded from plain
two-body Keplerian to a J2-perturbed Cowell propagator (hapsira's
`CowellPropagator` with a custom RHS adding `hapsira.core.perturbations.
J2_perturbation` to the two-body acceleration). Result, same scenario:

| | Two-body (Phase 3) | J2-aware (Phase 4) |
|---|---|---|
| Position error vs. Basilisk, 3-day propagation | ~2,960,787 m | ~6,836 m |

A **~433x reduction**, from adding exactly one well-understood
perturbation term. The residual ~6.8km over 3 days is consistent with
higher-order terms (J3+, tesseral/sectoral harmonics from the full
GGM03S degree-10 field) that aren't modeled — a much smaller, expected
gap relative to realistic miss-distance scales (median ~12km per
`05-datasets.md`), not a red flag.

**Practical implication this fix addresses**: without J2, the targeting
solver's initial conditions would not have actually produced the intended
conjunction once flown through Basilisk's real dynamics for any
multi-day lead time — the ~3000km drift dwarfs realistic miss distances
entirely. This was a real correctness issue for the project, not just an
accuracy nicety, and Phase 4 catching it before building the full episode
loop on top of a broken targeting solver is exactly why this got checked
now rather than discovered later during training.

### A new problem the J2 upgrade introduced: integration failures

hapsira's Cowell integrator (`hapsira.core.propagation.cowell.cowell`)
hardcodes `atol=1e-12` in its `scipy.integrate.solve_ivp` call,
regardless of the problem's actual state magnitude (km-scale positions
and velocities — an absolute tolerance of `1e-12` against a ~1e4 km
state is unreasonably tight, effectively demanding double-precision-limit
accuracy). Empirically, this causes `RuntimeError: Integration failed`
for a real ~4% of sampled geometries (2/50 trials), not confined to
extreme relative speeds. Loosening `rtol` (the only tolerance the public
`CowellPropagator` API exposes) does not help, since `atol` is the
binding constraint and isn't exposed.

**Fix**: `solve_secondary_initial_state_robust()` retries with a
resampled relative-velocity direction on failure (up to `max_attempts`).
This is a legitimate fix, not a hidden change to the requested scenario —
the relative-velocity *direction* is already a free/sampled parameter
(not derived from real data), and the requested miss distance/relative
speed *magnitudes* are preserved exactly across retries. With retries,
200/200 trials succeeded; self-consistency error distribution with J2
is tighter and more uniform than the old two-body case (no more
hyperbolic-orbit fat tail): max 10.3m / 0.013 m/s, p99 6.4m / 0.011 m/s,
vs. the old max of 170m. Test tolerances in `tests/test_targeting.py`
updated accordingly.

## 2. `bsk_rl` API notes for building the environment

Verified by reading the actual installed source
(`.venv/lib/python3.11/site-packages/bsk_rl/`), not secondhand summaries.

- **A passive, non-agent secondary object must still be a full
  `Satellite`.** `ConjunctionDynModel`'s proximity check
  (`sim/dyn/relative_motion.py`) iterates `simulator.dynamics_list.values()`
  and only pairs objects that are both `isinstance(sat_dyn,
  ConjunctionDynModel)` — there's no path for a non-`Satellite` object to
  participate. bsk_rl's own RSO Inspection example handles this by giving
  the passive object a degenerate single-choice `action_spec` (e.g.
  `[act.NadirPoint(duration=1e9)]`); we use `[act.Drift(duration=1e9)]`
  for the same purpose (`env/satellites.py`'s `SecondarySatellite`).
- **`observation_spec` cannot be empty** — `vectorize_nested_dict`
  errors trying to concatenate zero arrays. Even a satellite nobody reads
  observations from needs at least one trivial entry (we mirror the RSO
  example's `SatProperties({"prop": "one", "fn": lambda _: 1.0})`).
- **`SatProperties`'s `prop` key is required even when using `fn`** —
  `get_obs()` unconditionally does `prop = obs_property["prop"]` before
  checking for `fn`. (An earlier secondhand research summary, from Phase 0,
  implied `fn` alone was sufficient — corrected here after reading the
  actual source.)
- **`sat_args` orbit specification is strict**: `setup_spacecraft_hub`
  requires *either* `(rN, vN)` *or* `(oe, mu)`, not both, and raises
  `KeyError("Orbit is overspecified")` if both are non-`None`. The
  default `sat_args` value for `oe` is `random_orbit` (a callable,
  evaluated to a real random orbit if not overridden) — **so providing
  `rN`/`vN` without also explicitly passing `oe=None` triggers the
  "overspecified" error**, since the default `oe` won't be `None` unless
  explicitly overridden. Easy to miss; caught by testing, not guessed.
- **bsk_rl deep-copies satellites at env construction** —
  `GeneralSatelliteTasking.__init__` does `self.satellites =
  deepcopy(satellites)`. The `Satellite` objects passed to the
  constructor are not the live objects the simulation actually updates;
  use `env.satellites[i]` (or `env.satellites[0].dynamics`, etc.) to read
  live state, not the original references held before construction.
- **`GeneralSatelliteTasking.step()` takes a tuple of actions, one per
  satellite**, in `self.satellites` order; `SatelliteTasking` (single-
  satellite convenience wrapper) is unusable here since it hard-asserts
  exactly one satellite. Our env wraps `GeneralSatelliteTasking` directly
  and internally constructs the joint action tuple, following the exact
  same pattern `SatelliteTasking` itself uses (thin subclass overriding
  `action_space`/`observation_space`/`step`/`_get_obs`/`_get_info`) —
  see `gym.py` lines 561-608 for the reference pattern we mirrored.
- **`ImpulsiveThrustHill(chief_name, ...)`**: the Hill frame is defined
  relative to whatever satellite `chief_name` names (found via
  `simulator.get_satellite(chief_name)` at `reset_post_sim_init`). We use
  the ego satellite's own name as its own chief — i.e. Δv is expressed in
  the ego's own RTN/Hill frame (its own orbit), which is the standard way
  a real maneuver would be described (radial/along-track/cross-track
  relative to your own orbit), not the target's frame. This wasn't fully
  pinned down in `07-action-space.md`'s original design and is settled
  here.
- **Numpy arrays are safe with `sat_args`' `rN`/`vN`** — unlike the raw
  Basilisk scripting in Phase 3 (where a bare numpy array silently
  mis-parsed), `setup_spacecraft_hub` explicitly wraps assignment via
  `np2EigenVectorXd(rN)` before setting `hub.r_CN_NInit`, so bsk_rl's own
  API handles this conversion correctly. Confirms deferring the fidelity
  work to Phase 4's `bsk_rl`-based approach (per `docs/16`) was the right
  call — this exact class of bug doesn't recur at this layer.

## 3. A real limitation found validating the full environment: TCA timing sensitivity

With the full `CollisionAvoidanceEnv` built (observation/reward wiring,
episode stepping — see `12-architecture.md`'s `env/` layout), a
never-maneuver baseline run with the env's original defaults
(`miss_distance_m=500`, `relative_speed_ms=8000`, a 5-day schedule)
crashed the episode outright: `Secondary: failed altitude_valid check` —
the targeting solver's backward-propagated initial condition put the
secondary on an orbit that dips below bsk_rl's default 200km minimum
altitude at some point in its history. Reducing to a shorter, calmer
scenario (0.2-day schedule, `relative_speed_ms=200`) avoided the crash but
surfaced a subtler, more important issue: the **realized** miss distance
at TCA (as actually flown by Basilisk) came out to ~4022m against a
**targeted** 300m — a ~13x discrepancy, on a timescale (0.2 days, ~3
orbits) where the J2-corrected propagator had already been shown to track
Basilisk closely (`~6.8km` residual over a **3-day**, ~45-orbit window —
see part 1 above).

**Diagnosis, confirmed not assumed**: the *closing rate* in the final
approach (`(152290m - 4022m) / 864s ≈ 171 m/s`) was close to the targeted
200 m/s relative speed — the trajectory shape was basically right, but
the position at our assumed TCA instant was off. That's the signature of
a **timing** error, not a trajectory error: our targeting solver defines
"the encounter" as a specific relative state at a precomputed instant
(`t0 + time_to_tca`), but Basilisk's true dynamics (residual terms beyond
J2 — J3+, tesseral/sectoral harmonics) shift *when* true closest approach
actually occurs by some small amount. A small timing shift, multiplied by
a large relative speed, becomes a large positional error:
`4022m / 200m/s ≈ 20s` of implied timing error — plausible given the
known ~km-scale residual dynamics gap over multi-day windows.

**Confirmed empirically**: rerunning the identical scenario with
`relative_speed_ms=20` (10x lower) gave a realized miss distance of
365.85m against the same 300m target — a ~66m gap, not ~3700m. The error
shrank roughly proportionally with relative speed, exactly matching the
timing-error-amplification hypothesis rather than a general fidelity
problem.

**Practical implication**: the current targeting solver is reliable for
low-to-moderate relative-speed scenarios (confirmed clean end-to-end
below), but for realistic Kelvins-derived relative speeds (up to ~17 km/s
per `05-datasets.md`) it will systematically produce encounters looser
than targeted. **Not fixed in Phase 4** — the correct fix is a proper
TCA-refinement step (numerically searching for the true local-minimum
separation time near the nominal TCA, rather than assuming it occurs
exactly at the precomputed instant), which is a real, well-scoped
follow-up, not attempted here to keep Phase 4 focused on proving the
pipeline runs correctly end-to-end. Flagged clearly rather than left as a
silent gap: `tests/test_env.py` and this environment's example usage
deliberately use a low-relative-speed regime where this effect stays
small, with an explicit comment pointing back to this section.

Separately: bsk_rl's default `min_orbital_radius` (200km altitude)
rejecting some backward-propagated secondary trajectories is also a real,
un-addressed gap in the scenario generator — it currently doesn't check
that the solved initial condition corresponds to a sane, non-terminating
orbit before handing it to the environment. Also a Phase 5 follow-up
(reject-and-resample, similar in spirit to the Cowell-integration-failure
retry in part 1).

## Status

`env/` (`satellites.py`, `observations.py`, `collision_avoidance_env.py`)
implemented and validated end-to-end in the low-relative-speed regime:
construction, `reset()`, full-episode `step()` with real actions,
`gymnasium.utils.env_checker` compliance, and a random/baseline-policy
sanity check confirming the reward function differentiates behaviors
sensibly (never-maneuver vs. always-max-thrust give different, explicable
total rewards driven by the risk/fuel/disruption terms as designed).
Throughput benchmarked at ~8.5 steps/sec on this machine (informs
`10-rl-algorithm.md`'s Phase 6 compute planning). Two real limitations
found and documented rather than hidden (part 3 above): TCA timing
sensitivity at high relative speed, and no orbit-sanity check in the
scenario generator — both flagged as Phase 5 follow-ups, not blockers for
Phase 4's "does the pipeline work" goal.
