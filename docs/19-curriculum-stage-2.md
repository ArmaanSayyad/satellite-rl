# 19 — Curriculum Stage 2: Sampled Geometry (Phase 5c)

Implements `03-scenario-design.md`'s curriculum stage 2: instead of one
fixed, hardcoded encounter (Phase 4), each episode samples a fresh real
event's geometry from the bootstrap table built in Phase 2
(`data/fitted/geometry_events.csv`, ~8,700 real Kelvins events — see
`15-distribution-fitting-results.md` for why bootstrap resampling was
chosen over independent parametric fits).

## What changed

`CollisionAvoidanceEnv(sample_geometry=True, ...)`: every `reset()` draws
a new real event's `(miss_distance, relative_speed, sigma_x, sigma_z,
combined_radius)` and a fresh (still free/uniform, per
`03-scenario-design.md`) encounter-plane orientation angle, solves for
the secondary's initial state via the Phase 5a/5b-hardened targeting
solver, and builds that episode's scenario from it. `sample_geometry=False`
(default) preserves Phase 4's exact fixed-scenario behavior — this was a
pure addition, not a breaking change (confirmed: all of Phase 4's
existing `test_env.py` tests still pass unmodified).

## Two real implementation problems this surfaced

### 1. bsk_rl evaluates sat_args callables independently — needed explicit coupling

`sat_args` values can be callables, re-evaluated fresh every `reset()`
(bsk_rl's own randomization mechanism — see `17-env-implementation-
notes.md`). Naively giving the secondary's `rN` and `vN` two *separate*
sampling closures would let them each draw their own random event,
producing a physically inconsistent position/velocity pair. Fixed with
`env/scenario_sampling.py`'s `SecondaryScenarioSampler`: a generation
counter, incremented by the env's `reset()`, that both `rN()` and `vN()`
check — whichever is called first for a new generation does the actual
sample+solve and caches it; the other just reads the cache. This doesn't
rely on any assumption about which of `rN`/`vN` bsk_rl calls first.

### 2. The Pc observation's sigma/combined_radius could no longer be class-level constants

Phase 4's `make_ego_satellite_class(sigma, combined_radius, ...)` baked
these into the observation function as closure constants at class-
creation time — fine when there's one fixed scenario for the env's whole
lifetime, broken once they vary per episode. Fixed by making them mutable
per-satellite state (`satellite._pc_sigma`, `satellite._pc_combined_radius`,
set by the env's `reset()` before calling `super().reset()`), mirroring
the pattern Phase 4 already used for `_time_to_tca_s`.
`observations.make_collision_pc_fn` and `satellites.make_ego_satellite_class`
both simplified accordingly (dropped `sigma`/`combined_radius` params
entirely).

**Ordering bug caught before it shipped**: `GeneralSatelliteTasking.reset()`
calls `_get_obs()` (which needs `_pc_sigma`/`_pc_combined_radius` already
set) at its own end. Setting these mutable attributes *after* calling
`super().reset()` — the natural-looking order — would read stale or
missing values on the very first observation of every episode. Fixed by
setting them *before* `super().reset()`, relying on the sampler's
generation-counter idempotency (triggering the sample+solve via
`current_sigma`/`current_combined_radius` before `rN()`/`vN()` are even
called; both reuse the same cached result).

## What's deliberately kept from Phase 4

The `conjunction_radius` used for bsk_rl's own terminal collision check
(`ConjunctionDynModel`) stays **fixed** (a `conjunction_radius_m`
constructor parameter, independent of the sampled `combined_radius`).
These represent genuinely different things: a real physical collision
threshold (bsk_rl's job) versus our own risk model's assumed combined
object size for the Pc *estimate* (which should vary per real event,
since different events involve different actual object sizes). Conflating
them would be a modeling error, not a simplification worth avoiding.

Sigma remains isotropic (per `observations.make_collision_pc_fn`'s
existing design) — for stage 2, computed as the geometric mean of the
sampled event's real `sigma_x`/`sigma_z`, so it's a genuine per-event
value, not a global constant, while still not modeling the real
anisotropy or its evolution across an episode's decision points. Both of
those remain curriculum stage 3 scope.

## A real, deliberate Gym-convention deviation: seeding

`gymnasium.utils.env_checker` warns: *"Step observations are not equal
although similar given the same seed and action."* This is expected and
was checked, not just noticed and ignored: `CollisionAvoidanceEnv`'s
internal sampling RNG (`np.random.default_rng(targeting_seed)`) is
created once at `__init__` and keeps advancing across `reset()` calls —
it is **not** reseeded by the `seed` argument passed to `reset()`. This
means calling `env.reset(seed=42)` twice in a row does **not** reproduce
the same sampled scenario the second time, which technically violates
strict Gym determinism conventions.

This is deliberate, not an oversight: for curriculum training, the whole
point of `sample_geometry=True` is that each episode sees a *different*
real event — freezing the sample on every `reset(seed=X)` call would
defeat that. `targeting_seed` still controls overall reproducibility (two
environments constructed with the same `targeting_seed` draw the same
*sequence* of scenarios across resets), which is the reproducibility
guarantee that actually matters for this use case. Documented here so
it's a known, intentional trade-off if it surfaces again (e.g. in a
future evaluation harness that does want exact reset-to-reset
reproducibility for a fixed benchmark — see `11-evaluation.md` — that use
case should hold `sample_geometry=False` with a fixed scenario, or add a
dedicated re-seeding hook, not assume this env's default sampling mode is
reproducible per-reset).

## Validated

`env_checker` passes (warning above aside — it's advisory, not a
failure) in both modes. Confirmed across repeated resets: genuinely
different real events get sampled (miss distance/relative speed/combined
radius all vary), `_pc_sigma` actually tracks the sampled event (not
stuck at a stale value), and full episodes run to completion with valid
terminal Pc values. 10/10 `test_env.py` tests pass (4 new, 6 unchanged
from Phase 4).
