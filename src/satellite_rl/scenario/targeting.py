"""Backward-propagation targeting solver: given the ego satellite's known
orbit and a desired encounter geometry at TCA, solve for the secondary
object's initial state (at episode start, t0) such that propagation
produces that encounter. See docs/03-scenario-design.md "Generating a
conjunction: the targeting problem".

Uses hapsira's Cowell propagator with a J2 perturbation term (see
docs/17-env-implementation-notes.md). This was originally plain two-body
Keplerian propagation, but Phase 4 found that matters a lot: J2 causes
real, large secular RAAN (nodal) precession for LEO orbits -- ~14 deg over
3 days for a 500km/51.6deg example orbit, empirically confirmed against
bsk_rl's real Basilisk dynamics -- so plain two-body targeting produced
initial conditions that diverged by ~2960km from the targeted encounter
once actually flown through Basilisk over a 3-day lead time. Adding J2 to
the propagator here cut that to ~6.8km (a ~433x reduction) for the same
scenario. Higher-order terms (J3+, tesseral/sectoral harmonics) account
for the remainder -- not added here, since J2 captures the dominant
secular effect and the residual is much smaller relative to realistic
miss-distance scales.

Round-trip (forward-then-backward) propagation is still highly accurate
with J2 included, though not to the machine-precision level of the pure
two-body closed-form solution (which has an exact analytic inverse) --
see docs/17 for the measured self-consistency numbers with J2 enabled.

Dependency note: hapsira 0.18.0 (its only PyPI release as of Aug 2026)
requires astropy<7 -- see the pin and comment in pyproject.toml.
"""

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from hapsira.bodies import Earth
from hapsira.core.perturbations import J2_perturbation
from hapsira.core.propagation.base import func_twobody
from hapsira.twobody import Orbit
from hapsira.twobody.propagation import CowellPropagator

from ..pc.geometry import encounter_plane_basis

_EARTH_J2 = Earth.J2.value
_EARTH_R_KM = Earth.R.to(u.km).value


def _j2_accel(t0, u_, k):
    """Cowell-propagator RHS: two-body acceleration plus J2. Operates in
    km/km-s internally, matching hapsira's core propagation functions
    (see their docstrings) -- unrelated to the m/m-s convention used
    everywhere else in this module's public API.
    """
    du = func_twobody(t0, u_, k)
    ax, ay, az = J2_perturbation(t0, u_, k, J2=_EARTH_J2, R=_EARTH_R_KM)
    du[3] += ax
    du[4] += ay
    du[5] += az
    return du


_J2_PROPAGATOR = CowellPropagator(f=_j2_accel)

_EARTH_MU_M3S2 = Earth.k.to(u.m**3 / u.s**2).value
_EARTH_R_M = Earth.R.to(u.m).value
DEFAULT_MIN_ALTITUDE_M = 200e3  # matches bsk_rl's own default min_orbital_radius margin
# Added Phase 7e (docs/26-precise-targeting.md): the hyperbolic-orbit
# check (eccentricity >= 1) rejects the clearest failures but not
# borderline ones -- a real event produced eccentricity=0.982 (BOUND, so
# not caught) with apoapsis near GEO altitude and r_sec_t0 73,584km from
# Earth, a wildly unrealistic "LEO debris object" and numerically
# ill-conditioned (near-parabolic orbits are inherently sensitive to
# small dynamics-model differences -- exactly what made a Basilisk-
# ground-truth correction converge against one setup while the actual
# training environment produced a completely different outcome for the
# same "corrected" state). This dataset's conjunctions are between LEO
# objects (docs/01-problem-scope.md's scope), so 2,000km -- the common
# industry LEO definition ceiling -- is a physically justified bound, not
# an arbitrary one.
DEFAULT_MAX_APOAPSIS_ALTITUDE_M = 2_000e3


def osculating_periapsis_altitude_m(r_m: np.ndarray, v_m: np.ndarray) -> float:
    """Periapsis altitude (m above Earth's surface) of the two-body
    osculating orbit through state (r_m, v_m), via vis-viva + the
    eccentricity vector -- valid for elliptical AND hyperbolic orbits
    (periapsis is well-defined in both cases: r_p = a(1-e), with a<0 for
    e>1). Periapsis is an intrinsic property of the orbit (conserved
    under two-body dynamics), so evaluating it from any single point on
    the trajectory -- not just t0 -- gives the same answer; used as a
    cheap orbit-sanity gate without needing a full propagation.

    See docs/17-env-implementation-notes.md's "no orbit-sanity check"
    follow-up and docs/18-scenario-generator-hardening.md for why this
    matters: sampled relative-velocity directions can otherwise produce
    secondary trajectories that dip below a physically sane altitude.
    """
    r = np.asarray(r_m, dtype=float)
    v = np.asarray(v_m, dtype=float)
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    specific_energy = v_mag**2 / 2 - _EARTH_MU_M3S2 / r_mag
    semi_major_axis = -_EARTH_MU_M3S2 / (2 * specific_energy)
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / _EARTH_MU_M3S2 - r / r_mag
    eccentricity = np.linalg.norm(e_vec)
    periapsis_radius = semi_major_axis * (1 - eccentricity)
    return periapsis_radius - _EARTH_R_M


def osculating_apoapsis_altitude_m(r_m: np.ndarray, v_m: np.ndarray) -> float:
    """Apoapsis altitude (m above Earth's surface) of the two-body
    osculating orbit through state (r_m, v_m) -- only meaningful for
    BOUND orbits (eccentricity < 1); for hyperbolic orbits this returns a
    negative/meaningless value (no apoapsis exists), so callers must
    check `osculating_eccentricity(...) < 1.0` first, not rely on this
    function alone to reject hyperbolic cases. See
    `osculating_eccentricity`'s docstring for why apoapsis needs its own
    check separate from eccentricity/periapsis: an orbit can be BOUND
    (e<1) with a perfectly fine periapsis and still have an unrealistic,
    near-GEO-altitude apoapsis.
    """
    r = np.asarray(r_m, dtype=float)
    v = np.asarray(v_m, dtype=float)
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    specific_energy = v_mag**2 / 2 - _EARTH_MU_M3S2 / r_mag
    semi_major_axis = -_EARTH_MU_M3S2 / (2 * specific_energy)
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / _EARTH_MU_M3S2 - r / r_mag
    eccentricity = np.linalg.norm(e_vec)
    apoapsis_radius = semi_major_axis * (1 + eccentricity)
    return apoapsis_radius - _EARTH_R_M


def osculating_eccentricity(r_m: np.ndarray, v_m: np.ndarray) -> float:
    """Eccentricity of the two-body osculating orbit through state
    (r_m, v_m). Added Phase 7e (docs/26-precise-targeting.md) alongside
    `osculating_periapsis_altitude_m`'s existing lower-bound (periapsis-
    too-low) sanity check, after finding a real event (relative_speed=
    14,919 m/s, well above typical LEO orbital speed) produced a
    secondary state with a perfectly fine periapsis altitude (487km,
    comfortably above the floor) but eccentricity 7.03 -- HYPERBOLIC.
    Periapsis altitude alone can't catch this: it's well-defined and can
    look completely normal for a hyperbolic flyby (the object passes
    Earth once and never returns), which `osculating_periapsis_altitude_m`
    correctly computes but has no way to flag as unrealistic on its own.
    A hyperbolic "secondary satellite" isn't a physically sane object to
    be conducting a conjunction assessment against -- real catalogued
    objects are in bound, periodic orbits.
    """
    r = np.asarray(r_m, dtype=float)
    v = np.asarray(v_m, dtype=float)
    r_mag = np.linalg.norm(r)
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / _EARTH_MU_M3S2 - r / r_mag
    return float(np.linalg.norm(e_vec))


@dataclass
class TargetedScenario:
    """Result of solving for a secondary object's initial state."""

    r_sec_t0: np.ndarray  # secondary's position at episode start, m (ECI/GCRS)
    v_sec_t0: np.ndarray  # secondary's velocity at episode start, m/s
    r_ego_tca: np.ndarray  # ego's targeted position at TCA, m
    v_ego_tca: np.ndarray  # ego's targeted velocity at TCA, m
    r_sec_tca_target: np.ndarray  # secondary's TARGETED position at TCA, m
    v_sec_tca_target: np.ndarray  # secondary's TARGETED velocity at TCA, m/s
    miss_distance_target: float  # m, as specified (sanity: should equal input)
    relative_speed_target: float  # m/s, as specified


def propagate_state(
    r0_m: np.ndarray, v0_ms: np.ndarray, dt_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a Cartesian state (meters, m/s) by dt_s seconds (positive
    or negative) using J2-perturbed (Cowell) dynamics -- see module
    docstring for why plain two-body isn't sufficient.
    """
    orbit = Orbit.from_vectors(Earth, np.asarray(r0_m) * u.m, np.asarray(v0_ms) * (u.m / u.s))
    propagated = orbit.propagate(dt_s * u.s, method=_J2_PROPAGATOR)
    r = propagated.r.to(u.m).value
    v = propagated.v.to(u.m / u.s).value
    return np.asarray(r), np.asarray(v)


def solve_secondary_initial_state(
    ego_r0_m: np.ndarray,
    ego_v0_ms: np.ndarray,
    time_to_tca_s: float,
    miss_distance_m: float,
    relative_speed_ms: float,
    orientation_angle_rad: float,
    rng: np.random.Generator,
) -> TargetedScenario:
    """Solve for the secondary object's state at t0 such that two-body
    propagation from t0 to TCA (= t0 + time_to_tca_s) produces a conjunction
    with the ego satellite at exactly the specified miss distance and
    relative speed.

    The relative-velocity DIRECTION at TCA is a free parameter -- not
    constrained by the Kelvins-fit magnitudes we sample (miss_distance,
    relative_speed are magnitudes only) -- so it's sampled uniformly on the
    unit sphere. `orientation_angle_rad` places the miss vector within the
    resulting encounter plane (also a free parameter given only a scalar
    miss distance).

    Args:
        ego_r0_m, ego_v0_ms: ego satellite's state at episode start t0.
        time_to_tca_s: time from t0 to the desired closest approach.
        miss_distance_m, relative_speed_ms: targeted encounter magnitudes.
        orientation_angle_rad: angle of the miss vector within the
            encounter plane, in [0, 2*pi).
        rng: used to sample the relative-velocity direction.
    """
    r_ego_tca, v_ego_tca = propagate_state(ego_r0_m, ego_v0_ms, time_to_tca_s)

    v_rel_hat = rng.normal(size=3)
    v_rel_hat /= np.linalg.norm(v_rel_hat)
    v_rel = relative_speed_ms * v_rel_hat

    basis = encounter_plane_basis(v_rel)  # (3, 2): [e1, e2], both perp to v_rel
    miss_2d = miss_distance_m * np.array(
        [np.cos(orientation_angle_rad), np.sin(orientation_angle_rad)]
    )
    r_rel = basis @ miss_2d  # perpendicular to v_rel by construction -> valid TCA

    r_sec_tca = r_ego_tca + r_rel
    v_sec_tca = v_ego_tca + v_rel

    r_sec_t0, v_sec_t0 = propagate_state(r_sec_tca, v_sec_tca, -time_to_tca_s)

    return TargetedScenario(
        r_sec_t0=r_sec_t0,
        v_sec_t0=v_sec_t0,
        r_ego_tca=r_ego_tca,
        v_ego_tca=v_ego_tca,
        r_sec_tca_target=r_sec_tca,
        v_sec_tca_target=v_sec_tca,
        miss_distance_target=miss_distance_m,
        relative_speed_target=relative_speed_ms,
    )


def solve_secondary_initial_state_robust(
    ego_r0_m: np.ndarray,
    ego_v0_ms: np.ndarray,
    time_to_tca_s: float,
    miss_distance_m: float,
    relative_speed_ms: float,
    orientation_angle_rad: float,
    rng: np.random.Generator,
    max_attempts: int = 3000,
    min_altitude_m: float = DEFAULT_MIN_ALTITUDE_M,
    max_apoapsis_altitude_m: float = DEFAULT_MAX_APOAPSIS_ALTITUDE_M,
) -> TargetedScenario:
    """Same as `solve_secondary_initial_state`, but retries with a
    resampled relative-velocity direction on (a) integration failure, (b)
    an unsafe resulting secondary orbit, (c) a hyperbolic one, or (d) a
    bound-but-unrealistically-stretched one.

    (a) hapsira's Cowell integrator (`solve_secondary_initial_state`'s J2
    propagation) hardcodes `atol=1e-12` internally regardless of the
    problem's actual state magnitude (km-scale positions/velocities) --
    empirically, this causes `RuntimeError: Integration failed` for a
    real, non-negligible fraction of sampled geometries (~4% observed
    over 50 trials, not exclusively at extreme relative speeds -- see
    docs/17-env-implementation-notes.md).

    (b) A sampled relative-velocity direction can put the secondary on an
    orbit whose periapsis dips below a physically sane altitude --
    bsk_rl's own `altitude_valid` aliveness check would fail mid-episode
    if this reaches the environment (found empirically building
    Phase 4's env -- see docs/17). Checked here via
    `osculating_periapsis_altitude_m` against `min_altitude_m` (default
    matches bsk_rl's own `min_orbital_radius` margin).

    (c) A sampled relative-velocity direction/magnitude can put the
    secondary on a HYPERBOLIC orbit (eccentricity >= 1) -- physically
    unrealistic for a catalogued object (real satellites/debris are all
    in bound, periodic orbits), but NOT caught by (b): periapsis is
    well-defined and can look perfectly normal for a hyperbolic flyby.
    Found in Phase 7e (docs/26-precise-targeting.md) via a real event
    with `relative_speed_ms=14,919` -- well above typical LEO orbital
    speed (~7,600 m/s) -- which produced a secondary state with periapsis
    altitude 487km (comfortably valid) but eccentricity 7.03. This
    mattered in practice, not just in principle: such a state propagated
    to a t0 position hundreds of thousands of km away (a near-lunar-
    distance "LEO debris object"), which was numerically unstable enough
    that a Basilisk-ground-truth correction loop checking it could
    converge to a small 2D miss-vector error while the underlying 3D
    state remained physically absurd -- silently producing garbage that
    passed a narrow correctness check. High relative speeds are common
    among real high-risk Kelvins events specifically, so this isn't a
    rare corner case for exactly the regime docs/24-risk-stratified-
    sampling.md's elevated-risk pool draws from.

    (d) Rejecting only hyperbolic orbits (c) turned out insufficient on
    its own: a different real event produced a BOUND orbit (eccentricity
    0.982, so not caught by (c)) with periapsis altitude 415km
    (comfortably valid, so not caught by (b) either) but an apoapsis near
    GEO altitude -- `r_sec_t0` 73,584km from Earth, a wildly unrealistic
    "LEO debris object" for a dataset whose conjunctions are between LEO
    objects (docs/01-problem-scope.md), and just as numerically fragile
    as a hyperbolic orbit for the same reason (near-parabolic orbits are
    inherently sensitive to small dynamics-model differences). Checked
    via `osculating_apoapsis_altitude_m` against `max_apoapsis_altitude_m`
    (default 2,000km, the common industry LEO ceiling) -- ONLY meaningful
    once (c) has already confirmed the orbit is bound, since apoapsis
    altitude is undefined (returns a meaningless negative value) for
    hyperbolic orbits.

    **This second failure mode is common, not a rare edge case** --
    measured per-attempt valid-orbit rates of only ~18-42% depending on
    relative speed (higher speed -> lower valid rate; see
    docs/18-scenario-generator-hardening.md for the full sweep), so
    `max_attempts` needs real headroom, not just a safety margin.

    **Default raised from 50 to 3,000 in Phase 7e** (docs/26-precise-
    targeting.md) after adding checks (c) and (d) exposed that 50 wasn't
    actually enough across the full realistic parameter range these
    docstrings already claimed to cover: at `relative_speed_ms=14,919`
    (a real Kelvins event, well above typical LEO orbital speed), the
    measured per-attempt success rate under the new (correct) checks was
    roughly 0.3%, not the ~18-42% the old estimate was based on (that
    estimate only reflected failure mode (b), never (c)/(d), since
    neither existed as a check yet). 300 attempts gave only 7/10 for that
    case; 3,000 gave 10/10 in a real test (~8s wall-clock for that
    specific worst-case draw). J2 attempts are cheap (~10ms each), so
    this doesn't meaningfully slow the common case, which still converges
    in a handful of attempts regardless of this ceiling.

    Since the relative-velocity DIRECTION is already a free parameter
    we're sampling (not something with a real-world-derived distribution
    -- see `solve_secondary_initial_state`'s docstring), resampling it on
    either failure mode is a legitimate retry, not silently changing the
    requested scenario's actual physical parameters (miss distance and
    relative speed magnitude are preserved exactly across retries).
    """
    last_error: Exception = RuntimeError("max_attempts must be >= 1")
    for _attempt in range(max_attempts):
        try:
            scenario = solve_secondary_initial_state(
                ego_r0_m,
                ego_v0_ms,
                time_to_tca_s,
                miss_distance_m,
                relative_speed_ms,
                orientation_angle_rad,
                rng,
            )
        except RuntimeError as exc:
            last_error = exc
            continue

        periapsis_altitude = osculating_periapsis_altitude_m(
            scenario.r_sec_t0, scenario.v_sec_t0
        )
        if periapsis_altitude < min_altitude_m:
            last_error = ValueError(
                f"secondary orbit periapsis altitude {periapsis_altitude:.0f}m "
                f"below minimum {min_altitude_m:.0f}m"
            )
            continue

        eccentricity = osculating_eccentricity(scenario.r_sec_t0, scenario.v_sec_t0)
        if eccentricity >= 1.0:
            last_error = ValueError(
                f"secondary orbit is hyperbolic (eccentricity={eccentricity:.3f})"
            )
            continue

        apoapsis_altitude = osculating_apoapsis_altitude_m(scenario.r_sec_t0, scenario.v_sec_t0)
        if apoapsis_altitude > max_apoapsis_altitude_m:
            last_error = ValueError(
                f"secondary orbit apoapsis altitude {apoapsis_altitude:.0f}m "
                f"above maximum {max_apoapsis_altitude_m:.0f}m (eccentricity={eccentricity:.3f})"
            )
            continue

        return scenario

    raise RuntimeError(
        f"solve_secondary_initial_state failed after {max_attempts} attempts "
        f"(miss_distance={miss_distance_m}, relative_speed={relative_speed_ms}): {last_error}"
    )


def validate_self_consistency(
    scenario: TargetedScenario, time_to_tca_s: float
) -> tuple[float, float]:
    """Forward-propagate the solved secondary initial state (with the same
    two-body model used to solve it) and check it reproduces the targeted
    TCA state. This validates the solver's own algebra/propagator
    round-trip, NOT Basilisk fidelity -- see
    docs/16-targeting-validation-results.md for the Basilisk-fidelity
    check, which is a separate, more important question this self-check
    can't answer.

    Returns:
        (position_error_m, velocity_error_ms)
    """
    r_check, v_check = propagate_state(scenario.r_sec_t0, scenario.v_sec_t0, time_to_tca_s)
    pos_error = float(np.linalg.norm(r_check - scenario.r_sec_tca_target))
    vel_error = float(np.linalg.norm(v_check - scenario.v_sec_tca_target))
    return pos_error, vel_error


def example_leo_orbit() -> tuple[np.ndarray, np.ndarray]:
    """A representative circular-ish LEO orbit (~500 km altitude, ISS-like
    51.6 deg inclination), for tests and examples. Returns (r0, v0) in
    meters / m/s.
    """
    altitude_m = 500e3
    r_mag = Earth.R.to(u.m).value + altitude_m
    inclination_rad = np.radians(51.6)
    v_circular = np.sqrt(Earth.k.to(u.m**3 / u.s**2).value / r_mag)

    r0 = np.array([r_mag, 0.0, 0.0])
    v0 = np.array(
        [0.0, v_circular * np.cos(inclination_rad), v_circular * np.sin(inclination_rad)]
    )
    return r0, v0
