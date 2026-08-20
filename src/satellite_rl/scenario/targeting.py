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
    max_attempts: int = 10,
) -> TargetedScenario:
    """Same as `solve_secondary_initial_state`, but retries with a
    resampled relative-velocity direction on integration failure.

    hapsira's Cowell integrator (`solve_secondary_initial_state`'s J2
    propagation) hardcodes `atol=1e-12` internally regardless of the
    problem's actual state magnitude (km-scale positions/velocities) --
    empirically, this causes `RuntimeError: Integration failed` for a
    real, non-negligible fraction of sampled geometries (~4% observed
    over 50 trials, not exclusively at extreme relative speeds -- see
    docs/17-env-implementation-notes.md). Since the relative-velocity
    DIRECTION is already a free parameter we're sampling (not something
    with a real-world-derived distribution -- see the docstring above),
    resampling it on failure is a legitimate retry, not silently changing
    the requested scenario's actual physical parameters (miss distance
    and relative speed magnitude are preserved exactly across retries).
    """
    last_error = None
    for _attempt in range(max_attempts):
        try:
            return solve_secondary_initial_state(
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
