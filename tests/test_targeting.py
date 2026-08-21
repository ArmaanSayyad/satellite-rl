"""Tests for satellite_rl.scenario.targeting.

Covers the self-consistency validation this module can make reliable and
fast (J2-aware two-body, no Basilisk needed -- CI-friendly), and the
orbit-sanity/integration-failure retry logic. The separate question of
Basilisk-fidelity drift was resolved in Phase 4 (real J2 nodal
precession, not a bug -- see docs/17-env-implementation-notes.md) and the
TCA-timing-sensitivity follow-up is covered in
docs/18-scenario-generator-hardening.md, not here.
"""

import numpy as np
import pytest

from satellite_rl.scenario.targeting import (
    DEFAULT_MAX_APOAPSIS_ALTITUDE_M,
    DEFAULT_MIN_ALTITUDE_M,
    example_leo_orbit,
    osculating_apoapsis_altitude_m,
    osculating_eccentricity,
    osculating_periapsis_altitude_m,
    solve_secondary_initial_state_robust,
    validate_self_consistency,
)

TIME_TO_TCA_S = 3 * 86400.0  # 3 days, representative of a mid-schedule CDM lead time


@pytest.mark.parametrize("seed", range(20))
def test_solver_reproduces_targeted_geometry(seed):
    """The solved secondary state, when the TCA-relative vectors are
    recombined, must reproduce the exact targeted miss distance and
    relative speed -- this is checking the solver's own vector algebra,
    independent of propagation accuracy.
    """
    rng = np.random.default_rng(seed)
    ego_r0, ego_v0 = example_leo_orbit()
    miss_distance = rng.uniform(20.0, 50_000.0)
    relative_speed = rng.uniform(100.0, 15_000.0)
    orientation = rng.uniform(0, 2 * np.pi)

    scenario = solve_secondary_initial_state_robust(
        ego_r0, ego_v0, TIME_TO_TCA_S, miss_distance, relative_speed, orientation, rng
    )

    r_rel = scenario.r_sec_tca_target - scenario.r_ego_tca
    v_rel = scenario.v_sec_tca_target - scenario.v_ego_tca
    assert np.linalg.norm(r_rel) == pytest.approx(miss_distance, rel=1e-9)
    assert np.linalg.norm(v_rel) == pytest.approx(relative_speed, rel=1e-9)
    # TCA definition: relative position must be perpendicular to relative
    # velocity (closest approach), not just the right magnitude.
    cos_angle = np.dot(r_rel, v_rel) / (np.linalg.norm(r_rel) * np.linalg.norm(v_rel))
    assert cos_angle == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("seed", range(20))
def test_self_consistency_batch(seed):
    """Forward-propagating the solved initial state must reproduce the
    targeted TCA state to a tight tolerance -- this is the "batch
    validation" the roadmap calls for, for the (now J2-aware, Phase 4)
    propagator layer. (Basilisk-fidelity validation is separate, see
    module docstring -- now resolved, see docs/17-env-implementation-notes.md.)

    Uses the robust (retrying) solver: hapsira's Cowell integrator fails
    outright (RuntimeError) for a real ~4% of sampled geometries due to a
    hardcoded internal atol -- see docs/17. Since the failing direction is
    itself a free/sampled parameter, retrying with a resampled direction
    is a legitimate fix, not a hidden change to the requested scenario.
    """
    rng = np.random.default_rng(seed)
    ego_r0, ego_v0 = example_leo_orbit()
    miss_distance = rng.uniform(20.0, 50_000.0)
    relative_speed = rng.uniform(100.0, 15_000.0)
    orientation = rng.uniform(0, 2 * np.pi)

    scenario = solve_secondary_initial_state_robust(
        ego_r0, ego_v0, TIME_TO_TCA_S, miss_distance, relative_speed, orientation, rng
    )
    pos_error_m, vel_error_ms = validate_self_consistency(scenario, TIME_TO_TCA_S)

    # Empirically (Phase 4, docs/17-env-implementation-notes.md, 200
    # trials with the J2 propagator): max observed 10.3m / 0.013 m/s,
    # p99 6.4m / 0.011 m/s -- a much tighter, more uniform distribution
    # than the old pure-two-body case (which had a fat tail to ~170m from
    # hyperbolic orbits). 50m/0.1 m/s bounds are generous relative to
    # that, not tight; the actual distribution is in the results doc.
    assert pos_error_m < 50.0
    assert vel_error_ms < 0.1


def test_example_leo_orbit_is_physically_reasonable():
    r0, v0 = example_leo_orbit()
    altitude_km = (np.linalg.norm(r0) - 6378136.6) / 1000.0
    speed_kms = np.linalg.norm(v0) / 1000.0
    assert 400 < altitude_km < 600  # roughly ISS-altitude LEO
    assert 7.0 < speed_kms < 8.0  # roughly circular LEO orbital speed


def test_periapsis_altitude_circular_orbit_equals_own_altitude():
    """For a circular orbit, periapsis = apoapsis = the orbit's own
    (constant) radius -- an independent sanity check of the vis-viva +
    eccentricity-vector formula, not just self-consistency.
    """
    r0, v0 = example_leo_orbit()
    expected_altitude = np.linalg.norm(r0) - 6378136.6
    computed = osculating_periapsis_altitude_m(r0, v0)
    assert computed == pytest.approx(expected_altitude, rel=1e-6)


def test_periapsis_altitude_known_eccentric_orbit():
    """Hand-constructed eccentric orbit: apoapsis velocity of a
    r_a=7000km / r_p=6600km ellipse, checked against the closed-form
    vis-viva periapsis speed -- an independent numerical check, not
    derived from the function under test.
    """
    mu = 3.986004418e14
    r_a, r_p = 7_000_000.0, 6_600_000.0
    a = (r_a + r_p) / 2
    v_at_apoapsis = np.sqrt(mu * (2 / r_a - 1 / a))
    r0 = np.array([r_a, 0.0, 0.0])
    v0 = np.array([0.0, v_at_apoapsis, 0.0])
    computed_altitude = osculating_periapsis_altitude_m(r0, v0)
    expected_altitude = r_p - 6378136.6
    assert computed_altitude == pytest.approx(expected_altitude, rel=1e-6)


def test_robust_solver_rejects_low_periapsis_scenario():
    """The exact parameters that crashed Phase 4's environment with a
    bsk_rl `altitude_valid` failure mid-episode (docs/17) must now yield
    a scenario whose secondary orbit clears the minimum altitude.
    """
    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(0)
    scenario = solve_secondary_initial_state_robust(
        ego_r0,
        ego_v0,
        time_to_tca_s=5 * 86400.0,
        miss_distance_m=500.0,
        relative_speed_ms=8000.0,
        orientation_angle_rad=1.0,
        rng=rng,
    )
    altitude = osculating_periapsis_altitude_m(scenario.r_sec_t0, scenario.v_sec_t0)
    assert altitude >= DEFAULT_MIN_ALTITUDE_M


def test_eccentricity_circular_orbit_is_zero():
    r0, v0 = example_leo_orbit()
    assert osculating_eccentricity(r0, v0) == pytest.approx(0.0, abs=1e-6)


def test_eccentricity_known_eccentric_orbit():
    # Same r_a=7000km/r_p=6600km ellipse as
    # test_periapsis_altitude_known_eccentric_orbit, cross-checked against
    # the standard e = (r_a - r_p) / (r_a + r_p) closed form.
    r_a, r_p = 7_000_000.0, 6_600_000.0
    mu = 3.986004418e14
    a = (r_a + r_p) / 2
    v_at_apoapsis = np.sqrt(mu * (2 / r_a - 1 / a))
    r0 = np.array([r_a, 0.0, 0.0])
    v0 = np.array([0.0, v_at_apoapsis, 0.0])
    expected = (r_a - r_p) / (r_a + r_p)
    assert osculating_eccentricity(r0, v0) == pytest.approx(expected, rel=1e-6)


def test_eccentricity_hyperbolic_orbit_exceeds_one():
    # Escape velocity at this radius is ~10,900 m/s -- well above it is
    # unambiguously hyperbolic.
    r0 = np.array([6_878_136.6, 0.0, 0.0])
    v0 = np.array([0.0, 15_000.0, 0.0])
    assert osculating_eccentricity(r0, v0) > 1.0


def test_robust_solver_rejects_hyperbolic_scenario():
    """docs/26-precise-targeting.md: the exact real-event parameters that
    produced a hyperbolic secondary "orbit" (periapsis altitude looked
    fine at 487km, but eccentricity was 7.03) must now be rejected --
    every returned scenario's secondary orbit must be bound AND within a
    realistic LEO apoapsis ceiling (eccentricity alone wasn't enough, per
    docs/26 -- see the osculating_apoapsis_altitude_m tests below for the
    apoapsis check's own correctness)."""
    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(0)
    scenario = solve_secondary_initial_state_robust(
        ego_r0,
        ego_v0,
        time_to_tca_s=0.2 * 86400.0,
        miss_distance_m=237.0,
        relative_speed_ms=14_919.0,
        orientation_angle_rad=1.0,
        rng=rng,
        max_attempts=3000,
    )
    assert osculating_eccentricity(scenario.r_sec_t0, scenario.v_sec_t0) < 1.0
    assert (
        osculating_apoapsis_altitude_m(scenario.r_sec_t0, scenario.v_sec_t0)
        <= DEFAULT_MAX_APOAPSIS_ALTITUDE_M
    )


def test_apoapsis_altitude_circular_orbit_equals_own_altitude():
    r0, v0 = example_leo_orbit()
    expected_altitude = np.linalg.norm(r0) - 6378136.6
    assert osculating_apoapsis_altitude_m(r0, v0) == pytest.approx(expected_altitude, rel=1e-6)


def test_apoapsis_altitude_known_eccentric_orbit():
    # Same r_a=7000km/r_p=6600km ellipse as the periapsis/eccentricity
    # tests above -- apoapsis should reproduce r_a exactly.
    mu = 3.986004418e14
    r_a, r_p = 7_000_000.0, 6_600_000.0
    a = (r_a + r_p) / 2
    v_at_apoapsis = np.sqrt(mu * (2 / r_a - 1 / a))
    r0 = np.array([r_a, 0.0, 0.0])
    v0 = np.array([0.0, v_at_apoapsis, 0.0])
    computed_altitude = osculating_apoapsis_altitude_m(r0, v0)
    expected_altitude = r_a - 6378136.6
    assert computed_altitude == pytest.approx(expected_altitude, rel=1e-6)


def test_apoapsis_altitude_flags_bound_but_unrealistic_orbit():
    # eccentricity=0.982-like case from docs/26: bound (e<1) with a
    # perfectly fine periapsis, but an apoapsis far beyond any realistic
    # LEO regime -- this is exactly what osculating_eccentricity alone
    # can't catch, and what max_apoapsis_altitude_m exists to reject.
    mu = 3.986004418e14
    r_p = 6_378_136.6 + 400_000.0  # 400km periapsis altitude, fine on its own
    r_a = 73_584_000.0  # near-GEO apoapsis, per the real event found in docs/26
    a = (r_a + r_p) / 2
    v_at_periapsis = np.sqrt(mu * (2 / r_p - 1 / a))
    r0 = np.array([r_p, 0.0, 0.0])
    v0 = np.array([0.0, v_at_periapsis, 0.0])
    assert osculating_eccentricity(r0, v0) < 1.0  # bound -- (c) alone wouldn't catch this
    assert osculating_apoapsis_altitude_m(r0, v0) > DEFAULT_MAX_APOAPSIS_ALTITUDE_M
