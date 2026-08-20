"""Tests for satellite_rl.scenario.targeting.

Covers the self-consistency validation this phase can make reliable and
fast (pure two-body, no Basilisk needed -- CI-friendly). The separate
question of Basilisk-fidelity drift is NOT covered here -- see
docs/16-targeting-validation-results.md for why that's deferred to
Phase 4, and scripts/validate_targeting_against_basilisk.py for the
current (unresolved, clearly flagged) state of that investigation.
"""

import numpy as np
import pytest

from satellite_rl.scenario.targeting import (
    example_leo_orbit,
    solve_secondary_initial_state,
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

    scenario = solve_secondary_initial_state(
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
    validation" the roadmap calls for, for the two-body propagator layer
    (Basilisk-fidelity validation is separate, see module docstring).
    """
    rng = np.random.default_rng(seed)
    ego_r0, ego_v0 = example_leo_orbit()
    miss_distance = rng.uniform(20.0, 50_000.0)
    relative_speed = rng.uniform(100.0, 15_000.0)
    orientation = rng.uniform(0, 2 * np.pi)

    scenario = solve_secondary_initial_state(
        ego_r0, ego_v0, TIME_TO_TCA_S, miss_distance, relative_speed, orientation, rng
    )
    pos_error_m, vel_error_ms = validate_self_consistency(scenario, TIME_TO_TCA_S)

    # Empirically (Phase 3, docs/16-targeting-validation-results.md, 200
    # trials): median error is ~1e-6 m (floating point), but cases where
    # the sampled relative speed pushes the secondary's TCA state onto a
    # HYPERBOLIC orbit (ecc > 1) show real, explained precision loss --
    # up to ~170 m observed, p99 ~106 m -- from hapsira's Kepler solver
    # being less precise for high eccentricity. 500m is a deliberately
    # generous bound to avoid flaky failures on that real tail, not a
    # tight one; the actual distribution is reported in the results doc.
    assert pos_error_m < 500.0
    assert vel_error_ms < 1.0


def test_example_leo_orbit_is_physically_reasonable():
    r0, v0 = example_leo_orbit()
    altitude_km = (np.linalg.norm(r0) - 6378136.6) / 1000.0
    speed_kms = np.linalg.norm(v0) / 1000.0
    assert 400 < altitude_km < 600  # roughly ISS-altitude LEO
    assert 7.0 < speed_kms < 8.0  # roughly circular LEO orbital speed
