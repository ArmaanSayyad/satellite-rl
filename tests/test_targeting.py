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
