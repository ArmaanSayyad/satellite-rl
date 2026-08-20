"""Tests for satellite_rl.env.scenario_sampling.

SecondaryScenarioSampler itself only needs scenario/targeting.py
(hapsira-based, CI-friendly, see test_targeting.py's docstring) -- but
importing anything from `satellite_rl.env` runs that package's
`__init__.py`, which imports bsk_rl regardless of which submodule was
actually requested. Skipped via `pytest.importorskip` in the lightweight
CI environment, same as test_env.py, rather than fighting the package
structure for this test file alone. Uses a small synthetic geometry_df
rather than the full real 8,672-row table, to keep runtime reasonable and
the elevated-risk pool's membership exactly known.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("bsk_rl")

from satellite_rl.env.scenario_sampling import SecondaryScenarioSampler
from satellite_rl.scenario.targeting import example_leo_orbit

NOMINAL_TCA_S = 0.2 * 86400.0  # matches curriculum stage 2's shortest schedule


def _make_geometry_df(n: int = 20) -> pd.DataFrame:
    """n rows with strictly increasing native_pc (row i has native_pc =
    i * 1e-6), so the top-K-by-native_pc pool's membership is exactly
    known: the last `k` rows by index.
    """
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "miss_distance": rng.uniform(100.0, 500.0, n),
            "relative_speed": rng.uniform(1000.0, 5000.0, n),
            "sigma_x": rng.uniform(10.0, 50.0, n),
            "sigma_z": rng.uniform(50.0, 200.0, n),
            "combined_radius": rng.uniform(1.0, 5.0, n),
            "alignment_angle_rad": rng.uniform(0.0, 2 * np.pi, n),
            "native_pc": np.arange(n) * 1e-6,
        }
    )


def _make_sampler(**kwargs) -> SecondaryScenarioSampler:
    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(42)
    return SecondaryScenarioSampler(
        _make_geometry_df(), ego_r0, ego_v0, rng, nominal_tca_s=NOMINAL_TCA_S, **kwargs
    )


def test_high_risk_fraction_zero_never_draws_from_pool():
    sampler = _make_sampler(high_risk_fraction=0.0)
    for gen in range(15):
        sampler.generation = gen
        sampler.rN()  # triggers _ensure_current
        assert sampler.current_sample["drawn_from_high_risk_pool"] is False


def test_high_risk_fraction_one_always_draws_from_pool():
    sampler = _make_sampler(high_risk_fraction=1.0, high_risk_pool_fraction=0.25)
    # Pool = top 5 of 20 rows by native_pc -- native_pc in [15e-6, 19e-6].
    for gen in range(15):
        sampler.generation = gen
        sampler.rN()
        sample = sampler.current_sample
        assert sample["drawn_from_high_risk_pool"] is True
        assert sample["native_pc"] >= 15e-6 - 1e-12


def test_high_risk_fraction_default_is_zero_and_backward_compatible():
    sampler = _make_sampler()  # no high_risk_fraction kwarg
    assert sampler.high_risk_fraction == 0.0
    assert sampler.high_risk_df is None


def test_high_risk_fraction_partial_draws_from_both_pools():
    sampler = _make_sampler(high_risk_fraction=0.5, high_risk_pool_fraction=0.25)
    seen_high_risk = set()
    for gen in range(60):
        sampler.generation = gen
        sampler.rN()
        seen_high_risk.add(sampler.current_sample["drawn_from_high_risk_pool"])
    # With 60 independent 50/50 draws, seeing only one outcome would be a
    # ~2*(0.5)^60 event -- not a hard guarantee, but this isn't that.
    assert seen_high_risk == {True, False}


def test_high_risk_fraction_out_of_range_rejected():
    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        SecondaryScenarioSampler(
            _make_geometry_df(),
            ego_r0,
            ego_v0,
            rng,
            nominal_tca_s=NOMINAL_TCA_S,
            high_risk_fraction=1.5,
        )
