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
    i * 1e-6), so the top-K-by-pool-rank pool's membership is exactly
    known: the last `k` rows by index. esa_reported_pc is always half of
    native_pc, so it never changes which row wins max(native_pc,
    esa_reported_pc) -- keeps the pool-membership math from the
    native_pc-only design (docs/24) valid for these tests, since Phase 7d
    (docs/25) changed ranking to that max.
    """
    rng = np.random.default_rng(0)
    native_pc = np.arange(n) * 1e-6
    return pd.DataFrame(
        {
            "miss_distance": rng.uniform(100.0, 500.0, n),
            "relative_speed": rng.uniform(1000.0, 5000.0, n),
            "sigma_x": rng.uniform(10.0, 50.0, n),
            "sigma_z": rng.uniform(50.0, 200.0, n),
            "combined_radius": rng.uniform(1.0, 5.0, n),
            "alignment_angle_rad": rng.uniform(0.0, 2 * np.pi, n),
            "native_pc": native_pc,
            "esa_reported_pc": native_pc / 2.0,
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
    # high_risk_augment=False isolates pool SELECTION from augmentation
    # (which deliberately changes native_pc away from the source row's
    # value -- see the augmentation-specific tests below).
    # high_risk_precise_targeting=False: not what this test is about, and
    # leaving it on would make each of the 15 draws below run 1-3 real
    # Basilisk calls (~1.5-3s each) -- see the dedicated, low-iteration-
    # count precise-targeting tests further down instead.
    sampler = _make_sampler(
        high_risk_fraction=1.0,
        high_risk_pool_fraction=0.25,
        high_risk_augment=False,
        high_risk_precise_targeting=False,
    )
    # Pool = top 5 of 20 rows by max(native_pc, esa_reported_pc) -- since
    # esa_reported_pc = native_pc/2 in this fixture, that's equivalent to
    # ranking by native_pc alone: native_pc in [15e-6, 19e-6].
    for gen in range(15):
        sampler.generation = gen
        sampler.rN()
        sample = sampler.current_sample
        assert sample["drawn_from_high_risk_pool"] is True
        assert sample["native_pc"] >= 15e-6 - 1e-12
        assert sample["augmented"] is False


def test_high_risk_fraction_default_is_zero_and_backward_compatible():
    sampler = _make_sampler()  # no high_risk_fraction kwarg
    assert sampler.high_risk_fraction == 0.0
    assert sampler.high_risk_df is None


def test_high_risk_fraction_partial_draws_from_both_pools():
    # high_risk_precise_targeting=False -- 60 draws, half from the pool,
    # would otherwise cost ~30 real Basilisk calls here for no reason;
    # this test is about the draw-source coin flip, not targeting.
    sampler = _make_sampler(
        high_risk_fraction=0.5, high_risk_pool_fraction=0.25, high_risk_precise_targeting=False
    )
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


def test_esa_reported_pc_can_win_pool_ranking():
    # Row n-1 (highest native_pc) is NOT in the pool once esa_reported_pc
    # for an earlier row is made to dominate -- confirms ranking uses
    # max(native_pc, esa_reported_pc), not native_pc alone (docs/25:
    # our own recomputed Pc materially under-counts risk for some real
    # events relative to ESA's own reported assessment).
    df = _make_geometry_df(n=20)
    df.loc[0, "esa_reported_pc"] = 1.0  # dominates every native_pc in the fixture
    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(42)
    sampler = SecondaryScenarioSampler(
        df,
        ego_r0,
        ego_v0,
        rng,
        nominal_tca_s=NOMINAL_TCA_S,
        high_risk_fraction=1.0,
        high_risk_pool_fraction=0.05,  # pool size 1 -- must be exactly row 0
        high_risk_augment=False,
        high_risk_precise_targeting=False,
    )
    sampler.generation = 1
    sampler.rN()
    # Row 0's miss_distance is the unique fingerprint here (rng-generated,
    # not a round number like native_pc), used to confirm which row won.
    assert sampler.current_sample["miss_distance"] == pytest.approx(df.loc[0, "miss_distance"])


def test_high_risk_augment_produces_dissimilar_variants():
    # A single-row pool drawn many times with augmentation on must NOT
    # always reproduce the exact same miss_distance -- otherwise the
    # elevated pool would just be the same handful of geometries repeated
    # (the memorization risk docs/24/25 flag augmentation as fixing).
    df = _make_geometry_df(n=1)
    df.loc[0, ["sigma_x", "sigma_z"]] = [50.0, 200.0]  # sizable, so jitter is visible
    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(7)
    sampler = SecondaryScenarioSampler(
        df,
        ego_r0,
        ego_v0,
        rng,
        nominal_tca_s=NOMINAL_TCA_S,
        high_risk_fraction=1.0,
        high_risk_pool_fraction=1.0,
        high_risk_augment=True,
        high_risk_precise_targeting=False,
    )
    miss_distances = []
    native_pcs = []
    for gen in range(20):
        sampler.generation = gen
        sampler.rN()
        sample = sampler.current_sample
        assert sample["augmented"] is True
        miss_distances.append(sample["miss_distance"])
        native_pcs.append(sample["native_pc"])
    assert len(set(miss_distances)) > 1  # genuinely dissimilar draws
    assert len(set(native_pcs)) > 1  # native_pc actually recomputed, not copied


def test_high_risk_augment_false_reproduces_exact_row():
    df = _make_geometry_df(n=1)
    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(7)
    sampler = SecondaryScenarioSampler(
        df,
        ego_r0,
        ego_v0,
        rng,
        nominal_tca_s=NOMINAL_TCA_S,
        high_risk_fraction=1.0,
        high_risk_pool_fraction=1.0,
        high_risk_augment=False,
        high_risk_precise_targeting=False,
    )
    for gen in range(10):
        sampler.generation = gen
        sampler.rN()
        sample = sampler.current_sample
        assert sample["augmented"] is False
        assert sample["miss_distance"] == pytest.approx(df.loc[0, "miss_distance"])
        assert sample["native_pc"] == pytest.approx(df.loc[0, "native_pc"])


def test_high_risk_precise_targeting_reports_sub_meter_error():
    # docs/26-precise-targeting.md: gated to pool draws (small target miss
    # distances), a couple of real Basilisk calls per draw -- kept to a
    # single generation here since each call costs real wall-clock time.
    df = _make_geometry_df(n=1)
    df.loc[0, "miss_distance"] = 38.0  # small enough for the fix to matter
    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(1010)  # the oscillation edge case from development
    sampler = SecondaryScenarioSampler(
        df,
        ego_r0,
        ego_v0,
        rng,
        nominal_tca_s=NOMINAL_TCA_S,
        high_risk_fraction=1.0,
        high_risk_pool_fraction=1.0,
        high_risk_augment=False,
        high_risk_precise_targeting=True,
    )
    sampler.generation = 1
    sampler.rN()
    sample = sampler.current_sample
    assert sample["precise_targeting_error_m"] is not None
    assert sample["precise_targeting_error_m"] < 1.0


def test_high_risk_precise_targeting_off_reports_none():
    df = _make_geometry_df(n=1)
    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(7)
    sampler = SecondaryScenarioSampler(
        df,
        ego_r0,
        ego_v0,
        rng,
        nominal_tca_s=NOMINAL_TCA_S,
        high_risk_fraction=1.0,
        high_risk_pool_fraction=1.0,
        high_risk_precise_targeting=False,
    )
    sampler.generation = 1
    sampler.rN()
    assert sampler.current_sample["precise_targeting_error_m"] is None


def test_high_risk_precise_targeting_not_applied_to_full_table_draws():
    # Even with high_risk_precise_targeting=True, a draw from the FULL
    # table (not the pool) must never pay the Basilisk-correction cost --
    # gated strictly to drawing_high_risk, per docs/26.
    sampler = _make_sampler(high_risk_fraction=0.0, high_risk_precise_targeting=True)
    sampler.generation = 1
    sampler.rN()
    sample = sampler.current_sample
    assert sample["drawn_from_high_risk_pool"] is False
    assert sample["precise_targeting_error_m"] is None
