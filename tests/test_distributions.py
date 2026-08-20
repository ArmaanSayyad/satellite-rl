"""Tests for satellite_rl.scenario.distributions.

Uses synthetic data throughout -- the real Kelvins dataset (232 MB) is not
part of the repo or CI (see docs/05-datasets.md), so functions that load it
directly (fit_geometry_distributions, fit_covariance_shrink_ratio,
extract_schedule_library) are exercised manually via
`python -m satellite_rl.scenario.distributions` (see
docs/15-distribution-fitting-results.md for real-data results), not here.
"""

import numpy as np
import pandas as pd
import pytest

from satellite_rl.scenario.distributions import (
    fit_lognormal,
    sample_scenario_geometry,
    sample_scenario_geometry_bootstrap,
)


def test_fit_lognormal_recovers_known_parameters():
    rng = np.random.default_rng(0)
    true_mu, true_sigma = 2.0, 0.5
    samples = rng.lognormal(mean=true_mu, sigma=true_sigma, size=50_000)

    fitted = fit_lognormal(samples)

    assert fitted.mu == pytest.approx(true_mu, abs=0.02)
    assert fitted.sigma == pytest.approx(true_sigma, abs=0.02)
    assert fitted.n_samples == 50_000
    # A correctly-specified lognormal fit on genuinely lognormal data
    # should NOT be rejected by KS at a reasonable significance level --
    # this is the contrapositive check to docs/15's finding that the real
    # Kelvins data does reject: confirms fit_lognormal's fitting/testing
    # machinery itself is correct, not that real data must pass.
    assert fitted.ks_pvalue > 0.01


def test_fit_lognormal_drops_nonpositive_values():
    data = np.array([1.0, 2.0, -5.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    fitted = fit_lognormal(data)
    assert fitted.n_samples == 8  # the -5.0 and 0.0 dropped


def test_sample_scenario_geometry_returns_all_keys():
    rng = np.random.default_rng(0)
    fitted = {
        "miss_distance": fit_lognormal(rng.lognormal(9.0, 1.0, 1000)),
        "relative_speed": fit_lognormal(rng.lognormal(9.0, 1.0, 1000)),
        "sigma_x": fit_lognormal(rng.lognormal(4.0, 1.0, 1000)),
        "sigma_z": fit_lognormal(rng.lognormal(6.0, 1.0, 1000)),
        "combined_radius": fit_lognormal(rng.lognormal(0.0, 0.5, 1000)),
    }
    sample = sample_scenario_geometry(fitted, rng)
    assert set(sample.keys()) == {
        "miss_distance",
        "relative_speed",
        "sigma_x",
        "sigma_z",
        "combined_radius",
    }
    assert all(v > 0 for v in sample.values())


def test_sample_scenario_geometry_bootstrap_returns_a_real_row():
    geometry_df = pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "miss_distance": [100.0, 200.0, 300.0],
            "relative_speed": [1000.0, 2000.0, 3000.0],
            "sigma_x": [10.0, 20.0, 30.0],
            "sigma_z": [15.0, 25.0, 35.0],
            "combined_radius": [1.0, 2.0, 3.0],
        }
    )
    rng = np.random.default_rng(0)
    samples = [sample_scenario_geometry_bootstrap(geometry_df, rng) for _ in range(50)]

    # Every sampled tuple must be EXACTLY one of the three real rows --
    # this is the whole point of bootstrap resampling (preserves real
    # joint correlations) versus independent marginal sampling.
    real_rows = {
        (100.0, 1000.0, 10.0, 15.0, 1.0),
        (200.0, 2000.0, 20.0, 25.0, 2.0),
        (300.0, 3000.0, 30.0, 35.0, 3.0),
    }
    for s in samples:
        key = (
            s["miss_distance"],
            s["relative_speed"],
            s["sigma_x"],
            s["sigma_z"],
            s["combined_radius"],
        )
        assert key in real_rows

    # With 50 draws from 3 rows, seeing only one unique row would be a
    # sign sampling isn't actually varying -- not a hard guarantee, but a
    # reasonable sanity check at this sample size.
    unique_draws = {
        (s["miss_distance"], s["relative_speed"], s["sigma_x"], s["sigma_z"], s["combined_radius"])
        for s in samples
    }
    assert len(unique_draws) > 1
