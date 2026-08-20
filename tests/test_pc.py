"""Validation tests for the Pc module (docs/04-collision-probability.md,
docs/13-roadmap.md Phase 1).

Validation strategy, in order of trustworthiness -- deliberately NOT based
on "known worked examples" recalled from a paper, since we can't be fully
confident we'd recall such numbers correctly:

1. Exact closed-form (circular case): when sigma_x == sigma_z, Pc has an
   exact closed form via the noncentral chi-squared CDF (a standard,
   independently-derivable result: R^2/sigma^2 ~ noncentral chi2(df=2,
   nc=R0^2/sigma^2) for an offset circular bivariate Gaussian). This is
   ground truth, computed via a completely different method (scipy's
   noncentral chi-squared CDF) than Foster's numerical quadrature.
2. Monte Carlo cross-check (general elliptical case): sample directly from
   the offset Gaussian and count the fraction landing inside the HBR disk.
   Independent of the quadrature method, so agreement is a real check, not
   circular reasoning.
3. Chan vs. Foster agreement, across the same test geometries -- Chan
   makes an additional disk-approximation Foster doesn't, so exact
   agreement isn't guaranteed; we measure and report the actual deviation
   rather than asserting a specific tolerance a priori.
4. Boundary/sanity properties that must hold for any correct Pc
   implementation regardless of method.
"""

import numpy as np
import pytest
from scipy.stats import ncx2

from satellite_rl.pc import compute_pc
from satellite_rl.pc.chan import pc_chan
from satellite_rl.pc.foster import pc_foster
from satellite_rl.pc.geometry import EncounterGeometry2D


def exact_circular_pc(x0: float, z0: float, sigma: float, hbr: float) -> float:
    """Exact Pc for the circular (sigma_x == sigma_z == sigma) case via the
    noncentral chi-squared CDF. Independent of both Foster's and Chan's
    implementations -- this is the ground truth for this special case.
    """
    r0_sq_over_sigma_sq = (x0**2 + z0**2) / sigma**2
    hbr_sq_over_sigma_sq = hbr**2 / sigma**2
    return float(ncx2.cdf(hbr_sq_over_sigma_sq, df=2, nc=r0_sq_over_sigma_sq))


CIRCULAR_CASES = [
    (0.0, 0.0, 100.0, 10.0),  # dead-on hit, small HBR relative to sigma
    (0.0, 0.0, 100.0, 200.0),  # dead-on hit, HBR bigger than sigma
    (150.0, 0.0, 100.0, 10.0),  # offset miss, small HBR
    (300.0, 400.0, 200.0, 20.0),  # larger offset (miss distance 500 m)
    (50.0, 50.0, 50.0, 15.0),  # offset comparable to sigma
]

ELLIPTICAL_CASES = [
    # x0, z0, sigma_x, sigma_z, hbr
    (100.0, 0.0, 150.0, 60.0, 20.0),
    (0.0, 0.0, 200.0, 80.0, 15.0),
    (250.0, -100.0, 300.0, 100.0, 25.0),
    (50.0, 30.0, 40.0, 40.0, 12.0),
]


@pytest.mark.parametrize("x0,z0,sigma,hbr", CIRCULAR_CASES)
def test_foster_matches_exact_circular_case(x0, z0, sigma, hbr):
    geometry = EncounterGeometry2D(x0=x0, z0=z0, sigma_x=sigma, sigma_z=sigma)
    pc_numeric = pc_foster(geometry, hbr)
    pc_exact = exact_circular_pc(x0, z0, sigma, hbr)
    assert pc_numeric == pytest.approx(pc_exact, abs=1e-6)


@pytest.mark.parametrize("x0,z0,sigma,hbr", CIRCULAR_CASES)
def test_chan_matches_exact_circular_case(x0, z0, sigma, hbr):
    geometry = EncounterGeometry2D(x0=x0, z0=z0, sigma_x=sigma, sigma_z=sigma)
    pc_chan_val = pc_chan(geometry, hbr)
    pc_exact = exact_circular_pc(x0, z0, sigma, hbr)
    # Chan's method makes an additional disk-approximation of the collision
    # region even in the circular case where that approximation is exact,
    # so this should agree tightly -- looser tolerance than Foster's only
    # because it's a truncated series, not because of a modeling difference.
    assert pc_chan_val == pytest.approx(pc_exact, abs=1e-4)


@pytest.mark.parametrize("seed,case", list(enumerate(ELLIPTICAL_CASES)))
def test_foster_matches_monte_carlo(seed, case):
    x0, z0, sigma_x, sigma_z, hbr = case
    geometry = EncounterGeometry2D(x0=x0, z0=z0, sigma_x=sigma_x, sigma_z=sigma_z)
    pc_numeric = pc_foster(geometry, hbr)

    rng = np.random.default_rng(seed)
    n_samples = 2_000_000
    samples_u = rng.normal(x0, sigma_x, n_samples)
    samples_w = rng.normal(z0, sigma_z, n_samples)
    inside = (samples_u**2 + samples_w**2) <= hbr**2
    pc_mc = inside.mean()
    se = np.sqrt(pc_mc * (1 - pc_mc) / n_samples) if 0 < pc_mc < 1 else 1.0 / n_samples
    tolerance = max(5 * se, 1e-6)
    assert pc_numeric == pytest.approx(pc_mc, abs=tolerance)


@pytest.mark.parametrize("case", ELLIPTICAL_CASES)
def test_chan_vs_foster_elliptical_deviation(case):
    """Chan's disk approximation vs. Foster's direct integration, on
    genuinely elliptical (sigma_x != sigma_z) cases where they're not
    mathematically forced to agree. We assert a generous bound and print
    the actual deviation -- the real point of this test is to have a
    number on record, not to rubber-stamp agreement.
    """
    x0, z0, sigma_x, sigma_z, hbr = case
    geometry = EncounterGeometry2D(x0=x0, z0=z0, sigma_x=sigma_x, sigma_z=sigma_z)
    pc_f = pc_foster(geometry, hbr)
    pc_c = pc_chan(geometry, hbr)
    deviation = abs(pc_f - pc_c)
    print(f"case={case} foster={pc_f:.6g} chan={pc_c:.6g} abs_dev={deviation:.3g}")
    # Generous bound for Phase 1 -- tighten once we have a documented
    # sense of typical deviation magnitude across the realistic parameter
    # range (see docs/04-collision-probability.md).
    assert deviation < 0.05


def test_pc_zero_radius_is_zero():
    geometry = EncounterGeometry2D(x0=100.0, z0=0.0, sigma_x=100.0, sigma_z=100.0)
    assert pc_foster(geometry, 0.0) == 0.0
    assert pc_chan(geometry, 0.0) == 0.0


def test_pc_increases_with_hbr():
    geometry = EncounterGeometry2D(x0=50.0, z0=0.0, sigma_x=100.0, sigma_z=100.0)
    pcs = [pc_foster(geometry, hbr) for hbr in [1.0, 10.0, 50.0, 200.0]]
    assert pcs == sorted(pcs)
    # HBR=200 with sigma=100, offset=50: computed value is ~0.83 (not ~1 --
    # a disk of radius 2*sigma centered near the mean still excludes real
    # Gaussian tail mass, so don't over-assert here; the real check is
    # monotonicity above and the exact/Monte-Carlo tests elsewhere).
    assert pcs[-1] > 0.75


def test_pc_bounded_in_unit_interval():
    geometry = EncounterGeometry2D(x0=0.0, z0=0.0, sigma_x=10.0, sigma_z=10.0)
    assert 0.0 <= pc_foster(geometry, 1000.0) <= 1.0
    assert 0.0 <= pc_chan(geometry, 1000.0) <= 1.0


def test_compute_pc_end_to_end():
    """Sanity check the full pipeline: 3D relative state + combined 3x3
    covariance -> encounter-plane projection -> Pc, with a simple,
    hand-constructible geometry (relative velocity along x, so the
    encounter plane is the y-z plane, and the miss vector is purely in y).
    """
    r_rel = np.array([0.0, 120.0, 0.0])  # 120 m miss distance, all in y
    v_rel = np.array([7500.0, 0.0, 0.0])  # typical LEO relative speed, m/s
    cov_combined = np.diag([500.0**2, 80.0**2, 80.0**2])  # along-track irrelevant (projected out)
    pc = compute_pc(r_rel, v_rel, cov_combined, combined_radius=20.0)
    assert 0.0 < pc < 1.0

    pc_chan_val = compute_pc(r_rel, v_rel, cov_combined, combined_radius=20.0, method="chan")
    assert 0.0 < pc_chan_val < 1.0
    assert pc_chan_val == pytest.approx(pc, abs=0.01)


def test_compute_pc_rejects_unknown_method():
    r_rel = np.array([0.0, 120.0, 0.0])
    v_rel = np.array([7500.0, 0.0, 0.0])
    cov_combined = np.diag([500.0**2, 80.0**2, 80.0**2])
    with pytest.raises(ValueError):
        compute_pc(r_rel, v_rel, cov_combined, combined_radius=20.0, method="nonexistent")
