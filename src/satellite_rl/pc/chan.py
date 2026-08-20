"""Chan (1997) fast series-expansion method for 2D collision probability.

Ported faithfully from Orekit's Chan1997.java
(org.orekit.ssa.collision.shorttermencounter.probability.twod), verified
against the actual Orekit source (github.com/CS-SI/Orekit, Aug 2026) --
not reconstructed from memory. See docs/04-collision-probability.md.

Reference: Chan, K., "Collision Probability Analyses for Earth Orbiting
Satellites," ISCOPS 1997, Nagasaki, Advances in the Astronautical Sciences
96; Chan, F.K. et al., Spacecraft Collision Probability, Aerospace Press,
2008.

Same underlying assumptions as Foster's method (short-encounter linear
relative motion, spherical combined object, diagonalized/uncorrelated
positional covariance, Gaussian uncertainty, deterministic relative
velocity) -- Chan's method additionally approximates the collision region
with a rescaled-coordinate series expansion rather than Foster's direct
quadrature. Treat Foster as the accuracy reference and this as the fast,
training-time-safe approximation (see docs/10-rl-algorithm.md); see
tests/test_pc.py for the measured agreement between the two.
"""

import math

from .geometry import EncounterGeometry2D


def _select_order_m(u: float, v: float) -> int:
    """Number of series terms, per Orekit's Chan1997 thresholds (verbatim)."""
    if u <= 0.01 or v <= 1:
        return 3
    if u <= 1 or v <= 9:
        return 10
    if u <= 25 or v <= 25:
        return 20
    return 60


def pc_chan(geometry: EncounterGeometry2D, combined_radius: float) -> float:
    """Compute Pc via Chan's (1997) fast series expansion.

    Args:
        geometry: encounter-plane geometry (principal-axis frame).
        combined_radius: combined hard-body radius (primary + secondary), m.

    Returns:
        Probability of collision, in [0, 1].
    """
    if combined_radius <= 0:
        return 0.0

    xm, ym = geometry.x0, geometry.z0
    sigma_x, sigma_y = geometry.sigma_x, geometry.sigma_z

    u = combined_radius**2 / (sigma_x * sigma_y)
    v = xm**2 / sigma_x**2 + ym**2 / sigma_y**2
    m = _select_order_m(u, v)

    t = 1.0
    s = 1.0
    running_s = 1.0
    value = math.exp(-v / 2.0) * t - math.exp(-(u + v) / 2.0) * t * running_s

    for i in range(1, m):
        t = (v / 2.0) / i * t
        s = (u / 2.0) / i * s
        running_s += s
        value += math.exp(-v / 2.0) * t - math.exp(-(u + v) / 2.0) * t * running_s

    return float(min(max(value, 0.0), 1.0))
