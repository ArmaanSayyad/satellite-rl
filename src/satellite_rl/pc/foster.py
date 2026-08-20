"""Foster (1992) numerical double-integral method for 2D collision probability.

See docs/04-collision-probability.md for the derivation. Integrates the
offset bivariate Gaussian density over a disk of radius `combined_radius`
(the combined hard-body radius), centered at the origin of the
encounter-plane coordinate system (zero relative separation = an actual
collision), in polar coordinates. This is the accuracy reference -- no
disk/ellipse approximation beyond the standard short-term-encounter model
itself (see geometry.py).
"""

import numpy as np
from scipy import integrate

from .geometry import EncounterGeometry2D


def pc_foster(
    geometry: EncounterGeometry2D,
    combined_radius: float,
    epsabs: float = 1e-10,
    epsrel: float = 1e-8,
) -> float:
    """Compute Pc via direct numerical integration of the offset bivariate
    Gaussian over the hard-body-radius disk.

    Args:
        geometry: encounter-plane geometry (principal-axis frame).
        combined_radius: combined hard-body radius (primary + secondary), m.
        epsabs, epsrel: passed through to scipy.integrate.dblquad.

    Returns:
        Probability of collision, in [0, 1].
    """
    if combined_radius <= 0:
        return 0.0

    x0, z0 = geometry.x0, geometry.z0
    sx, sz = geometry.sigma_x, geometry.sigma_z
    norm_const = 1.0 / (2.0 * np.pi * sx * sz)

    def integrand(r: float, theta: float) -> float:
        u = r * np.cos(theta)
        w = r * np.sin(theta)
        exponent = -0.5 * (((u - x0) / sx) ** 2 + ((w - z0) / sz) ** 2)
        return norm_const * np.exp(exponent) * r

    pc, _abserr = integrate.dblquad(
        integrand,
        0.0,
        2.0 * np.pi,
        lambda _theta: 0.0,
        lambda _theta: combined_radius,
        epsabs=epsabs,
        epsrel=epsrel,
    )
    return float(np.clip(pc, 0.0, 1.0))
