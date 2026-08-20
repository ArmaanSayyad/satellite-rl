"""Collision probability (Pc) computation for satellite conjunction assessment.

See docs/04-collision-probability.md for the method selection and
validation approach.
"""

import numpy as np

from .chan import pc_chan
from .foster import pc_foster
from .geometry import EncounterGeometry2D, project_to_encounter_plane

__all__ = [
    "EncounterGeometry2D",
    "compute_pc",
    "pc_chan",
    "pc_foster",
    "project_to_encounter_plane",
]

_METHODS = {"foster": pc_foster, "chan": pc_chan}


def compute_pc(
    r_rel: np.ndarray,
    v_rel: np.ndarray,
    cov_combined: np.ndarray,
    combined_radius: float,
    method: str = "foster",
) -> float:
    """Compute the probability of collision for a conjunction at TCA.

    Args:
        r_rel: relative position at TCA (secondary - primary), (3,), meters.
        v_rel: relative velocity at TCA, (3,) m/s (direction only matters).
        cov_combined: combined (summed) 3x3 position covariance of both
            objects, same frame as r_rel/v_rel, meters^2.
        combined_radius: combined hard-body radius (sum of both objects'
            radii), meters.
        method: "foster" (accuracy reference, default) or "chan" (fast
            series approximation -- see docs/10-rl-algorithm.md for when
            to prefer it, e.g. inside a training loop).

    Returns:
        Probability of collision, in [0, 1].
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown Pc method: {method!r}, expected one of {list(_METHODS)}")

    geometry = project_to_encounter_plane(
        np.asarray(r_rel, dtype=float),
        np.asarray(v_rel, dtype=float),
        np.asarray(cov_combined, dtype=float),
    )
    return _METHODS[method](geometry, combined_radius)
