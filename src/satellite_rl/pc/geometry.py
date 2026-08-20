"""Reduce a 3D conjunction geometry to the 2D encounter-plane representation
used by the Pc computation. See docs/04-collision-probability.md.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncounterGeometry2D:
    """Encounter-plane geometry in the principal-axis (diagonalized) frame."""

    x0: float  # miss-vector component along principal axis 1, meters
    z0: float  # miss-vector component along principal axis 2, meters
    sigma_x: float  # std dev along principal axis 1, meters
    sigma_z: float  # std dev along principal axis 2, meters


def encounter_plane_basis(v_rel: np.ndarray) -> np.ndarray:
    """Return a 3x2 matrix [e1 e2] spanning the plane normal to v_rel.

    The encounter plane is defined, per the standard short-term-encounter
    assumption, as the plane through the miss vector normal to the relative
    velocity direction (linear relative motion near TCA).
    """
    v_hat = v_rel / np.linalg.norm(v_rel)
    reference = np.array([0.0, 0.0, 1.0]) if abs(v_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(v_hat, reference)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(v_hat, e1)
    return np.column_stack([e1, e2])


def project_to_encounter_plane(
    r_rel: np.ndarray,
    v_rel: np.ndarray,
    cov_combined: np.ndarray,
) -> EncounterGeometry2D:
    """Project a 3D relative position + combined 3x3 position covariance onto
    the 2D encounter plane, then diagonalize to the principal-axis frame.

    Args:
        r_rel: relative position at TCA (secondary minus primary), (3,), m.
        v_rel: relative velocity at TCA, (3,) m/s. Only its direction is used.
        cov_combined: combined (summed) 3x3 position covariance of both
            objects, in the same frame as r_rel/v_rel, m^2.
    """
    basis = encounter_plane_basis(v_rel)  # (3, 2)
    miss_2d = basis.T @ r_rel  # (2,)
    cov_2d = basis.T @ cov_combined @ basis  # (2, 2)

    eigvals, eigvecs = np.linalg.eigh(cov_2d)  # ascending order, symmetric matrix
    if np.any(eigvals <= 0):
        raise ValueError(
            f"Encounter-plane covariance is not positive definite: eigenvalues {eigvals}"
        )
    sigma = np.sqrt(eigvals)
    miss_principal = eigvecs.T @ miss_2d

    return EncounterGeometry2D(
        x0=float(miss_principal[0]),
        z0=float(miss_principal[1]),
        sigma_x=float(sigma[0]),
        sigma_z=float(sigma[1]),
    )
