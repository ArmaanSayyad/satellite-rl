"""Observation helper functions for the collision-avoidance environment.

See docs/06-state-space.md for the design and
docs/17-env-implementation-notes.md / docs/19-curriculum-stage-2.md for
Phase 4/5 implementation notes.
"""

import numpy as np

from ..pc import compute_pc
from ..pc.geometry import encounter_plane_basis
from ..scenario.targeting import propagate_state


def make_collision_pc_fn(secondary_name: str):
    """Build a SatProperties `fn` callback that computes Pc from the
    PREDICTED encounter at TCA, not the instantaneous current separation.

    Real conjunction risk is about where two objects will be at their
    predicted closest approach, not where they are right now -- so this
    forward-propagates both satellites' current true state to the
    remaining time-to-TCA (via the same J2-aware propagator used for
    scenario targeting, see scenario/targeting.py) before computing Pc.
    This reuses Phase 3's propagator directly, tying scenario generation
    and in-episode risk assessment to the same physics model.

    Args:
        secondary_name: name of the (passive) secondary satellite.

    Reads `sigma_x`/`sigma_z` and `combined_radius` from the satellite's
    own `_pc_sigma_x`/`_pc_sigma_z`/`_pc_combined_radius` attributes (set
    by the env wrapper at reset/step time) rather than taking them as
    fixed arguments here -- curriculum stage 2 (docs/19-curriculum-
    stage-2.md) samples a fresh scenario, with its own sigma/
    combined_radius, every episode, so these can no longer be baked in as
    constants at class-creation time the way Phase 4's fixed-scenario
    design did.

    The covariance is built anisotropically, aligned to
    `encounter_plane_basis(v_rel)` at the SAME basis convention
    `scenario/targeting.py`'s `solve_secondary_initial_state` used to
    place the miss vector at scenario-generation time (both live in
    pc/geometry.py) -- e1 (`sigma_x`, the tight axis) and e2 (`sigma_z`,
    the loose axis). Before docs/23-anisotropic-covariance-fix.md, this
    used a single isotropic sigma (geometric mean of sigma_x/sigma_z),
    justified as "looks the same under any encounter-plane projection" --
    checked directly and found false when eccentricity is large (real
    median sigma_z/sigma_x ~5.8x, docs/22-evaluation-results.md): the
    isotropic simplification suppressed computed Pc far below what the
    real anisotropic geometry implies, which is why curriculum stage 2/3
    training/eval could never produce a genuinely high-risk episode
    (docs/22's central finding). Using the fixed-scenario mode's single
    `sigma_m` still works unchanged here -- the env wrapper sets
    `_pc_sigma_x == _pc_sigma_z == sigma_m` for that mode, which reduces
    this construction to the isotropic case exactly, since diag(s, s)
    embedded via an orthonormal basis is just s*I.

    Requires the satellite passed to the returned function to also have a
    `_time_to_tca_s` attribute (set by the env wrapper each step) giving
    the remaining time to the scenario's TCA, in seconds.
    """

    def _compute_pc(satellite) -> float:
        secondary = satellite.simulator.get_satellite(secondary_name)
        r_ego = np.array(satellite.dynamics.r_BN_N)
        v_ego = np.array(satellite.dynamics.v_BN_N)
        r_sec = np.array(secondary.dynamics.r_BN_N)
        v_sec = np.array(secondary.dynamics.v_BN_N)
        dt = getattr(satellite, "_time_to_tca_s", 0.0)
        sigma_x = satellite._pc_sigma_x
        sigma_z = satellite._pc_sigma_z
        combined_radius = satellite._pc_combined_radius

        try:
            r_ego_p, v_ego_p = propagate_state(r_ego, v_ego, dt)
            r_sec_p, v_sec_p = propagate_state(r_sec, v_sec, dt)
        except RuntimeError:
            # hapsira's Cowell integrator can fail for some geometries
            # (docs/17) -- degrade gracefully to the instantaneous
            # (unpropagated) relative state rather than crashing the
            # training loop. Logged distinctly so degraded-Pc steps are
            # identifiable during training analysis, not silently mixed in.
            satellite.logger.warning(
                "Pc forward-propagation failed, falling back to instantaneous state"
            )
            r_ego_p, v_ego_p = r_ego, v_ego
            r_sec_p, v_sec_p = r_sec, v_sec

        r_rel = r_sec_p - r_ego_p
        v_rel = v_sec_p - v_ego_p
        basis = encounter_plane_basis(v_rel)  # (3, 2): [e1 (tight), e2 (loose)]
        cov_2d = np.diag([sigma_x**2, sigma_z**2])
        cov_combined = basis @ cov_2d @ basis.T
        return compute_pc(r_rel, v_rel, cov_combined, combined_radius, method="chan")

    return _compute_pc
