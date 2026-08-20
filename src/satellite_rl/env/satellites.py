"""bsk_rl Satellite subclasses for the collision-avoidance environment.

See docs/06-state-space.md, docs/07-action-space.md,
docs/02-bsk_rl-architecture.md for the design, and
docs/17-env-implementation-notes.md for Phase 4 implementation decisions
and the API details verified directly against the installed bsk_rl source
(not assumed from earlier secondhand research).
"""

from typing import ClassVar

from bsk_rl import act, obs, sats
from bsk_rl.sim import dyn, fsw

from .observations import make_collision_pc_fn

EGO_NAME = "Ego"
SECONDARY_NAME = "Secondary"


class SecondarySatellite(sats.Satellite):
    """A passive object (debris / other satellite) on a scripted trajectory.

    Not an RL agent -- per docs/17, bsk_rl's ConjunctionDynModel requires
    both sides of a conjunction check to be full Satellite instances, so
    this exists purely to be propagated and checked for proximity, with a
    trivial single-choice action space (mirrors bsk_rl's own RSO Inspection
    example's pattern for a passive target object).
    """

    # A non-empty observation_spec is required (bsk_rl errors trying to
    # vectorize zero arrays) even though nothing ever reads this
    # satellite's observation -- mirrors bsk_rl's own RSO Inspection
    # example's pattern for a passive, non-agent object.
    observation_spec: ClassVar[list] = [obs.SatProperties({"prop": "one", "fn": lambda _: 1.0})]
    action_spec: ClassVar[list] = [act.Drift(duration=1e9)]
    dyn_type: ClassVar = (dyn.ConjunctionDynModel,)
    fsw_type: ClassVar = ()


def make_ego_satellite_class(
    sigma: float,
    combined_radius: float,
    max_dv: float = 10.0,
    dv_available_init: float = 100.0,
    secondary_name: str = SECONDARY_NAME,
    ego_name: str = EGO_NAME,
) -> type:
    """Build an EgoSatellite subclass configured for one fixed scenario.

    `observation_spec`/`action_spec` are bsk_rl ClassVars (class-level,
    not instance-level) -- see docs/17 -- so per-scenario parameters (the
    secondary's name, the fixed covariance/HBR used for the Pc
    observation) must be baked in at class-definition time via this
    factory rather than passed to `__init__`.

    Args:
        sigma: fixed isotropic position-uncertainty std dev for the Pc
            observation, meters (see observations.make_collision_pc_fn).
        combined_radius: fixed combined hard-body radius, meters.
        max_dv: maximum Δv magnitude per maneuver, m/s.
        secondary_name, ego_name: must match the names used when
            instantiating the two satellites.
    """
    pc_fn = make_collision_pc_fn(secondary_name, sigma, combined_radius)

    def _time_to_tca(satellite) -> float:
        return getattr(satellite, "_time_to_tca_s", 0.0)

    class EgoSatellite(sats.Satellite):
        """The RL-controlled satellite deciding whether/when to maneuver."""

        observation_spec: ClassVar[list] = [
            obs.SatProperties(
                {"prop": "collision_prob", "fn": pc_fn},
                {"prop": "dv_available", "module": "fsw", "norm": dv_available_init},
                {"prop": "time_to_tca", "fn": _time_to_tca},
            ),
            obs.RelativeProperties(
                {"prop": "r_DC_N", "norm": 1e4},
                {"prop": "v_DC_N", "norm": 1e4},
                chief_name=secondary_name,
            ),
        ]
        action_spec: ClassVar[list] = [
            act.ImpulsiveThrustHill(chief_name=ego_name, max_dv=max_dv, max_drift_duration=1e9)
        ]
        dyn_type: ClassVar = (dyn.ConjunctionDynModel,)
        fsw_type: ClassVar = (fsw.MagicOrbitalManeuverFSWModel,)

    return EgoSatellite
