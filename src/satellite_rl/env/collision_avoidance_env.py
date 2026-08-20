"""The Phase 4 collision-avoidance Gymnasium environment: curriculum
stage 1 (single fixed geometry, fixed schedule) per
docs/03-scenario-design.md and docs/09-episode-design.md.

Wraps bsk_rl's GeneralSatelliteTasking (two satellites: ego + a passive
secondary) following the same thin-subclass pattern bsk_rl's own
SatelliteTasking uses for a single satellite (see docs/17-env-
implementation-notes.md) -- exposing only the ego's action/observation
externally, injecting the secondary's fixed no-op action internally.
"""


import numpy as np
from bsk_rl import GeneralSatelliteTasking
from gymnasium import spaces

from ..scenario.targeting import example_leo_orbit, solve_secondary_initial_state_robust
from .observations import make_collision_pc_fn
from .satellites import EGO_NAME, SECONDARY_NAME, SecondarySatellite, make_ego_satellite_class

UTC_INIT = "2018 SEP 29 21:00:00.000 (UTC)"
DEADZONE_MS = 1e-3  # see docs/07-action-space.md: deadzone for maneuver-count reporting


def _risk_penalty(pc: float, threshold: float) -> float:
    """Nonlinear thresholded risk penalty, per docs/08-reward-function.md:
    roughly linear below the operational Pc threshold, steeper above it.
    """
    if pc <= 0.0:
        return 0.0
    ratio = pc / threshold
    if pc <= threshold:
        return ratio
    return 1.0 + np.log(ratio)


class CollisionAvoidanceEnv(GeneralSatelliteTasking):
    """Single ego satellite deciding whether/when to maneuver against one
    fixed, scripted conjunction, per a fixed sequence of decision points.
    """

    def __init__(
        self,
        miss_distance_m: float = 500.0,
        relative_speed_ms: float = 8000.0,
        orientation_angle_rad: float = 1.0,
        combined_radius_m: float = 10.0,
        sigma_m: float = 200.0,
        schedule_days_before_tca: tuple = (5.0, 3.0, 1.0, 0.5, 0.1, 0.0),
        max_dv_ms: float = 10.0,
        dv_available_init: float = 100.0,
        risk_weight: float = 1.0,
        fuel_weight: float = 0.01,
        disruption_weight: float = 0.05,
        pc_threshold: float = 1e-4,
        targeting_seed: int = 0,
        **kwargs,
    ) -> None:
        """
        Args:
            miss_distance_m, relative_speed_ms, orientation_angle_rad:
                the fixed encounter geometry (curriculum stage 1 -- see
                docs/03-scenario-design.md; stage 2/3 sampling is Phase 5).
            combined_radius_m: fixed combined hard-body radius.
            sigma_m: fixed isotropic position-uncertainty std dev used for
                the Pc observation and terminal reward (see
                observations.make_collision_pc_fn for why isotropic).
            schedule_days_before_tca: fixed decision-point schedule,
                descending, must end at 0.0 (TCA).
            max_dv_ms: per-maneuver Δv bound.
            risk_weight, fuel_weight, disruption_weight, pc_threshold:
                reward weights, see docs/08-reward-function.md.
            targeting_seed: seed for the targeting solver's relative-
                velocity-direction sampling (see scenario/targeting.py) --
                does not change the requested miss_distance/relative_speed,
                only the (otherwise free) encounter-plane orientation
                sampling used internally by the solver.
            **kwargs: passed to GeneralSatelliteTasking (e.g. sim_rate).
        """
        schedule_s = tuple(d * 86400.0 for d in schedule_days_before_tca)
        if schedule_s != tuple(sorted(schedule_s, reverse=True)) or schedule_s[-1] != 0.0:
            raise ValueError("schedule_days_before_tca must be descending and end at 0.0")
        self.schedule_s = schedule_s
        self.risk_weight = risk_weight
        self.fuel_weight = fuel_weight
        self.disruption_weight = disruption_weight
        self.pc_threshold = pc_threshold
        self.max_dv_ms = max_dv_ms

        ego_r0, ego_v0 = example_leo_orbit()
        rng = np.random.default_rng(targeting_seed)
        scenario = solve_secondary_initial_state_robust(
            ego_r0,
            ego_v0,
            schedule_s[0],
            miss_distance_m,
            relative_speed_ms,
            orientation_angle_rad,
            rng,
        )

        ego_satellite_class = make_ego_satellite_class(
            sigma=sigma_m,
            combined_radius=combined_radius_m,
            max_dv=max_dv_ms,
            dv_available_init=dv_available_init,
        )
        self._pc_fn = make_collision_pc_fn(SECONDARY_NAME, sigma_m, combined_radius_m)

        ego = ego_satellite_class(
            EGO_NAME,
            sat_args={
                "rN": list(ego_r0),
                "vN": list(ego_v0),
                "oe": None,
                "utc_init": UTC_INIT,
                "conjunction_radius": combined_radius_m / 2,
                "dv_available_init": dv_available_init,
            },
        )
        secondary = SecondarySatellite(
            SECONDARY_NAME,
            sat_args={
                "rN": list(scenario.r_sec_t0),
                "vN": list(scenario.v_sec_t0),
                "oe": None,
                "utc_init": UTC_INIT,
                "conjunction_radius": combined_radius_m / 2,
            },
        )

        super().__init__(
            satellites=[ego, secondary],
            time_limit=schedule_s[0] + 1000.0,
            terminate_on_time_limit=False,
            failure_penalty=-1000.0,  # safety-net signal for a real ConjunctionDynModel collision, see docs/09
            **kwargs,
        )
        self.schedule_index = 0
        self.cumulative_fuel_used_ms = 0.0
        self.maneuver_count = 0

    @property
    def action_space(self) -> spaces.Box:
        """Δv only (m/s, ego's own Hill/RTN frame) -- duration is env-
        controlled (fixed to the schedule interval), not agent-chosen; a
        deliberate Phase 4 simplification of docs/07-action-space.md's
        original 4D design, noted there.
        """
        return spaces.Box(low=-self.max_dv_ms, high=self.max_dv_ms, shape=(3,), dtype=np.float32)

    @property
    def observation_space(self) -> spaces.Box:
        if not hasattr(self, "simulator"):
            self.reset(seed=self.seed)
        return self.satellites[0].observation_space

    def reset(self, seed: int | None = None, options=None):
        self.schedule_index = 0
        self.cumulative_fuel_used_ms = 0.0
        self.maneuver_count = 0
        # self.satellites persists across resets (only .dynamics/.fsw get
        # rebuilt) -- see docs/17-env-implementation-notes.md.
        self.satellites[0]._time_to_tca_s = self.schedule_s[0]
        tuple_obs, info = super().reset(seed=seed, options=options)
        return tuple_obs[0], info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -self.max_dv_ms, self.max_dv_ms)

        current_idx = self.schedule_index
        next_idx = current_idx + 1
        duration_s = self.schedule_s[current_idx] - self.schedule_s[next_idx]
        full_ego_action = np.concatenate([action, [duration_s]]).astype(np.float32)

        fuel_before = self.satellites[0].fsw.dv_available
        self.satellites[0]._time_to_tca_s = self.schedule_s[next_idx]
        self.schedule_index = next_idx
        is_final_step = self.schedule_index == len(self.schedule_s) - 1

        tuple_obs, base_reward, terminated, truncated, info = super().step(
            [full_ego_action, 0]
        )

        fuel_after = self.satellites[0].fsw.dv_available
        dv_used_ms = max(0.0, fuel_before - fuel_after)
        self.cumulative_fuel_used_ms += dv_used_ms
        maneuvered = dv_used_ms > DEADZONE_MS
        if maneuvered:
            self.maneuver_count += 1

        reward = base_reward
        reward -= self.fuel_weight * dv_used_ms
        reward -= self.disruption_weight * float(maneuvered)

        if is_final_step and not terminated:
            pc_final = self._pc_fn(self.satellites[0])
            info["pc_final"] = pc_final
            reward -= self.risk_weight * _risk_penalty(pc_final, self.pc_threshold)
            terminated = True

        info["cumulative_fuel_used_ms"] = self.cumulative_fuel_used_ms
        info["maneuver_count"] = self.maneuver_count

        return tuple_obs[0], reward, terminated, truncated, info
