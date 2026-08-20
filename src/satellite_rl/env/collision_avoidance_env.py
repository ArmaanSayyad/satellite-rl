"""The collision-avoidance Gymnasium environment.

Three modes, curriculum stages 1-3 per docs/03-scenario-design.md and
docs/09-episode-design.md:
- `sample_geometry=False` (default, Phase 4): a single fixed encounter
  geometry and fixed decision-point schedule, constant across all
  episodes.
- `sample_geometry=True, evolve_uncertainty=False` (Phase 5c, curriculum
  stage 2, see docs/19-curriculum-stage-2.md): a fresh geometry -- miss
  distance, relative speed, sigma, combined radius -- sampled from the
  real Kelvins-derived bootstrap table every reset, but still on the
  fixed decision-point schedule and with sigma constant within an episode.
- `sample_geometry=True, evolve_uncertainty=True` (Phase 5d, curriculum
  stage 3, see docs/20-curriculum-stage-3.md): additionally samples a
  real per-event CDM-timing schedule (irregular, variable length) and
  lets sigma evolve within the episode (geometric interpolation between
  the sampled event's real first-CDM and last-CDM covariance magnitude)
  -- the actual v1 target environment per docs/03.

Wraps bsk_rl's GeneralSatelliteTasking (two satellites: ego + a passive
secondary) following the same thin-subclass pattern bsk_rl's own
SatelliteTasking uses for a single satellite (see docs/17-env-
implementation-notes.md) -- exposing only the ego's action/observation
externally, injecting the secondary's fixed no-op action internally.
"""

import json

import numpy as np
import pandas as pd
from bsk_rl import GeneralSatelliteTasking
from gymnasium import spaces

from ..scenario.distributions import FITTED_DIR
from ..scenario.targeting import example_leo_orbit, solve_secondary_initial_state_robust
from .observations import make_collision_pc_fn
from .satellites import EGO_NAME, SECONDARY_NAME, SecondarySatellite, make_ego_satellite_class
from .scenario_sampling import SecondaryScenarioSampler

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
    """Single ego satellite deciding whether/when to maneuver against a
    scripted conjunction, per a sequence of decision points.
    """

    def __init__(
        self,
        sample_geometry: bool = False,
        evolve_uncertainty: bool = False,
        miss_distance_m: float = 500.0,
        relative_speed_ms: float = 8000.0,
        orientation_angle_rad: float = 1.0,
        combined_radius_m: float = 10.0,
        sigma_m: float = 200.0,
        conjunction_radius_m: float = 10.0,
        schedule_days_before_tca: tuple = (5.0, 3.0, 1.0, 0.5, 0.1, 0.0),
        max_dv_ms: float = 10.0,
        dv_available_init: float = 100.0,
        risk_weight: float = 1.0,
        fuel_weight: float = 0.01,
        disruption_weight: float = 0.05,
        pc_threshold: float = 1e-4,
        targeting_seed: int = 0,
        high_risk_fraction: float = 0.0,
        high_risk_pool_fraction: float = 0.05,
        **kwargs,
    ) -> None:
        """
        Args:
            sample_geometry: if True, sample a fresh (miss_distance,
                relative_speed, sigma, combined_radius) from the real
                Kelvins-derived bootstrap table
                (data/fitted/geometry_events.csv) every reset (curriculum
                stage 2, docs/19-curriculum-stage-2.md), ignoring
                miss_distance_m/relative_speed_ms/combined_radius_m/
                sigma_m below. If False (default), those fixed values are
                used for every episode (curriculum stage 1, Phase 4).
            evolve_uncertainty: if True (requires `sample_geometry=True`),
                additionally sample a real per-event CDM-timing schedule
                (ignoring `schedule_days_before_tca`) and let sigma evolve
                within the episode via geometric interpolation between the
                sampled event's real first/last covariance magnitude
                (curriculum stage 3, docs/20-curriculum-stage-3.md).
            miss_distance_m, relative_speed_ms, orientation_angle_rad:
                the fixed encounter geometry, used only when
                `sample_geometry=False`.
            combined_radius_m: fixed combined hard-body radius for the Pc
                observation/reward, used only when `sample_geometry=False`
                (with sampling, this comes from the sampled event instead).
            sigma_m: fixed isotropic position-uncertainty std dev used for
                the Pc observation and terminal reward (see
                observations.make_collision_pc_fn for why isotropic),
                used only when `sample_geometry=False`.
            conjunction_radius_m: the ACTUAL physical hard-body radius
                (per satellite, i.e. half the combined value) used for
                bsk_rl's own terminal collision check (`ConjunctionDynModel`)
                -- deliberately kept fixed and independent of the sampled/
                fixed combined_radius_m used for the Pc *risk estimate*,
                since those represent different things (a real physical
                collision threshold vs. our own uncertainty-aware risk
                model's assumed object size).
            schedule_days_before_tca: fixed decision-point schedule,
                descending, must end at 0.0 (TCA). Ignored when
                `evolve_uncertainty=True` (a real schedule is sampled
                instead).
            max_dv_ms: per-maneuver Δv bound.
            risk_weight, fuel_weight, disruption_weight, pc_threshold:
                reward weights, see docs/08-reward-function.md.
            targeting_seed: seed for the targeting solver's relative-
                velocity-direction sampling (see scenario/targeting.py),
                and, when `sample_geometry=True`, also for which real
                event/schedule/covariance pair gets sampled each reset.
            high_risk_fraction, high_risk_pool_fraction: only used when
                `sample_geometry=True`. Probability of drawing the
                geometry row from the elevated-risk pool (top
                `high_risk_pool_fraction` of real events by `native_pc`)
                instead of uniformly from the full real table. Default
                0.0 -- unmodified real-distribution sampling. See
                docs/24-risk-stratified-sampling.md: real actionable-risk
                events are ~1-in-8,672 (docs/23), so uniform sampling
                essentially never exposes training to one; this is a
                deliberate, explicit, tunable departure from the
                unmodified distribution, not a silent one.
            **kwargs: passed to GeneralSatelliteTasking (e.g. sim_rate).
        """
        if evolve_uncertainty and not sample_geometry:
            raise ValueError("evolve_uncertainty=True requires sample_geometry=True")

        schedule_s = tuple(d * 86400.0 for d in schedule_days_before_tca)
        if schedule_s != tuple(sorted(schedule_s, reverse=True)) or schedule_s[-1] != 0.0:
            raise ValueError("schedule_days_before_tca must be descending and end at 0.0")
        self.schedule_s = schedule_s
        self.risk_weight = risk_weight
        self.fuel_weight = fuel_weight
        self.disruption_weight = disruption_weight
        self.pc_threshold = pc_threshold
        self.max_dv_ms = max_dv_ms
        self.sample_geometry = sample_geometry
        self.evolve_uncertainty = evolve_uncertainty

        ego_r0, ego_v0 = example_leo_orbit()
        rng = np.random.default_rng(targeting_seed)

        ego_satellite_class = make_ego_satellite_class(
            max_dv=max_dv_ms,
            dv_available_init=dv_available_init,
        )
        self._pc_fn = make_collision_pc_fn(SECONDARY_NAME)

        ego = ego_satellite_class(
            EGO_NAME,
            sat_args={
                "rN": list(ego_r0),
                "vN": list(ego_v0),
                "oe": None,
                "utc_init": UTC_INIT,
                "conjunction_radius": conjunction_radius_m / 2,
                "dv_available_init": dv_available_init,
            },
        )

        if sample_geometry:
            geometry_df = pd.read_csv(FITTED_DIR / "geometry_events.csv")
            if evolve_uncertainty:
                with open(FITTED_DIR / "schedule_library.json") as f:
                    schedule_library = json.load(f)
                evolution_df = pd.read_csv(FITTED_DIR / "covariance_evolution_events.csv")
                self._sampler = SecondaryScenarioSampler(
                    geometry_df,
                    ego_r0,
                    ego_v0,
                    rng,
                    schedule_library=schedule_library,
                    evolution_df=evolution_df,
                    high_risk_fraction=high_risk_fraction,
                    high_risk_pool_fraction=high_risk_pool_fraction,
                )
            else:
                self._sampler = SecondaryScenarioSampler(
                    geometry_df,
                    ego_r0,
                    ego_v0,
                    rng,
                    nominal_tca_s=schedule_s[0],
                    high_risk_fraction=high_risk_fraction,
                    high_risk_pool_fraction=high_risk_pool_fraction,
                )
            self._fixed_sigma_m = None
            self._fixed_combined_radius_m = None
            secondary_sat_args = {
                "rN": self._sampler.rN,
                "vN": self._sampler.vN,
                "oe": None,
                "utc_init": UTC_INIT,
                "conjunction_radius": conjunction_radius_m / 2,
            }
        else:
            self._sampler = None
            self._fixed_sigma_m = sigma_m
            self._fixed_combined_radius_m = combined_radius_m
            scenario = solve_secondary_initial_state_robust(
                ego_r0,
                ego_v0,
                schedule_s[0],
                miss_distance_m,
                relative_speed_ms,
                orientation_angle_rad,
                rng,
            )
            secondary_sat_args = {
                "rN": list(scenario.r_sec_t0),
                "vN": list(scenario.v_sec_t0),
                "oe": None,
                "utc_init": UTC_INIT,
                "conjunction_radius": conjunction_radius_m / 2,
            }

        secondary = SecondarySatellite(SECONDARY_NAME, sat_args=secondary_sat_args)

        super().__init__(
            satellites=[ego, secondary],
            # A callable: self.schedule_s may change every reset (stage
            # 3's real, variable-length schedules), and this is
            # re-evaluated by bsk_rl's own _randomize_time_limit() inside
            # super().reset() -- AFTER our reset() override has already
            # updated self.schedule_s for the current episode (see below).
            time_limit=lambda: self.schedule_s[0] + 1000.0,
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
        if self._sampler is not None:
            if seed is not None:
                # Explicit seed -> honor the standard Gym contract
                # (reset(seed=X) must be reproducible): reseed the
                # sampler's RNG fresh so the same seed always samples the
                # same scenario. When seed is None (the common case during
                # real training loops), the RNG keeps advancing across
                # resets instead -- that's what curriculum sampling wants
                # (a different real event each episode), and doing so is
                # itself the documented, deliberate part of the design
                # (see docs/19-curriculum-stage-2.md) -- only the "same
                # seed must reproduce" contract needed fixing, not the
                # underlying continued-advancement behavior.
                self._sampler.rng = np.random.default_rng(seed)
            self._sampler.generation += 1
            # Triggers this episode's sample+solve now (idempotent per
            # generation -- see scenario_sampling.py) so schedule_s and
            # _pc_sigma_x/_pc_sigma_z/_pc_combined_radius are set BEFORE
            # super().reset(), which calls _get_obs() (needing them
            # already set) and re-evaluates the time_limit callable
            # (needing self.schedule_s already updated) at its own
            # end/start, before this method gets control back. rN()/vN(),
            # called inside super().reset() via bsk_rl's sat_args
            # callables, then just reuse the same cached scenario.
            if self.evolve_uncertainty:
                self.schedule_s = self._sampler.current_schedule_s
            sigma_x, sigma_z = self._sampler.sigma_xz_at_fraction(0.0)
            self.satellites[0]._pc_sigma_x = sigma_x
            self.satellites[0]._pc_sigma_z = sigma_z
            self.satellites[0]._pc_combined_radius = self._sampler.current_combined_radius
        else:
            # Fixed-scenario mode (curriculum stage 1) has no real-event
            # anisotropy to preserve -- sigma_m is a single deliberately
            # simple isotropic value, per docs/03-scenario-design.md.
            # Setting both axes equal makes observations.py's anisotropic
            # covariance construction reduce to the isotropic case exactly
            # (diag(s, s) embedded via an orthonormal basis is s*I), so no
            # separate code path is needed there for this mode.
            self.satellites[0]._pc_sigma_x = self._fixed_sigma_m
            self.satellites[0]._pc_sigma_z = self._fixed_sigma_m
            self.satellites[0]._pc_combined_radius = self._fixed_combined_radius_m
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
        if self.evolve_uncertainty:
            total_span = self.schedule_s[0]
            fraction = 1.0 if total_span == 0.0 else 1.0 - self.schedule_s[next_idx] / total_span
            sigma_x, sigma_z = self._sampler.sigma_xz_at_fraction(fraction)
            self.satellites[0]._pc_sigma_x = sigma_x
            self.satellites[0]._pc_sigma_z = sigma_z

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
        info["schedule_length"] = len(self.schedule_s)

        return tuple_obs[0], reward, terminated, truncated, info
