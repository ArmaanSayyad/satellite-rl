"""Couples bsk_rl's per-parameter sat_args randomizer callables (see
docs/17-env-implementation-notes.md: `sat_args` values can be functions,
re-evaluated fresh on every `reset()`) to a single, jointly-consistent
scenario sample per episode -- curriculum stages 2 and 3, see
docs/19-curriculum-stage-2.md and docs/20-curriculum-stage-3.md.

bsk_rl evaluates each sat_args callable independently
(`generate_sat_args`: `{k: v() if callable(v) else v for k, v in ...}`),
so naively giving `rN` and `vN` two SEPARATE sampling closures would let
them draw from two different random scenarios. `SecondaryScenarioSampler`
avoids relying on any assumption about which gets called first: both
`rN()` and `vN()` check a generation counter (incremented by the env's
`reset()` before calling `super().reset()`) and (re)sample together the
first time either is called for a new generation, then both return
values from that single cached sample.

Stage 3 (schedule_library/evolution_df provided) additionally samples a
real per-event CDM-timing schedule and a real (first, last) covariance
pair as part of that same cached generation -- both are needed before the
targeting solve even runs (the schedule's total span is the solve's
time_to_tca), so they can't be deferred to separate accessor calls the
way sigma/combined_radius conceptually could.
"""


import numpy as np
import pandas as pd

from ..scenario.targeting import TargetedScenario, solve_secondary_initial_state_robust

MIN_DECISION_INTERVAL_S = 60.0  # merge real CDM timestamps closer together than this


def _clean_schedule(raw_days: list) -> list:
    """Real per-event time_to_tca sequences (days) can have near-duplicate
    timestamps, occasional NEGATIVE values (some real CDMs are issued
    slightly after TCA -- confirmed in docs/14-pc-validation-results.md,
    min observed time_to_tca was -0.15 days), and don't necessarily
    include an exact 0.0 (TCA) point. Drop negative entries (our schedule
    model only covers decision points before TCA -- see
    docs/09-episode-design.md), dedupe/sort descending, merge points
    closer than MIN_DECISION_INTERVAL_S (bsk_rl's ImpulsiveThrustHill
    enforces a minimum drift duration of 2*sim_rate -- real gaps smaller
    than that would silently get stretched, desyncing our own schedule
    bookkeeping from what actually happens), and force the schedule to
    end at exactly 0.0 regardless of the real data's minimum reported
    (non-negative) time_to_tca.
    """
    schedule_s = sorted({d * 86400.0 for d in raw_days if d >= 0.0}, reverse=True)
    if len(schedule_s) < 1 or schedule_s[0] < MIN_DECISION_INTERVAL_S:
        # Degenerate case: every real timestamp for this event was
        # negative/at TCA already, or too close to TCA to leave room for
        # even one decision step -- fall back to a fixed 1-day default
        # rather than a schedule with no room for the agent to act at all.
        return [86400.0, 0.0]
    filtered = [schedule_s[0]]
    for t in schedule_s[1:]:
        if filtered[-1] - t >= MIN_DECISION_INTERVAL_S:
            filtered.append(t)
    if filtered[-1] != 0.0:
        filtered.append(0.0)
    return filtered


class SecondaryScenarioSampler:
    """Samples a fresh conjunction scenario (geometry from the real
    Kelvins-derived bootstrap table, per docs/15-distribution-fitting-
    results.md) each time `generation` is incremented, and exposes it as
    bsk_rl sat_args callables plus the resulting Pc-relevant parameters.
    """

    def __init__(
        self,
        geometry_df: pd.DataFrame,
        ego_r0: np.ndarray,
        ego_v0: np.ndarray,
        rng: np.random.Generator,
        nominal_tca_s: float | None = None,
        schedule_library: list | None = None,
        evolution_df: pd.DataFrame | None = None,
    ) -> None:
        """
        Args:
            nominal_tca_s: fixed total schedule span (stage 2) -- required
                if `schedule_library` is None.
            schedule_library, evolution_df: real per-event CDM-timing
                schedules and (first, last) covariance pairs (stage 3,
                both required together) -- if provided, a fresh schedule
                and covariance pair are sampled each generation instead
                of using `nominal_tca_s`/a constant sigma.
        """
        if (schedule_library is None) != (evolution_df is None):
            raise ValueError("schedule_library and evolution_df must be provided together")
        if schedule_library is None and nominal_tca_s is None:
            raise ValueError("nominal_tca_s is required when schedule_library is not provided")
        self.geometry_df = geometry_df
        self.ego_r0 = ego_r0
        self.ego_v0 = ego_v0
        self.nominal_tca_s = nominal_tca_s
        self.schedule_library = schedule_library
        self.evolution_df = evolution_df
        self.rng = rng
        self.generation = 0
        self._cached_generation: int | None = None
        self._cached_scenario: TargetedScenario | None = None
        self._cached_sample: dict | None = None
        self._cached_schedule_s: list | None = None
        self._cached_evolution: dict | None = None

    def _ensure_current(self) -> None:
        if self._cached_generation == self.generation:
            return

        if self.schedule_library is not None:
            raw_days = self.schedule_library[self.rng.integers(0, len(self.schedule_library))]
            schedule_s = _clean_schedule(raw_days)
        else:
            schedule_s = [self.nominal_tca_s, 0.0] if self.nominal_tca_s != 0.0 else [0.0]
        nominal_tca_s = schedule_s[0]

        row = self.geometry_df.iloc[self.rng.integers(0, len(self.geometry_df))]
        sample = {
            "miss_distance": float(row["miss_distance"]),
            "relative_speed": float(row["relative_speed"]),
            "sigma_x": float(row["sigma_x"]),
            "sigma_z": float(row["sigma_z"]),
            "combined_radius": float(row["combined_radius"]),
        }
        orientation_angle_rad = self.rng.uniform(0, 2 * np.pi)
        scenario = solve_secondary_initial_state_robust(
            self.ego_r0,
            self.ego_v0,
            nominal_tca_s,
            sample["miss_distance"],
            sample["relative_speed"],
            orientation_angle_rad,
            self.rng,
        )

        if self.evolution_df is not None:
            evo_row = self.evolution_df.iloc[self.rng.integers(0, len(self.evolution_df))]
            evolution = {
                "sigma_x_first": float(evo_row["sigma_x_first"]),
                "sigma_z_first": float(evo_row["sigma_z_first"]),
                "sigma_x_last": float(evo_row["sigma_x_last"]),
                "sigma_z_last": float(evo_row["sigma_z_last"]),
            }
        else:
            evolution = None

        self._cached_scenario = scenario
        self._cached_sample = sample
        self._cached_schedule_s = schedule_s
        self._cached_evolution = evolution
        self._cached_generation = self.generation

    def rN(self) -> list:
        """bsk_rl sat_args callable for the secondary's initial position."""
        self._ensure_current()
        return list(self._cached_scenario.r_sec_t0)

    def vN(self) -> list:
        """bsk_rl sat_args callable for the secondary's initial velocity."""
        self._ensure_current()
        return list(self._cached_scenario.v_sec_t0)

    @property
    def current_schedule_s(self) -> list:
        """This episode's decision-point schedule, in seconds before TCA,
        descending, ending at 0.0 -- either a fixed [nominal_tca_s, 0.0]
        (stage 2) or a real, cleaned per-event schedule (stage 3).
        """
        self._ensure_current()
        return list(self._cached_schedule_s)

    @property
    def current_sigma(self) -> float:
        """Isotropic Pc-observation sigma for the current scenario at
        TCA (i.e. the "final"/constant value): the geometric mean of the
        real event's sigma_x/sigma_z. For a constant-uncertainty episode
        (stage 2, or stage 3 without querying sigma_at_fraction), this is
        the value used throughout.
        """
        self._ensure_current()
        return float(np.sqrt(self._cached_sample["sigma_x"] * self._cached_sample["sigma_z"]))

    def sigma_at_fraction(self, fraction: float) -> float:
        """Interpolated isotropic sigma at `fraction` of the way from
        episode start (0.0) to TCA (1.0), via geometric (log-linear)
        interpolation between the sampled event's real first/last
        combined-sigma magnitude -- covariance shrinks multiplicatively
        (median ~8.36x per docs/15), not additively, so linear
        interpolation in sigma itself would be the wrong shape. Falls
        back to the constant `current_sigma` if no evolution data was
        provided (stage 2).
        """
        self._ensure_current()
        if self._cached_evolution is None:
            return self.current_sigma
        fraction = float(np.clip(fraction, 0.0, 1.0))
        mag_first = np.sqrt(
            self._cached_evolution["sigma_x_first"] * self._cached_evolution["sigma_z_first"]
        )
        mag_last = np.sqrt(
            self._cached_evolution["sigma_x_last"] * self._cached_evolution["sigma_z_last"]
        )
        log_sigma = np.log(mag_first) + fraction * (np.log(mag_last) - np.log(mag_first))
        return float(np.exp(log_sigma))

    @property
    def current_combined_radius(self) -> float:
        self._ensure_current()
        return float(self._cached_sample["combined_radius"])

    @property
    def current_sample(self) -> dict:
        """The raw sampled (miss_distance, relative_speed, sigma_x,
        sigma_z, combined_radius) -- for logging/diagnostics.
        """
        self._ensure_current()
        return dict(self._cached_sample)
