"""Couples bsk_rl's per-parameter sat_args randomizer callables (see
docs/17-env-implementation-notes.md: `sat_args` values can be functions,
re-evaluated fresh on every `reset()`) to a single, jointly-consistent
scenario sample per episode -- curriculum stage 2, see
docs/19-curriculum-stage-2.md.

bsk_rl evaluates each sat_args callable independently
(`generate_sat_args`: `{k: v() if callable(v) else v for k, v in ...}`),
so naively giving `rN` and `vN` two SEPARATE sampling closures would let
them draw from two different random scenarios. `SecondaryScenarioSampler`
avoids relying on any assumption about which gets called first: both
`rN()` and `vN()` check a generation counter (incremented by the env's
`reset()` before calling `super().reset()`) and (re)sample together the
first time either is called for a new generation, then both return
values from that single cached sample.
"""


import numpy as np
import pandas as pd

from ..scenario.targeting import TargetedScenario, solve_secondary_initial_state_robust


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
        nominal_tca_s: float,
        rng: np.random.Generator,
    ) -> None:
        self.geometry_df = geometry_df
        self.ego_r0 = ego_r0
        self.ego_v0 = ego_v0
        self.nominal_tca_s = nominal_tca_s
        self.rng = rng
        self.generation = 0
        self._cached_generation: int | None = None
        self._cached_scenario: TargetedScenario | None = None
        self._cached_sample: dict | None = None

    def _ensure_current(self) -> None:
        if self._cached_generation == self.generation:
            return
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
            self.nominal_tca_s,
            sample["miss_distance"],
            sample["relative_speed"],
            orientation_angle_rad,
            self.rng,
        )
        self._cached_scenario = scenario
        self._cached_sample = sample
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
    def current_sigma(self) -> float:
        """Isotropic Pc-observation sigma for the current scenario: the
        geometric mean of the real event's sigma_x/sigma_z -- a scalar
        summary consistent with the isotropic simplification documented
        in observations.make_collision_pc_fn (full anisotropic covariance
        remains curriculum stage 3 scope).
        """
        self._ensure_current()
        return float(np.sqrt(self._cached_sample["sigma_x"] * self._cached_sample["sigma_z"]))

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
