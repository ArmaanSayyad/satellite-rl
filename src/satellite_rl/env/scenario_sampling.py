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

from ..pc import compute_pc
from ..scenario.targeting import TargetedScenario, solve_secondary_initial_state_robust
from ..scenario.tca_refinement import correct_targeting_geometry

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
        high_risk_fraction: float = 0.0,
        high_risk_pool_fraction: float = 0.05,
        high_risk_augment: bool = True,
        high_risk_precise_targeting: bool = True,
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
            high_risk_fraction: probability, per episode, of drawing the
                geometry row from the elevated-risk pool instead of
                uniformly from the full real table. Default 0.0 --
                unmodified uniform-real-distribution sampling, the
                pre-Phase-7c behavior. See docs/24-risk-stratified-
                sampling.md: real actionable-risk events are ~1-in-8,672
                by our own recomputed `native_pc`, too rare for uniform
                sampling to expose training to one in any practical
                budget. This is a deliberate, explicit, tunable departure
                from the unmodified real distribution -- the elevated
                pool is still 100% real events, just resampled with a
                different (documented, not silent) weighting.
            high_risk_pool_fraction: size of the elevated-risk pool, as a
                fraction of `geometry_df`'s rows -- ranked by
                `max(native_pc, esa_reported_pc)` (not a value threshold,
                which would tie on the majority of rows sitting at
                Pc=0; and not `native_pc` alone, since docs/25-
                augmentation-and-threshold-findings.md found our own
                recomputed Pc materially under-counts risk relative to
                ESA's own reported assessment for several real events --
                using the max of both is a safety-oriented choice: either
                signal suggesting real risk is enough to include a row).
                Only used when `high_risk_fraction > 0`.
            high_risk_augment: when drawing from the elevated-risk pool,
                perturb the drawn row's miss vector by resampling it from
                the SAME Gaussian its own reported covariance defines
                (`N((x0, z0), diag(sigma_x^2, sigma_z^2))`) instead of
                using its exact real (x0, z0) every time. This is not an
                invented perturbation: the covariance IS the real event's
                own statement of "here is the distribution of plausible
                true offsets consistent with this measurement" -- so a
                draw from it is a genuinely different, still-real-
                measurement-consistent encounter, not a synthetic one.
                Without this, the elevated pool -- a few hundred rows at
                most -- would produce the exact same handful of geometries
                every time it's drawn, risking the policy memorizing those
                specific instances rather than learning to generalize.
                Default True; only takes effect when `high_risk_fraction
                > 0`. See docs/25 for the empirical validation (how many
                actionable variants this produces, and how dissimilar).
            high_risk_precise_targeting: when drawing from the elevated-
                risk pool, correct the J2 targeting solver's initial
                state via `tca_refinement.correct_targeting_geometry`
                (1-3 extra Basilisk calls, ~1.5-3s) so the actual
                simulated encounter lands near the intended small miss
                distance instead of ~100-200m off (docs/26-precise-
                targeting.md: this error is invisible at the km-scale
                miss distances every other draw uses, but swamps the
                tens-of-meters targets high-risk real events need).
                Default True; only takes effect when `high_risk_fraction
                > 0`, since that's the only regime where target miss
                distances are small enough for this to matter.
        """
        if (schedule_library is None) != (evolution_df is None):
            raise ValueError("schedule_library and evolution_df must be provided together")
        if schedule_library is None and nominal_tca_s is None:
            raise ValueError("nominal_tca_s is required when schedule_library is not provided")
        if not 0.0 <= high_risk_fraction <= 1.0:
            raise ValueError("high_risk_fraction must be in [0, 1]")
        self.geometry_df = geometry_df
        self.ego_r0 = ego_r0
        self.ego_v0 = ego_v0
        self.nominal_tca_s = nominal_tca_s
        self.schedule_library = schedule_library
        self.evolution_df = evolution_df
        self.rng = rng
        self.generation = 0
        self.high_risk_fraction = high_risk_fraction
        self.high_risk_augment = high_risk_augment
        self.high_risk_precise_targeting = high_risk_precise_targeting
        self.high_risk_df = None
        if high_risk_fraction > 0.0:
            n_top = max(1, int(len(geometry_df) * high_risk_pool_fraction))
            pool_rank = geometry_df[["native_pc", "esa_reported_pc"]].max(axis=1)
            self.high_risk_df = (
                geometry_df.loc[pool_rank.nlargest(n_top).index].reset_index(drop=True)
            )
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

        # Elevated-risk stratified draw (docs/24-risk-stratified-
        # sampling.md): with probability high_risk_fraction, draw from
        # the precomputed top-native_pc pool instead of the full table.
        # Independent per-episode coin flip (not tied to the schedule/
        # evolution draws above), so it composes cleanly with stage 3's
        # separately-sampled schedule/evolution pair.
        drawing_high_risk = self.high_risk_df is not None and self.rng.random() < self.high_risk_fraction
        pool = self.high_risk_df if drawing_high_risk else self.geometry_df
        row = pool.iloc[self.rng.integers(0, len(pool))]
        miss_distance = float(row["miss_distance"])
        alignment_angle_rad = float(row["alignment_angle_rad"])
        sigma_x = float(row["sigma_x"])
        sigma_z = float(row["sigma_z"])
        combined_radius = float(row["combined_radius"])
        native_pc = float(row["native_pc"])
        augmented = False

        if drawing_high_risk and self.high_risk_augment:
            # Posterior-resampling augmentation (docs/25-augmentation-and-
            # threshold-findings.md): this row's covariance IS its own
            # statement of "here is the distribution of plausible true
            # miss-vector offsets consistent with this real measurement" --
            # (sigma_x, sigma_z) describes uncertainty about where the
            # real encounter actually was, not noise we're inventing. A
            # fresh draw from that same distribution is a different,
            # equally real-measurement-consistent encounter, not a
            # synthetic one. Without this, the elevated pool (a few
            # hundred rows at most) would produce the exact same handful
            # of geometries every time it's drawn -- risking the policy
            # memorizing those specific instances.
            x0 = miss_distance * np.cos(alignment_angle_rad)
            z0 = miss_distance * np.sin(alignment_angle_rad)
            x0 = self.rng.normal(x0, sigma_x)
            z0 = self.rng.normal(z0, sigma_z)
            miss_distance = float(np.hypot(x0, z0))
            alignment_angle_rad = float(np.arctan2(z0, x0))
            native_r_rel = np.array([x0, z0, 0.0])
            native_v_rel = np.array([0.0, 0.0, 1.0])
            native_cov = np.diag([sigma_x**2, sigma_z**2, 1e-12])
            native_pc = float(
                compute_pc(native_r_rel, native_v_rel, native_cov, combined_radius, method="chan")
            )
            augmented = True

        sample = {
            "miss_distance": miss_distance,
            "relative_speed": float(row["relative_speed"]),
            "sigma_x": sigma_x,
            "sigma_z": sigma_z,
            "combined_radius": combined_radius,
            "alignment_angle_rad": alignment_angle_rad,
            "native_pc": native_pc,
            "drawn_from_high_risk_pool": drawing_high_risk,
            "augmented": augmented,
            "precise_targeting_error_m": None,
        }
        # The real event's own miss-vector-vs-covariance alignment, not a
        # fresh uniform-random draw -- see docs/23-anisotropic-covariance-
        # fix.md. `solve_secondary_initial_state`'s `orientation_angle_rad`
        # places the miss vector within `encounter_plane_basis(v_rel)`,
        # the SAME basis convention `project_to_encounter_plane` used to
        # compute this angle in the first place (both live in
        # pc/geometry.py), so reusing it here reproduces this specific
        # real event's actual relationship between its miss vector and its
        # (tight sigma_x, loose sigma_z) covariance axes -- not the exact
        # 3D orientation (that was never physically meaningful to preserve,
        # since v_rel's direction here is independently sampled below),
        # just the relative geometry that actually determines Pc.
        orientation_angle_rad = sample["alignment_angle_rad"]
        # solve_secondary_initial_state_robust's own default max_attempts
        # (3,000, raised in Phase 7e -- docs/26-precise-targeting.md) is
        # what makes this reliable for real events with relative speeds
        # well above typical LEO orbital speed; not overridden here.
        scenario = solve_secondary_initial_state_robust(
            self.ego_r0,
            self.ego_v0,
            nominal_tca_s,
            sample["miss_distance"],
            sample["relative_speed"],
            orientation_angle_rad,
            self.rng,
        )

        if drawing_high_risk and self.high_risk_precise_targeting:
            # docs/26-precise-targeting.md: the J2-only solver above is
            # only accurate to ~100-200m at these lead times, regardless
            # of target size -- invisible for the km-scale miss distances
            # every other draw uses, but it swamps the tens-of-meters
            # targets real high-risk events need. Corrects r_sec_t0/v_sec
            # _t0 via 1-3 extra Basilisk calls (~1.5-3s total) so the
            # ACTUAL simulated encounter lands near the intended one, not
            # just the J2 solver's approximation of it. Gated to pool
            # draws specifically -- this cost is real and not worth
            # paying on every episode, only where precision matters.
            scenario, diagnostics = correct_targeting_geometry(
                self.ego_r0, self.ego_v0, scenario, nominal_tca_s
            )
            sample["precise_targeting_error_m"] = diagnostics["final_error_m"]

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
    def current_sigma_xz(self) -> tuple[float, float]:
        """(sigma_x, sigma_z) -- the real event's own encounter-plane
        principal std devs (tight axis, loose axis), for the current
        scenario at TCA. For a constant-uncertainty episode (stage 2, or
        stage 3 without querying sigma_xz_at_fraction), this is the pair
        used throughout.

        Replaces the pre-Phase-7b isotropic `current_sigma` (geometric
        mean) -- see docs/23-anisotropic-covariance-fix.md: collapsing to
        one isotropic value discarded the real covariance ellipse's
        eccentricity (median ~5.8x, per docs/22-evaluation-results.md),
        which materially suppressed computed Pc relative to the real
        anisotropic geometry.
        """
        self._ensure_current()
        return float(self._cached_sample["sigma_x"]), float(self._cached_sample["sigma_z"])

    def sigma_xz_at_fraction(self, fraction: float) -> tuple[float, float]:
        """Interpolated (sigma_x, sigma_z) at `fraction` of the way from
        episode start (0.0) to TCA (1.0), via geometric (log-linear)
        interpolation between the sampled event's real first/last
        per-axis sigma -- covariance shrinks multiplicatively (median
        ~8.36x per docs/15), not additively, so linear interpolation
        would be the wrong shape. Each axis is interpolated independently
        (not collapsed to a magnitude first) so the eccentricity is
        preserved throughout the episode, not just at its endpoints.
        Falls back to the constant `current_sigma_xz` if no evolution
        data was provided (stage 2).
        """
        self._ensure_current()
        if self._cached_evolution is None:
            return self.current_sigma_xz
        fraction = float(np.clip(fraction, 0.0, 1.0))

        def _interp(first: float, last: float) -> float:
            log_val = np.log(first) + fraction * (np.log(last) - np.log(first))
            return float(np.exp(log_val))

        sigma_x = _interp(
            self._cached_evolution["sigma_x_first"], self._cached_evolution["sigma_x_last"]
        )
        sigma_z = _interp(
            self._cached_evolution["sigma_z_first"], self._cached_evolution["sigma_z_last"]
        )
        return sigma_x, sigma_z

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
