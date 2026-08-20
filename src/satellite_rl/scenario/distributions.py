"""Fit and sample distributions from real Kelvins CDM statistics, to ground
synthetic scenario generation in reality rather than invented numbers.
See docs/03-scenario-design.md.

Three things get fit here, matching docs/03-scenario-design.md's design:
1. Encounter geometry (miss distance, relative speed, encounter-plane
   sigma_x/sigma_z, combined radius) -- the per-scenario parameters for
   curriculum stage 2 ("single conjunction, sampled geometry").
2. Covariance shrink ratio (first-CDM vs. last-CDM combined-sigma
   magnitude per event) -- for curriculum stage 3's evolving-uncertainty
   model.
3. A schedule library (real per-event time_to_tca sequences) -- sampled by
   bootstrap resampling of actual event schedules rather than a fitted
   parametric distribution, since real schedules are irregular (counts
   ranging 1-23 CDMs/event; see docs/14 for the empirical check). This
   matches docs/09-episode-design.md's "decision points sampled from the
   real distribution of time_to_tca values ... within a Kelvins event."
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..pc.geometry import project_to_encounter_plane
from .kelvins_loader import (
    DEFAULT_TRAIN_CSV,
    build_covariance_from_cdm_row,
    combined_radius_from_rcs,
    load_events,
    relative_state_columns,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FITTED_DIR = REPO_ROOT / "data" / "fitted"


@dataclass
class FittedLognormal:
    """A lognormal fit: X = exp(mu + sigma * Z), Z ~ N(0,1). Equivalently,
    scipy's (shape=sigma, loc=0, scale=exp(mu)) parameterization.
    """

    mu: float
    sigma: float
    n_samples: int
    ks_statistic: float
    ks_pvalue: float

    def sample(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray | float:
        draws = rng.lognormal(mean=self.mu, sigma=self.sigma, size=size)
        return draws

    def to_dict(self) -> dict:
        return asdict(self)


def fit_lognormal(data: np.ndarray) -> FittedLognormal:
    """Fit a lognormal to strictly-positive data via scipy (location fixed
    at 0, the standard approach for a variable with no natural offset),
    and report a Kolmogorov-Smirnov goodness-of-fit statistic honestly --
    not just assumed to fit well because "orbital mechanics is usually
    right-skewed."
    """
    data = np.asarray(data, dtype=float)
    data = data[data > 0]
    shape, _loc, scale = stats.lognorm.fit(data, floc=0)
    mu = float(np.log(scale))
    sigma = float(shape)
    ks_stat, ks_p = stats.kstest(data, "lognorm", args=(shape, 0, scale))
    return FittedLognormal(
        mu=mu, sigma=sigma, n_samples=len(data), ks_statistic=float(ks_stat), ks_pvalue=float(ks_p)
    )


def _add_encounter_geometry(df):
    """Row-wise: project each event onto the encounter plane, derive
    combined radius from RCS. Drops rows that fail (missing RCS, singular
    covariance) and returns (augmented_df, n_dropped).
    """
    rows = []
    n_dropped = 0
    for _, row in df.iterrows():
        try:
            r_rel, v_rel = relative_state_columns(row)
            cov_combined = build_covariance_from_cdm_row(row, "t") + build_covariance_from_cdm_row(
                row, "c"
            )
            geometry = project_to_encounter_plane(r_rel, v_rel, cov_combined)
            radius = combined_radius_from_rcs(row)
            rows.append(
                {
                    "event_id": row["event_id"],
                    "miss_distance": row["miss_distance"],
                    "relative_speed": row["relative_speed"],
                    "sigma_x": geometry.sigma_x,
                    "sigma_z": geometry.sigma_z,
                    "combined_radius": radius,
                }
            )
        except Exception:  # noqa: BLE001 -- counting failures, not debugging one
            n_dropped += 1

    return pd.DataFrame(rows), n_dropped


def fit_geometry_distributions(
    csv_path: Path = DEFAULT_TRAIN_CSV,
) -> tuple[dict[str, FittedLognormal], pd.DataFrame]:
    """Fit lognormal distributions to final-CDM event geometry (miss
    distance, relative speed, encounter-plane sigma_x/sigma_z, RCS-derived
    combined radius), and return the underlying per-event table alongside
    the fits -- the table is what `sample_scenario_geometry_bootstrap`
    uses, since the marginal fits alone are a statistically poor
    approximation (see docs/15-distribution-fitting-results.md).
    """
    final_cdms = load_events(csv_path, which="final")
    required = ["t_rcs_estimate", "c_rcs_estimate"]
    final_cdms = final_cdms.dropna(subset=required)

    geometry_df, n_dropped = _add_encounter_geometry(final_cdms)
    print(
        f"fit_geometry_distributions: {len(geometry_df)} events used, "
        f"{n_dropped} dropped (projection/covariance failures)."
    )

    fits = {
        "miss_distance": fit_lognormal(geometry_df["miss_distance"].to_numpy()),
        "relative_speed": fit_lognormal(geometry_df["relative_speed"].to_numpy()),
        "sigma_x": fit_lognormal(geometry_df["sigma_x"].to_numpy()),
        "sigma_z": fit_lognormal(geometry_df["sigma_z"].to_numpy()),
        "combined_radius": fit_lognormal(geometry_df["combined_radius"].to_numpy()),
    }
    return fits, geometry_df


def fit_covariance_shrink_ratio(csv_path: Path = DEFAULT_TRAIN_CSV) -> FittedLognormal:
    """Fit the ratio of (first-CDM combined-sigma magnitude) to
    (last-CDM combined-sigma magnitude) per event -- how much the
    covariance typically shrinks between an event's first and final
    reported CDM. Used by docs/03-scenario-design.md's evolving-
    uncertainty model (curriculum stage 3).

    "Combined-sigma magnitude" here is geometric mean(sigma_x, sigma_z) in
    the encounter-plane projection, i.e. a single scalar uncertainty-size
    summary -- consistent with what fit_geometry_distributions fits.
    """
    first_cdms = load_events(csv_path, which="first").dropna(
        subset=["t_rcs_estimate", "c_rcs_estimate"]
    )
    last_cdms = load_events(csv_path, which="final").dropna(
        subset=["t_rcs_estimate", "c_rcs_estimate"]
    )

    first_geom, _ = _add_encounter_geometry(first_cdms)
    last_geom, _ = _add_encounter_geometry(last_cdms)

    merged = first_geom.merge(last_geom, on="event_id", suffixes=("_first", "_last"))
    # Only meaningful for events with >1 CDM -- events with exactly one CDM
    # have first == last by construction and would just inject a spurious
    # spike at ratio=1.
    cdm_counts = load_events(csv_path, which="all").groupby("event_id").size()
    multi_cdm_events = cdm_counts[cdm_counts > 1].index
    merged = merged[merged["event_id"].isin(multi_cdm_events)]

    mag_first = np.sqrt(merged["sigma_x_first"] * merged["sigma_z_first"])
    mag_last = np.sqrt(merged["sigma_x_last"] * merged["sigma_z_last"])
    ratio = (mag_first / mag_last).to_numpy()

    print(f"fit_covariance_shrink_ratio: {len(ratio)} multi-CDM events used.")
    return fit_lognormal(ratio)


def extract_schedule_library(csv_path: Path = DEFAULT_TRAIN_CSV, max_events: int = 5000) -> list:
    """Real per-event time_to_tca sequences (days before TCA, descending),
    for bootstrap resampling by the episode generator (docs/09-episode-
    design.md), rather than a fitted parametric schedule model -- real
    schedules are too irregular (1-23 CDMs/event) for a clean parametric
    fit to be worth the complexity.
    """
    df = load_events(csv_path, which="all")
    schedules = []
    for _event_id, group in df.groupby("event_id"):
        times = sorted(group["time_to_tca"].tolist(), reverse=True)
        schedules.append(times)
        if len(schedules) >= max_events:
            break
    print(f"extract_schedule_library: {len(schedules)} event schedules extracted.")
    return schedules


def sample_scenario_geometry(fitted: dict[str, FittedLognormal], rng: np.random.Generator) -> dict:
    """Draw one synthetic scenario's geometry parameters from INDEPENDENT
    per-parameter lognormal fits.

    NOTE (see docs/15-distribution-fitting-results.md): the KS goodness-of-
    fit tests for these marginal fits are statistically poor (p ~ 0 for
    every parameter, KS statistics 0.10-0.25), and independent marginals
    also discard real cross-parameter correlations (e.g. miss distance and
    covariance size are not actually independent in the real data). This
    function is kept as a documented, simple, always-available fallback
    (e.g. for generating scenarios without the full geometry table on
    disk) -- prefer `sample_scenario_geometry_bootstrap` when the real
    per-event table is available, which is the actual recommendation.
    """
    return {
        "miss_distance": float(fitted["miss_distance"].sample(rng)),
        "relative_speed": float(fitted["relative_speed"].sample(rng)),
        "sigma_x": float(fitted["sigma_x"].sample(rng)),
        "sigma_z": float(fitted["sigma_z"].sample(rng)),
        "combined_radius": float(fitted["combined_radius"].sample(rng)),
    }


def sample_scenario_geometry_bootstrap(geometry_df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Draw one synthetic scenario's geometry by resampling an entire real
    event's geometry tuple (with replacement), preserving real
    cross-parameter correlations that independent marginal fits discard.
    This is the recommended sampling method -- see the note on
    `sample_scenario_geometry` and docs/15-distribution-fitting-results.md.
    """
    row = geometry_df.iloc[rng.integers(0, len(geometry_df))]
    return {
        "miss_distance": float(row["miss_distance"]),
        "relative_speed": float(row["relative_speed"]),
        "sigma_x": float(row["sigma_x"]),
        "sigma_z": float(row["sigma_z"]),
        "combined_radius": float(row["combined_radius"]),
    }


def save_fitted(
    geometry_fits: dict[str, FittedLognormal],
    geometry_df: pd.DataFrame,
    shrink_ratio_fit: FittedLognormal,
    schedule_library: list,
    out_dir: Path = FITTED_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "geometry_distributions.json", "w") as f:
        json.dump({k: v.to_dict() for k, v in geometry_fits.items()}, f, indent=2)
    geometry_df.to_csv(out_dir / "geometry_events.csv", index=False)
    with open(out_dir / "covariance_shrink_ratio.json", "w") as f:
        json.dump(shrink_ratio_fit.to_dict(), f, indent=2)
    with open(out_dir / "schedule_library.json", "w") as f:
        json.dump(schedule_library, f)
    print(f"Saved fitted distributions to {out_dir}/")


def main() -> None:
    print("=== Fitting geometry distributions ===")
    geometry_fits, geometry_df = fit_geometry_distributions()
    for name, fit in geometry_fits.items():
        print(
            f"  {name}: mu={fit.mu:.3f} sigma={fit.sigma:.3f} "
            f"n={fit.n_samples} KS_stat={fit.ks_statistic:.4f} KS_p={fit.ks_pvalue:.2e}"
        )

    print("\n=== Fitting covariance shrink ratio ===")
    shrink_fit = fit_covariance_shrink_ratio()
    print(
        f"  shrink_ratio: mu={shrink_fit.mu:.3f} sigma={shrink_fit.sigma:.3f} "
        f"n={shrink_fit.n_samples} KS_stat={shrink_fit.ks_statistic:.4f} "
        f"KS_p={shrink_fit.ks_pvalue:.2e}"
    )
    # median ratio = exp(mu); report directly since it's the most
    # interpretable single number ("covariance typically shrinks by ~Nx").
    print(f"  median shrink ratio: {np.exp(shrink_fit.mu):.2f}x")

    print("\n=== Extracting schedule library ===")
    schedules = extract_schedule_library()
    counts = [len(s) for s in schedules]
    print(f"  CDM count per event: min={min(counts)} median={np.median(counts):.0f} max={max(counts)}")

    save_fitted(geometry_fits, geometry_df, shrink_fit, schedules)


if __name__ == "__main__":
    main()
