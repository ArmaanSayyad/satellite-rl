#!/usr/bin/env python3
"""Phase 1 validation: compare our Pc implementation against the ESA Kelvins
dataset's real, historical `risk` values (see docs/13-roadmap.md Phase 1,
docs/04-collision-probability.md).

Empirically established (see this script's exploratory run, Aug 2026):
- `risk` in the dataset is log10(Pc), floored at -30.
- Position/covariance columns are in meters / m/s (consistent units
  throughout -- verified by comparing magnitude ranges, not assumed).
- The dataset provides the FULL position covariance (not just diagonal
  sigmas): {t,c}_sigma_{r,t,n} are the diagonal (variance = sigma^2), and
  {t,c}_ct_r, {t,c}_cn_r, {t,c}_cn_t are the off-diagonal covariance terms,
  matching the standard CCSDS CDM covariance-matrix layout.
- The dataset does NOT provide a hard-body-radius column directly (this
  gap was correctly flagged in docs/05-datasets.md). We derive a combined
  radius from {t,c}_rcs_estimate (radar cross-section, m^2) via
  r = sqrt(RCS / pi) -- a real physical proxy, not an invented constant,
  but still an approximation (RCS != physical cross-sectional area in
  general; treat this as a documented modeling choice, not ground truth).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from satellite_rl.pc import compute_pc

DATA_PATH = (
    REPO_ROOT
    / "data"
    / "kelvins_cdm"
    / "Collision Avoidance Challenge - Dataset"
    / "kelvins_competition_data"
    / "train_data.csv"
)

RISK_FLOOR = -30.0


def build_covariance(row: pd.Series, prefix: str) -> np.ndarray:
    sr, st, sn = row[f"{prefix}_sigma_r"], row[f"{prefix}_sigma_t"], row[f"{prefix}_sigma_n"]
    ct_r, cn_r, cn_t = row[f"{prefix}_ct_r"], row[f"{prefix}_cn_r"], row[f"{prefix}_cn_t"]
    return np.array(
        [
            [sr**2, ct_r, cn_r],
            [ct_r, st**2, cn_t],
            [cn_r, cn_t, sn**2],
        ]
    )


def combined_radius_from_rcs(row: pd.Series) -> float:
    r_t = np.sqrt(row["t_rcs_estimate"] / np.pi)
    r_c = np.sqrt(row["c_rcs_estimate"] / np.pi)
    return float(r_t + r_c)


def main() -> None:
    print(f"Loading {DATA_PATH.name}...")
    df = pd.read_csv(DATA_PATH)
    print(f"{len(df)} total CDM rows across {df['event_id'].nunique()} events.")

    # Final CDM per event (closest to TCA) -- the most operationally
    # meaningful single risk assessment per event, and avoids
    # double-counting the same underlying encounter many times.
    final_cdms = df.loc[df.groupby("event_id")["time_to_tca"].idxmin()].copy()
    print(f"{len(final_cdms)} final-CDM rows (one per event).")

    required_cols = ["t_rcs_estimate", "c_rcs_estimate", "risk"]
    final_cdms = final_cdms.dropna(subset=required_cols)
    print(f"{len(final_cdms)} rows with non-null RCS + risk.")

    sample_n = min(3000, len(final_cdms))
    sample = final_cdms.sample(sample_n, random_state=0)
    print(f"Validating on a random sample of {sample_n} events (method=chan for speed)...")

    computed_log10_pc = []
    reported_risk = []
    n_failed = 0
    failure_reasons = {}

    for _, row in sample.iterrows():
        try:
            r_rel = np.array(
                [row["relative_position_r"], row["relative_position_t"], row["relative_position_n"]]
            )
            v_rel = np.array(
                [row["relative_velocity_r"], row["relative_velocity_t"], row["relative_velocity_n"]]
            )
            cov_t = build_covariance(row, "t")
            cov_c = build_covariance(row, "c")
            cov_combined = cov_t + cov_c
            radius = combined_radius_from_rcs(row)

            pc = compute_pc(r_rel, v_rel, cov_combined, radius, method="chan")
            log10_pc = np.log10(max(pc, 10**RISK_FLOOR))
            computed_log10_pc.append(log10_pc)
            reported_risk.append(row["risk"])
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, we're counting/categorizing failures
            n_failed += 1
            reason = type(exc).__name__
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    n_ok = len(computed_log10_pc)
    print(f"\n{n_ok} succeeded, {n_failed} failed ({n_failed / sample_n:.1%}).")
    if failure_reasons:
        print("Failure reasons:", failure_reasons)

    computed_log10_pc = np.array(computed_log10_pc)
    reported_risk = np.array(reported_risk)

    pearson_r, pearson_p = stats.pearsonr(computed_log10_pc, reported_risk)
    spearman_r, spearman_p = stats.spearmanr(computed_log10_pc, reported_risk)
    diff = computed_log10_pc - reported_risk

    print("\n--- Agreement between computed log10(Pc) and dataset `risk` ---")
    print(f"Pearson r  = {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"Spearman r = {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"diff (computed - reported): mean={diff.mean():.3f}, median={np.median(diff):.3f}, "
          f"std={diff.std():.3f}")
    print(f"diff percentiles: 5%={np.percentile(diff, 5):.2f}, 25%={np.percentile(diff, 25):.2f}, "
          f"75%={np.percentile(diff, 75):.2f}, 95%={np.percentile(diff, 95):.2f}")

    at_floor_reported = (reported_risk <= RISK_FLOOR + 1e-9).mean()
    at_floor_computed = (computed_log10_pc <= RISK_FLOOR + 1e-9).mean()
    print(f"\nFraction at floor ({RISK_FLOOR}): reported={at_floor_reported:.1%}, "
          f"computed={at_floor_computed:.1%}")

    # Excluding floored values on both sides -- the floor is a dataset
    # artifact (they don't report Pc below 1e-30), not a real agreement
    # signal, and our own floor-clipping means both sides pile up at -30
    # regardless of true underlying magnitude once it's astronomically small.
    not_floored = (reported_risk > RISK_FLOOR + 1e-9) & (computed_log10_pc > RISK_FLOOR + 1e-9)
    if not_floored.sum() > 10:
        pr2, _ = stats.pearsonr(computed_log10_pc[not_floored], reported_risk[not_floored])
        sr2, _ = stats.spearmanr(computed_log10_pc[not_floored], reported_risk[not_floored])
        print(f"\nExcluding floored rows ({not_floored.sum()} rows remain):")
        print(f"Pearson r  = {pr2:.4f}")
        print(f"Spearman r = {sr2:.4f}")
        d2 = (computed_log10_pc - reported_risk)[not_floored]
        print(f"diff: mean={d2.mean():.3f}, median={np.median(d2):.3f}, std={d2.std():.3f}")


if __name__ == "__main__":
    main()
