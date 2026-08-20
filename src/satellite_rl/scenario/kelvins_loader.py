"""Shared utilities for loading and interpreting the ESA Kelvins CDM dataset.

Column semantics documented here were established empirically (Phase 1,
see docs/14-pc-validation-results.md and docs/05-datasets.md's
"Corrections" note), not assumed from the original secondhand research:
`time_to_tca` is in days, `risk` is log10(Pc) floored at -30, covariance
columns include full off-diagonal cross-terms in the standard CCSDS CDM
layout, and there is no direct hard-body-radius column (we derive one from
`{t,c}_rcs_estimate`).
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_TRAIN_CSV = (
    REPO_ROOT
    / "data"
    / "kelvins_cdm"
    / "Collision Avoidance Challenge - Dataset"
    / "kelvins_competition_data"
    / "train_data.csv"
)

RISK_FLOOR = -30.0


def build_covariance_from_cdm_row(row: pd.Series, prefix: str) -> np.ndarray:
    """Build the 3x3 position covariance for one object (prefix 't' or 'c')
    from a Kelvins CDM row, including the off-diagonal cross-terms.
    """
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
    """Approximate combined hard-body radius from radar cross-section
    (m^2), treating each object as an equivalent circular cross-section.
    See docs/14-pc-validation-results.md for why this is an approximation,
    not ground truth, and how well it validated empirically.
    """
    r_t = np.sqrt(row["t_rcs_estimate"] / np.pi)
    r_c = np.sqrt(row["c_rcs_estimate"] / np.pi)
    return float(r_t + r_c)


def load_raw(csv_path: Path = DEFAULT_TRAIN_CSV) -> pd.DataFrame:
    """Load the full Kelvins CDM CSV, no filtering."""
    return pd.read_csv(csv_path)


def load_events(csv_path: Path = DEFAULT_TRAIN_CSV, which: str = "final") -> pd.DataFrame:
    """Load one row per event: the final (closest-to-TCA), first
    (farthest-from-TCA), or all CDM rows.

    Args:
        which: "final" (min time_to_tca per event), "first" (max
            time_to_tca per event), or "all" (every row, unfiltered).
    """
    df = load_raw(csv_path)
    if which == "all":
        return df
    if which == "final":
        return df.loc[df.groupby("event_id")["time_to_tca"].idxmin()].copy()
    if which == "first":
        return df.loc[df.groupby("event_id")["time_to_tca"].idxmax()].copy()
    raise ValueError(f"Unknown which={which!r}, expected 'final', 'first', or 'all'")


def relative_state_columns(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Extract (r_rel, v_rel) in the RTN frame from a Kelvins CDM row."""
    r_rel = np.array(
        [row["relative_position_r"], row["relative_position_t"], row["relative_position_n"]]
    )
    v_rel = np.array(
        [row["relative_velocity_r"], row["relative_velocity_t"], row["relative_velocity_n"]]
    )
    return r_rel, v_rel
