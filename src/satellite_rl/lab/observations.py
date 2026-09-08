"""Versioned features for generated CDM estimates; never consumes scoring truth."""

import numpy as np

OBSERVATION_VERSION = "encounter-history-v2.0"
FEATURE_NAMES = (
    "log_pc",
    "log_sigma_1",
    "log_sigma_2",
    "correlation",
    "miss_1_scaled",
    "miss_2_scaled",
    "log_radius",
    "time_days",
    "next_update_days",
    "fuel_fraction",
    "update_fraction",
    "previous_r",
    "previous_t",
    "previous_n",
    "basis_r1",
    "basis_r2",
    "basis_t1",
    "basis_t2",
    "basis_n1",
    "basis_n2",
    "noise_scale",
    "history_valid",
    "log_miss",
    "log_lead",
)


def encode_estimate(packet, previous_action, update_fraction):
    cov = np.asarray(packet["covariance_plane_m2"])
    sigma = np.sqrt(np.diag(cov))
    miss = np.asarray(packet["miss_plane_m"])
    features = [
        np.log10(max(packet["estimated_pc"], 1e-30)) / 30,
        *(np.log10(np.maximum(sigma, 1e-6)) / 6),
        cov[0, 1] / np.prod(sigma),
        *(np.arcsinh(miss / np.maximum(sigma, 1e-6)) / 10),
        np.log10(max(packet["radius_m"], 1e-6)) / 3,
        packet["time_to_tca_s"] / 86400 / 7,
        packet["next_update_s"] / 86400 / 7,
        packet["fuel_fraction"],
        update_fraction,
        *(np.asarray(previous_action) / 10),
        *np.asarray(packet["basis_rtn"]).ravel(),
        packet["noise_scale"],
        1.0,
        np.log10(max(np.linalg.norm(miss), 1e-6)) / 7,
        np.log10(max(packet["time_to_tca_s"], 1)) / 6,
    ]
    return np.clip(np.asarray(features, dtype=np.float32), -10, 10)
