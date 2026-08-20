"""Baselines 3 and 4 from docs/11-evaluation.md: the threshold heuristic
and the hindsight-optimal oracle. Baselines 1/2 (never-maneuver,
always-max-thrust) and the random policy are simple enough to stay
inline in evaluate.py; these two need real logic.

Observation layout (verified empirically against a live env, not assumed
from the observation_spec order -- see docs/22-evaluation-results.md):
`obs[0]` is `collision_prob` (the current predicted Pc), matching
`env._pc_fn`'s own return value exactly for the same satellite state.
"""

import numpy as np

PC_OBS_INDEX = 0

# A fixed radial (R, away from the primary) burn direction: real CAM
# practice favors radial burns for short-notice encounters (the fast,
# bounded, immediately-effective axis), versus in-track burns which rely
# on secular along-track drift accumulating over longer lead times than
# stage 2's short schedule (0.2 days down to TCA) provides. This is a
# deliberate simplification of docs/11's "e.g. computed via a simple
# closed-form miss-distance-increase targeting" -- no attempt is made to
# solve for the direction that actually increases miss distance for a
# specific geometry, just an operationally-plausible fixed axis. Doesn't
# guarantee the burn helps for every sampled geometry; that's a real,
# reported limitation, not hidden.
RADIAL_DV_DIRECTION = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def threshold_heuristic_action(obs: np.ndarray, max_dv_ms: float, pc_threshold: float) -> np.ndarray:
    """Fixed-direction, fixed-magnitude radial burn at max_dv_ms whenever
    observed Pc exceeds pc_threshold, otherwise no action. Mirrors the
    real operational pattern (act once risk crosses a threshold, don't
    otherwise) without a learned or targeted direction/magnitude.
    """
    if obs[PC_OBS_INDEX] > pc_threshold:
        return RADIAL_DV_DIRECTION * max_dv_ms
    return np.zeros(3, dtype=np.float32)


def hindsight_oracle_fuel(
    env,
    seed: int,
    max_dv_ms: float,
    pc_threshold: float,
    tol_ms: float = 0.05,
    max_iters: int = 12,
) -> tuple[float, bool]:
    """Bisection search (real re-simulation, not a closed-form derivation
    -- see docs/22-evaluation-results.md for why) for the minimum-
    magnitude single first-decision-step radial burn that achieves final
    Pc <= pc_threshold, for one fixed held-out scenario (same `seed`
    reproduces the same scenario, per the env's reset(seed=X) contract).

    This is a genuine "hindsight" computation -- it requires re-running
    the same true future multiple times to find the answer, using
    information (the outcome of a given burn) unavailable to a realistic
    policy acting once, at decision time -- but it is NOT a fully general
    minimum-fuel oracle: it only searches over burn magnitude, fixed to
    the radial direction and the earliest decision point (the cheapest
    place to burn, by CW-equation intuition). A true global oracle would
    also search over direction and burn timing, which is more expensive
    and out of scope here; this is a valid upper bound on required fuel
    among "single first-step radial burns," not a true global minimum.

    Returns (fuel_ms, achieved) -- achieved=False if even max_dv_ms in
    the radial direction can't bring Pc under threshold for this
    scenario (reported honestly, not silently extrapolated past the
    action bound).
    """

    def run(magnitude: float) -> float:
        _obs, info = env.reset(seed=seed)
        terminated = truncated = False
        first = True
        while not (terminated or truncated):
            if first:
                action = RADIAL_DV_DIRECTION * magnitude
                first = False
            else:
                action = np.zeros(3, dtype=np.float32)
            _obs, _reward, terminated, truncated, info = env.step(action)
        return info["pc_final"]

    if run(0.0) <= pc_threshold:
        return 0.0, True
    if run(max_dv_ms) > pc_threshold:
        return max_dv_ms, False

    lo, hi = 0.0, max_dv_ms
    for _ in range(max_iters):
        if hi - lo < tol_ms:
            break
        mid = (lo + hi) / 2.0
        if run(mid) <= pc_threshold:
            hi = mid
        else:
            lo = mid
    return hi, True
