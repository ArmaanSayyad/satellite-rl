"""Process-isolated full-dynamics sensor forecasts from the present simulated state.

Use an external interpreter, not multiprocessing children: SB3 vector workers
are daemonic. SPICE kernels belong exclusively to the child process.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

BASE_EPOCH = datetime(2018, 9, 29, 21, tzinfo=timezone.utc)
FORECAST_VERSION = "basilisk-epoch-aware-v1"


def forecast_states(ego_r, ego_v, sec_r, sec_v, lead_s, elapsed_s, sim_rate_s=1.0):
    """Return no-further-burn terminal states, never reusing the parent's SPICE state."""
    payload = {
        "ego_r": np.asarray(ego_r).tolist(),
        "ego_v": np.asarray(ego_v).tolist(),
        "sec_r": np.asarray(sec_r).tolist(),
        "sec_v": np.asarray(sec_v).tolist(),
        "lead_s": float(lead_s),
        "elapsed_s": float(elapsed_s),
        "sim_rate_s": float(sim_rate_s),
    }
    if lead_s < 0 or elapsed_s < 0 or sim_rate_s <= 0:
        raise ValueError("Forecast times must be nonnegative and integration step positive")
    if lead_s == 0:
        return tuple(np.asarray(payload[k]) for k in ("ego_r", "ego_v", "sec_r", "sec_v"))
    completed = subprocess.run(
        [sys.executable, "-m", "satellite_rl.lab.forecast"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Basilisk forecast subprocess failed: {completed.stderr[-2000:]}")
    result = json.loads(completed.stdout)
    states = tuple(np.asarray(result[k], dtype=float) for k in ("ego_r", "ego_v", "sec_r", "sec_v"))
    if any(state.shape != (3,) or not np.all(np.isfinite(state)) for state in states):
        raise RuntimeError("Basilisk forecast returned invalid states")
    return states


def main():
    from satellite_rl.scenario.tca_refinement import _fly_passive_pair

    payload = json.load(sys.stdin)
    epoch = BASE_EPOCH + timedelta(seconds=payload["elapsed_s"])
    utc = epoch.strftime("%Y %b %d %H:%M:%S.%f (UTC)")
    _, r1, v1, r2, v2 = _fly_passive_pair(
        *[np.asarray(payload[k]) for k in ("ego_r", "ego_v", "sec_r", "sec_v")],
        payload["lead_s"],
        payload["sim_rate_s"],
        utc_init=utc,
    )
    json.dump(
        {
            "ego_r": r1[-1].tolist(),
            "ego_v": v1[-1].tolist(),
            "sec_r": r2[-1].tolist(),
            "sec_v": v2[-1].tolist(),
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
