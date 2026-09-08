"""Generate attributed schedules with source IDs retained, unlike the v1 library."""

import hashlib
import json

import pandas as pd

from satellite_rl.lab.data import FITTED_DIR
from satellite_rl.scenario.kelvins_loader import DEFAULT_TRAIN_CSV


def main():
    df = pd.read_csv(DEFAULT_TRAIN_CSV, usecols=["event_id", "time_to_tca"])
    artifact = {
        "schema_version": "1.0",
        "source": "ESA Kelvins Collision Avoidance Challenge training data",
        "source_sha256": hashlib.sha256(DEFAULT_TRAIN_CSV.read_bytes()).hexdigest(),
        "schedules_days_before_tca": {
            str(int(event)): sorted(group.time_to_tca.tolist(), reverse=True)
            for event, group in df.groupby("event_id")
        },
    }
    path = FITTED_DIR / "schedules_by_event.json"
    path.write_text(json.dumps(artifact, separators=(",", ":")) + "\n")
    print(f"Wrote {len(artifact['schedules_days_before_tca'])} source schedules to {path}")


if __name__ == "__main__":
    main()
