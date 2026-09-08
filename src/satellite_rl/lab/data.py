"""Source-family partitioning. Split before generating any scenario variants."""

import hashlib
import json
from pathlib import Path

import numpy as np

FITTED_DIR = Path(__file__).resolve().parents[3] / "data" / "fitted"


def grouped_split(event_ids, seed=42, fractions=(0.7, 0.15, 0.15)):
    if len(fractions) != 3 or any(f < 0 for f in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("Three nonnegative split fractions must sum to one")
    ids = sorted({int(i) for i in event_ids})
    ids.sort(key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    a, b = int(len(ids) * fractions[0]), int(len(ids) * sum(fractions[:2]))
    return {"train": ids[:a], "validation": ids[a:b], "test": ids[b:]}


def load_keyed_schedules():
    path = FITTED_DIR / "schedules_by_event.json"
    if not path.exists():
        raise FileNotFoundError(
            "Run python scripts/build_keyed_schedules.py with the ESA data installed"
        )
    return json.loads(path.read_text())["schedules_days_before_tca"]
