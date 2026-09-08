# APSIS

### A conjunction research instrument

[![CI](https://github.com/ArmaanSayyad/satellite-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/ArmaanSayyad/satellite-rl/actions/workflows/ci.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-lightgrey)](LICENSE)

Explore the moment a satellite must decide: **maneuver now, or wait for a better warning?**

Apsis brings this repository's satellite collision-avoidance research into a runnable
mission-control application: real Basilisk trajectory recordings, policy comparisons,
encounter-plane uncertainty, decision evidence, and the experiments behind the models.
Geometry comes from ESA's Kelvins CDMs. Reference orbits and sensing errors are generated;
these are scientific simulations, not reconstructions of historical satellite operations.

## Open the instrument

With **Node 24.15+** (24.20 tested):

```sh
npm run dev
```

Open the Vite address printed in the terminal. Verified example replays load immediately.
No Python, model download, or Basilisk installation is required to browse them.

Or run the application service and built UI together:

```sh
docker compose up --build
```

Open **http://localhost:8000**. This default container serves verified replays and the
typed catalog API. See [deployment and live physics](docs/DEPLOYMENT.md) for live jobs.

## What is measured—and what is unresolved

**An experimental state-conditional prototype is available; acceptance is not met.**
Planner imitation followed by 64 PPO steps mitigated 2/3 dangerous sensing repeats
and waited on 3/3 safe repeats. Only one dangerous and one safe family were tested;
the dangerous family was seen during training. Teacher-only decisions were identical,
so there is no demonstrated PPO benefit. See the [model card](docs/MODEL_CARD.md).

The earlier runs remain visible as negative findings.
The 64-step continuous proof maneuvers on both safe held-out examples. A 256-step
categorical pilot, despite an explicit wait action, repeats the same `−0.1 m/s`
along-track burn across tested risk/covariance interventions. These are small proofs,
not adequate training budgets or evidence of general safety.

The audit also found a forecasting mismatch: one source-8767 J2 forecast predicts
about **305 m**, while the actual Basilisk encounter is about **38 m**. Measurement
covariance alone does not describe that dynamics-model error. See the
[technical report](TECHNICAL.md), [registered experiment protocol](docs/EXPERIMENT_PROTOCOL.md),
and downloadable evidence in the application.

One controlled fixed-schedule, seed-0 replay of event 8767 illustrates the comparison:

| Policy | Fuel (m/s) | Final Pc, Basilisk state + Foster |
|---|---:|---:|
| Coast | 0 | 1.6226e-4 |
| Published v1 PPO | 1.3614 | numerical zero |
| Legacy radial threshold | 10 | numerical zero |
| Noisy-estimate geometry planner | 0.0200 | 2.3931e-6 |
| Experimental v2 (missed intervention) | 0 | 1.6226e-4 |

This is **one source family and seed**, under different information regimes: v1/legacy
baselines receive truth-derived forecasts; the planner receives noisy estimates.
It is not a population comparison. Zero here is numerical output, not proof of zero
physical risk. The planner fails other measured warning sequences; inspect the evidence.

The library includes 17 verified policy replays across four source families, including
an uncapped 4.70-day, 14-update source-schedule safe event. See the
[verification record](docs/VERIFICATION.md) for the checks actually run.

## Research workflow

Python 3.11 is the tested full-stack interpreter. Install Basilisk before this package:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install 'bsk[all]'
pip install -e '.[dev]' -r web/backend/requirements.txt
python -m scripts.download_checkpoint
python scripts/download_kelvins.py
```

The checkpoint downloader verifies the release SHA-256 and preserves mismatched existing
files. The dataset downloader verifies the canonical Zenodo archive's MD5. Small derived
geometry and event-linked schedules are committed, so routine simulations do not require
the 232 MB CSV. Rebuild schedules with `python scripts/build_keyed_schedules.py`.

```sh
# Train with source-family partitions, real irregular schedules, noisy history,
# process isolation, manifests, periodic checkpoints, and an explicit wait action.
python -m satellite_rl.lab.train --out runs/v2-serious --steps 100000 \
  --workers 4 --action-mode discrete19 --risk-pool-fraction 0.001 \
  --forecast-fidelity basilisk --imitation-samples 2048

# Evaluate without silently allowing training-source overlap.
python -m satellite_rl.lab.evaluate --help

# Produce real recorder-verified comparisons.
python -m scripts.generate_replays --events 8767 --policies v1 never_maneuver planner

# Full Python suite; physics-dependent tests skip when Basilisk is absent.
python -m pytest -q
ruff check src tests scripts web/backend
```

Exact completed experiments, resume commands, acceptance gates, and limitations are in
the [experiment protocol](docs/EXPERIMENT_PROTOCOL.md). A full training run is expensive:
short-window throughput must not be extrapolated to multi-day precision targeting.

## Inside

- **Briefing:** learn CDMs, TCA, Pc, covariance and Δv through a measured encounter.
- **Scenario laboratory:** switch source events and policies, seek linked playback,
  inspect orbital context and the encounter plane, and test the displayed radius assumption.
- **Evidence:** inspect measured run configuration, learning curves, evaluation matrices,
  feature interventions and model limitations. Download the underlying JSON.
- **Live service:** opt-in isolated simulation jobs, polling, cancellation, deterministic
  cache keys, configurable models/origins, and structured failures.

[Technical report](TECHNICAL.md) · [Data card](docs/DATA_CARD.md) ·
[Architecture](docs/ARCHITECTURE.md) · [Deployment](docs/DEPLOYMENT.md) ·
[Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) · [Model card](docs/MODEL_CARD.md)

## License and scope

Code: [MIT](LICENSE). ESA/Kelvins data and derived geometry/schedules/replays carry
**CC-BY-4.0 attribution**, separately from the code license; see
[SOURCE.md](data/kelvins_cdm/SOURCE.md) and the [data card](docs/DATA_CARD.md).
Earth imagery attribution is in [web/README.md](web/README.md).

This is a single-ego research simulator, not a flight-qualified maneuver advisory system.
Constellation scheduling remains deferred until single-satellite risk gating is established.
