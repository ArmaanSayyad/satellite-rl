# satellite-rl

[![CI](https://github.com/ArmaanSayyad/satellite-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/ArmaanSayyad/satellite-rl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Open-source reinforcement learning for satellite collision avoidance:
an RL agent decides whether to burn fuel to dodge a predicted close
approach ("conjunction") with another object, or wait for a better risk
estimate — trading collision risk against limited fuel, over the
multi-day sequence of refining warnings a real operator actually gets.

Built on [`bsk_rl`](https://github.com/AVSLab/bsk_rl) (real orbital
dynamics via Basilisk), with every conjunction scenario grounded in
real historical data from ESA's Kelvins Collision Avoidance Challenge
dataset — not invented numbers.

## Result

On the 7 real historical ESA events found to be genuinely dangerous
(collision probability above the standard `1e-4` operational
threshold), the trained policy drove risk to zero on every one, using
**23–27× less fuel** than the cruder baselines that also achieve zero
risk, and was the first checkpoint in this project to beat both a
"do nothing" baseline and a realistic threshold-based heuristic. The
full result — including the honest caveat that this isn't yet a
policy that *reacts* to risk state, and the five real bugs that had to
be found and fixed before the environment could produce a genuinely
dangerous training episode at all — is in
**[TECHNICAL.md](TECHNICAL.md)**.

Try it interactively: **[web demo](web/README.md)** — run the trained
policy against a real event in your browser with a live 3D
visualization.

## Quickstart

1. **Install Basilisk first** — it is not a normal pip dependency.
   Follow the [AVSLab install docs](https://avslab.github.io/basilisk/);
   `pip install "bsk[all]"` covers most platforms with a prebuilt wheel.
   macOS/Linux are preferred over Windows.
2. ```
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. `python scripts/download_kelvins.py` — fetches and checksums the
   Kelvins dataset into `data/kelvins_cdm/` (232 MB, not committed to
   the repo).
4. `pytest` — the full test suite, including Basilisk-dependent tests.
5. Optional, reproduces the validation results in
   [TECHNICAL.md](TECHNICAL.md#6-development-history-what-was-found-and-fixed):
   `python scripts/validate_pc_against_kelvins.py` and
   `python -m satellite_rl.scenario.distributions`.

### Running the trained policy / web demo

The trained checkpoint (`runs/ppo_stage2_riskaware_run1.zip`) is
gitignored like all training artifacts. Get it either way:

- **Download the pretrained checkpoint** from this repo's
  [Releases](https://github.com/ArmaanSayyad/satellite-rl/releases)
  page and unzip it into `runs/`.
- **Or train it yourself** (~90 minutes) with the exact settings used
  for the result above — see [`web/README.md`](web/README.md#prerequisite-a-trained-model-checkpoint).

Then see [`web/README.md`](web/README.md) to run the interactive demo.

## Repository layout

```
src/satellite_rl/    the package -- pc/, scenario/, env/, training/
tests/                pytest suite
scripts/              dataset download + validation scripts
data/                  Kelvins dataset (gitignored except provenance) +
                       small committed derived artifacts
web/                   interactive 3D demo (FastAPI + React/Three.js)
TECHNICAL.md          problem, data, architecture, training, results,
                       findings, limitations -- the full write-up
```

## License

Code: MIT (see [LICENSE](LICENSE)). The `data/kelvins_cdm/` dataset is
CC-BY-4.0 (ESA/Kelvins, see `data/kelvins_cdm/SOURCE.md`) — a different
license than the code, kept deliberately separate.
