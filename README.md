# satellite-rl

Open-source reinforcement learning for satellite collision avoidance:
an RL agent learns when to burn fuel to dodge a predicted close approach
("conjunction") with another object, versus when to wait for a better
risk estimate — trading collision risk against limited fuel.

Built on [`bsk_rl`](https://github.com/AVSLab/bsk_rl) (real orbital
dynamics via Basilisk), with conjunction scenarios grounded in real
historical data from ESA's Kelvins Collision Avoidance Challenge dataset.

**Status: Phase 2 of 9 complete** (collision-probability module validated
against real historical data; scenario-generation distributions fitted).
No trainable environment yet — see [`docs/13-roadmap.md`](docs/13-roadmap.md).

## Why this project exists

See [`docs/00-plan.md`](docs/00-plan.md) for the full design process and
[`docs/01-problem-scope.md`](docs/01-problem-scope.md) for what's in v1 vs.
deferred. Short version: real conjunction warnings arrive as a *sequence*
of refining risk estimates over several days, not a single alert — so this
is genuinely a sequential decision problem under shrinking uncertainty,
which is exactly the kind of problem RL is suited to. The full reasoning,
including the collision-probability math, dataset licensing research, and
every design decision (state/action/reward/evaluation), is written up in
[`docs/`](docs/) as a sequence of numbered design docs, including results
write-ups as each phase lands (`docs/14-pc-validation-results.md`,
`docs/15-distribution-fitting-results.md`, ...).

## Repository layout

```
docs/                 design docs (read 00-plan.md first)
data/
  kelvins_cdm/          ESA Kelvins CDM dataset (CC-BY-4.0) -- gitignored
                          except SOURCE.md; train_data.csv alone is 232 MB,
                          over GitHub's 100 MB push limit, so fetch it with
                          scripts/download_kelvins.py, don't expect it committed
  fitted/                small derived artifacts (fitted distributions, the
                          real per-event geometry table, schedule library) --
                          committed, see docs/15-distribution-fitting-results.md
  celestrak_cache/       gitignored, fetched at runtime
  scratch/                gitignored
src/satellite_rl/
  pc/                    collision-probability (Pc) computation
  scenario/              conjunction scenario generator + Kelvins data loading
  env/                   the Gymnasium environment
  training/               PPO training scripts
  eval/                   baselines + evaluation
tests/
scripts/               data-download + validation utilities
```

## Installation

1. **Install Basilisk first** — it is not a normal pip dependency.
   Follow the [AVSLab install docs](https://avslab.github.io/basilisk/);
   `pip install "bsk[all]"` covers most platforms with a prebuilt wheel.
   macOS/Linux are preferred over Windows.
2. `python3 -m venv .venv && source .venv/bin/activate`
3. `pip install -e ".[dev]"`
4. `python scripts/download_kelvins.py` — fetches + checksums the Kelvins
   dataset into `data/kelvins_cdm/` (232 MB, not committed to the repo).
5. `pytest`
6. Optional, requires step 4 first: `python scripts/validate_pc_against_kelvins.py`
   and `python -m satellite_rl.scenario.distributions` reproduce the
   Phase 1/2 results docs against the real data.

## License

Code: MIT. The `data/kelvins_cdm/` dataset is CC-BY-4.0 (ESA/Kelvins,
see `data/kelvins_cdm/SOURCE.md`) — a different license than the code,
kept deliberately separate; see `docs/13-roadmap.md` Phase 8.
