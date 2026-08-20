# satellite-rl

Open-source reinforcement learning for satellite collision avoidance:
an RL agent learns when to burn fuel to dodge a predicted close approach
("conjunction") with another object, versus when to wait for a better
risk estimate — trading collision risk against limited fuel.

Built on [`bsk_rl`](https://github.com/AVSLab/bsk_rl) (real orbital
dynamics via Basilisk), with conjunction scenarios grounded in real
historical data from ESA's Kelvins Collision Avoidance Challenge dataset.

**Status: early scaffolding (Phase 0 of the roadmap). Not yet functional.**

## Why this project exists

See [`docs/00-plan.md`](docs/00-plan.md) for the full design process and
[`docs/01-problem-scope.md`](docs/01-problem-scope.md) for what's in v1 vs.
deferred. Short version: real conjunction warnings arrive as a *sequence*
of refining risk estimates over several days, not a single alert — so this
is genuinely a sequential decision problem under shrinking uncertainty,
which is exactly the kind of problem RL is suited to. The full reasoning,
including the collision-probability math, dataset licensing research, and
every design decision (state/action/reward/evaluation), is written up in
[`docs/`](docs/) as a sequence of numbered design docs (`01` through `13`).

## Repository layout

```
docs/                 design docs (read 00-plan.md first)
data/
  kelvins_cdm/          ESA Kelvins CDM dataset (CC-BY-4.0), see SOURCE.md
  celestrak_cache/       gitignored, fetched at runtime
  scratch/                gitignored
src/satellite_rl/
  pc/                    collision-probability (Pc) computation
  scenario/              conjunction scenario generator
  env/                   the Gymnasium environment
  training/               PPO training scripts
  eval/                   baselines + evaluation
tests/
scripts/               data-download utilities
```

## Installation

1. **Install Basilisk first** — it is not a normal pip dependency.
   Follow the [AVSLab install docs](https://avslab.github.io/basilisk/);
   `pip install "bsk[all]"` covers most platforms with a prebuilt wheel.
   macOS/Linux are preferred over Windows.
2. `python3 -m venv .venv && source .venv/bin/activate`
3. `pip install -e ".[dev]"`
4. `python scripts/download_kelvins.py` — fetches the Kelvins dataset into
   `data/kelvins_cdm/` (not yet implemented — Phase 2).
5. `pytest`

## License

Code: MIT. The `data/kelvins_cdm/` dataset is CC-BY-4.0 (ESA/Kelvins,
see `data/kelvins_cdm/SOURCE.md`) — a different license than the code,
kept deliberately separate; see `docs/13-roadmap.md` Phase 8.
