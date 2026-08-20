# 12 — Software Architecture & Repo Structure

## Proposed layout

```
satellite-rl/
  docs/                      # this design doc series (already exists)
  data/
    kelvins_cdm/              # CC-BY-4.0 dataset -- gitignored except SOURCE.md;
                                # train_data.csv is 232 MB, over GitHub's 100 MB
                                # hard per-file limit (found in Phase 1), so this
                                # is fetched via scripts/download_kelvins.py, not committed
      SOURCE.md                # Zenodo DOI, checksum, download date, license text
    celestrak_cache/           # gitignored, fetched at runtime
    scratch/                   # gitignored
  src/satellite_rl/
    pc/                        # collision-probability module (04)
      foster.py                 # numerical double-integral reference implementation
      chan.py                   # fast series-expansion implementation (training-time)
      __init__.py                # public compute_pc() dispatching to the configured method
    scenario/                  # conjunction scenario generator (03)
      distributions.py          # fit + sample from Kelvins statistics
      targeting.py               # hapsira-based backward-propagation solver
      cdm_schedule.py            # decision-point schedule generation (fixed + sampled)
    env/                       # bsk_rl-based Gymnasium env (06, 07, 08, 09)
      satellite.py               # custom Satellite subclass (dyn_type/fsw_type, action_spec, obs_spec)
      conjunction_dyn.py         # ConjunctionDynModel subclass + createNewEvent callbacks
      reward.py                  # ComposedReward implementation
    training/                  # SB3 PPO training scripts, configs (10)
    eval/                      # baselines, metrics, Kelvins-replay validation (11)
  tests/
    test_pc.py                  # Pc validated against exact closed-form + Monte Carlo (see 14)
    test_targeting.py           # realized-vs-targeted miss distance error distribution
    test_env.py                 # Gymnasium API compliance (gymnasium.utils.env_checker)
  scripts/
    download_kelvins.py          # Zenodo fetch + checksum verification
    fetch_celestrak.py           # rate-limited CelesTrak GP data fetch, 2h+ cache
  pyproject.toml
  README.md
```

## Dependency management

`pyproject.toml`, core runtime dependencies:
- `bsk-rl` (MIT) — requires Basilisk installed separately first (per
  `02-bsk_rl-architecture.md` §8; document this explicitly in README as a
  pre-`pip install` step, since it's not a normal transitive pip
  dependency — platform-specific, `pip install "bsk[all]"` covers most
  platforms per verified findings).
- `gymnasium`
- `hapsira` (poliastro's maintained fork, per `03-scenario-design.md`'s
  targeting solver) — **not** `poliastro` (archived, per earlier research).
- `numpy`, `scipy` (Pc numerical integration).
- `stable-baselines3` (per `10-rl-algorithm.md`).
- `sgp4` (per `05-datasets.md`, if/when CelesTrak background population is
  added in v1.1).

Dev/optional dependencies:
- `orekit_jpype` — **dev/test-only**, per `04-collision-probability.md`'s
  decision to validate against it but not depend on it at runtime. Keep
  this in a `[project.optional-dependencies].dev` group specifically so a
  normal `pip install satellite-rl` never pulls in a JVM dependency.
- `pytest`, `matplotlib` (eval plots).

## Installation instructions (README outline)

1. Install Basilisk (platform-specific, link to AVSLab docs; note macOS/
   Linux preferred per `02` §8).
2. `pip install -e ".[dev]"` for local development, or `pip install
   satellite-rl` once packaged for plain use.
3. `python scripts/download_kelvins.py` — fetches and checksums the
   Zenodo dataset into `data/kelvins_cdm/`.
4. `pytest` — should pass with no additional setup beyond the above (no
   live Space-Track/CelesTrak network dependency in the core test suite —
   those are fetched-at-runtime/cached, not required for correctness
   tests of the Pc/env/targeting logic, which should be testable against
   the committed Kelvins data and synthetic fixtures alone).

## Why this structure

The `pc/`, `scenario/`, and `env/` separation mirrors the actual
dependency direction established across the design docs: `env/` depends on
`pc/` and `scenario/`, but `pc/` and `scenario/` don't depend on `env/` or
on Basilisk at all (the Pc module is pure math; the scenario/targeting
module needs only `hapsira`, not Basilisk) — meaning both can be unit
tested fast, without spinning up a full Basilisk simulation, which matters
a lot for iteration speed and CI runtime on an open-source repo where
contributors' first PR shouldn't require debugging a slow full-stack test.
