# 13 — Milestone Roadmap

Phased so that every phase produces something runnable/checkable — no
phase depends on "trust me, it'll come together later."

## Phase 0 — Scaffolding — DONE (Aug 2026)
- Repo skeleton per `12-architecture.md`, `pyproject.toml`, CI (lint +
  test, no Basilisk dependency needed for lint).
- Basilisk + `bsk-rl` install verified locally, `import bsk_rl` succeeds.
- **Deliverable**: empty package installs and imports cleanly.
- **Verified locally** (macOS 26.2, arm64, Python 3.11.4, `.venv`):
  `pip install "bsk[all]"` pulled a prebuilt `bsk-2.11.1` wheel (no
  from-source build needed on this platform), `bsk-rl==1.3.4`,
  `hapsira==0.18.0`, `gymnasium==1.3.0`, `stable-baselines3==2.9.0`,
  `scipy==1.17.1` all installed without conflicts. `import bsk_rl`,
  `from bsk_rl.gym import SatelliteTasking`, and `import hapsira` all
  succeed. `ruff check` and `pytest` (1 placeholder test) both pass.
  Git repo initialized locally, not yet committed.

## Phase 1 — Collision probability module — DONE (Aug 2026)
- Implement Foster's method (`pc/foster.py`) — numerical double integral.
- Unit tests against textbook/paper worked examples.
- Implement Chan's method (`pc/chan.py`), cross-validated against Foster
  on a batch of cases (should agree closely; document observed max
  deviation, don't just assert agreement).
- Validate both against a sample of Kelvins dataset events using their
  reported `risk` column.
- **Deliverable**: `compute_pc()` with a test suite and a short written
  validation summary (add as `docs/14-pc-validation-results.md` once run —
  not written yet, since it depends on actual numbers).
- **Done**: `src/satellite_rl/pc/{geometry,foster,chan}.py` implemented —
  Chan ported verbatim from Orekit's real Java source (not reconstructed
  from memory), Foster via direct `scipy.integrate.dblquad`. 24/24 tests
  pass in `tests/test_pc.py`, validated against an independent exact
  closed-form (noncentral chi-squared) and Monte Carlo, not just
  self-consistency. External validation against 3,000 real ESA Kelvins
  events: 0 computation failures, Spearman r = 0.93 (all events) / 0.77
  (excluding floored-risk events) between our computed Pc and ESA's real
  reported risk. Full results and known limitations in
  `14-pc-validation-results.md`. Kelvins dataset downloaded, checksummed,
  and several dataset facts corrected from the earlier secondhand research
  summary (see `05-datasets.md`'s "Corrections" note) based on actually
  opening the file.

## Phase 2 — Data pipeline — DONE (Aug 2026)
- `scripts/download_kelvins.py` (Zenodo fetch + checksum).
- Fit distributions (`scenario/distributions.py`) to Kelvins
  miss-distance/relative-speed/covariance columns, per `03-scenario-
  design.md`.
- `scripts/fetch_celestrak.py` (rate-limited, cached) — lower priority,
  needed only once v1.1's background population is in scope. **Not done
  yet** — deferred, since it's only needed for the background-object
  population enhancement (`03-scenario-design.md`), not v1's core loop.
- **Deliverable**: distribution-fitting script producing plots/summary
  stats, checked into `data/` (fitted parameters, not raw re-derivable
  data) or a results doc.
- **Done**: `src/satellite_rl/scenario/{kelvins_loader,distributions}.py`
  implemented. Real finding, reported honestly rather than glossed over:
  independent per-parameter lognormal fits are statistically poor
  (KS-rejected for every parameter) and structurally lose real
  cross-parameter correlation — **pivoted to joint bootstrap resampling**
  (`sample_scenario_geometry_bootstrap()`) as the recommended method,
  sampling real event tuples from `data/fitted/geometry_events.csv`
  instead of independent marginals. Covariance shrink ratio measured at a
  real, substantial **8.36× median** (first-CDM vs. final-CDM). Schedule
  library of 5,000 real event CDM-timing sequences extracted for
  `09-episode-design.md`'s bootstrap-resampled schedules. Full results in
  `15-distribution-fitting-results.md`; `03-scenario-design.md` updated to
  match. 4 new tests (synthetic-data-only, since the real dataset isn't
  in CI), 27/27 passing.

## Phase 3 — Scenario generator (targeting) — DONE, with one item carried to Phase 4 (Aug 2026)
- `scenario/targeting.py`: backward-propagation solver (`hapsira`) for a
  secondary object's initial state given a desired TCA encounter geometry.
- `tests/test_targeting.py`: batch validation of realized-vs-targeted miss
  distance once forward-propagated through actual Basilisk dynamics — per
  `03-scenario-design.md`'s flagged open risk, this is where we find out
  how large that drift actually is.
- **Deliverable**: targeting solver with measured accuracy numbers
  (not assumed accuracy).
- **Done**: `src/satellite_rl/scenario/targeting.py` implemented and
  thoroughly validated — geometry-reproduction exact to floating-point
  precision (20 trials), self-consistency errors floating-point-perfect
  in the median case with a real, root-caused tail (up to ~170m, from
  hyperbolic-orbit numerical precision at high sampled relative speed).
  68/68 tests passing. Real dependency bug found and fixed along the way:
  `hapsira` 0.18.0 (its only PyPI release) is broken against current
  astropy; pinned `astropy<7` in `pyproject.toml`, verified `bsk-rl` still
  works fine with the older astropy too.
- **Carried to Phase 4, not resolved here**: the Basilisk-fidelity
  cross-check (how much does real 10th-degree-spherical-harmonics
  dynamics diverge from the two-body targeting model). Extensive
  debugging found and fixed 3 real Basilisk API bugs but didn't converge
  to a trustworthy number — decided to defer this to Phase 4, using
  `bsk_rl`'s own (already-correct) `DynamicsModel` instead of continuing
  to hand-roll raw Basilisk scripting. Full honest writeup in
  `16-targeting-validation-results.md`; `03-scenario-design.md`'s "open
  technical risk" section updated to match. `scripts/
  validate_targeting_against_basilisk.py` kept, clearly marked as not
  yet a trustworthy result, for whoever picks this up in Phase 4.

## Phase 4 — Custom Gym environment (fixed scenario)
- `env/satellite.py`, `env/conjunction_dyn.py`, observation/action/reward
  wiring per `06`, `07`, `08`, curriculum stage 1 (single fixed geometry,
  fixed schedule, per `03`/`09`).
- **Carried over from Phase 3**: the Basilisk-fidelity cross-check
  (realized-vs-targeted miss distance once the targeting solver's initial
  conditions are actually flown through `bsk_rl`'s `DynamicsModel`) — do
  this using the real custom `Satellite`/env being built here, not a
  separate hand-rolled script; see `16-targeting-validation-results.md`.
- `gymnasium.utils.env_checker` compliance test.
- Benchmark single-env step throughput (informs `10-rl-algorithm.md`'s
  compute-budget planning).
- Sanity check with a random policy — confirm reward signs/magnitudes
  behave as designed (e.g. always-max-Δv policy incurs high fuel penalty,
  never-act policy's terminal risk penalty responds to Pc as expected).
- **Deliverable**: a working, tested Gym env on the simplest scenario,
  plus the Basilisk-fidelity numbers `03`/`16` deferred here.

## Phase 5 — Curriculum stages 2–3
- Sampled geometry (stage 2), then CDM-sequence uncertainty evolution
  (stage 3, the actual v1 target environment) per `03-scenario-design.md`.
- **Deliverable**: full v1 environment, still validated with a random/
  scripted policy before any learning is attempted.

## Phase 6 — Training
- SB3 PPO, starting hyperparameters per `10-rl-algorithm.md`, tuned based
  on observed training curves.
- Reward-weight sensitivity analysis per `08-reward-function.md`.
- **Deliverable**: a trained policy checkpoint + training curves.

## Phase 7 — Evaluation
- All four baselines (`11-evaluation.md`), full metric suite, held-out
  Kelvins-event replay validation.
- **Deliverable**: results write-up — including an honest report if PPO
  doesn't beat the threshold heuristic, per `11`'s "honest negative
  results" section.

## Phase 8 — Open-source polish
- README, install instructions, worked example notebook, CI green,
  license file (note: MIT/Apache2 recommended for the code itself; the
  Kelvins data subdirectory needs its own CC-BY-4.0 attribution per
  `05-datasets.md`, distinct from the code's license — don't blanket the
  whole repo under one LICENSE file without carving this out correctly).
- **Deliverable**: repo ready for a public release / blog-post-style
  write-up of Phase 7's results.

## Phase 9 (stretch, v2) — Constellation scheduling
- Multi-satellite extension via `ConstellationTasking`, per `02-bsk_rl-
  architecture.md` §6, likely migrating training to RLlib per `10-rl-
  algorithm.md`'s noted path. Explicitly out of scope until v1 (Phases
  0–8) is complete and validated, per `01-problem-scope.md`.

## Sequencing notes

Phases 1–3 (Pc, data, targeting) have no dependency on each other and
could be parallelized by a team; for a solo learning-focused project,
sequential order is preferable so each piece is understood before the
next is built on top of it — this roadmap is written in the order it will
actually be worked, not the order that would minimize wall-clock time for
a team.
