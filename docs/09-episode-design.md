# 09 — Episode & Training Scenario Design

Builds on `03-scenario-design.md` (CDM-sequence structure, curriculum
stages), `06-state-space.md`, `07-action-space.md`, `08-reward-function.md`.

## Episode = one conjunction's CDM sequence

One episode = one scripted conjunction event, decided step-by-step across
its CDM-update schedule, ending at (or just before) TCA. This directly
matches the decision cadence established in `03` and `07`.

## Decision-point schedule

**v1 (curriculum stages 1–2, per `03-scenario-design.md`)**: a **fixed**
schedule of decision points, e.g. `T-5d, T-3d, T-1d, T-12h, T-2h, TCA` (6
steps) — chosen for implementation simplicity while the pipeline is being
built and debugged (Phases 1–4 in `13-roadmap.md`). Fixed-length episodes
are easier to reason about when validating the pipeline end-to-end.

**v1 curriculum stage 3 (the actual target environment)**: decision points
**sampled from the real distribution of `time_to_tca` values observed
across CDMs within a Kelvins event** (per-event schedules in the real data
are irregular and vary in count, averaging ~12 — see `05-datasets.md`).
This is a variable-length episode (different events get different numbers
of decision points) — Gymnasium/`bsk_rl` handle this natively (episode
length is just whenever `terminated`/`truncated` fires), so this is not an
engineering blocker, just a note that evaluation code must handle variable
episode lengths correctly (`11-evaluation.md`).

## Reset logic

On `reset()`:
1. Sample a scenario (miss-distance/relative-speed/covariance geometry,
   HBR, decision-point schedule) per `03-scenario-design.md`'s sampling
   procedure for the active curriculum stage.
2. Run the targeting solver (backward-propagation via `hapsira`) to get
   the secondary object's initial state.
3. Register the secondary as a second dynamics object and set up the
   `ConjunctionDynModel` terminal check plus our own scripted
   `createNewEvent` callbacks at each decision point.
4. Reset ego satellite's fuel budget (`dv_available`) to full.

## Termination / truncation

- **Terminated** (true collision, per `ConjunctionDynModel`'s built-in
  check, per `02-bsk_rl-architecture.md` §9): included as a safety-net
  check even though, per `08-reward-function.md`, we don't rely on it as
  the primary reward signal — if it ever fires, that's a signal something
  is badly wrong (either the scenario generator produced an unrealistic
  guaranteed-hit geometry, or the policy is catastrophically bad), worth
  logging distinctly for debugging rather than just folding into the
  normal reward path.
- **Terminated (normal case)**: reaching the final decision point (at or
  just past TCA), where the terminal risk-term reward is computed per
  `08-reward-function.md`.
- **No truncation by external time limit** in v1 — episode length is
  fully determined by the (fixed or sampled) decision-point schedule, so a
  separate `time_limit` truncation shouldn't trigger under normal
  operation; it's still configured (via `bsk_rl`'s `time_limit` env
  constructor arg, per `02` §1) as a safety bound in case a scenario's
  targeting solver produces a degenerate/unexpectedly long schedule.

## Curriculum progression mechanics

Not automatic/adaptive in v1 — curriculum stage is a config value we
change manually between training runs (Phase 5, `13-roadmap.md`), not a
learned or automatically-triggered progression. This is a deliberate
simplification: automatic curriculum learning is itself a research topic
bsk_rl's docs mention as an available training recipe (per
`02-bsk_rl-architecture.md` §7) — worth revisiting only after v1's fixed
3-stage manual progression is working and validated, not before.

## Train/eval scenario split

Per `03-scenario-design.md`, Kelvins events are split into a fitting
partition (used to derive the sampling distributions that generate
*synthetic* training episodes) and a held-out partition (real events,
replayed directly — not resampled — as an out-of-distribution sanity
check, per `01-problem-scope.md` success criterion #2 and detailed in
`11-evaluation.md`). Training never sees the held-out partition's actual
events, only the distributional statistics fit from the training
partition — this is what makes the held-out replay a meaningful check
rather than circular.
