# satellite-rl — Research & Design Plan

Open-source project: reinforcement learning for satellite collision avoidance
(and, as a stretch goal, constellation task scheduling), built on `bsk_rl`
(Basilisk) with real conjunction-probability math and real historical
conjunction data (ESA Kelvins CDM dataset) grounding the scenario generator.

Goal of this planning phase: think through every design decision *before*
writing simulation/training code, and leave a paper trail of docs explaining
why each decision was made. Each item below becomes its own doc in this
folder as we work through it.

## Status legend
- [ ] not started
- [~] in progress
- [x] done — see linked doc

## Todo list

1. [x] **Problem scope** — v1 scope (single-satellite collision avoidance)
   vs. stretch scope (multi-satellite constellation scheduling). What's in
   v1, what's explicitly deferred. → `01-problem-scope.md`
2. [x] **bsk_rl / Basilisk environment architecture** — how `bsk_rl` actually
   expects you to build a custom Gymnasium environment: satellite/dynamics/
   FSW model conventions, existing example environments, multi-satellite
   support, what we get for free vs. what we must build ourselves.
   → `02-bsk_rl-architecture.md`
3. [x] **Orbital scenario & conjunction-event modeling** — how conjunction
   (close-approach) events get injected into an episode: synthetic geometry
   sampling vs. grounding in real historical event statistics, background
   object population (debris field), scenario difficulty/curriculum.
   → `03-scenario-design.md`
4. [x] **Collision probability (Pc) computation** — which formula(s) to use
   (2D Foster/Alfriend etc.), what inputs are required (covariance, combined
   hard-body radius, miss distance, relative velocity), Orekit vs. a
   from-scratch Python implementation. → `04-collision-probability.md`
5. [x] **Datasets** — ESA Kelvins CDM dataset (schema, size, download,
   preprocessing), CelesTrak TLEs, Space-Track (optional), directory layout
   under `data/`, licensing/redistribution constraints. → `05-datasets.md`
6. [x] **State (observation) space design** — exact observation vector,
   normalization, how uncertainty/covariance is represented, partial
   observability. → `06-state-space.md`
7. [x] **Action space design** — discrete vs. continuous maneuver decisions,
   mapping RL actions to Basilisk thruster/Δv commands, action timing and
   frequency. → `07-action-space.md`
8. [x] **Reward function design** — collision-risk term, fuel-cost term,
   mission-utility term, weighting/shaping, terminal conditions.
   → `08-reward-function.md`
9. [x] **Episode & training scenario design** — episode length/horizon,
   reset logic, scenario sampling distribution, curriculum learning.
   → `09-episode-design.md`
10. [x] **RL algorithm & training infrastructure** — algorithm choice,
    library (SB3 vs. CleanRL vs. RLlib), handling the hybrid action space,
    hyperparameters, compute budget. → `10-rl-algorithm.md`
11. [x] **Baselines & evaluation methodology** — heuristic baselines
    (always-avoid, threshold rule, no-action), validation against real
    historical ESA CDM outcomes, metrics. → `11-evaluation.md`
12. [x] **Software architecture & repo structure** — how everything fits
    together: package layout, dependency management, install instructions.
    → `12-architecture.md`
13. [x] **Milestone roadmap** — phased implementation plan with concrete,
    checkable deliverables per phase. → `13-roadmap.md`

## Status: planning phase complete

All 13 items researched and documented (Aug 2026). Next actual work is
Phase 0 of `13-roadmap.md` (repo scaffolding) — this plan doc's job is
done; implementation progress now tracks against the roadmap doc instead.

## Working method

We go through items 1–13 roughly in order (later items depend on earlier
ones), research each with live, verified sources where facts are needed
(library APIs, dataset schemas, formulas), and write a doc before moving on.
This file gets updated (status + links) as each doc lands.
