# 03 — Orbital Scenario & Conjunction-Event Modeling

Synthesizes `02-bsk_rl-architecture.md` (the mechanism: `ConjunctionDynModel`
+ `createNewEvent`, no built-in scenario scripting) and `04/05` (the Pc math
and real-world grounding data). This is the biggest genuine engineering gap
identified so far — bsk_rl gives us primitives, not a scenario generator —
so this doc is where we design that generator.

## The key real-world fact this scenario design is built around

A real conjunction is not a single alert — the Kelvins dataset shows
**~12 CDMs per event on average**, issued over the days leading up to TCA,
each with a *refined* (generally shrinking) covariance as ground-based
tracking gets more observations. This matters for the RL problem: it's not
"see one risk number, decide once" — it's a **sequential decision problem
under shrinking uncertainty**, where waiting for a better estimate has
value, but waiting too long makes any needed maneuver more costly/less
effective (an operational reality: avoidance burns planned further from TCA
need less Δv for the same miss-distance change).

**Design decision**: v1 models each conjunction as a **sequence of
CDM-like updates** over the episode (not a single snapshot), directly
mirroring the real data's structure. This is both more realistic and a
more interesting RL problem (explore/wait value is real, not just a risk
threshold classifier).

## Generating a conjunction: the targeting problem

`ConjunctionDynModel` (verified in `02`) needs a **second dynamics object**
on a trajectory that produces a chosen close approach with our ego
satellite at a chosen time. Basilisk integrates forward from initial
conditions — it has no "encounter targeting" API — so we solve for the
secondary's initial state ourselves:

1. Sample a desired encounter geometry at TCA: miss-distance vector and
   relative-velocity vector in the encounter plane, **drawn from
   distributions fit to the Kelvins dataset's real `miss_distance`,
   `relative_speed`, and covariance columns** (not invented numbers) —
   see "Sampling realistic geometry" below.
2. Propagate the ego satellite's own (known) orbit forward to the chosen
   TCA to get its inertial state at that instant.
3. Compute the secondary's inertial state at TCA = ego's TCA state +
   the sampled relative state (converted from encounter-plane/RTN
   coordinates to inertial).
4. **Backward-propagate** that TCA state to the episode's start time to
   get the secondary's initial condition. This backward-propagation step
   only needs two-body/J2 fidelity (it's an initial-condition-targeting
   utility, not the simulated dynamics that matters for training) — use
   `hapsira` (the maintained poliastro fork) for this, kept as a
   scenario-generation-time dependency, separate from Basilisk which runs
   the actual training-time dynamics.
5. Register the secondary as a second `Satellite`/dynamics object in the
   `bsk_rl` sim with that initial state, and use `ConjunctionDynModel`'s
   `createNewEvent` mechanism (per `02-bsk_rl-architecture.md` §9) both for
   the terminal actual-collision check and — more importantly for us — for
   our own scripted callback that fires at each CDM-update time to
   recompute Pc from the (still Basilisk-propagated, ground-truth) relative
   state plus an evolving covariance model (see next section) and push it
   into the observation.

## Sampling realistic geometry from Kelvins statistics

Rather than hand-picking miss distances/covariances, v1 fits simple
distributions (log-normal is a reasonable first choice for miss distance
and relative speed, given they're strictly positive and right-skewed in
orbital mechanics) to the relevant Kelvins columns:
- `miss_distance`, `relative_speed` — encounter geometry.
- `c_sigma_r/t/n`, `t_sigma_r/t/n` (and cross-terms) — covariance
  magnitude and shape, combined per the Pc doc's method.
- `risk` — used **only for validation** (does our generated scenario's
  computed Pc land in a realistic range compared to real events with
  similar miss-distance/covariance inputs?), never as a direct input to
  scenario generation (that would be circular).

This keeps the simulator's difficulty distribution empirically grounded
instead of arbitrary, and gives us a natural train/held-out-eval split:
fit distributions on a training partition of Kelvins events, hold out a
partition of *actual* real events (full CDM sequences, real outcomes) for
the v1 success-criterion #2 sanity check (`01-problem-scope.md`).

## Covariance evolution across the CDM sequence

Real covariance shrinks (usually) as TCA approaches and tracking improves.
v1 models this simply: interpolate covariance magnitude between an initial
(larger, first-CDM-like) value and a final (smaller, last-CDM-like) value
sampled from the Kelvins dataset's *actual* first-CDM vs. last-CDM
covariance ratio for real events, rather than inventing a shrink rate. Not
every real event's covariance shrinks monotonically (tracking gaps or
maneuvers by either object can cause jumps) — v1 explicitly does **not**
model those anomalies; monotonic shrink is a deliberate simplification,
noted here so it isn't silently assumed to be more realistic than it is.

## Background object population (secondary priority for v1)

For v1's core loop, a single scripted conjunction per episode (against an
otherwise-empty background) is sufficient to make the core risk/fuel
tradeoff learnable and testable. A realistic background debris/satellite
field (sampled from CelesTrak's `active` and fragmentation-debris groups,
per `05-datasets.md`) is a **v1.1 enhancement**, not a v1 blocker — it adds
scenario realism (occasional coincidental close approaches from objects the
agent isn't being tested on) but isn't required to answer the core research
question. Flagging this explicitly so it doesn't silently expand v1 scope.

## Difficulty / curriculum design

Ordered by what we'll actually implement first:

1. **Single conjunction, fixed geometry** — one hand-picked, realistic
   (Kelvins-typical) encounter per episode, deterministic. Purpose: prove
   the pipeline (targeting → sim → Pc → reward → training) works at all
   before adding randomness.
2. **Single conjunction, sampled geometry** — miss distance/relative
   speed/covariance drawn per-episode from the fitted Kelvins
   distributions (previous section). Purpose: agent must generalize across
   the realistic difficulty range instead of memorizing one scenario.
3. **Single conjunction, sampled geometry + CDM-sequence uncertainty
   evolution** — the shrinking-covariance sequence described above, so the
   wait-vs-act tradeoff is real. This is the actual v1 target environment.
4. **(v1.1+) Multiple simultaneous/sequential conjunctions per episode**,
   and/or a populated background field — deferred per `01-problem-scope.md`.

## Open technical risk to flag honestly

The backward-propagation targeting step (via `hapsira`, two-body/J2) will
not perfectly match Basilisk's own 10th-degree-spherical-harmonics forward
propagation (per `02-bsk_rl-architecture.md` §2) — the secondary's actual
TCA state, once forward-propagated by Basilisk from our computed initial
condition, will drift slightly from the exact targeted miss distance. For
v1 this is acceptable (real conjunction geometry has this kind of
uncertainty anyway, and it's small over the episode timescales involved —
hours to a few days), but it should be **measured and reported**, not
assumed away: part of implementing this generator is a validation script
that checks realized-vs-targeted miss distance across a batch of sampled
scenarios and reports the error distribution. If the drift turns out
large, a closed-loop correction (re-target using Basilisk itself, or a
higher-fidelity two-body propagator) becomes a follow-up task.
