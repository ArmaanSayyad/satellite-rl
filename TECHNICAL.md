# Technical Report

This document is the single technical write-up for the project: the
problem, the data, the system architecture, the environment/training
design, the empirical results, and — deliberately, at length — the bugs
that were found and fixed along the way and what's still unresolved.
It replaces a series of 28 phase-by-phase design/results docs written
during development; those are preserved outside this repository (not
deleted) for anyone who wants that level of detail, but everything
load-bearing is consolidated here.

Every number in this document comes from either a real, cited source
(NASA CARA reports, the ESA Kelvins dataset, verified library source
code) or an actual run of this project's own code — nothing here is an
estimate presented as a measurement. Where a result was negative,
partial, or still-unresolved at the end of development, it's reported
as such, not smoothed over.

## Table of contents

1. [Problem statement](#1-problem-statement)
2. [Data](#2-data)
3. [System architecture](#3-system-architecture)
4. [Environment design (the MDP)](#4-environment-design-the-mdp)
5. [Training setup](#5-training-setup)
6. [Development history: what was found and fixed](#6-development-history-what-was-found-and-fixed)
7. [Results](#7-results)
8. [Web demo](#8-web-demo)
9. [Limitations and future work](#9-limitations-and-future-work)
10. [Repository layout](#10-repository-layout)

---

## 1. Problem statement

Operators of satellites in low Earth orbit (LEO) periodically receive
**Conjunction Data Messages (CDMs)** — warnings that another tracked
object (debris or another satellite) is predicted to pass close to
their spacecraft at some future time. A real conjunction is not a
single alert: the ESA Kelvins dataset shows real events receive
**~12 CDMs on average**, refined over the days leading up to the
predicted closest approach (TCA — time of closest approach) as
ground-based tracking improves and the covariance (positional
uncertainty) shrinks. An operator decides, at each update, whether to
burn fuel to maneuver away from the risk or wait for a better estimate
— waiting is free of fuel cost but costs decision time, and a maneuver
planned closer to TCA needs more Δv for the same effect.

This is a genuine **sequential decision problem under shrinking
uncertainty** — not "see one risk number, decide once" — which is
exactly the class of problem reinforcement learning is suited to: an
agent that must trade a probabilistic risk (of collision) against a
resource cost (fuel) over a multi-step decision horizon, without
knowing the true outcome until it's too late to change course.

**Project scope (v1, what's actually built here)**: a single "ego"
satellite in a realistic LEO orbit, simulated with real orbital
dynamics, deciding whether/when to execute an impulsive maneuver
against one scripted conjunction per episode, where the conjunction's
geometry and risk are grounded in real historical ESA data rather than
invented numbers. Explicitly **out of scope**: multi-satellite
constellation scheduling (a natural v2 extension, needing multi-agent
RL), a populated background debris field, live data feeds, and
detailed attitude control during maneuvers (maneuvers are modeled as
impulsive Δv events, not a pointing/control problem).

**Success criteria set at the start of the project**: (1) the trained
policy should measurably outperform naive baselines on real,
genuinely-dangerous encounters; (2) its behavior should be sane when
checked against real historical events (not acting on negligible risk,
acting on real elevated risk); (3) the whole pipeline — scenario
generation, collision-probability math, environment, training,
evaluation — should be reproducible from a fresh clone. All three are
addressed in [§7](#7-results), including where the honest answer is
more nuanced than a clean "yes."

## 2. Data

**ESA Kelvins Collision Avoidance Challenge Dataset** (Zenodo DOI
`10.5281/zenodo.4463683`, CC-BY-4.0, see `data/kelvins_cdm/SOURCE.md`
for full attribution) is this project's grounding dataset: real,
anonymized CDMs ESA received 2015–2019. 162,634 rows / 13,154 unique
events in the training partition (~12 CDMs/event), 103 columns per
row including miss distance, relative speed, the full position
covariance (not just diagonal sigmas — it includes off-diagonal
cross-correlation terms, matching the standard CCSDS CDM layout), and
ESA's own reported risk assessment (`risk = log10(Pc)`, floored at
-30). No direct hard-body-radius column exists; this project uses
radar cross-section (`{t,c}_rcs_estimate`, present for ~67.5% of
events) as an approximate radius proxy via `r = sqrt(RCS/π)` — a real
physical proxy, not ground truth (radar reflectivity depends on
material/shape, not just size).

**Why this dataset and not others**: Space-Track.org's CDM archive is
restricted to objects an account owns (not public); CelesTrak/SOCRATES
gives lower-precision public TLEs without operator covariance data;
commercial providers (LeoLabs, Slingshot, Kayhan) are paid services
with no open research data. The Kelvins dataset is the one unambiguous
option that's both real (not synthetic) and legally redistributable.

**How it's actually used**: not as a single fitted parametric
distribution. An early plan (independent lognormal fits per parameter)
was implemented and tested — every parameter's fit was rejected by a
Kolmogorov-Smirnov test at overwhelming significance (KS statistics
0.10–0.25, meaning the fitted curve was off from the real data by
10–25 percentage points at its worst point) — see
[§6](#6-development-history-what-was-found-and-fixed). The project
pivoted to **joint bootstrap resampling**: scenarios are generated by
drawing an entire real event's `(miss_distance, relative_speed,
sigma_x, sigma_z, combined_radius, alignment_angle)` tuple from a
derived 8,672-row table (`data/fitted/geometry_events.csv`), preserving
real joint structure instead of independent marginals. A library of
5,000 real event CDM-timing schedules (1–22 CDMs/event, median 13) is
similarly bootstrap-sampled for episode structure, not parametrically
generated.

## 3. System architecture

```
src/satellite_rl/
  pc/          collision-probability (Pc) computation -- pure math, no
               Basilisk dependency
    foster.py    numerical double-integral reference method (Foster 1992)
    chan.py      fast series-expansion method (Chan 1997), used at
                 training time; ported from Orekit's real Java source
    geometry.py  encounter-plane projection, covariance basis math
  scenario/    conjunction scenario generation -- needs `hapsira`, not
               Basilisk (backward-propagation targeting is a two-body/
               J2-fidelity problem, separate from the training-time
               high-fidelity simulation)
    distributions.py  Kelvins data loading, bootstrap sampling
    targeting.py       backward-propagation solver for the secondary
                        object's initial state given a desired TCA geometry
    tca_refinement.py  finds the true closest-approach time via real
                        Basilisk propagation; also the correction loop
                        used to hit real small (tens-of-meters) miss
                        distances precisely (see §6)
  env/         the Gymnasium environment, built on `bsk_rl`/Basilisk
    satellites.py            ego + passive-secondary Satellite subclasses
    scenario_sampling.py     SecondaryScenarioSampler (bootstrap draws,
                              risk-stratified pool, augmentation)
    observations.py          the live Pc computation hooked into the
                              observation
    collision_avoidance_env.py   reward, episode stepping, curriculum modes
  training/    PPO training (Stable-Baselines3) and evaluation
    train_ppo.py       training entry point
    baselines.py        threshold heuristic, hindsight oracle
    evaluate.py, full_evaluation.py   baseline comparison / metrics
```

**Why this split**: `pc/` and `scenario/` don't depend on Basilisk or
each other's heavier pieces, so they're fast to unit-test without
spinning up a full simulation — a fresh contributor's first PR doesn't
need to debug a slow full-stack test. `env/` depends on both; `training/`
depends on `env/`.

**Generating one conjunction** (what happens on `reset()`):
1. Draw a real event's geometry tuple from the bootstrap table (§2).
2. Propagate the ego satellite's own orbit forward to the chosen TCA.
3. Compute the secondary's target inertial state at TCA = ego's TCA
   state + the sampled relative geometry.
4. **Backward-propagate** that TCA state to the episode's start time
   (via `hapsira`, a J2-perturbed Cowell propagator) to get the
   secondary's initial condition — the actual state handed to Basilisk.
5. For small-miss-distance (high-risk) scenarios specifically, run a
   Newton/fixed-point correction loop (`correct_targeting_geometry`)
   that flies the candidate state through real Basilisk dynamics,
   measures the achieved miss vector, and corrects — because the J2-only
   backward-propagation model isn't accurate enough on its own at these
   scales (see [§6](#6-development-history-what-was-found-and-fixed)).
6. Register the secondary as a second Basilisk `Satellite` and run the
   actual episode under full high-fidelity dynamics (10th-degree
   spherical harmonics Earth gravity + Sun third-body perturbation +
   SPICE ephemeris).

Collision probability is computed **outside** Basilisk (pure Python,
from the relative state Basilisk exposes) via Chan's method at training
time (fast, series-expansion) with Foster's method (numerical double
integral, the historical reference) available for validation. No
external Pc library or JVM dependency (Orekit) is used at runtime —
only for offline cross-validation during development.

## 4. Environment design (the MDP)

**Observation space** (9 scalars, verified directly against the
shipped code, not the original design intent — see the note below):

| Feature | Description |
|---|---|
| `collision_prob` | Live-computed Pc, from the real simulated relative state |
| `dv_available` | Remaining fuel fraction (normalized) |
| `time_to_tca` | Time remaining to closest approach |
| `r_DC_N` (3 components) | Relative position (secondary − ego), normalized |
| `v_DC_N` (3 components) | Relative velocity, normalized |

*A note on honesty here*: the original design (see the archived phase
docs, `06-state-space.md`) called for the agent to observe a **noisy**
estimate of relative state (true state + noise drawn from the current
covariance) plus explicit covariance-shape features (σx, σz,
orientation), specifically to make this a genuinely partially-observable
problem. **What actually shipped is simpler**: the agent sees the true
relative state directly (no injected observation noise), and only the
resulting scalar Pc — not the covariance shape that produced it. This
was found late in development (Phase 7b) and is recorded here rather
than left as an undocumented gap between the design docs and the code.
It means the agent can't currently distinguish "close miss, tight
uncertainty" from "far miss, huge smear" if they happen to produce a
similar Pc, and there's no partial-observability pressure from
observation noise specifically (though the environment remains
partially observable in the sense that the agent never sees the
*true* future outcome at decision time).

**Action space**: a single continuous 3-vector, `[dv_R, dv_T, dv_N]` —
an impulsive Δv expressed in the ego satellite's own Hill/RTN frame
(radial/along-track/cross-track relative to its own orbit — the
standard way a real maneuver is described). There is no separate
discrete "wait" action; waiting is what the policy learns to output
when Δv ≈ 0, which is both simpler to train (PPO's native continuous
Gaussian policy head needs no custom hybrid action architecture) and
not a meaningful loss of realism (a real operator's "do nothing this
cycle" *is* choosing zero burn). Action magnitudes are bounded by
`max_dv` (10 m/s in the trained checkpoint used throughout this
report); a maneuver's coast duration is bounded by the time to the
next decision point.

**Reward**, computed per step:

```
reward_t = -w_fuel * |Δv_t| - w_disruption * 1[|Δv_t| > deadzone]
           + (risk_penalty  if t is the final step else 0)

risk_penalty = -w_risk * f(Pc_final)
f(Pc) = Pc / Pc_threshold                 if Pc <= Pc_threshold
f(Pc) = 1 + log(Pc / Pc_threshold)        if Pc  > Pc_threshold
```

`Pc_threshold = 1e-4` — verified against NASA CARA's real operational
maneuver-decision threshold, not an arbitrary round number (see
[§6](#6-development-history-what-was-found-and-fixed)). The risk term
uses the collision probability itself (not a binary "did a collision
happen" outcome) because realized collisions are, by design, extremely
rare even for a bad policy — a sparse binary signal would give almost
no training gradient. This mirrors real operational practice, which
also scores decisions against a probability threshold, not an
after-the-fact outcome.

**Episode structure**: one episode = one conjunction's decision
sequence. Three curriculum modes exist in the code:
1. **Fixed geometry, fixed schedule** — one deterministic scenario, used
   only to validate the pipeline works at all.
2. **Sampled geometry, fixed schedule** (`sample_geometry=True`) — a
   fresh real event's geometry is bootstrap-sampled every reset, on a
   fixed 4-decision-point schedule (`T-0.2d, T-0.1d, T-0.05d, T-0.01d,
   TCA`). **This is the mode actually used for training** (see
   [§5](#5-training-setup)).
3. **Sampled geometry + evolving uncertainty** (`+ evolve_uncertainty=
   True`) — additionally samples a real per-event CDM schedule
   (irregular, up to 20+ decision points) and lets covariance shrink
   within the episode (geometric interpolation between the sampled
   event's real first/last covariance — a real, measured effect:
   median shrink ratio **8.36×** across 8,482 real multi-CDM events).
   This is the full v1 target environment by design, but was measured
   at only **~1.3 steps/sec** single-process throughput (vs. ~8-9
   steps/sec for the fixed-schedule mode) — too slow for a feasible
   single-process training budget, so **it was never actually used for
   the training run reported in §7.** This is a real, honestly-flagged
   gap between the intended full environment and what was trained.

## 5. Training setup

**Algorithm**: PPO (Stable-Baselines3) — on-policy, native continuous
action support (fits the action space directly), and more forgiving of
reward-scale/hyperparameter mistakes than off-policy alternatives
(SAC/TD3), which mattered for a project where environment correctness
and algorithm tuning needed to be debugged separately, not conflated.

**Hyperparameters** (both training runs): `n_steps=64, batch_size=32,
gamma=0.95, gae_lambda=0.9, ent_coef=0.01, clip_range=0.2,
learning_rate=3e-4`. `n_steps=64` (much smaller than SB3's 2048
default) was a deliberate, measured choice: at 2048, even the fast
curriculum mode would need ~17 minutes of wall-clock time *before a
single gradient update*, making a realistic training budget afford only
a couple of updates total. `gamma=0.95` (not 0.99) fits the short
4-6-decision-step episodes. `ent_coef=0.01` (SB3 default is 0.0) adds
explicit exploration pressure against collapsing to an always-near-zero
action.

**Compute**: CPU-only (Basilisk has no GPU path; the policy network is
small enough that a GPU wouldn't help). The real bottleneck is
environment step throughput — each step is a real Basilisk propagation.
No environment parallelization (`SubprocVecEnv`) was used in either
training run; this is the clearest lever for a larger future run.

**Two training runs were done, not one** — reported separately in
[§7](#7-results) because the second run only happened after a scenario-
generator bug fix that fundamentally changed what the training data
could contain:

- **Run 1** ("Phase 6"): 5,000 timesteps, ~53 min wall clock, on the
  scenario generator as it existed before the fixes in
  [§6](#6-development-history-what-was-found-and-fixed).
- **Run 2** ("Phase 7g", the checkpoint shipped with this repo and used
  by the web demo): 8,000 timesteps, ~92 min wall clock, `targeting_
  seed=0`, `high_risk_fraction=0.5`, `high_risk_pool_fraction=0.01`,
  `high_risk_augment=True`, `high_risk_precise_targeting=True` — on the
  fixed scenario generator, deliberately oversampling real dangerous
  events (see [§6](#6-development-history-what-was-found-and-fixed)).

## 6. Development history: what was found and fixed

This project was built and validated in phases, each checked against
real data or real simulation output before moving to the next. The
project's standing practice throughout was to report negative and
partial results plainly rather than smooth them into a cleaner
narrative — several of the findings below directly overturned an
earlier phase's own conclusion. That history is preserved here because
it's the most concrete evidence the final result is trustworthy: every
major claim in this document was checked, not assumed, and wrong
conclusions were caught and corrected rather than left standing.

### Collision-probability math, validated against real ESA assessments

`compute_pc` (Foster and Chan methods) was validated three ways before
being trusted: against an independently-derivable exact closed-form
result (circular-covariance case, via the noncentral chi-squared CDF) —
agreement to `1e-6`; against 2-million-sample Monte Carlo on elliptical
covariance cases — agreement within 5× the Monte Carlo standard error;
and against **3,000 real ESA Kelvins events**, comparing this project's
independently-computed Pc to ESA's own reported risk assessment:
**Spearman r = 0.927** (all events) / **0.774** (excluding events at
ESA's reporting floor) — strong rank correlation using only public data
and an RCS-derived radius approximation, with zero computation failures
across all 3,000 real covariance matrices.

### J2 nodal precession: a 433× accuracy fix

Early validation compared the (two-body) targeting solver's predicted
trajectory against real Basilisk propagation and found a
~2,960,787-meter divergence over a 3-day, 500km-altitude orbit —
initially (incorrectly) attributed to unresolved bugs in raw Basilisk
API usage. Building the real environment on `bsk_rl` (which handles
Basilisk's gravity/SPICE setup correctly) reproduced the *same*
divergence, ruling that out. **Root cause, confirmed by directly
checking orbital elements**: real J2 nodal (RAAN) precession — the
orbit's ascending node drifted by measured `-14.31°` over 3 days,
matching the standard J2 secular-drift formula's prediction of
`-14.26°` almost exactly. The two-body targeting solver simply didn't
model this large, genuine physical effect. **Fix**: upgrading the
targeting propagator from plain two-body to a J2-perturbed Cowell
propagator cut the divergence to **~6,836 meters** — a **433× reduction**
— with the residual consistent with unmodeled higher-order terms (J3+,
tesseral/sectoral harmonics), small relative to realistic miss
distances.

### The scenario generator couldn't produce a genuinely dangerous episode — twice

**First finding (Phase 7)**: after the first training run (below),
held-out evaluation on 60 real events found **zero** episodes where
even doing nothing (`never_maneuver`) resulted in Pc above the 1e-4
threshold — confirmed not a fluke via a 300-episode sweep and a check
against the training seed itself (0/150). Investigated rather than
assumed: real Kelvins geometry is genuinely mostly benign (median miss
distance 14,458m vs. median combined radius ~1.1m), **and** the
environment's isotropic-covariance simplification (collapsing the two
real encounter-plane principal standard deviations to one geometric-mean
value) was compounding that — real encounter-plane covariance has a
median eccentricity of 5.76× (up to 4,248×), nowhere near isotropic, and
recomputing Pc with the real anisotropic shape shifted the exceedance
rate from 0% to 3.2% on the same sample.

**Fix (Phase 7b)**: preserve each real event's actual miss-vector/
covariance alignment (previously computed and then discarded before
being saved to the geometry table) and use real anisotropic covariance
in both scenario generation and the live Pc computation. **Result,
reported honestly**: the fix moved the achievable Pc ceiling by a real,
measured **~50×** (max observed Pc 4.5e-9 → 2.2e-7 across a 300-episode
sweep) but **did not by itself produce `Pc > 1e-4` training episodes**.
Computing Pc directly from the full real 8,672-event table with true
alignment (no simulation needed) found the honest reason why: **only 1
event in the entire real dataset (0.012%) has Pc above the 1e-4
threshold.** At that base rate, uniform bootstrap sampling would need
tens of thousands of episodes to reliably encounter it even once.

### Chasing one real actionable event to seven

**Risk-stratified sampling (Phase 7c)** added an opt-in mechanism to
oversample a real "elevated-risk" pool (top-N events by computed Pc)
instead of drawing uniformly. Verified working exactly as designed, but
honestly quantified as insufficient alone: with the pool still 99.8%
below-threshold internally, the expected-hit arithmetic for the single
known actionable event was exact and small (~1.15 hits per 1,000
episodes at the default settings) — oversampling a mostly-benign pool
doesn't manufacture more above-threshold examples.

**Augmentation and re-ranking (Phase 7d)** attacked the scarcity from
three angles: searching for more real data (a genuine dead end — every
other source checked was either not publicly redistributable or not a
CDM archive at all); generating physically-grounded synthetic variety
by resampling each real event's own reported covariance (validated:
**over 5,000 distinct actionable synthetic variants** generated from
110,000 draws across 8 real source events — not invented noise, a
direct statement of "here is the distribution of plausible true offsets
consistent with this real measurement"); and checking whether the
1e-4 threshold itself was miscalibrated (**no** — verified against NASA
CARA's real published threshold, which is exactly 1e-4). Along the way,
a real bug was found: this project's own Pc pipeline **under-counts**
risk relative to ESA's own assessment — 13 real events exceed `Pc >
1e-4` by ESA's own numbers, not 1; the combined-radius (RCS-based)
approximation was the main source of the gap, understating risk on
ESA's single riskiest real event by ~300×. **Fix**: rank the elevated-
risk pool by whichever of (this project's own Pc, ESA's reported Pc) is
higher — a safety-oriented choice — which took the pool's count of
known-actionable real events from 1 to **7**.

### Five bugs standing between "correctly identified" and "correctly simulated"

Even with 7 known-actionable real events identified and augmented for
diversity, the *actually simulated* outcome for these small
(tens-of-meters) miss-distance scenarios still didn't match what the
risk math predicted — sometimes by 50+ orders of magnitude. Root-causing
this (Phase 7e) found five separate, genuinely distinct bugs:

1. **Missing Sun gravity** in the correction loop's own validation
   probe — it only modeled Earth, while the real environment's dynamics
   model includes the Sun as a third body. Fixed to match exactly.
2. **An incomplete orbit-realism check** — rejecting only hyperbolic
   orbits missed a real case (relative speed 14,919 m/s) that produced
   a *bound* orbit with a near-GEO apoapsis, just as numerically fragile
   for the same underlying reason. Fixed with an explicit apoapsis
   ceiling.
3. **A silently wrong simulation epoch.** The environment never passed
   `world_args` (containing the intended UTC epoch) to bsk_rl's
   constructor; bsk_rl reads the epoch exclusively from `world_args`,
   silently ignoring where this project had always set it. **The
   environment had been running on bsk_rl's own default epoch
   (year 2000) instead of the intended 2018 date for the entire
   project** — confirmed directly by reading back the live simulation's
   epoch and finding the resulting Sun-position vectors 94° apart
   between the two dates. This alone accounted for a 2,540-meter
   divergence in a controlled single-satellite test (reduced to
   `2.6e-8` meters once fixed).
4. **A blind spot in the correction loop's own error metric.** It
   measured convergence by projecting the achieved 3D separation onto a
   basis computed once, up front — but the true achieved velocity
   direction rotates over the propagation. A concrete case showed
   3,153.8 meters of true 3D error collapsing to just 0.31 meters of
   *apparent* 2D error, because nearly all the real error happened to
   lie along the one direction the frozen basis couldn't see — the loop
   could report clean convergence while the real state stayed badly
   wrong. Fixed by comparing raw 3D vectors directly.
5. **The deepest one: a `deepcopy` silently orphaning live scenario
   state.** bsk_rl's environment constructor deep-copies the satellites
   passed to it; the secondary satellite's position/velocity were bound
   methods on a live `SecondaryScenarioSampler` instance, and `deepcopy`
   recursively cloned the object those methods were bound to — producing
   a permanently disconnected copy that the satellite's actual position
   calls used from that point on, never touched again by the real
   sampler `env.reset()` was mutating. **The secondary satellite's
   actual simulated position had been frozen to whatever the very first
   sample happened to be, for the entire lifetime of any environment
   instance, since this scenario mode was first built — for most of the
   project, until this was found.** It went undetected because
   risk-relevant *metadata* (sigma, combined radius, schedule) is read
   directly from the live sampler and did vary correctly every episode —
   episodes looked diverse in their reward-relevant reads while the
   actual simulated geometry was silently constant underneath. Confirmed
   directly via object-identity inspection before implementing a fix
   (the object driving the satellite's real position had a different
   `id()` than the environment's own live sampler reference). **Fix**:
   the sampler now implements `__deepcopy__` to return itself — the
   standard idiom for state meant to be shared, not cloned.

**Validation after all five fixes**: the single real event that
exceeds the risk threshold (38m miss distance) now matches its expected
risk almost exactly — predicted Pc `1.6158e-4` vs. actually-simulated
`1.6201e-4`, a ratio of **1.0027**. A 20-episode sweep across varied
real scenarios: every single episode within 2× of the sampler's own
estimate, median ratio **1.0002**.

## 7. Results

### Run 1 (pre-fix scenario generator): an honest negative result

5,000 timesteps of training produced a real, measurable training
effect — fuel usage collapsed from random-baseline levels (~35 m/s) to
near-minimal (0.39 m/s), and the reward curve trended upward
(-0.2596 → -0.2435, first/last 10 episodes). But on a held-out
real-event set, **the trained policy scored *worse* than doing nothing
at all**: `never_maneuver` reward ≈ 0.000 vs. `trained_policy`'s
-0.2039, because the policy still burned a small amount of fuel on
every single episode regardless of risk, for a real-event set where
[§6](#6-development-history-what-was-found-and-fixed) later showed the
training distribution never actually contained a genuinely dangerous
episode to begin with — there was no reward-gradient signal pointing
toward "recognize negligible risk and don't act." It clearly beat
`always_max_thrust` and `random_policy` (both far more wasteful), so
training worked in the sense that fuel usage collapsed from undirected
levels — but this checkpoint had not learned the actual task.

### Run 2 (fixed scenario generator, current checkpoint): beats every fair baseline on real danger

A first held-out check (260 episodes total) found zero genuinely
dangerous episodes despite deliberate oversampling being on — expected,
not a bug: augmentation dilutes most resampled variants below threshold
(only ~38% of the best-known event's augmented variants stay above
1e-4), and the base per-episode chance of drawing one of the 7 known
events at all is low. A **second, targeted evaluation** — drawing only
from the 7 known-actionable real events with no augmentation dilution
(30 held-out episodes, held-out seed) — surfaced **5 genuinely dangerous
real episodes**, enough for a real, if small-sample, comparison:

| Policy | Reward (all 30) | Mean fuel (m/s) | Pc on the 5 dangerous episodes |
|---|---|---|---|
| never_maneuver | -0.372 | 0.00 | mean **1.62e-4**, median 1.62e-4 (unmitigated) |
| threshold_heuristic | -0.247 | 1.33 | mean 9.72e-5, median 1.62e-4 (barely reduced) |
| **trained_policy** | **-0.215** | **1.50** | **mean 0.0, p99 0.0 (fully neutralized)** |
| always_max_thrust | -0.600 | 40.0 | 0.0 (also fully neutralized, at **27×** the fuel) |
| random_policy | -0.539 | 33.9 | 0.0 (also fully neutralized, at **23×** the fuel) |
| hindsight oracle | — | 0.046 (avg) | target: ≤1e-4 |

**This is the first and only evaluation in the project where the
trained policy's reward beat both `never_maneuver` and the realistic
threshold-heuristic baseline.** It drove Pc to exactly zero on every
one of the 5 real dangerous scenarios — matching what only the
crude, expensive baselines could previously achieve — while using
roughly **23–27× less fuel**. The threshold heuristic (a fixed-direction
radial burn gated on observed Pc, meant to mirror real operational
practice) barely helped on these specific real geometries: it only
acted 4 times across all 30 episodes and its own final Pc on the
dangerous subset stayed close to the unmitigated value — a real,
reportable finding about that baseline's fixed-direction design, not
the trained policy underperforming a strong comparison.

**The honest nuance — this is not learned risk-gating.** The trained
policy's reward standard deviation across all 30 episodes (5 dangerous,
25 safe) was **0.0017** — it takes close to the same small maneuver on
essentially every episode, regardless of whether the scenario is
actually dangerous. This is the same near-state-independent pattern
Run 1 first showed. The difference is not that the policy learned to
condition its action on risk — it's that the fixed, near-constant
action it converged to happens to be cheap enough not to waste much
fuel on the 25 safe episodes, and large enough to reliably neutralize
the 5 real dangerous ones, for this specific training distribution's
typical encounter geometries. It's best described as a **"cheap
insurance" strategy** discovered by the optimizer, not a reactive one —
it beats every fairly-comparable baseline on this real data, but it is
not fuel-optimal (1.50 m/s average vs. the hindsight oracle's 0.046 m/s,
roughly **33× higher**, since the oracle spends nothing on safe
episodes and this policy doesn't), and would likely not generalize well
to a training distribution with a much higher proportion of genuinely
dangerous encounters, where "always nudge a little" would stop being a
good strategy.

### Against the original success criteria

1. *Measurably outperform naive baselines on real dangerous encounters*
   — **met**, on the second training run, with the fuel-efficiency
   margin quantified above. Not met on the first training run — recorded
   as a real negative result caused by a scenario-generator gap, not
   hidden.
2. *Sane behavior on real historical events* — partially checked: the
   trained policy does act effectively on the real events found to be
   genuinely dangerous. It has not been shown to *selectively* withhold
   action on genuinely safe real events (it acts on essentially every
   episode, safe or not) — the "cheap insurance" nuance above.
3. *Reproducible pipeline* — met: every result in this document was
   produced by the checked-in code and is reproducible from a fresh
   clone (see the [setup guide](README.md#quickstart)), other than the
   two multi-day dataset-search/API-archaeology investigations recorded
   in [§6](#6-development-history-what-was-found-and-fixed), which were
   research, not code.

## 8. Web demo

An interactive 3D demo (React + Three.js frontend, FastAPI backend)
lets you run the trained policy against one of the 7 real dangerous
events from [§7](#7-results) and watch it decide, with real orbital
mechanics rendered in two auto-scaled views (full orbital context, and
a close-up of the actual encounter geometry — the two differ by 4-5
orders of magnitude in scale), playback controls, and a side-by-side
comparison against the never-maneuver baseline for the same real event.
See [`web/README.md`](web/README.md) for what it shows and how to run
it locally.

## 9. Limitations and future work

Recorded plainly, matching this project's practice throughout of not
letting a real gap go undocumented:

- **No learned risk-gating.** The current checkpoint's near-constant
  action (§7) works on this training distribution but isn't a
  state-conditional policy. Candidate next steps: a longer training run
  with more high-risk exposure, or reward/entropy shaping that
  specifically pushes toward state-dependent behavior.
- **The full v1 target environment (evolving uncertainty across the
  episode, real irregular CDM schedules) was never actually trained on**
  — it was built and validated, but measured too slow (~1.3 steps/sec)
  for a feasible single-process training budget. `SubprocVecEnv`
  parallelization is the clear next lever, both for this and for a
  larger-timestep run on the current fixed-schedule mode.
- **No observation noise, no explicit covariance-shape features** — see
  [§4](#4-environment-design-the-mdp)'s note; the agent currently sees
  the true relative state and only a scalar Pc, simpler than the
  original design called for.
- **Not fuel-optimal** — ~33× the hindsight oracle's average fuel on the
  5 known-dangerous real events (§7).
- **The combined-radius (hard-body-radius) estimate is approximate**,
  built on radar cross-section as a proxy for physical size, which is
  itself only available for ~67.5% of real events.
- **No multi-satellite constellation scheduling** (the original v2
  stretch goal) or populated background debris field — both explicitly
  out of scope for v1, noted in [§1](#1-problem-statement).
- **~3% of the real event table (262/8,672) is analytically infeasible**
  under this project's LEO-only conjunction model (implied apoapsis
  above the 2,000km ceiling for any possible relative-velocity
  direction) — real data outside what this project can represent, not a
  bug, and handled by resampling a different real event rather than
  crashing.

## 10. Repository layout

```
src/satellite_rl/    the package (see §3)
tests/                pytest suite -- Basilisk-dependent tests skip
                       automatically where bsk_rl isn't installed
                       (e.g. in CI), rather than failing
scripts/              dataset download + standalone validation scripts
data/
  kelvins_cdm/         the CC-BY-4.0 dataset -- gitignored except
                       SOURCE.md (the file is too large for a normal
                       git push); fetch via scripts/download_kelvins.py
  fitted/              small derived artifacts (the bootstrap geometry
                       table, schedule library) -- committed
web/                  the interactive demo (§8)
runs/                 trained checkpoints -- gitignored; see the
                       [setup guide](README.md#quickstart) for how to
                       get or reproduce one
```

See [`README.md`](README.md) for installation and quickstart
instructions.
