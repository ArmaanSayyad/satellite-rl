# 08 — Reward Function Design

Builds on `02-bsk_rl-architecture.md` §5 (`GlobalReward`/`ComposedReward`,
`ResourceReward` as a structural template) and `04-collision-probability.md`
(Pc as the risk signal).

## The core problem: realized collisions are too rare to be a training signal

If collision-avoidance is being modeled realistically, actual simulated
collisions should be extremely rare events (that's the point of avoidance —
a well-behaved policy essentially never lets a true collision happen). A
reward based only on "did a collision actually occur this episode" is
almost always zero, giving the policy gradient almost nothing to learn
from — the classic sparse-reward problem, and a bad fit here specifically
because the *desired* behavior makes the informative event vanishingly
rare.

**Decision**: use the **computed collision probability (Pc) itself** as the
risk signal, not a binary realized-collision outcome. This is not a hack —
it's literally how real conjunction-avoidance decisions are already scored
operationally: an analyst evaluates and acts based on a *probability*
threshold, not on waiting to see if a collision happens. Using Pc as the
reward's risk term is therefore both more tractable *and* more faithful to
the real decision problem than a sparse binary signal would be.

## Reward structure

Composed of three terms (implemented as separable pieces via
`bsk_rl.data.ComposedReward`, per `02` §5, so each is independently
testable):

**1. Risk term (terminal, dominant)** — computed once per episode, at the
final decision point before TCA, using the **true** relative state (not
the noisy observed one — the risk term scores the actual outcome, while
the *observation* the agent acted on was necessarily noisy; this asymmetry
is intentional and mirrors reality: the world doesn't grade you on your
estimate, it grades you on what actually happens):

```
risk_penalty = -w_risk * f(Pc_final)
```

where `f` is **not** simply linear in Pc. Real operational practice treats
risk nonlinearly around a threshold (commonly-cited operational Pc
thresholds sit in the 1e-4–1e-5 range for crewed/high-value assets — this
is a widely-cited order of magnitude in the conjunction-assessment
literature, not a number independently re-verified from a primary source
in this research pass, and should be confirmed before being treated as
load-bearing). v1 uses a smooth thresholded penalty:

```
f(Pc) = Pc / Pc_threshold                      if Pc <= Pc_threshold
f(Pc) = 1 + log(Pc / Pc_threshold)              if Pc > Pc_threshold
```

— i.e., roughly linear (small) penalty below the operational threshold,
switching to a much steeper (log-then-effectively-large, since we'll clip
at a large finite max rather than let it diverge) penalty above it. This
shapes the reward so the agent isn't indifferent between Pc=1e-6 and
Pc=1e-5 in absolute terms, but strongly avoids crossing into the regime
real operators would consider actionable risk. `Pc_threshold` is a config
value, not hardcoded, specifically so we can sensitivity-test the learned
policy against different threshold assumptions later.

**2. Fuel-cost term (per step)**:

```
fuel_penalty = -w_fuel * |Δv_t|
```

Applied every step the agent takes a nonzero action (per
`07-action-space.md`'s deadzone note — the *true* Δv is charged here, not
the deadzoned reporting value). This is what makes "always maneuver
maximally" a bad policy and creates the actual risk/fuel tradeoff that is
the point of the whole project.

**3. Maneuver-count term (per step, small)**:

```
disruption_penalty = -w_disruption * 1[Δv_t > deadzone]
```

A small fixed cost per maneuver performed (independent of its magnitude),
representing the real operational overhead of planning/executing a
maneuver (re-planning, coordination, schedule disruption) beyond just
propellant. Without this term, a policy has no reason to prefer one
well-timed burn over several small nudges that sum to the same total Δv —
in reality those aren't equivalent, and this term is what teaches the
policy to consolidate.

## Total reward per step

```
reward_t = fuel_penalty_t + disruption_penalty_t
           + (risk_penalty if t == final_step else 0)
```

## Weight selection (`w_risk`, `w_fuel`, `w_disruption`)

Not fixed a priori — v1's Phase 6 (`13-roadmap.md`) includes weight
sensitivity analysis as an explicit task, not a one-shot guess. Starting
point: set `w_risk` large enough that even the smallest realistic `w_fuel *
max_dv` fuel cost is dominated by crossing `Pc_threshold`, ensuring the
policy never learns to "accept" threshold-crossing risk purely to save
fuel — then tune `w_fuel`/`w_disruption` downward from there until the
learned policy's maneuver frequency roughly matches the threshold-heuristic
baseline's frequency (`11-evaluation.md`) as a calibration sanity check,
before comparing whether the *learned* policy actually does better than
that baseline.

## What's deliberately not in the reward

- **No shaping/potential-based reward for "getting closer to a good
  decision" mid-episode** — v1 keeps the fuel/disruption terms as the only
  per-step signal and lets the terminal risk term carry the actual
  objective, to avoid a shaping term accidentally biasing the learned
  policy's risk tolerance (a known failure mode of naive reward shaping).
  If training proves too slow/sparse in practice, revisit this — noted as
  an open risk, not a settled decision, in `13-roadmap.md`.
- **No mission-utility/task-completion bonus** — v1's satellite has no
  competing task (per `01-problem-scope.md`, we're isolating the
  risk/fuel tradeoff); this becomes relevant once the environment is
  extended to include the satellite's primary mission (e.g. imaging
  tasking) as a competing objective, a natural v1.1/v2 extension.
