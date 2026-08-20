# 11 — Baselines & Evaluation Methodology

## Baselines

**1. Never maneuver** — zero Δv, always. Establishes the raw risk exposure
of doing nothing; the learned policy must clearly beat this on the risk
term, trivially (it's a strict floor, included for completeness/sanity
rather than as a serious comparison).

**2. Always maneuver maximally on any detected conjunction** — burn at
`max_dv` the moment a conjunction is scheduled. Establishes the fuel-cost
ceiling and shows what "risk-averse to a fault" costs — a useful contrast
point on the fuel/disruption axis, not something we expect (or want) the
learned policy to resemble either.

**3. Threshold heuristic (the real baseline to beat)** — maneuver
(a fixed, non-learned Δv magnitude/direction, e.g. computed via a simple
closed-form miss-distance-increase targeting) once observed Pc exceeds
`Pc_threshold` (per `08-reward-function.md`), otherwise wait. This mirrors
actual operational practice and is the baseline the learned policy needs
to meaningfully outperform — on fuel efficiency and/or disruption count at
matched or better risk levels — for the project's central claim ("RL adds
value here") to hold up. If the learned policy doesn't beat this, that's a
real, reportable finding, not a failure to hide — see "Honest negative
results" below.

**4. Hindsight-optimal oracle (context only, not a fair comparison)** —
given the *true* final encounter geometry (unavailable to any realistic
policy at decision time), compute the minimum-fuel maneuver that would
have kept Pc below threshold. This is an upper bound / sanity ceiling: no
real policy can beat it, but it tells us how much headroom exists between
the threshold heuristic and theoretical best, i.e., whether there's
meaningful value on the table for a learned wait-and-refine strategy to
capture at all before we spend effort trying to capture it.

## Metrics

- **Final Pc distribution** — mean, median, and tail percentiles (e.g.
  95th/99th) across evaluation episodes; tail behavior matters more than
  the mean for a risk problem (a policy with a good mean but a fat tail of
  rare high-Pc outcomes is worse than one with a slightly higher mean and
  a tighter tail — report both, don't collapse to a single number).
- **Total fuel used per episode** (Δv magnitude, true un-deadzoned value
  per `07-action-space.md`).
- **Maneuver count per episode** (deadzoned, per `07`), as the
  disruption-term proxy.
- **Regret vs. hindsight-optimal oracle** — fuel used minus the oracle's
  fuel, at matched final-Pc outcome; the metric that most directly answers
  "how close to optimal is this."
- **Timing behavior** — at which decision point (early/mid/late in the
  schedule) does the policy typically act? This is qualitative but
  important: does it learn to wait for better information when the
  scenario's covariance is still shrinking fast, and act promptly once
  waiting stops paying off? Worth a plot, not just a scalar.

## Real-data validation (v1 success criterion #2, `01-problem-scope.md`)

Replay the held-out partition of real Kelvins events (per
`09-episode-design.md`'s train/eval split) through the trained policy in
"evaluation mode" (policy observes the real event's actual reported
CDM sequence — real miss distance, real covariance evolution, real
schedule — and its actions are scored, but obviously can't be validated
against a real counterfactual "what if it had maneuvered differently"
outcome, since the real events already had their own real-world
resolution). What this check *can* validate:
- The policy doesn't act (waste fuel) on real historical events with
  consistently low reported risk.
- The policy does act on real historical events that were flagged as
  higher-risk (large `max_risk_estimate` in the Kelvins data) — sanity
  aligning with what real analysts flagged as actionable.
- The policy's behavior degrades gracefully (doesn't produce wildly
  out-of-range Δv or NaN/crash) on real events whose statistics may fall
  outside the training distribution's typical range — an out-of-
  distribution robustness check, not a performance benchmark.

This is explicitly a **sanity check**, not a rigorous benchmark — real
events don't have a "correct answer" we can score against (we don't know
what would have happened under a different maneuver decision than the one
real operators actually made, and even real operators' decisions aren't
necessarily optimal). Framing this correctly in any write-up matters for
intellectual honesty.

## Statistical rigor given observation noise

Per `06-state-space.md`'s noted caveat: because observations include
injected noise, per-episode outcomes have real variance even for a fixed
underlying scenario. Evaluation runs **N repeated episodes per scenario
type** (not just N different scenarios) and reports distributions, not
single-run point estimates — with N chosen large enough that baseline-vs-
learned-policy differences are distinguishable from observation-noise
variance (a basic power/significance check before claiming "the learned
policy is better," not just eyeballing mean differences).

## Honest negative results

If the learned PPO policy fails to beat the threshold heuristic (baseline
3), that itself is a legitimate, reportable outcome for an open-source
research project — and given that real conjunction-assessment practice
already uses well-tuned threshold-based rules refined over decades of
operational experience, it is a genuinely plausible outcome, not a
strawman we're setting up to beat. The project's value doesn't depend on
RL "winning" — it depends on the pipeline (grounded scenario generation,
correct Pc computation, a well-posed MDP, and a fair comparison) being
correct and reproducible enough that the result — whichever way it comes
out — is trustworthy.
