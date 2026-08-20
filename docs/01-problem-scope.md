# 01 — Problem Scope

## v1 scope: single-satellite collision avoidance

We are building, first, an RL agent that controls **one satellite's** decision
of whether and when to execute a collision-avoidance maneuver in response to
a predicted close approach ("conjunction") with another tracked object
(debris or another satellite).

This is deliberately narrower than "constellation scheduling" for a concrete
reason: it is the smallest version of the problem that still contains the
core, genuinely-open research/engineering question — trading collision risk
against fuel cost under uncertainty — without also requiring multi-agent
coordination, which is a separate hard problem layered on top.

**In scope for v1:**
- One "ego" satellite in a realistic LEO orbit, simulated with real orbital
  dynamics (via `bsk_rl`/Basilisk).
- A stream of conjunction events against a background object population,
  with realistic miss-distance/covariance statistics grounded in the ESA
  Kelvins CDM dataset (see `05-datasets.md`).
- A collision-probability (Pc) computation the agent's observation is based
  on (see `04-collision-probability.md`).
- An action space letting the agent choose to wait or execute a maneuver
  (impulsive Δv) to reduce Pc.
- A reward trading off collision risk, fuel expenditure, and (a simplified
  proxy for) mission disruption from maneuvering.
- Training via a single-agent RL algorithm (PPO, see `10-rl-algorithm.md`).
- Evaluation against simple heuristic baselines and, where possible, sanity
  checks against real historical CDM outcomes.

**Explicitly deferred (not v1):**
- Multi-satellite constellation scheduling (assigning observation/tasking
  requests across satellites) — this is the stretch goal from the original
  project brainstorm and becomes a v2 extension once v1 is working, reusing
  the same Pc/reward machinery but adding a scheduling action space and
  (likely) multi-agent RL.
- High-fidelity attitude control / sensor pointing during avoidance
  maneuvers — we treat maneuvers as impulsive Δv events, not detailed
  attitude-control problems (Basilisk supports far more fidelity here than
  we need for v1; we intentionally under-use it at first).
- Real-time/live data feeds from Space-Track — v1 trains and evaluates on
  historical/synthetic data, not a live operational pipeline.
- Onboard autonomy constraints (limited compute, communication delay) —
  we assume the RL policy runs with full information at decision time.

## Why this ordering

The full pipeline (dynamics → conjunction generation → Pc computation →
RL loop → evaluation) has to work end-to-end and be trustworthy on the
single-agent problem before multi-agent coordination is added on top —
otherwise we can't tell whether a scheduling policy is failing because of
the scheduling logic or because the underlying risk/reward model is wrong.

## Success criteria for v1

1. The agent learns a policy that noticeably outperforms a naive
   "never maneuver" baseline (lower realized collision rate) and a naive
   "always maneuver on any conjunction" baseline (lower fuel expenditure /
   fewer unnecessary maneuvers), on held-out scenarios.
2. The policy's behavior is sane when checked against real historical CDM
   events from the ESA Kelvins dataset (e.g., it doesn't maneuver for
   objects with negligible Pc, and does act on high-Pc real events).
3. The whole pipeline — scenario generation, Pc computation, env, training,
   eval — is documented and reproducible from a fresh clone (this is an
   open-source project; reproducibility is a first-class requirement, not
   an afterthought).
