# 02 — bsk_rl / Basilisk Environment Architecture

Verified against `avslab.github.io/bsk_rl/` docs (v1.3.4) and the
`AVSLab/bsk_rl` `develop` branch source (`src/bsk_rl/...`) directly, Aug 2026.
Class/function names below are quoted as found, not paraphrased guesses.

## 1. Core architecture

Env classes (`bsk_rl.gym`):
- `GeneralSatelliteTasking(Env)` — base class, N satellites, tuple obs/action.
- `SatelliteTasking` — single-satellite convenience wrapper (unwraps the tuple).
  **This is what we use for v1.**
- `ConstellationTasking(GeneralSatelliteTasking, ParallelEnv)` — PettingZoo
  multi-agent wrapper. **This is what v2 (constellation scheduling) would use.**

Constructor composition objects:
- `satellites` — one or more `Satellite` instances
- `scenario` — a `bsk_rl.scene.Scenario`
- `rewarder` — a `bsk_rl.data.GlobalReward`
- `communicator` — a `bsk_rl.comm.CommunicationMethod`
- `world_type`/`world_args`, `sim_rate`, `time_limit`, `failure_penalty`

Each `Satellite` owns a `dyn_type` (dynamics model, `bsk_rl.sim.dyn`) and
`fsw_type` (flight software model, `bsk_rl.sim.fsw`). `Simulator`
(`bsk_rl.sim.simulator`) builds the actual Basilisk sim on each `reset()`.
Reset lifecycle hooks — `reset_overwrite_previous`, `reset_pre_sim_init`,
`reset_during_sim_init`, `reset_post_sim_init` — exist on `Scenario`,
`GlobalReward`, `Communicator`, and each `Satellite`/`DynamicsModel`/
`FSWModel`. **This is where our custom conjunction-scenario code will live**
(see §9 below and `03-scenario-design.md`).

## 2. Satellite / dynamics model

Dynamics classes (`sim/dyn/base.py`, `relative_motion.py`):
`DynamicsModelABC` → `DynamicsModel` (orbital-element properties:
`semi_major_axis`, `eccentricity`, `inclination`, `ascending_node`,
`argument_of_periapsis`, `true_anomaly`, `beta_angle`, plus `r_BN_N`,
`v_BN_N`, `sigma_BN`, Hill frame `HN`).

**Gravity fidelity is real, not toy**: `WorldModel.setup_gravity_bodies`
calls `useSphericalHarmonicsGravityModel(GGM03S_path, 10)` — a **10th-degree
spherical-harmonics Earth gravity model** (includes J2+), plus Sun via SPICE.
`BasicDynamicsModel` adds reaction wheels, momentum-desaturation RCS
thrusters, solar panel/battery/power. Specialized subclasses include
`ConjunctionDynModel`, `MaxRangeDynModel`, `ImagingDynModel`, etc.

**Actuators**: reaction wheels + desat thrusters (attitude only) by default.
**Δv maneuvers require `fsw_type` including `MagicOrbitalManeuverFSWModel`**,
which does an **impulsive** (instantaneous velocity-change) burn via
`action_impulsive_thrust(dv_N)` — directly edits `v_BN_N` in the Basilisk
state. It tracks `dv_available` (default `dv_available_init = 100.0` m/s) as
a simple fuel budget, with a `fuel_remaining` aliveness check. **No
continuous/finite-burn thrust model ships out of the box** — impulsive Δv is
the right fidelity level for v1 (matches how real conjunction-avoidance
maneuvers are typically modeled operationally, at the "delay this much
Δv along this vector" level, not detailed burn-arc modeling).

## 3. Action space

Two families (`bsk_rl.act`):
- `DiscreteAction` → `spaces.Discrete` (task list, e.g. `Charge`, `Drift`,
  `NadirPoint`, `Desat`, `Downlink`, `Image`, `Scan`, `Broadcast`).
- `ContinuousAction` → `spaces.Box` (only **one** continuous action allowed
  per satellite — `ContinuousActionBuilder` asserts `len(action_spec)==1`).

Verified continuous classes (`act/continuous_actions.py`):
- **`ImpulsiveThrust(name, max_dv=inf, max_drift_duration=inf, fsw_action=None)`**
  — action vector is exactly `[dV_N_x, dV_N_y, dV_N_z, duration]` (m/s, s).
  Calls `fsw.action_impulsive_thrust(dv_N)`, then coasts. **This maps almost
  exactly onto our "execute a Δv burn, or wait" decision** — "wait" is just
  a near-zero-dV action with a chosen drift duration.
- `ImpulsiveThrustHill(chief_name, ...)` — same, but Δv specified in a named
  chief satellite's Hill (RTN) frame, rotated to inertial internally. This is
  likely more natural for us: conjunction geometry is usually reasoned about
  relative to the secondary object, i.e. in a Hill/RTN-like frame.
- `AttitudeSetpoint(control_period=60)` — 3-vector MRP attitude command
  (not needed for v1, since we're treating maneuvers as impulsive Δv).

**Decision for v1**: use `ImpulsiveThrustHill` (or plain `ImpulsiveThrust`)
as the base continuous action, giving the agent direct control of burn
vector + coast duration. A literal "do nothing" is expressed as a zero (or
near-zero, to avoid singularities) Δv with a chosen drift duration. See
`07-action-space.md` for the full design, including whether we discretize
the "maneuver / wait" decision on top of this continuous primitive.

## 4. Observation space

`bsk_rl.obs`: `Observation` ABC (`get_obs()`). `SatProperties(dict(prop=...,
norm=..., name=...), ...)` pulls named properties off `fsw`/`dynamics`
(any of the orbital-element properties in §2, plus `storage_level_fraction`,
`battery_charge_fraction`, etc.). Critically, `SatProperties` supports an
arbitrary `fn` key — **a callable taking the satellite and returning a
value** — so an externally computed scalar (our collision probability, see
`04-collision-probability.md`) can be injected directly into the observation
via `SatProperties(dict(fn=my_pc_fn, name="collision_prob"))` with no need
to subclass anything.

`RelativeProperties` gives multi-satellite relative state (e.g. Hill-frame
position of another object) — directly useful for exposing the secondary
object's relative position/velocity in our observation. `Time`,
`OpportunityProperties`, `Eclipse` are also available. All observations
concatenate into a flat `Box` per satellite.

## 5. Reward / data model

Fully hookable: `GlobalReward` ABC — override
`calculate_reward(data: dict[sat_id, Data]) -> dict[sat_id, float]`, plus
optional `initial_data()`, `create_data_store()`, `reset_overwrite_previous()`.
Pairs with a `DataStore` subclass that accumulates `Data` from sim logs each
step. Shipped rewarders (`NoReward`, `UniqueImageReward`,
`ScanningTimeReward`, `ResourceReward`, `RSOInspectionReward`) don't cover
our case, but `ResourceReward` (arbitrary resource deltas → weighted reward)
is a good structural template for "-1 × fuel used, -large × collision
indicator." `ComposedReward` lets multiple reward terms be combined —
useful for keeping the risk/fuel/mission-utility terms as separate,
independently-testable pieces (see `08-reward-function.md`).

## 6. Multi-satellite support (relevant to v2 stretch goal only)

Native. `GeneralSatelliteTasking`/`ConstellationTasking` accept a list of
(possibly heterogeneous) `Satellite`s. `ConstellationTasking` implements
PettingZoo's `ParallelEnv`, with `meta_agent_groupings` to cluster multiple
satellites under one policy. Communication layer (`bsk_rl.comm`):
`NoCommunication`, `FreeCommunication`, `LOSCommunication`, others. Not
needed for v1, but confirms the v2 constellation-scheduling stretch goal is
architecturally supported without switching frameworks.

## 7. Shipped examples (what we get for free vs. must build)

Docs ship: Getting Started (single scanning satellite), Satellite
Configuration, Multi-Agent Environments, Fault Environment, Broadcast
Communication, Cloud Environment (+ re-imaging), Agile Earth-Observing
Satellite benchmark, RSO Inspection, plus training recipes (RLlib PPO,
time-discounted GAE, async multi-agent, curriculum learning, action-masking/
"shielded" training).

**No collision-avoidance environment ships.** Closest existing primitives
are `ConjunctionDynModel` / `MaxRangeDynModel` (§9) and `RelativeProperties`/
Hill-frame tooling — we build the CA scenario, reward, and observation
ourselves on top of these.

## 8. Installation

`pip install bsk-rl` (PyPI, latest `1.3.4`, MIT license). Requires Python
**≥3.10**. Basilisk itself installs separately first (`pip install bsk`, or
build from source per-platform); docs note macOS/Linux are preferable to
Windows. No GPU requirement anywhere in docs/PyPI metadata — this is a pure
CPU numerical sim (Basilisk core is C++).

## 9. Scripted events / injecting a conjunction — the key mechanism for us

Two relevant, verified mechanisms (`sim/dyn/relative_motion.py`):

- **`ConjunctionDynModel`** — purpose-built for exactly our use case. Each
  pair of satellites using this dyn model gets a Basilisk event registered
  via `self.simulator.createNewEvent(name, rate, conditionFunction=...,
  actionFunction=..., terminal=True)` that fires when inter-satellite
  distance ≤ sum of `conjunction_radius` (default 10 m); on trigger it logs
  a collision and fails the `conjunction_valid` aliveness check, ending the
  episode.
- **`MaxRangeDynModel`** — symmetric mechanism, keeps a deputy within
  `max_range_radius` of a named chief.

Both are thin wrappers around Basilisk's generic
`createNewEvent(conditionFunction, actionFunction, terminal, exactRateMatch)`
primitive. **This is the general scripted-event hook we reuse** to inject a
custom secondary object on a collision-course trajectory: add it as a second
dynamics object with a scripted/ballistic trajectory, then register our own
`createNewEvent` condition on separation distance/time to trigger our own
Pc-computation and reward logic (rather than only Basilisk's built-in
terminal-collision check, which is too coarse — we need continuous Pc
tracking as the conjunction approaches, not just a binary hit/miss at the
very end).

**Important gap**: there is **no higher-level scenario-scripting API** (no
YAML/JSON event timeline, no built-in TLE-based conjunction generator). We
have to assemble scripted conjunction scenarios ourselves from
`createNewEvent` + custom `reset_pre_sim_init` code + our own conjunction
geometry generator (grounded in the ESA CDM dataset — see
`03-scenario-design.md` and `05-datasets.md`). This is real, non-trivial
engineering work — flagging it clearly here so it's sized correctly in the
roadmap (`13-roadmap.md`), not treated as "just wire two libraries together."

## Implications for later design docs

- `03-scenario-design.md`: conjunction injection = custom secondary
  object + `createNewEvent`, geometry parameters drawn from real CDM
  statistics.
- `04-collision-probability.md`: Pc is computed **outside** Basilisk/bsk_rl
  (Python-side, from relative position + covariance), then fed into the
  observation via `SatProperties(fn=...)`.
- `07-action-space.md`: base primitive is `ImpulsiveThrustHill`
  (Δv in Hill/RTN frame + coast duration); "wait" is a near-zero-Δv action.
- `08-reward-function.md`: build on `ComposedReward` with separate risk /
  fuel / mission-utility terms, following the `ResourceReward` pattern.
