#!/usr/bin/env python3
"""Phase 3 validation (STATUS: INCOMPLETE / NOT TRUSTWORTHY YET -- see
docs/16-targeting-validation-results.md): how much does Basilisk's actual
high-fidelity dynamics (10th-degree spherical harmonics Earth gravity,
matching bsk_rl's own DynamicsModel -- see docs/02-bsk_rl-architecture.md)
diverge from the simpler two-body model our targeting solver uses to
compute initial conditions? This is the open risk flagged in
docs/03-scenario-design.md's "Open technical risk to flag honestly"
section.

This script is kept as-is, NOT as a validated result, because three real
bugs were found and fixed while building it (numpy-array initial
conditions silently mis-parsing, missing SPICE planet-orientation
message, missing explicit GravBodyVector assignment -- each documented
inline below) but the final single-spacecraft result still didn't
converge to a physically plausible number even after all three fixes, and
further debugging showed sensitivity to details (e.g. `zeroBase` string
capitalization) that weren't fully run to ground. Rather than keep
hand-rolling raw Basilisk scripting, Phase 4 will do this validation
properly using bsk_rl's own Satellite/DynamicsModel class, which already
wires this up correctly in a maintained, working RL framework -- see
docs/16 for the full reasoning. Do not treat this script's output as a
real result; it's kept for reference (the real, fixed bugs below remain
useful) for whoever picks this back up in Phase 4.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd
from Basilisk.simulation import spacecraft
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody

from satellite_rl.scenario.distributions import FITTED_DIR, sample_scenario_geometry_bootstrap
from satellite_rl.scenario.targeting import example_leo_orbit, solve_secondary_initial_state


def fly_two_body_basilisk(
    r_ego0: np.ndarray,
    v_ego0: np.ndarray,
    r_sec0: np.ndarray,
    v_sec0: np.ndarray,
    duration_s: float,
    timestep_s: float = 30.0,
) -> dict:
    """Propagate two spacecraft forward under Basilisk's real 10th-degree
    spherical harmonics Earth gravity, returning their final states.
    """
    sim_task_name = "simTask"
    sim_process_name = "simProcess"

    scSim = SimulationBaseClass.SimBaseClass()
    dyn_process = scSim.CreateNewProcess(sim_process_name)
    sim_time_step = macros.sec2nano(timestep_s)
    dyn_process.addTask(scSim.CreateNewTask(sim_task_name, sim_time_step))

    ego = spacecraft.Spacecraft()
    ego.ModelTag = "ego"
    sec = spacecraft.Spacecraft()
    sec.ModelTag = "secondary"
    ego.syncDynamicsIntegration(sec)

    scSim.AddModelToTask(sim_task_name, ego)
    scSim.AddModelToTask(sim_task_name, sec)

    grav_factory = simIncludeGravBody.gravBodyFactory()
    planet = grav_factory.createEarth()
    planet.isCentralBody = True
    from Basilisk.utilities.supportDataTools.dataFetcher import DataFile, get_path

    path_grav_data = str(get_path(DataFile.LocalGravData.GGM03S))
    planet.useSphericalHarmonicsGravityModel(path_grav_data, 10)

    # REQUIRED for any spherical-harmonics field of degree/order >= 1: the
    # tesseral/sectoral terms are evaluated in the Earth-fixed rotating
    # frame, so a planet-orientation message must be connected via a SPICE
    # interface, or Earth is silently treated as non-rotating -- producing
    # a large, spurious secular drift (empirically ~3000 km over 3 days in
    # this scenario before this fix; see docs/16-targeting-validation-
    # results.md). Verified against bsk_rl's own world.py AND Basilisk's
    # own examples/scenarioOrbitConsistencyVerification.py (a real,
    # maintained Basilisk test written specifically to check this exact
    # thing -- see docs/16), which is also where `zeroBase = "Earth"`
    # (capitalized) and the explicit GravBodyVector line below come from.
    grav_factory.createSpiceInterface(time="2018 SEP 29 21:00:00.000 (UTC)")
    grav_factory.spiceObject.zeroBase = "Earth"
    scSim.AddModelToTask(sim_task_name, grav_factory.spiceObject, ModelPriority=100)

    # Explicit assignment (matching scenarioOrbitConsistencyVerification.py)
    # rather than gravFactory.addBodiesTo(...) -- tried both individually
    # and together; none produced a result matching the two-body reference
    # at short (600s) timescales. Left as the closest-to-reference-example
    # version for whoever debugs this further in Phase 4.
    ego.gravField.gravBodies = spacecraft.GravBodyVector(list(grav_factory.gravBodies.values()))
    sec.gravField.gravBodies = spacecraft.GravBodyVector(list(grav_factory.gravBodies.values()))

    # MUST be plain Python lists, not numpy arrays: assigning a numpy
    # ndarray directly to hub.r_CN_NInit/v_CN_NInit silently mis-parses
    # through Basilisk's SWIG/Eigen binding (produces a wrong initial
    # state with no error/warning) -- empirically confirmed (Phase 3, see
    # docs/16-targeting-validation-results.md): 473m error over 3 days
    # with numpy arrays vs. 0.02m with plain lists, all else identical.
    ego.hub.r_CN_NInit = list(r_ego0)
    ego.hub.v_CN_NInit = list(v_ego0)
    sec.hub.r_CN_NInit = list(r_sec0)
    sec.hub.v_CN_NInit = list(v_sec0)

    ego_rec = ego.scStateOutMsg.recorder()
    sec_rec = sec.scStateOutMsg.recorder()
    scSim.AddModelToTask(sim_task_name, ego_rec)
    scSim.AddModelToTask(sim_task_name, sec_rec)

    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(macros.sec2nano(duration_s))
    scSim.ExecuteSimulation()

    result = {
        "r_ego_final": np.array(ego_rec.r_BN_N[-1]),
        "v_ego_final": np.array(ego_rec.v_BN_N[-1]),
        "r_sec_final": np.array(sec_rec.r_BN_N[-1]),
        "v_sec_final": np.array(sec_rec.v_BN_N[-1]),
    }
    grav_factory.unloadSpiceKernels()
    return result


def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    time_to_tca_s = 3 * 86400.0  # 3 days -- representative of a mid-schedule CDM lead time

    geometry_path = FITTED_DIR / "geometry_events.csv"
    geometry_df = pd.read_csv(geometry_path)
    print(f"Loaded {len(geometry_df)} real events from {geometry_path.name}")

    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(42)

    results = []
    for i in range(n_trials):
        real_event = sample_scenario_geometry_bootstrap(geometry_df, rng)
        orientation = rng.uniform(0, 2 * np.pi)

        scenario = solve_secondary_initial_state(
            ego_r0,
            ego_v0,
            time_to_tca_s,
            real_event["miss_distance"],
            real_event["relative_speed"],
            orientation,
            rng,
        )

        flown = fly_two_body_basilisk(
            ego_r0, ego_v0, scenario.r_sec_t0, scenario.v_sec_t0, time_to_tca_s
        )

        realized_miss = np.linalg.norm(flown["r_sec_final"] - flown["r_ego_final"])
        realized_speed = np.linalg.norm(flown["v_sec_final"] - flown["v_ego_final"])

        ego_drift = np.linalg.norm(flown["r_ego_final"] - scenario.r_ego_tca)
        sec_drift = np.linalg.norm(flown["r_sec_final"] - scenario.r_sec_tca_target)

        miss_error = realized_miss - scenario.miss_distance_target
        speed_error = realized_speed - scenario.relative_speed_target

        print(
            f"trial {i}: target_miss={scenario.miss_distance_target:8.2f} m  "
            f"basilisk_miss={realized_miss:8.2f} m  miss_error={miss_error:+9.2f} m  "
            f"target_speed={scenario.relative_speed_target:8.2f}  "
            f"basilisk_speed={realized_speed:8.2f}  speed_error={speed_error:+7.3f} m/s  "
            f"ego_2body_drift={ego_drift:9.2f} m  sec_2body_drift={sec_drift:9.2f} m"
        )
        results.append(
            {
                "miss_error_m": miss_error,
                "speed_error_ms": speed_error,
                "ego_drift_m": ego_drift,
                "sec_drift_m": sec_drift,
                "target_miss_m": scenario.miss_distance_target,
            }
        )

    miss_errors = np.array([r["miss_error_m"] for r in results])
    speed_errors = np.array([r["speed_error_ms"] for r in results])
    ego_drifts = np.array([r["ego_drift_m"] for r in results])
    sec_drifts = np.array([r["sec_drift_m"] for r in results])

    print(f"\n=== Summary (n={n_trials}) ===")
    print(f"miss_distance error (Basilisk - target):  mean={miss_errors.mean():.2f} m, "
          f"std={miss_errors.std():.2f} m, max_abs={np.abs(miss_errors).max():.2f} m")
    print(f"relative_speed error (Basilisk - target): mean={speed_errors.mean():.4f} m/s, "
          f"std={speed_errors.std():.4f} m/s, max_abs={np.abs(speed_errors).max():.4f} m/s")
    print(f"ego single-body two-body-vs-Basilisk drift:       mean={ego_drifts.mean():.2f} m, "
          f"max={ego_drifts.max():.2f} m")
    print(f"secondary single-body two-body-vs-Basilisk drift: mean={sec_drifts.mean():.2f} m, "
          f"max={sec_drifts.max():.2f} m")


if __name__ == "__main__":
    main()
