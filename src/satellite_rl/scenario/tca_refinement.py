"""Refine the targeting solver's nominal TCA to the TRUE local-minimum-
separation time as it actually occurs under Basilisk's real dynamics,
rather than the precomputed instant our J2-based targeting solver
assumes. See docs/18-scenario-generator-hardening.md part 2 for the full
diagnosis: residual dynamics beyond J2 shift *when* true closest approach
occurs by a small amount, which gets amplified into a large positional
error at high relative speed (timing_error_s * relative_speed_ms).

Why a short high-fidelity probe near the nominal TCA alone doesn't work:
the Basilisk-vs-J2 divergence accumulates gradually over the ENTIRE
trajectory (days), not just near closest approach -- so the only way to
find the true minimum is to propagate both objects under full Basilisk
dynamics from t0 all the way to the vicinity of TCA. Benchmarked at
~1.5-3s wall-clock for a realistic 3-day, two-satellite scenario at
5-10s sampling resolution (Phase 5b) -- a one-time cost per scenario
(paid at environment/scenario construction), not per RL training step.

API/setup verified against Phase 4's working `bsk_rl`-based environment
and Phase 3's raw-Basilisk debugging (gravity/SPICE/GravBodyVector setup
-- see docs/17-env-implementation-notes.md); this module uses raw
Basilisk scripting (not bsk_rl's Gym wrapper) specifically to get direct
access to a full fine-grained state recorder, which isn't straightforward
to extract through bsk_rl's step()-based interface.
"""

import numpy as np
from Basilisk.simulation import spacecraft
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody
from Basilisk.utilities.supportDataTools.dataFetcher import DataFile, get_path

UTC_INIT = "2018 SEP 29 21:00:00.000 (UTC)"


def _fly_passive_pair(
    r1_0: np.ndarray,
    v1_0: np.ndarray,
    r2_0: np.ndarray,
    v2_0: np.ndarray,
    duration_s: float,
    sim_rate_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate two passive spacecraft under full Basilisk dynamics
    (10th-degree spherical harmonics + SPICE Earth orientation, matching
    bsk_rl's own DynamicsModel), recording their full trajectories.
    """
    sim_task_name = "simTask"
    sim_process_name = "simProcess"

    scSim = SimulationBaseClass.SimBaseClass()
    dyn_process = scSim.CreateNewProcess(sim_process_name)
    dyn_process.addTask(scSim.CreateNewTask(sim_task_name, macros.sec2nano(sim_rate_s)))

    sc1 = spacecraft.Spacecraft()
    sc1.ModelTag = "probe1"
    sc2 = spacecraft.Spacecraft()
    sc2.ModelTag = "probe2"
    sc1.syncDynamicsIntegration(sc2)
    scSim.AddModelToTask(sim_task_name, sc1)
    scSim.AddModelToTask(sim_task_name, sc2)

    grav_factory = simIncludeGravBody.gravBodyFactory()
    planet = grav_factory.createEarth()
    planet.isCentralBody = True
    path_grav_data = str(get_path(DataFile.LocalGravData.GGM03S))
    planet.useSphericalHarmonicsGravityModel(path_grav_data, 10)

    grav_factory.createSpiceInterface(time=UTC_INIT)
    grav_factory.spiceObject.zeroBase = "Earth"
    scSim.AddModelToTask(sim_task_name, grav_factory.spiceObject, ModelPriority=100)

    sc1.gravField.gravBodies = spacecraft.GravBodyVector(list(grav_factory.gravBodies.values()))
    sc2.gravField.gravBodies = spacecraft.GravBodyVector(list(grav_factory.gravBodies.values()))

    sc1.hub.r_CN_NInit = list(r1_0)
    sc1.hub.v_CN_NInit = list(v1_0)
    sc2.hub.r_CN_NInit = list(r2_0)
    sc2.hub.v_CN_NInit = list(v2_0)

    rec1 = sc1.scStateOutMsg.recorder()
    rec2 = sc2.scStateOutMsg.recorder()
    scSim.AddModelToTask(sim_task_name, rec1)
    scSim.AddModelToTask(sim_task_name, rec2)

    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(macros.sec2nano(duration_s))
    scSim.ExecuteSimulation()

    times_s = np.array(rec1.times()) * macros.NANO2SEC
    r1 = np.array(rec1.r_BN_N)
    r2 = np.array(rec2.r_BN_N)
    grav_factory.unloadSpiceKernels()
    return times_s, r1, r2


def refine_tca(
    ego_r0: np.ndarray,
    ego_v0: np.ndarray,
    sec_r0: np.ndarray,
    sec_v0: np.ndarray,
    nominal_tca_s: float,
    sim_rate_s: float = 5.0,
    margin_s: float | None = None,
) -> dict:
    """Find the true local-minimum-separation time near `nominal_tca_s`
    by propagating both objects under full Basilisk dynamics from t0.

    Args:
        ego_r0, ego_v0: ego's state at t0.
        sec_r0, sec_v0: secondary's state at t0 (from the targeting
            solver, e.g. `TargetedScenario.r_sec_t0`/`v_sec_t0`).
        nominal_tca_s: the targeting solver's assumed time-to-TCA.
        sim_rate_s: recorder/integration resolution -- 5-10s gives ample
            precision for typical residual timing offsets (tens of
            seconds, per docs/18) via the quadratic refinement below,
            while keeping wall-clock cost low.
        margin_s: how far past nominal_tca_s to keep propagating, to
            ensure the true minimum (which could be slightly before or
            after the nominal instant) is bracketed. Defaults to
            max(300s, 1% of nominal_tca_s).

    Returns:
        dict with `refined_tca_s`, `min_separation_m`,
        `nominal_separation_m` (separation at the closest recorded sample
        to the original nominal instant, for comparison), and
        `sample_resolution_s`.
    """
    if margin_s is None:
        margin_s = max(300.0, 0.01 * nominal_tca_s)
    duration_s = nominal_tca_s + margin_s

    times_s, r1, r2 = _fly_passive_pair(ego_r0, ego_v0, sec_r0, sec_v0, duration_s, sim_rate_s)
    separation = np.linalg.norm(r1 - r2, axis=1)

    i_min = int(np.argmin(separation))
    # Quadratic (parabolic) interpolation through the 3 points bracketing
    # the discrete minimum -- standard technique for sub-sample-resolution
    # precision without a second, finer simulation pass.
    if 0 < i_min < len(times_s) - 1:
        t0, t1 = times_s[i_min - 1], times_s[i_min]
        d0, d1, d2 = separation[i_min - 1], separation[i_min], separation[i_min + 1]
        denom = d0 - 2 * d1 + d2
        refined_t = t1 + 0.5 * (d0 - d2) / denom * (t1 - t0) if abs(denom) > 1e-9 else t1
    else:
        refined_t = float(times_s[i_min])

    nominal_idx = int(np.argmin(np.abs(times_s - nominal_tca_s)))
    return {
        "refined_tca_s": float(refined_t),
        "min_separation_m": float(separation[i_min]),
        "nominal_separation_m": float(separation[nominal_idx]),
        "sample_resolution_s": sim_rate_s,
    }
