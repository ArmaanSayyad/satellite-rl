// Mirrors web/backend/simulation_runner.py's run_simulation() return
// shape exactly. Every field here is a real number from an actual
// Basilisk physics simulation of the trained policy -- nothing here is
// mocked or scripted.

export interface Vec3 {
  0: number;
  1: number;
  2: number;
}

export interface Frame {
  t_s: number;
  ego_r: [number, number, number];
  ego_v: [number, number, number];
  sec_r: [number, number, number];
  sec_v: [number, number, number];
}

export interface Keyframe extends Frame {
  fuel_ms: number;
  pc_estimate: number | null;
}

export interface Decision {
  t_s: number;
  action_dv_ms: [number, number, number];
  action_magnitude_ms: number;
  fuel_before_ms: number;
  fuel_after_ms: number;
}

export interface Scenario {
  seed: number;
  miss_distance_m: number;
  relative_speed_ms: number;
  native_pc: number;
  combined_radius_m: number;
}

export interface SimulationResult {
  seed: number;
  scenario: {
    miss_distance_m: number;
    relative_speed_ms: number;
    native_pc: number;
    combined_radius_m: number;
  };
  constants: {
    earth_mu_m3s2: number;
    earth_radius_m: number;
    max_dv_ms: number;
    pc_threshold: number;
  };
  keyframes: Keyframe[];
  decisions: Decision[];
  dense_frames: Frame[];
  result: {
    pc_final: number;
    total_fuel_used_ms: number;
    maneuver_count: number;
    collision_occurred: boolean;
  };
  baseline: {
    policy: string;
    pc_final: number;
    collision_occurred: boolean;
  };
}
