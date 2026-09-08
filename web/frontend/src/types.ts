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
  encounter?: Encounter;
  truth_encounter?: Encounter;
}

export interface Encounter {
  miss_vector_m: [number, number];
  covariance_m2: [[number, number], [number, number]];
  time_to_tca_s: number;
}
export interface ReplayEntry {
  id: string;
  title: string;
  event_id: string | number;
  policy_id: string;
  path: string;
  verified: boolean;
  summary: { pc_final: number; total_fuel_used_ms: number };
}
export interface ReplayIndex {
  schema_version: string;
  replays: ReplayEntry[];
}

export interface Decision {
  t_s: number;
  action_dv_ms: [number, number, number];
  action_magnitude_ms: number;
  fuel_before_ms: number;
  fuel_after_ms: number;
  observation?: { pc_estimate: number; vector: number[]; version: string };
  encounter?: Encounter;
}

export interface Scenario {
  seed: number;
  miss_distance_m: number;
  relative_speed_ms: number;
  native_pc: number;
  combined_radius_m: number;
}

export interface SimulationResult {
  id?: string;
  policy_id?: string;
  schema_version?: string;
  provenance?: {
    source_event_id: string | number;
    kind: string;
    simulator: string;
    generated_at: string;
    seed: number;
    checkpoint_sha256?: string;
    fidelity_note?: string;
    schedule_kind?: string;
    observation_note?: string;
    verification?: Record<string, unknown>;
  };
  seed: number;
  scenario: {
    esa_reported_pc?: number;
    source_native_pc?: number;
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
    collision_occurred: boolean | null;
    status?: "completed" | "failed";
    radius_sensitivity_foster?: Record<string, number>;
  };
  baseline: {
    policy: string;
    pc_final: number;
    collision_occurred: boolean | null;
  };
}
