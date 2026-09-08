"""Versioned application contract. Importing it never loads the simulator."""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def require_scored_episode(info: dict) -> None:
    """Never publish failure sentinels as verified terminal physics."""
    pc = info.get("pc_final")
    if (info.get("status") != "completed"
            or info.get("pc_final_valid") is not True
            or not isinstance(pc, (int, float))
            or not math.isfinite(pc) or not 0 <= pc <= 1):
        raise ValueError(info.get("failure_reason") or "Simulator did not produce a valid terminal risk score")


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: str = Field(pattern=r"^kelvins-[0-9]+$")
    policy_id: Literal["v1", "never_maneuver", "threshold", "planner", "v2"] = "v1"
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    variant: Literal["historical_geometry", "posterior_variant"] = "historical_geometry"
    radius_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    schedule_mode: Literal["controlled", "source"] = "controlled"
    max_lead_days: float | None = Field(default=None, ge=0.02, le=7.0)
    observation_seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class JobResponse(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    id: str
    status: Literal["running", "completed", "failed", "cancelled"]
    progress: float = Field(ge=0, le=1)
    cached: bool = False
    artifact_url: str | None = None
    error: dict[str, str] | None = None


class Frame(BaseModel):
    t_s: float
    ego_r: tuple[float, float, float]
    ego_v: tuple[float, float, float]
    sec_r: tuple[float, float, float]
    sec_v: tuple[float, float, float]


class ReplayArtifact(BaseModel):
    """Validate the stable envelope; scientific packets remain extensible."""
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    schema_version: Literal["2.0"]
    id: str
    policy_id: str
    seed: int
    provenance: dict
    scenario: dict
    constants: dict
    keyframes: list[dict] = Field(min_length=2)
    decisions: list[dict] = Field(min_length=1)
    dense_frames: list[Frame] = Field(min_length=2)
    result: dict
    baseline: dict
