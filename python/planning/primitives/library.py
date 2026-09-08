"""Runtime loading and contract validation for XMflight motion primitives."""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
from typing import Any, Dict
import numpy as np
import yaml

from planning.common.paths import planning_package_root, resolve_package_path

def rotation_z(yaw: float) -> np.ndarray:
    """Return the ROS-map rotation matrix for a yaw angle in radians."""
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

def load_motion_primitive_config(path: str = "") -> Dict[str, Any]:
    if path:
        config_path = Path(path).expanduser()
        if not config_path.is_absolute() and not config_path.exists():
            config_path = planning_package_root() / config_path
    else:
        config_path = planning_package_root() / "config" / "motion_primitives.yaml"
    with config_path.resolve().open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid motion primitive config: {config_path}")
    return payload

class MotionPrimitiveLibrary:
    """Immutable, validated 105-action motion-primitive library."""

    def __init__(
        self,
        npz_path: str = "",
        metadata_json: str = "",
        validate_contract: bool = True,
    ):
        if not npz_path:
            environment_npz = str(
                os.environ.get("PLANNING_MOTION_PRIMITIVES_NPZ", "")
            ).strip()
            environment_metadata = str(
                os.environ.get("PLANNING_MOTION_PRIMITIVES_JSON", "")
            ).strip()
            if environment_npz or environment_metadata:
                if not environment_npz or not environment_metadata:
                    raise ValueError(
                        "PLANNING_MOTION_PRIMITIVES_NPZ and "
                        "PLANNING_MOTION_PRIMITIVES_JSON must be set together"
                    )
                npz_path = environment_npz
                metadata_json = environment_metadata
            else:
                config = load_motion_primitive_config()
                npz_path = config["paths"]["motion_primitives_npz"]
                metadata_json = config["paths"]["metadata_json"]

        self.npz_path = resolve_package_path(npz_path)
        self.metadata_path = (
            resolve_package_path(metadata_json)
            if metadata_json
            else self.npz_path.with_name(self.npz_path.stem + "_meta.json")
        )
        if not self.npz_path.exists():
            raise FileNotFoundError(
                f"motion primitive file not found: {self.npz_path}. "
                "Run: rosrun planning generate_motion_primitives.py"
            )
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"motion primitive metadata not found: {self.metadata_path}")

        with np.load(self.npz_path, allow_pickle=False) as data:
            self.pos_ref = np.ascontiguousarray(data["pos_ref"], dtype=np.float32)
            self.vel_ref = np.ascontiguousarray(data["vel_ref"], dtype=np.float32)
            self.acc_ref = np.ascontiguousarray(data["acc_ref"], dtype=np.float32)
            self.yaw_ref = np.ascontiguousarray(data["yaw_ref"], dtype=np.float32)
            self.cmd_seq = np.ascontiguousarray(data["cmd_seq"], dtype=np.float32)
            self.t_ref = np.ascontiguousarray(data["t_ref"], dtype=np.float32)
            self.t_cmd = np.ascontiguousarray(data["t_cmd"], dtype=np.float32)
            self.y_end = np.ascontiguousarray(data["lateral_end"], dtype=np.float32)
            self.z_end = np.ascontiguousarray(data["vertical_end"], dtype=np.float32)
            self.terminal_heading_rad = np.ascontiguousarray(
                data["terminal_heading_rad"], dtype=np.float32
            )
            self.horizontal_index = np.ascontiguousarray(data["horizontal_index"], dtype=np.int32)
            self.vertical_index = np.ascontiguousarray(data["vertical_index"], dtype=np.int32)

        metadata_object = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.meta = dict(metadata_object["meta"])
        self.actions = list(metadata_object["actions"])
        if validate_contract:
            self.validate_contract()

    def validate_contract(self, config_path: str = "") -> None:
        config = load_motion_primitive_config(config_path)
        dynamics = config["dynamics"]
        geometry = config["geometry"]
        target_speed = float(dynamics["target_forward_speed_mps"])
        forward_distance = float(geometry["forward_distance_m"])
        control_dt = float(dynamics["control_dt_s"])
        expected_frames = int(round((forward_distance / target_speed) / control_dt))
        expected_duration = expected_frames * control_dt
        errors = []
        if self.num_actions != 105:
            errors.append(f"num_actions={self.num_actions}, expected 105")
        if self.frames != expected_frames:
            errors.append(f"frames={self.frames}, expected {expected_frames}")
        if abs(self.duration_s - expected_duration) > 1.0e-6:
            errors.append(f"duration_s={self.duration_s}, expected {expected_duration}")
        if abs(self.control_dt_s - control_dt) > 1.0e-8:
            errors.append(f"control_dt_s={self.control_dt_s}, expected {control_dt}")
        if abs(self.forward_distance_m - forward_distance) > 1.0e-6:
            errors.append(
                f"forward_distance_m={self.forward_distance_m}, expected {forward_distance}"
            )
        if abs(self.target_forward_speed_mps - target_speed) > 1.0e-6:
            errors.append(
                f"target_forward_speed_mps={self.target_forward_speed_mps}, expected {target_speed}"
            )
        center = self.endpoint(self.center_action_id)
        expected_center = np.asarray([forward_distance, 0.0, 0.0], dtype=np.float32)
        if not np.allclose(center, expected_center, atol=2.0e-4):
            errors.append(f"center endpoint={center.tolist()}, expected {expected_center.tolist()}")
        failed = sorted(
            name for name, passed in dict(self.meta.get("feasibility", {})).items() if not bool(passed)
        )
        if failed:
            errors.append("failed feasibility checks: " + ",".join(failed))
        contract_hash = str(self.meta.get("contract_sha256", ""))
        if len(contract_hash) != 64:
            errors.append("missing or malformed contract_sha256")
        if errors:
            raise RuntimeError(
                "motion primitive contract mismatch: " + "; ".join(errors) + ". "
                "Regenerate with: rosrun planning generate_motion_primitives.py"
            )

    @property
    def contract_sha256(self) -> str:
        return str(self.meta["contract_sha256"])

    @property
    def num_actions(self) -> int:
        return int(self.cmd_seq.shape[0])

    @property
    def frames(self) -> int:
        return int(self.cmd_seq.shape[1])

    @property
    def control_dt_s(self) -> float:
        return float(self.meta["control_dt_s"])

    @property
    def duration_s(self) -> float:
        return float(self.meta["duration_s"])

    @property
    def forward_distance_m(self) -> float:
        return float(self.meta["forward_distance_m"])

    @property
    def target_forward_speed_mps(self) -> float:
        return float(self.meta["target_forward_speed_mps"])

    @property
    def center_action_id(self) -> int:
        return int(self.meta["center_action_id"])

    def command_sequence(self, action_id: int) -> np.ndarray:
        self._check_action_id(action_id)
        return self.cmd_seq[action_id].copy()

    def reference_path(self, action_id: int) -> np.ndarray:
        self._check_action_id(action_id)
        return self.pos_ref[action_id].copy()

    def action_metadata(self, action_id: int) -> Dict[str, Any]:
        self._check_action_id(action_id)
        return dict(self.actions[action_id])

    def endpoint(self, action_id: int) -> np.ndarray:
        self._check_action_id(action_id)
        return self.pos_ref[action_id, -1].copy()

    def valid_action_mask(
        self,
        current_z: float,
        z_min: float = 1.0,
        z_max: float = 3.0,
        margin: float = 0.0,
    ) -> np.ndarray:
        terminal_z = float(current_z) + self.z_end
        return (
            (terminal_z >= float(z_min) + float(margin))
            & (terminal_z <= float(z_max) - float(margin))
        ).astype(np.bool_)

    def nearest_action_by_endpoint(self, lateral_m: float, vertical_m: float) -> int:
        distance_squared = (self.y_end - float(lateral_m)) ** 2 + (
            self.z_end - float(vertical_m)
        ) ** 2
        return int(np.argmin(distance_squared))

    def _check_action_id(self, action_id: int) -> None:
        if action_id < 0 or action_id >= self.num_actions:
            raise IndexError(f"action_id {action_id} outside [0,{self.num_actions})")
