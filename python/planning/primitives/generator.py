#!/usr/bin/env python3
"""Generate the 105-action XMflight motion-primitive library.

The primary motion contract is defined by:
  * target forward speed
  * forward displacement
  * control period

Primitive duration and frame count are derived and validated. The generated
metadata includes a content hash so rollout, label, checkpoint, and evaluation
artifacts can reject incompatible motion libraries.
"""

from __future__ import annotations

from pathlib import Path


import argparse
import hashlib
import json
import math
from typing import Dict, List, Tuple

import numpy as np

from planning.primitives.library import load_motion_primitive_config, resolve_package_path


def _save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _quintic_smoothstep(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(s, dtype=np.float64)
    q = 10.0 * values**3 - 15.0 * values**4 + 6.0 * values**5
    dq = 30.0 * values**2 - 60.0 * values**3 + 30.0 * values**4
    ddq = 60.0 * values - 180.0 * values**2 + 120.0 * values**3
    return q, dq, ddq


def _quintic_hermite_normalized(
    s: np.ndarray,
    p0: float,
    dp0: float,
    ddp0: float,
    p1: float,
    dp1: float,
    ddp1: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(s, dtype=np.float64)
    c0 = float(p0)
    c1 = float(dp0)
    c2 = 0.5 * float(ddp0)
    rhs = np.asarray(
        [
            float(p1) - (c0 + c1 + c2),
            float(dp1) - (c1 + 2.0 * c2),
            float(ddp1) - 2.0 * c2,
        ],
        dtype=np.float64,
    )
    matrix = np.asarray(
        [[1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [6.0, 12.0, 20.0]],
        dtype=np.float64,
    )
    c3, c4, c5 = np.linalg.solve(matrix, rhs)
    position = c0 + c1 * values + c2 * values**2 + c3 * values**3 + c4 * values**4 + c5 * values**5
    first = c1 + 2.0 * c2 * values + 3.0 * c3 * values**2 + 4.0 * c4 * values**3 + 5.0 * c5 * values**4
    second = 2.0 * c2 + 6.0 * c3 * values + 12.0 * c4 * values**2 + 20.0 * c5 * values**3
    return position, first, second


def _derive_timing(config: Dict) -> Tuple[float, float, float, int, float]:
    dynamics = config["dynamics"]
    geometry = config["geometry"]
    target_speed = float(dynamics["target_forward_speed_mps"])
    forward_distance = float(geometry["forward_distance_m"])
    control_dt = float(dynamics["control_dt_s"])
    tolerance = float(dynamics.get("center_speed_tolerance_mps", 0.01))
    if target_speed <= 0.0 or forward_distance <= 0.0 or control_dt <= 0.0:
        raise ValueError("target speed, forward distance, and control period must be positive")

    requested_duration = forward_distance / target_speed
    command_frames = int(round(requested_duration / control_dt))
    if command_frames < 2:
        raise ValueError("motion primitive is too short for the configured control period")
    actual_duration = command_frames * control_dt
    actual_speed = forward_distance / actual_duration
    if abs(actual_speed - target_speed) > tolerance:
        raise ValueError(
            "forward distance and target speed do not align with the control period: "
            f"requested_duration={requested_duration:.9f}s frames={command_frames} "
            f"actual_duration={actual_duration:.9f}s actual_speed={actual_speed:.9f}m/s"
        )
    return target_speed, forward_distance, control_dt, command_frames, actual_duration


def _path_at_s(
    s: np.ndarray,
    forward_distance: float,
    lateral_endpoint: float,
    terminal_heading_rad: float,
    vertical_endpoint: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(s, dtype=np.float64)
    x = forward_distance * values
    dx = np.full_like(values, forward_distance)
    ddx = np.zeros_like(values)

    terminal_lateral_derivative = forward_distance * math.tan(float(terminal_heading_rad))
    y, dy, ddy = _quintic_hermite_normalized(
        values,
        p0=0.0,
        dp0=0.0,
        ddp0=0.0,
        p1=float(lateral_endpoint),
        dp1=terminal_lateral_derivative,
        ddp1=0.0,
    )
    q, dq, ddq = _quintic_smoothstep(values)
    z = float(vertical_endpoint) * q
    dz = float(vertical_endpoint) * dq
    ddz = float(vertical_endpoint) * ddq
    position = np.column_stack([x, y, z]).astype(np.float32)
    first = np.column_stack([dx, dy, dz]).astype(np.float32)
    second = np.column_stack([ddx, ddy, ddz]).astype(np.float32)
    return position, first, second


def _compute_profiles(
    forward_distance: float,
    lateral_endpoint: float,
    terminal_heading_rad: float,
    vertical_endpoint: float,
    duration: float,
    control_dt: float,
    command_frames: int,
) -> Dict[str, np.ndarray]:
    t_ref = np.linspace(0.0, duration, command_frames + 1, dtype=np.float64)
    s_ref = t_ref / duration
    pos_ref, dpos_ref, ddpos_ref = _path_at_s(
        s_ref,
        forward_distance,
        lateral_endpoint,
        terminal_heading_rad,
        vertical_endpoint,
    )
    vel_ref = dpos_ref / duration
    acc_ref = ddpos_ref / (duration * duration)
    yaw_ref = np.unwrap(np.arctan2(vel_ref[:, 1], vel_ref[:, 0])).astype(np.float32)
    yaw_rate = np.diff(yaw_ref) / control_dt

    t_cmd = (np.arange(command_frames, dtype=np.float64) + 0.5) * control_dt
    s_mid = np.clip(t_cmd / duration, 0.0, 1.0)
    _, dpos_mid, ddpos_mid = _path_at_s(
        s_mid,
        forward_distance,
        lateral_endpoint,
        terminal_heading_rad,
        vertical_endpoint,
    )
    velocity_world = dpos_mid / duration
    acceleration_world = ddpos_mid / (duration * duration)
    yaw_mid = 0.5 * (yaw_ref[:-1] + yaw_ref[1:])
    cosine = np.cos(yaw_mid)
    sine = np.sin(yaw_mid)
    vx_world = velocity_world[:, 0]
    vy_world = velocity_world[:, 1]
    body_velocity = np.column_stack(
        [
            cosine * vx_world + sine * vy_world,
            -sine * vx_world + cosine * vy_world,
            velocity_world[:, 2],
        ]
    )
    command_sequence = np.column_stack([body_velocity, yaw_rate]).astype(np.float32)
    return {
        "t_ref": t_ref.astype(np.float32),
        "t_cmd": t_cmd.astype(np.float32),
        "pos_ref": pos_ref,
        "vel_ref": vel_ref.astype(np.float32),
        "acc_ref": acc_ref.astype(np.float32),
        "yaw_ref": yaw_ref,
        "velocity_world_mid": velocity_world.astype(np.float32),
        "acceleration_world_mid": acceleration_world.astype(np.float32),
        "command_sequence": command_sequence,
    }


def _check_fov(
    pos_ref: np.ndarray,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
    margin_m: float,
    min_forward_m: float,
) -> Tuple[float, float]:
    x = pos_ref[:, 0].astype(np.float64)
    y = pos_ref[:, 1].astype(np.float64)
    z = pos_ref[:, 2].astype(np.float64)
    active = x >= max(float(min_forward_m), 1.0e-3)
    if not np.any(active):
        return 0.0, 0.0
    horizontal_half = math.radians(horizontal_fov_deg) * 0.5
    vertical_half = math.radians(vertical_fov_deg) * 0.5
    horizontal_denominator = np.maximum(x[active] * math.tan(horizontal_half), 1.0e-6)
    vertical_denominator = np.maximum(x[active] * math.tan(vertical_half), 1.0e-6)
    margin = max(0.0, float(margin_m))
    horizontal_ratio = float(np.max((np.abs(y[active]) + margin) / horizontal_denominator))
    vertical_ratio = float(np.max((np.abs(z[active]) + margin) / vertical_denominator))
    return horizontal_ratio, vertical_ratio


def _kinematics(
    command_sequence: np.ndarray,
    velocity_world_mid: np.ndarray,
    acceleration_world_mid: np.ndarray,
) -> Dict[str, float]:
    speed = np.linalg.norm(velocity_world_mid, axis=1)
    body_velocity = command_sequence[:, :3]
    return {
        "max_speed_mps": float(np.max(speed)),
        "max_horizontal_speed_mps": float(np.max(np.linalg.norm(velocity_world_mid[:, :2], axis=1))),
        "max_vertical_speed_mps": float(np.max(np.abs(velocity_world_mid[:, 2]))),
        "max_acceleration_mps2": float(np.max(np.linalg.norm(acceleration_world_mid, axis=1))),
        "max_horizontal_acceleration_mps2": float(np.max(np.linalg.norm(acceleration_world_mid[:, :2], axis=1))),
        "max_vertical_acceleration_mps2": float(np.max(np.abs(acceleration_world_mid[:, 2]))),
        "max_yaw_rate_radps": float(np.max(np.abs(command_sequence[:, 3]))),
        "max_abs_body_lateral_speed_mps": float(np.max(np.abs(command_sequence[:, 1]))),
        "max_speed_step_mps": float(np.max(np.abs(np.diff(speed)))) if speed.size > 1 else 0.0,
        "max_body_velocity_step_mps": float(np.max(np.linalg.norm(np.diff(body_velocity, axis=0), axis=1))) if body_velocity.shape[0] > 1 else 0.0,
    }


def _classify_lateral(value: float) -> str:
    if value < -0.20:
        return "left"
    if value > 0.20:
        return "right"
    return "straight"


def _classify_vertical(value: float) -> str:
    if value > 1.0e-3:
        return "up"
    if value < -1.0e-3:
        return "down"
    return "level"


def _contract_hash(config_contract: Dict, arrays: Dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(config_contract, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def generate_library(config: Dict) -> Dict:
    dynamics = config["dynamics"]
    geometry = config["geometry"]
    collision = config["collision"]
    target_speed, forward_distance, control_dt, command_frames, duration = _derive_timing(config)

    lateral_samples = np.asarray(geometry["lateral_endpoint_samples_m"], dtype=np.float32)
    heading_samples = np.deg2rad(np.asarray(geometry["terminal_heading_samples_deg"], dtype=np.float32))
    vertical_samples = np.asarray(geometry["vertical_endpoint_samples_m"], dtype=np.float32)
    if lateral_samples.shape != heading_samples.shape:
        raise ValueError("lateral endpoint and terminal heading arrays must have equal length")
    if lateral_samples.size != 15 or vertical_samples.size != 7:
        raise ValueError("motion library contract requires 15 horizontal x 7 vertical choices")

    action_count = int(lateral_samples.size * vertical_samples.size)
    reference_frames = command_frames + 1
    pos_ref = np.empty((action_count, reference_frames, 3), dtype=np.float32)
    vel_ref = np.empty_like(pos_ref)
    acc_ref = np.empty_like(pos_ref)
    yaw_ref = np.empty((action_count, reference_frames), dtype=np.float32)
    cmd_seq = np.empty((action_count, command_frames, 4), dtype=np.float32)
    lateral_end = np.empty((action_count,), dtype=np.float32)
    vertical_end = np.empty((action_count,), dtype=np.float32)
    terminal_heading = np.empty((action_count,), dtype=np.float32)
    horizontal_index = np.empty((action_count,), dtype=np.int32)
    vertical_index = np.empty((action_count,), dtype=np.int32)
    action_metadata: List[Dict] = []
    kinematic_rows: List[Dict[str, float]] = []

    horizontal_fov = float(geometry["horizontal_fov_deg"])
    vertical_fov = float(geometry["vertical_fov_deg"])
    fov_margin = float(geometry.get("fov_margin_m", 0.10))
    fov_min_forward = float(geometry.get("fov_check_min_forward_m", 0.50))

    for horizontal_id, (lateral, heading) in enumerate(zip(lateral_samples, heading_samples)):
        for vertical_id, vertical in enumerate(vertical_samples):
            action_id = horizontal_id * vertical_samples.size + vertical_id
            profile = _compute_profiles(
                forward_distance=forward_distance,
                lateral_endpoint=float(lateral),
                terminal_heading_rad=float(heading),
                vertical_endpoint=float(vertical),
                duration=duration,
                control_dt=control_dt,
                command_frames=command_frames,
            )
            pos_ref[action_id] = profile["pos_ref"]
            vel_ref[action_id] = profile["vel_ref"]
            acc_ref[action_id] = profile["acc_ref"]
            yaw_ref[action_id] = profile["yaw_ref"]
            cmd_seq[action_id] = profile["command_sequence"]
            horizontal_ratio, vertical_ratio = _check_fov(
                profile["pos_ref"], horizontal_fov, vertical_fov, fov_margin, fov_min_forward
            )
            row = _kinematics(
                profile["command_sequence"],
                profile["velocity_world_mid"],
                profile["acceleration_world_mid"],
            )
            row["max_horizontal_fov_ratio"] = horizontal_ratio
            row["max_vertical_fov_ratio"] = vertical_ratio
            kinematic_rows.append(row)

            lateral_end[action_id] = lateral
            vertical_end[action_id] = vertical
            terminal_heading[action_id] = heading
            horizontal_index[action_id] = horizontal_id
            vertical_index[action_id] = vertical_id
            action_metadata.append(
                {
                    "id": int(action_id),
                    "horizontal_index": int(horizontal_id),
                    "vertical_index": int(vertical_id),
                    "lateral_endpoint_m": float(lateral),
                    "vertical_endpoint_m": float(vertical),
                    "terminal_heading_rad": float(heading),
                    "terminal_heading_deg": float(math.degrees(float(heading))),
                    "lateral_mode": _classify_lateral(float(lateral)),
                    "vertical_mode": _classify_vertical(float(vertical)),
                    "duration_s": duration,
                    "control_dt_s": control_dt,
                    "command_frames": command_frames,
                    **row,
                }
            )

    summary = {
        key: {
            "min": float(min(row[key] for row in kinematic_rows)),
            "mean": float(np.mean([row[key] for row in kinematic_rows])),
            "max": float(max(row[key] for row in kinematic_rows)),
        }
        for key in kinematic_rows[0]
    }
    center_action = 7 * vertical_samples.size + 3
    center_speed = float(np.linalg.norm(cmd_seq[center_action, :, :3], axis=1).mean())
    feasibility = {
        "center_speed_ok": abs(center_speed - target_speed) <= float(dynamics["center_speed_tolerance_mps"]),
        "speed_ok": summary["max_speed_mps"]["max"] <= float(dynamics["max_speed_mps"]),
        "acceleration_ok": summary["max_acceleration_mps2"]["max"] <= float(dynamics["max_acceleration_mps2"]),
        "vertical_speed_ok": summary["max_vertical_speed_mps"]["max"] <= float(dynamics["max_vertical_speed_mps"]),
        "vertical_acceleration_ok": summary["max_vertical_acceleration_mps2"]["max"] <= float(dynamics["max_vertical_acceleration_mps2"]),
        "yaw_rate_ok": summary["max_yaw_rate_radps"]["max"] <= math.radians(float(dynamics["max_yaw_rate_degps"])),
        "speed_step_ok": summary["max_speed_step_mps"]["max"] <= float(dynamics["max_speed_step_mps"]),
        "body_velocity_step_ok": summary["max_body_velocity_step_mps"]["max"] <= float(dynamics["max_body_velocity_step_mps"]),
        "horizontal_fov_ok": summary["max_horizontal_fov_ratio"]["max"] <= 1.0,
        "vertical_fov_ok": summary["max_vertical_fov_ratio"]["max"] <= 1.0,
    }

    arrays = {
        "pos_ref": pos_ref,
        "vel_ref": vel_ref,
        "acc_ref": acc_ref,
        "yaw_ref": yaw_ref,
        "cmd_seq": cmd_seq,
        "t_ref": np.linspace(0.0, duration, reference_frames, dtype=np.float32),
        "t_cmd": (np.arange(command_frames, dtype=np.float32) + 0.5) * control_dt,
        "lateral_end": lateral_end,
        "vertical_end": vertical_end,
        "terminal_heading_rad": terminal_heading,
        "horizontal_index": horizontal_index,
        "vertical_index": vertical_index,
    }
    config_contract = {
        "target_forward_speed_mps": target_speed,
        "actual_forward_speed_mps": forward_distance / duration,
        "forward_distance_m": forward_distance,
        "duration_s": duration,
        "control_dt_s": control_dt,
        "command_frames": command_frames,
        "lateral_endpoint_samples_m": lateral_samples.tolist(),
        "terminal_heading_samples_deg": np.rad2deg(heading_samples).tolist(),
        "vertical_endpoint_samples_m": vertical_samples.tolist(),
        "inflate_radius_m": float(collision["inflate_radius_m"]),
    }
    contract_sha256 = _contract_hash(config_contract, arrays)
    metadata = {
        "contract_sha256": contract_sha256,
        "num_actions": action_count,
        "num_horizontal": int(lateral_samples.size),
        "num_vertical": int(vertical_samples.size),
        **config_contract,
        "reference_frames": reference_frames,
        "horizontal_fov_deg": horizontal_fov,
        "vertical_fov_deg": vertical_fov,
        "inflate_radius_m": float(collision["inflate_radius_m"]),
        "fov_margin_m": fov_margin,
        "fov_check_min_forward_m": fov_min_forward,
        "center_action_id": center_action,
        "center_speed_mps": center_speed,
        "kinematic_summary": summary,
        "feasibility": feasibility,
    }
    return {**arrays, "metadata": metadata, "actions": action_metadata}


def save_library(library: Dict, config: Dict) -> Tuple[Path, Path]:
    npz_path = resolve_package_path(config["paths"]["motion_primitives_npz"])
    metadata_path = resolve_package_path(config["paths"]["metadata_json"])
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in library.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(npz_path, **arrays)
    _save_json(metadata_path, {"meta": library["metadata"], "actions": library["actions"]})
    return npz_path, metadata_path


def print_summary(library: Dict, npz_path: Path, metadata_path: Path) -> None:
    metadata = library["metadata"]
    kinematics = metadata["kinematic_summary"]
    feasibility = metadata["feasibility"]
    print("MOTION_PRIMITIVE_GENERATION={}".format("PASS" if all(feasibility.values()) else "FAIL"))
    print("  npz:", npz_path)
    print("  metadata:", metadata_path)
    print("  contract_sha256:", metadata["contract_sha256"])
    print("  actions:", metadata["num_actions"])
    print("  target_forward_speed_mps: {:.3f}".format(metadata["target_forward_speed_mps"]))
    print("  actual_forward_speed_mps: {:.3f}".format(metadata["actual_forward_speed_mps"]))
    print("  forward_distance_m: {:.3f}".format(metadata["forward_distance_m"]))
    print("  duration_s: {:.3f}".format(metadata["duration_s"]))
    print("  control_dt_s: {:.3f}".format(metadata["control_dt_s"]))
    print("  command_frames:", metadata["command_frames"])
    print("  max_speed_mps: {:.3f}".format(kinematics["max_speed_mps"]["max"]))
    print("  max_acceleration_mps2: {:.3f}".format(kinematics["max_acceleration_mps2"]["max"]))
    print("  max_vertical_acceleration_mps2: {:.3f}".format(kinematics["max_vertical_acceleration_mps2"]["max"]))
    print("  max_yaw_rate_degps: {:.1f}".format(math.degrees(kinematics["max_yaw_rate_radps"]["max"])))
    print("  max_speed_step_mps: {:.4f}".format(kinematics["max_speed_step_mps"]["max"]))
    print("  max_body_velocity_step_mps: {:.4f}".format(kinematics["max_body_velocity_step_mps"]["max"]))
    print("  feasibility:", feasibility)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/motion_primitives.yaml")
    args = parser.parse_args()
    config = load_motion_primitive_config(args.config)
    library = generate_library(config)
    npz_path, metadata_path = save_library(library, config)
    print_summary(library, npz_path, metadata_path)
    return 0 if all(library["metadata"]["feasibility"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
