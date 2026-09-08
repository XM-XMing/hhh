#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure continuous Unity execution of a motion-primitive sequence.

This test uses UnityForestEnv.execute_primitive_stream(), the same continuous
execution path used by rollout collection.  It records every /xm/state sample,
writes a CSV trace, and reports per-primitive and steady-cruise statistics.
"""

from __future__ import annotations

from pathlib import Path


import argparse
import csv
import json
import math
import threading
import time
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import rospy

from planning.msg import XMState
from planning.runtime.unity_env import EnvConfig, UnityForestEnv


def _yaw_from_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _rotate_world_to_body(vector_xyz: Sequence[float], yaw: float) -> np.ndarray:
    vector = np.asarray(vector_xyz, dtype=np.float64).reshape(3)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.asarray(
        [
            cosine * vector[0] + sine * vector[1],
            -sine * vector[0] + cosine * vector[1],
            vector[2],
        ],
        dtype=np.float64,
    )


def _percentile(values: Sequence[float], q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, q)) if array.size else float("nan")


class StateRecorder:
    """Thread-safe recorder for the native Unity state stream."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._rows: List[Dict[str, Any]] = []
        self._subscriber = rospy.Subscriber("/xm/state", XMState, self._callback, queue_size=500)

    def _callback(self, message: XMState) -> None:
        orientation = message.orientation
        yaw = _yaw_from_xyzw(
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        )
        velocity_world = np.asarray(
            [message.velocity.x, message.velocity.y, message.velocity.z], dtype=np.float64
        )
        acceleration_world = np.asarray(
            [message.acceleration.x, message.acceleration.y, message.acceleration.z],
            dtype=np.float64,
        )
        velocity_body = _rotate_world_to_body(velocity_world, yaw)
        acceleration_body = _rotate_world_to_body(acceleration_world, yaw)
        stamp = message.header.stamp
        row = {
            "receive_wall_s": time.time(),
            "ros_stamp_ns": int(stamp.to_nsec()) if hasattr(stamp, "to_nsec") else 0,
            "sim_time_ns": int(message.sim_time_ns),
            "state_id": int(message.state_id),
            "x_m": float(message.position.x),
            "y_m": float(message.position.y),
            "z_m": float(message.position.z),
            "yaw_rad": float(yaw),
            "yaw_deg": float(math.degrees(yaw)),
            "vx_world_mps": float(velocity_world[0]),
            "vy_world_mps": float(velocity_world[1]),
            "vz_world_mps": float(velocity_world[2]),
            "speed_world_mps": float(np.linalg.norm(velocity_world)),
            "vx_body_mps": float(velocity_body[0]),
            "vy_body_mps": float(velocity_body[1]),
            "vz_body_mps": float(velocity_body[2]),
            "speed_body_mps": float(np.linalg.norm(velocity_body)),
            "ax_world_mps2": float(acceleration_world[0]),
            "ay_world_mps2": float(acceleration_world[1]),
            "az_world_mps2": float(acceleration_world[2]),
            "ax_body_mps2": float(acceleration_body[0]),
            "ay_body_mps2": float(acceleration_body[1]),
            "az_body_mps2": float(acceleration_body[2]),
            "min_clearance_m": float(message.min_clearance),
            "collided": int(bool(message.collided)),
            "altitude_violation": int(bool(message.altitude_violation)),
            "flags": int(message.flags),
            "primitive_index": -1,
            "action_id": -1,
        }
        with self._lock:
            self._rows.append(row)

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._rows]


def _parse_actions(text: str) -> List[int]:
    actions = [int(token.strip()) for token in text.split(",") if token.strip()]
    if not actions:
        raise ValueError("--actions must contain at least one action id")
    return actions


def _observation_state_time_ns(observation: Dict[str, Any]) -> int:
    return int(observation["state"].get("sim_time_ns", 0))


def _assign_primitive_windows(
    rows: List[Dict[str, Any]], boundaries: Sequence[Dict[str, Any]]
) -> None:
    windows = []
    for boundary in boundaries:
        if bool(boundary.get("skipped_invalid", False)) or "obs_after" not in boundary:
            continue
        start_ns = _observation_state_time_ns(boundary["obs_before"])
        end_ns = _observation_state_time_ns(boundary["obs_after"])
        windows.append(
            (
                int(boundary["stream_index"]),
                int(boundary["action_id"]),
                start_ns,
                end_ns,
            )
        )

    for row in rows:
        sim_time_ns = int(row["sim_time_ns"])
        for primitive_index, action_id, start_ns, end_ns in windows:
            if start_ns < sim_time_ns <= end_ns:
                row["primitive_index"] = primitive_index
                row["action_id"] = action_id
                break


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("no /xm/state samples were recorded")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _boundary_summary(
    boundaries: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for boundary in boundaries:
        primitive_index = int(boundary["stream_index"])
        action_id = int(boundary["action_id"])
        if bool(boundary.get("skipped_invalid", False)) or "obs_after" not in boundary:
            output.append(
                {
                    "primitive_index": primitive_index,
                    "action_id": action_id,
                    "skipped_invalid": True,
                    "break_reason": str(boundary.get("stream_break_reason", "invalid")),
                }
            )
            continue

        before = boundary["obs_before"]
        after = boundary["obs_after"]
        position_before = np.asarray(before["state"]["position"], dtype=np.float64)
        position_after = np.asarray(after["state"]["position"], dtype=np.float64)
        yaw_before = float(before["state"]["yaw"])
        yaw_after = float(after["state"]["yaw"])
        delta_body = _rotate_world_to_body(position_after - position_before, yaw_before)
        duration_s = max(
            0.0,
            (_observation_state_time_ns(after) - _observation_state_time_ns(before)) * 1.0e-9,
        )
        samples = [row for row in rows if int(row["primitive_index"]) == primitive_index]
        body_vx = [float(row["vx_body_mps"]) for row in samples]
        body_vy = [float(row["vy_body_mps"]) for row in samples]
        speeds = [float(row["speed_world_mps"]) for row in samples]
        output.append(
            {
                "primitive_index": primitive_index,
                "action_id": action_id,
                "skipped_invalid": False,
                "duration_s": duration_s,
                "delta_body_x_m": float(delta_body[0]),
                "delta_body_y_m": float(delta_body[1]),
                "delta_body_z_m": float(delta_body[2]),
                "endpoint_error_body_m": float(boundary["endpoint_error_body_m"]),
                "mean_body_vx_mps": float(np.mean(body_vx)) if body_vx else float("nan"),
                "mean_body_vy_mps": float(np.mean(body_vy)) if body_vy else float("nan"),
                "mean_speed_mps": float(np.mean(speeds)) if speeds else float("nan"),
                "p95_speed_mps": _percentile(speeds, 95.0),
                "max_speed_mps": float(np.max(speeds)) if speeds else float("nan"),
                "yaw_change_deg": float(math.degrees(_wrap_angle(yaw_after - yaw_before))),
                "collided": bool(boundary.get("collided_after", False)),
                "hard_altitude": bool(boundary.get("hard_altitude_after", False)),
                "success": bool(boundary.get("success_after", False)),
                "break_reason": str(boundary.get("stream_break_reason", "")),
                "sample_count": len(samples),
            }
        )
    return output


def _write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test continuous Unity execution and record the native 50 Hz state trace."
    )
    parser.add_argument("--actions", default="52,52,52,52,52,52,52,52,52,52")
    parser.add_argument("--start", type=float, nargs=3, default=[0.0, 0.0, 2.0])
    parser.add_argument("--goal", type=float, nargs=3, default=[40.0, 0.0, 2.0])
    parser.add_argument("--continuous", action="store_true", help="Accepted for CLI clarity; execution is always continuous.")
    parser.add_argument("--out-csv", default="/tmp/unity_motion_sequence.csv")
    parser.add_argument("--settle", type=float, default=0.30)
    parser.add_argument("--ready-timeout", type=float, default=20.0)
    parser.add_argument("--warmup-primitives", type=int, default=2)
    parser.add_argument("--speed-tolerance-mps", type=float, default=0.15)
    parser.add_argument("--forward-distance-tolerance-m", type=float, default=0.10)
    parser.add_argument("--max-lateral-drift-m", type=float, default=0.25)
    parser.add_argument("--max-yaw-drift-deg", type=float, default=2.0)
    parser.add_argument("--minimum-state-rate-hz", type=float, default=45.0)
    parser.add_argument("--maximum-state-gap-s", type=float, default=0.06)
    parser.add_argument("--ignore-action-mask", action="store_true")
    args = parser.parse_args()

    actions = _parse_actions(args.actions)
    output_path = Path(args.out_csv).expanduser().resolve()
    summary_csv_path = output_path.with_name(output_path.stem + "_primitives.csv")
    summary_json_path = output_path.with_name(output_path.stem + "_summary.json")

    config = EnvConfig(
        depth_out_width=160,
        depth_out_height=90,
        reset_settle_s=float(args.settle),
        primitive_post_wait_s=0.0,
        stop_at_primitive_end=False,
    )
    env = UnityForestEnv(start=args.start, goal=args.goal, config=config)
    recorder = StateRecorder()

    try:
        env.wait_until_ready(timeout_s=float(args.ready_timeout))
        observation_start = env.reset(start=args.start, goal=args.goal)
        recorder.clear()
        rospy.sleep(0.05)

        boundaries = env.execute_primitive_stream(
            actions,
            stop_at_end=True,
            require_current_mask=not bool(args.ignore_action_mask),
            expected_end_positions=None,
            max_drift_m=0.0,
            stop_on_terminal=True,
            stop_on_drift=False,
        )
        rospy.sleep(0.10)
        rows = recorder.snapshot()
    finally:
        env.stop()

    _assign_primitive_windows(rows, boundaries)
    _write_csv(output_path, rows)
    primitive_rows = _boundary_summary(boundaries, rows)
    _write_summary_csv(summary_csv_path, primitive_rows)

    assigned_rows = [row for row in rows if int(row["primitive_index"]) >= 0]
    sim_times = np.asarray([int(row["sim_time_ns"]) for row in assigned_rows], dtype=np.int64)
    if sim_times.size >= 2:
        unique_times = np.unique(sim_times)
        gaps_s = np.diff(unique_times).astype(np.float64) * 1.0e-9
        duration_s = max(1.0e-9, (unique_times[-1] - unique_times[0]) * 1.0e-9)
        state_rate_hz = float((unique_times.size - 1) / duration_s)
        max_state_gap_s = float(np.max(gaps_s)) if gaps_s.size else float("nan")
    else:
        state_rate_hz = float("nan")
        max_state_gap_s = float("nan")

    warmup = max(0, min(int(args.warmup_primitives), len(actions)))
    stable_rows = [row for row in assigned_rows if int(row["primitive_index"]) >= warmup]
    stable_primitive_rows = [
        row
        for row in primitive_rows
        if not bool(row.get("skipped_invalid", False))
        and int(row["primitive_index"]) >= warmup
    ]

    stable_body_vx = [float(row["vx_body_mps"]) for row in stable_rows]
    stable_speed = [float(row["speed_world_mps"]) for row in stable_rows]
    stable_forward_displacements = [
        float(row["delta_body_x_m"]) for row in stable_primitive_rows
    ]

    start_position = np.asarray(observation_start["state"]["position"], dtype=np.float64)
    start_yaw = float(observation_start["state"]["yaw"])
    completed_boundaries = [
        boundary
        for boundary in boundaries
        if not bool(boundary.get("skipped_invalid", False)) and "obs_after" in boundary
    ]
    if completed_boundaries:
        end_observation = completed_boundaries[-1]["obs_after"]
        end_position = np.asarray(end_observation["state"]["position"], dtype=np.float64)
        end_yaw = float(end_observation["state"]["yaw"])
    else:
        end_position = start_position.copy()
        end_yaw = start_yaw
    total_delta_body = _rotate_world_to_body(end_position - start_position, start_yaw)
    total_yaw_drift_deg = float(math.degrees(_wrap_angle(end_yaw - start_yaw)))

    center_action_id = int(env.mpl.center_action_id)
    target_forward_speed = float(env.mpl.target_forward_speed_mps)
    target_forward_distance = float(env.mpl.forward_distance_m)
    center_cruise = bool(actions and all(action_id == center_action_id for action_id in actions))

    invalid_count = sum(bool(row.get("skipped_invalid", False)) for row in primitive_rows)
    collision_seen = any(bool(row.get("collided", 0)) for row in rows) or any(
        bool(row.get("collided", False)) for row in primitive_rows
    )
    altitude_seen = any(bool(row.get("altitude_violation", 0)) for row in rows) or any(
        bool(row.get("hard_altitude", False)) for row in primitive_rows
    )
    complete = len(completed_boundaries) == len(actions)

    checks: Dict[str, bool] = {
        "all_primitives_completed": bool(complete),
        "no_invalid_action": invalid_count == 0,
        "no_collision": not collision_seen,
        "no_hard_altitude_violation": not altitude_seen,
        "state_rate_ok": bool(np.isfinite(state_rate_hz) and state_rate_hz >= float(args.minimum_state_rate_hz)),
        "state_gap_ok": bool(np.isfinite(max_state_gap_s) and max_state_gap_s <= float(args.maximum_state_gap_s)),
    }

    stable_mean_body_vx = _finite_mean(stable_body_vx)
    stable_mean_speed = _finite_mean(stable_speed)
    stable_mean_forward_displacement = _finite_mean(stable_forward_displacements)
    if center_cruise:
        checks.update(
            {
                "stable_forward_speed_ok": bool(
                    np.isfinite(stable_mean_body_vx)
                    and abs(stable_mean_body_vx - target_forward_speed)
                    <= float(args.speed_tolerance_mps)
                ),
                "stable_forward_distance_ok": bool(
                    np.isfinite(stable_mean_forward_displacement)
                    and abs(stable_mean_forward_displacement - target_forward_distance)
                    <= float(args.forward_distance_tolerance_m)
                ),
                "lateral_drift_ok": abs(float(total_delta_body[1])) <= float(args.max_lateral_drift_m),
                "yaw_drift_ok": abs(total_yaw_drift_deg) <= float(args.max_yaw_drift_deg),
            }
        )

    result = {
        "motion_primitive_contract_sha256": str(env.mpl.contract_sha256),
        "actions": actions,
        "center_action_id": center_action_id,
        "center_cruise_check": center_cruise,
        "target_forward_speed_mps": target_forward_speed,
        "target_forward_distance_m": target_forward_distance,
        "primitive_duration_s": float(env.mpl.duration_s),
        "completed_primitives": len(completed_boundaries),
        "requested_primitives": len(actions),
        "state_samples": len(assigned_rows),
        "state_rate_hz": state_rate_hz,
        "max_state_gap_s": max_state_gap_s,
        "warmup_primitives": warmup,
        "stable_sample_count": len(stable_rows),
        "stable_mean_body_vx_mps": stable_mean_body_vx,
        "stable_p05_body_vx_mps": _percentile(stable_body_vx, 5.0),
        "stable_p95_body_vx_mps": _percentile(stable_body_vx, 95.0),
        "stable_mean_speed_mps": stable_mean_speed,
        "stable_p95_speed_mps": _percentile(stable_speed, 95.0),
        "stable_mean_forward_displacement_m": stable_mean_forward_displacement,
        "total_delta_body_m": [float(value) for value in total_delta_body],
        "total_yaw_drift_deg": total_yaw_drift_deg,
        "checks": checks,
        "result": "PASS" if all(checks.values()) else "FAIL",
        "state_csv": str(output_path),
        "primitive_csv": str(summary_csv_path),
    }
    summary_json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("UNITY_MOTION_SEQUENCE")
    print("  contract_sha256:", result["motion_primitive_contract_sha256"])
    print("  actions:", ",".join(str(action_id) for action_id in actions))
    print("  completed: {}/{}".format(len(completed_boundaries), len(actions)))
    print("  state_samples:", len(assigned_rows))
    print("  state_rate_hz: {:.3f}".format(state_rate_hz))
    print("  max_state_gap_s: {:.4f}".format(max_state_gap_s))
    print("  warmup_primitives:", warmup)
    print("  stable_mean_body_vx_mps: {:.3f}".format(stable_mean_body_vx))
    print("  stable_body_vx_p05/p95_mps: {:.3f}/{:.3f}".format(
        result["stable_p05_body_vx_mps"], result["stable_p95_body_vx_mps"]
    ))
    print("  stable_mean_speed_mps: {:.3f}".format(stable_mean_speed))
    print("  stable_mean_forward_displacement_m: {:.3f}".format(stable_mean_forward_displacement))
    print("  total_delta_body_m:", result["total_delta_body_m"])
    print("  total_yaw_drift_deg: {:.3f}".format(total_yaw_drift_deg))
    print("  checks:", checks)
    print("  state_csv:", output_path)
    print("  primitive_csv:", summary_csv_path)
    print("  summary_json:", summary_json_path)
    print("RESULT=" + result["result"])
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
