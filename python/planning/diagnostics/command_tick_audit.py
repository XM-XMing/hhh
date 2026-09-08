#!/usr/bin/env python3
"""Compare command publication and observed physics ticks across repeated audits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


AUDIT_CONTRACT_ID = "[DEBUG-CMD-TICK-c83e]command_frame_audit"


def _load(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episode = payload["episodes"][0]
    primitive = episode["primitives"][0]
    audit = primitive.get("command_frame_audit")
    if audit is None or audit.get("contract_id") != AUDIT_CONTRACT_ID:
        raise ValueError("missing command-frame audit: {}".format(path))
    return {
        "name": path.parent.name,
        "path": path,
        "episode": episode,
        "primitive": primitive,
        "audit": audit,
    }


def _load_bridge_commands(trace_path: Path) -> List[Dict]:
    path = trace_path.parent / "runtime_logs" / "bridge_command_audit.jsonl"
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        action = record.get("action", [])
        if (
            int(record.get("mode", -1)) == 3
            and len(action) == 4
            and abs(float(action[0]) - 3.0) <= 1.0e-6
            and max(abs(float(value)) for value in action[1:]) <= 1.0e-6
        ):
            records.append(record)
    return records


def _distance(left, right) -> float:
    return float(
        np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
    )


def _unique_states(audit: Dict, baseline: Dict, endpoint: Dict) -> List[Dict]:
    by_id = {int(baseline["state_id"]): dict(baseline)}
    for state in audit.get("states", []):
        by_id[int(state["state_id"])] = dict(state)
    by_id[int(endpoint["state_id"])] = dict(endpoint)
    return [by_id[key] for key in sorted(by_id)]


def _summarize(trace: Dict, movement_tolerance_m: float) -> Dict:
    primitive = trace["primitive"]
    audit = trace["audit"]
    before = primitive["before"]
    after = primitive["after"]
    baseline_id = int(before["state_id"])
    states = _unique_states(audit, before, after)
    states = [
        state
        for state in states
        if baseline_id <= int(state["state_id"]) <= int(after["state_id"])
    ]
    moving = []
    for left, right in zip(states, states[1:]):
        delta = _distance(left["position"], right["position"])
        if delta > float(movement_tolerance_m):
            moving.append({
                "tick_offset": int(right["state_id"]) - baseline_id,
                "sim_time_offset_ns": int(right["sim_time_ns"]) - int(before["sim_time_ns"]),
                "position_delta_m": delta,
            })

    commands = []
    execution = primitive.get("primitive_execution")
    bridge_commands = [] if execution is not None else _load_bridge_commands(trace["path"])
    if execution is not None:
        applied_frames = execution.get("applied_frames", [])
        for command, frame in zip(audit["commands"], applied_frames):
            commands.append({
                "command_index": int(command["command_index"]),
                "command": command["command"],
                "latest_state_tick_offset": None,
                "first_observed_tick_offset": int(frame["applied_state_id"]) - baseline_id,
                "publish_phase_after_latest_state_ns": None,
                "first_state_observation_latency_ns": None,
                "applied_execution_id": int(frame["execution_id"]),
                "applied_command_id": int(frame["command_id"]),
                "applied_state_id": int(frame["applied_state_id"]),
            })
    else:
        for command in audit["commands"]:
            latest = command["latest_state_before_publish"]
            first = command.get("first_observed_state_after_publish")
            commands.append({
                "command_index": int(command["command_index"]),
                "command": command["command"],
                "latest_state_tick_offset": None if latest is None else int(latest["state_id"]) - baseline_id,
                "first_observed_tick_offset": None if first is None else int(first["state_id"]) - baseline_id,
                "publish_phase_after_latest_state_ns": None if latest is None else (
                    int(command["publish_monotonic_ns"]) - int(latest["receive_monotonic_ns"])
                ),
                "first_state_observation_latency_ns": None if first is None else (
                    int(first["receive_monotonic_ns"])
                    - int(command["publish_completed_monotonic_ns"])
                ),
            })
    if bridge_commands:
        if len(bridge_commands) != len(commands):
            raise ValueError(
                "bridge/Python command count mismatch for {}: {} != {}".format(
                    trace["name"], len(bridge_commands), len(commands)
                )
            )
        for command, bridge in zip(commands, bridge_commands):
            command["bridge"] = {
                "command_id": int(bridge["command_id"]),
                "latest_unity_tick_offset": int(bridge["latest_unity_state_id"]) - baseline_id,
                "latest_unity_sim_time_ns": int(bridge["latest_unity_sim_time_ns"]),
                "publish_to_bridge_callback_ns": (
                    int(bridge["callback_monotonic_ns"])
                    - int(audit["commands"][command["command_index"]]["publish_monotonic_ns"])
                ),
                "publish_to_zmq_send_complete_ns": (
                    int(bridge["send_completed_monotonic_ns"])
                    - int(audit["commands"][command["command_index"]]["publish_monotonic_ns"])
                ),
                "bridge_send_duration_ns": (
                    int(bridge["send_completed_monotonic_ns"])
                    - int(bridge["send_started_monotonic_ns"])
                ),
                "send_rc": int(bridge["send_rc"]),
            }

    displacement = _distance(before["position"], after["position"])
    return {
        "repeat": trace["name"],
        "action": int(primitive["selected_action"]),
        "ros_use_sim_time": bool(audit["ros_use_sim_time"]),
        "command_count": len(commands),
        "commands": commands,
        "baseline_state_id": baseline_id,
        "endpoint_state_id": int(after["state_id"]),
        "endpoint_tick_offset": int(after["state_id"]) - baseline_id,
        "endpoint_sim_time_offset_ns": int(after["sim_time_ns"]) - int(before["sim_time_ns"]),
        "endpoint_position": after["position"],
        "endpoint_displacement_m": displacement,
        "endpoint_error_body_m": float(primitive["primitive_endpoint_error_body_m"]),
        "first_moving_tick_offset": None if not moving else moving[0]["tick_offset"],
        "effective_moving_tick_count": (
            len(execution["applied_frames"]) if execution is not None else len(moving)
        ),
        "moving_ticks": moving,
    }


def _first_command_mapping_mismatch(left: Dict, right: Dict):
    for left_cmd, right_cmd in zip(left["commands"], right["commands"]):
        left_mapping = (
            left_cmd["latest_state_tick_offset"], left_cmd["first_observed_tick_offset"]
        )
        right_mapping = (
            right_cmd["latest_state_tick_offset"], right_cmd["first_observed_tick_offset"]
        )
        if left_mapping != right_mapping:
            return {
                "command_index": int(left_cmd["command_index"]),
                "left_repeat": left["repeat"],
                "right_repeat": right["repeat"],
                "left_mapping": left_mapping,
                "right_mapping": right_mapping,
                "left_publish_phase_ns": left_cmd["publish_phase_after_latest_state_ns"],
                "right_publish_phase_ns": right_cmd["publish_phase_after_latest_state_ns"],
            }
    return None


def _one_tick_model_delta(command_speed: float, dt_s: float, tau_s: float, ticks: int) -> float:
    velocity = 0.0
    position = 0.0
    alpha = float(dt_s) / (float(tau_s) + float(dt_s))
    for _ in range(int(ticks)):
        old_velocity = velocity
        velocity = old_velocity + (float(command_speed) - old_velocity) * alpha
        position += (old_velocity + velocity) * 0.5 * float(dt_s)
    return position


def diagnose(root: Path, args) -> Dict:
    root = Path(root).expanduser().resolve()
    paths = sorted(root.glob("repeat_*/first_divergence_trace.json"))
    if len(paths) < int(args.min_repeats):
        raise ValueError("need at least {} traces; found {}".format(args.min_repeats, len(paths)))
    summaries = [
        _summarize(_load(path), float(args.movement_tolerance_m)) for path in paths
    ]
    if any(summary["command_count"] != 25 for summary in summaries):
        raise ValueError("expected exactly 25 command frames in every repeat")

    mismatches = []
    for left_index in range(len(summaries)):
        for right_index in range(left_index + 1, len(summaries)):
            mismatch = _first_command_mapping_mismatch(
                summaries[left_index], summaries[right_index]
            )
            if mismatch is not None:
                mismatches.append(mismatch)

    effective_counts = [item["effective_moving_tick_count"] for item in summaries]
    first_motion_ticks = [item["first_moving_tick_offset"] for item in summaries]
    first_consumption_relative_to_publish_state = [
        None
        if item["first_moving_tick_offset"] is None
        or item["commands"][0]["latest_state_tick_offset"] is None
        else (
            int(item["first_moving_tick_offset"])
            - int(item["commands"][0]["latest_state_tick_offset"])
        )
        for item in summaries
    ]
    first_consumption_relative_to_bridge_state = [
        None
        if item["first_moving_tick_offset"] is None
        or not item["commands"][0].get("bridge")
        else (
            int(item["first_moving_tick_offset"])
            - int(item["commands"][0]["bridge"]["latest_unity_tick_offset"])
        )
        for item in summaries
    ]
    endpoint_ticks_after_bridge_state_at_first_send = [
        None
        if not item["commands"][0].get("bridge")
        else (
            int(item["endpoint_tick_offset"])
            - int(item["commands"][0]["bridge"]["latest_unity_tick_offset"])
        )
        for item in summaries
    ]
    displacement_values = [item["endpoint_displacement_m"] for item in summaries]
    observed_span = max(displacement_values) - min(displacement_values)
    min_count = min(effective_counts)
    max_count = max(effective_counts)
    command_speed = float(np.linalg.norm(np.asarray(summaries[0]["commands"][0]["command"][:3])))
    predicted_span = _one_tick_model_delta(
        command_speed, float(args.control_dt_s), float(args.xy_time_constant_s), max_count
    ) - _one_tick_model_delta(
        command_speed, float(args.control_dt_s), float(args.xy_time_constant_s), min_count
    )
    integration_match_error = abs(observed_span - predicted_span)
    report = {
        "verdict": "RED" if len(set(effective_counts)) > 1 else "GREEN",
        "symptom": "same_command_sequence_effective_tick_count_mismatch",
        "repeat_count": len(summaries),
        "first_publish_mapping_mismatch": None if not mismatches else mismatches[0],
        "first_mismatching_command_index": (
            0 if len(set(first_consumption_relative_to_publish_state)) > 1 else None
        ),
        "first_inferred_command_consumption_tick_offsets": first_motion_ticks,
        "first_consumption_tick_after_latest_state_at_publish": (
            first_consumption_relative_to_publish_state
        ),
        "first_consumption_tick_after_latest_state_at_bridge_send": (
            first_consumption_relative_to_bridge_state
        ),
        "endpoint_tick_after_latest_state_at_first_bridge_send": (
            endpoint_ticks_after_bridge_state_at_first_send
        ),
        "effective_command_tick_counts": effective_counts,
        "endpoint_displacement_m": displacement_values,
        "endpoint_delta_m": observed_span,
        "one_tick_model_delta_m": predicted_span,
        "one_tick_integration_match_error_m": integration_match_error,
        "one_tick_integration_match": bool(
            max_count - min_count == 1
            and integration_match_error <= float(args.integration_match_tolerance_m)
        ),
        "ros_use_sim_time": [item["ros_use_sim_time"] for item in summaries],
        "repeats": summaries,
        "observability": {
            "publish_side": "direct",
            "first_motion_tick": "inferred_from_state_position_delta",
            "bridge_callback_send": (
                "direct" if all(item["commands"][0].get("bridge") for item in summaries)
                else "not_instrumented"
            ),
            "unity_receive_and_applied_command_id": "not_exposed_by_protocol",
            "per_frame_consumption_for_identical_commands": "not_identifiable_without_Unity_ack",
        },
    }
    report_path = root / "command_tick_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--min-repeats", type=int, default=5)
    parser.add_argument("--movement-tolerance-m", type=float, default=1.0e-6)
    parser.add_argument("--integration-match-tolerance-m", type=float, default=2.0e-4)
    parser.add_argument("--control-dt-s", type=float, default=0.02)
    parser.add_argument("--xy-time-constant-s", type=float, default=0.06)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = _parser().parse_args(argv)
    report = diagnose(args.root, args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        mismatch = report["first_publish_mapping_mismatch"]
        print("COMMAND_TICK_DIAGNOSIS verdict={} repeats={}".format(
            report["verdict"], report["repeat_count"]
        ))
        print("first_mismatching_command_index={}".format(
            "none" if report["first_mismatching_command_index"] is None
            else report["first_mismatching_command_index"]
        ))
        print("first_inferred_consumption_tick_offsets={}".format(
            report["first_inferred_command_consumption_tick_offsets"]
        ))
        print("first_consumption_tick_after_latest_state_at_publish={}".format(
            report["first_consumption_tick_after_latest_state_at_publish"]
        ))
        print("first_consumption_tick_after_latest_state_at_bridge_send={}".format(
            report["first_consumption_tick_after_latest_state_at_bridge_send"]
        ))
        print("endpoint_tick_after_latest_state_at_first_bridge_send={}".format(
            report["endpoint_tick_after_latest_state_at_first_bridge_send"]
        ))
        print("effective_command_tick_counts={}".format(
            report["effective_command_tick_counts"]
        ))
        print("endpoint_delta_m={:.9f} one_tick_model_delta_m={:.9f} match={}".format(
            report["endpoint_delta_m"], report["one_tick_model_delta_m"],
            report["one_tick_integration_match"],
        ))
        print("report={}".format(Path(args.root).resolve() / "command_tick_audit_report.json"))
        print("RESULT={}".format(report["verdict"]))
    return 1 if report["verdict"] == "RED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
