#!/usr/bin/env python3
"""Locate the first divergence across repeated single-mission Unity evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np


TRACE_CONTRACT_ID = "[DEBUG-EVAL-REPRO-7c91]policy_eval_first_divergence"


def _load_trace(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract_id") != TRACE_CONTRACT_ID:
        raise ValueError("first-divergence trace contract mismatch: {}".format(path))
    episodes = payload.get("episodes", [])
    if len(episodes) != 1:
        raise ValueError("reproducibility trace must contain exactly one episode")
    episode = dict(episodes[0])
    if episode.get("reset") is None:
        raise ValueError("reproducibility trace has no reset evidence")
    episode["trace_path"] = str(path)
    return episode


def _vector(record: Mapping, key: str) -> np.ndarray:
    return np.asarray(record[key], dtype=np.float64)


def _angle_delta(left: float, right: float) -> float:
    return abs(math.atan2(math.sin(left - right), math.cos(left - right)))


def _state_delta(left: Mapping, right: Mapping) -> Dict[str, float]:
    return {
        "position_m": float(np.linalg.norm(_vector(left, "position") - _vector(right, "position"))),
        "velocity_mps": float(np.linalg.norm(_vector(left, "velocity") - _vector(right, "velocity"))),
        "yaw_rad": _angle_delta(float(left["yaw"]), float(right["yaw"])),
    }


def _timing_delta(left: Mapping, right: Mapping) -> Dict[str, int]:
    return {
        "sensor_skew_ns": abs(int(left["sensor_skew_ns"]) - int(right["sensor_skew_ns"])),
        "state_sequence": abs(int(left["state_sequence"]) - int(right["state_sequence"])),
        "depth_sequence": abs(int(left["depth_sequence"]) - int(right["depth_sequence"])),
    }


def _numeric_diverged(delta: Mapping[str, float], args) -> bool:
    return bool(
        float(delta["position_m"]) > float(args.position_tolerance_m)
        or float(delta["velocity_mps"]) > float(args.velocity_tolerance_mps)
        or float(delta["yaw_rad"]) > float(args.yaw_tolerance_rad)
    )


def _logit_delta(left: Mapping, right: Mapping) -> float:
    left_values = {
        int(item["action"]): float(item["logit"])
        for item in left.get("actor_top_k", [])
    }
    right_values = {
        int(item["action"]): float(item["logit"])
        for item in right.get("actor_top_k", [])
    }
    common = set(left_values).intersection(right_values)
    if not common:
        return float("inf")
    return max(abs(left_values[index] - right_values[index]) for index in common)


def _stage_divergence(
    left: Mapping,
    right: Mapping,
    *,
    stage: str,
    step: int,
    args,
    include_timing: bool = True,
) -> Dict:
    categories = []
    evidence = {}
    state_delta = _state_delta(left, right)
    timing_delta = _timing_delta(left, right)
    evidence["state_delta"] = state_delta
    evidence["timing_delta"] = timing_delta
    if _numeric_diverged(state_delta, args):
        categories.append("A" if stage == "reset" else "E")
    if bool(include_timing) and timing_delta["sensor_skew_ns"] > int(args.sensor_skew_tolerance_ns):
        categories.append("F")
    if left.get("depth_fingerprint") != right.get("depth_fingerprint"):
        categories.append("B")
    if (
        "action_mask_fingerprint" in left
        and left.get("action_mask_fingerprint") != right.get("action_mask_fingerprint")
    ):
        categories.append("C")
    if (
        "actor_logits_fingerprint" in left
        and left.get("actor_logits_fingerprint") != right.get("actor_logits_fingerprint")
    ):
        categories.append("D")
        evidence["top_k_logit_max_common_delta"] = _logit_delta(left, right)
        evidence["left_top_k"] = left.get("actor_top_k", [])
        evidence["right_top_k"] = right.get("actor_top_k", [])
    return {
        "stage": stage,
        "step": int(step),
        "categories": categories,
        "evidence": evidence,
    }


def _compare(left: Mapping, right: Mapping, args, *, include_timing: bool = True) -> Dict:
    if (
        left.get("checkpoint_sha256") != right.get("checkpoint_sha256")
        or left.get("mission_id") != right.get("mission_id")
        or left.get("episode_id") != right.get("episode_id")
    ):
        raise ValueError("repeated traces do not bind the same checkpoint/mission")

    reset_left = dict(left["reset"])
    reset_right = dict(right["reset"])
    reset_left["action_mask_fingerprint"] = reset_left.get("action_mask_fingerprint")
    reset_right["action_mask_fingerprint"] = reset_right.get("action_mask_fingerprint")
    first = _stage_divergence(
        reset_left, reset_right, stage="reset", step=-1, args=args,
        include_timing=include_timing,
    )
    if first["categories"]:
        return first

    left_primitives = list(left.get("primitives", []))
    right_primitives = list(right.get("primitives", []))
    shared = min(len(left_primitives), len(right_primitives))
    for index in range(shared):
        left_step = left_primitives[index]
        right_step = right_primitives[index]
        before_left = {
            **left_step["before"],
            "action_mask_fingerprint": left_step["action_mask_fingerprint"],
            "actor_logits_fingerprint": left_step["actor_logits_fingerprint"],
            "actor_top_k": left_step["actor_top_k"],
        }
        before_right = {
            **right_step["before"],
            "action_mask_fingerprint": right_step["action_mask_fingerprint"],
            "actor_logits_fingerprint": right_step["actor_logits_fingerprint"],
            "actor_top_k": right_step["actor_top_k"],
        }
        first = _stage_divergence(
            before_left, before_right, stage="primitive_before", step=index, args=args,
            include_timing=include_timing,
        )
        if int(left_step["selected_action"]) != int(right_step["selected_action"]):
            first["evidence"]["selected_action"] = [
                int(left_step["selected_action"]),
                int(right_step["selected_action"]),
            ]
            if "D" not in first["categories"]:
                first["categories"].append("D")
        if first["categories"]:
            return first

        after_left = left_step["after"]
        after_right = right_step["after"]
        first = _stage_divergence(
            after_left, after_right, stage="primitive_after", step=index, args=args,
            include_timing=include_timing,
        )
        if first["categories"]:
            first["evidence"]["selected_action"] = int(left_step["selected_action"])
            first["evidence"]["endpoint_error_body_m"] = [
                float(left_step["primitive_endpoint_error_body_m"]),
                float(right_step["primitive_endpoint_error_body_m"]),
            ]
            return first

    if len(left_primitives) != len(right_primitives):
        return {
            "stage": "episode_length",
            "step": shared,
            "categories": ["E"],
            "evidence": {"primitive_counts": [len(left_primitives), len(right_primitives)]},
        }
    return {"stage": "none", "step": -1, "categories": [], "evidence": {}}


def _correlation(values, outcomes):
    values = np.asarray(values, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if values.size < 3 or np.std(values) == 0.0 or np.std(outcomes) == 0.0:
        return None
    return float(np.corrcoef(values, outcomes)[0, 1])


def diagnose(root: Path, args) -> Dict:
    root = Path(root).expanduser().resolve()
    paths = sorted(root.glob("repeat_*/first_divergence_trace.json"))
    if len(paths) < int(args.min_repeats):
        raise ValueError(
            "need at least {} completed traces under {}; found {}".format(
                args.min_repeats, root, len(paths)
            )
        )
    traces = [_load_trace(path) for path in paths]
    comparisons = []
    content_comparisons = []
    for left_index in range(len(traces)):
        for right_index in range(left_index + 1, len(traces)):
            result = _compare(traces[left_index], traces[right_index], args)
            result.update({
                "left": str(paths[left_index].parent.name),
                "right": str(paths[right_index].parent.name),
            })
            comparisons.append(result)
            content_result = _compare(
                traces[left_index], traces[right_index], args, include_timing=False
            )
            content_result.update({
                "left": str(paths[left_index].parent.name),
                "right": str(paths[right_index].parent.name),
            })
            content_comparisons.append(content_result)
    divergent = [item for item in comparisons if item["categories"]]
    stage_rank = {"reset": 0, "primitive_before": 1, "primitive_after": 2, "episode_length": 3, "none": 4}
    earliest = min(
        divergent,
        key=lambda item: (
            -1 if item["step"] < 0 else item["step"],
            stage_rank[item["stage"]],
            item["left"],
            item["right"],
        ),
    ) if divergent else None
    content_divergent = [item for item in content_comparisons if item["categories"]]
    earliest_content = min(
        content_divergent,
        key=lambda item: (
            -1 if item["step"] < 0 else item["step"],
            stage_rank[item["stage"]],
            item["left"],
            item["right"],
        ),
    ) if content_divergent else None

    reset_speed = [float(trace["reset"]["speed_mps"]) for trace in traces]
    reset_error = [float(trace["reset"]["reset_position_error_m"]) for trace in traces]
    reset_yaw = [float(trace["reset"]["yaw"]) for trace in traces]
    reset_skew = [int(trace["reset"]["sensor_skew_ns"]) for trace in traces]
    outcomes = [1.0 if trace.get("outcome") == "success" else 0.0 for trace in traces]
    report = {
        "verdict": "RED" if divergent else "GREEN",
        "symptom": "same_checkpoint_mission_first_divergence",
        "root": str(root),
        "repeat_count": len(traces),
        "checkpoint_sha256": traces[0]["checkpoint_sha256"],
        "mission_id": traces[0]["mission_id"],
        "episode_id": traces[0]["episode_id"],
        "outcomes": [trace.get("outcome") for trace in traces],
        "earliest_divergence": earliest,
        "earliest_content_divergence": earliest_content,
        "pairwise_comparisons": comparisons,
        "pairwise_content_comparisons": content_comparisons,
        "reset_evidence": {
            "speed_mps": reset_speed,
            "position_error_m": reset_error,
            "yaw_rad": reset_yaw,
            "sensor_skew_ns": reset_skew,
            "success_correlation": {
                "speed": _correlation(reset_speed, outcomes),
                "position_error": _correlation(reset_error, outcomes),
                "yaw": _correlation(reset_yaw, outcomes),
                "sensor_skew": _correlation(reset_skew, outcomes),
            },
        },
    }
    (root / "first_divergence_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--min-repeats", type=int, default=5)
    parser.add_argument("--position-tolerance-m", type=float, default=1.0e-4)
    parser.add_argument("--velocity-tolerance-mps", type=float, default=1.0e-4)
    parser.add_argument("--yaw-tolerance-rad", type=float, default=1.0e-5)
    parser.add_argument("--sensor-skew-tolerance-ns", type=int, default=1_000_000)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = _parser().parse_args(argv)
    report = diagnose(args.root, args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        earliest = report["earliest_divergence"]
        print("EVAL_REPRO_DIAGNOSIS verdict={} repeats={} outcomes={}".format(
            report["verdict"], report["repeat_count"], ",".join(report["outcomes"])
        ))
        if earliest is not None:
            print("first_divergence pair={}/{} stage={} step={} categories={}".format(
                earliest["left"], earliest["right"], earliest["stage"],
                earliest["step"], ",".join(earliest["categories"]),
            ))
            print("evidence={}".format(json.dumps(earliest["evidence"], sort_keys=True)))
        earliest_content = report["earliest_content_divergence"]
        if earliest_content is not None:
            print("first_content_divergence pair={}/{} stage={} step={} categories={}".format(
                earliest_content["left"], earliest_content["right"],
                earliest_content["stage"], earliest_content["step"],
                ",".join(earliest_content["categories"]),
            ))
            print("content_evidence={}".format(
                json.dumps(earliest_content["evidence"], sort_keys=True)
            ))
        print("report={}".format(Path(report["root"]) / "first_divergence_report.json"))
        print("RESULT={}".format(report["verdict"]))
    return 1 if report["verdict"] == "RED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
