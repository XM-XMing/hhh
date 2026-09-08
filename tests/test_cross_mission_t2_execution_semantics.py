"""RED specifications for Layer-2 cross-run execution semantics."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from planning.diagnostics.cross_mission_t2_risk import summarize_mission_repeats


def _execution(
    *,
    execution_id: int,
    command_seed: int,
    first_state_id: int,
    first_sim_time_ns: int,
):
    frames = []
    for frame_index in range(25):
        frames.append(
            {
                "frame_index": frame_index,
                "applied_state_id": first_state_id + frame_index,
                "sim_time_ns": first_sim_time_ns + 20_000_000 * frame_index,
                "execution_status": 2 if frame_index == 24 else 1,
                "execution_id": execution_id,
                "command_id": (
                    command_seed ^ (0x1F123BB5 * (frame_index + 1))
                ) & ((1 << 63) - 1),
                "command_payload_fingerprint": "payload-{:02d}".format(frame_index),
            }
        )
    return {
        "execution_id": execution_id,
        "frame_count": 25,
        "requested_frame_count": 25,
        "status": "complete",
        "effective_integration_ticks": 25,
        "primitive_completed": True,
        "terminal_abort": False,
        "applied_frames": frames,
        "endpoint_state_id": first_state_id + 24,
    }


def _trace(execution):
    return {
        "checkpoint_sha256": "checkpoint",
        "mission_id": "mission",
        "episode_id": 1,
        "outcome": "success",
        "steps": 1,
        "primitives": [
            {
                "step": 0,
                "before": {
                    "state_fingerprint": "state",
                    "depth_fingerprint": "depth",
                    "observation_fingerprint": "observation",
                    "state_sequence": 10,
                    "state_id": 10,
                    "state_timestamp_ns": 100,
                    "depth_sequence": 20,
                    "depth_timestamp_ns": 90,
                    "sensor_skew_ns": 10,
                },
                "after": {"observation_fingerprint": "after"},
                "actor_logits_fingerprint": "logits",
                "action_mask_fingerprint": "mask",
                "action_mask": [True, True],
                "selected_action": 4,
                "terminal_reward_input_fingerprint": "terminal",
                "valid_action_count": 2,
                "top1_top2_margin": 0.1,
                "depth_mask_boundary_margin_m": 0.02,
                "minimum_safety_clearance_m": 0.4,
                "primitive_execution": execution,
            }
        ],
    }


def _rollout(outcome="success"):
    flags = {
        "success": "False",
        "collision": "False",
        "dead_end": "False",
        "timeout": "False",
        "far": "False",
        "hard_altitude": "False",
    }
    flags[outcome] = "True"
    return {
        "steps": "1",
        **flags,
        "stop_reason": outcome,
        "final_x": "1.0",
        "final_y": "2.0",
        "final_z": "1.5",
        "final_distance_xy": "0.2",
    }


def _summary(*traces, outcome="success"):
    return summarize_mission_repeats(
        mission={"episode_id": 1, "mission_id": "mission", "baseline_outcome": "success"},
        repeats=[
            {
                "repeat": "repeat_{:03d}".format(index),
                "trace": trace,
                "rollout": _rollout(outcome),
            }
            for index, trace in enumerate(traces, start=1)
        ],
    )


@pytest.mark.unit
def test_random_execution_and_command_correlation_ids_are_not_a_layer2_divergence():
    run_a = _trace(
        _execution(
            execution_id=5490942030757484029,
            command_seed=2519423526881718667,
            first_state_id=33,
            first_sim_time_ns=640_000_000,
        )
    )
    run_b = _trace(
        _execution(
            execution_id=4116063715641398704,
            command_seed=7985669837782709871,
            first_state_id=734,
            first_sim_time_ns=8_640_000_000,
        )
    )

    summary = _summary(run_a, run_b)

    assert summary["classification"] == "M0"


@pytest.mark.unit
def test_legacy_execution_without_payload_evidence_stays_m0_and_is_audited():
    left = _trace(
        _execution(
            execution_id=101,
            command_seed=1001,
            first_state_id=500,
            first_sim_time_ns=1_000_000_000,
        )
    )
    right = copy.deepcopy(left)
    for trace in (left, right):
        for frame in trace["primitives"][0]["primitive_execution"]["applied_frames"]:
            del frame["command_payload_fingerprint"]

    summary = _summary(left, right)

    assert summary["classification"] == "M0"
    assert summary["execution_payload_evidence_available"] is False


@pytest.mark.unit
def test_mixed_payload_evidence_fails_diagnostic_completeness():
    left = _trace(
        _execution(
            execution_id=101,
            command_seed=1001,
            first_state_id=500,
            first_sim_time_ns=1_000_000_000,
        )
    )
    right = copy.deepcopy(left)
    for frame in right["primitives"][0]["primitive_execution"]["applied_frames"]:
        del frame["command_payload_fingerprint"]

    with pytest.raises(ValueError, match="payload evidence"):
        _summary(left, right)


@pytest.mark.unit
@pytest.mark.parametrize(
    "difference",
    [
        "short_frame_sequence",
        "reordered_frames",
        "normalized_state_gap",
        "execution_status_sequence",
        "command_payload",
        "normalized_sim_time_gap",
        "endpoint_relation",
    ],
)
def test_execution_structural_differences_remain_layer2_m4(difference):
    left = _trace(
        _execution(
            execution_id=101,
            command_seed=1001,
            first_state_id=500,
            first_sim_time_ns=1_000_000_000,
        )
    )
    right = copy.deepcopy(left)
    execution = right["primitives"][0]["primitive_execution"]
    frames = execution["applied_frames"]

    if difference == "short_frame_sequence":
        del frames[-1]
    elif difference == "reordered_frames":
        frames[10], frames[11] = frames[11], frames[10]
    elif difference == "normalized_state_gap":
        for frame in frames[14:]:
            frame["applied_state_id"] += 1
        execution["endpoint_state_id"] += 1
    elif difference == "execution_status_sequence":
        frames[10]["execution_status"] = 2
    elif difference == "command_payload":
        frames[10]["command_payload_fingerprint"] = "changed-payload"
    elif difference == "normalized_sim_time_gap":
        for frame in frames[14:]:
            frame["sim_time_ns"] += 20_000_000
    elif difference == "endpoint_relation":
        execution["endpoint_state_id"] -= 1
    else:  # pragma: no cover - parameter list is the test contract.
        raise AssertionError(difference)

    assert _summary(left, right)["classification"] == "M4"


def _terminal_abort_execution(
    *,
    execution_id: int,
    command_seed: int,
    first_state_id: int,
    first_sim_time_ns: int,
    applied_tick_count: int = 14,
    classification: str = "TERMINAL_ABORT",
):
    completed = _execution(
        execution_id=execution_id,
        command_seed=command_seed,
        first_state_id=first_state_id,
        first_sim_time_ns=first_sim_time_ns,
    )
    frames = copy.deepcopy(completed["applied_frames"][:applied_tick_count])
    for frame in frames:
        frame["execution_status"] = 1
    terminal_state = {
        "execution_id": execution_id,
        "command_id": -1,
        "frame_index": -1,
        "execution_status": 3,
        "applied_state_id": first_state_id + applied_tick_count,
        "sim_time_ns": first_sim_time_ns + 20_000_000 * applied_tick_count,
        "collided": True,
    }
    return {
        "kind": classification,
        "terminal_reason": "collision",
        "effective_integration_ticks": applied_tick_count,
        "primitive_completed": False,
        "terminal_abort": classification == "TERMINAL_ABORT",
        "partial_receipt": {
            "execution_id": execution_id,
            "requested_frame_count": 25,
            "received_frames": frames + [terminal_state],
        },
        "terminal_state": terminal_state,
    }


def _terminal_abort_trace(execution_result):
    trace = _trace(None)
    trace["outcome"] = "collision"
    primitive = trace["primitives"][0]
    primitive.pop("primitive_execution")
    primitive.update(
        {
            "primitive_completed": False,
            "terminal_abort": True,
            "effective_integration_ticks": execution_result[
                "effective_integration_ticks"
            ],
            "primitive_execution_result": execution_result,
        }
    )
    return trace


@pytest.mark.unit
def test_terminal_abort_with_only_random_correlation_ids_is_layer2_m0():
    run_a = _terminal_abort_trace(
        _terminal_abort_execution(
            execution_id=8031253137307485836,
            command_seed=8702847692985510960,
            first_state_id=4815,
            first_sim_time_ns=96_279_997_847,
        )
    )
    run_b = _terminal_abort_trace(
        _terminal_abort_execution(
            execution_id=6406354282098454189,
            command_seed=122363948353366542,
            first_state_id=54719,
            first_sim_time_ns=1_094_359_975_539,
        )
    )

    assert _summary(run_a, run_b, outcome="collision")["classification"] == "M0"


@pytest.mark.unit
@pytest.mark.parametrize(
    "difference",
    ["applied_tick_count", "command_payload", "terminal_classification"],
)
def test_terminal_abort_structural_differences_remain_layer2_m4(difference):
    left = _terminal_abort_trace(
        _terminal_abort_execution(
            execution_id=10,
            command_seed=100,
            first_state_id=400,
            first_sim_time_ns=7_000_000_000,
        )
    )
    right = copy.deepcopy(left)
    result = right["primitives"][0]["primitive_execution_result"]
    if difference == "applied_tick_count":
        replacement = _terminal_abort_execution(
            execution_id=10,
            command_seed=100,
            first_state_id=400,
            first_sim_time_ns=7_000_000_000,
            applied_tick_count=13,
        )
        right = _terminal_abort_trace(replacement)
    elif difference == "command_payload":
        result["partial_receipt"]["received_frames"][10][
            "command_payload_fingerprint"
        ] = "changed-payload"
    elif difference == "terminal_classification":
        result["kind"] = "PROTOCOL_ERROR"
    else:  # pragma: no cover - parameter list is the test contract.
        raise AssertionError(difference)

    assert _summary(left, right, outcome="collision")["classification"] == "M4"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("difference", "expected"),
    [
        ("random_correlation_ids", "M0"),
        ("content", "M1"),
        ("mask", "M2"),
        ("action", "M3"),
        ("trajectory", "M4"),
    ],
)
def test_layer2_mission_classifier_preserves_m0_through_m4_semantics(difference, expected):
    traces = []
    for repeat in range(5):
        execution = _execution(
            execution_id=1_000 + repeat,
            command_seed=10_000 + repeat * 131,
            first_state_id=500 + repeat * 100,
            first_sim_time_ns=1_000_000_000 + repeat * 5_000_000_000,
        )
        traces.append(_trace(execution))
    if difference == "content":
        traces[-1]["primitives"][0]["before"]["depth_fingerprint"] = "other-depth"
        traces[-1]["primitives"][0]["before"]["observation_fingerprint"] = "other-observation"
        traces[-1]["primitives"][0]["actor_logits_fingerprint"] = "other-logits"
    elif difference == "mask":
        traces[-1]["primitives"][0]["action_mask_fingerprint"] = "other-mask"
    elif difference == "action":
        traces[-1]["primitives"][0]["selected_action"] = 5
    elif difference == "trajectory":
        traces[-1]["primitives"][0]["after"]["observation_fingerprint"] = "other-after"
    elif difference != "random_correlation_ids":  # pragma: no cover
        raise AssertionError(difference)

    assert _summary(*traces)["classification"] == expected


def _artifact_repeats():
    root = Path("/tmp/gate3_layer2_risk_screen_20260810/missions/episode_000451")
    if not root.exists():
        pytest.skip("current Gate-3 episode-451 artifact is not available")
    repeats = []
    for trace_path in sorted(root.glob("repeat_*/first_divergence_trace.json")):
        with (trace_path.parent / "rollout_index.csv").open(encoding="utf-8", newline="") as stream:
            rollout_rows = list(csv.DictReader(stream))
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        repeats.append(
            {
                "repeat": trace_path.parent.name,
                "trace": payload["episodes"][0],
                "rollout": rollout_rows[0],
            }
        )
    if len(repeats) != 5:
        pytest.skip("current Gate-3 episode-451 artifact does not contain five repeats")
    return repeats


def _artifact_execution_identity_signature(trace):
    return tuple(
        (
            primitive["primitive_execution"]["execution_id"],
            tuple(
                frame["command_id"]
                for frame in primitive["primitive_execution"]["applied_frames"]
            ),
        )
        for primitive in trace["primitives"]
    )


def _artifact_normalized_execution_structure(trace):
    signature = []
    for primitive in trace["primitives"]:
        execution = primitive["primitive_execution"]
        frames = execution["applied_frames"]
        first_state_id = frames[0]["applied_state_id"]
        first_sim_time_ns = frames[0]["sim_time_ns"]
        signature.append(
            (
                execution["requested_frame_count"],
                tuple(
                    (
                        frame["frame_index"],
                        frame["applied_state_id"] - first_state_id,
                        round(
                            (frame["sim_time_ns"] - first_sim_time_ns)
                            / 20_000_000
                        ),
                        frame["execution_status"],
                    )
                    for frame in frames
                ),
                execution["endpoint_state_id"] - first_state_id,
            )
        )
    return tuple(signature)


@pytest.mark.unit
def test_current_episode451_artifact_is_m0_after_execution_id_canonicalization():
    repeats = _artifact_repeats()
    traces = [item["trace"] for item in repeats]
    report = summarize_mission_repeats(
        mission={
            "episode_id": 451,
            "mission_id": traces[0]["mission_id"],
            "baseline_outcome": "success",
        },
        repeats=repeats,
    )

    assert len({_artifact_execution_identity_signature(trace) for trace in traces}) == 5
    assert len({_artifact_normalized_execution_structure(trace) for trace in traces}) == 1
    assert report["unique_execution_correlation_identity_trajectories"] == 5
    assert report["unique_execution_semantic_trajectories"] == 1
    assert report["primitive_count_range"] == [29, 29]
    assert report["unique_terminal_outcomes"] == 1
    assert len(
        {
            tuple(item["selected_action"] for item in trace["primitives"])
            for trace in traces
        }
    ) == 1
    for field in ("state_fingerprint", "depth_fingerprint", "observation_fingerprint"):
        assert len(
            {
                tuple(item["before"][field] for item in trace["primitives"])
                for trace in traces
            }
        ) == 1
    assert len(
        {
            tuple(item["actor_logits_fingerprint"] for item in trace["primitives"])
            for trace in traces
        }
    ) == 1
    assert len(
        {
            tuple(item["action_mask_fingerprint"] for item in trace["primitives"])
            for trace in traces
        }
    ) == 1
    assert report["classification"] == "M0"
