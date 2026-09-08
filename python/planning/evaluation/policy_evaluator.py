#!/usr/bin/env python3
"""Evaluate a normalized learned policy in Unity."""

from __future__ import annotations

from pathlib import Path

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import time
from typing import Any, Dict, List, Mapping, Tuple, Union

import numpy as np
import zmq

from planning.bc.model import VectorNormalizer, build_model, mask_logits, require_torch
from planning.awac.model import build_critic
from planning.common.config import parse_bool
from planning.safety.depth_safety import DepthSafetyConfig, local_depth_action_mask
from planning.runtime.unity_env import (
    EnvConfig,
    PrimitiveExecutionAbortedError,
    UnityForestEnv,
)
from planning.contracts.primitive_execution import (
    PRIMITIVE_EXECUTION_RESULT_TERMINAL_ABORT,
    PrimitiveExecutionContractError,
    classify_primitive_execution_result,
)
from planning.contracts.feature import (
    FEATURE_CONTRACT_ID,
    INITIAL_PREV_ACTION,
    NUM_ACTIONS,
    POLICY_VECTOR_DIM,
    action_onehot,
    obs_continuous_vector,
    policy_input_contract_sha256,
    validate_checkpoint_policy_input_contract,
)
from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    LEGACY_ASYNC_OBSERVATION_CONTRACT,
)
from planning.evaluation.observation_provenance import (
    build_evaluation_provenance_summary,
    resolve_evaluation_observation_provenance,
)
from planning.mission.spec import (
    DEFAULT_MAX_PRIMITIVE_STEPS,
    GOAL_RADIUS_XY_M,
    MISSION_PLANAR_DISTANCE_M,
    TASK_CONTRACT_ID,
    mission_id_from_row,
    row_start_goal,
    task_contract_sha256,
    validate_mission_rows,
)
from planning.contracts.task import (
    resolve_policy_checkpoint_task_contract,
    task_contract_fields,
)
from planning.primitives.library import MotionPrimitiveLibrary, resolve_package_path
from planning.contracts.policy_runtime import (
    EXECUTION_MODES,
    MIN_FORMAL_HOLDOUT_EPISODES,
    POLICY_RUNTIME_CONTRACT_ID,
    POLICY_UNITY_EVALUATION_CONTRACT_ID,
    SAFETY_MASKS,
    checkpoint_depth_mask_numeric_config,
    resolve_evaluation_depth_mask_numeric_config,
    resolve_evaluation_runtime,
    summarize_policy_outcomes,
    terminal_done_reason,
)
from planning.runtime.reliable_endpoint_snapshot_provider import (
    BridgeSnapshotEndpointProvider,
    ZmqBridgeSnapshotRetriever,
)
from planning.runtime.reliable_unity_env_backend import ZmqReliableV4EnvironmentBackend
from planning.contracts.offpolicy import (
    validate_policy_checkpoint_algorithm,
)
from planning.diagnostics.pairing_counterfactual import (
    PAIRING_COUNTERFACTUAL_CONTRACT_ID,
    enumerate_admissible_pairs,
    risk_rankings,
    summarize_counterfactual_steps,
)
from planning.common.hashing import file_sha256
from planning.common import print_progress, read_csv, write_csv_atomic


FIRST_DIVERGENCE_TRACE_CONTRACT_ID = "[DEBUG-EVAL-REPRO-7c91]policy_eval_first_divergence"


def _load_torch_checkpoint(torch, path: Path, device):
    """Load a checkpoint without changing the frozen file on disk."""

    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(str(path), map_location=device)


def resolve_policy_checkpoint_normalizer(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    normalizer_checkpoint_path: Union[str, Path],
    torch,
    device,
    expected_max_primitive_steps: int,
) -> Tuple[VectorNormalizer, Path]:
    """Resolve the feature normalizer with an explicit provenance boundary.

    BC checkpoints are self-contained.  Historical AWAC checkpoints may omit
    the normalizer even though AWAC was initialized from BC; those checkpoints
    must name a source checkpoint whose complete-file SHA equals the recorded
    ``source_bc_checkpoint_sha256`` before its normalizer can be reused.
    """

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if "feature_mean" in checkpoint and "feature_std" in checkpoint:
        return VectorNormalizer.from_checkpoint(dict(checkpoint)), checkpoint_path

    source_value = str(normalizer_checkpoint_path or "").strip()
    if not source_value:
        raise ValueError(
            "policy checkpoint has no feature normalization; "
            "--normalizer-checkpoint is required for this legacy AWAC artifact"
        )
    source_path = Path(source_value).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            "normalizer source checkpoint does not exist: {}".format(source_path)
        )
    expected_source_sha = str(
        checkpoint.get("source_bc_checkpoint_sha256", "")
    ).strip().lower()
    if len(expected_source_sha) != 64:
        raise ValueError(
            "policy checkpoint is missing source_bc_checkpoint_sha256 for "
            "normalizer compatibility"
        )
    actual_source_sha = file_sha256(source_path).lower()
    if actual_source_sha != expected_source_sha:
        raise ValueError(
            "normalizer source SHA mismatch: received={} expected={}".format(
                actual_source_sha, expected_source_sha
            )
        )
    source_checkpoint = _load_torch_checkpoint(torch, source_path, device)
    validate_checkpoint_policy_input_contract(source_checkpoint)
    resolve_policy_checkpoint_task_contract(
        source_checkpoint,
        expected_max_primitive_steps=int(expected_max_primitive_steps),
        path="normalizer source task contract",
    )
    for field in ("observation_contract", "observation_source"):
        if source_checkpoint.get(field) != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
            raise ValueError("normalizer source {} mismatch".format(field))
    for field in (
        "feature_contract_id",
        "policy_input_contract_sha256",
        "vec_dim",
        "num_actions",
        "depth_history_frames",
        "initial_prev_action",
    ):
        if source_checkpoint.get(field) != checkpoint.get(field):
            raise ValueError("normalizer source {} mismatch".format(field))
    return VectorNormalizer.from_checkpoint(source_checkpoint), source_path


def normalize_policy_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fill legacy runtime fields from the checkpoint's resolved config.

    Historical AWAC checkpoints persisted the resolved runtime contract inside
    ``resolved_training_config`` but did not duplicate ``safety_mask`` and
    ``execution_mode`` at the checkpoint root.  Keep this compatibility seam
    strict: copy only fields that are absent at the root, and let the existing
    runtime-contract validator reject missing or invalid values.  No defaults
    are invented and the checkpoint on disk is never modified.
    """

    normalized = dict(checkpoint)
    resolved = checkpoint.get("resolved_training_config", {})
    if not isinstance(resolved, Mapping):
        return normalized
    for field in (
        "safety_mask",
        "execution_mode",
        "policy_runtime_contract_id",
    ):
        if field not in normalized and field in resolved:
            normalized[field] = resolved[field]
    return normalized


def array_fingerprint(value) -> str:
    """Return a stable fingerprint that binds array dtype, shape, and bytes."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def observation_trace(obs: Dict, *, previous_action: int) -> Dict:
    state = obs["state"]
    sensor_time = obs.get("sensor_time", {})
    sensor_seq = obs.get("sensor_seq", {})
    position = np.asarray(state["position"], dtype=np.float32)
    velocity = np.asarray(state["velocity"], dtype=np.float32)
    acceleration = np.asarray(state["acceleration"], dtype=np.float32)
    depth = np.asarray(obs["depth"], dtype=np.float32)
    fingerprint_payload = np.concatenate(
        (
            position,
            velocity,
            acceleration,
            np.asarray([float(state["yaw"]), float(previous_action)], dtype=np.float32),
            np.asarray(obs["goal"]["relative"], dtype=np.float32),
            depth.reshape(-1),
        )
    )
    safety = obs.get("safety", {})
    state_fingerprint_payload = np.concatenate((
        position,
        velocity,
        acceleration,
        np.asarray(state.get("orientation_xyzw", []), dtype=np.float32).reshape(-1),
        np.asarray([
            float(state["yaw"]),
            float(state.get("z", position[2])),
            float(state.get("flags", 0)),
            float(safety.get("min_clearance", np.nan)),
            float(bool(safety.get("collided", False))),
            float(bool(safety.get("altitude_violation", False))),
        ], dtype=np.float32),
        np.asarray(safety.get("front_clearances", []), dtype=np.float32).reshape(-1),
    ))
    return {
        "observation_fingerprint": array_fingerprint(fingerprint_payload),
        "state_fingerprint": array_fingerprint(state_fingerprint_payload),
        "depth_fingerprint": array_fingerprint(depth),
        "position": position.astype(float).tolist(),
        "velocity": velocity.astype(float).tolist(),
        "speed_mps": float(np.linalg.norm(velocity)),
        "yaw": float(state["yaw"]),
        "state_sequence": int(sensor_seq.get("state", -1)),
        "depth_sequence": int(sensor_seq.get("depth", -1)),
        "state_id": int(state.get("state_id", -1)),
        "sim_time_ns": int(state.get("sim_time_ns", 0)),
        "state_timestamp_ns": int(sensor_time.get("state_stamp_ns", 0)),
        "depth_timestamp_ns": int(sensor_time.get("depth_stamp_ns", 0)),
        "sensor_skew_ns": int(sensor_time.get("skew_ns", 0)),
    }


def parse_int_set(value: str) -> set:
    return {int(part.strip()) for part in str(value).split(",") if part.strip()}


def action_metric(info: Dict, key: str, action: int, default):
    values = info.get(key)
    if values is None or int(action) < 0:
        return default
    array = np.asarray(values)
    if int(action) >= int(array.size):
        return default
    return array.reshape(-1)[int(action)].item()


def resolve_policy_action_contract(temperature: float, *, audit_only: bool):
    """Resolve an explicit deterministic/stochastic evaluation contract."""
    value = float(temperature)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("--temperature must be finite and >= 0")
    action_mode = (
        "deterministic_argmax" if value == 0.0 else "masked_categorical"
    )
    if action_mode != "deterministic_argmax" and not bool(audit_only):
        raise ValueError(
            "formal policy evaluation requires --temperature 0; "
            "use --audit-only for stochastic masked-categorical diagnostics"
        )
    return action_mode, value


def depth_safety_config(env_config: EnvConfig, max_patch_radius_px: int) -> DepthSafetyConfig:
    return DepthSafetyConfig(
        horizontal_fov_deg=float(env_config.depth_camera_hfov_deg),
        vertical_fov_deg=float(env_config.depth_camera_vfov_deg),
        min_forward_m=float(env_config.depth_mask_min_forward_m),
        path_sample_stride=max(1, int(env_config.depth_mask_path_sample_stride)),
        collision_radius_m=float(env_config.depth_mask_collision_radius_m),
        depth_slack_m=float(env_config.depth_mask_slack_m),
        patch_radius_px=max(0, int(env_config.depth_mask_patch_radius_px)),
        max_patch_radius_px=max(0, int(max_patch_radius_px)),
        valid_depth_min_m=float(env_config.depth_min_m),
        valid_depth_max_m=float(env_config.depth_sensor_max_m) - 0.05,
    )


def append_depth_history(history: List[np.ndarray], depth: np.ndarray, frames: int) -> List[np.ndarray]:
    """Append the current depth frame and left-pad at an episode boundary."""
    if int(frames) <= 0:
        raise ValueError("depth history length must be positive")
    history.append(np.asarray(depth, dtype=np.float32))
    del history[:-int(frames)]
    return [history[0]] * (int(frames) - len(history)) + history


def observation_tensors(obs: Dict, prev_action: int, normalizer: VectorNormalizer, torch, device, depth_history=None):
    frames = depth_history if depth_history is not None else [np.asarray(obs["depth"], dtype=np.float32)]
    depth = np.stack(frames, axis=0)[None, :, :, :]
    continuous = normalizer.transform_continuous(obs_continuous_vector(obs))
    vector = np.concatenate([continuous, action_onehot(prev_action)], axis=0)[None, :]
    return (
        torch.from_numpy(depth).to(device=device, dtype=torch.float32),
        torch.from_numpy(vector).to(device=device, dtype=torch.float32),
    )


def choose_action(
    model,
    obs,
    prev_action,
    action_mask,
    normalizer,
    torch,
    device,
    temperature: float,
    depth_history=None,
    return_diagnostics: bool = False,
    return_full_logits: bool = False,
    return_full_probabilities: bool = False,
):
    depth, vector = observation_tensors(obs, prev_action, normalizer, torch, device, depth_history)
    mask = np.asarray(action_mask, dtype=np.bool_).reshape(1, -1)
    mask_tensor = torch.from_numpy(mask).to(device=device, dtype=torch.bool)
    with torch.no_grad():
        actor_logits = model(depth, vector)
        logits = mask_logits(actor_logits, mask_tensor)
        probabilities = torch.softmax(logits if temperature <= 0.0 else logits / float(temperature), dim=1)
        if temperature > 0.0:
            action = int(torch.multinomial(probabilities, 1).item())
        else:
            action = int(torch.argmax(logits, dim=1).item())
        confidence = float(probabilities[0, action].item())
        top5 = torch.topk(logits, k=5, dim=1).indices[0].cpu().numpy().astype(int).tolist()
        if bool(return_diagnostics):
            actor_logits_cpu = actor_logits[0].detach().cpu().numpy().astype(np.float32)
            top_indices = np.asarray(top5, dtype=np.int64)
            diagnostics = {
                "actor_logits_fingerprint": array_fingerprint(actor_logits_cpu),
                "actor_top_k": [
                    {
                        "action": int(index),
                        "logit": float(actor_logits_cpu[index]),
                    }
                    for index in top_indices
                ],
            }
            if bool(return_full_logits):
                diagnostics["actor_logits"] = actor_logits_cpu.astype(float).tolist()
            if bool(return_full_probabilities):
                diagnostics["action_probabilities"] = (
                    probabilities[0].detach().cpu().numpy().astype(np.float64).tolist()
                )
        else:
            diagnostics = None
    if bool(return_diagnostics):
        return action, confidence, top5, diagnostics
    return action, confidence, top5


@contextmanager
def preserve_torch_rng(torch):
    """Run diagnostic-only shadow inference without consuming policy RNG."""

    cpu_state = torch.random.get_rng_state()
    cuda_states = None
    if bool(torch.cuda.is_available()):
        cuda_states = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _distribution_kl(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("policy probability vectors must have the same non-zero shape")
    left = np.clip(left, 0.0, None)
    right = np.clip(right, 0.0, None)
    left = left / max(float(left.sum()), 1.0e-12)
    right = right / max(float(right.sum()), 1.0e-12)
    return float(np.sum(left * (np.log(np.clip(left, 1.0e-12, None)) - np.log(np.clip(right, 1.0e-12, None)))))


def _distribution_js(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("policy probability vectors must have the same non-zero shape")
    left = np.clip(left, 0.0, None)
    right = np.clip(right, 0.0, None)
    left = left / max(float(left.sum()), 1.0e-12)
    right = right / max(float(right.sum()), 1.0e-12)
    midpoint = 0.5 * (left + right)
    return float(0.5 * _distribution_kl(left, midpoint) + 0.5 * _distribution_kl(right, midpoint))


def _diagnostic_top1_and_margin(diagnostics: Mapping[str, Any]) -> Tuple[int, float, float]:
    probabilities = np.asarray(diagnostics.get("action_probabilities", []), dtype=np.float64)
    if probabilities.size == 0:
        raise ValueError("shadow trace requires full action probabilities")
    order = np.argsort(-probabilities, kind="stable")
    top1 = int(order[0])
    top1_probability = float(probabilities[top1])
    top2_probability = float(probabilities[order[1]]) if order.size > 1 else 0.0
    return top1, top1_probability, top1_probability - top2_probability


def build_shadow_trace_record(
    *,
    mission_id: str,
    episode_id: int,
    step: int,
    previous_action: int,
    valid_action_count: int,
    primary_action: int,
    shadow_action: int,
    primary_label: str,
    shadow_label: str,
    primary_diagnostics: Mapping[str, Any],
    shadow_diagnostics: Mapping[str, Any],
    observation_fingerprint: str = "",
    action_mask_fingerprint: str = "",
    critic_values: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build one immutable shadow row; it has no environment side effects."""

    primary_top1, primary_top1_probability, primary_margin = _diagnostic_top1_and_margin(
        primary_diagnostics
    )
    shadow_top1, shadow_top1_probability, shadow_margin = _diagnostic_top1_and_margin(
        shadow_diagnostics
    )
    primary_probabilities = np.asarray(
        primary_diagnostics["action_probabilities"], dtype=np.float64
    )
    shadow_probabilities = np.asarray(
        shadow_diagnostics["action_probabilities"], dtype=np.float64
    )
    primary_name = str(primary_label).strip().lower()
    shadow_name = str(shadow_label).strip().lower()
    bc_to_awac = None
    if primary_name in {"bc", "bc60k", "bc_calibration", "behavior_cloning"} and shadow_name in {
        "awac", "awac10k", "awac_online"
    }:
        bc_to_awac = _distribution_kl(primary_probabilities, shadow_probabilities)
    elif shadow_name in {"bc", "bc60k", "bc_calibration", "behavior_cloning"} and primary_name in {
        "awac", "awac10k", "awac_online"
    }:
        bc_to_awac = _distribution_kl(shadow_probabilities, primary_probabilities)
    record = {
        "mission_id": str(mission_id),
        "episode_id": int(episode_id),
        "step": int(step),
        "previous_action": int(previous_action),
        "valid_action_count": int(valid_action_count),
        "primary_policy_label": str(primary_label),
        "shadow_policy_label": str(shadow_label),
        "primary_action": int(primary_action),
        "shadow_action": int(shadow_action),
        "primary_top1_action": int(primary_top1),
        "shadow_top1_action": int(shadow_top1),
        "primary_top1_probability": float(primary_top1_probability),
        "shadow_top1_probability": float(shadow_top1_probability),
        "primary_selected_probability": float(
            primary_probabilities[int(primary_action)]
        ),
        "shadow_selected_probability": float(
            shadow_probabilities[int(shadow_action)]
        ),
        "primary_top1_margin": float(primary_margin),
        "shadow_top1_margin": float(shadow_margin),
        "bc_to_awac_kl": None if bc_to_awac is None else float(bc_to_awac),
        "js_divergence": float(_distribution_js(primary_probabilities, shadow_probabilities)),
        "top1_disagreement": bool(primary_top1 != shadow_top1),
        "selected_action_disagreement": bool(int(primary_action) != int(shadow_action)),
        "observation_fingerprint": str(observation_fingerprint),
        "action_mask_fingerprint": str(action_mask_fingerprint),
        "q_primary_action": None,
        "q_shadow_action": None,
        "q_primary_top1": None,
        "q_shadow_top1": None,
    }
    if critic_values is not None:
        record.update(dict(critic_values))
    return record


def critic_trace_values(
    *,
    obs: Mapping[str, Any],
    previous_action: int,
    action_mask: np.ndarray,
    depth_history,
    primary_diagnostics: Mapping[str, Any],
    shadow_diagnostics: Mapping[str, Any],
    primary_label: str,
    shadow_label: str,
    critic1,
    critic2,
    normalizer: VectorNormalizer,
    torch,
    device,
) -> Dict[str, Any]:
    """Evaluate frozen twin-Q values for a shadow row without updating state."""

    primary_top1 = _diagnostic_top1_and_margin(primary_diagnostics)[0]
    shadow_top1 = _diagnostic_top1_and_margin(shadow_diagnostics)[0]
    primary_name = str(primary_label).strip().lower()
    shadow_name = str(shadow_label).strip().lower()
    bc_names = {"bc", "bc60k", "bc_calibration", "behavior_cloning"}
    awac_names = {"awac", "awac10k", "awac_online"}
    if primary_name in bc_names and shadow_name in awac_names:
        bc_action, awac_action = primary_top1, shadow_top1
    elif primary_name in awac_names and shadow_name in bc_names:
        bc_action, awac_action = shadow_top1, primary_top1
    else:
        return {
            "critic_source_policy_label": "",
            "q_bc_top1": None,
            "q_awac_top1": None,
            "q_awac_minus_bc": None,
            "critic_prefers_awac": None,
        }
    depth, vector = observation_tensors(
        dict(obs), previous_action, normalizer, torch, device, depth_history
    )
    with torch.no_grad():
        q1 = critic1(depth, vector)[0].detach().cpu().numpy().astype(np.float64)
        q2 = critic2(depth, vector)[0].detach().cpu().numpy().astype(np.float64)
    qmin = np.minimum(q1, q2)
    valid = np.asarray(action_mask, dtype=np.bool_).reshape(-1)
    if not valid[int(bc_action)] or not valid[int(awac_action)]:
        raise ValueError("policy top-1 action is not valid under the shared mask")
    delta = float(qmin[int(awac_action)] - qmin[int(bc_action)])
    primary_q = float(qmin[primary_top1])
    shadow_q = float(qmin[shadow_top1])
    return {
        "critic_source_policy_label": str(
            "awac" if primary_name in awac_names else "awac"
        ),
        "q1_primary_action": float(q1[primary_top1]),
        "q2_primary_action": float(q2[primary_top1]),
        "q_primary_action": primary_q,
        "q1_shadow_action": float(q1[shadow_top1]),
        "q2_shadow_action": float(q2[shadow_top1]),
        "q_shadow_action": shadow_q,
        "q_primary_top1": primary_q,
        "q_shadow_top1": shadow_q,
        "q1_bc_top1": float(q1[bc_action]),
        "q2_bc_top1": float(q2[bc_action]),
        "q1_awac_top1": float(q1[awac_action]),
        "q2_awac_top1": float(q2[awac_action]),
        "q_bc_top1": float(qmin[bc_action]),
        "q_awac_top1": float(qmin[awac_action]),
        "q_awac_minus_bc": delta,
        "critic_prefers_awac": bool(delta > 0.0),
    }


def state_candidate_metadata(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return complete state evidence without hashing callback timing into content."""
    message = record["message"]
    orientation = np.asarray([
        message.orientation.x,
        message.orientation.y,
        message.orientation.z,
        message.orientation.w,
    ], dtype=np.float64)
    content = np.concatenate([
        np.asarray([message.state_id, message.flags], dtype=np.float64),
        np.asarray([message.position.x, message.position.y, message.position.z], dtype=np.float64),
        orientation,
        np.asarray([message.velocity.x, message.velocity.y, message.velocity.z], dtype=np.float64),
        np.asarray([message.acceleration.x, message.acceleration.y, message.acceleration.z], dtype=np.float64),
        np.asarray([
            float(message.min_clearance),
            *[float(value) for value in message.front_clearances],
            float(bool(message.collided)),
            float(bool(message.altitude_violation)),
        ], dtype=np.float64),
    ])
    full = np.concatenate([
        content,
        np.asarray([
            message.sim_time_ns,
            getattr(message, "applied_execution_id", -1),
            getattr(message, "applied_execution_frame_index", -1),
            getattr(message, "applied_command_id", -1),
            getattr(message, "execution_status", 0),
        ], dtype=np.float64),
    ])
    return {
        "state_sequence": int(record["state_sequence"]),
        "state_id": int(message.state_id),
        "state_timestamp_ns": int(message.sim_time_ns),
        "receive_monotonic_ns": int(record["receive_monotonic_ns"]),
        "position": [float(message.position.x), float(message.position.y), float(message.position.z)],
        "velocity": [float(message.velocity.x), float(message.velocity.y), float(message.velocity.z)],
        "acceleration": [
            float(message.acceleration.x), float(message.acceleration.y), float(message.acceleration.z)
        ],
        "yaw": float(math.atan2(
            2.0 * (orientation[3] * orientation[2] + orientation[0] * orientation[1]),
            1.0 - 2.0 * (orientation[1] ** 2 + orientation[2] ** 2),
        )),
        "collided": bool(message.collided),
        "altitude_violation": bool(message.altitude_violation),
        "min_clearance": float(message.min_clearance),
        "front_clearances": [float(value) for value in message.front_clearances],
        "state_content_fingerprint": array_fingerprint(content),
        "state_full_fingerprint": array_fingerprint(full),
        "_record": record,
    }


def depth_candidate_metadata(record: Mapping[str, Any]) -> Dict[str, Any]:
    message = record["message"]
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(str(message.encoding).encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.asarray(
        [message.height, message.width, message.step, message.is_bigendian], dtype=np.int64
    ).tobytes())
    digest.update(raw.tobytes())
    return {
        "depth_sequence": int(record["depth_sequence"]),
        "depth_timestamp_ns": int(UnityForestEnv._stamp_ns(message)),
        "receive_monotonic_ns": int(record["receive_monotonic_ns"]),
        "image_shape": [int(message.height), int(message.width)],
        "encoding": str(message.encoding),
        "depth_fingerprint": digest.hexdigest(),
        "_record": record,
    }


def terminal_reward_input_fingerprint(
    obs: Mapping[str, Any], mask: np.ndarray, *, episode_step: int, previous_action: int
) -> str:
    safety = obs["safety"]
    payload = np.concatenate([
        np.asarray(obs["state"]["position"], dtype=np.float64).reshape(3),
        np.asarray([
            obs["state"]["z"],
            obs["goal"]["distance_xy"],
            obs["goal"]["dz"],
            safety["min_clearance"],
            float(bool(safety["collided"])),
            float(bool(safety["altitude_violation"])),
            int(episode_step),
            int(previous_action),
        ], dtype=np.float64),
        np.asarray(safety["front_clearances"], dtype=np.float64).reshape(-1),
        np.asarray(mask, dtype=np.uint8).reshape(-1),
    ])
    return array_fingerprint(payload)


def audit_mask_boundary_margin_m(env: UnityForestEnv, mask_info: Mapping[str, Any]) -> float:
    """Return the nearest production depth-mask boundary as audit evidence only."""
    clearances = np.asarray(
        mask_info.get("depth_action_min_ray_clearance_m", []), dtype=np.float32
    )
    finite = clearances[np.isfinite(clearances)]
    if finite.size == 0:
        return float("nan")
    required = float(env.config.depth_mask_collision_radius_m) + float(
        env.config.depth_mask_slack_m
    )
    return float(np.min(np.abs(finite - required)))


def compact_pairing_candidate_counts(
    *, capture: Mapping[str, Any], max_sensor_skew_ns: int
) -> list[Dict[str, Any]]:
    """Count current-contract candidate pairs without replaying or saving depth."""
    records = []
    for boundary_index, boundary in enumerate(capture["boundaries"]):
        boundary_states = [
            state_candidate_metadata(record) for record in boundary["states"]
        ]
        boundary_depths = [
            depth_candidate_metadata(record) for record in boundary["depths"]
        ]
        pairs = enumerate_admissible_pairs(
            boundary_kind=str(boundary["boundary_kind"]),
            states=boundary_states,
            depths=boundary_depths,
            previous_state_sequence=int(boundary["previous_state_sequence"]),
            previous_depth_sequence=int(boundary["previous_depth_sequence"]),
            max_sensor_skew_ns=int(max_sensor_skew_ns),
            required_state_sequence=boundary.get("required_state_sequence"),
            required_state_timestamp_ns=boundary.get("required_state_timestamp_ns"),
        )
        records.append({
            "boundary_index": int(boundary_index),
            "boundary_kind": str(boundary["boundary_kind"]),
            "candidate_state_count": len(boundary_states),
            "candidate_depth_count": len(boundary_depths),
            "admissible_pair_count": len(pairs),
        })
    return records


def replay_pairing_counterfactuals(
    *,
    env: UnityForestEnv,
    capture: Mapping[str, Any],
    decision_contexts: List[Mapping[str, Any]],
    model,
    normalizer: VectorNormalizer,
    torch,
    device,
    max_sensor_skew_ns: int,
) -> Dict[str, Any]:
    """Replay all admissible pairs through production observation/mask/model seams."""
    states = [state_candidate_metadata(record) for record in capture["states"]]
    depths = [depth_candidate_metadata(record) for record in capture["depths"]]
    state_by_sequence = {item["state_sequence"]: item for item in states}
    depth_by_sequence = {item["depth_sequence"]: item for item in depths}
    replay_steps = []

    boundaries = list(capture["boundaries"])
    if len(boundaries) < len(decision_contexts):
        raise RuntimeError(
            "pairing audit captured {} boundaries for {} policy decisions".format(
                len(boundaries), len(decision_contexts)
            )
        )
    for step, context in enumerate(decision_contexts):
        boundary = boundaries[step]
        boundary_states = [
            state_by_sequence[int(record["state_sequence"])]
            for record in boundary["states"]
        ]
        boundary_depths = [
            depth_by_sequence[int(record["depth_sequence"])]
            for record in boundary["depths"]
        ]
        pairs = enumerate_admissible_pairs(
            boundary_kind=str(boundary["boundary_kind"]),
            states=boundary_states,
            depths=boundary_depths,
            previous_state_sequence=int(boundary["previous_state_sequence"]),
            previous_depth_sequence=int(boundary["previous_depth_sequence"]),
            max_sensor_skew_ns=int(max_sensor_skew_ns),
            required_state_sequence=boundary.get("required_state_sequence"),
            required_state_timestamp_ns=boundary.get("required_state_timestamp_ns"),
        )
        selected_state_sequence = (
            int(boundary["required_state_sequence"])
            if boundary.get("required_state_sequence") is not None
            else int(boundary["selected_state_sequence"])
        )
        selected_depth_sequence = int(boundary["selected_depth_sequence"])
        previous_action = int(context["previous_action"])
        actual_depth_stack = [
            np.asarray(frame, dtype=np.float32) for frame in context["depth_stack"]
        ]
        results = []
        for pair in pairs:
            state_metadata = pair["state"]
            depth_metadata = pair["depth"]
            obs = env.build_pairing_candidate_observation(
                state_metadata["_record"],
                depth_metadata["_record"],
                boundary["camera_info"],
            )
            mask, mask_info = env.get_action_mask(obs, return_info=True)
            depth_stack = list(actual_depth_stack[:-1]) + [np.asarray(obs["depth"], dtype=np.float32)]
            action, _, _, diagnostics = choose_action(
                model,
                obs,
                previous_action,
                mask,
                normalizer,
                torch,
                device,
                0.0,
                depth_stack,
                return_diagnostics=True,
                return_full_logits=True,
            )
            actor_logits = np.asarray(diagnostics.pop("actor_logits"), dtype=np.float32)
            masked_indices = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
            ordered = masked_indices[np.argsort(actor_logits[masked_indices])[::-1]]
            top1_logit = float(actor_logits[int(ordered[0])])
            top2_logit = float(actor_logits[int(ordered[1])]) if ordered.size > 1 else float("-inf")
            clearance = np.asarray(
                mask_info.get("depth_action_min_ray_clearance_m"), dtype=np.float32
            )
            finite_clearance = clearance[np.isfinite(clearance)]
            required_clearance = float(env.config.depth_mask_collision_radius_m) + float(
                env.config.depth_mask_slack_m
            )
            mask_boundary_margin = (
                float(np.min(np.abs(finite_clearance - required_clearance)))
                if finite_clearance.size else float("nan")
            )
            trace = observation_trace(obs, previous_action=previous_action)
            results.append({
                "state_sequence": int(pair["state_sequence"]),
                "state_id": int(state_metadata["state_id"]),
                "state_timestamp_ns": int(pair["state_timestamp_ns"]),
                "state_content_fingerprint": state_metadata["state_content_fingerprint"],
                "depth_sequence": int(pair["depth_sequence"]),
                "depth_timestamp_ns": int(pair["depth_timestamp_ns"]),
                "sensor_skew_ns": int(pair["sensor_skew_ns"]),
                "observation_fingerprint": trace["observation_fingerprint"],
                "depth_fingerprint": trace["depth_fingerprint"],
                "raw_depth_fingerprint": depth_metadata["depth_fingerprint"],
                "mask_fingerprint": array_fingerprint(mask),
                "logits_fingerprint": diagnostics["actor_logits_fingerprint"],
                "actor_top_k": diagnostics["actor_top_k"],
                "top1_action": int(action),
                "top1_logit": top1_logit,
                "top2_logit": top2_logit,
                "top1_top2_margin": top1_logit - top2_logit,
                "valid_action_count": int(np.count_nonzero(mask)),
                "mask_boundary_margin_m": mask_boundary_margin,
                "terminal_reward_input_fingerprint": terminal_reward_input_fingerprint(
                    obs,
                    mask,
                    episode_step=step,
                    previous_action=previous_action,
                ),
                "_mask": np.asarray(mask, dtype=np.bool_),
                "_logits": actor_logits,
                "_is_selected": bool(
                    int(pair["state_sequence"]) == selected_state_sequence
                    and int(pair["depth_sequence"]) == selected_depth_sequence
                ),
            })
        selected = [result for result in results if result["_is_selected"]]
        if len(selected) != 1:
            raise RuntimeError(
                "pairing audit boundary {} has {} selected pairs".format(step, len(selected))
            )
        baseline = selected[0]
        for result in results:
            result["logits_linf_from_selected"] = float(
                np.max(np.abs(result["_logits"] - baseline["_logits"]))
            )
            result["mask_hamming_from_selected"] = int(
                np.count_nonzero(result["_mask"] != baseline["_mask"])
            )
            result["top1_logit_delta_from_selected"] = abs(
                float(result["top1_logit"]) - float(baseline["top1_logit"])
            )
            result.pop("_mask")
            result.pop("_logits")
        unique = lambda key: len({str(result[key]) for result in results})
        categories = []
        if len(results) > 1 and unique("observation_fingerprint") == 1:
            categories.append("A")
        if unique("observation_fingerprint") > 1:
            categories.append("B")
        if unique("logits_fingerprint") > 1 and unique("top1_action") == 1:
            categories.append("C")
        if unique("mask_fingerprint") > 1 and unique("top1_action") == 1:
            categories.append("D")
        if unique("top1_action") > 1:
            categories.append("E")
        if unique("terminal_reward_input_fingerprint") > 1:
            categories.append("F")
        replay_steps.append({
            "step": int(step),
            "boundary_kind": str(boundary["boundary_kind"]),
            "policy_decision": bool(context.get("policy_decision", True)),
            "categories": categories,
            "pairs": results,
        })

    summary = summarize_counterfactual_steps(
        replay_steps, max_sensor_skew_ns=int(max_sensor_skew_ns)
    )
    summary.update({
        "total_captured_state_frames": len(states),
        "total_captured_depth_frames": len(depths),
        "captured_boundary_count": len(boundaries),
        "risk_rankings": risk_rankings(replay_steps, limit=10),
    })
    return {
        "contract_id": PAIRING_COUNTERFACTUAL_CONTRACT_ID,
        "summary": summary,
        "states": [{key: value for key, value in item.items() if key != "_record"} for item in states],
        "depths": [{key: value for key, value in item.items() if key != "_record"} for item in depths],
        "steps": replay_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--shadow-policy-checkpoint",
        default="",
        help="optional diagnostic-only second policy; it never controls Unity",
    )
    parser.add_argument(
        "--trace-jsonl",
        default="",
        help="write one diagnostic-only primary/shadow row per attempted primitive",
    )
    parser.add_argument(
        "--primary-policy-label",
        default="bc",
        help="label used to orient BC-to-AWAC divergence metrics",
    )
    parser.add_argument(
        "--shadow-policy-label",
        default="awac",
        help="label used to orient BC-to-AWAC divergence metrics",
    )
    parser.add_argument(
        "--normalizer-checkpoint",
        default="",
        help=(
            "explicit BC source for a legacy AWAC checkpoint that omitted "
            "feature normalization; its SHA must match source_bc_checkpoint_sha256"
        ),
    )
    parser.add_argument("--index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_PRIMITIVE_STEPS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tensorboard-log-dir", default="")
    parser.add_argument("--tensorboard-step", type=int, default=-1)
    parser.add_argument("--depth-width", type=int, default=160)
    parser.add_argument("--depth-height", type=int, default=90)
    parser.add_argument("--reset-settle", type=float, default=0.30)
    parser.add_argument("--reset-timeout", type=float, default=10.0)
    parser.add_argument("--worker-ready-timeout", type=float, default=30.0)
    parser.add_argument(
        "--reliable-v4",
        action="store_true",
        help="step through the reliable v4 command/result/snapshot path",
    )
    parser.add_argument("--reliable-v4-runtime-instance-id", default="")
    parser.add_argument("--reliable-v4-command-endpoint", default="")
    parser.add_argument("--reliable-v4-result-endpoint", default="")
    parser.add_argument("--reliable-v4-snapshot-endpoint", default="")
    parser.add_argument("--reliable-v4-timeout", type=float, default=10.0)
    parser.add_argument(
        "--expected-observation-contract",
        default=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        help=(
            "require checkpoint and runtime provenance to match this contract; "
            "formal evaluation defaults to reliable exact endpoint snapshots"
        ),
    )
    parser.add_argument("--post-wait", type=float, default=0.0)
    parser.add_argument("--safety-mask", choices=("inherit",) + SAFETY_MASKS, default="inherit")
    parser.add_argument("--execution-mode", choices=("inherit",) + EXECUTION_MODES, default="inherit")
    parser.add_argument("--audit-allow-runtime-override", action="store_true")
    parser.add_argument("--max-sensor-skew-ms", type=float, default=80.0)
    parser.add_argument(
        "--collision-cache",
        default=str(resolve_package_path("data/map_data/forest_voxels_10cm.npz")),
    )
    parser.add_argument("--collision-voxel-size", type=float, default=0.10)
    parser.add_argument("--collision-inflate-radius", type=float, default=0.35)
    parser.add_argument("--collision-check-step", type=int, default=1)
    parser.add_argument(
        "--depth-mask-collision-radius", type=float, default=None,
        help="inherit from checkpoint; an explicit mismatch requires audit override",
    )
    parser.add_argument(
        "--depth-mask-slack", type=float, default=None,
        help="inherit from checkpoint; an explicit mismatch requires audit override",
    )
    parser.add_argument(
        "--depth-mask-sample-stride", type=int, default=None,
        help="inherit from checkpoint; an explicit mismatch requires audit override",
    )
    parser.add_argument(
        "--depth-mask-max-patch-radius-px",
        type=int,
        default=None,
        help="inherit from checkpoint; an explicit mismatch requires audit override",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--only-checkpoint-val-episodes", action="store_true")
    parser.add_argument(
        "--episode-ids",
        default="",
        help="optional comma-separated episode ids; all requested ids must exist after other filters",
    )
    parser.add_argument(
        "--save-depth-audit",
        action="store_true",
        help="retain every pre-action raw depth frame, intrinsics, masks, and projection diagnostics",
    )
    parser.add_argument(
        "--depth-audit-patch-radii-px",
        default="14,28,70",
        help="comma-separated max patch radii evaluated offline without changing the executed mask",
    )
    parser.add_argument("--expected-planar-distance", type=float, default=MISSION_PLANAR_DISTANCE_M)
    parser.add_argument("--planar-distance-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--min-success-rate", type=float, default=0.60)
    parser.add_argument("--max-collision-rate", type=float, default=0.05)
    parser.add_argument("--max-dead-end-rate", type=float, default=0.25)
    parser.add_argument("--max-timeout-rate", type=float, default=0.05)
    parser.add_argument("--max-far-rate", type=float, default=0.0)
    parser.add_argument(
        "--audit-only", action="store_true",
        help="write diagnostic evaluation evidence without applying the deployment quality gate",
    )
    parser.add_argument(
        "--first-divergence-trace",
        action="store_true",
        help="capture read-only reset/primitive evidence for repeated-evaluation diagnosis",
    )
    parser.add_argument(
        "--command-tick-audit",
        action="store_true",
        help="audit every publish/state boundary of the first primitive; requires first-divergence trace",
    )
    parser.add_argument(
        "--execution-transport-audit",
        action="store_true",
        help=(
            "capture in-memory planning receipt ledgers and write them after each "
            "diagnostic episode; requires first-divergence trace"
        ),
    )
    parser.add_argument("--execution-transport-audit-run-id", default="")
    parser.add_argument(
        "--execution-transport-audit-unity-runtime-identity", default=""
    )
    parser.add_argument(
        "--admissible-pairing-audit",
        action="store_true",
        help=(
            "capture bounded neighboring sensor frames and replay every pair admitted by "
            "the existing synchronization contract after the episode"
        ),
    )
    parser.add_argument(
        "--pairing-candidate-count-audit",
        action="store_true",
        help=(
            "count state/depth pairs admitted by the current synchronization "
            "contract without policy replay or raw-depth persistence"
        ),
    )
    args = parser.parse_args()
    if bool(args.reliable_v4):
        missing_v4 = [
            name
            for name, value in (
                ("--reliable-v4-runtime-instance-id", args.reliable_v4_runtime_instance_id),
                ("--reliable-v4-command-endpoint", args.reliable_v4_command_endpoint),
                ("--reliable-v4-result-endpoint", args.reliable_v4_result_endpoint),
                ("--reliable-v4-snapshot-endpoint", args.reliable_v4_snapshot_endpoint),
            )
            if not str(value).strip()
        ]
        if missing_v4:
            raise ValueError("--reliable-v4 requires: {}".format(", ".join(missing_v4)))
        if float(args.reliable_v4_timeout) <= 0.0:
            raise ValueError("--reliable-v4-timeout must be positive")
    if int(args.tensorboard_step) < -1:
        raise ValueError("--tensorboard-step must be -1 or non-negative")
    if bool(args.first_divergence_trace) and not bool(args.audit_only):
        raise ValueError("--first-divergence-trace requires --audit-only")
    if bool(args.command_tick_audit) and not bool(args.first_divergence_trace):
        raise ValueError("--command-tick-audit requires --first-divergence-trace")
    if bool(args.execution_transport_audit) and not bool(args.first_divergence_trace):
        raise ValueError(
            "--execution-transport-audit requires --first-divergence-trace"
        )
    if bool(args.execution_transport_audit) and (
        not str(args.execution_transport_audit_run_id).strip()
        or not str(args.execution_transport_audit_unity_runtime_identity).strip()
    ):
        raise ValueError(
            "--execution-transport-audit requires run and Unity runtime identities"
        )
    if bool(args.admissible_pairing_audit) and not bool(args.audit_only):
        raise ValueError("--admissible-pairing-audit requires --audit-only")
    if bool(args.pairing_candidate_count_audit) and not bool(args.audit_only):
        raise ValueError("--pairing-candidate-count-audit requires --audit-only")
    if str(args.trace_jsonl).strip() and not str(args.shadow_policy_checkpoint).strip():
        raise ValueError("--trace-jsonl requires --shadow-policy-checkpoint")
    policy_action_mode, policy_temperature = resolve_policy_action_contract(
        args.temperature,
        audit_only=bool(args.audit_only),
    )
    args.temperature = policy_temperature

    torch, nn, _, _, _ = require_torch()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    index_path = Path(args.index).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    collision_cache_path = Path(args.collision_cache).expanduser()
    if not collision_cache_path.is_absolute():
        collision_cache_path = collision_cache_path.resolve()
    steps_dir = out_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    depth_audit_dir = out_dir / "depth_audit"
    depth_audit_index_path = out_dir / "depth_audit_index.csv"
    if bool(args.save_depth_audit):
        depth_audit_dir.mkdir(parents=True, exist_ok=True)
    shadow_trace_path = None
    if str(args.trace_jsonl).strip():
        shadow_trace_path = Path(args.trace_jsonl).expanduser()
        if not shadow_trace_path.is_absolute():
            shadow_trace_path = out_dir / shadow_trace_path
        shadow_trace_path = shadow_trace_path.resolve()
        shadow_trace_path.parent.mkdir(parents=True, exist_ok=True)

    evaluation_started = time.time()
    checkpoint = _load_torch_checkpoint(torch, checkpoint_path, device)
    if str(checkpoint.get("feature_contract_id", "")) != FEATURE_CONTRACT_ID:
        raise ValueError("checkpoint feature contract mismatch: {}".format(checkpoint.get("feature_contract_id")))
    validate_checkpoint_policy_input_contract(checkpoint)
    checkpoint = resolve_policy_checkpoint_task_contract(
        checkpoint,
        expected_max_primitive_steps=int(args.max_steps),
        path="policy checkpoint task contract",
    )
    checkpoint = normalize_policy_checkpoint_metadata(checkpoint)
    evaluation_observation_provenance = resolve_evaluation_observation_provenance(
        checkpoint,
        expected_observation_contract=str(args.expected_observation_contract),
        runtime_observation_contract=(
            EXACT_ENDPOINT_OBSERVATION_CONTRACT
            if bool(args.reliable_v4)
            else LEGACY_ASYNC_OBSERVATION_CONTRACT
        ),
        runtime_observation_source=(
            EXACT_ENDPOINT_OBSERVATION_CONTRACT
            if bool(args.reliable_v4)
            else LEGACY_ASYNC_OBSERVATION_CONTRACT
        ),
        reliable_execution_enabled=bool(args.reliable_v4),
        telemetry_observation_enabled=not bool(args.reliable_v4),
        telemetry_fallback_enabled=not bool(args.reliable_v4),
        state_depth_exact_endpoint_binding=bool(args.reliable_v4),
        allow_override=bool(args.audit_allow_runtime_override),
    )
    if int(checkpoint.get("vec_dim", -1)) != POLICY_VECTOR_DIM:
        raise ValueError("checkpoint vec_dim mismatch")
    if int(checkpoint.get("num_actions", -1)) != NUM_ACTIONS:
        raise ValueError("checkpoint num_actions mismatch")
    depth_history_frames = int(checkpoint.get("depth_history_frames", 1))
    if depth_history_frames <= 0:
        raise ValueError("checkpoint depth history is invalid")
    initial_prev_action = int(checkpoint.get("initial_prev_action", INITIAL_PREV_ACTION))
    if initial_prev_action != INITIAL_PREV_ACTION:
        raise ValueError("current feature contract requires initial_prev_action=-1")
    normalizer, normalizer_source_path = resolve_policy_checkpoint_normalizer(
        checkpoint,
        checkpoint_path=checkpoint_path,
        normalizer_checkpoint_path=args.normalizer_checkpoint,
        torch=torch,
        device=device,
        expected_max_primitive_steps=int(args.max_steps),
    )
    model_type = str(checkpoint.get("model_type", "bc_unknown"))
    state_key = "actor_state_dict" if "actor_state_dict" in checkpoint else "model_state_dict"
    if state_key not in checkpoint:
        raise KeyError("checkpoint contains neither actor_state_dict nor model_state_dict")
    offpolicy_algorithm = validate_policy_checkpoint_algorithm(
        checkpoint, state_key
    )
    algorithm_id = (
        offpolicy_algorithm.algorithm_id
        if offpolicy_algorithm is not None
        else "behavior_cloning"
    )
    model = build_model(nn, depth_channels=depth_history_frames).to(device)
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    shadow_checkpoint = None
    shadow_checkpoint_path = None
    shadow_normalizer = None
    shadow_model = None
    if str(args.shadow_policy_checkpoint).strip():
        shadow_checkpoint_path = Path(args.shadow_policy_checkpoint).expanduser().resolve()
        if not shadow_checkpoint_path.is_file():
            raise FileNotFoundError(
                "shadow policy checkpoint does not exist: {}".format(
                    shadow_checkpoint_path
                )
            )
        shadow_checkpoint = normalize_policy_checkpoint_metadata(
            _load_torch_checkpoint(torch, shadow_checkpoint_path, device)
        )
        if str(shadow_checkpoint.get("feature_contract_id", "")) != FEATURE_CONTRACT_ID:
            raise ValueError("shadow checkpoint feature contract mismatch")
        validate_checkpoint_policy_input_contract(shadow_checkpoint)
        shadow_checkpoint = resolve_policy_checkpoint_task_contract(
            shadow_checkpoint,
            expected_max_primitive_steps=int(args.max_steps),
            path="shadow policy checkpoint task contract",
        )
        shadow_provenance = resolve_evaluation_observation_provenance(
            shadow_checkpoint,
            expected_observation_contract=str(args.expected_observation_contract),
            runtime_observation_contract=(
                EXACT_ENDPOINT_OBSERVATION_CONTRACT
                if bool(args.reliable_v4)
                else LEGACY_ASYNC_OBSERVATION_CONTRACT
            ),
            runtime_observation_source=(
                EXACT_ENDPOINT_OBSERVATION_CONTRACT
                if bool(args.reliable_v4)
                else LEGACY_ASYNC_OBSERVATION_CONTRACT
            ),
            reliable_execution_enabled=bool(args.reliable_v4),
            telemetry_observation_enabled=not bool(args.reliable_v4),
            telemetry_fallback_enabled=not bool(args.reliable_v4),
            state_depth_exact_endpoint_binding=bool(args.reliable_v4),
            allow_override=bool(args.audit_allow_runtime_override),
        )
        if shadow_provenance != evaluation_observation_provenance:
            raise ValueError("shadow policy observation provenance mismatch")
        for field in (
            "task_contract_id",
            "task_contract_sha256",
            "mpl_contract_sha256",
            "vec_dim",
            "num_actions",
            "depth_history_frames",
            "initial_prev_action",
        ):
            if shadow_checkpoint.get(field) != checkpoint.get(field):
                raise ValueError("shadow policy {} mismatch".format(field))
        shadow_state_key = (
            "actor_state_dict"
            if "actor_state_dict" in shadow_checkpoint
            else "model_state_dict"
        )
        if shadow_state_key not in shadow_checkpoint:
            raise KeyError(
                "shadow checkpoint contains neither actor_state_dict nor model_state_dict"
            )
        shadow_normalizer, _ = resolve_policy_checkpoint_normalizer(
            shadow_checkpoint,
            checkpoint_path=shadow_checkpoint_path,
            normalizer_checkpoint_path=args.normalizer_checkpoint,
            torch=torch,
            device=device,
            expected_max_primitive_steps=int(args.max_steps),
        )
        shadow_depth_history_frames = int(
            shadow_checkpoint.get("depth_history_frames", depth_history_frames)
        )
        if shadow_depth_history_frames != depth_history_frames:
            raise ValueError("shadow policy depth history mismatch")
        shadow_model = build_model(nn, depth_channels=shadow_depth_history_frames).to(device)
        shadow_model.load_state_dict(shadow_checkpoint[shadow_state_key])
        shadow_model.eval()
    trace_critic1 = None
    trace_critic2 = None
    trace_critic_normalizer = None
    trace_critic_source_label = ""
    if str(args.trace_jsonl).strip():
        critic_candidates = [
            (checkpoint, normalizer, str(args.primary_policy_label)),
            (shadow_checkpoint, shadow_normalizer, str(args.shadow_policy_label)),
        ]
        for candidate_checkpoint, candidate_normalizer, candidate_label in critic_candidates:
            if candidate_checkpoint is None or candidate_normalizer is None:
                continue
            if not isinstance(candidate_checkpoint.get("critic1_state_dict"), Mapping):
                continue
            if not isinstance(candidate_checkpoint.get("critic2_state_dict"), Mapping):
                continue
            trace_critic1 = build_critic(
                nn, depth_channels=depth_history_frames
            ).to(device)
            trace_critic2 = build_critic(
                nn, depth_channels=depth_history_frames
            ).to(device)
            trace_critic1.load_state_dict(candidate_checkpoint["critic1_state_dict"], strict=True)
            trace_critic2.load_state_dict(candidate_checkpoint["critic2_state_dict"], strict=True)
            trace_critic1.eval()
            trace_critic2.eval()
            trace_critic_normalizer = candidate_normalizer
            trace_critic_source_label = str(candidate_label)
            break
    safety_mask, execution_mode, runtime_override = resolve_evaluation_runtime(
        checkpoint,
        requested_mask=str(args.safety_mask),
        requested_execution=str(args.execution_mode),
        allow_override=bool(args.audit_allow_runtime_override),
    )
    depth_mask_config, depth_mask_runtime_override = (
        resolve_evaluation_depth_mask_numeric_config(
            checkpoint,
            {
                "depth_mask_collision_radius_m": args.depth_mask_collision_radius,
                "depth_mask_slack_m": args.depth_mask_slack,
                "depth_mask_sample_stride": args.depth_mask_sample_stride,
                "depth_mask_max_patch_radius_px": args.depth_mask_max_patch_radius_px,
            },
            allow_override=bool(args.audit_allow_runtime_override),
        )
    )
    runtime_override = bool(
        runtime_override
        or depth_mask_runtime_override
        or evaluation_observation_provenance["runtime_contract_override"]
    )
    evaluation_provenance_summary = build_evaluation_provenance_summary(
        evaluation_observation_provenance,
        runtime_contract_override=runtime_override,
    )
    args.depth_mask_collision_radius = depth_mask_config[
        "depth_mask_collision_radius_m"
    ]
    args.depth_mask_slack = depth_mask_config["depth_mask_slack_m"]
    args.depth_mask_sample_stride = depth_mask_config[
        "depth_mask_sample_stride"
    ]
    args.depth_mask_max_patch_radius_px = depth_mask_config[
        "depth_mask_max_patch_radius_px"
    ]
    use_depth_collision_mask = safety_mask == "depth"
    use_global_collision_mask = safety_mask == "global"

    rows = [row for row in read_csv(index_path) if parse_bool(row.get("execute_ok", True))]
    validate_mission_rows(
        rows,
        args.expected_planar_distance,
        args.planar_distance_tolerance,
        expected_max_primitive_steps=int(args.max_steps),
    )
    if args.only_checkpoint_val_episodes:
        val_mission_ids = {str(value) for value in checkpoint.get("val_mission_ids", [])}
        if val_mission_ids:
            rows = [row for row in rows if mission_id_from_row(row) in val_mission_ids]
        else:
            val_ids = set(int(value) for value in checkpoint.get("val_episodes", []))
            rows = [
                row for row in rows
                if int(float(row.get("episode_id", -1))) in val_ids
            ]
    requested_episode_ids = parse_int_set(args.episode_ids)
    if requested_episode_ids:
        rows = [
            row for row in rows
            if int(float(row.get("episode_id", -1))) in requested_episode_ids
        ]
        found_episode_ids = {int(float(row.get("episode_id", -1))) for row in rows}
        missing_episode_ids = sorted(requested_episode_ids - found_episode_ids)
        if missing_episode_ids:
            raise ValueError("requested episode ids unavailable after filtering: {}".format(missing_episode_ids))
    rows.sort(key=lambda row: int(float(row.get("episode_id", 0))))
    if int(args.max_episodes) > 0:
        rows = rows[: int(args.max_episodes)]
    if not rows:
        raise RuntimeError("no episodes to evaluate")
    formal_quality_requested = bool(
        not args.audit_only
        and not requested_episode_ids
        and not runtime_override
        and policy_action_mode == "deterministic_argmax"
        and policy_temperature == 0.0
    )
    if formal_quality_requested and len(rows) < MIN_FORMAL_HOLDOUT_EPISODES:
        raise ValueError(
            "formal fixed-holdout evaluation requires at least {} episodes; "
            "got {} (use --audit-only for a smaller diagnostic run)".format(
                MIN_FORMAL_HOLDOUT_EPISODES, len(rows)
            )
        )

    audit_patch_radii = sorted(parse_int_set(args.depth_audit_patch_radii_px))
    if bool(args.save_depth_audit) and not audit_patch_radii:
        raise ValueError("--depth-audit-patch-radii-px must contain at least one integer")
    if any(value < 0 for value in audit_patch_radii):
        raise ValueError("depth audit patch radii must be non-negative")

    mpl = MotionPrimitiveLibrary()
    checkpoint_mpl_hash = str(checkpoint.get("mpl_contract_sha256", ""))
    if checkpoint_mpl_hash != str(mpl.contract_sha256):
        raise ValueError(
            "checkpoint motion-primitive contract {} != runtime {}".format(
                checkpoint_mpl_hash or "<missing>", mpl.contract_sha256
            )
        )
    reliable_v4_context = None
    reliable_v4_backend = None
    reliable_v4_snapshot_retriever = None
    if bool(args.reliable_v4):
        reliable_v4_context = zmq.Context()
        reliable_v4_snapshot_retriever = ZmqBridgeSnapshotRetriever(
            reliable_v4_context,
            args.reliable_v4_snapshot_endpoint,
            timeout_ms=max(1, int(float(args.reliable_v4_timeout) * 1000.0)),
        )
        reliable_v4_backend = ZmqReliableV4EnvironmentBackend(
            reliable_v4_context,
            runtime_instance_id=args.reliable_v4_runtime_instance_id,
            command_endpoint=args.reliable_v4_command_endpoint,
            result_endpoint=args.reliable_v4_result_endpoint,
            snapshot_provider=BridgeSnapshotEndpointProvider(reliable_v4_snapshot_retriever),
            timeout_s=float(args.reliable_v4_timeout),
        )
    env = UnityForestEnv(
        start=[0.0, 0.0, 2.0],
        goal=[40.0, 0.0, 2.0],
        mpl=mpl,
        config=EnvConfig(
            depth_out_width=int(args.depth_width),
            depth_out_height=int(args.depth_height),
            reset_timeout=float(args.reset_timeout),
            reset_settle_s=float(args.reset_settle),
            primitive_post_wait_s=float(args.post_wait),
            stop_at_primitive_end=execution_mode == "stop",
            enforce_sensor_sync=True,
            max_sensor_skew_s=max(0.0, float(args.max_sensor_skew_ms) * 1.0e-3),
            max_episode_steps=int(args.max_steps),
            goal_radius_xy=GOAL_RADIUS_XY_M,
            use_depth_collision_mask=use_depth_collision_mask,
            depth_mask_collision_radius_m=float(args.depth_mask_collision_radius),
            depth_mask_slack_m=float(args.depth_mask_slack),
            depth_mask_path_sample_stride=max(1, int(args.depth_mask_sample_stride)),
            depth_mask_max_patch_radius_px=max(0, int(args.depth_mask_max_patch_radius_px)),
            use_global_collision_mask=use_global_collision_mask,
            collision_cache_npz=str(collision_cache_path),
            collision_voxel_size=float(args.collision_voxel_size),
            collision_inflate_radius=float(args.collision_inflate_radius),
            collision_check_step=max(1, int(args.collision_check_step)),
            terminate_on_dead_end=True,
            terminate_on_invalid_action=True,
        ),
        reliable_v4_backend=reliable_v4_backend,
    )
    env.wait_until_ready(float(args.worker_ready_timeout))

    def write_execution_transport_audit(
        *,
        episode_id: int,
        metadata: Mapping[str, Any],
    ) -> None:
        if not bool(args.execution_transport_audit):
            return
        payload = env.end_execution_transport_audit()
        payload.update({
            "episode_id": int(episode_id),
            **dict(metadata),
        })
        (out_dir / "execution_transport_audit_episode_{:06d}.json".format(
            int(episode_id)
        )).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("POLICY_UNITY_EVAL_START")
    print("  checkpoint:", checkpoint_path)
    print("  algorithm_id:", algorithm_id)
    print("  model_type:", model_type)
    print("  normalizer_source:", normalizer_source_path)
    print("  reward_contract_id:", checkpoint.get("reward_contract_id", "<missing>"))
    print("  index:", index_path)
    print("  episodes:", len(rows))
    print("  max_steps:", args.max_steps)
    print("  safety_mask:", safety_mask)
    print("  execution_mode:", execution_mode)
    print("  reliable_v4:", bool(args.reliable_v4))
    print(
        "  observation_contract:",
        evaluation_observation_provenance["runtime_observation_contract"],
    )
    print(
        "  expected_observation_contract:",
        evaluation_observation_provenance["expected_observation_contract"],
    )
    print(
        "  runtime_contract_override:",
        runtime_override,
    )
    print("  policy_action_mode:", policy_action_mode)
    print("  policy_temperature:", policy_temperature)
    print("  shadow_policy_checkpoint:", shadow_checkpoint_path or "")
    print("  shadow_trace_jsonl:", shadow_trace_path or "")
    print("  shadow_controls_environment:", False)
    print("  runtime_contract_override:", runtime_override)
    print("  depth_mask_runtime_override:", depth_mask_runtime_override)
    print("  policy_input: depth_history + state + goal + previous_action")
    print("  policy_input_contract_sha256:", policy_input_contract_sha256())
    print("  privileged_runtime_inputs: []")
    print("  local_depth_collision_mask:", use_depth_collision_mask)
    print("  depth_mask_max_patch_radius_px:", int(args.depth_mask_max_patch_radius_px))
    print("  save_depth_audit:", bool(args.save_depth_audit))
    print("  admissible_pairing_audit:", bool(args.admissible_pairing_audit))
    print("  depth_audit_patch_radii_px:", audit_patch_radii)
    print("  global_collision_mask:", use_global_collision_mask)
    print("  collision_cache:", collision_cache_path)
    print("  depth_history_frames:", depth_history_frames)
    print("  initial_prev_action:", initial_prev_action)
    print("  mpl_duration_s:", mpl.duration_s)
    print("  mpl_forward_distance_m:", mpl.forward_distance_m)

    rollout_rows = []
    depth_audit_rows = []
    first_divergence_traces = []
    pairing_counterfactual_reports = []
    pairing_candidate_count_reports = []
    shadow_trace_stream = (
        shadow_trace_path.open("w", encoding="utf-8")
        if shadow_trace_path is not None
        else None
    )
    shadow_trace_enabled = shadow_model is not None and shadow_trace_stream is not None

    def capture_depth_audit(
        episode_id: int,
        step_index: int,
        obs: Dict,
        mask_info: Dict,
        combined_mask: np.ndarray,
        action: int,
    ):
        if not bool(args.save_depth_audit):
            return None
        raw_depth_m = np.asarray(obs["depth_m"], dtype=np.float32)
        intrinsics = dict(obs["depth_intrinsics"])
        height_mask = np.asarray(mask_info["height_mask"], dtype=np.bool_)
        counter_depth_masks = []
        counter_combined_masks = []
        counter_valid_counts = {}
        for radius in audit_patch_radii:
            counter_depth_mask, _ = local_depth_action_mask(
                mpl,
                raw_depth_m,
                depth_safety_config(env.config, radius),
                intrinsics=intrinsics,
            )
            counter_depth_masks.append(counter_depth_mask)
            counter_combined_masks.append(height_mask & counter_depth_mask)
            counter_valid_counts[str(radius)] = int(counter_depth_mask.sum())

        artifact = depth_audit_dir / "episode_{:06d}_step_{:03d}.npz".format(
            int(episode_id), int(step_index)
        )
        diagnostic_keys = (
            "depth_action_checked_sample_count",
            "depth_action_valid_patch_sample_count",
            "depth_action_capped_patch_sample_count",
            "depth_action_max_unclipped_patch_radius_px",
            "depth_action_mean_invalid_patch_fraction",
            "depth_action_min_ray_clearance_m",
        )
        payload = {
            "raw_depth_m": raw_depth_m,
            "intrinsics_fx_fy_cx_cy": np.asarray(
                [intrinsics[key] for key in ("fx", "fy", "cx", "cy")], dtype=np.float32
            ),
            "episode_id": np.asarray(int(episode_id), dtype=np.int64),
            "step": np.asarray(int(step_index), dtype=np.int64),
            "chosen_action": np.asarray(int(action), dtype=np.int64),
            "height_mask": height_mask,
            "executed_combined_mask": np.asarray(combined_mask, dtype=np.bool_),
            "executed_depth_mask": np.asarray(
                mask_info["depth_mask"] if mask_info.get("depth_mask") is not None else [], dtype=np.bool_
            ),
            "counterfactual_patch_radii_px": np.asarray(audit_patch_radii, dtype=np.int32),
            "counterfactual_depth_masks": np.stack(counter_depth_masks, axis=0),
            "counterfactual_combined_masks": np.stack(counter_combined_masks, axis=0),
        }
        for key in diagnostic_keys:
            values = mask_info.get(key)
            payload[key] = np.asarray(values if values is not None else [], dtype=np.float32)
        np.savez_compressed(str(artifact), **payload)

        record = {
            "episode_id": int(episode_id),
            "step": int(step_index),
            "chosen_action": int(action),
            "artifact": str(artifact.relative_to(out_dir)),
            "combined_valid_count": int(mask_info["combined_valid_count"]),
            "depth_valid_count": int(mask_info.get("depth_valid_count", -1)),
            "depth_frame_valid_fraction": float(mask_info.get("depth_frame_valid_fraction", np.nan)),
            "chosen_checked_samples": int(action_metric(mask_info, "depth_action_checked_sample_count", action, -1)),
            "chosen_valid_patch_samples": int(action_metric(mask_info, "depth_action_valid_patch_sample_count", action, -1)),
            "chosen_capped_patch_samples": int(action_metric(mask_info, "depth_action_capped_patch_sample_count", action, -1)),
            "chosen_max_unclipped_patch_radius_px": int(action_metric(mask_info, "depth_action_max_unclipped_patch_radius_px", action, -1)),
            "chosen_mean_invalid_patch_fraction": float(action_metric(mask_info, "depth_action_mean_invalid_patch_fraction", action, np.nan)),
            "chosen_min_ray_clearance_m": float(action_metric(mask_info, "depth_action_min_ray_clearance_m", action, np.nan)),
            "counterfactual_depth_valid_counts": json.dumps(counter_valid_counts, sort_keys=True),
            "success": False,
            "collision": False,
            "dead_end": False,
            "timeout": False,
            "far": False,
            "hard_altitude": False,
        }
        depth_audit_rows.append(record)
        return record

    try:
        for rank, row in enumerate(rows, start=1):
            episode_id = int(float(row.get("episode_id", rank - 1)))
            start, goal_list = row_start_goal(row)
            goal = np.asarray(goal_list, dtype=np.float32)
            if bool(args.execution_transport_audit):
                env.begin_execution_transport_audit(
                    run_id=str(args.execution_transport_audit_run_id),
                    unity_runtime_identity=str(
                        args.execution_transport_audit_unity_runtime_identity
                    ),
                )
            if bool(args.admissible_pairing_audit) or bool(args.pairing_candidate_count_audit):
                env.begin_pairing_candidate_audit(history_limit=64)
            obs = env.reset(start=start, goal=goal)
            previous_action = initial_prev_action
            depth_history = []
            pairing_decision_contexts = []
            step_rows = []
            success = collision = dead_end = timeout = far = hard_altitude = False
            stop_reason = "max_steps"
            episode_started = time.time()
            trajectory_z = [float(obs["state"]["z"])]
            episode_trace = {
                "contract_id": FIRST_DIVERGENCE_TRACE_CONTRACT_ID,
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "mission_index_sha256": file_sha256(index_path),
                "observation_provenance": dict(evaluation_provenance_summary),
                "mission_id": mission_id_from_row(row),
                "episode_id": episode_id,
                "start_target": [float(value) for value in start[:3]],
                "goal_target": [float(value) for value in goal[:3]],
                "reset": None,
                "primitives": [],
            } if bool(args.first_divergence_trace) else None

            for step_index in range(int(args.max_steps)):
                depth_stack = append_depth_history(depth_history, obs["depth"], depth_history_frames)
                if bool(args.admissible_pairing_audit):
                    pairing_decision_contexts.append({
                        "previous_action": int(previous_action),
                        "depth_stack": list(depth_stack),
                        "policy_decision": True,
                    })
                action_mask, mask_info = env.get_action_mask(obs, return_info=True)
                if int(np.count_nonzero(action_mask)) == 0:
                    audit_record = capture_depth_audit(
                        episode_id, step_index, obs, mask_info, action_mask, -1
                    )
                    if audit_record is not None:
                        audit_record["dead_end"] = True
                    dead_end = True
                    stop_reason = "dead_end"
                    env.stop()
                    break
                diagnostic_mode = bool(args.first_divergence_trace or shadow_trace_enabled)
                action_result = choose_action(
                    model,
                    obs,
                    previous_action,
                    action_mask,
                    normalizer,
                    torch,
                    device,
                    float(args.temperature),
                    depth_stack,
                    return_diagnostics=diagnostic_mode,
                    return_full_logits=diagnostic_mode,
                    return_full_probabilities=bool(shadow_trace_enabled),
                )
                if diagnostic_mode:
                    action, confidence, top5, actor_diagnostics = action_result
                else:
                    action, confidence, top5 = action_result
                    actor_diagnostics = None
                shadow_action = None
                shadow_diagnostics = None
                shadow_trace_record = None
                if shadow_trace_enabled:
                    # The shadow has the same observation, previous action and
                    # mask, but its result is never passed to Unity or the next
                    # primary decision.  Preserve RNG so even stochastic audit
                    # mode remains counterfactual.
                    with preserve_torch_rng(torch):
                        shadow_result = choose_action(
                            shadow_model,
                            obs,
                            previous_action,
                            action_mask,
                            shadow_normalizer,
                            torch,
                            device,
                            float(args.temperature),
                            depth_stack,
                            return_diagnostics=True,
                            return_full_logits=True,
                            return_full_probabilities=True,
                        )
                    shadow_action, _, _, shadow_diagnostics = shadow_result
                    critic_values = None
                    if trace_critic1 is not None and trace_critic2 is not None:
                        critic_values = critic_trace_values(
                            obs=obs,
                            previous_action=previous_action,
                            action_mask=action_mask,
                            depth_history=depth_stack,
                            primary_diagnostics=actor_diagnostics,
                            shadow_diagnostics=shadow_diagnostics,
                            primary_label=str(args.primary_policy_label),
                            shadow_label=str(args.shadow_policy_label),
                            critic1=trace_critic1,
                            critic2=trace_critic2,
                            normalizer=trace_critic_normalizer,
                            torch=torch,
                            device=device,
                        )
                    shadow_trace_record = build_shadow_trace_record(
                        mission_id=mission_id_from_row(row),
                        episode_id=episode_id,
                        step=step_index,
                        previous_action=previous_action,
                        valid_action_count=int(np.count_nonzero(action_mask)),
                        primary_action=action,
                        shadow_action=shadow_action,
                        primary_label=str(args.primary_policy_label),
                        shadow_label=str(args.shadow_policy_label),
                        primary_diagnostics=actor_diagnostics,
                        shadow_diagnostics=shadow_diagnostics,
                        observation_fingerprint=observation_trace(
                            obs, previous_action=previous_action
                        )["observation_fingerprint"],
                        action_mask_fingerprint=array_fingerprint(action_mask),
                        critic_values=critic_values,
                    )
                if episode_trace is not None and episode_trace["reset"] is None:
                    reset_trace = observation_trace(
                        obs, previous_action=previous_action
                    )
                    reset_trace.update({
                        "reset_position_error_m": float(
                            np.linalg.norm(
                                np.asarray(obs["state"]["position"], dtype=np.float32)
                                - np.asarray(start[:3], dtype=np.float32)
                            )
                        ),
                        "action_mask_fingerprint": array_fingerprint(action_mask),
                        "valid_action_count": int(np.count_nonzero(action_mask)),
                    })
                    episode_trace["reset"] = reset_trace
                audit_record = capture_depth_audit(
                    episode_id, step_index, obs, mask_info, action_mask, action
                )
                position_before = np.asarray(obs["state"]["position"], dtype=np.float32).copy()
                distance_before = float(obs["goal"]["distance_xy"])
                try:
                    obs_next, reward, done, info = env.step_primitive(
                        action,
                        obs_before=obs,
                        precomputed_mask=action_mask,
                        precomputed_mask_info=mask_info,
                        audit_command_frames=bool(
                            args.command_tick_audit and step_index == 0
                        ),
                    )
                except PrimitiveExecutionAbortedError as error:
                    execution_result = error.execution_result
                    if execution_result is None:
                        # Test and diagnostic callers can construct the typed
                        # exception directly.  A live UnityForestEnv always
                        # supplies this result after validating submitted IDs.
                        execution_result = classify_primitive_execution_result(
                            error.audit_receipt,
                            expected_frame_count=int(
                                error.audit_receipt["requested_frame_count"]
                            ),
                        )
                        error.execution_result = execution_result
                    if execution_result.get("kind") == (
                        PRIMITIVE_EXECUTION_RESULT_TERMINAL_ABORT
                    ):
                        obs_next, reward, done, info = env.materialize_terminal_abort(
                            error,
                            action_id=action,
                            obs_before=obs,
                            precomputed_mask=action_mask,
                            precomputed_mask_info=mask_info,
                        )
                    else:
                        if episode_trace is not None:
                            episode_trace["execution_failure"] = {
                                "step": int(step_index),
                                "before": observation_trace(
                                    obs, previous_action=previous_action
                                ),
                                "action_mask_fingerprint": array_fingerprint(action_mask),
                                "valid_action_count": int(np.count_nonzero(action_mask)),
                                "selected_action": int(action),
                                **actor_diagnostics,
                                "receipt": error.audit_receipt,
                            }
                            (out_dir / "first_divergence_trace.partial.json").write_text(
                                json.dumps(
                                    {
                                        "contract_id": FIRST_DIVERGENCE_TRACE_CONTRACT_ID,
                                        "observation_provenance": dict(
                                            evaluation_provenance_summary
                                        ),
                                        "episodes": [episode_trace],
                                    },
                                    indent=2,
                                    sort_keys=True,
                                ),
                                encoding="utf-8",
                            )
                        raise
                except PrimitiveExecutionContractError as error:
                    failure = {
                        "failure_kind": "PrimitiveExecutionContractError",
                        "error": str(error),
                        "step": int(step_index),
                        "selected_action": int(action),
                        "previous_action": int(previous_action),
                        "before": observation_trace(
                            obs, previous_action=previous_action
                        ),
                        "action_mask_fingerprint": array_fingerprint(action_mask),
                        "valid_action_count": int(np.count_nonzero(action_mask)),
                        **actor_diagnostics,
                    }
                    if episode_trace is not None:
                        episode_trace["execution_failure"] = dict(failure)
                        (out_dir / "first_divergence_trace.partial.json").write_text(
                            json.dumps(
                                {
                                    "contract_id": FIRST_DIVERGENCE_TRACE_CONTRACT_ID,
                                    "observation_provenance": dict(
                                        evaluation_provenance_summary
                                    ),
                                    "episodes": [episode_trace],
                                },
                                indent=2,
                                sort_keys=True,
                            ),
                            encoding="utf-8",
                        )
                    write_execution_transport_audit(
                        episode_id=episode_id,
                        metadata=failure,
                    )
                    raise
                if episode_trace is not None:
                    primitive_trace = {
                        "step": int(step_index),
                        "previous_action": int(previous_action),
                        "before": observation_trace(
                            obs, previous_action=previous_action
                        ),
                        "action_mask_fingerprint": array_fingerprint(action_mask),
                        "action_mask": np.asarray(action_mask, dtype=np.bool_).astype(bool).tolist(),
                        "valid_action_count": int(np.count_nonzero(action_mask)),
                        "top1_top2_margin": float(
                            actor_diagnostics["actor_top_k"][0]["logit"]
                            - actor_diagnostics["actor_top_k"][1]["logit"]
                        ),
                        "depth_mask_boundary_margin_m": audit_mask_boundary_margin_m(
                            env, mask_info
                        ),
                        "minimum_safety_clearance_m": float(
                            obs["safety"].get("min_clearance", np.nan)
                        ),
                        **actor_diagnostics,
                        "selected_action": int(action),
                        "after": observation_trace(
                            obs_next, previous_action=action
                        ),
                        "primitive_endpoint_error_body_m": float(
                            info.get("primitive_endpoint_error_body_m", np.nan)
                        ),
                        "done": bool(done),
                        "done_reason": str(info.get("done_reason", "")),
                        "terminal_reward_input_fingerprint": terminal_reward_input_fingerprint(
                            obs_next,
                            np.asarray(obs_next["action_mask"], dtype=np.bool_),
                            episode_step=step_index + 1,
                            previous_action=action,
                        ),
                    }
                    if "command_frame_audit" in info:
                        primitive_trace["command_frame_audit"] = info[
                            "command_frame_audit"
                        ]
                    if "primitive_execution" in info:
                        primitive_trace["primitive_execution"] = info[
                            "primitive_execution"
                        ]
                    if "primitive_execution_result" in info:
                        primitive_trace["primitive_execution_result"] = info[
                            "primitive_execution_result"
                        ]
                    episode_trace["primitives"].append(primitive_trace)
                if audit_record is not None:
                    audit_record["success"] = bool(info.get("success", False))
                    audit_record["collision"] = bool(info.get("collided", False))
                    audit_record["dead_end"] = bool(info.get("dead_end", False))
                    audit_record["timeout"] = bool(info.get("timeout", False))
                    audit_record["far"] = bool(info.get("far", False))
                    audit_record["hard_altitude"] = bool(
                        info.get("hard_altitude_violation", False)
                    )
                if shadow_trace_record is not None:
                    shadow_trace_record.update({
                        "distance_before": float(distance_before),
                        "distance_after": float(obs_next["goal"]["distance_xy"]),
                        "progress_m": float(
                            distance_before - float(obs_next["goal"]["distance_xy"])
                        ),
                        "reward": float(reward),
                        "done": bool(done),
                        "done_reason": str(info.get("done_reason", "")),
                        "minimum_safety_clearance_before_m": float(
                            obs["safety"].get("min_clearance", np.nan)
                        ),
                        "minimum_safety_clearance_after_m": float(
                            obs_next["safety"].get("min_clearance", np.nan)
                        ),
                        "success": bool(info.get("success", False)),
                        "collision": bool(info.get("collided", False)),
                        "dead_end": bool(info.get("dead_end", False)),
                        "timeout": bool(info.get("timeout", False)),
                        "hard_altitude": bool(
                            info.get("hard_altitude_violation", False)
                        ),
                        "terminal_abort": bool(info.get("terminal_abort", False)),
                    })
                    shadow_trace_stream.write(
                        json.dumps(shadow_trace_record, sort_keys=True) + "\n"
                    )
                    shadow_trace_stream.flush()
                if bool(args.save_depth_audit) and bool(info.get("dead_end", False)):
                    next_mask_info = info.get("next_action_mask_info", {})
                    terminal_mask = np.asarray(
                        obs_next.get("action_mask", np.zeros((mpl.num_actions,), dtype=np.bool_)),
                        dtype=np.bool_,
                    )
                    terminal_audit_record = capture_depth_audit(
                        episode_id,
                        step_index + 1,
                        obs_next,
                        next_mask_info,
                        terminal_mask,
                        -1,
                    )
                    if terminal_audit_record is not None:
                        terminal_audit_record["dead_end"] = True
                step_rows.append(
                    {
                        "episode_id": episode_id,
                        "step": step_index,
                        "observation_contract": evaluation_provenance_summary[
                            "observation_contract"
                        ],
                        "observation_source": evaluation_provenance_summary[
                            "observation_source"
                        ],
                        "checkpoint_observation_contract": evaluation_provenance_summary[
                            "checkpoint_observation_contract"
                        ],
                        "checkpoint_observation_source": evaluation_provenance_summary[
                            "checkpoint_observation_source"
                        ],
                        "prev_action": previous_action,
                        "action": action,
                        "confidence": confidence,
                        "top5": " ".join(str(value) for value in top5),
                        "reward": float(reward),
                        "distance_before": distance_before,
                        "distance_after": float(obs_next["goal"]["distance_xy"]),
                        "x_before": float(position_before[0]),
                        "y_before": float(position_before[1]),
                        "z_before": float(position_before[2]),
                        "x_after": float(obs_next["state"]["position"][0]),
                        "y_after": float(obs_next["state"]["position"][1]),
                        "z_after": float(obs_next["state"]["position"][2]),
                        "valid_action_count": int(mask_info["combined_valid_count"]),
                        "height_valid_action_count": int(np.asarray(mask_info["height_mask"], dtype=np.bool_).sum()),
                        "depth_valid_action_count": int(mask_info.get("depth_valid_count", -1)),
                        "depth_blocked_action_count": int(mask_info.get("depth_blocked_count", -1)),
                        "global_valid_action_count": int(mask_info.get("global_valid_count", -1)),
                        "chosen_action_depth_valid": bool(mask_info["depth_mask"][action]) if mask_info.get("depth_mask") is not None else "",
                        "chosen_action_global_valid": bool(mask_info["global_mask"][action]) if mask_info.get("global_mask") is not None else "",
                        "next_valid_action_count": int(info.get("next_action_mask_info", {}).get("combined_valid_count", -1)),
                        "next_depth_valid_action_count": int(info.get("next_action_mask_info", {}).get("depth_valid_count", -1)),
                        "next_global_valid_action_count": int(info.get("next_action_mask_info", {}).get("global_valid_count", -1)),
                        "endpoint_error_body_m": float(info.get("primitive_endpoint_error_body_m", np.nan)),
                        "primitive_completed": bool(info.get("primitive_completed", True)),
                        "terminal_abort": bool(info.get("terminal_abort", False)),
                        "terminal_abort_reason": str(info.get("terminal_reason", "")),
                        "effective_integration_ticks": int(
                            info.get("effective_integration_ticks", -1)
                        ),
                        "applied_frame_count": int(info.get("applied_frame_count", -1)),
                        "success": bool(info.get("success", False)),
                        "collision": bool(info.get("collided", False)),
                        "dead_end": bool(info.get("dead_end", False)),
                        "timeout": bool(info.get("timeout", False)),
                        "far": bool(info.get("far", False)),
                        "hard_altitude": bool(
                            info.get("hard_altitude_violation", False)
                        ),
                    }
                )
                previous_action = int(action)
                obs = obs_next
                trajectory_z.append(float(obs["state"]["z"]))
                success = bool(info.get("success", False))
                collision = bool(info.get("collided", False))
                dead_end = bool(info.get("dead_end", False))
                timeout = bool(info.get("timeout", False))
                far = bool(info.get("far", False))
                hard_altitude = bool(info.get("hard_altitude_violation", False))
                if done:
                    stop_reason = str(info.get("done_reason", "")).strip()
                    if not stop_reason:
                        stop_reason = terminal_done_reason(
                            success=success,
                            collision=collision,
                            dead_end=dead_end,
                            timeout=timeout,
                            far=far,
                            hard_altitude=hard_altitude,
                        )
                    break
            if not any(
                (success, collision, dead_end, timeout, far, hard_altitude)
            ):
                timeout = True
                stop_reason = "timeout"
            env.stop()
            write_execution_transport_audit(
                episode_id=episode_id,
                metadata={
                    "failure_kind": "",
                    "stop_reason": str(stop_reason),
                },
            )
            if bool(args.admissible_pairing_audit):
                pairing_capture = env.finish_pairing_candidate_audit()
                if len(pairing_capture["boundaries"]) == len(pairing_decision_contexts) + 1:
                    terminal_depth_history = append_depth_history(
                        list(depth_history), obs["depth"], depth_history_frames
                    )
                    pairing_decision_contexts.append({
                        "previous_action": int(previous_action),
                        "depth_stack": list(terminal_depth_history),
                        "policy_decision": False,
                    })
                pairing_report = replay_pairing_counterfactuals(
                    env=env,
                    capture=pairing_capture,
                    decision_contexts=pairing_decision_contexts,
                    model=model,
                    normalizer=normalizer,
                    torch=torch,
                    device=device,
                    max_sensor_skew_ns=int(round(env.config.max_sensor_skew_s * 1.0e9)),
                )
                pairing_report.update({
                    "episode_id": int(episode_id),
                    "mission_id": mission_id_from_row(row),
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "mission_index_sha256": file_sha256(index_path),
                    "observation_provenance": dict(evaluation_provenance_summary),
                    "max_sensor_skew_ns": int(round(env.config.max_sensor_skew_s * 1.0e9)),
                })
                raw_depth_artifact = out_dir / "pairing_candidates_episode_{:06d}.npz".format(
                    int(episode_id)
                )
                raw_depth_payload = {}
                for depth_record in pairing_capture["depths"]:
                    message = depth_record["message"]
                    sequence = int(depth_record["depth_sequence"])
                    raw_depth_payload["depth_{:08d}_bytes".format(sequence)] = np.frombuffer(
                        bytes(message.data), dtype=np.uint8
                    )
                    raw_depth_payload["depth_{:08d}_shape_step".format(sequence)] = np.asarray(
                        [message.height, message.width, message.step, message.is_bigendian],
                        dtype=np.int64,
                    )
                    raw_depth_payload["depth_{:08d}_timestamp_ns".format(sequence)] = np.asarray(
                        UnityForestEnv._stamp_ns(message), dtype=np.int64
                    )
                np.savez_compressed(str(raw_depth_artifact), **raw_depth_payload)
                pairing_report["raw_depth_artifact"] = str(
                    raw_depth_artifact.relative_to(out_dir)
                )
                pairing_report_path = out_dir / "pairing_counterfactual_episode_{:06d}.json".format(
                    int(episode_id)
                )
                pairing_report_path.write_text(
                    json.dumps(pairing_report, indent=2, sort_keys=True), encoding="utf-8"
                )
                pairing_counterfactual_reports.append({
                    "episode_id": int(episode_id),
                    "report": str(pairing_report_path.relative_to(out_dir)),
                    **pairing_report["summary"],
                })
            elif bool(args.pairing_candidate_count_audit):
                pairing_capture = env.finish_pairing_candidate_audit()
                pairing_candidate_count_reports.append({
                    "episode_id": int(episode_id),
                    "mission_id": mission_id_from_row(row),
                    "boundaries": compact_pairing_candidate_counts(
                        capture=pairing_capture,
                        max_sensor_skew_ns=int(
                            round(env.config.max_sensor_skew_s * 1.0e9)
                        ),
                    ),
                })
            final_position = np.asarray(obs["state"]["position"], dtype=np.float32)
            final_distance = float(np.linalg.norm(final_position[:2] - goal[:2]))
            final_abs_goal_dz = abs(float(final_position[2] - goal[2]))
            step_path = steps_dir / "episode_{:06d}.csv".format(episode_id)
            if step_rows:
                write_csv_atomic(step_path, step_rows, fieldnames=list(step_rows[0].keys()))
            rollout = {
                **task_contract_fields(int(args.max_steps)),
                "episode_id": episode_id,
                "mission_id": mission_id_from_row(row),
                "observation_contract": evaluation_provenance_summary[
                    "observation_contract"
                ],
                "observation_source": evaluation_provenance_summary[
                    "observation_source"
                ],
                "checkpoint_observation_contract": evaluation_provenance_summary[
                    "checkpoint_observation_contract"
                ],
                "checkpoint_observation_source": evaluation_provenance_summary[
                    "checkpoint_observation_source"
                ],
                "steps": len(step_rows),
                "success": success,
                "collision": collision,
                "dead_end": dead_end,
                "timeout": timeout,
                "far": far,
                "hard_altitude": hard_altitude,
                "stop_reason": stop_reason,
                "return": float(
                    sum(float(step["reward"]) for step in step_rows)
                ),
                "final_distance_xy": final_distance,
                "final_abs_goal_dz": final_abs_goal_dz,
                "min_z": min(trajectory_z),
                "max_z": max(trajectory_z),
                "final_x": float(final_position[0]),
                "final_y": float(final_position[1]),
                "final_z": float(final_position[2]),
                "elapsed_s": time.time() - episode_started,
                "step_csv": os.path.relpath(str(step_path), str(out_dir)),
            }
            rollout_rows.append(rollout)
            if episode_trace is not None:
                episode_trace["outcome"] = stop_reason
                episode_trace["steps"] = len(step_rows)
                first_divergence_traces.append(episode_trace)
            if depth_audit_rows:
                write_csv_atomic(
                    depth_audit_index_path,
                    depth_audit_rows,
                    fieldnames=list(depth_audit_rows[0].keys()),
                )
            print_progress(
                component="policy_unity_evaluation",
                processed=rank,
                passing=sum(bool(item["success"]) for item in rollout_rows),
                estimated_stop=len(rows),
                elapsed_s=time.time() - evaluation_started,
                episode=episode_id,
                outcome=stop_reason,
                steps=len(step_rows),
                final_distance="{:.3f}".format(final_distance),
            )
    finally:
        env.stop()
        if shadow_trace_stream is not None:
            shadow_trace_stream.close()
        if reliable_v4_backend is not None:
            reliable_v4_backend.close()
        if reliable_v4_snapshot_retriever is not None:
            reliable_v4_snapshot_retriever.close()
        if reliable_v4_context is not None:
            reliable_v4_context.term()

    evaluation_observation_provenance = resolve_evaluation_observation_provenance(
        checkpoint,
        expected_observation_contract=str(args.expected_observation_contract),
        runtime_observation_contract=(
            EXACT_ENDPOINT_OBSERVATION_CONTRACT
            if bool(args.reliable_v4)
            else LEGACY_ASYNC_OBSERVATION_CONTRACT
        ),
        runtime_observation_source=(
            EXACT_ENDPOINT_OBSERVATION_CONTRACT
            if bool(args.reliable_v4)
            else LEGACY_ASYNC_OBSERVATION_CONTRACT
        ),
        reliable_execution_enabled=bool(args.reliable_v4),
        telemetry_observation_enabled=not bool(args.reliable_v4),
        telemetry_fallback_enabled=not bool(args.reliable_v4),
        state_depth_exact_endpoint_binding=bool(args.reliable_v4),
        snapshot_missing_count=max(
            int(getattr(env, "snapshot_missing_count", 0)),
            int(getattr(reliable_v4_backend, "snapshot_missing_count", 0)),
        ),
        telemetry_lookup_count=int(getattr(env, "telemetry_lookup_count", 0)),
        allow_override=bool(args.audit_allow_runtime_override),
    )
    evaluation_provenance_summary = build_evaluation_provenance_summary(
        evaluation_observation_provenance,
        runtime_contract_override=bool(
            runtime_override
            or evaluation_observation_provenance["runtime_contract_override"]
        ),
    )
    for trace in first_divergence_traces:
        trace["observation_provenance"] = dict(evaluation_provenance_summary)

    rollout_path = out_dir / "rollout_index.csv"
    write_csv_atomic(rollout_path, rollout_rows, fieldnames=list(rollout_rows[0].keys()))
    if bool(args.first_divergence_trace):
        trace_path = out_dir / "first_divergence_trace.json"
        trace_path.write_text(
            json.dumps(
                {
                    "contract_id": FIRST_DIVERGENCE_TRACE_CONTRACT_ID,
                    "observation_provenance": dict(evaluation_provenance_summary),
                    "episodes": first_divergence_traces,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    if pairing_candidate_count_reports:
        pairing_count_path = out_dir / "pairing_candidate_counts.json"
        pairing_count_path.write_text(
            json.dumps(pairing_candidate_count_reports, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if depth_audit_rows:
        write_csv_atomic(
            depth_audit_index_path,
            depth_audit_rows,
            fieldnames=list(depth_audit_rows[0].keys()),
        )
    episodes = len(rollout_rows)
    outcome_summary = summarize_policy_outcomes(rollout_rows)
    episode_returns = np.asarray(
        [float(row["return"]) for row in rollout_rows], dtype=np.float64
    )
    tensorboard_step = (
        int(args.tensorboard_step)
        if int(args.tensorboard_step) >= 0
        else int(checkpoint.get("global_step", 0))
    )
    summary = {
        "evaluation_contract_id": POLICY_UNITY_EVALUATION_CONTRACT_ID,
        "episodes": episodes,
        "max_steps": int(args.max_steps),
        **outcome_summary,
        "episode_return_mean": float(np.mean(episode_returns)),
        "episode_return_median": float(np.median(episode_returns)),
        "episode_return_std": float(np.std(episode_returns)),
        "episode_steps_mean": float(
            np.mean([float(row["steps"]) for row in rollout_rows])
        ),
        "final_distance_mean": float(np.mean([row["final_distance_xy"] for row in rollout_rows])),
        "final_abs_goal_dz_mean": float(np.mean([row["final_abs_goal_dz"] for row in rollout_rows])),
        "observed_min_z": float(min(row["min_z"] for row in rollout_rows)),
        "observed_max_z": float(max(row["max_z"] for row in rollout_rows)),
        "policy_runtime_contract_id": POLICY_RUNTIME_CONTRACT_ID,
        "safety_mask": safety_mask,
        "execution_mode": execution_mode,
        "reliable_v4": bool(args.reliable_v4),
        **evaluation_provenance_summary,
        "reliable_v4_runtime_instance_id": (
            str(args.reliable_v4_runtime_instance_id)
            if bool(args.reliable_v4) else ""
        ),
        "policy_action_mode": policy_action_mode,
        "policy_temperature": policy_temperature,
        "depth_mask_runtime_override": depth_mask_runtime_override,
        "depth_mask_config": {
            "collision_radius_m": depth_mask_config[
                "depth_mask_collision_radius_m"
            ],
            "slack_m": depth_mask_config["depth_mask_slack_m"],
            "sample_stride": depth_mask_config["depth_mask_sample_stride"],
            "max_patch_radius_px": depth_mask_config[
                "depth_mask_max_patch_radius_px"
            ],
        },
        "checkpoint_depth_mask_numeric_config": (
            checkpoint_depth_mask_numeric_config(checkpoint)
        ),
        "local_depth_collision_mask": use_depth_collision_mask,
        "depth_mask_collision_radius_m": float(args.depth_mask_collision_radius),
        "depth_mask_slack_m": float(args.depth_mask_slack),
        "depth_mask_sample_stride": int(args.depth_mask_sample_stride),
        "depth_mask_max_patch_radius_px": int(args.depth_mask_max_patch_radius_px),
        "depth_audit_saved": bool(args.save_depth_audit),
        "depth_audit_patch_radii_px": audit_patch_radii,
        "depth_audit_index": depth_audit_index_path.name if depth_audit_rows else "",
        "first_divergence_trace": (
            "first_divergence_trace.json" if bool(args.first_divergence_trace) else ""
        ),
        "admissible_pairing_audit": bool(args.admissible_pairing_audit),
        "pairing_counterfactual_reports": pairing_counterfactual_reports,
        "pairing_candidate_count_audit": bool(args.pairing_candidate_count_audit),
        "pairing_candidate_count_report": (
            "pairing_candidate_counts.json" if pairing_candidate_count_reports else ""
        ),
        "episode_filter_ids": sorted(requested_episode_ids),
        "quality_gate_applicable": (
            formal_quality_requested
            and episodes >= MIN_FORMAL_HOLDOUT_EPISODES
        ),
        "minimum_formal_holdout_episodes": MIN_FORMAL_HOLDOUT_EPISODES,
        "global_collision_mask": use_global_collision_mask,
        **task_contract_fields(int(args.max_steps)),
        "max_steps": int(args.max_steps),
        "checkpoint": os.path.relpath(str(checkpoint_path), str(out_dir)),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "normalizer_checkpoint": os.path.relpath(
            str(normalizer_source_path), str(out_dir)
        ),
        "normalizer_checkpoint_sha256": file_sha256(normalizer_source_path),
        "mission_index_sha256": file_sha256(index_path),
        "algorithm_id": algorithm_id,
        "model_type": model_type,
        "rollout_csv": rollout_path.name,
        "runtime_observation_manifest": "runtime_observation_manifest.json",
        "tensorboard_log_dir": str(args.tensorboard_log_dir),
        "tensorboard_step": tensorboard_step,
        "shadow_policy_checkpoint": (
            str(shadow_checkpoint_path) if shadow_checkpoint_path is not None else ""
        ),
        "shadow_policy_checkpoint_sha256": (
            file_sha256(shadow_checkpoint_path)
            if shadow_checkpoint_path is not None
            else ""
        ),
        "shadow_trace_jsonl": (
            str(shadow_trace_path) if shadow_trace_path is not None else ""
        ),
        "shadow_trace_enabled": bool(shadow_trace_enabled),
        "shadow_trace_critic_source_label": str(trace_critic_source_label),
        "shadow_trace_critic_enabled": bool(
            trace_critic1 is not None and trace_critic2 is not None
        ),
        "shadow_controls_environment": False,
        "shadow_controls_primary_action": False,
        "shadow_controls_seed": False,
        "primary_policy_label": str(args.primary_policy_label),
        "shadow_policy_label": str(args.shadow_policy_label),
        "quality_thresholds": {
            "min_success_rate": float(args.min_success_rate),
            "max_collision_rate": float(args.max_collision_rate),
            "max_dead_end_rate": float(args.max_dead_end_rate),
            "max_timeout_rate": float(args.max_timeout_rate),
            "max_far_rate": float(args.max_far_rate),
        },
    }
    quality_pass = bool(
        summary["quality_gate_applicable"]
        and summary["outcome_partition_valid"]
        and summary["success_rate"] >= float(args.min_success_rate)
        and summary["collision_rate"] <= float(args.max_collision_rate)
        and summary["dead_end_rate"] <= float(args.max_dead_end_rate)
        and summary["timeout_rate"] <= float(args.max_timeout_rate)
        and summary["far_rate"] <= float(args.max_far_rate)
        and summary["altitude_violation_count"] == 0
    )
    summary["quality_pass"] = quality_pass
    (out_dir / "runtime_observation_manifest.json").write_text(
        json.dumps(
            {
                "schema": "policy_unity_evaluation_observation_provenance_v1",
                "evaluation_contract_id": POLICY_UNITY_EVALUATION_CONTRACT_ID,
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "mission_index_sha256": summary["mission_index_sha256"],
                **evaluation_provenance_summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if args.tensorboard_log_dir:
        from planning.awac.tensorboard import AWACTensorBoardLogger

        logger = AWACTensorBoardLogger(
            Path(args.tensorboard_log_dir).expanduser().resolve()
        )
        try:
            logger.log_dev100(
                summary,
                global_step=tensorboard_step,
                bc_baseline={
                    "success_rate": 0.69,
                    "collision_rate": 0.09,
                    "dead_end_rate": 0.22,
                },
            )
        finally:
            logger.close()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("POLICY_UNITY_EVAL_SUMMARY")
    for key, value in summary.items():
        print("  {}: {}".format(key, value))
    if not bool(summary["quality_gate_applicable"]):
        print("RESULT=AUDIT_COMPLETE")
        return 0
    print("RESULT={}".format("PASS" if quality_pass else "FAIL"))
    return 0 if quality_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
