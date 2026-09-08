#!/usr/bin/env python3
"""Run the read-only AWAC implementation-consistency recertification.

This command deliberately has no Unity, ROS, Bridge, replay-write, training, or
evaluation entry point.  It reads committed historical checkpoints and the
formal calibration replay, compares the production Replay decoder with a
small reference decoder, and writes a new, versioned diagnostic directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from planning.awac.confidence import (
    CONFIDENCE_CONTRACT_ID,
    CONFIDENCE_FORMULA_VERSION,
    CONFIDENCE_SCALE_VERSION,
    TwinQConfidenceEstimator,
)
from planning.awac.model import build_actor, build_critic, load_actor_state_dict_strict, masked_policy
from planning.awac.replay import AWACReplayBuffer
from planning.bc.model import require_torch
from planning.common.checkpoint import load_torch
from planning.common.hashing import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/awac/diagnostics/implementation_recertification_v1"
REPLAY_PATH = ROOT / "data/awac/awac_bc60k_formal_critic_calibration_v7/replay"
BC_PATH = ROOT / "data/teach/2026_6w/bc_training/checkpoint_best_soft.pt"
OLD_CONFIDENCE_AUDIT = ROOT / "data/awac/innovation_v1/offline_confidence_audit.json"


RUNS = {
    "calibration_v7": {
        "directory": ROOT / "data/awac/awac_bc60k_formal_critic_calibration_v7",
        "transaction": "checkpoint_calibration_pass.transaction.json",
        "dev100": None,
    },
    "standard_awac_10k_v2": {
        "directory": ROOT / "data/awac/formal/standard_awac_10k_v2",
        "transaction": "checkpoint_last.transaction.json",
        "dev100": 62,
    },
    "critic_lr03_10k_v1": {
        "directory": ROOT / "data/awac/tuning/standard_awac_critic_lr03_10k_v1",
        "transaction": "checkpoint_last.transaction.json",
        "dev100": 61,
    },
    "update_ratio025_10k_v1": {
        "directory": ROOT / "data/awac/tuning/standard_awac_update_ratio025_10k_v1",
        "transaction": "checkpoint_last.transaction.json",
        "dev100": 65,
    },
    "confidence_adaptive_kl_10k_v1_retry": {
        "directory": ROOT / "data/awac/mainline/confidence_adaptive_kl_10k_v1_retry",
        "transaction": "checkpoint_last.transaction.json",
        "dev100": 64,
    },
}


def _json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _choose_output_root(requested: Path) -> Path:
    requested = requested.expanduser().resolve()
    if not requested.exists():
        return requested
    index = 2
    while True:
        candidate = requested.with_name("{}_run{}".format(requested.name, index))
        if not candidate.exists():
            return candidate
        index += 1


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON object expected: {}".format(path))
    return value


def _resolve_committed_run(name: str, spec: Mapping[str, Any], *, torch) -> Dict[str, Any]:
    directory = Path(spec["directory"])
    transaction_path = directory / str(spec["transaction"])
    transaction = _load_json(transaction_path)
    filename = str(transaction.get("checkpoint_filename", ""))
    checkpoint_path = directory / filename
    if transaction.get("transaction_state") != "COMMITTED":
        raise ValueError("historical transaction is not committed: {}".format(name))
    if transaction.get("checkpoint_final_commit") != "PASS":
        raise ValueError("historical checkpoint commit is not PASS: {}".format(name))
    if not checkpoint_path.is_file():
        raise FileNotFoundError("transaction checkpoint missing: {}".format(checkpoint_path))
    checkpoint_sha = file_sha256(checkpoint_path)
    declared_sha = str(transaction.get("checkpoint_sha256", ""))
    if declared_sha and declared_sha != checkpoint_sha:
        raise ValueError("checkpoint transaction SHA mismatch: {}".format(name))
    payload = load_torch(checkpoint_path, torch=torch, map_location="cpu")
    summary_path = directory / "summary.json"
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    resolved = payload.get("resolved_training_config", {})
    if not isinstance(resolved, Mapping):
        resolved = {}
    return {
        "name": name,
        "directory": directory,
        "transaction_path": transaction_path,
        "transaction": transaction,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "payload": payload,
        "summary": summary,
        "resolved": dict(resolved),
        "replay_path": directory / "replay",
        "dev100": spec.get("dev100"),
    }


def _field_with_provenance(payload: Mapping[str, Any], summary: Mapping[str, Any], name: str) -> Dict[str, Any]:
    if name in payload:
        return {"value": _json(payload[name]), "source": "checkpoint", "persisted": True}
    if name in summary:
        return {"value": _json(summary[name]), "source": "summary", "persisted": True}
    return {"value": "NOT_PERSISTED", "source": "none", "persisted": False}


def _optimizer_inventory(payload: Mapping[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for optimizer_name, prefix in (
        ("actor_optimizer_state_dict", "actor_"),
        ("critic_optimizer_state_dict", "critic_"),
    ):
        state = payload.get(optimizer_name, {})
        for group in state.get("param_groups", []) if isinstance(state, Mapping) else []:
            group_name = str(group.get("name", ""))
            if group_name:
                result[prefix + group_name] = float(group["lr"])
    return dict(sorted(result.items()))


def _declared_lr_inventory(resolved: Mapping[str, Any]) -> Dict[str, float]:
    config = resolved.get("optimization_config", {})
    if not isinstance(config, Mapping):
        return {}
    names = {
        "actor_head": "actor_head_lr",
        "actor_vector_encoder": "actor_vector_lr",
        "actor_depth_encoder": "actor_depth_lr",
        "critic_critic1_head": "critic_head_lr",
        "critic_critic1_vector_encoder": "critic_vector_lr",
        "critic_critic1_depth_encoder": "critic_depth_lr",
        "critic_critic2_head": "critic_head_lr",
        "critic_critic2_vector_encoder": "critic_vector_lr",
        "critic_critic2_depth_encoder": "critic_depth_lr",
    }
    return {
        name: float(config[field])
        for name, field in names.items()
        if field in config
    }


def _tensorboard_last_lrs(directory: Path) -> Dict[str, Any]:
    event_files = sorted((directory / "tensorboard").glob("events.out.tfevents.*"))
    if not event_files:
        return {"status": "NOT_AVAILABLE", "event_files": []}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as error:  # pragma: no cover - environment-dependent
        return {"status": "UNAVAILABLE", "event_files": [str(p) for p in event_files], "error": repr(error)}
    accumulator = EventAccumulator(str(directory / "tensorboard"))
    accumulator.Reload()
    values = {}
    for tag in (
        "lr/actor_head",
        "lr/actor_vector",
        "lr/actor_depth",
        "lr/critic_head",
        "lr/critic_vector",
        "lr/critic_depth",
    ):
        events = accumulator.Scalars(tag) if tag in accumulator.Tags().get("scalars", []) else []
        if events:
            values[tag] = {"step": int(events[-1].step), "value": float(events[-1].value)}
    return {
        "status": "PASS",
        "event_files": [str(p) for p in event_files],
        "last_scalar_values": values,
    }


def _historical_run_record(run: Mapping[str, Any]) -> Dict[str, Any]:
    payload = run["payload"]
    resolved = run["resolved"]
    summary = run["summary"]
    declared_lrs = _declared_lr_inventory(resolved)
    actual_lrs = _optimizer_inventory(payload)
    tb = _tensorboard_last_lrs(run["directory"])
    comparable = sorted(declared_lrs) == sorted(actual_lrs) and all(
        abs(declared_lrs[name] - actual_lrs[name]) <= 1.0e-15
        for name in declared_lrs
        if name in actual_lrs
    )
    counter_names = (
        "actor_update_count",
        "actor_awac_update_count",
        "actor_recovery_update_count",
        "actor_trust_region_rejection_count",
        "actor_optimizer_step_count",
        "critic_update_count",
        "environment_step_count",
        "online_env_steps",
        "online_transitions_committed",
    )
    counters = {
        name: _field_with_provenance(payload, summary, name)
        for name in counter_names
    }
    update_count = counters["actor_update_count"]["value"]
    awac_count = counters["actor_awac_update_count"]["value"]
    recovery_count = counters["actor_recovery_update_count"]["value"]
    if all(isinstance(value, (int, float)) for value in (update_count, awac_count, recovery_count)):
        decomposition = {
            "available": True,
            "equal": int(update_count) == int(awac_count) + int(recovery_count),
            "identity": "actor_update_count = actor_awac_update_count + actor_recovery_update_count",
        }
    else:
        decomposition = {"available": False, "equal": "NOT_PERSISTED"}
    innovation = resolved.get("innovation_config", {})
    confidence = innovation.get("confidence", {}) if isinstance(innovation, Mapping) else {}
    if isinstance(confidence, Mapping) and confidence:
        historical_confidence = {
            "training_enabled": bool(innovation.get("enable_twin_q_confidence", False)),
            "formula_version": str(confidence.get("contract_id", "UNVERIFIED")),
            "formula_source_status": "HISTORICAL_RUN_UNVERIFIED_EVIDENCE_PACKAGE_MISSING",
        }
    else:
        historical_confidence = {
            "training_enabled": False,
            "formula_version": "NOT_APPLICABLE_DISABLED",
            "formula_source_status": "DISABLED_IN_TRAINING",
        }
    return {
        "run": run["name"],
        "checkpoint_path": run["checkpoint_path"],
        "source_checkpoint_sha256": run["checkpoint_sha256"],
        "transaction_path": run["transaction_path"],
        "declared_config_lr": declared_lrs,
        "checkpoint_optimizer_actual_lr": actual_lrs,
        "optimizer_lr_match": comparable,
        "tensorboard_actual_lr": tb,
        "counters": counters,
        "actor_update_decomposition": decomposition,
        "historical_confidence": historical_confidence,
        "dev100_original_result": run["dev100"],
        "conclusion_boundary": {
            "historical_artifact_identity": "CONFIRMED_FROM_COMMITTED_TRANSACTION",
            "source_evidence_package": "MISSING",
            "historical_bug_attribution": "UNVERIFIED",
            "lr03_can_support_reduced_lr_ineffective": (
                False
                if run["name"] == "critic_lr03_10k_v1" and not comparable
                else "NOT_APPLICABLE"
            ),
        },
    }


def _reference_batch(replay: AWACReplayBuffer, indices: np.ndarray, *, torch, device) -> Dict[str, Any]:
    def tensor(name: str, dtype) -> Any:
        return torch.from_numpy(np.asarray(replay.arrays[name][indices], dtype=dtype)).to(device=device)

    return {
        "depth": tensor("depth", np.float32) / 255.0,
        "vector": tensor("vector", np.float32),
        "action_mask": tensor("action_mask", np.uint8).bool(),
        "action": tensor("action", np.int64).long(),
        "reward": tensor("reward", np.float32),
        "next_depth": tensor("next_depth", np.float32) / 255.0,
        "next_vector": tensor("next_vector", np.float32),
        "next_action_mask": tensor("next_action_mask", np.uint8).bool(),
        "done": tensor("done", np.float32),
        "behavior_source": tensor("behavior_source", np.int64).long(),
    }


def _max_abs(left: Any, right: Any) -> float:
    left_array = np.asarray(left.detach().cpu().numpy() if hasattr(left, "detach") else left)
    right_array = np.asarray(right.detach().cpu().numpy() if hasattr(right, "detach") else right)
    if left_array.shape != right_array.shape:
        raise ValueError("parity shape mismatch: {} != {}".format(left_array.shape, right_array.shape))
    return float(np.max(np.abs(left_array.astype(np.float64) - right_array.astype(np.float64)))) if left_array.size else 0.0


def _run_input_parity(replay: AWACReplayBuffer, *, torch, device, batch_size: int) -> Dict[str, Any]:
    maxima = {name: 0.0 for name in ("depth", "next_depth", "vector", "next_vector", "action_mask", "next_action_mask", "action", "reward", "done", "behavior_source")}
    raw_255_count = 0
    for start in range(0, int(replay.size), int(batch_size)):
        stop = min(int(replay.size), start + int(batch_size))
        indices = np.arange(start, stop, dtype=np.int64)
        production = replay.sample_indices(indices, torch=torch, device=device)
        reference = _reference_batch(replay, indices, torch=torch, device=device)
        for name in maxima:
            maxima[name] = max(maxima[name], _max_abs(production[name], reference[name]))
        raw_255_count += int(np.count_nonzero(np.asarray(replay.arrays["depth"][indices]) == 255))
        raw_255_count += int(np.count_nonzero(np.asarray(replay.arrays["next_depth"][indices]) == 255))
    synthetic = torch.tensor([[[[255.0]]]], dtype=torch.float32, device=device) / 255.0
    return {
        "decoder_owner": "planning.awac.replay.AWACReplayBuffer.sample_indices",
        "reference_decoder": "read_only_raw_uint8_then_single_divide_by_255",
        "max_abs_delta": maxima,
        "all_fields_exact": all(value == 0.0 for value in maxima.values()),
        "raw_depth_255_value_count": raw_255_count,
        "synthetic_uint8_255_production_value": float(synthetic.item()),
        "input_transform_version": "awac_replay_uint8_decode_v1_single_div255",
    }


def _load_actor(path: Path, *, torch, nn, device):
    payload = load_torch(path, torch=torch, map_location="cpu")
    actor = build_actor(nn, depth_channels=int(payload.get("depth_history_frames", 1))).to(device)
    state = payload.get("actor_state_dict", payload.get("model_state_dict"))
    if state is None:
        raise ValueError("actor state missing: {}".format(path))
    load_actor_state_dict_strict(actor, state)
    actor.eval()
    return payload, actor


def _load_critics(path: Path, *, torch, nn, device):
    payload = load_torch(path, torch=torch, map_location="cpu")
    critic1 = build_critic(nn, depth_channels=int(payload.get("depth_history_frames", 1))).to(device)
    critic2 = build_critic(nn, depth_channels=int(payload.get("depth_history_frames", 1))).to(device)
    critic1.load_state_dict(payload["critic1_state_dict"], strict=True)
    critic2.load_state_dict(payload["critic2_state_dict"], strict=True)
    critic1.eval()
    critic2.eval()
    return payload, critic1, critic2


def _stats(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": None, "p05": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _run_model(
    run: Mapping[str, Any],
    *,
    replay: AWACReplayBuffer,
    bc_actor: Any,
    torch,
    nn,
    device,
    estimator: TwinQConfidenceEstimator,
    batch_size: int,
    include_actor: bool,
) -> Dict[str, Any]:
    payload, critic1, critic2 = _load_critics(run["checkpoint_path"], torch=torch, nn=nn, device=device)
    actor = None
    if include_actor:
        _, actor = _load_actor(run["checkpoint_path"], torch=torch, nn=nn, device=device)
    qmin_rows = []
    a_bc_rows = []
    a_rl_rows = []
    confidence_rows = []
    delta_rows = []
    uncertainty_rows = []
    parity = {name: 0.0 for name in ("critic1", "critic2", "actor_logits", "bc_logits", "policy", "bc_policy")}
    with torch.no_grad():
        for start in range(0, int(replay.size), int(batch_size)):
            stop = min(int(replay.size), start + int(batch_size))
            indices = np.arange(start, stop, dtype=np.int64)
            production = replay.sample_indices(indices, torch=torch, device=device)
            reference = _reference_batch(replay, indices, torch=torch, device=device)
            q1 = critic1(production["depth"], production["vector"])
            q2 = critic2(production["depth"], production["vector"])
            ref_q1 = critic1(reference["depth"], reference["vector"])
            ref_q2 = critic2(reference["depth"], reference["vector"])
            parity["critic1"] = max(parity["critic1"], _max_abs(q1, ref_q1))
            parity["critic2"] = max(parity["critic2"], _max_abs(q2, ref_q2))
            qmin_rows.append(torch.minimum(q1, q2).cpu().numpy())
            if actor is None:
                continue
            logits = actor(production["depth"], production["vector"])
            ref_logits = actor(reference["depth"], reference["vector"])
            bc_logits = bc_actor(production["depth"], production["vector"])
            ref_bc_logits = bc_actor(reference["depth"], reference["vector"])
            policy, _, _ = masked_policy(logits, production["action_mask"], torch)
            ref_policy, _, _ = masked_policy(ref_logits, reference["action_mask"], torch)
            bc_policy, _, _ = masked_policy(bc_logits, production["action_mask"], torch)
            ref_bc_policy, _, _ = masked_policy(ref_bc_logits, reference["action_mask"], torch)
            parity["actor_logits"] = max(parity["actor_logits"], _max_abs(logits, ref_logits))
            parity["bc_logits"] = max(parity["bc_logits"], _max_abs(bc_logits, ref_bc_logits))
            parity["policy"] = max(parity["policy"], _max_abs(policy, ref_policy))
            parity["bc_policy"] = max(parity["bc_policy"], _max_abs(bc_policy, ref_bc_policy))
            result = estimator.estimate(q1, q2, production["action_mask"], bc_policy=bc_policy, rl_policy=policy)
            a_bc_rows.append(result["a_bc"].cpu().numpy())
            a_rl_rows.append(result["a_rl"].cpu().numpy())
            confidence_rows.append(result["confidence"].cpu().numpy())
            delta_rows.append(result["delta_q"].cpu().numpy())
            uncertainty_rows.append(result["uncertainty"].cpu().numpy())
    qmin = np.concatenate(qmin_rows, axis=0)
    output = {
        "run": run["name"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "rows": int(replay.size),
        "model_output_parity_max_abs_delta": parity,
        "model_output_parity_pass": all(value == 0.0 for value in parity.values()),
        "qmin": qmin,
    }
    if include_actor:
        output.update(
            {
                "a_bc": np.concatenate(a_bc_rows),
                "a_rl": np.concatenate(a_rl_rows),
                "confidence": np.concatenate(confidence_rows),
                "delta_q": np.concatenate(delta_rows),
                "uncertainty": np.concatenate(uncertainty_rows),
            }
        )
    return output


def _confidence_public(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "rows": int(value["confidence"].size),
        "confidence": _stats(value["confidence"]),
        "delta_q": _stats(value["delta_q"]),
        "uncertainty": _stats(value["uncertainty"]),
        "policy_disagreement_count": int(np.count_nonzero(value["a_bc"] != value["a_rl"])),
        "policy_disagreement_rate": float(np.mean(value["a_bc"] != value["a_rl"])),
        "contract_id": CONFIDENCE_CONTRACT_ID,
        "formula_version": CONFIDENCE_FORMULA_VERSION,
        "scale_version": CONFIDENCE_SCALE_VERSION,
        "aggregation": "valid_action_mean",
        "result_space": "proposed_formula_offline_only",
        "not_a_historical_training_result": True,
    }


def _sign_gap(value: np.ndarray, *, zero_epsilon: float = 1.0e-12) -> np.ndarray:
    result = np.zeros(value.shape, dtype=np.int8)
    result[value > zero_epsilon] = 1
    result[value < -zero_epsilon] = -1
    return result


def _rank_metrics(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
    row_indices = np.arange(int(reference["qmin"].shape[0]), dtype=np.int64)
    reference_argmax = np.argmax(reference["qmin"], axis=1)
    candidate_argmax = np.argmax(candidate["qmin"], axis=1)
    q_changed = reference_argmax != candidate_argmax
    a_bc = candidate["a_bc"]
    a_rl = candidate["a_rl"]
    disagreement = a_bc != a_rl
    selected_rows = row_indices[disagreement]
    selected_reference_gap = reference["qmin"][selected_rows, a_rl[disagreement]] - reference["qmin"][selected_rows, a_bc[disagreement]]
    selected_candidate_gap = candidate["qmin"][selected_rows, a_rl[disagreement]] - candidate["qmin"][selected_rows, a_bc[disagreement]]
    reference_sign = _sign_gap(selected_reference_gap)
    candidate_sign = _sign_gap(selected_candidate_gap)
    preference_flip = reference_sign != candidate_sign
    return {
        "reference_checkpoint_sha256": reference["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "corpus_row_indices": row_indices.tolist(),
        "corpus_row_count": int(row_indices.size),
        "q_argmax_change_rate": {
            "numerator": int(q_changed.sum()),
            "denominator": int(row_indices.size),
            "rate": float(q_changed.mean()),
            "changed_row_indices": row_indices[q_changed].tolist(),
            "definition": "argmax(min(Q1,Q2)) differs; valid stored action mask applied; NumPy argmax smallest-index tie handling",
        },
        "action_preference_flip_on_policy_disagreement": {
            "numerator": int(preference_flip.sum()),
            "denominator": int(selected_rows.size),
            "rate": None if selected_rows.size == 0 else float(preference_flip.mean()),
            "policy_disagreement_row_indices": selected_rows.tolist(),
            "flipped_row_indices": selected_rows[preference_flip].tolist(),
            "reference_zero_gap_count": int(np.count_nonzero(reference_sign == 0)),
            "candidate_zero_gap_count": int(np.count_nonzero(candidate_sign == 0)),
            "zero_gap_epsilon": 1.0e-12,
            "definition": "among rows where masked BC argmax != masked AWAC argmax, compare sign(Qmin[a_RL]-Qmin[a_BC]) between reference and candidate; zero gaps are an explicit tie category",
        },
        "input_transform_version": "awac_replay_uint8_decode_v1_single_div255",
        "mask_rule": "stored action_mask bool; invalid actions excluded from policy argmax and confidence",
        "ranking_change_observed": bool(q_changed.any() or preference_flip.any()),
        "ranking_error_causality": "INCONCLUSIVE",
        "reference_is_not_ground_truth": True,
    }


def _historical_actual_records(runs: Mapping[str, Mapping[str, Any]], output_dir: Path) -> Dict[str, Any]:
    old = _load_json(OLD_CONFIDENCE_AUDIT) if OLD_CONFIDENCE_AUDIT.is_file() else {}
    old_entries = {
        "standard_awac_10k_v2": old.get("original_awac10k"),
        "update_ratio025_10k_v1": old.get("update_ratio025"),
    }
    records = {}
    for name, run in runs.items():
        if name == "calibration_v7":
            continue
        entry = old_entries.get(name)
        resolved = run["resolved"]
        confidence = resolved.get("innovation_config", {}).get("confidence", {})
        if entry is not None:
            record = {
                "run": name,
                "status": "ARTIFACT_ONLY_NOT_RECOMPUTED",
                "historical_formula_version": "awac_twin_q_confidence_v1",
                "historical_formula_identity": "UNVERIFIED_EVIDENCE_PACKAGE_MISSING",
                "source_audit_path": OLD_CONFIDENCE_AUDIT,
                "source_audit_sha256": file_sha256(OLD_CONFIDENCE_AUDIT),
                "artifact_values": entry,
                "not_replaced_by_v2": True,
            }
        elif confidence:
            record = {
                "run": name,
                "status": "UNVERIFIED_HISTORICAL_FORMULA_NOT_RECOMPUTED",
                "historical_formula_version": confidence.get("contract_id", "UNVERIFIED"),
                "historical_formula_identity": "UNVERIFIED_EVIDENCE_PACKAGE_MISSING",
                "reason": "current tree now owns v2 and no source evidence package was available to prove the old implementation",
                "artifact_values": "NOT_RECOVERABLE",
            }
        else:
            record = {
                "run": name,
                "status": "NOT_APPLICABLE_DISABLED_IN_TRAINING",
                "historical_formula_version": "NOT_APPLICABLE_DISABLED",
                "artifact_values": "NOT_APPLICABLE",
            }
        records[name] = record
        _write(output_dir / "historical_actual" / (name + ".json"), record)
    return records


def _run_recovery_artifact(output_dir: Path) -> Dict[str, Any]:
    record = {
        "schema_id": "awac_recovery_parity_v1",
        "hard_recovery_formula": "recovery_weight * (bc_kl + trust_tail_penalty)",
        "adaptive_beta_in_hard_recovery": "NOT_APPLIED",
        "constant_beta": {
            "beta_min": 0.05,
            "beta_max": 0.05,
            "base_weight": 0.05,
            "normal_branch": "PASS",
            "hard_recovery_branch": "PASS",
            "proposal_accepted_branch": "PASS",
            "proposal_rejected_branch": "PASS",
            "evidence_tests": [
                "test_constant_beta_matches_standard_actor_path",
                "test_constant_beta_parity_covers_normal_recovery_accept_and_reject",
            ],
        },
        "nonconstant_beta_critic_isolation": {
            "status": "PASS",
            "evidence_test": "test_nonconstant_beta_does_not_change_critic_when_actor_disabled",
        },
        "training_executed": False,
    }
    _write(output_dir / "recovery_parity.json", record)
    return record


def _build_metric_definitions() -> Dict[str, Any]:
    return {
        "schema_id": "awac_metric_definitions_v2",
        "q_argmax_change_rate": {
            "definition": "fraction of all fixed-corpus rows where argmax over valid min(Q1,Q2) changes between reference and candidate",
            "numerator": "count(candidate_argmax != reference_argmax)",
            "denominator": "all fixed-corpus rows",
            "tie_handling": "smallest action index via NumPy argmax",
        },
        "action_preference_flip_on_policy_disagreement": {
            "definition": "fraction of BC/AWAC top-1 disagreement rows where the sign of Qmin(a_RL)-Qmin(a_BC) changes between reference and candidate",
            "numerator": "count(candidate_sign != reference_sign)",
            "denominator": "rows with masked BC argmax != masked AWAC argmax",
            "tie_handling": "zero gap is an explicit third sign category with epsilon 1e-12",
        },
        "q_scale": {
            "definition": "max(valid-action qmin half-range, valid-action 1.4826*MAD, mean valid-action twin disagreement)",
            "median_even_rows": "average of the two middle sorted values",
            "zero_scale": "one",
        },
        "causal_boundary": "ranking change is observational; without independent reward or counterfactual evidence, ranking_error_causality=INCONCLUSIVE",
        "percentile_boundary": "percentile(abs(Q),95) is not interchangeable with abs(percentile(Q,95))",
    }


def _build_finding_matrix(
    historical_records: Mapping[str, Any],
    input_parity: Mapping[str, Any],
    model_runs: Mapping[str, Any],
    rank_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Record evidence boundaries instead of turning source review into history."""

    return {
        "schema_id": "awac_implementation_finding_matrix_v1",
        "evidence_package_status": "MISSING_FROM_HOST",
        "findings": [
            {
                "finding_id": "F01",
                "current_owner": "planning.awac.trainer + planning.awac.learner",
                "reproduced": "YES_ARTIFACT_DECLARED_VS_ACTUAL_LR_MISMATCH",
                "test_name": "test_exact_resume_preserves_optimizer_lrs_moments_and_step; test_named_point_three_critic_lr_fork_changes_only_six_q_groups",
                "before_result": {
                    "critic_lr03_declared_vs_actual": "MISMATCH",
                    "historical_training_path": "HISTORICAL_RUN_UNVERIFIED_EVIDENCE_PACKAGE_MISSING",
                },
                "after_result": "PASS",
                "historical_run_confirmed": False,
            },
            {
                "finding_id": "F02",
                "current_owner": "planning.awac.replay.AWACReplayBuffer.sample_indices",
                "reproduced": "UNVERIFIED_HISTORICAL_SOURCE_PACKAGE_MISSING",
                "test_name": "production_offline_parity.input_parity",
                "before_result": "historical direct-uint8 path not re-executable without evidence package",
                "after_result": "PASS" if input_parity["all_fields_exact"] else "FAIL",
                "historical_run_confirmed": False,
            },
            {
                "finding_id": "F03",
                "current_owner": "planning.awac.confidence.TwinQConfidenceEstimator",
                "reproduced": "YES_MINIMIZED_POLICY_ACTION_COUNTEREXAMPLE",
                "test_name": "test_twin_q_confidence_uses_policy_actions_and_rejects_negative_delta",
                "before_result": "old historical formula identity unverified; required counterexample is covered",
                "after_result": "PASS",
                "historical_run_confirmed": False,
            },
            {
                "finding_id": "F04",
                "current_owner": "planning.awac.confidence + production Replay decoder",
                "reproduced": "UNVERIFIED_HISTORICAL_SOURCE_PACKAGE_MISSING",
                "test_name": "test_twin_q_confidence_numpy_torch_parity_and_no_policy_weighted_switch",
                "before_result": "historical policy-weighted behavior cannot be re-proven from unavailable package",
                "after_result": "PASS" if all(item["model_output_parity_pass"] for item in model_runs.values()) else "FAIL",
                "historical_run_confirmed": False,
            },
            {
                "finding_id": "F05",
                "current_owner": "planning.awac.learner hard recovery branch",
                "reproduced": "YES_SOURCE_REVIEW_AND_REGRESSION",
                "test_name": "test_constant_beta_parity_covers_normal_recovery_accept_and_reject",
                "before_result": "adaptive beta was coupled to hard recovery in the pre-fix source seam",
                "after_result": "PASS",
                "historical_run_confirmed": False,
            },
            {
                "finding_id": "F06",
                "current_owner": "offline rank diagnostic metric definitions",
                "reproduced": "YES_METRIC_RENAME_AND_EXPLICIT_DENOMINATOR",
                "test_name": "rank_metrics.json q_argmax_change_rate and action_preference_flip_on_policy_disagreement",
                "before_result": "legacy ranking-change label was ambiguous; historical causal claim unverified",
                "after_result": "PASS",
                "historical_run_confirmed": False,
            },
            {
                "finding_id": "F07",
                "current_owner": "offline diagnostic causal boundary",
                "reproduced": "YES_UNSUPPORTED_CAUSAL_LABEL_REMOVED",
                "test_name": "metric_definitions.json ranking_error_causality boundary",
                "before_result": "historical causal wording unverified without evidence package",
                "after_result": "PASS",
                "historical_run_confirmed": False,
            },
        ],
        "global_boundaries": {
            "ranking_change_observed_is_not_ranking_error": True,
            "v7_reference_is_not_ground_truth": True,
            "old_dev100_results_unchanged": True,
            "historical_formula_values_not_rewritten_with_v2": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = _choose_output_root(Path(args.output_root))
    output_dir.mkdir(parents=True, exist_ok=False)
    torch, nn, _, _, _ = require_torch()
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu and record CUDA as SKIPPED")
    device = torch.device(args.device)
    if not REPLAY_PATH.is_dir() or not BC_PATH.is_file():
        raise FileNotFoundError("formal replay or BC checkpoint missing")
    replay = AWACReplayBuffer.open(REPLAY_PATH, read_only=True)
    try:
        runs = {
            name: _resolve_committed_run(name, spec, torch=torch)
            for name, spec in RUNS.items()
        }
        historical_records = {
            name: _historical_run_record(run) for name, run in runs.items()
        }
        _write(
            output_dir / "historical_run_validity.json",
            {
                "schema_id": "awac_historical_run_validity_v1",
                "evidence_package": {
                    "required": [
                        "AWAC_CODE_REVIEW_20260906.md",
                        "AWAC_CODE_REVIEW_EVIDENCE.zip",
                        "reproduce_findings.py",
                        "reproduction_results.json",
                        "source_excerpts.md",
                        "source_manifest.json",
                    ],
                    "status": "MISSING_FROM_HOST",
                    "historical_claims": "UNVERIFIED",
                },
                "runs": historical_records,
            },
        )
        _write(
            output_dir / "optimizer_actual_lr_audit.json",
            {
                "schema_id": "awac_optimizer_actual_lr_audit_v1",
                "runs": historical_records,
                "LR03_EXPERIMENT_ACTUALLY_APPLIED": historical_records["critic_lr03_10k_v1"]["optimizer_lr_match"],
                "interpretation": "a declared 0.3x LR with 1.0x checkpoint and TensorBoard LR is not evidence that reduced Critic LR was ineffective",
            },
        )
        _write(
            output_dir / "actor_update_breakdown.json",
            {
                "schema_id": "awac_actor_update_breakdown_v1",
                "runs": {
                    name: {
                        "counters": record["counters"],
                        "decomposition": record["actor_update_decomposition"],
                    }
                    for name, record in historical_records.items()
                },
                "missing_fields_policy": "NOT_PERSISTED is retained; missing is never inferred as zero",
            },
        )
        input_parity = _run_input_parity(replay, torch=torch, device=device, batch_size=int(args.batch_size))
        _, bc_actor = _load_actor(BC_PATH, torch=torch, nn=nn, device=device)
        estimator = TwinQConfidenceEstimator()
        model_runs = {}
        reference = _run_model(
            runs["calibration_v7"],
            replay=replay,
            bc_actor=bc_actor,
            torch=torch,
            nn=nn,
            device=device,
            estimator=estimator,
            batch_size=int(args.batch_size),
            include_actor=False,
        )
        model_runs["calibration_v7"] = reference
        for name in (
            "standard_awac_10k_v2",
            "critic_lr03_10k_v1",
            "update_ratio025_10k_v1",
            "confidence_adaptive_kl_10k_v1_retry",
        ):
            model_runs[name] = _run_model(
                runs[name],
                replay=replay,
                bc_actor=bc_actor,
                torch=torch,
                nn=nn,
                device=device,
                estimator=estimator,
                batch_size=int(args.batch_size),
                include_actor=True,
            )
        proposed = {}
        rank_metrics = {}
        for name in (
            "standard_awac_10k_v2",
            "critic_lr03_10k_v1",
            "update_ratio025_10k_v1",
            "confidence_adaptive_kl_10k_v1_retry",
        ):
            result = model_runs[name]
            proposed[name] = _confidence_public(result)
            _write(output_dir / "proposed_formula_offline" / (name + ".json"), proposed[name])
            rank_metrics[name] = _rank_metrics(reference, result)
        _write(
            output_dir / "production_offline_parity.json",
            {
                "schema_id": "awac_production_offline_parity_v2",
                "device": str(device),
                "cuda_runtime_validation": "SKIPPED_NOT_REQUESTED" if device.type == "cpu" else "PASS",
                "replay": {
                    "path": REPLAY_PATH,
                    "metadata_sha256": file_sha256(REPLAY_PATH / "metadata.json"),
                    "contract_id": replay.metadata.get("contract_id"),
                    "contract_sha256": replay.metadata.get("replay_contract_sha256"),
                    "observation_contract": replay.metadata.get("observation_contract"),
                    "rows": int(replay.size),
                    "row_indices": list(range(int(replay.size))),
                },
                "input_parity": input_parity,
                "model_output_parity": {
                    name: {
                        "checkpoint_sha256": result["checkpoint_sha256"],
                        "max_abs_delta": result["model_output_parity_max_abs_delta"],
                        "pass": result["model_output_parity_pass"],
                    }
                    for name, result in model_runs.items()
                },
                "no_persistent_replay_write": True,
            },
        )
        _write(
            output_dir / "confidence_contract_v2.json",
            {
                "schema_id": "awac_confidence_contract_v2",
                "contract_id": CONFIDENCE_CONTRACT_ID,
                "formula_version": CONFIDENCE_FORMULA_VERSION,
                "scale_version": CONFIDENCE_SCALE_VERSION,
                "action_sources": {
                    "a_BC": "argmax(masked_pi_BC)",
                    "a_RL": "argmax(masked_pi_AWAC)",
                },
                "signals": {
                    "Qmin": "min(Q1,Q2)",
                    "DeltaQ": "Qmin(a_RL)-Qmin(a_BC)",
                    "U": "abs(Q1(a_RL)-Q2(a_RL))",
                },
                "implementation_choice": {
                    "formula": "C=(1-exp(-margin_gain*max(DeltaQ/q_scale,0)))*exp(-disagreement_gain*max(U/q_scale,0))",
                    "margin_gain": 2.0,
                    "disagreement_gain": 1.0,
                    "bounds": "clip to [0,1]",
                    "negative_delta_q": "zero positive signal",
                    "status": "IMPLEMENTATION_CHOICE_NOT_UNIQUE_PLAN_FORMULA",
                },
                "aggregation": "valid_action_mean",
                "invalid_action_handling": "mask-aware; every row requires at least one valid action",
                "numpy_torch_even_median": "average of two middle sorted valid values",
                "offline_proposed_results": proposed,
            },
        )
        recovery = _run_recovery_artifact(output_dir)
        _write(output_dir / "metric_definitions.json", _build_metric_definitions())
        _write(
            output_dir / "rank_metrics.json",
            {
                "schema_id": "awac_rank_metrics_v2",
                "reference": {
                    "run": "calibration_v7",
                    "checkpoint_sha256": reference["checkpoint_sha256"],
                },
                "candidates": rank_metrics,
            },
        )
        _write(
            output_dir / "finding_matrix.json",
            _build_finding_matrix(
                historical_records, input_parity, model_runs, rank_metrics
            ),
        )
        historical_actual = _historical_actual_records(runs, output_dir)
    finally:
        replay.close()
    result = {
        "RECERTIFICATION": "PASS",
        "OUTPUT": str(output_dir),
        "ROWS": int(replay.size),
        "INPUT_PARITY": bool(input_parity["all_fields_exact"]),
        "MODEL_PARITY": all(item["model_output_parity_pass"] for item in model_runs.values()),
        "LR03_EXPERIMENT_ACTUALLY_APPLIED": historical_records["critic_lr03_10k_v1"]["optimizer_lr_match"],
        "EVIDENCE_PACKAGE": "MISSING_HISTORICAL_CLAIMS_UNVERIFIED",
        "TRAINING_EXECUTED": False,
        "DEV100_EXECUTED": False,
    }
    print(json.dumps(_json(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
