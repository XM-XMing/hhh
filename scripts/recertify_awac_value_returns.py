#!/usr/bin/env python3
"""Read-only AWAC value-return and calibration-gate recertification.

The command intentionally evaluates only committed checkpoints on the raw,
persisted Calibration V7 holdout.  It has no Unity, ROS, Bridge, replay write,
optimizer, checkpoint-save, or policy-evaluation path.  Its only writes are a
new versioned diagnostic directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from planning.awac.calibration import (
    HOLDOUT_RETURN_SEMANTICS,
    RHO_COMPARISON,
    CriticCalibrationConfig,
    assess_value_policy_alignment,
    calibration_gate_contract,
    calibration_gate_contract_from_config,
    episode_monte_carlo_returns,
    spearman_rank_correlation,
)
from planning.awac.checkpoint import (
    calibration_holdout_records_sha256,
    validate_calibration_exact_resume_state,
)
from planning.awac.model import build_critic
from planning.bc.model import require_torch
from planning.common.checkpoint import load_torch
from planning.common.hashing import file_sha256


ROOT = Path(__file__).resolve().parents[1]
XM_ROOT = ROOT.parents[2]
DEFAULT_V7_CHECKPOINT = (
    ROOT
    / "data/awac/awac_bc60k_formal_critic_calibration_v7"
    / "checkpoint_calibration_pass.pt"
)
DEFAULT_CURRENT_25K_CHECKPOINT = (
    ROOT
    / "data/awac/mainline/confidence_adaptive_kl_10k_v2_retry"
    / "checkpoint_last.pt"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/awac/diagnostics/value_return_recertification_v1"
EXPECTED_V7_SHA256 = "1a0edc9949f8e67290dfd4d84d455123302f44afbebe92e8e80ed4a352dec8ea"
EXPECTED_CURRENT_25K_SHA256 = (
    "605d3b2071ad27a9ba69392b9d4233f208a84c53944a67dc0f09fe51fbe2e6c3"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _array_sha256(values: Sequence[float]) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    if not bool(np.isfinite(array).all()):
        raise ValueError("offline recertification array contains non-finite values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p01": float(np.percentile(array, 1.0)),
        "p50": float(np.percentile(array, 50.0)),
        "p99": float(np.percentile(array, 99.0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _choose_output_dir(requested: Path) -> Path:
    requested = Path(requested).expanduser().resolve()
    if not requested.exists():
        return requested
    suffix = 2
    while True:
        candidate = requested.with_name("{}_run{}".format(requested.name, suffix))
        if not candidate.exists():
            return candidate
        suffix += 1


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON object expected: {}".format(path))
    return dict(value)


def _validate_committed_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
) -> Dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    actual_sha256 = file_sha256(checkpoint_path)
    if actual_sha256 != str(expected_sha256):
        raise ValueError(
            "checkpoint SHA mismatch: actual={} expected={}".format(
                actual_sha256, expected_sha256
            )
        )
    transaction_path = checkpoint_path.with_suffix(".transaction.json")
    if not transaction_path.is_file():
        raise FileNotFoundError("checkpoint transaction missing: {}".format(transaction_path))
    transaction = _read_json(transaction_path)
    if transaction.get("transaction_state") != "COMMITTED":
        raise ValueError("checkpoint transaction is not COMMITTED: {}".format(path))
    if transaction.get("checkpoint_final_commit") != "PASS":
        raise ValueError("checkpoint transaction final commit is not PASS: {}".format(path))
    if str(transaction.get("checkpoint_sha256", "")) != actual_sha256:
        raise ValueError("checkpoint transaction SHA mismatch: {}".format(path))
    generation = checkpoint_path.parent / str(transaction.get("checkpoint_filename", ""))
    if not generation.is_file():
        raise FileNotFoundError("transaction generation missing: {}".format(generation))
    if file_sha256(generation) != actual_sha256:
        raise ValueError("canonical and committed generation SHA mismatch: {}".format(path))
    return {
        "path": checkpoint_path,
        "sha256": actual_sha256,
        "transaction_path": transaction_path,
        "transaction": transaction,
        "generation_path": generation,
    }


def _policy_contexts(
    *,
    bc_checkpoint_sha256: str,
    v7_checkpoint_sha256: str,
    current_checkpoint_sha256: str,
) -> Dict[str, Dict[str, Any]]:
    """Describe observed behavior, Bellman target, and MC continuation policy."""

    behavior = {
        "policy_id": "frozen_bc_argmax",
        "checkpoint_sha256": str(bc_checkpoint_sha256),
        "mask_contract": "depth_action_mask",
        "selection_mode": "deterministic_argmax",
        "temperature": 0.0,
    }
    v7_target = {
        "policy_id": "frozen_bc_masked_soft_distribution",
        "checkpoint_sha256": str(bc_checkpoint_sha256),
        "mask_contract": "depth_action_mask",
        "selection_mode": "masked_softmax",
        "temperature": 1.0,
    }
    current_target = {
        "policy_id": "current_awac_actor_masked_soft_distribution",
        "checkpoint_sha256": str(current_checkpoint_sha256),
        "mask_contract": "depth_action_mask",
        "selection_mode": "masked_softmax",
        "temperature": 1.0,
    }

    def context(target_policy: Mapping[str, Any], checkpoint_sha256: str) -> Dict[str, Any]:
        return {
            "critic_checkpoint_sha256": str(checkpoint_sha256),
            "behavior_policy": dict(behavior),
            "target_policy": dict(target_policy),
            "return_policy": dict(behavior),
            "alignment": assess_value_policy_alignment(
                behavior_policy=behavior,
                target_policy=target_policy,
                return_policy=behavior,
            ),
        }

    return {
        "v7": context(v7_target, v7_checkpoint_sha256),
        "current_25k": context(current_target, current_checkpoint_sha256),
    }


def _offline_value_certification(
    *,
    complete_episode_count: int,
    usable_row_count: int,
    mc_rho: Optional[float],
    value_policy_alignment: str,
    independent_snapshot_count: int,
) -> Dict[str, Any]:
    """Report the versioned gate boundary without forging historic windows."""

    config = CriticCalibrationConfig()
    reasons = []
    if int(complete_episode_count) <= 0 or int(usable_row_count) <= 0:
        reasons.append("complete_mc_episode_evidence_missing")
    if int(independent_snapshot_count) < int(config.min_stability_windows):
        reasons.append("independent_mc_windows_below_required_stability_count")
    if str(value_policy_alignment) != "MATCH":
        reasons.append("value_policy_alignment_{}".format(str(value_policy_alignment).lower()))
    if mc_rho is None or not math.isfinite(float(mc_rho)):
        reasons.append("q_mc_rank_correlation_unavailable")
    elif not float(mc_rho) > float(config.min_q_mc_rank_correlation):
        reasons.append("q_mc_rank_correlation_below_threshold")

    if not reasons:
        # This read-only tool intentionally does not invent the required
        # multi-time TD/disagreement windows, so it cannot promote a run.
        reasons.append("runtime_gate_stability_not_recomputed_by_read_only_tool")
    return {
        "state": "PENDING",
        "reason": (
            "PENDING_INSUFFICIENT_DATA"
            if "independent_mc_windows_below_required_stability_count" in reasons
            else "PENDING_EVIDENCE"
        ),
        "reason_details": reasons,
        "evidence_status": (
            "PENDING_INSUFFICIENT_DATA"
            if "independent_mc_windows_below_required_stability_count" in reasons
            else "PENDING_EVIDENCE"
        ),
        "would_open": False,
        "independent_mc_snapshot_count": int(independent_snapshot_count),
        "required_independent_stability_windows": int(config.min_stability_windows),
        "rho_threshold": float(config.min_q_mc_rank_correlation),
        "rho_comparison": RHO_COMPARISON,
        "policy_alignment_blocks_target_value_certification": (
            str(value_policy_alignment) != "MATCH"
        ),
        "historical_pass_not_auto_upgraded": True,
    }


def _public_return_contract(contract: Mapping[str, Any]) -> Dict[str, Any]:
    returns = list(contract["returns"])
    result = {
        key: _jsonable(value)
        for key, value in contract.items()
        if key != "returns"
    }
    result.update(
        {
        "returns_array_sha256": _array_sha256(returns),
        "returns_summary": _summary(returns),
        }
    )
    return result


def _evaluate_critic_on_holdout(
    *,
    checkpoint: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    returns: Sequence[float],
    torch,
    nn,
    batch_size: int,
) -> Dict[str, Any]:
    """Read exactly the behavior actions from the immutable holdout rows."""

    payload = checkpoint["payload"]
    device = torch.device("cpu")
    depth_channels = int(payload.get("depth_history_frames", 1))
    critic1 = build_critic(nn, depth_channels=depth_channels).to(device)
    critic2 = build_critic(nn, depth_channels=depth_channels).to(device)
    critic1.load_state_dict(payload["critic1_state_dict"], strict=True)
    critic2.load_state_dict(payload["critic2_state_dict"], strict=True)
    critic1.eval()
    critic2.eval()

    values = []
    depth_min = math.inf
    depth_max = -math.inf
    with torch.no_grad():
        for start in range(0, len(records), int(batch_size)):
            rows = records[start : start + int(batch_size)]
            depth = np.ascontiguousarray(
                np.asarray([row["depth"] for row in rows], dtype=np.float32)
            )
            vector = np.ascontiguousarray(
                np.asarray([row["vector"] for row in rows], dtype=np.float32)
            )
            mask = np.asarray([row["action_mask"] for row in rows], dtype=np.bool_)
            action = np.asarray([row["action"] for row in rows], dtype=np.int64)
            if not bool(np.isfinite(depth).all()) or not bool(np.isfinite(vector).all()):
                raise ValueError("holdout observation contains non-finite values")
            if np.any(action < 0) or np.any(action >= mask.shape[1]):
                raise ValueError("holdout action is outside action-mask width")
            if not bool(mask[np.arange(action.size), action].all()):
                raise ValueError("holdout behavior action is invalid under its mask")
            depth_min = min(depth_min, float(depth.min()))
            depth_max = max(depth_max, float(depth.max()))
            depth_tensor = torch.from_numpy(depth).to(device=device)
            vector_tensor = torch.from_numpy(vector).to(device=device)
            action_tensor = torch.from_numpy(action).to(device=device)
            q1 = critic1(depth_tensor, vector_tensor)
            q2 = critic2(depth_tensor, vector_tensor)
            qmin = torch.minimum(q1, q2)
            values.extend(
                qmin.gather(1, action_tensor[:, None]).squeeze(1).cpu().tolist()
            )
    q_values = np.asarray(values, dtype=np.float64)
    if q_values.size != len(records) or not bool(np.isfinite(q_values).all()):
        raise ValueError("critic holdout values are missing or non-finite")
    return_values = np.asarray(returns, dtype=np.float64)
    if return_values.shape != q_values.shape:
        raise ValueError("Q/MC return row alignment mismatch")
    return {
        "q_values": q_values,
        "q_summary": _summary(q_values),
        "q_absolute_extreme": float(np.max(np.abs(q_values))),
        "q_mc_spearman": spearman_rank_correlation(q_values, return_values),
        "holdout_depth_storage_dtype": "float32",
        "holdout_depth_range": {"min": float(depth_min), "max": float(depth_max)},
        "holdout_depth_decoder": (
            "exact_resume_float32_normalized_observation; no uint8 replay decode "
            "path is used by this holdout measurement"
        ),
    }


def _probe_before_after() -> Dict[str, Any]:
    evidence_paths = {
        "awac_md": XM_ROOT / "AWAC.md",
        "review_md": XM_ROOT / "AWAC_MD_VS_CODE_REVIEW_20260906.md",
        "evidence_zip": XM_ROOT / "AWAC_MD_VS_CODE_REVIEW_EVIDENCE_20260906.zip",
    }
    return {
        "schema_id": "awac_value_return_probe_before_after_v1",
        "source_evidence_package": {
            "paths": {name: str(path) for name, path in evidence_paths.items()},
            "available": {
                name: bool(path.is_file()) for name, path in evidence_paths.items()
            },
            "reproduce_review_py_executed": False,
            "reason": (
                "AWAC.md is available; the requested review markdown and evidence "
                "zip are absent, so their original reproduction script was not run"
            ),
        },
        "current_production_regressions": [
            {
                "id": "P01",
                "before": "negative rho could satisfy the historical gate",
                "after": "strict rho > 0.4 is required",
                "test": "test_gate_requires_strict_positive_mc_rank_evidence_and_policy_alignment",
            },
            {
                "id": "P02",
                "before": "missing or constant rho could pass",
                "after": "undefined rho is PENDING_EVIDENCE",
                "test": "test_gate_requires_strict_positive_mc_rank_evidence_and_policy_alignment",
            },
            {
                "id": "P03",
                "before": "decreasing TD/disagreement could be called sustained worsening",
                "after": "only signed increases can trigger sustained worsening",
                "test": "test_decreasing_metrics_are_not_misclassified_as_divergence",
            },
            {
                "id": "P04",
                "before": "negative Q extreme was not checked",
                "after": "max(abs(q_min), abs(q_max)) is checked",
                "test": "test_negative_q_extreme_is_a_hard_divergence",
            },
            {
                "id": "P05",
                "before": "single scaled reward was named return",
                "after": "per-episode backward Monte Carlo return",
                "test": "test_episode_monte_carlo_return_matches_worked_literal",
            },
            {
                "id": "P10",
                "before": "ESS existed only in learner metrics",
                "after": "raw and normalized/capped ESS fractions are TensorBoard scalars",
                "test": "test_tensorboard_snapshot_exposes_ess_independently_from_weight_clipping",
            },
            {
                "id": "P13",
                "before": "static ranking change could emit HIGH causal critic diagnosis",
                "after": "causality remains INCONCLUSIVE without action-return evidence",
                "test": "test_static_ranking_change_never_claims_high_causal_critic_error",
            },
        ],
    }


def _metric_definitions() -> Dict[str, Any]:
    return {
        "schema_id": "awac_value_return_metric_definitions_v1",
        "holdout_return": {
            "semantics": HOLDOUT_RETURN_SEMANTICS,
            "formula": "G_t = reward_scale * raw_reward_t + gamma * G_(t+1)",
            "grouping": "(mission_id, episode_id), sorted by episode_transition_index",
            "row_order": "restored to original persisted holdout order",
            "incomplete_episode_policy": "excluded with explicit contract error; never zero-bootstrapped",
        },
        "q_mc_rank_correlation": {
            "metric": "Spearman correlation between behavior-action Qmin(s_t,a_behavior) and behavior-policy MC return",
            "threshold": 0.4,
            "comparison": RHO_COMPARISON,
            "undefined_policy": "PENDING_EVIDENCE; no numeric substitute for constant or insufficient arrays",
            "target_policy_certificate_requires": "VALUE_POLICY_ALIGNMENT=MATCH",
        },
        "q_argmax_change_rate": {
            "definition": "changed valid-action Qmin argmax / comparable fixed-corpus states",
            "causal_status": "observational only",
        },
        "action_preference_flip_on_policy_disagreement": {
            "definition": "changed sign of Qmin(a_AWAC)-Qmin(a_BC) / states with BC/AWAC action disagreement",
            "causal_status": "observational only",
        },
        "ess_tensorboard": {
            "raw_weight_ess_fraction": "mean learner awac_raw_weight_ess_fraction in runner update window",
            "normalized_clipped_weight_ess_fraction": "mean learner awac_weight_ess_fraction in runner update window; normalized/capped actor-loss weights",
            "batch_size": "StandardAWACOnlineRunner.batch_size",
            "aggregation_window_updates": "len(StandardAWACOnlineRunner._online_metric_rows)",
            "clip_fraction": "separate diagnostic; not a substitute for ESS",
        },
    }


def _summary_markdown(
    *,
    v7: Mapping[str, Any],
    current: Mapping[str, Any],
    frozen_unchanged: bool,
) -> str:
    return "\n".join(
        (
            "# AWAC value-return offline recertification",
            "",
            "- This report is read-only: no runtime, replay write, optimizer, checkpoint, Dev100, or Unity action occurred.",
            "- V7 historical gate PASS is retained as historical evidence only; it is not a V2 MC-return certification.",
            "- V7 MC rho: `{}`; new gate: `{}` / `{}`.".format(
                v7["behavior_mc"]["q_mc_spearman"],
                v7["new_gate_certification"]["state"],
                v7["new_gate_certification"]["reason"],
            ),
            "- Current 25K MC rho: `{}`; new value certification: `{}` / `{}`.".format(
                current["behavior_mc"]["q_mc_spearman"],
                current["new_gate_certification"]["state"],
                current["new_gate_certification"]["reason"],
            ),
            "- Both comparisons are cross-policy diagnostics when their value-policy alignment is `MISMATCH`; they do not certify target-policy values.",
            "- Frozen input hashes unchanged after report generation: `{}`.".format(
                "YES" if frozen_unchanged else "NO"
            ),
            "- Missing review markdown/evidence zip are recorded as missing; the original reproduction script was not claimed or executed.",
            "",
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-checkpoint", type=Path, default=DEFAULT_V7_CHECKPOINT)
    parser.add_argument(
        "--current-25k-checkpoint", type=Path, default=DEFAULT_CURRENT_25K_CHECKPOINT
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if int(args.batch_size) <= 0 or int(args.cpu_threads) <= 0:
        raise ValueError("batch-size and cpu-threads must be positive")
    out_dir = _choose_output_dir(Path(args.out_dir))
    v7_checkpoint = _validate_committed_checkpoint(
        Path(args.v7_checkpoint), expected_sha256=EXPECTED_V7_SHA256
    )
    current_checkpoint = _validate_committed_checkpoint(
        Path(args.current_25k_checkpoint), expected_sha256=EXPECTED_CURRENT_25K_SHA256
    )
    frozen_paths = {
        "v7_checkpoint": v7_checkpoint["path"],
        "v7_transaction": v7_checkpoint["transaction_path"],
        "current_25k_checkpoint": current_checkpoint["path"],
        "current_25k_transaction": current_checkpoint["transaction_path"],
    }
    frozen_before = {name: file_sha256(path) for name, path in frozen_paths.items()}

    torch, nn, _, _, _ = require_torch()
    torch.set_num_threads(int(args.cpu_threads))
    v7_payload = load_torch(v7_checkpoint["path"], torch=torch, map_location="cpu")
    current_payload = load_torch(
        current_checkpoint["path"], torch=torch, map_location="cpu"
    )
    v7_checkpoint["payload"] = v7_payload
    current_checkpoint["payload"] = current_payload
    exact_resume = validate_calibration_exact_resume_state(
        v7_payload.get("exact_resume_state", {})
    )
    records = list(exact_resume["raw_holdout_records"])
    corpus_sha256 = calibration_holdout_records_sha256(records)
    if corpus_sha256 != str(exact_resume["raw_holdout_records_sha256"]):
        raise ValueError("V7 raw holdout corpus SHA mismatch")
    training_contract = v7_payload.get("training_contract", {})
    if not isinstance(training_contract, Mapping):
        raise ValueError("V7 training contract missing")
    gamma = float(training_contract["gamma"])
    reward_scale = float(v7_payload["reward_scale"])
    current_training_contract = current_payload.get("training_contract", {})
    if not isinstance(current_training_contract, Mapping):
        raise ValueError("current 25K training contract missing")
    if float(current_training_contract["gamma"]) != gamma:
        raise ValueError("V7/current gamma mismatch for shared MC corpus")
    if float(current_payload["reward_scale"]) != reward_scale:
        raise ValueError("V7/current reward scale mismatch for shared MC corpus")
    return_contract = episode_monte_carlo_returns(
        records,
        gamma=gamma,
        reward_scale=reward_scale,
    )
    if str(return_contract["return_semantics"]) != HOLDOUT_RETURN_SEMANTICS:
        raise ValueError("unexpected MC return semantics")
    if int(return_contract["complete_episode_count"]) <= 0:
        raise ValueError("V7 does not contain complete holdout episodes")
    if current_payload.get("exact_resume_state") not in (None, {}):
        raise ValueError("current 25K unexpectedly owns a second calibration corpus")
    if str(current_payload.get("source_calibration_checkpoint_sha256", "")) != str(
        v7_checkpoint["sha256"]
    ):
        raise ValueError("current 25K source calibration checkpoint mismatch")

    contexts = _policy_contexts(
        bc_checkpoint_sha256=str(v7_payload["source_bc_checkpoint_sha256"]),
        v7_checkpoint_sha256=str(v7_checkpoint["sha256"]),
        current_checkpoint_sha256=str(current_checkpoint["sha256"]),
    )
    v7_eval = _evaluate_critic_on_holdout(
        checkpoint=v7_checkpoint,
        records=records,
        returns=return_contract["returns"],
        torch=torch,
        nn=nn,
        batch_size=int(args.batch_size),
    )
    current_eval = _evaluate_critic_on_holdout(
        checkpoint=current_checkpoint,
        records=records,
        returns=return_contract["returns"],
        torch=torch,
        nn=nn,
        batch_size=int(args.batch_size),
    )
    legacy_returns = [float(row["reward"]) * reward_scale for row in records]
    public_return = _public_return_contract(return_contract)
    corpus = {
        "source": "Calibration V7 exact_resume_state.raw_holdout_records",
        "sha256": corpus_sha256,
        "row_indices": list(range(len(records))),
        "row_count": len(records),
        "episode_keys": public_return["episode_keys"],
        "complete_episode_count": int(return_contract["complete_episode_count"]),
        "usable_row_count": int(return_contract["usable_row_count"]),
        "excluded_episode_count": len(return_contract["invalid_episodes"]),
        "excluded_episodes": return_contract["invalid_episodes"],
        "terminal_contract": training_contract.get("terminal_contract", {}),
        "current_25k_uses_v7_fixed_corpus": True,
    }

    def report_for(
        name: str,
        checkpoint: Mapping[str, Any],
        evaluation: Mapping[str, Any],
        context: Mapping[str, Any],
        historical_gate: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        alignment = context["alignment"]
        certification = _offline_value_certification(
            complete_episode_count=int(return_contract["complete_episode_count"]),
            usable_row_count=int(return_contract["usable_row_count"]),
            mc_rho=evaluation["q_mc_spearman"],
            value_policy_alignment=str(alignment["status"]),
            independent_snapshot_count=1,
        )
        return {
            "schema_id": "awac_value_return_offline_recertification_v1",
            "run": name,
            "checkpoint": {
                "path": checkpoint["path"],
                "sha256": checkpoint["sha256"],
                "transaction_path": checkpoint["transaction_path"],
                "generation_path": checkpoint["generation_path"],
            },
            "fixed_corpus": corpus,
            "gamma": gamma,
            "reward_scale": reward_scale,
            "reward_scale_applied_once": True,
            "old_single_step_reward_diagnostic": {
                "semantics": "scaled_immediate_reward_historical_diagnostic_only",
                "returns_array_sha256": _array_sha256(legacy_returns),
                "returns_summary": _summary(legacy_returns),
                "q_spearman": spearman_rank_correlation(
                    evaluation["q_values"], legacy_returns
                ),
            },
            "behavior_mc": {
                "semantics": HOLDOUT_RETURN_SEMANTICS,
                "returns_array_sha256": public_return["returns_array_sha256"],
                "returns_summary": public_return["returns_summary"],
                "q_mc_spearman": evaluation["q_mc_spearman"],
            },
            "q_summary": evaluation["q_summary"],
            "q_absolute_extreme": evaluation["q_absolute_extreme"],
            "holdout_depth_contract": {
                "storage_dtype": evaluation["holdout_depth_storage_dtype"],
                "range": evaluation["holdout_depth_range"],
                "decoder": evaluation["holdout_depth_decoder"],
            },
            "value_policy": context,
            "value_policy_alignment": alignment,
            "new_gate_certification": certification,
            "signed_stability": {
                "status": "INSUFFICIENT_REAL_MULTI_TIME_SNAPSHOTS",
                "independent_snapshot_count": 1,
                "signed_td_changes": [],
                "signed_disagreement_changes": [],
                "reason": (
                    "Only the final checkpoint was remeasured with the fixed V7 "
                    "MC corpus; repeated final snapshots are not stability windows"
                ),
            },
            "historical_gate": dict(historical_gate or {}),
            "historical_pass_not_auto_upgraded": True,
            "no_training_executed": True,
        }

    v7_history = v7_payload.get("calibration_metrics", {})
    v7_report = report_for(
        "calibration_v7",
        v7_checkpoint,
        v7_eval,
        contexts["v7"],
        {
            "historical_gate_state": v7_payload.get("calibration_gate_state"),
            "historical_calibration_state": (
                v7_history.get("state") if isinstance(v7_history, Mapping) else None
            ),
            "historical_gate_contract": calibration_gate_contract_from_config(
                training_contract.get("calibration")
                if isinstance(training_contract.get("calibration"), Mapping)
                else None
            ),
        },
    )
    current_report = report_for(
        "current_25k",
        current_checkpoint,
        current_eval,
        contexts["current_25k"],
        {
            "historical_gate_state": "NOT_A_CALIBRATION_CHECKPOINT",
            "historical_calibration_state": "NOT_APPLICABLE",
            "source_calibration_checkpoint_sha256": current_payload.get(
                "source_calibration_checkpoint_sha256"
            ),
        },
    )
    frozen_after = {name: file_sha256(path) for name, path in frozen_paths.items()}
    frozen_unchanged = frozen_before == frozen_after
    if not frozen_unchanged:
        raise RuntimeError("read-only recertification changed a frozen artifact")

    output = {
        "return_contract": {
            "schema_id": "awac_holdout_return_contract_v2",
            **public_return,
            "corpus": corpus,
            "reward_scale_owner": v7_payload.get("reward_scale_owner"),
        },
        "gate_contract": {
            "schema_id": "awac_calibration_gate_contract_report_v2",
            "future_runtime_contract": calibration_gate_contract(
                CriticCalibrationConfig()
            ),
            "legacy_v7_contract": v7_report["historical_gate"][
                "historical_gate_contract"
            ],
            "historical_pass_not_auto_upgraded": True,
        },
        "probes": _probe_before_after(),
        "value_policy_alignment": {
            "schema_id": "awac_value_policy_alignment_v1",
            "v7": contexts["v7"],
            "current_25k": contexts["current_25k"],
            "minimum_future_evidence": (
                "Collect a complete fixed-policy holdout whose behavior, return, "
                "and Bellman target policy identities match, then measure at real "
                "independent checkpoints"
            ),
        },
        "v7": v7_report,
        "current": current_report,
        "metric_definitions": _metric_definitions(),
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    _write_json(out_dir / "return_contract.json", output["return_contract"])
    _write_json(out_dir / "gate_contract.json", output["gate_contract"])
    _write_json(out_dir / "probe_before_after.json", output["probes"])
    _write_json(
        out_dir / "value_policy_alignment.json", output["value_policy_alignment"]
    )
    _write_json(out_dir / "v7_offline_recertification.json", v7_report)
    _write_json(out_dir / "25k_offline_recertification.json", current_report)
    _write_json(out_dir / "metric_definitions.json", output["metric_definitions"])
    (out_dir / "recertification_summary.md").write_text(
        _summary_markdown(
            v7=v7_report,
            current=current_report,
            frozen_unchanged=frozen_unchanged,
        ),
        encoding="utf-8",
    )
    return {
        "out_dir": out_dir,
        "v7": v7_report,
        "current": current_report,
        "frozen_artifacts_unchanged": frozen_unchanged,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    result = run(build_parser().parse_args(argv))
    print("RL_VALUE_RETURN_OFFLINE_RECERTIFICATION=PASS")
    print(
        json.dumps(
            {
                "out_dir": str(result["out_dir"]),
                "v7_mc_rho": result["v7"]["behavior_mc"]["q_mc_spearman"],
                "v7_new_gate": result["v7"]["new_gate_certification"]["state"],
                "current_25k_mc_rho": result["current"]["behavior_mc"][
                    "q_mc_spearman"
                ],
                "current_25k_new_gate": result["current"][
                    "new_gate_certification"
                ]["state"],
                "frozen_artifacts_unchanged": result["frozen_artifacts_unchanged"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
