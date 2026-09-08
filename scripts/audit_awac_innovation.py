#!/usr/bin/env python3
"""Read-only confidence/rank audit over the formal V7 calibration Replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from planning.awac.confidence import TwinQConfidenceEstimator
from planning.awac.model import build_actor, build_critic, load_actor_state_dict_strict, masked_policy
from planning.awac.replay import AWACReplayBuffer
from planning.bc.model import require_torch
from planning.common.checkpoint import load_torch
from planning.common.hashing import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = ROOT / "data/awac/awac_bc60k_formal_critic_calibration_v7/replay"
DEFAULT_ORIGINAL = ROOT / "data/awac/formal/standard_awac_10k_v2/checkpoint_last.pt"
DEFAULT_TUNED = ROOT / "data/awac/tuning/standard_awac_update_ratio025_10k_v1/checkpoint_last.pt"
DEFAULT_BC = ROOT / "data/teach/2026_6w/bc_training/checkpoint_best_soft.pt"
DEFAULT_OUTPUT = ROOT / "data/awac/innovation_v1/offline_confidence_audit.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", default=str(DEFAULT_REPLAY))
    parser.add_argument("--original-checkpoint", default=str(DEFAULT_ORIGINAL))
    parser.add_argument("--tuned-checkpoint", default=str(DEFAULT_TUNED))
    parser.add_argument("--bc-checkpoint", default=str(DEFAULT_BC))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _rank_vector(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, -1, dtype=np.int64)
    valid = np.flatnonzero(mask)
    order = valid[np.argsort(values[valid], kind="stable")]
    result[order] = np.arange(order.size, dtype=np.int64)
    return result


def _spearman(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    left_rank = _rank_vector(left, mask)
    right_rank = _rank_vector(right, mask)
    valid = np.flatnonzero(mask)
    if valid.size < 2:
        return 1.0
    x = left_rank[valid].astype(np.float64)
    y = right_rank[valid].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return 1.0 if denominator == 0.0 else float(np.dot(x, y) / denominator)


def _load_critic(path: Path, *, torch, nn, device):
    payload = load_torch(path, torch=torch, map_location="cpu")
    model = build_critic(nn, depth_channels=int(payload.get("depth_history_frames", 1))).to(device)
    model.load_state_dict(payload["critic1_state_dict"], strict=True)
    critic2 = build_critic(nn, depth_channels=int(payload.get("depth_history_frames", 1))).to(device)
    critic2.load_state_dict(payload["critic2_state_dict"], strict=True)
    model.eval()
    critic2.eval()
    return payload, model, critic2


def _load_actor(path: Path, *, torch, nn, device):
    payload = load_torch(path, torch=torch, map_location="cpu")
    actor = build_actor(nn, depth_channels=int(payload.get("depth_history_frames", 1))).to(device)
    state_dict = payload.get("actor_state_dict", payload.get("model_state_dict"))
    if state_dict is None:
        raise ValueError("checkpoint lacks actor_state_dict or model_state_dict: {}".format(path))
    load_actor_state_dict_strict(actor, state_dict)
    actor.eval()
    return payload, actor


def _audit_model(
    *,
    replay,
    critic1,
    critic2,
    actor,
    bc_actor,
    estimator,
    torch,
    device,
    batch_size: int,
) -> dict:
    confidence = []
    margins = []
    disagreements = []
    delta_q = []
    uncertainty = []
    normalized_delta_q = []
    normalized_uncertainty = []
    qmin_rows = []
    actor_agree = []
    actor_confidence = []
    actor_top1_rows = []
    bc_top1_rows = []
    for start in range(0, int(replay.size), int(batch_size)):
        stop = min(int(replay.size), start + int(batch_size))
        # Replay.sample_indices is the production decode owner.  In
        # particular, uint8 depth is converted and divided by 255 exactly
        # once before it crosses the NumPy -> Torch boundary.
        indices = np.arange(start, stop, dtype=np.int64)
        batch = replay.sample_indices(indices, torch=torch, device=device)
        depth = batch["depth"]
        vector = batch["vector"]
        mask = batch["action_mask"]
        with torch.no_grad():
            q1 = critic1(depth, vector)
            q2 = critic2(depth, vector)
            logits = actor(depth, vector)
            bc_logits = bc_actor(depth, vector)
            policy, _, _ = masked_policy(logits, mask, torch)
            bc_policy, _, _ = masked_policy(bc_logits, mask, torch)
            result = estimator.estimate(
                q1,
                q2,
                mask,
                bc_policy=bc_policy,
                rl_policy=policy,
            )
            actor_top1 = policy.argmax(dim=1)
            bc_top1 = bc_policy.argmax(dim=1)
            actor_agree.extend((actor_top1 == bc_top1).cpu().numpy().tolist())
            actor_top1_rows.extend(actor_top1.cpu().numpy().tolist())
            bc_top1_rows.extend(bc_top1.cpu().numpy().tolist())
            actor_confidence.extend(result["confidence"].cpu().numpy().tolist())
            confidence.extend(result["confidence"].cpu().numpy().tolist())
            margins.extend(result["normalized_margin"].cpu().numpy().tolist())
            disagreements.extend(result["normalized_twin_disagreement"].cpu().numpy().tolist())
            delta_q.extend(result["delta_q"].cpu().numpy().tolist())
            uncertainty.extend(result["uncertainty"].cpu().numpy().tolist())
            normalized_delta_q.extend(
                result["normalized_delta_q"].cpu().numpy().tolist()
            )
            normalized_uncertainty.extend(
                result["normalized_uncertainty"].cpu().numpy().tolist()
            )
            qmin_rows.extend(torch.minimum(q1, q2).cpu().numpy())
    return {
        "confidence": np.asarray(confidence, dtype=np.float64),
        "normalized_margin": np.asarray(margins, dtype=np.float64),
        "normalized_disagreement": np.asarray(disagreements, dtype=np.float64),
        "delta_q": np.asarray(delta_q, dtype=np.float64),
        "uncertainty": np.asarray(uncertainty, dtype=np.float64),
        "normalized_delta_q": np.asarray(normalized_delta_q, dtype=np.float64),
        "normalized_uncertainty": np.asarray(
            normalized_uncertainty, dtype=np.float64
        ),
        "qmin": np.asarray(qmin_rows, dtype=np.float64),
        "actor_agree": np.asarray(actor_agree, dtype=bool),
        "actor_top1": np.asarray(actor_top1_rows, dtype=np.int64),
        "bc_top1": np.asarray(bc_top1_rows, dtype=np.int64),
        "actor_confidence": np.asarray(actor_confidence, dtype=np.float64),
    }


def _summarize_audit(values: Mapping[str, np.ndarray], *, q_argmax_stable: np.ndarray, q_argmax_changed: np.ndarray) -> dict:
    confidence = values["confidence"]
    agree = values["actor_agree"]
    result = {
        "confidence": _stats(confidence),
        "normalized_margin": _stats(values["normalized_margin"]),
        "normalized_twin_disagreement": _stats(values["normalized_disagreement"]),
        "actor_policy_agree_confidence_mean": float(np.mean(confidence[agree])) if np.any(agree) else None,
        "actor_policy_disagree_confidence_mean": float(np.mean(confidence[~agree])) if np.any(~agree) else None,
        "q_argmax_stable_count": int(q_argmax_stable.sum()),
        "q_argmax_change_count": int(q_argmax_changed.sum()),
        "confidence_mean_q_argmax_stable": float(np.mean(confidence[q_argmax_stable])) if np.any(q_argmax_stable) else None,
        "confidence_mean_q_argmax_change": float(np.mean(confidence[q_argmax_changed])) if np.any(q_argmax_changed) else None,
    }
    if not np.isfinite(confidence).all():
        raise FloatingPointError("offline confidence audit produced non-finite values")
    return result


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    replay_path = Path(args.replay).expanduser().resolve()
    original_path = Path(args.original_checkpoint).expanduser().resolve()
    tuned_path = Path(args.tuned_checkpoint).expanduser().resolve()
    bc_path = Path(args.bc_checkpoint).expanduser().resolve()
    for path in (replay_path, original_path, tuned_path, bc_path):
        if not path.exists():
            raise FileNotFoundError("audit input does not exist: {}".format(path))
    torch, nn, _, _, _ = require_torch()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    replay = AWACReplayBuffer.open(replay_path, read_only=True)
    try:
        original_payload, original_critic1, original_critic2 = _load_critic(
            original_path, torch=torch, nn=nn, device=device
        )
        tuned_payload, tuned_critic1, tuned_critic2 = _load_critic(
            tuned_path, torch=torch, nn=nn, device=device
        )
        bc_payload, bc_actor = _load_actor(bc_path, torch=torch, nn=nn, device=device)
        _, original_actor = _load_actor(original_path, torch=torch, nn=nn, device=device)
        _, tuned_actor = _load_actor(tuned_path, torch=torch, nn=nn, device=device)
        estimator = TwinQConfidenceEstimator()
        original = _audit_model(
            replay=replay, critic1=original_critic1, critic2=original_critic2,
            actor=original_actor, bc_actor=bc_actor, estimator=estimator,
            torch=torch, device=device, batch_size=int(args.batch_size)
        )
        tuned = _audit_model(
            replay=replay, critic1=tuned_critic1, critic2=tuned_critic2,
            actor=tuned_actor, bc_actor=bc_actor, estimator=estimator,
            torch=torch, device=device, batch_size=int(args.batch_size)
        )
        q_argmax_changed = []
        spearman = []
        top3_overlap = []
        arrays = replay.arrays
        masks = np.asarray(arrays["action_mask"][: int(replay.size)], dtype=bool)
        for index in range(int(replay.size)):
            left = original["qmin"][index]
            right = tuned["qmin"][index]
            valid = masks[index]
            left_order = np.flatnonzero(valid)[np.argsort(left[valid])[::-1]]
            right_order = np.flatnonzero(valid)[np.argsort(right[valid])[::-1]]
            left_argmax = int(valid[np.argmax(left[valid])])
            right_argmax = int(valid[np.argmax(right[valid])])
            q_argmax_changed.append(bool(left_argmax != right_argmax))
            top3_overlap.append(float(len(set(left_order[:3]).intersection(set(right_order[:3]))) / 3.0))
            spearman.append(_spearman(left, right, valid))
        q_argmax_changed = np.asarray(q_argmax_changed, dtype=bool)
        q_argmax_stable = ~q_argmax_changed
        original_summary = _summarize_audit(
            original, q_argmax_stable=q_argmax_stable, q_argmax_changed=q_argmax_changed
        )
        tuned_summary = _summarize_audit(
            tuned, q_argmax_stable=q_argmax_stable, q_argmax_changed=q_argmax_changed
        )
        disagreement = tuned["actor_top1"] != tuned["bc_top1"]
        disagreement_rows = np.flatnonzero(disagreement)
        reference_gap = (
            original["qmin"][disagreement_rows, tuned["actor_top1"][disagreement]]
            - original["qmin"][disagreement_rows, tuned["bc_top1"][disagreement]]
        )
        candidate_gap = (
            tuned["qmin"][disagreement_rows, tuned["actor_top1"][disagreement]]
            - tuned["qmin"][disagreement_rows, tuned["bc_top1"][disagreement]]
        )
        reference_sign = np.where(reference_gap > 1.0e-12, 1, np.where(reference_gap < -1.0e-12, -1, 0))
        candidate_sign = np.where(candidate_gap > 1.0e-12, 1, np.where(candidate_gap < -1.0e-12, -1, 0))
        preference_changed = reference_sign != candidate_sign
        report = {
            "schema_id": "awac_offline_confidence_audit_v2",
            "read_only": True,
            "replay": str(replay_path),
            "replay_rows": int(replay.size),
            "replay_observation_contract": replay.metadata.get("observation_contract"),
            "original_checkpoint": str(original_path),
            "original_checkpoint_sha256": file_sha256(original_path),
            "tuned_checkpoint": str(tuned_path),
            "tuned_checkpoint_sha256": file_sha256(tuned_path),
            "bc_checkpoint": str(bc_path),
            "bc_checkpoint_sha256": file_sha256(bc_path),
            "original_awac10k": original_summary,
            "update_ratio025": tuned_summary,
            "q_argmax_change_rate": {
                "numerator": int(q_argmax_changed.sum()),
                "denominator": int(q_argmax_changed.size),
                "rate": float(np.mean(q_argmax_changed)),
                "definition": "valid-action min(Q1,Q2) argmax change; smallest-index tie handling",
            },
            "action_preference_flip_on_policy_disagreement": {
                "numerator": int(preference_changed.sum()),
                "denominator": int(disagreement_rows.size),
                "rate": None if disagreement_rows.size == 0 else float(np.mean(preference_changed)),
                "zero_gap_epsilon": 1.0e-12,
                "definition": "sign change of Qmin(a_RL)-Qmin(a_BC) only on masked policy disagreement rows",
            },
            "q_rank_spearman_mean": float(np.mean(spearman)),
            "q_top3_overlap_mean": float(np.mean(top3_overlap)),
            "confidence_mean_policy_agree": tuned_summary["actor_policy_agree_confidence_mean"],
            "confidence_mean_policy_disagree": tuned_summary["actor_policy_disagree_confidence_mean"],
            "confidence_association_with_q_argmax_change": {
                "changed_mean": tuned_summary["confidence_mean_q_argmax_change"],
                "stable_mean": tuned_summary["confidence_mean_q_argmax_stable"],
                "status": "OBSERVED_ASSOCIATION_ONLY",
            },
            "ranking_change_observed": bool(q_argmax_changed.any() or preference_changed.any()),
            "ranking_error_causality": "INCONCLUSIVE",
            "reference_is_not_ground_truth": True,
            "threshold_auto_tuning": False,
        }
    finally:
        replay.close()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "OFFLINE_CONFIDENCE_AUDIT": "PASS",
        "ROWS": report["replay_rows"],
        "RANKING_ERROR_CAUSALITY": report["ranking_error_causality"],
        "OUTPUT": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
