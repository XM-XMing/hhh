#!/usr/bin/env python3
"""Train normalized BC with exact teacher CE and soft-target KL supervision."""

from __future__ import annotations

from pathlib import Path
import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from planning.bc.model import VectorNormalizer, build_model, mask_logits, require_torch
from planning.data.bc_mmap import (
    BC_MMAP_DATASET_CONTRACT_ID,
    MappedSoftRolloutDataset,
    read_episode_ids_file,
)
from planning.data.rollout import (
    LABEL_CONTRACT_ID,
    TeacherLabelStore,
    load_rollout_episode,
    resolve_dataset_path,
)
from planning.common.config import parse_bool
from planning.safety.depth_mask import DepthActionMaskStore
from planning.contracts.feature import (
    CONTINUOUS_DIM,
    FEATURE_CONTRACT_ID,
    INITIAL_PREV_ACTION,
    NUM_ACTIONS,
    POLICY_VECTOR_DIM,
    actions_onehot,
    policy_input_contract,
    policy_input_contract_sha256,
)
from planning.mission.spec import (
    MISSION_PLANAR_DISTANCE_M,
    TASK_CONTRACT_ID,
    mission_id_from_row,
    task_contract_sha256,
    validate_mission_rows,
)
from planning.contracts.task import (
    DEFAULT_MAX_PRIMITIVE_STEPS,
    task_contract_fields,
    validate_task_contract,
)
from planning.common import canonical_json_sha256, file_sha256, print_progress, read_csv, write_csv_atomic
from planning.primitives.library import MotionPrimitiveLibrary
from planning.diagnostics.observability import StructuredRunLogger
from planning.contracts.policy_runtime import (
    EXECUTION_MODES,
    POLICY_RUNTIME_CONTRACT_ID,
    SAFETY_MASKS,
)

from planning.common.hardware import HardwareProfile, configure_torch_runtime
from planning.common import LatencyTracker, seed_everything, write_json_atomic
from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    observation_contract,
)


_BC_MMAP_PROVENANCE_FIELDS = (
    "source_index_sha256",
    "source_labels_sha256",
    "source_depth_masks_sha256",
    "mpl_contract_sha256",
)


@dataclass(frozen=True)
class BCTrainingConfig:
    """Canonical algorithm defaults projected by the BC CLI."""

    epochs: int = 40
    batch_size: int = 128
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4


DEFAULT_BC_TRAINING_CONFIG = BCTrainingConfig()


def _manifest_count(manifest: Mapping[str, Any], key: str, path: str) -> int:
    if key not in manifest:
        raise ValueError("{} missing {}".format(path, key))
    try:
        value = int(manifest[key])
    except (TypeError, ValueError):
        raise ValueError("{} {} must be an integer".format(path, key))
    if value < 0:
        raise ValueError("{} {} must be non-negative".format(path, key))
    return value


def validate_bc_mmap_provenance(
    manifest: Mapping[str, Any],
    *,
    expected_observation_contract: str = EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    path: str = "BC mmap manifest",
) -> Dict[str, Any]:
    """Validate the immutable provenance boundary before BC dataset loading."""

    if str(manifest.get("contract_id", "")) != BC_MMAP_DATASET_CONTRACT_ID:
        raise ValueError("{} dataset contract mismatch".format(path))
    expected_max_steps = manifest.get(
        "max_steps",
        manifest.get("max_primitive_steps", DEFAULT_MAX_PRIMITIVE_STEPS),
    )
    validate_task_contract(
        manifest,
        expected_max_primitive_steps=expected_max_steps,
        path="{} task contract".format(path),
    )
    source = str(manifest.get("observation_source", "")).strip()
    if not source:
        raise ValueError("{} missing observation source".format(path))
    contract = observation_contract(manifest, path=path)
    expected = str(expected_observation_contract).strip()
    if expected != EXACT_ENDPOINT_OBSERVATION_CONTRACT:
        raise ValueError(
            "{} requires observation contract {}".format(
                path, EXACT_ENDPOINT_OBSERVATION_CONTRACT
            )
        )
    if contract != expected:
        raise ValueError(
            "{} observation contract mismatch: received={} expected={}".format(
                path, contract, expected
            )
        )

    total = _manifest_count(manifest, "transition_count", path)
    reliable = _manifest_count(manifest, "reliable_rows", path)
    legacy = _manifest_count(manifest, "legacy_rows", path)
    if legacy != 0:
        raise ValueError("{} reliable dataset contains legacy rows".format(path))
    if reliable != total:
        raise ValueError(
            "{} reliable_rows {} != transition_count {}".format(
                path, reliable, total
            )
        )
    for field in _BC_MMAP_PROVENANCE_FIELDS:
        value = str(manifest.get(field, "")).strip().lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("{} missing or invalid {}".format(path, field))

    return {
        "observation_contract": contract,
        "observation_source": source,
        "reliable_rows": reliable,
        "legacy_rows": legacy,
        "transition_count": total,
        "dataset_manifest_contract_id": str(manifest["contract_id"]),
        "source_index_sha256": str(manifest["source_index_sha256"]),
        "source_labels_sha256": str(manifest["source_labels_sha256"]),
        "source_depth_masks_sha256": str(manifest["source_depth_masks_sha256"]),
        "mpl_contract_sha256": str(manifest["mpl_contract_sha256"]),
        "task_contract_id": str(manifest["task_contract_id"]),
        "task_contract_schema_version": int(
            manifest["task_contract_schema_version"]
        ),
        "task_contract_sha256": str(manifest["task_contract_sha256"]),
        "max_primitive_steps": int(manifest["max_primitive_steps"]),
        "max_steps": int(manifest.get("max_steps", manifest["max_primitive_steps"])),
    }


def _normalizer_sha256(normalizer: VectorNormalizer) -> str:
    digest = hashlib.sha256()
    for name in ("mean", "std"):
        values = np.asarray(getattr(normalizer, name), dtype=np.float32)
        digest.update(name.encode("utf-8"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _relative_training_args(args, base: Path) -> Dict[str, Any]:
    values = dict(vars(args))
    for key in (
        "index", "labels", "depth_action_masks", "dataset_cache", "out_dir",
        "tensorboard_log_dir", "val_episodes_from_checkpoint",
    ):
        value = values.get(key)
        if value:
            values[key] = os.path.relpath(
                str(Path(value).expanduser().resolve()), str(Path(base).resolve())
            )
    return values


def _resolved_training_config(args, out_dir: Path, provenance: Mapping[str, Any]) -> Dict[str, Any]:
    """Persist the final CLI values and the resolved observation contract."""

    config = {
        "args": _relative_training_args(args, out_dir),
        "observation_contract": str(provenance["observation_contract"]),
        "observation_source": str(provenance["observation_source"]),
        "reliable_rows": int(provenance["reliable_rows"]),
        "legacy_rows": int(provenance["legacy_rows"]),
        "task_contract_id": str(provenance["task_contract_id"]),
        "task_contract_schema_version": int(provenance["task_contract_schema_version"]),
        "task_contract_sha256": str(provenance["task_contract_sha256"]),
        "max_primitive_steps": int(provenance["max_primitive_steps"]),
    }
    return config


def split_episodes(rows: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    """Split by stable mission identity to prevent rollout leakage."""
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        groups.setdefault(mission_id_from_row(row), []).append(row)
    mission_ids = sorted(groups)
    rng = random.Random(int(seed))
    rng.shuffle(mission_ids)
    n_val = max(1, int(round(len(mission_ids) * float(val_ratio)))) if len(mission_ids) > 1 else 0
    val_ids = set(mission_ids[:n_val])
    train_rows = [row for mission_id in mission_ids if mission_id not in val_ids for row in groups[mission_id]]
    val_rows = [row for mission_id in mission_ids if mission_id in val_ids for row in groups[mission_id]]
    return train_rows, val_rows


def split_rows_from_checkpoint(
    rows: List[Dict],
    split_checkpoint: Mapping[str, Any],
    *,
    allow_subset: bool = False,
    path: str = "validation-split checkpoint",
) -> Tuple[List[Dict], List[Dict]]:
    """Apply a checkpoint split to either the full index or a known subset.

    A scale-ablation dataset is a canonical subset of the same source index.
    In that case the checkpoint's full validation set is intentionally only
    partially present; the subset must still contain only known train/val
    missions and must contain at least one validation mission.  Full-index
    callers retain the original exact-split requirement.
    """

    if not rows:
        raise RuntimeError("fixed validation split has no episodes")
    if not isinstance(split_checkpoint, Mapping):
        raise ValueError("{} must be a mapping".format(path))

    if split_checkpoint.get("val_mission_ids"):
        val_mission_ids = {
            str(value) for value in split_checkpoint["val_mission_ids"]
        }
        available_mission_ids = {mission_id_from_row(row) for row in rows}
        train_mission_ids = {
            str(value) for value in split_checkpoint.get("train_mission_ids", [])
        }
        known_mission_ids = train_mission_ids | val_mission_ids
        unknown = available_mission_ids - known_mission_ids
        if unknown:
            raise ValueError(
                "selected index contains missions absent from checkpoint split: {}".format(
                    sorted(unknown)[:10]
                )
            )
        selected_val_mission_ids = available_mission_ids & val_mission_ids
        missing = val_mission_ids - available_mission_ids
        if missing and not allow_subset:
            raise ValueError(
                "{} is missing checkpoint validation missions: {}".format(
                    path, sorted(missing)[:10]
                )
            )
        if not selected_val_mission_ids:
            raise ValueError(
                "selected index contains no checkpoint validation missions"
            )
        val_rows = [
            row for row in rows
            if mission_id_from_row(row) in selected_val_mission_ids
        ]
        train_rows = [
            row for row in rows
            if mission_id_from_row(row) not in selected_val_mission_ids
        ]
    else:
        # Compatibility for checkpoints created before mission-group splitting.
        val_ids = {
            int(value) for value in split_checkpoint.get("val_episodes", [])
        }
        if not val_ids:
            raise ValueError("{} has no validation split".format(path))
        available_ids = {
            int(float(row.get("episode_id", -1))) for row in rows
        }
        known_ids = {
            int(value)
            for value in split_checkpoint.get("train_episodes", [])
        } | val_ids
        unknown = available_ids - known_ids if known_ids else set()
        if unknown:
            raise ValueError(
                "selected index contains episodes absent from checkpoint split: {}".format(
                    sorted(unknown)[:10]
                )
            )
        selected_val_ids = available_ids & val_ids
        missing = val_ids - available_ids
        if missing and not allow_subset:
            raise ValueError(
                "{} is missing checkpoint validation episodes: {}".format(
                    path, sorted(missing)[:10]
                )
            )
        if not selected_val_ids:
            raise ValueError(
                "selected index contains no checkpoint validation episodes"
            )
        val_rows = [
            row for row in rows
            if int(float(row.get("episode_id", -1))) in selected_val_ids
        ]
        train_rows = [
            row for row in rows
            if int(float(row.get("episode_id", -1))) not in selected_val_ids
        ]

    if not train_rows:
        raise RuntimeError("fixed validation split leaves no training episodes")
    if not val_rows:
        raise RuntimeError("fixed validation split leaves no validation episodes")
    return train_rows, val_rows


class SoftRolloutDataset:
    def __init__(
        self,
        rows: List[Dict],
        index_path: Path,
        labels: TeacherLabelStore,
        depth_masks: DepthActionMaskStore = None,
    ):
        self.depths = []
        self.continuous = []
        self.prev_actions = []
        self.height_masks = []
        self.local_depth_masks = []
        self.behavior_actions = []
        self.teacher_actions = []
        self.soft_targets = []
        self.history_starts = []
        self.episode_ids = []
        self.vectors = None

        transition_offset = 0
        for row in rows:
            rollout_path = resolve_dataset_path(row, index_path)
            episode = load_rollout_episode(rollout_path, validate=True)
            behavior = np.asarray(episode["behavior_actions"], dtype=np.int64)
            label = labels.get_episode(rollout_path, expected_behavior_actions=behavior)
            transition_count = int(behavior.shape[0])
            self.depths.append(np.asarray(episode["depths"], dtype=np.float16))
            self.continuous.append(
                np.concatenate(
                    [
                        np.asarray(episode["states"], dtype=np.float32),
                        np.asarray(episode["goals"], dtype=np.float32),
                    ],
                    axis=1,
                )
            )
            self.prev_actions.append(np.asarray(episode["prev_actions"], dtype=np.int64))
            self.height_masks.append(np.asarray(episode["height_action_masks"], dtype=np.bool_))
            if depth_masks is None:
                self.local_depth_masks.append(np.ones((transition_count, NUM_ACTIONS), dtype=np.bool_))
            else:
                self.local_depth_masks.append(depth_masks.get_episode(rollout_path, transition_count))
            self.behavior_actions.append(behavior)
            self.teacher_actions.append(np.asarray(label["teacher_argmax"], dtype=np.int64))
            self.soft_targets.append(np.asarray(label["soft_targets"], dtype=np.float32))
            self.history_starts.append(np.full(transition_count, transition_offset, dtype=np.int64))
            transition_offset += transition_count
            episode_id = int(float(row.get("episode_id", -1)))
            self.episode_ids.extend([episode_id] * transition_count)

        if not self.depths:
            raise RuntimeError("empty dataset")
        self.depths = np.concatenate(self.depths, axis=0).astype(np.float16, copy=False)
        self.continuous = np.concatenate(self.continuous, axis=0).astype(np.float32, copy=False)
        self.prev_actions = np.concatenate(self.prev_actions, axis=0).astype(np.int64, copy=False)
        self.height_masks = np.concatenate(self.height_masks, axis=0).astype(np.bool_, copy=False)
        self.local_depth_masks = np.concatenate(self.local_depth_masks, axis=0).astype(np.bool_, copy=False)
        self.behavior_actions = np.concatenate(self.behavior_actions, axis=0).astype(np.int64, copy=False)
        self.teacher_actions = np.concatenate(self.teacher_actions, axis=0).astype(np.int64, copy=False)
        self.soft_targets = np.concatenate(self.soft_targets, axis=0).astype(np.float32, copy=False)
        self.history_starts = np.concatenate(self.history_starts, axis=0).astype(np.int64, copy=False)
        self.episode_ids = np.asarray(self.episode_ids, dtype=np.int64)
        if self.continuous.shape[1] != CONTINUOUS_DIM:
            raise ValueError("continuous feature dim mismatch: {}".format(self.continuous.shape))
        if self.history_starts.shape != (len(self),):
            raise ValueError("depth history index shape mismatch")

    def apply_normalizer(self, normalizer: VectorNormalizer) -> None:
        normalized = normalizer.transform_continuous(self.continuous)
        previous_onehot = actions_onehot(self.prev_actions, NUM_ACTIONS)
        self.vectors = np.concatenate([normalized, previous_onehot], axis=1).astype(np.float32)
        if self.vectors.shape[1] != POLICY_VECTOR_DIM:
            raise RuntimeError("policy vector dim mismatch")

    def __len__(self) -> int:
        return int(self.behavior_actions.shape[0])

    def teacher_hist(self) -> np.ndarray:
        valid = self.teacher_actions[(self.teacher_actions >= 0) & (self.teacher_actions < NUM_ACTIONS)]
        return np.bincount(valid, minlength=NUM_ACTIONS)

    def behavior_hist(self) -> np.ndarray:
        return np.bincount(self.behavior_actions, minlength=NUM_ACTIONS)

def make_torch_dataset(base, torch, Dataset, depth_history_frames: int):
    if getattr(base, "normalizer", None) is None and getattr(base, "vectors", None) is None:
        raise RuntimeError("normalizer has not been applied")
    if int(depth_history_frames) <= 0:
        raise ValueError("depth_history_frames must be positive")

    class TorchDataset(Dataset):
        def __len__(self):
            return len(base)

        def __getitem__(self, index: int):
            if isinstance(base, MappedSoftRolloutDataset):
                global_index, history_indices, vector = base.sample(index, depth_history_frames)
                return (
                    torch.from_numpy(np.asarray(base.depths[history_indices], dtype=np.float16)),
                    torch.from_numpy(vector),
                    torch.from_numpy(np.array(base.height_masks[global_index], dtype=np.bool_, copy=True)),
                    torch.from_numpy(np.array(base.local_depth_masks[global_index], dtype=np.bool_, copy=True)),
                    torch.tensor(int(base.behavior_actions[global_index]), dtype=torch.long),
                    torch.tensor(int(base.teacher_actions[global_index]), dtype=torch.long),
                    torch.from_numpy(np.array(base.soft_targets[global_index], dtype=np.float32, copy=True)),
                )
            history_indices = np.maximum(
                int(index) - np.arange(int(depth_history_frames) - 1, -1, -1),
                base.history_starts[index],
            )
            return (
                torch.from_numpy(base.depths[history_indices]),
                torch.from_numpy(base.vectors[index]),
                torch.from_numpy(base.height_masks[index]),
                torch.from_numpy(base.local_depth_masks[index]),
                torch.tensor(base.behavior_actions[index], dtype=torch.long),
                torch.tensor(base.teacher_actions[index], dtype=torch.long),
                torch.from_numpy(base.soft_targets[index]),
            )

    return TorchDataset()


def compute_loss_and_metrics(
    model,
    loader,
    optimizer,
    scaler,
    device,
    torch,
    train: bool,
    ce_weight: float,
    kl_weight: float,
    ce_target: str,
    loss_mask: str,
    use_amp: bool,
    grad_clip: float,
) -> Dict[str, float]:
    """Run one epoch without forcing invalid targets back into the mask."""
    model.train(mode=train)
    totals = {
        "ce_sum": 0.0,
        "kl_sum": 0.0,
        "ce_samples": 0,
        "kl_samples": 0,
        "teacher_top1_correct": 0,
        "teacher_top5_correct": 0,
        "teacher_samples": 0,
        "behavior_top1_correct": 0,
        "behavior_top5_correct": 0,
        "behavior_samples": 0,
        "invalid_teacher": 0,
        "invalid_behavior": 0,
        "dropped": 0,
    }
    batch_latency = LatencyTracker()
    processed_samples = 0

    for depth, vector, height_mask, local_depth_mask, behavior, teacher, soft in loader:
        batch_started = time.perf_counter()
        processed_samples += int(behavior.shape[0])
        depth = depth.to(device=device, dtype=torch.float32, non_blocking=True)
        vector = vector.to(device=device, dtype=torch.float32, non_blocking=True)
        height_mask = height_mask.to(device=device, dtype=torch.bool, non_blocking=True)
        local_depth_mask = local_depth_mask.to(device=device, dtype=torch.bool, non_blocking=True)
        behavior = behavior.to(device=device, dtype=torch.long, non_blocking=True)
        teacher = teacher.to(device=device, dtype=torch.long, non_blocking=True)
        soft = soft.to(device=device, dtype=torch.float32, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)

        if loss_mask == "height":
            effective_mask = height_mask
        elif loss_mask == "height_depth":
            effective_mask = height_mask & local_depth_mask
        else:
            effective_mask = torch.ones_like(height_mask)

        batch_ids = torch.arange(behavior.shape[0], device=device)
        teacher_in_range = (teacher >= 0) & (teacher < NUM_ACTIONS)
        safe_teacher = teacher.clamp(0, NUM_ACTIONS - 1)
        teacher_valid = teacher_in_range & effective_mask[batch_ids, safe_teacher]
        behavior_valid = effective_mask[batch_ids, behavior]

        # Remove probability mass outside the selected training mask and
        # renormalize. Empty soft targets are skipped instead of fabricated.
        soft = soft * effective_mask.float()
        soft_sum = soft.sum(dim=1, keepdim=True)
        soft_valid = soft_sum.squeeze(1) > 1.0e-8
        soft = soft / soft_sum.clamp_min(1.0e-8)

        ce_actions = teacher if ce_target == "teacher" else behavior
        ce_valid = teacher_valid if ce_target == "teacher" else behavior_valid
        contributes = (ce_valid & (float(ce_weight) != 0.0)) | (soft_valid & (float(kl_weight) != 0.0))
        totals["invalid_teacher"] += int((~teacher_valid).sum().item())
        totals["invalid_behavior"] += int((~behavior_valid).sum().item())
        totals["dropped"] += int((~contributes).sum().item())
        if not bool(torch.any(contributes)):
            continue

        with torch.set_grad_enabled(train):
            if bool(use_amp):
                try:
                    amp_context = torch.amp.autocast(device_type="cuda", enabled=True)
                except (AttributeError, TypeError):
                    amp_context = torch.cuda.amp.autocast(enabled=True)
            else:
                amp_context = nullcontext()
            with amp_context:
                logits = model(depth, vector)
                logits_for_loss = mask_logits(logits, effective_mask) if loss_mask != "none" else logits
                ce_per = torch.nn.functional.cross_entropy(
                    logits_for_loss, ce_actions.clamp(0, NUM_ACTIONS - 1), reduction="none"
                )
                log_prob = torch.log_softmax(logits_for_loss, dim=1)
                kl_per = torch.sum(
                    soft * (torch.log(soft.clamp_min(1.0e-8)) - log_prob), dim=1
                )
                zero = logits_for_loss.sum() * 0.0
                ce = ce_per[ce_valid].mean() if bool(torch.any(ce_valid)) else zero
                kl = kl_per[soft_valid].mean() if bool(torch.any(soft_valid)) else zero
                loss = float(ce_weight) * ce + float(kl_weight) * kl
            if train:
                scaler.scale(loss).backward()
                if float(grad_clip) > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                scaler.step(optimizer)
                scaler.update()

        with torch.no_grad():
            logits_eval = mask_logits(logits, effective_mask) if loss_mask != "none" else logits
            if bool(torch.any(ce_valid)):
                totals["ce_sum"] += float(ce_per[ce_valid].sum().item())
                totals["ce_samples"] += int(ce_valid.sum().item())
            if bool(torch.any(soft_valid)):
                totals["kl_sum"] += float(kl_per[soft_valid].sum().item())
                totals["kl_samples"] += int(soft_valid.sum().item())

            if bool(torch.any(teacher_valid)):
                teacher_logits = logits_eval[teacher_valid]
                teacher_targets = teacher[teacher_valid]
                k = min(5, int(teacher_logits.shape[1]))
                predictions = teacher_logits.topk(k, dim=1).indices
                totals["teacher_top1_correct"] += int((predictions[:, 0] == teacher_targets).sum().item())
                totals["teacher_top5_correct"] += int(
                    predictions.eq(teacher_targets[:, None]).any(dim=1).sum().item()
                )
                totals["teacher_samples"] += int(teacher_targets.numel())
            if bool(torch.any(behavior_valid)):
                behavior_logits = logits_eval[behavior_valid]
                behavior_targets = behavior[behavior_valid]
                k = min(5, int(behavior_logits.shape[1]))
                predictions = behavior_logits.topk(k, dim=1).indices
                totals["behavior_top1_correct"] += int((predictions[:, 0] == behavior_targets).sum().item())
                totals["behavior_top5_correct"] += int(
                    predictions.eq(behavior_targets[:, None]).any(dim=1).sum().item()
                )
                totals["behavior_samples"] += int(behavior_targets.numel())
        batch_latency.add(time.perf_counter() - batch_started)

    ce_count = max(1, int(totals["ce_samples"]))
    kl_count = max(1, int(totals["kl_samples"]))
    teacher_count = max(1, int(totals["teacher_samples"]))
    behavior_count = max(1, int(totals["behavior_samples"]))
    ce_mean = totals["ce_sum"] / ce_count
    kl_mean = totals["kl_sum"] / kl_count
    action_loss = float(ce_weight) * ce_mean + float(kl_weight) * kl_mean
    latency = batch_latency.summary()
    measured_s = max(1.0e-12, float(sum(batch_latency.values_s)))
    return {
        "loss": action_loss,
        "action_loss": action_loss,
        "ce": ce_mean,
        "kl": kl_mean,
        "teacher_top1": totals["teacher_top1_correct"] / teacher_count,
        "teacher_top5": totals["teacher_top5_correct"] / teacher_count,
        "behavior_top1": totals["behavior_top1_correct"] / behavior_count,
        "behavior_top5": totals["behavior_top5_correct"] / behavior_count,
        "samples": max(int(totals["ce_samples"]), int(totals["kl_samples"])),
        "ce_samples": int(totals["ce_samples"]),
        "kl_samples": int(totals["kl_samples"]),
        "invalid_teacher": int(totals["invalid_teacher"]),
        "invalid_behavior": int(totals["invalid_behavior"]),
        "dropped": int(totals["dropped"]),
        "batch_latency_p50_ms": latency["p50_s"] * 1000.0,
        "batch_latency_p95_ms": latency["p95_s"] * 1000.0,
        "batch_latency_p99_ms": latency["p99_s"] * 1000.0,
        "compute_samples_per_s": processed_samples / measured_s,
    }


def save_checkpoint(
    path: Path, model, epoch: int, args, normalizer: VectorNormalizer,
    train_rows, val_rows, metrics: Dict, mpl_contract_sha256: str,
    provenance: Mapping[str, Any] = None,
) -> None:
    import torch

    saved_args = dict(vars(args))
    for key in (
        "index", "labels", "depth_action_masks", "dataset_cache", "out_dir",
        "tensorboard_log_dir", "val_episodes_from_checkpoint",
    ):
        value = saved_args.get(key)
        if value:
            saved_args[key] = os.path.relpath(
                str(Path(value).expanduser().resolve()), str(path.parent)
            )
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "args": saved_args,
        "model_type": "bc_depth_history" if int(model.depth_channels) > 1 else "bc_soft",
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "policy_input_contract": policy_input_contract(),
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "privileged_runtime_inputs": [],
        **task_contract_fields(
            int(
                (provenance or {}).get(
                    "max_primitive_steps", DEFAULT_MAX_PRIMITIVE_STEPS
                )
            )
        ),
        "max_steps": int(
            (provenance or {}).get(
                "max_primitive_steps", DEFAULT_MAX_PRIMITIVE_STEPS
            )
        ),
        "teacher_label_contract_id": LABEL_CONTRACT_ID,
        "mpl_contract_sha256": str(mpl_contract_sha256),
        "vec_dim": POLICY_VECTOR_DIM,
        "num_actions": NUM_ACTIONS,
        "depth_history_frames": int(model.depth_channels),
        "initial_prev_action": INITIAL_PREV_ACTION,
        "policy_runtime_contract_id": POLICY_RUNTIME_CONTRACT_ID,
        "safety_mask": str(args.deployment_safety_mask),
        "execution_mode": str(args.deployment_execution_mode),
        "feature_mean": normalizer.mean,
        "feature_std": normalizer.std,
        "train_episodes": [int(float(row.get("episode_id", -1))) for row in train_rows],
        "val_episodes": [int(float(row.get("episode_id", -1))) for row in val_rows],
        "train_mission_ids": sorted({mission_id_from_row(row) for row in train_rows}),
        "val_mission_ids": sorted({mission_id_from_row(row) for row in val_rows}),
    }
    if provenance:
        payload.update(dict(provenance))
    torch.save(payload, str(path))


def write_metrics(path: Path, rows: List[Dict]) -> None:
    write_csv_atomic(path, rows, fieldnames=list(rows[0].keys()))


def maybe_plot(out_dir: Path, metrics: List[Dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in metrics]
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [row["train_loss"] for row in metrics], label="train")
        plt.plot(epochs, [row["val_loss"] for row in metrics], label="val")
        plt.xlabel("epoch")
        plt.ylabel("CE + KL")
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(out_dir / "loss_curve.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [row["val_teacher_top1"] for row in metrics], label="teacher top1")
        plt.plot(epochs, [row["val_teacher_top5"] for row in metrics], label="teacher top5")
        plt.plot(epochs, [row["val_behavior_top1"] for row in metrics], label="behavior top1")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(out_dir / "accuracy_curve.png"), dpi=160)
        plt.close()
    except Exception as error:
        print("PLOT_WARNING:", repr(error))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--depth-action-masks", default="",)
    parser.add_argument("--dataset-cache", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_BC_TRAINING_CONFIG.epochs)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument(
        "--episode-ids-file",
        default="",
        help="select canonical episode ids from a newline/comma text or JSON file",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BC_TRAINING_CONFIG.batch_size
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_BC_TRAINING_CONFIG.lr)
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_BC_TRAINING_CONFIG.weight_decay
    )
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=24)
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--depth-history-frames", type=int, default=1)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.3)
    parser.add_argument("--ce-target", choices=["teacher", "behavior"], default="teacher")
    parser.add_argument("--loss-mask", choices=["height", "height_depth", "none"], default="height")
    parser.add_argument("--deployment-safety-mask", choices=SAFETY_MASKS, required=True)
    parser.add_argument("--deployment-execution-mode", choices=EXECUTION_MODES, required=True)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--save-plots", action="store_true", help="write PNG training curves")
    parser.add_argument("--tensorboard-log-dir", default="")
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--val-episodes-from-checkpoint", default="")
    parser.add_argument(
        "--observation-contract",
        default=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        help="require the source rollout/cache observation provenance contract",
    )
    parser.add_argument("--expected-planar-distance", type=float, default=MISSION_PLANAR_DISTANCE_M)
    parser.add_argument("--planar-distance-tolerance", type=float, default=1.0e-3)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    torch, nn, _, Dataset, DataLoader = require_torch()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    seed_everything(args.seed, torch=torch)
    if int(args.cpu_threads) <= 0 or int(args.num_workers) < 0 or int(args.prefetch_factor) <= 0:
        raise ValueError("CPU threads, workers and prefetch factor must be positive/non-negative")
    torch.set_num_threads(int(args.cpu_threads))
    hardware_profile = HardwareProfile(
        cpu_threads=int(args.cpu_threads),
        dataloader_workers=int(args.num_workers),
        dataloader_prefetch_factor=int(args.prefetch_factor),
        pin_memory=bool(device.type == "cuda"),
        persistent_workers=bool(int(args.num_workers) > 0),
        allow_tf32=not bool(args.disable_tf32),
    )
    hardware_runtime = configure_torch_runtime(torch, hardware_profile)

    index_path = Path(args.index).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = "bc-train-{}".format(
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_logger = StructuredRunLogger(
        out_dir / "training.jsonl",
        run_id=run_id,
        component="bc_training",
    )
    rows = [row for row in read_csv(index_path) if parse_bool(row.get("execute_ok", True))]
    validate_mission_rows(rows, args.expected_planar_distance, args.planar_distance_tolerance)
    rows.sort(key=lambda row: int(float(row.get("episode_id", 0))))
    if str(args.episode_ids_file).strip():
        wanted = set(read_episode_ids_file(Path(args.episode_ids_file)))
        available = {int(float(row.get("episode_id", -1))) for row in rows}
        missing = sorted(wanted - available)
        if missing:
            raise ValueError(
                "selected episodes absent from rollout index: {}".format(missing[:10])
            )
        rows = [
            row for row in rows
            if int(float(row.get("episode_id", -1))) in wanted
        ]
    if int(args.max_episodes) > 0:
        rows = rows[:int(args.max_episodes)]
    if len(rows) < 2:
        raise RuntimeError("need at least two accepted episodes")
    if args.val_episodes_from_checkpoint:
        checkpoint_path = Path(args.val_episodes_from_checkpoint).expanduser().resolve()
        try:
            split_checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        except TypeError:
            split_checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        validate_task_contract(
            split_checkpoint,
            expected_max_primitive_steps=int(args.max_steps)
            if hasattr(args, "max_steps")
            else DEFAULT_MAX_PRIMITIVE_STEPS,
            path="validation-split checkpoint task contract",
        )
        train_rows, val_rows = split_rows_from_checkpoint(
            rows,
            split_checkpoint,
            allow_subset=bool(str(args.episode_ids_file).strip()),
            path=str(checkpoint_path),
        )
    else:
        train_rows, val_rows = split_episodes(rows, args.val_ratio, args.seed)
    if int(args.depth_history_frames) <= 0:
        raise ValueError("--depth-history-frames must be positive")
    mpl_contract = MotionPrimitiveLibrary()
    cache_dir = Path(args.dataset_cache).expanduser().resolve() if args.dataset_cache else None
    cache_manifest = None
    cache_manifest_path = None
    if cache_dir is not None:
        cache_manifest_path = cache_dir / "manifest.json"
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    labels = None if cache_dir is not None else TeacherLabelStore(labels_path)
    labels_mpl_hash = str(
        labels.metadata.get("mpl_contract_sha256", "")
        if labels is not None
        else cache_manifest.get("mpl_contract_sha256", "")
    )
    if labels_mpl_hash != str(mpl_contract.contract_sha256):
        raise ValueError(
            "label motion-primitive contract {} != runtime {}".format(
                labels_mpl_hash or "<missing>", mpl_contract.contract_sha256
            )
        )
    depth_masks = (
        DepthActionMaskStore(Path(args.depth_action_masks))
        if args.depth_action_masks and cache_dir is None else None
    )
    if args.loss_mask == "height_depth" and not args.depth_action_masks:
        raise ValueError("--loss-mask height_depth requires --depth-action-masks")
    if cache_dir is not None:
        input_provenance = validate_bc_mmap_provenance(
            cache_manifest,
            expected_observation_contract=args.observation_contract,
            path=str(cache_manifest_path),
        )
        masks_path = Path(args.depth_action_masks).expanduser().resolve() if args.depth_action_masks else None
        train_base = MappedSoftRolloutDataset(
            cache_dir, train_rows, index_path, labels_path, masks_path
        )
        val_base = MappedSoftRolloutDataset(
            cache_dir, val_rows, index_path
        )
        normalizer = VectorNormalizer.fit(train_base.normalizer_values())
    else:
        train_base = SoftRolloutDataset(train_rows, index_path, labels, depth_masks)
        val_base = SoftRolloutDataset(val_rows, index_path, labels, depth_masks)
        normalizer = VectorNormalizer.fit(train_base.continuous)
        source = observation_contract(labels.metadata, path=str(labels.path))
        if source != str(args.observation_contract):
            raise ValueError(
                "observation contract mismatch: received={} expected={}".format(
                    source, args.observation_contract
                )
            )
        if not str(labels.metadata.get("observation_source", "")).strip():
            raise ValueError("label metadata is missing observation source")
        selected_total = len(train_base) + len(val_base)
        input_provenance = {
            "observation_contract": source,
            "observation_source": str(labels.metadata["observation_source"]),
            "reliable_rows": selected_total,
            "legacy_rows": 0,
            "transition_count": selected_total,
            "dataset_manifest_contract_id": "",
            "source_index_sha256": file_sha256(index_path),
            "source_labels_sha256": file_sha256(labels_path),
            "source_depth_masks_sha256": (
                file_sha256(depth_masks.path) if depth_masks is not None else ""
            ),
            "mpl_contract_sha256": str(mpl_contract.contract_sha256),
            **task_contract_fields(
                int(labels.metadata["max_primitive_steps"])
            ),
        }
    train_base.apply_normalizer(normalizer)
    val_base.apply_normalizer(normalizer)

    if cache_dir is not None:
        input_provenance["dataset_manifest_sha256"] = file_sha256(
            cache_manifest_path
        )
    else:
        input_provenance["dataset_manifest_sha256"] = ""
    resolved_training_config = _resolved_training_config(
        args, out_dir, input_provenance
    )
    resolved_training_config_sha256 = canonical_json_sha256(
        resolved_training_config, ensure_ascii=False
    )
    write_json_atomic(out_dir / "resolved_training_config.json", resolved_training_config)
    checkpoint_provenance = {
        **input_provenance,
        "normalizer_sha256": _normalizer_sha256(normalizer),
        "resolved_training_config": resolved_training_config,
        "resolved_training_config_sha256": resolved_training_config_sha256,
    }

    train_dataset = make_torch_dataset(train_base, torch, Dataset, args.depth_history_frames)
    val_dataset = make_torch_dataset(val_base, torch, Dataset, args.depth_history_frames)
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": bool(device.type == "cuda"),
        "persistent_workers": bool(int(args.num_workers) > 0),
        "drop_last": False,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    model = build_model(nn, depth_channels=args.depth_history_frames).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    teacher_hist = train_base.teacher_hist()
    behavior_hist = train_base.behavior_hist()
    print("SOFT_BC_TRAIN_START")
    print("  index:", index_path)
    print("  labels:", labels_path)
    print("  depth_action_masks:", Path(args.depth_action_masks).expanduser().resolve() if args.depth_action_masks else "none")
    print("  dataset_cache:", cache_dir if cache_dir is not None else "memory")
    print("  out_dir:", out_dir)
    print("  train_episodes:", len(train_rows))
    print("  val_episodes:", len(val_rows))
    print("  train_transitions:", len(train_base))
    print("  val_transitions:", len(val_base))
    print("  nonzero_teacher_actions:", int(np.count_nonzero(teacher_hist)), "/", NUM_ACTIONS)
    print("  nonzero_behavior_actions:", int(np.count_nonzero(behavior_hist)), "/", NUM_ACTIONS)
    print("  feature_contract_id:", FEATURE_CONTRACT_ID)
    print("  policy_input_contract_sha256:", policy_input_contract_sha256())
    print("  observation_contract:", input_provenance["observation_contract"])
    print("  observation_source:", input_provenance["observation_source"])
    print("  reliable_rows:", input_provenance["reliable_rows"])
    print("  legacy_rows:", input_provenance["legacy_rows"])
    print("  dataset_manifest_sha256:", input_provenance["dataset_manifest_sha256"])
    print("  resolved_training_config_sha256:", resolved_training_config_sha256)
    print("  privileged_runtime_inputs: []")
    print("  vec_dim:", POLICY_VECTOR_DIM)
    print("  depth_history_frames:", int(args.depth_history_frames))
    print("  initial_prev_action:", INITIAL_PREV_ACTION)
    print("  ce_target:", args.ce_target)
    print("  loss_mask:", args.loss_mask)
    print("  amp:", use_amp)
    print("  device:", device)
    print("  hardware_runtime:", hardware_runtime)

    tensorboard_dir = (
        Path(args.tensorboard_log_dir).expanduser().resolve()
        if args.tensorboard_log_dir else out_dir / "tensorboard"
    )
    writer = None
    if not bool(args.disable_tensorboard):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=str(tensorboard_dir))
        except Exception as error:
            print("TENSORBOARD_WARNING:", repr(error))

    metrics = []
    metrics_path = out_dir / "metrics.csv"
    best_soft_loss = float("inf")
    best_hard_score = -float("inf")
    best_soft_epoch = -1
    best_hard_epoch = -1
    started = time.time()
    run_logger.event(
        "started",
        episode=0,
        train_episodes=len(train_rows),
        val_episodes=len(val_rows),
        train_transitions=len(train_base),
        val_transitions=len(val_base),
        device=str(device),
        observation_contract=input_provenance["observation_contract"],
        observation_source=input_provenance["observation_source"],
        reliable_rows=input_provenance["reliable_rows"],
        legacy_rows=input_provenance["legacy_rows"],
        dataset_manifest_sha256=input_provenance["dataset_manifest_sha256"],
        resolved_training_config_sha256=resolved_training_config_sha256,
    )
    for epoch in range(1, int(args.epochs) + 1):
        epoch_started = time.time()
        train_metrics = compute_loss_and_metrics(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            torch,
            train=True,
            ce_weight=args.ce_weight,
            kl_weight=args.kl_weight,
            ce_target=args.ce_target,
            loss_mask=args.loss_mask,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
        )
        val_metrics = compute_loss_and_metrics(
            model,
            val_loader,
            optimizer,
            scaler,
            device,
            torch,
            train=False,
            ce_weight=args.ce_weight,
            kl_weight=args.kl_weight,
            ce_target=args.ce_target,
            loss_mask=args.loss_mask,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
        )
        row = {"epoch": epoch, "elapsed_s": time.time() - started}
        for prefix, values in (("train", train_metrics), ("val", val_metrics)):
            for key, value in values.items():
                row["{}_{}".format(prefix, key)] = value
        metrics.append(row)
        write_metrics(metrics_path, metrics)
        if writer is not None:
            for key, value in row.items():
                if key != "epoch" and isinstance(value, (int, float)):
                    writer.add_scalar(key.replace("_", "/", 1), value, epoch)
            epoch_elapsed = max(1.0e-9, row["elapsed_s"] - (metrics[-2]["elapsed_s"] if len(metrics) > 1 else 0.0))
            writer.add_scalar("performance/transitions_per_second", (len(train_base) + len(val_base)) / epoch_elapsed, epoch)
            if device.type == "cuda":
                writer.add_scalar("performance/cuda_max_memory_allocated_bytes", torch.cuda.max_memory_allocated(), epoch)
            writer.flush()
        run_logger.event(
            "epoch_completed",
            episode=epoch,
            step=len(train_base) + len(val_base),
            duration_ms=(time.time() - epoch_started) * 1000.0,
            train_loss=row["train_loss"],
            val_action_loss=row["val_action_loss"],
            val_teacher_top1=row["val_teacher_top1"],
            val_teacher_top5=row["val_teacher_top5"],
        )
        print_progress(
            component="bc_training",
            processed=epoch,
            passing=epoch,
            estimated_stop=int(args.epochs),
            elapsed_s=time.time() - started,
            train_loss="{:.4f}".format(row["train_loss"]),
            val_loss="{:.4f}".format(row["val_action_loss"]),
            val_kl="{:.4f}".format(row["val_kl"]),
            teacher_top1="{:.3f}".format(row["val_teacher_top1"]),
            teacher_top5="{:.3f}".format(row["val_teacher_top5"]),
            dropped=row["val_dropped"],
        )

        # Select the deployable action model by its action objective.  The
        # training-only auxiliary head must not redefine "best policy".
        if float(row["val_action_loss"]) < best_soft_loss:
            best_soft_loss = float(row["val_action_loss"])
            best_soft_epoch = epoch
            save_checkpoint(
                out_dir / "checkpoint_best_soft.pt",
                model,
                epoch,
                args,
                normalizer,
                train_rows,
                val_rows,
                row,
                mpl_contract.contract_sha256,
                checkpoint_provenance,
            )
        hard_score = float(row["val_teacher_top1"] + 0.25 * row["val_teacher_top5"])
        if hard_score > best_hard_score:
            best_hard_score = hard_score
            best_hard_epoch = epoch
            save_checkpoint(
                out_dir / "checkpoint_best_hard.pt",
                model,
                epoch,
                args,
                normalizer,
                train_rows,
                val_rows,
                row,
                mpl_contract.contract_sha256,
                checkpoint_provenance,
            )

    save_checkpoint(
        out_dir / "checkpoint_last.pt",
        model,
        int(args.epochs),
        args,
        normalizer,
        train_rows,
        val_rows,
        metrics[-1],
        mpl_contract.contract_sha256,
        checkpoint_provenance,
    )
    shutil.copy2(out_dir / "checkpoint_best_soft.pt", out_dir / "checkpoint_best.pt")
    write_metrics(metrics_path, metrics)
    checkpoint_sha256 = file_sha256(out_dir / "checkpoint_best.pt")
    checkpoint_last_sha256 = file_sha256(out_dir / "checkpoint_last.pt")
    total_val = max(1, len(val_base))
    summary = {
        "train_episodes": len(train_rows),
        "val_episodes": len(val_rows),
        "train_transitions": len(train_base),
        "val_transitions": len(val_base),
        "best_soft_epoch": best_soft_epoch,
        "best_soft_val_action_loss": best_soft_loss,
        "best_soft_val_loss": best_soft_loss,
        "best_hard_epoch": best_hard_epoch,
        "best_hard_score": best_hard_score,
        "last_val_teacher_top1": metrics[-1]["val_teacher_top1"],
        "last_val_teacher_top5": metrics[-1]["val_teacher_top5"],
        "last_val_invalid_teacher_rate": metrics[-1]["val_invalid_teacher"] / total_val,
        "last_val_dropped_rate": metrics[-1]["val_dropped"] / total_val,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "policy_input_contract": policy_input_contract(),
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "privileged_runtime_inputs": [],
        **task_contract_fields(int(input_provenance["max_primitive_steps"])),
        "max_steps": int(input_provenance["max_primitive_steps"]),
        "teacher_label_contract_id": LABEL_CONTRACT_ID,
        "observation_contract": input_provenance["observation_contract"],
        "observation_source": input_provenance["observation_source"],
        "reliable_rows": int(input_provenance["reliable_rows"]),
        "legacy_rows": int(input_provenance["legacy_rows"]),
        "dataset_manifest_contract_id": input_provenance[
            "dataset_manifest_contract_id"
        ],
        "dataset_manifest_sha256": input_provenance["dataset_manifest_sha256"],
        "source_index_sha256": input_provenance["source_index_sha256"],
        "source_labels_sha256": input_provenance["source_labels_sha256"],
        "source_depth_masks_sha256": input_provenance[
            "source_depth_masks_sha256"
        ],
        "mpl_contract_sha256": str(mpl_contract.contract_sha256),
        "normalizer_sha256": checkpoint_provenance["normalizer_sha256"],
        "resolved_training_config": resolved_training_config,
        "resolved_training_config_sha256": resolved_training_config_sha256,
        "vec_dim": POLICY_VECTOR_DIM,
        "depth_history_frames": int(args.depth_history_frames),
        "initial_prev_action": INITIAL_PREV_ACTION,
        "checkpoint_best": "checkpoint_best.pt",
        "checkpoint_best_soft": "checkpoint_best_soft.pt",
        "checkpoint_best_hard": "checkpoint_best_hard.pt",
        "checkpoint_last": "checkpoint_last.pt",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_last_sha256": checkpoint_last_sha256,
        "metrics_csv": metrics_path.name,
        "dataset_mode": "mmap" if cache_dir is not None else "memory",
        "dataset_cache": os.path.relpath(str(cache_dir), str(out_dir)) if cache_dir is not None else "",
        "hardware_runtime": hardware_runtime,
        "tensorboard_enabled": writer is not None,
        "tensorboard_log_dir": os.path.relpath(str(tensorboard_dir), str(out_dir)) if writer is not None else "",
    }
    if args.loss_mask == "height_depth":
        # The privileged global teacher may nominate an action that the current
        # local depth screen cannot verify. KL still supervises any remaining
        # local support, so this is a data property rather than a train failure.
        quality_ok = bool(
            best_soft_epoch > 0 and metrics[-1]["val_kl_samples"] > 0
        )
    else:
        quality_ok = bool(
            summary["last_val_invalid_teacher_rate"] <= 1.0e-3
            and best_soft_epoch > 0
        )
    summary["quality_pass"] = quality_ok
    write_json_atomic(out_dir / "summary.json", summary)
    if bool(args.save_plots):
        maybe_plot(out_dir, metrics)
    if writer is not None:
        writer.close()
    run_logger.event(
        "completed" if quality_ok else "quality_rejected",
        episode=int(args.epochs),
        step=len(train_base) + len(val_base),
        best_soft_epoch=best_soft_epoch,
        best_soft_val_action_loss=best_soft_loss,
        best_hard_epoch=best_hard_epoch,
        quality_pass=quality_ok,
        observation_contract=input_provenance["observation_contract"],
        observation_source=input_provenance["observation_source"],
        reliable_rows=input_provenance["reliable_rows"],
        legacy_rows=input_provenance["legacy_rows"],
        dataset_manifest_sha256=input_provenance["dataset_manifest_sha256"],
        checkpoint_sha256=checkpoint_sha256,
        resolved_training_config_sha256=resolved_training_config_sha256,
    )
    run_logger.close()
    print("SOFT_BC_TRAIN_SUMMARY")
    for key, value in summary.items():
        print("  {}: {}".format(key, value))
    print("RESULT={}".format("PASS" if quality_ok else "FAIL"))
    return 0 if quality_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
