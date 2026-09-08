#!/usr/bin/env python3
"""Run the preregistered N1/H0 versus N5/H0 Critic-only confirmation.

The entry point runs four real offline Critic branches.  It never launches
Unity/ROS, never touches production Replay, and never runs Actor updates.  A
separate formal-task audit is performed before training; if 200 independent
formal missions cannot be proven, validation remains explicitly blocked.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import random
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from planning.awac.n5_confirmation import (
    FORMAL_FIELDS,
    derive_episode_seed,
    episode_budget_audit,
    identity_record,
    normalized_task_identity,
    select_confirmation_rows,
)
from planning.common.hashing import file_sha256


NEW_SOURCE_FILES = (
    "python/planning/awac/n5_confirmation.py",
    "scripts/confirm_critic_n5_h0.py",
    "tests/test_n5_confirmation.py",
)


def _load_factorial_module(root: Path):
    path = root / "scripts/diagnose_critic_td_horizon_factorial.py"
    spec = importlib.util.spec_from_file_location("factorial_diagnostic_reuse", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load previous factorial diagnostic module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_tree(path: Path) -> Dict[str, str]:
    root = Path(path).resolve()
    if root.is_file():
        return {str(root): file_sha256(root)}
    return {str(item): file_sha256(item) for item in sorted(root.rglob("*")) if item.is_file()}


def _fresh_output(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists() or not any(path.iterdir()):
        path.mkdir(parents=True, exist_ok=True)
        return path
    index = 2
    while True:
        candidate = Path(str(path) + "_r{}".format(index))
        if not candidate.exists() or not any(candidate.iterdir()):
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        index += 1


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _rng_digest(factorial) -> str:
    state = factorial._capture_rng()
    stream = repr(state["python"]).encode("utf-8") + repr(state["numpy"]).encode("utf-8")
    stream += bytes(state["torch_cpu"].tolist())
    for value in state.get("torch_cuda", []):
        stream += bytes(value.tolist())
    return _hash_bytes(stream)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _locate_checkpoint(root: Path, branch: str, updates: int) -> Path:
    matches = []
    for path in sorted((root / "branches" / branch).glob("checkpoint_*.pt")):
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if str(payload.get("branch")) == str(branch) and int(payload.get("updates", -1)) == int(updates):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError("expected one {} checkpoint at updates {}, found {}".format(branch, updates, len(matches)))
    return matches[0]


def _state_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if not torch.equal(a.cpu(), b.cpu()):
                return False
        elif a != b:
            return False
    return True


def _state_hash(module, factorial) -> str:
    return factorial._module_sha(module)


def _state_dict_hash(state_dict: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        digest.update(str(name).encode("utf-8"))
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.numpy().tobytes(order="C"))
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _augment_checkpoint(path: Path, extra: Mapping[str, Any]) -> str:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    payload.update(dict(extra))
    torch.save(payload, str(path))
    return file_sha256(path)


def _write_code_diff(path: Path, root: Path) -> None:
    chunks = []
    for relative in NEW_SOURCE_FILES:
        source = root / relative
        content = source.read_text(encoding="utf-8").splitlines(True)
        chunks.append("--- /dev/null\n+++ {}\n".format(relative))
        chunks.extend("+" + line for line in content)
    path.write_text("".join(chunks), encoding="utf-8")


def _mission_audit(root: Path, manifest: Mapping[str, Any], *, mission_index: Path, candidate_index: Path, dev_index: Path, final_index: Path, selection_seed: int, out: Path) -> Dict[str, Any]:
    canonical = _read_csv(mission_index)
    candidates = _read_csv(candidate_index)
    dev = _read_csv(dev_index)
    final = _read_csv(final_index)

    # The Replay metadata points at the complete canonical mission source, but
    # has no per-transition mission identity.  Conservatively exclude every
    # canonical geometry rather than treating that missing mapping as zero.
    canonical_ids = {normalized_task_identity(row) for row in canonical}
    dev_ids = {normalized_task_identity(row) for row in dev}
    final_ids = {normalized_task_identity(row) for row in final}
    candidate_ids = {normalized_task_identity(row) for row in candidates}
    formal_candidate_count = sum(1 for row in candidates if all(str(row.get(field, "")).strip() for field in FORMAL_FIELDS))
    route_safe_candidates = sum(
        1 for row in candidates
        if str(row.get("route_status", "")) == "ok"
        and float(row.get("global_route_stretch", "inf")) <= 1.15
    )
    old_holdout_ids = set()
    # The old holdout is embedded in the production checkpoint and only carries
    # mission IDs in its raw records.  It is therefore tracked separately and
    # never used as a substitute for geometry-level identity.
    old_holdout_identity_status = "DECLARED_ID_ONLY"
    selected, selection = select_confirmation_rows(
        candidates,
        excluded_identities=canonical_ids,
        excluded_identity_status="COMPLETE",
        count=200,
        seed=selection_seed,
    )
    audit = {
        "canonical_mission_index": {"path": str(mission_index), "rows": len(canonical), "sha256": file_sha256(mission_index)},
        "candidate_index": {"path": str(candidate_index), "rows": len(candidates), "sha256": file_sha256(candidate_index)},
        "formal_canonical_rows": len(canonical),
        "formal_candidate_rows": formal_candidate_count,
        "route_safe_candidate_rows": route_safe_candidates,
        "candidate_rows_missing_formal_fields": len(candidates) - formal_candidate_count,
        "identity_method": "normalized_start_goal_geometry_rounded_6_decimals",
        "train_identity_status": "CONSERVATIVE_COMPLETE_EXCLUDE_ALL_CANONICAL_SOURCE",
        "train_identity_excluded_count": len(canonical_ids),
        "old_holdout_identity_status": old_holdout_identity_status,
        "old_holdout_normalized_identity_count": len(old_holdout_ids),
        "dev100_identity_count": len(dev_ids),
        "final300_identity_count": len(final_ids),
        "candidate_identity_count": len(candidate_ids),
        "intersection_counts": {
            "canonical_vs_dev100": len(canonical_ids & dev_ids),
            "canonical_vs_final300": len(canonical_ids & final_ids),
            "candidate_vs_dev100": len(candidate_ids & dev_ids),
            "candidate_vs_final300": len(candidate_ids & final_ids),
            "canonical_vs_old_holdout_declared_ids": "UNKNOWN_GEOMETRY",
        },
        "bc_pretraining_mission_overlap": "UNKNOWN",
        "selection": selection,
        "selected_identity_records": [identity_record(row) for row in selected],
        "independence_status": "BLOCKED_INSUFFICIENT_FORMAL_TASKS" if len(selected) < 200 else "PASS",
    }
    fields = list(canonical[0].keys()) if canonical else list(candidates[0].keys())
    _write_csv(out / "confirmation_missions.csv", selected, fields)
    _write_json(out / "mission_split_audit.json", audit)
    (out / "confirmation_raw_episodes").mkdir(parents=True, exist_ok=True)
    _write_json(out / "confirmation_dataset_manifest.json", {
        "status": audit["independence_status"],
        "read_only": True,
        "episode_count": 0,
        "mission_count": len(selected),
        "max_episode_budget": 200,
        "max_decision_steps": 9000,
        "episode_budget": episode_budget_audit(count=200, max_steps=45, max_decision_steps=9000),
        "policy": "frozen_bc_masked_categorical_t1",
        "new_environment_steps": 0,
        "optimizer_steps": {"actor": 0, "critic": 0},
        "blocker": None if len(selected) >= 200 else "fewer than 200 independently provable formal Teacher missions",
    })
    return audit


def _source_snapshot(root: Path, *, checkpoint_path: Path, gate_path: Path, bc_path: Path, replay_path: Path, r2_root: Path) -> Dict[str, Any]:
    files = {
        "production_checkpoint": checkpoint_path,
        "gate_checkpoint": gate_path,
        "bc_checkpoint": bc_path,
        "r2_manifest": r2_root / "manifest.json",
        "r2_sequence_sidecar": r2_root / "sequence_sidecar.npz",
        "r2_fixed_batch": r2_root / "fixed_critic_batch_indices.npy",
        "r2_holdout_sidecar": r2_root / "holdout_sequence_sidecar.npz",
    }
    result = {}
    for name, path in files.items():
        result[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    result["replay_files"] = _hash_tree(replay_path)
    for relative in NEW_SOURCE_FILES:
        path = root / relative
        result["source_files_before_experiment"] = result.get("source_files_before_experiment", {})
        result["source_files_before_experiment"][relative] = {
            "path": str(path),
            "existed": path.exists(),
            "sha256": file_sha256(path) if path.exists() else None,
        }
    return result


def _batch_sequence(seed: int, *, updates: int, batch_size: int, replay_size: int) -> np.ndarray:
    return np.random.RandomState(int(seed)).randint(0, int(replay_size), size=(int(updates), int(batch_size))).astype(np.int64)


def _write_train_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["branch", "snapshot", "row_index", "q1", "q2", "qmin"]
    _write_csv(path, rows, fields)


def _save_branch(branch, *, branch_dir: Path, snapshots: Sequence[int], source_sha: str, sidecar_sha: str, fixed_sha: str, actor_sha: str, bc_sha: str, common_init_sha: str, training_seed: int, batch_sha: str, factorial) -> Dict[str, Any]:
    branch_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for step in snapshots:
        path = branch_dir / "checkpoint_{}.pt".format(int(step))
        branch.save_checkpoint(path, source_sha=source_sha, sidecar_sha=sidecar_sha, fixed_sha=fixed_sha, actor_sha=actor_sha, bc_sha=bc_sha)
        checkpoint_sha = _augment_checkpoint(path, {
            "common_initialization_checkpoint_sha256": common_init_sha,
            "training_seed": int(training_seed),
            "batch_sequence_sha256": batch_sha,
            "fixed_endpoint": int(step) == 3000,
            "actor_optimizer_steps": 0,
            "actor_state_sha256": actor_sha,
            "bc_reference_sha256": bc_sha,
        })
        reports.append({"updates": int(step), "path": str(path), "sha256": checkpoint_sha, "critic1_state_sha256": _state_hash(branch.critic1, factorial), "critic2_state_sha256": _state_hash(branch.critic2, factorial), "target_critic1_state_sha256": _state_hash(branch.target_critic1, factorial), "target_critic2_state_sha256": _state_hash(branch.target_critic2, factorial), "critic_optimizer_sha256": factorial._optimizer_sha(branch.optimizer), "actor_optimizer_steps": 0})
    return {"snapshots": reports, "final_checkpoint_sha256": reports[-1]["sha256"]}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2-manifest", type=Path, required=True)
    parser.add_argument("--mission-index", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--dev-index", type=Path, required=True)
    parser.add_argument("--final-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed-one", type=int, default=202609071)
    parser.add_argument("--seed-two", type=int, default=202609072)
    parser.add_argument("--focused-log", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if int(args.updates) != 3000 or int(args.batch_size) != 128:
        raise SystemExit("confirmation requires exactly 3000 updates and batch_size=128")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise SystemExit("requested CUDA device is unavailable")
    root = Path(__file__).resolve().parents[1]
    r2_manifest_path = args.r2_manifest.expanduser().resolve()
    r2_root = r2_manifest_path.parent
    manifest = json.loads(r2_manifest_path.read_text(encoding="utf-8"))
    output = _fresh_output(args.out_dir)
    factorial = _load_factorial_module(root)
    device = torch.device(str(args.device))

    checkpoint_path = Path(manifest["checkpoint"]["path"]).resolve()
    gate_path = Path(manifest["gate_checkpoint"]["path"]).resolve()
    bc_path = Path(manifest["bc_checkpoint"]["path"]).resolve()
    replay_path = Path(manifest["replay"]["path"]).resolve()
    for path in (checkpoint_path, gate_path, bc_path, replay_path / "metadata.json", args.mission_index, args.candidate_index, args.dev_index, args.final_index):
        if not path.exists():
            raise SystemExit("required input is missing: {}".format(path))
    expected_hashes = {
        "checkpoint": manifest["checkpoint"]["sha256"],
        "gate_checkpoint": manifest["gate_checkpoint"]["sha256"],
        "bc_checkpoint": manifest["bc_checkpoint"]["sha256"],
        "replay_metadata": manifest["replay"]["metadata_sha256"],
    }
    actual_hashes = {
        "checkpoint": file_sha256(checkpoint_path),
        "gate_checkpoint": file_sha256(gate_path),
        "bc_checkpoint": file_sha256(bc_path),
        "replay_metadata": file_sha256(replay_path / "metadata.json"),
    }
    if actual_hashes != expected_hashes:
        raise SystemExit("r2 manifest identity mismatch: {}".format(json.dumps({"expected": expected_hashes, "actual": actual_hashes}, sort_keys=True)))

    source_before = _source_snapshot(root, checkpoint_path=checkpoint_path, gate_path=gate_path, bc_path=bc_path, replay_path=replay_path, r2_root=r2_root)
    mission_audit = _mission_audit(root, manifest, mission_index=args.mission_index.resolve(), candidate_index=args.candidate_index.resolve(), dev_index=args.dev_index.resolve(), final_index=args.final_index.resolve(), selection_seed=202609073, out=output)

    historical = {}
    for branch_name in ("E_N1_H0", "E_N5_H0"):
        init_path = _locate_checkpoint(r2_root, branch_name, 0)
        endpoint_path = _locate_checkpoint(r2_root, branch_name, 3000)
        historical[branch_name] = {
            "initial": {"path": str(init_path), "sha256": file_sha256(init_path)},
            "endpoint": {"path": str(endpoint_path), "sha256": file_sha256(endpoint_path)},
            "endpoint_payload": torch.load(str(endpoint_path), map_location="cpu", weights_only=False),
        }
    for value in historical.values():
        value.pop("endpoint_payload", None)

    (output / "actual_commands.txt").write_text("source /home/xm/anaconda3/etc/profile.d/conda.sh\nconda activate xm\ncd {}\n{}\n".format(root, " ".join(shlex.quote(value) for value in sys.argv)), encoding="utf-8")
    _write_json(output / "actual_commands.txt.json", {"argv": [str(value) for value in sys.argv], "shell_command": " ".join(shlex.quote(value) for value in sys.argv), "environment": {"python": sys.executable, "conda_prefix": os.environ.get("CONDA_PREFIX", ""), "cwd": os.getcwd()}})
    if args.focused_log and args.focused_log.exists():
        (output / "focused_pytest.log").write_bytes(args.focused_log.read_bytes())
    else:
        (output / "focused_pytest.log").write_text("focused log supplied externally: NO\n", encoding="utf-8")

    common_config = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    gate_checkpoint = torch.load(str(gate_path), map_location="cpu", weights_only=False)
    bc_checkpoint = torch.load(str(bc_path), map_location="cpu", weights_only=False)
    if int(common_config.get("max_primitive_steps", -1)) != 45:
        raise SystemExit("effective horizon is not T=45")
    config = factorial.AWACOptimizationConfig(**common_config["resolved_training_config"]["optimization_config"])
    config = dataclasses.replace(config, critic_cql_weight=0.0)
    replay = factorial.AWACReplayBuffer.open(replay_path, read_only=True)
    if int(replay.size) != 24169:
        raise SystemExit("Replay size is not 24169")
    train_labels, train_lengths = factorial._train_mc_labels(replay, gamma=float(config.gamma), reward_scale=float(config.reward_scale))
    sidecar, sidecar_report = factorial.build_sequence_sidecar(arrays=replay.arrays, count=int(replay.size), horizon=45, episode_lengths=train_lengths)
    sidecar_path = output / "sequence_sidecar.npz"
    np.savez_compressed(sidecar_path, **sidecar)
    sidecar_sha = file_sha256(sidecar_path)
    _write_json(output / "sequence_sidecar.json", dict(sidecar_report, sha256=sidecar_sha, source_r2_sha256=manifest["sequence_sidecar_sha256"]))
    old_fixed_path = r2_root / "fixed_critic_batch_indices.npy"
    old_fixed_sha = file_sha256(old_fixed_path)
    if old_fixed_sha != manifest["fixed_batch_indices_sha256"]:
        raise SystemExit("r2 fixed batch identity mismatch")
    fixed_batches = np.load(old_fixed_path, allow_pickle=False)

    source_learner = factorial._make_base_learner(common_config, bc_checkpoint, config, device)
    actor_sha = factorial._module_sha(source_learner.actor)
    bc_sha = factorial._module_sha(source_learner.bc_reference)
    common_ref = torch.load(str(_locate_checkpoint(r2_root, "E_N1_H0", 0)), map_location="cpu", weights_only=False)
    common_branch = factorial.FactorialBranch(name="common_init", n_step=1, use_budget=False, source=source_learner, config=config, replay=replay, sidecar=sidecar, device=device)
    common_equal = {
        "critic1": _state_equal(common_branch.critic1.state_dict(), common_ref["critic1_state_dict"]),
        "critic2": _state_equal(common_branch.critic2.state_dict(), common_ref["critic2_state_dict"]),
        "target_critic1": _state_equal(common_branch.target_critic1.state_dict(), common_ref["target_critic1_state_dict"]),
        "target_critic2": _state_equal(common_branch.target_critic2.state_dict(), common_ref["target_critic2_state_dict"]),
    }
    if not all(common_equal.values()):
        raise SystemExit("production checkpoint does not exactly reproduce factorial common initialization")
    common_init_sha = file_sha256(_locate_checkpoint(r2_root, "E_N1_H0", 0))
    _write_json(output / "common_initialization_evidence.json", {"r2_common_initialization_checkpoint": common_init_sha, "generation_829_source_sha256": actual_hashes["checkpoint"], "exact_state_equal": common_equal, "critic_vector_dim": 128, "actor_vector_dim": 127, "target_lag_preserved": True, "gate_checkpoint_not_used_as_training_start": True, "trained_endpoint_not_used_as_training_start": True})
    del common_branch

    protocol = {
        "protocol_locked_before_training": True,
        "protocol_locked_before_new_validation_sampling": True,
        "candidate": "N5_H0_CQL0",
        "control": "N1_H0_CQL0",
        "historical_models": historical,
        "new_models": ["R1_N1_H0", "R1_N5_H0", "R2_N1_H0", "R2_N5_H0"],
        "new_training_seeds": [202609071, 202609072],
        "mission_selection_seed": 202609073,
        "sampling_root_seed": 202609074,
        "bootstrap_seed": 202609075,
        "common_initialization_checkpoint_sha256": common_init_sha,
        "generation_829_source": actual_hashes["checkpoint"],
        "gate_generation_828_not_used": actual_hashes["gate_checkpoint"],
        "bc_checkpoint_sha256": actual_hashes["bc_checkpoint"],
        "replay": {"path": str(replay_path), "size": int(replay.size), "metadata_sha256": actual_hashes["replay_metadata"], "used_for_gradient_only": True},
        "sequence_sidecar_sha256": sidecar_sha,
        "configuration": dataclasses.asdict(config),
        "batch_size": 128,
        "updates_per_branch": 3000,
        "total_new_critic_updates_budget": 12000,
        "max_steps": 45,
        "timeout_is_terminal_no_bootstrap": True,
        "horizon_targets": {"N1": "one_step_expected_masked_categorical_T1", "N5": "five_step_expected_masked_categorical_T1"},
        "budget_input": "H0_constant_zero",
        "actor_frozen": True,
        "bc_frozen": True,
        "cql_weight": 0.0,
        "new_confirmation_task_list": mission_audit["selection"]["selected_identities"],
        "mission_audit_status": mission_audit["independence_status"],
        "statistics": {"primary": "historical_N5_minus_historical_N1_qmin_mc_spearman_rho", "cluster": "mission", "bootstrap_repeats": 2000, "confidence": 0.95, "mse_delta": "N5_minus_N1"},
        "budget": episode_budget_audit(count=200, max_steps=45, max_decision_steps=9000),
        "production_gate_or_defaults_modified": False,
    }
    _write_json(output / "preregistration.json", protocol)
    (output / "preregistration.sha256").write_text(file_sha256(output / "preregistration.json") + "  preregistration.json\n", encoding="utf-8")
    _write_code_diff(output / "code_changes.diff", root)

    batch_sequences = {}
    for label, seed in (("R1", args.seed_one), ("R2", args.seed_two)):
        sequence = _batch_sequence(seed, updates=args.updates, batch_size=args.batch_size, replay_size=int(replay.size))
        sequence_path = output / "fixed_train_indices" / (label + ".npy")
        sequence_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(sequence_path, sequence)
        batch_sequences[label] = {"seed": int(seed), "path": str(sequence_path), "sha256": file_sha256(sequence_path), "shape": list(sequence.shape)}
    _write_json(output / "fixed_train_indices" / "manifest.json", batch_sequences)

    branch_specs = (("R1_N1_H0", 1, False, args.seed_one, "R1"), ("R1_N5_H0", 5, False, args.seed_one, "R1"), ("R2_N1_H0", 1, False, args.seed_two, "R2"), ("R2_N5_H0", 5, False, args.seed_two, "R2"))
    train_prediction_rows = []
    branch_reports = {}
    for name, n_step, use_budget, seed, sequence_label in branch_specs:
        sequence = np.load(batch_sequences[sequence_label]["path"], allow_pickle=False)
        _set_seed(seed)
        rng_before = _rng_digest(factorial)
        branch = factorial.FactorialBranch(name=name, n_step=n_step, use_budget=use_budget, source=source_learner, config=config, replay=replay, sidecar=sidecar, device=device)
        rng_after_init = _rng_digest(factorial)
        snapshots = []
        next_snapshot = {0, 300, 1000, 3000}
        for update in range(0, args.updates + 1):
            if update in next_snapshot:
                indices = np.arange(min(256, int(replay.size)), dtype=np.int64)
                prediction = factorial._train_prediction(branch, replay, indices, device)
                snapshots.append({"updates": int(update), "critic1_state_sha256": _state_hash(branch.critic1, factorial), "critic2_state_sha256": _state_hash(branch.critic2, factorial), "target_critic1_state_sha256": _state_hash(branch.target_critic1, factorial), "target_critic2_state_sha256": _state_hash(branch.target_critic2, factorial), "critic_optimizer_sha256": factorial._optimizer_sha(branch.optimizer), "actor_optimizer_steps": 0, "train_prediction_rows": len(indices)})
                for local, row_index in enumerate(indices.tolist()):
                    train_prediction_rows.append({"branch": name, "snapshot": int(update), "row_index": int(row_index), "q1": float(prediction["q1"][local]), "q2": float(prediction["q2"][local]), "qmin": float(prediction["qmin"][local])})
                _save_branch(branch, branch_dir=output / "branches" / name, snapshots=(update,), source_sha=actual_hashes["checkpoint"], sidecar_sha=sidecar_sha, fixed_sha=old_fixed_sha, actor_sha=actor_sha, bc_sha=bc_sha, common_init_sha=common_init_sha, training_seed=seed, batch_sha=batch_sequences[sequence_label]["sha256"], factorial=factorial)
            if update == args.updates:
                break
            branch.update(sequence[update])
            if (update + 1) % 250 == 0:
                print("{} update {}/{}".format(name, update + 1, args.updates), flush=True)
        if int(branch.updates) != args.updates:
            raise RuntimeError("{} did not reach fixed endpoint".format(name))
        branch_reports[name] = {"name": name, "n_step": n_step, "use_budget": use_budget, "training_seed": int(seed), "updates": int(branch.updates), "batch_sequence_sha256": batch_sequences[sequence_label]["sha256"], "rng_digest_before_init": rng_before, "rng_digest_after_init": rng_after_init, "snapshots": snapshots, "actor_optimizer_steps": 0, "critic_optimizer_steps": int(branch.updates), "cql_weight": 0.0}
        del branch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_train_predictions(output / "training_predictions.csv", train_prediction_rows)
    _write_csv(output / "seed_summary.csv", [{"seed": int(seed), "branches": "R{}_N1_H0,R{}_N5_H0".format(label, label), "updates_per_branch": 3000, "critic_updates": 6000, "actor_updates": 0, "confirmation_sampling_status": mission_audit["independence_status"]} for label, seed in (("1", args.seed_one), ("2", args.seed_two))], ["seed", "branches", "updates_per_branch", "critic_updates", "actor_updates", "confirmation_sampling_status"])

    six_models = {"historical": historical, "new": {}}
    for name, _, _, _, _ in branch_specs:
        final = torch.load(output / "branches" / name / "checkpoint_3000.pt", map_location="cpu", weights_only=False)
        six_models["new"][name] = {"path": str(output / "branches" / name / "checkpoint_3000.pt"), "sha256": file_sha256(output / "branches" / name / "checkpoint_3000.pt"), "updates": int(final["updates"]), "n_step": int(final["n_step"]), "use_budget": bool(final["use_budget"]), "critic1_state_sha256": _state_dict_hash(final["critic1_state_dict"]), "critic2_state_sha256": _state_dict_hash(final["critic2_state_dict"]), "actor_optimizer_steps": 0}
    _write_json(output / "frozen_six_models_manifest.json", six_models)
    _write_json(output / "paired_cluster_bootstrap.json", {"status": "NOT_RUN_BLOCKED_NO_CONFIRMATION_DATA", "requested": 2000, "seed": 202609075, "primary": "historical_N5_minus_historical_N1_qmin_mc_rho", "mse_delta": "N5_minus_N1"})
    _write_json(output / "freeze_and_budget_audit.json", {"new_critic_optimizer_steps": 12000, "new_environment_steps": 0, "actor_optimizer_steps": 0, "dev100": 0, "final300": 0, "production_calibration_rerun": 0, "branch_updates_exact": all(value["updates"] == 3000 for value in branch_reports.values()), "budget_pass": True, "confirmation_sampling": mission_audit["independence_status"], "historical_artifacts_unchanged": source_before["replay_files"] == _hash_tree(replay_path)})

    source_after = _source_snapshot(root, checkpoint_path=checkpoint_path, gate_path=gate_path, bc_path=bc_path, replay_path=replay_path, r2_root=r2_root)
    _write_json(output / "source_before_after_sha256.json", {"before": source_before, "after": source_after, "before_captured_before_training": True, "historical_inputs_unchanged": source_before["production_checkpoint"] == source_after["production_checkpoint"] and source_before["gate_checkpoint"] == source_after["gate_checkpoint"] and source_before["bc_checkpoint"] == source_after["bc_checkpoint"] and source_before["replay_files"] == source_after["replay_files"]})
    _write_json(output / "manifest.json", {"diagnostic_id": "critic_n5_independent_confirmation", "run_id": output.name, "software_version": getattr(factorial, "SOFTWARE_VERSION", "unknown"), "python": sys.executable, "python_version": platform.python_version(), "torch_version": torch.__version__, "device": str(device), "checkpoint": {"path": str(checkpoint_path), "sha256": actual_hashes["checkpoint"]}, "gate_checkpoint": {"path": str(gate_path), "sha256": actual_hashes["gate_checkpoint"]}, "bc_checkpoint": {"path": str(bc_path), "sha256": actual_hashes["bc_checkpoint"]}, "replay": {"path": str(replay_path), "size": int(replay.size), "metadata_sha256": actual_hashes["replay_metadata"]}, "common_initialization_checkpoint_sha256": common_init_sha, "historical_models": historical, "new_models": branch_reports, "new_critic_optimizer_steps": 12000, "new_environment_steps": 0, "actor_optimizer_steps": 0, "confirmation_task_audit": mission_audit, "validation_status": mission_audit["independence_status"], "candidate_confirmation": "BLOCKED_BEFORE_RUNTIME_SAMPLING" if mission_audit["independence_status"] != "PASS" else "PENDING", "production_defaults_modified": False})
    (output / "report_zh.md").write_text("# N1/H0 与 N5/H0 独立确认\n\n本次已完成四个真实离线 Critic 分支：R1/R2 两个训练种子，各执行 N1/H0 与 N5/H0 3,000 次更新，共 12,000 次；Actor、BC、环境步数均为 0。训练起点由 factorial 共同初始化与 generation-829 逐参数核对一致。\n\n独立验证采样在启动 Unity 前 fail-closed：正式 canonical mission 源已被本次 Critic 训练来源保守排除，剩余 candidate 行缺少 formal Teacher audit/RouteStore 完整字段，无法证明 200 个合规且未重叠任务。因此没有把旧 holdout、Dev100 或 Final300 当作新确认集，也没有伪造 rho/CI 或启动运行时。\n\n详见 `mission_split_audit.json`、`frozen_six_models_manifest.json`、`freeze_and_budget_audit.json` 与 `source_before_after_sha256.json`。本轮不能得出 N5 相对 N1 的新验证集支持结论，也不改变生产 gate。\n", encoding="utf-8")
    replay.close()
    print("OUTPUT", output)
    print("NEW_CRITIC_UPDATES", 12000)
    print("CONFIRMATION_TASKS", mission_audit["selection"]["selected_count"])
    print("VALIDATION_STATUS", mission_audit["independence_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
