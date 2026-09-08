#!/usr/bin/env python3
"""Collect a bounded diagnostic multi-action replay from the Unity runtime.

The collector is intentionally not a training or production replay entry
point.  For every selected failed-BC state it replays the recorded prefix
from a fresh reset and executes each legal candidate once.  Candidate rows
are written only to the separate ``multi_action_replay_v1`` artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from planning.diagnostics.multi_action_replay import (
    audit_multi_action_replay,
    plan_action_sources,
    write_multi_action_replay,
)


OBSERVATION_CONTRACT = "reliable_exact_endpoint_snapshot"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(path: Path, *, suffixes: Optional[Sequence[str]] = None) -> str:
    root = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    allowed = set(suffixes or ())
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if allowed and item.suffix not in allowed:
            continue
        digest.update(str(item.relative_to(root)).encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(_jsonable(payload), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: Any, *, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base).expanduser().resolve() / path).resolve()


def _load_missions(path: Path) -> Dict[str, Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = {str(row["mission_id"]): dict(row) for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError("mission index is empty: {}".format(path))
    return rows


def _load_failed_targets(
    bc_index: Path,
    bc_dir: Path,
    missions: Mapping[str, Mapping[str, str]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Select terminal-adjacent states from failed BC episodes only."""

    failed = []
    with Path(bc_index).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if _bool(row.get("success", False)):
                continue
            mission_id = str(row.get("mission_id", "")).strip()
            if mission_id not in missions:
                raise ValueError("failed mission missing from mission index: {}".format(mission_id))
            failed.append(dict(row))
    failed.sort(key=lambda row: (int(float(row.get("episode_id", -1))), str(row.get("mission_id", ""))))
    targets: List[Dict[str, Any]] = []
    for row in failed:
        raw_steps = str(row.get("step_csv", "")).strip()
        if not raw_steps:
            raise ValueError("failed BC row has no step_csv: {}".format(row.get("episode_id", "")))
        step_path = _resolve_path(raw_steps, base=Path(bc_index).parent)
        if not step_path.is_file():
            step_path = _resolve_path(raw_steps, base=bc_dir)
        if not step_path.is_file():
            raise FileNotFoundError("BC step CSV is missing: {}".format(step_path))
        with step_path.open(newline="", encoding="utf-8") as handle:
            steps = list(csv.DictReader(handle))
        steps.sort(key=lambda item: int(item["step"]))
        mission_id = str(row["mission_id"])
        for index in range(len(steps) - 1, -1, -1):
            item = steps[index]
            targets.append(
                {
                    "mission_id": mission_id,
                    "source_episode_id": str(row.get("episode_id", "")),
                    "source_failure_type": str(row.get("stop_reason", "")),
                    "step_id": int(item["step"]),
                    "recorded_action": int(item["action"]),
                    "recorded_position": [
                        float(item["x_before"]),
                        float(item["y_before"]),
                        float(item["z_before"]),
                    ],
                    "prefix_actions": [int(value["action"]) for value in steps[:index]],
                    "step_path": str(step_path),
                    "mission": dict(missions[mission_id]),
                    "route_id": str(
                        row.get("route_id")
                        or missions[mission_id].get("route_id")
                        or missions[mission_id].get("route_identity")
                        or "mission:{}".format(mission_id)
                    ),
                }
            )
            if len(targets) >= int(limit):
                return targets
    if len(targets) < int(limit):
        raise ValueError(
            "only {} failed BC states are available; {} required".format(
                len(targets), int(limit)
            )
        )
    return targets


def _state_content_hash(obs: Mapping[str, Any], mask: np.ndarray, previous_action: int) -> str:
    from planning.contracts.feature import obs_continuous_vector

    digest = hashlib.sha256()
    arrays = (
        np.ascontiguousarray(obs_continuous_vector(dict(obs)), dtype=np.float32),
        np.ascontiguousarray(np.asarray(obs["depth"], dtype=np.float16)),
        np.ascontiguousarray(np.asarray(mask, dtype=np.uint8).reshape(-1)),
        np.asarray([int(previous_action)], dtype=np.int64),
    )
    for array in arrays:
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _policy_vector(obs: Mapping[str, Any], previous_action: int, normalizer) -> np.ndarray:
    from planning.contracts.feature import action_onehot, obs_continuous_vector

    continuous = normalizer.transform_continuous(obs_continuous_vector(dict(obs)))
    return np.concatenate(
        (continuous, action_onehot(int(previous_action))), axis=0
    ).astype(np.float32, copy=False)


def _load_bc_actor(checkpoint_path: Path, *, torch_threads: int):
    import torch
    from planning.bc.model import VectorNormalizer, build_model, mask_logits

    torch.set_num_threads(max(1, int(torch_threads)))
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    actor = build_model(
        torch.nn,
        vec_dim=int(checkpoint.get("vec_dim", 127)),
        num_actions=int(checkpoint.get("num_actions", 105)),
        depth_channels=int(checkpoint.get("depth_history_frames", 1)),
    ).to(torch.device("cpu"))
    actor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actor.eval()
    return torch, actor, VectorNormalizer.from_checkpoint(checkpoint), checkpoint, mask_logits


def _bc_action(obs, mask, previous_action, actor, normalizer, torch, mask_logits) -> int:
    vector = _policy_vector(obs, previous_action, normalizer)
    depth = np.asarray(obs["depth"], dtype=np.float32)
    with torch.no_grad():
        logits = actor(
            torch.from_numpy(depth[None, None]),
            torch.from_numpy(vector[None]),
        )
        masked = mask_logits(
            logits,
            torch.from_numpy(np.asarray(mask, dtype=np.bool_)[None]),
        )
        action = int(torch.argmax(masked, dim=1).item())
    if not bool(np.asarray(mask, dtype=np.bool_)[action]):
        raise RuntimeError("BC selected an action outside the runtime mask")
    return action


def _load_teacher_entries(recovery_run: Optional[Path]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    if recovery_run is None:
        return {}
    index_path = Path(recovery_run) / "rollout_index.csv"
    if not index_path.is_file():
        raise FileNotFoundError("Teacher recovery rollout index is missing: {}".format(index_path))
    from planning.contracts.feature import obs_continuous_vector

    entries: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if not _bool(row.get("execute_ok", True)):
            continue
        raw = str(row.get("dataset_npz", "")).strip()
        if not raw:
            continue
        path = _resolve_path(raw, base=Path(recovery_run))
        if not path.is_file():
            continue
        with np.load(str(path), allow_pickle=False) as archive:
            states = np.asarray(archive["states"], dtype=np.float32)
            goals = np.asarray(archive["goals"], dtype=np.float32)
            depths = np.asarray(archive["depths"], dtype=np.float16)
            masks = np.asarray(archive["execution_action_masks"], dtype=np.bool_)
            actions = np.asarray(archive["behavior_actions"], dtype=np.int64)
            previous = np.asarray(archive["prev_actions"], dtype=np.int64)
            if not (len(states) == len(goals) == len(depths) == len(masks) == len(actions) == len(previous)):
                raise ValueError("Teacher NPZ arrays disagree: {}".format(path))
            for index in range(len(actions)):
                digest = hashlib.sha256()
                continuous = np.ascontiguousarray(
                    np.concatenate([states[index], goals[index]]), dtype=np.float32
                )
                for array in (
                    continuous,
                    np.ascontiguousarray(depths[index], dtype=np.float16),
                    np.ascontiguousarray(masks[index].astype(np.uint8)),
                    np.asarray([int(previous[index])], dtype=np.int64),
                ):
                    digest.update(array.tobytes(order="C"))
                entries.setdefault((str(row["mission_id"]), digest.hexdigest()), []).append(
                    {
                        "teacher_action": int(actions[index]),
                        "teacher_episode_id": str(row.get("episode_id", "")),
                        "teacher_step_id": int(index),
                    }
                )
    return entries


def _teacher_match(
    mission_id: str,
    obs: Mapping[str, Any],
    mask: np.ndarray,
    previous_action: int,
    teacher_entries: Mapping[Tuple[str, str], Sequence[Mapping[str, Any]]],
) -> Optional[Dict[str, Any]]:
    from planning.contracts.feature import obs_continuous_vector

    digest = hashlib.sha256()
    for array in (
        np.ascontiguousarray(obs_continuous_vector(dict(obs)), dtype=np.float32),
        np.ascontiguousarray(np.asarray(obs["depth"], dtype=np.float16)),
        np.ascontiguousarray(np.asarray(mask, dtype=np.uint8).reshape(-1)),
        np.asarray([int(previous_action)], dtype=np.int64),
    ):
        digest.update(array.tobytes(order="C"))
    matches = list(teacher_entries.get((str(mission_id), digest.hexdigest()), ()))
    if not matches:
        return None
    actions = {int(item["teacher_action"]) for item in matches}
    if len(actions) != 1:
        raise RuntimeError("Teacher state identity maps to conflicting actions")
    return dict(matches[0])


def _step(env, obs, mask, mask_info, action: int):
    from planning.runtime.unity_env import PrimitiveExecutionAbortedError
    from planning.contracts.primitive_execution import PRIMITIVE_EXECUTION_RESULT_TERMINAL_ABORT

    try:
        return env.step_primitive(
            int(action),
            obs_before=obs,
            precomputed_mask=mask,
            precomputed_mask_info=mask_info,
        )
    except PrimitiveExecutionAbortedError as error:
        result = error.execution_result
        if isinstance(result, Mapping) and result.get("kind") == PRIMITIVE_EXECUTION_RESULT_TERMINAL_ABORT:
            return env.materialize_terminal_abort(
                error,
                action_id=int(action),
                obs_before=obs,
                precomputed_mask=mask,
                precomputed_mask_info=mask_info,
            )
        raise


def _make_env(worker_spec_path: Path, max_steps: int):
    """Build the existing reliable-v4 environment; no runtime is launched here."""

    from planning.contracts.policy_runtime import DEPTH_MASK_NUMERIC_DEFAULTS
    from planning.mission.spec import GOAL_RADIUS_XY_M
    from planning.primitives.library import MotionPrimitiveLibrary
    from planning.runtime.reliable_training import (
        build_reliable_v4_backend,
        build_reliable_v4_runtime_config,
    )
    from planning.runtime.unity_env import (
        ACTION_MASK_Z_MARGIN_M,
        FLIGHT_Z_MAX_M,
        FLIGHT_Z_MIN_M,
        GOAL_TOLERANCE_Z_M,
        EnvConfig,
        UnityForestEnv,
    )
    from planning.runtime.worker import load_worker_runtime_spec_file

    specs = load_worker_runtime_spec_file(worker_spec_path)
    if len(specs) != 1:
        raise ValueError("collector requires exactly one worker runtime spec")
    spec = specs[0]
    config = build_reliable_v4_runtime_config(
        runtime_instance_id=spec.runtime_instance_id,
        command_endpoint=spec.python_command_endpoint,
        result_endpoint=spec.python_result_endpoint,
        snapshot_endpoint=spec.python_snapshot_endpoint,
        timeout_s=10.0,
    )
    backend = build_reliable_v4_backend(config)
    mpl = MotionPrimitiveLibrary()
    env = UnityForestEnv(
        start=(0.0, 0.0, 2.0),
        goal=(40.0, 0.0, 2.0),
        config=EnvConfig(
            depth_out_width=160,
            depth_out_height=90,
            reset_timeout=10.0,
            reset_settle_s=0.30,
            primitive_post_wait_s=0.0,
            stop_at_primitive_end=False,
            enforce_sensor_sync=True,
            max_sensor_skew_s=0.080,
            max_episode_steps=int(max_steps),
            goal_radius_xy=GOAL_RADIUS_XY_M,
            goal_tolerance_z=GOAL_TOLERANCE_Z_M,
            z_min=FLIGHT_Z_MIN_M,
            z_max=FLIGHT_Z_MAX_M,
            action_mask_z_margin=ACTION_MASK_Z_MARGIN_M,
            use_depth_collision_mask=True,
            depth_mask_collision_radius_m=DEPTH_MASK_NUMERIC_DEFAULTS["depth_mask_collision_radius_m"],
            depth_mask_slack_m=DEPTH_MASK_NUMERIC_DEFAULTS["depth_mask_slack_m"],
            depth_mask_path_sample_stride=DEPTH_MASK_NUMERIC_DEFAULTS["depth_mask_sample_stride"],
            depth_mask_max_patch_radius_px=DEPTH_MASK_NUMERIC_DEFAULTS["depth_mask_max_patch_radius_px"],
            terminate_on_dead_end=True,
            terminate_on_invalid_action=True,
        ),
        mpl=mpl,
        reliable_v4_backend=backend,
        legacy_command_path_enabled=False,
        node_name="multi_action_replay_collector",
        anonymous=False,
    )
    return env, backend, mpl, spec


def _mission_start_goal(row: Mapping[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    from planning.mission.spec import row_start_goal

    start, goal = row_start_goal(row)
    return np.asarray(start, dtype=np.float32), np.asarray(goal, dtype=np.float32)


def _replay_prefix(
    env,
    target: Mapping[str, Any],
    *,
    max_env_steps: int,
    env_steps_used: int,
):
    start, goal = _mission_start_goal(target["mission"])
    obs = env.reset(start=start, goal=goal)
    mask, mask_info = env.get_action_mask(obs, return_info=True)
    previous_action = -1
    prefix_return = 0.0
    for action in target["prefix_actions"]:
        if int(env_steps_used) >= int(max_env_steps):
            raise RuntimeError("environment-step budget exhausted during prefix replay")
        if not bool(mask[int(action)]):
            return None, env_steps_used, "prefix_action_invalid"
        obs, reward, done, info = _step(env, obs, mask, mask_info, int(action))
        env_steps_used += 1
        prefix_return += float(reward)
        if bool(done):
            return None, env_steps_used, "prefix_terminated"
        mask = np.asarray(obs.get("action_mask"), dtype=np.bool_)
        if mask.ndim != 1:
            mask, mask_info = env.get_action_mask(obs, return_info=True)
        else:
            mask_info = dict(info.get("next_action_mask_info", {}))
        previous_action = int(action)
    return (
        (
            obs,
            np.asarray(mask, dtype=np.bool_),
            dict(mask_info),
            previous_action,
            float(prefix_return),
            int(len(target["prefix_actions"])),
        ),
        env_steps_used,
        "ok",
    )


def _candidate_transition(
    env,
    target: Mapping[str, Any],
    *,
    action: int,
    expected_state_hash: str,
    source: str,
    max_env_steps: int,
    env_steps_used: int,
    action_dim: int,
    max_steps: int,
    actor,
    normalizer,
    torch,
    mask_logits,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], int, str]:
    branch = {
        "state_id": str(expected_state_hash),
        "action_source": str(source),
        "action": int(action),
        "status": "not_started",
    }
    replayed, env_steps_used, status = _replay_prefix(
        env, target, max_env_steps=max_env_steps, env_steps_used=env_steps_used
    )
    if replayed is None:
        branch.update({"status": status, "failure_reason": status})
        return None, branch, env_steps_used, status
    obs, mask, mask_info, previous_action, prefix_return, prefix_steps = replayed
    actual_hash = _state_content_hash(obs, mask, previous_action)
    if actual_hash != str(expected_state_hash):
        branch.update({"status": "state_hash_mismatch", "failure_reason": "state_hash_mismatch"})
        return None, branch, env_steps_used, "state_hash_mismatch"
    if not bool(mask[int(action)]):
        branch.update({"status": "invalid", "failure_reason": "candidate_action_invalid"})
        return None, branch, env_steps_used, "candidate_action_invalid"
    if int(env_steps_used) >= int(max_env_steps):
        raise RuntimeError("environment-step budget exhausted before candidate action")
    next_obs, reward, done, info = _step(env, obs, mask, mask_info, int(action))
    env_steps_used += 1
    candidate_done = bool(done)
    episode_return = float(prefix_return) + float(reward)
    episode_steps = int(prefix_steps) + 1
    terminal_info = dict(info)
    next_mask = np.asarray(next_obs.get("action_mask"), dtype=np.bool_)
    if next_mask.shape != (int(action_dim),):
        try:
            next_mask, _ = env.get_action_mask(next_obs, return_info=True)
        except Exception:
            next_mask = np.zeros((int(action_dim),), dtype=np.bool_)
    next_previous_action = int(action)
    next_hash = _state_content_hash(next_obs, next_mask, next_previous_action)

    # The candidate determines the first branch action.  The rest of the
    # branch follows the frozen deterministic BC policy solely to obtain a
    # comparable full-episode return and terminal outcome.  No optimizer or
    # production replay is touched.
    continuation_mask = next_mask
    continuation_mask_info = dict(info.get("next_action_mask_info", {}))
    continuation_previous_action = next_previous_action
    while not bool(done) and episode_steps < int(max_steps):
        if int(env_steps_used) >= int(max_env_steps):
            raise RuntimeError("environment-step budget exhausted during branch continuation")
        continuation_action = _bc_action(
            next_obs,
            continuation_mask,
            continuation_previous_action,
            actor,
            normalizer,
            torch,
            mask_logits,
        )
        next_obs, continuation_reward, done, continuation_info = _step(
            env,
            next_obs,
            continuation_mask,
            continuation_mask_info,
            int(continuation_action),
        )
        env_steps_used += 1
        episode_return += float(continuation_reward)
        episode_steps += 1
        terminal_info = dict(continuation_info)
        continuation_previous_action = int(continuation_action)
        if not bool(done):
            continuation_mask = np.asarray(next_obs.get("action_mask"), dtype=np.bool_)
            if continuation_mask.shape != (int(action_dim),):
                continuation_mask, continuation_mask_info = env.get_action_mask(
                    next_obs, return_info=True
                )
            else:
                continuation_mask_info = dict(
                    continuation_info.get("next_action_mask_info", {})
                )
    if not bool(done):
        terminal_info = dict(terminal_info)
        terminal_info["done_reason"] = "timeout"
    terminal_reason = str(terminal_info.get("done_reason", ""))
    if terminal_reason in ("", "unknown"):
        terminal_reason = "timeout" if not bool(done) else "terminal"
    branch.update(
        {
            "status": "completed",
            "episode_id": "multi-action-{}-{}-{}".format(
                target["source_episode_id"], target["step_id"], int(action)
            ),
            "episode_return": float(episode_return),
            "steps": int(episode_steps),
            "terminal_reason": terminal_reason,
            "success": bool(terminal_info.get("success", False)),
            "collision": bool(terminal_info.get("collided", terminal_reason == "collision")),
            "dead_end": bool(terminal_info.get("dead_end", terminal_reason == "dead_end")),
            "timeout": bool(terminal_info.get("timeout", terminal_reason == "timeout")),
        }
    )
    transition_id = hashlib.sha256(
        "{}:{}:{}".format(actual_hash, int(action), source).encode("utf-8")
    ).hexdigest()
    episode_id = "multi-action-{}-{}-{}".format(
        target["source_episode_id"], target["step_id"], int(action)
    )
    return {
        "state_id": actual_hash,
        "mission_id": str(target["mission_id"]),
        "episode_id": episode_id,
        "step_id": int(target["step_id"]),
        "route_id": str(target["route_id"]),
        "action_source": str(source),
        "action": int(action),
        "mask": np.asarray(mask, dtype=np.bool_),
        "reward": float(reward),
        "state_vector": __import__(
            "planning.contracts.feature", fromlist=["obs_continuous_vector"]
        ).obs_continuous_vector(dict(obs)),
        "state_depth": np.asarray(obs["depth"], dtype=np.float32),
        "next_state_id": next_hash,
        "next_state_vector": __import__(
            "planning.contracts.feature", fromlist=["obs_continuous_vector"]
        ).obs_continuous_vector(dict(next_obs)),
        "next_state_depth": np.asarray(next_obs["depth"], dtype=np.float32),
        "next_mask": np.asarray(next_mask, dtype=np.bool_),
        "done": candidate_done,
        "transition_id": transition_id,
    }, branch, env_steps_used, "accepted"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--worker-spec-file", type=Path, required=True)
    parser.add_argument("--mission-index", type=Path, required=True)
    parser.add_argument("--bc-index", type=Path, required=True)
    parser.add_argument("--bc-dir", type=Path, required=True)
    parser.add_argument("--recovery-run", type=Path, default=None)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=500)
    parser.add_argument("--min-transitions", type=int, default=2500)
    parser.add_argument("--max-transitions", type=int, default=4000)
    parser.add_argument("--max-episodes", type=int, default=3500)
    parser.add_argument("--max-env-steps", type=int, default=150000)
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--neighbor-count", type=int, default=2)
    parser.add_argument("--random-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--max-position-error-m", type=float, default=0.15)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if int(args.max_states) <= 0 or int(args.min_transitions) <= 0:
        raise ValueError("max-states and min-transitions must be positive")
    if int(args.max_transitions) < int(args.min_transitions):
        raise ValueError("max-transitions must be at least min-transitions")
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError("refusing non-empty diagnostic output directory: {}".format(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "states").mkdir(parents=True, exist_ok=True)

    missions = _load_missions(Path(args.mission_index))
    targets = _load_failed_targets(
        Path(args.bc_index), Path(args.bc_dir), missions, int(args.max_states)
    )
    teacher_entries = _load_teacher_entries(
        Path(args.recovery_run).expanduser().resolve() if args.recovery_run else None
    )
    torch, actor, normalizer, _bc_checkpoint, mask_logits_fn = _load_bc_actor(
        Path(args.bc_checkpoint).expanduser().resolve(), torch_threads=int(args.torch_threads)
    )
    from planning.awac.primitive_neighborhood import build_primitive_neighborhood

    env, backend, mpl, worker_spec = _make_env(Path(args.worker_spec_file), int(args.max_steps))
    neighborhood = build_primitive_neighborhood(mpl.pos_ref)
    started = time.time()
    protocol = {
        "schema_id": "multi_action_replay_v1",
        "status": "RUNNING",
        "diagnostic_only": True,
        "production_replay_modified": False,
        "training_executed": False,
        "actor_updates": 0,
        "critic_updates": 0,
        "max_states": int(args.max_states),
        "min_transitions": int(args.min_transitions),
        "max_transitions": int(args.max_transitions),
        "max_episodes": int(args.max_episodes),
        "max_environment_steps": int(args.max_env_steps),
        "max_steps": int(args.max_steps),
        "neighbor_count": int(args.neighbor_count),
        "random_count": int(args.random_count),
        "seed": int(args.seed),
        "observation_contract": OBSERVATION_CONTRACT,
        "bc_checkpoint_sha256": _sha256_file(Path(args.bc_checkpoint)),
        "bc_index_sha256": _sha256_file(Path(args.bc_index)),
        "mission_index_sha256": _sha256_file(Path(args.mission_index)),
        "bc_steps_tree_sha256": _sha256_tree(Path(args.bc_dir) / "steps", suffixes=(".csv",)),
        "teacher_entry_count": int(sum(len(values) for values in teacher_entries.values())),
        "worker_runtime_identity": str(worker_spec.runtime_instance_id),
        "action_selection": "BC + exact Teacher if matched + geometry neighbors + masked random; no Q selection",
        "state_selection": "failed BC episodes, deterministic episode/step order, terminal-adjacent first",
    }
    _write_json(out_dir / "protocol.json", protocol)

    rows: List[Dict[str, Any]] = []
    branch_records: List[Dict[str, Any]] = []
    state_records: List[Dict[str, Any]] = []
    env_steps_used = 0
    episode_count = 0
    state_failures: Dict[str, int] = {}
    runtime_failure = False
    try:
        env.wait_until_ready(timeout_s=30.0)
        for state_index, target in enumerate(targets):
            if len(rows) >= int(args.max_transitions) or episode_count >= int(args.max_episodes):
                break
            replayed, env_steps_used, status = _replay_prefix(
                env,
                target,
                max_env_steps=int(args.max_env_steps),
                env_steps_used=env_steps_used,
            )
            if replayed is None:
                state_failures[status] = state_failures.get(status, 0) + 1
                state_records.append({"state_index": state_index, "status": status, **target})
                continue
            obs, mask, mask_info, previous_action, _, _ = replayed
            position = np.asarray(obs["state"].get("position", ()), dtype=np.float32)
            recorded_position = np.asarray(target["recorded_position"], dtype=np.float32)
            position_error = float(np.linalg.norm(position - recorded_position)) if position.shape == (3,) else None
            if position_error is not None and position_error > float(args.max_position_error_m):
                state_failures["position_mismatch"] = state_failures.get("position_mismatch", 0) + 1
                state_records.append({"state_index": state_index, "status": "position_mismatch", "position_error_m": position_error, **target})
                continue
            bc_action = _bc_action(
                obs, mask, previous_action, actor, normalizer, torch, mask_logits_fn
            )
            if int(bc_action) != int(target["recorded_action"]):
                state_failures["bc_action_mismatch"] = state_failures.get("bc_action_mismatch", 0) + 1
                state_records.append({"state_index": state_index, "status": "bc_action_mismatch", "computed_bc_action": int(bc_action), **target})
                continue
            teacher = _teacher_match(
                str(target["mission_id"]), obs, mask, previous_action, teacher_entries
            )
            neighbor_actions = neighborhood.top_k(
                int(bc_action), int(args.neighbor_count), valid_mask=mask
            ).tolist()
            planned = plan_action_sources(
                mask,
                bc_action=int(bc_action),
                teacher_action=int(teacher["teacher_action"]) if teacher else None,
                neighbor_actions=neighbor_actions,
                random_count=int(args.random_count),
                seed=int(args.seed) + int(state_index) * 7919,
            )
            if not planned:
                state_failures["no_legal_candidates"] = state_failures.get("no_legal_candidates", 0) + 1
                continue
            state_hash = _state_content_hash(obs, mask, previous_action)
            state_records.append(
                {
                    "state_index": state_index,
                    "status": "collecting",
                    "state_id": state_hash,
                    "mission_id": target["mission_id"],
                    "source_episode_id": target["source_episode_id"],
                    "step_id": target["step_id"],
                    "recorded_action": target["recorded_action"],
                    "bc_action": int(bc_action),
                    "teacher_action": int(teacher["teacher_action"]) if teacher else None,
                    "neighbor_actions": [int(value) for value in neighbor_actions],
                    "planned_actions": [{"source": source, "action": int(action)} for source, action in planned],
                }
            )
            np.savez_compressed(
                out_dir / "states" / "state_{:04d}.npz".format(state_index),
                state_id=np.asarray([state_hash]),
                vector=np.asarray(__import__("planning.contracts.feature", fromlist=["obs_continuous_vector"]).obs_continuous_vector(dict(obs)), dtype=np.float32),
                depth=np.asarray(obs["depth"], dtype=np.float32),
                mask=np.asarray(mask, dtype=np.bool_),
                previous_action=np.asarray([int(previous_action)], dtype=np.int64),
            )
            state_row_start = len(rows)
            for source, action in planned:
                if len(rows) >= int(args.max_transitions) or episode_count >= int(args.max_episodes):
                    break
                episode_count += 1
                try:
                    transition, branch, env_steps_used, branch_status = _candidate_transition(
                        env,
                        target,
                        action=int(action),
                        expected_state_hash=state_hash,
                        source=source,
                        max_env_steps=int(args.max_env_steps),
                        env_steps_used=env_steps_used,
                        action_dim=int(mask.size),
                        max_steps=int(args.max_steps),
                        actor=actor,
                        normalizer=normalizer,
                        torch=torch,
                        mask_logits=mask_logits_fn,
                    )
                except Exception as error:
                    branch = {
                        "state_id": state_hash,
                        "action_source": str(source),
                        "action": int(action),
                        "status": "runtime",
                        "failure_reason": repr(error),
                    }
                    transition = None
                    branch_status = "runtime"
                    runtime_failure = True
                branch_records.append(
                    {
                        "state_index": int(state_index),
                        "mission_id": str(target["mission_id"]),
                        "source_episode_id": str(target["source_episode_id"]),
                        "step_id": int(target["step_id"]),
                        **branch,
                    }
                )
                if transition is None:
                    state_failures[branch_status] = state_failures.get(branch_status, 0) + 1
                else:
                    rows.append(transition)
                if runtime_failure:
                    break
            state_records[-1]["status"] = (
                "accepted" if len(rows) > state_row_start else "no_accepted_candidates"
            )
            print(
                "MULTI_ACTION_STATE {}/{} transitions={} episodes={} env_steps={}".format(
                    state_index + 1, len(targets), len(rows), episode_count, env_steps_used
                ),
                flush=True,
            )
            if runtime_failure:
                break
    finally:
        try:
            env.stop()
        finally:
            backend.close()

    with (out_dir / "state_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in state_records:
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
    with (out_dir / "candidate_branches.jsonl").open("w", encoding="utf-8") as handle:
        for record in branch_records:
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")

    accepted_state_count = len({str(row["state_id"]) for row in rows})
    status = "PASS"
    artifact = out_dir / "replay"
    if (
        runtime_failure
        or accepted_state_count < int(args.max_states)
        or len(rows) < int(args.min_transitions)
    ):
        status = "FAIL"
    else:
        # Strip collector-only diagnostics before passing rows to the contract.
        clean_rows = [
            {key: value for key, value in row.items() if key != "terminal_reason"}
            for row in rows
        ]
        metadata = {
            "observation_contract": OBSERVATION_CONTRACT,
            "observation_source": OBSERVATION_CONTRACT,
            "diagnostic_only": True,
            "production_replay": False,
            "training_consumed": False,
            "source_bc_checkpoint_sha256": protocol["bc_checkpoint_sha256"],
            "source_bc_index_sha256": protocol["bc_index_sha256"],
            "source_mission_index_sha256": protocol["mission_index_sha256"],
            "source_teacher_recovery_run": str(args.recovery_run or ""),
            "teacher_entry_count": protocol["teacher_entry_count"],
            "selection_max_states": int(args.max_states),
            "selection_min_transitions": int(args.min_transitions),
            "environment_steps": int(env_steps_used),
            "episodes_executed": int(episode_count),
            "random_seed": int(args.seed),
        }
        write_multi_action_replay(artifact, clean_rows, metadata)

    summary = {
        "schema_id": "multi_action_replay_v1",
        "status": status,
        "state_count": int(accepted_state_count),
        "transition_count": int(len(rows)),
        "episodes_executed": int(episode_count),
        "environment_steps_executed": int(env_steps_used),
        "state_failures": state_failures,
        "branch_count": int(len(branch_records)),
        "runtime_failure": bool(runtime_failure),
        "elapsed_s": float(time.time() - started),
        "replay_dir": str(artifact) if status == "PASS" else "",
        "production_replay_modified": False,
        "training_executed": False,
    }
    if status == "PASS":
        summary["audit"] = audit_multi_action_replay(
            artifact,
            min_states=int(args.max_states),
            min_transitions=int(args.min_transitions),
        )
    _write_json(out_dir / "collection_summary.json", summary)
    protocol["status"] = status
    protocol["state_count"] = int(accepted_state_count)
    protocol["transition_count"] = int(len(rows))
    protocol["episodes_executed"] = int(episode_count)
    protocol["environment_steps_executed"] = int(env_steps_used)
    protocol["state_failures"] = state_failures
    protocol["branch_count"] = int(len(branch_records))
    protocol["runtime_failure"] = bool(runtime_failure)
    _write_json(out_dir / "protocol.json", protocol)
    print("MULTI_ACTION_REPLAY_COLLECTION={}".format(status), flush=True)
    print("STATES={}".format(accepted_state_count), flush=True)
    print("TRANSITIONS={}".format(len(rows)), flush=True)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
