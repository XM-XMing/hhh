"""Multiprocessing-safe exact relabeling of collected Unity rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from planning.native.geometry import NativeGeometryContext
from planning.mission.global_route import GlobalRouteConfig
from planning.data.rollout import (
    load_rollout_episode_fields,
    validate_rollout_metadata,
)
from planning.data.mission_routes import MissionRouteStore, validate_route_references
from planning.contracts.feature import NUM_ACTIONS
from planning.primitives.library import MotionPrimitiveLibrary
from planning.teacher.policy import ReachabilityTeacher, TeacherConfig
from planning.teacher.policy import TEACHER_PLANNING_CONTRACT_ID
from planning.contracts.pipeline_provenance import validate_episode_provenance

_WORKER_TEACHER = None
_WORKER_INCLUDE_SCORES = False
_WORKER_GEOMETRY_CONTEXT = None
_WORKER_ROUTE_STORE = None
_WORKER_DIRECT_ARRAYS = None
LABELING_ROLLOUT_FIELDS = (
    "behavior_actions",
    "prev_actions",
    "poses_before",
    "velocities_before",
    "execution_action_masks",
    "start",
    "goal",
)

def init_label_worker(
    config_payload: Dict,
    collision_payload: Dict,
    include_scores: bool,
    route_store_prefix: str = "",
    direct_output: Optional[Dict] = None,
) -> None:
    """Initialize one process; loaded voxel/MPL data is reused for all episodes."""
    global _WORKER_TEACHER, _WORKER_INCLUDE_SCORES, _WORKER_GEOMETRY_CONTEXT, _WORKER_ROUTE_STORE, _WORKER_DIRECT_ARRAYS
    if _WORKER_GEOMETRY_CONTEXT is not None:
        _WORKER_GEOMETRY_CONTEXT.close()
    if _WORKER_ROUTE_STORE is not None:
        _WORKER_ROUTE_STORE.close()
    if not str(route_store_prefix).strip():
        raise ValueError("label worker requires a committed mission route store")
    _WORKER_ROUTE_STORE = MissionRouteStore.open(
        Path(route_store_prefix), validate=False
    )
    mpl = MotionPrimitiveLibrary()
    teacher_config = TeacherConfig(**config_payload)
    route_config = GlobalRouteConfig(
        resolution_m=float(teacher_config.global_route_resolution_m),
        flight_z_min_m=float(teacher_config.z_min),
        flight_z_max_m=float(teacher_config.z_max),
        lookahead_m=float(teacher_config.global_route_lookahead_m),
        tracking_margin_m=float(teacher_config.global_route_tracking_margin_m),
    )
    _WORKER_GEOMETRY_CONTEXT = NativeGeometryContext.from_voxel_cache(
        collision_payload["cache_npz"],
        voxel_size=float(collision_payload["voxel_size"]),
        route_config=route_config,
    )
    checker = _WORKER_GEOMETRY_CONTEXT.collision_checker(
        float(collision_payload["inflate_radius"])
    )
    _WORKER_TEACHER = ReachabilityTeacher(mpl, checker, teacher_config)
    _WORKER_INCLUDE_SCORES = bool(include_scores)
    _WORKER_DIRECT_ARRAYS = None
    if direct_output is not None:
        building = Path(str(direct_output["building"])).expanduser().resolve()
        total = int(direct_output["total"])
        arrays = {
            "soft_targets": np.load(building / "soft_targets.npy", mmap_mode="r+"),
            "global_masks": np.load(building / "global_masks.npy", mmap_mode="r+"),
            "teacher_argmax": np.load(building / "teacher_argmax.npy", mmap_mode="r+"),
            "behavior_actions": np.load(building / "behavior_actions.npy", mmap_mode="r+"),
            "valid_counts": np.load(building / "valid_counts.npy", mmap_mode="r+"),
            "entropies": np.load(building / "entropies.npy", mmap_mode="r+"),
        }
        if any(int(array.shape[0]) != total for array in arrays.values()):
            raise ValueError("direct label output arrays have inconsistent transition counts")
        if bool(direct_output.get("include_scores", False)):
            arrays["teacher_scores"] = np.load(
                building / "teacher_scores.npy", mmap_mode="r+"
            )
        _WORKER_DIRECT_ARRAYS = arrays

def _require_worker() -> ReachabilityTeacher:
    if _WORKER_TEACHER is None:
        raise RuntimeError("label worker is not initialized")
    return _WORKER_TEACHER

def label_rollout_episode(task: Tuple[int, str]) -> Dict:
    """Generate exact labels from the actual pose saved before each action."""
    teacher = _require_worker()
    episode_id, path_text = int(task[0]), task[1]
    direct_offset = int(task[2]) if len(task) > 2 else None
    direct_length = int(task[3]) if len(task) > 3 else None
    path = Path(path_text).expanduser().resolve()
    episode = load_rollout_episode_fields(path, LABELING_ROLLOUT_FIELDS)
    metadata = dict(episode.get("metadata", {}))
    validate_rollout_metadata(metadata, path)
    validate_episode_provenance(metadata, path=str(path), expected_episode_id=episode_id)
    duration = float(metadata.get("mpl_duration_s", float("nan")))
    forward_distance = float(metadata.get("mpl_forward_distance_m", float("nan")))
    contract_sha256 = str(metadata.get("mpl_contract_sha256", ""))
    planning_contract_id = str(metadata.get("teacher_planning_contract_id", ""))
    if planning_contract_id != TEACHER_PLANNING_CONTRACT_ID:
        raise ValueError(
            "{} teacher planning contract {} != runtime {}".format(
                path,
                planning_contract_id or "<missing>",
                TEACHER_PLANNING_CONTRACT_ID,
            )
        )
    behavior_config = metadata.get("teacher_config")
    if behavior_config != teacher.config.__dict__:
        raise ValueError(
            "{} teacher config differs between collection and relabeling".format(path)
        )
    motion_primitives = teacher.motion_primitives
    if not np.isfinite(duration) or abs(duration - float(motion_primitives.duration_s)) > 1.0e-6:
        raise ValueError(
            "{} MPL duration {} != runtime {}".format(
                path, duration, motion_primitives.duration_s
            )
        )
    if not np.isfinite(forward_distance) or abs(
        forward_distance - float(motion_primitives.forward_distance_m)
    ) > 1.0e-6:
        raise ValueError(
            "{} MPL forward distance {} != runtime {}".format(
                path, forward_distance, motion_primitives.forward_distance_m
            )
        )
    if contract_sha256 != str(motion_primitives.contract_sha256):
        raise ValueError(
            "{} motion-primitive contract {} != runtime {}".format(
                path, contract_sha256 or "<missing>",
                motion_primitives.contract_sha256,
            )
        )

    behavior = np.asarray(episode["behavior_actions"], dtype=np.int64)
    prev_actions = np.asarray(episode["prev_actions"], dtype=np.int64)
    poses = np.asarray(episode["poses_before"], dtype=np.float32)
    velocities = np.asarray(episode["velocities_before"], dtype=np.float32)
    goal = np.asarray(episode["goal"], dtype=np.float32).reshape(3)
    if _WORKER_ROUTE_STORE is None:
        raise RuntimeError("label worker route store is not initialized")
    try:
        route_index = int(float(metadata["global_route_index"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("{} rollout is missing global_route_index".format(path)) from error
    route_row = dict(metadata)
    route_row.update(
        {
            "global_route_index": route_index,
            "goal_x": float(goal[0]),
            "goal_y": float(goal[1]),
            "goal_z": float(goal[2]),
        }
    )
    validate_route_references(
        [route_row],
        _WORKER_ROUTE_STORE,
        candidate_index_sha256=_WORKER_ROUTE_STORE.metadata["candidate_index_sha256"],
        require_provenance=True,
    )
    route = _WORKER_ROUTE_STORE.validate(route_index, goal=goal)
    teacher.set_mission_route(route, goal)
    execution_masks = np.asarray(episode["execution_action_masks"], dtype=np.bool_)
    transition_count = int(behavior.shape[0])
    validate_episode_provenance(
        metadata,
        path=str(path),
        expected_episode_id=episode_id,
        expected_transition_count=transition_count,
    )
    expected_shapes = {
        "prev_actions": (transition_count,),
        "poses_before": (transition_count, 4),
        "velocities_before": (transition_count, 3),
        "execution_action_masks": (transition_count, NUM_ACTIONS),
        "start": (3,),
        "goal": (3,),
    }
    for key, expected in expected_shapes.items():
        actual = np.asarray(episode[key]).shape
        if actual != expected:
            raise ValueError(
                "{} {} shape {} != {}".format(path, key, actual, expected)
            )
    if transition_count <= 0:
        raise ValueError("{} contains zero transitions".format(path))
    if np.any((behavior < 0) | (behavior >= NUM_ACTIONS)):
        raise ValueError("{} behavior action out of range".format(path))
    if np.any((prev_actions < -1) | (prev_actions >= NUM_ACTIONS)):
        raise ValueError("{} previous action out of range".format(path))
    if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
        raise ValueError("{} contains non-finite labeling state".format(path))
    if not np.all(
        execution_masks[np.arange(transition_count), behavior]
    ):
        raise ValueError("{} behavior action outside execution mask".format(path))

    soft_targets = np.zeros((transition_count, NUM_ACTIONS), dtype=np.float16)
    global_masks = np.zeros((transition_count, NUM_ACTIONS), dtype=np.bool_)
    teacher_argmax = np.full((transition_count,), -1, dtype=np.int16)
    valid_counts = np.zeros((transition_count,), dtype=np.int16)
    entropies = np.zeros((transition_count,), dtype=np.float32)
    scores = (
        np.full((transition_count, NUM_ACTIONS), -np.inf, dtype=np.float32)
        if _WORKER_INCLUDE_SCORES
        else None
    )

    behavior_valid = 0
    teacher_valid = 0
    teacher_matches_behavior = 0
    mask_match_count = 0
    for index in range(transition_count):
        position = poses[index, :3]
        yaw = float(poses[index, 3])
        action_id, debug = teacher.score_state(
            position=position,
            yaw=yaw,
            velocity=velocities[index],
            goal=goal,
            prev_action=int(prev_actions[index]),
        )
        mask = np.asarray(debug["global_action_mask"], dtype=np.bool_)
        global_masks[index] = mask
        valid_counts[index] = int(np.count_nonzero(mask))
        mask_match_count += int(np.array_equal(mask, execution_masks[index]))
        behavior_id = int(behavior[index])
        if 0 <= behavior_id < NUM_ACTIONS and bool(mask[behavior_id]):
            behavior_valid += 1
        if action_id >= 0:
            teacher_argmax[index] = int(action_id)
            teacher_valid += int(bool(mask[int(action_id)]))
            teacher_matches_behavior += int(int(action_id) == behavior_id)
            target = np.asarray(debug["teacher_soft_target"], dtype=np.float32)
            if not np.isfinite(target).all() or abs(float(target.sum()) - 1.0) > 5.0e-4:
                raise ValueError(
                    "invalid teacher distribution at {} transition {}".format(path, index)
                )
            soft_targets[index] = target.astype(np.float16)
            entropies[index] = float(debug.get("teacher_entropy", 0.0))
            if scores is not None:
                scores[index] = np.asarray(debug["teacher_scores"], dtype=np.float32)

    result = {
        "episode_id": int(episode_id),
        "path": str(path),
        "length": transition_count,
        "soft_targets": soft_targets,
        "global_masks": global_masks,
        "teacher_argmax": teacher_argmax,
        "behavior_actions": behavior.astype(np.int16),
        "valid_counts": valid_counts,
        "entropies": entropies,
        "behavior_valid": behavior_valid,
        "teacher_valid": teacher_valid,
        "teacher_matches_behavior": teacher_matches_behavior,
        "mask_match_count": mask_match_count,
        "global_route_index": route_index,
    }
    if scores is not None:
        result["teacher_scores"] = scores
    if _WORKER_DIRECT_ARRAYS is not None:
        if direct_offset is None or direct_length is None:
            raise ValueError("direct label output requires an episode offset and length")
        if direct_length != transition_count:
            raise ValueError(
                "direct label length {} != computed {} for {}".format(
                    direct_length, transition_count, path
                )
            )
        target = slice(direct_offset, direct_offset + transition_count)
        _WORKER_DIRECT_ARRAYS["soft_targets"][target] = soft_targets
        _WORKER_DIRECT_ARRAYS["global_masks"][target] = global_masks
        _WORKER_DIRECT_ARRAYS["teacher_argmax"][target] = teacher_argmax
        _WORKER_DIRECT_ARRAYS["behavior_actions"][target] = behavior.astype(np.int16)
        _WORKER_DIRECT_ARRAYS["valid_counts"][target] = valid_counts
        _WORKER_DIRECT_ARRAYS["entropies"][target] = entropies
        if scores is not None:
            _WORKER_DIRECT_ARRAYS["teacher_scores"][target] = scores
        result["offset"] = direct_offset
        for key in (
            "soft_targets",
            "global_masks",
            "teacher_argmax",
            "behavior_actions",
            "valid_counts",
            "entropies",
            "teacher_scores",
        ):
            result.pop(key, None)
    return result
