"""Fork-pool worker functions for local depth action-mask generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from planning.data.rollout import load_rollout_episode_fields, validate_rollout_metadata
from planning.safety.depth_mask import normalized_depth_to_metric
from planning.safety.depth_safety import DepthSafetyConfig, local_depth_action_mask
from planning.primitives.library import MotionPrimitiveLibrary
from planning.contracts.pipeline_provenance import validate_episode_provenance
from planning.data.mission_routes import MissionRouteStore, validate_route_references

_WORKER_MPL = None
_WORKER_CONFIG = None
_WORKER_DEPTH_MIN_M = 0.30
_WORKER_DEPTH_MAX_M = 3.00
_WORKER_ROUTE_STORE = None
_WORKER_DIRECT_MASKS = None

def init_worker(
    config_payload: Dict,
    depth_min_m: float,
    depth_max_m: float,
    route_store_prefix: str = "",
    direct_output: Optional[Dict] = None,
) -> None:
    """Initialize static MPL/depth geometry once in each process."""
    global _WORKER_MPL, _WORKER_CONFIG, _WORKER_DEPTH_MIN_M, _WORKER_DEPTH_MAX_M, _WORKER_ROUTE_STORE, _WORKER_DIRECT_MASKS
    _WORKER_MPL = MotionPrimitiveLibrary()
    _WORKER_CONFIG = DepthSafetyConfig(**config_payload)
    _WORKER_DEPTH_MIN_M = float(depth_min_m)
    _WORKER_DEPTH_MAX_M = float(depth_max_m)
    if _WORKER_ROUTE_STORE is not None:
        _WORKER_ROUTE_STORE.close()
    if not str(route_store_prefix).strip():
        raise ValueError("depth-mask worker requires a committed mission route store")
    _WORKER_ROUTE_STORE = MissionRouteStore.open(
        Path(route_store_prefix), validate=False
    )
    _WORKER_DIRECT_MASKS = None
    if direct_output is not None:
        building = Path(str(direct_output["building"])).expanduser().resolve()
        masks = np.load(building / "local_action_masks.npy", mmap_mode="r+")
        if int(masks.shape[0]) != int(direct_output["total"]):
            raise ValueError("direct depth-mask output has an unexpected transition count")
        _WORKER_DIRECT_MASKS = masks

def build_episode_masks(task: Tuple[int, str]) -> Tuple[int, str, np.ndarray]:
    """Build one [T,105] local depth-only action mask array."""
    if _WORKER_MPL is None or _WORKER_CONFIG is None:
        raise RuntimeError("depth mask worker is not initialized")
    episode_id, path_text = int(task[0]), task[1]
    direct_offset = int(task[2]) if len(task) > 2 else None
    direct_length = int(task[3]) if len(task) > 3 else None
    path = Path(path_text).expanduser().resolve()
    episode = load_rollout_episode_fields(path, ("depths", "goal"))
    validate_rollout_metadata(episode["metadata"], path)
    depths = np.asarray(episode["depths"], dtype=np.float32)
    if depths.ndim != 3 or depths.shape[0] <= 0 or not np.isfinite(depths).all():
        raise ValueError("{} depth array is invalid: {}".format(path, depths.shape))
    goal = np.asarray(episode["goal"], dtype=np.float32)
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("{} goal array is invalid: {}".format(path, goal.shape))
    validate_episode_provenance(
        episode["metadata"],
        path=str(path),
        expected_episode_id=episode_id,
        expected_transition_count=int(depths.shape[0]),
    )
    if _WORKER_ROUTE_STORE is None:
        raise RuntimeError("depth-mask worker route store is not initialized")
    metadata = dict(episode["metadata"])
    try:
        route_index = int(float(metadata["global_route_index"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("{} rollout is missing global_route_index".format(path)) from error
    goal = goal.reshape(3)
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
    _WORKER_ROUTE_STORE.validate(route_index, goal=goal)
    masks = np.empty((depths.shape[0], _WORKER_MPL.num_actions), dtype=np.bool_)
    for index, depth in enumerate(depths):
        depth_m = normalized_depth_to_metric(depth, _WORKER_DEPTH_MIN_M, _WORKER_DEPTH_MAX_M)
        masks[index], _ = local_depth_action_mask(_WORKER_MPL, depth_m, _WORKER_CONFIG)
    if _WORKER_DIRECT_MASKS is not None:
        if direct_offset is None or direct_length is None:
            raise ValueError("direct depth-mask output requires an episode offset and length")
        if direct_length != int(masks.shape[0]):
            raise ValueError(
                "direct depth-mask length {} != computed {} for {}".format(
                    direct_length, masks.shape[0], path
                )
            )
        target = slice(direct_offset, direct_offset + int(masks.shape[0]))
        _WORKER_DIRECT_MASKS[target] = masks
        return int(episode_id), str(path), int(masks.shape[0])
    return int(episode_id), str(path), masks
