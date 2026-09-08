"""Safe exact-distance mission sampling with start, goal, and map-bound checks."""

from __future__ import annotations
import math
import multiprocessing as mp
import os
from typing import Callable, Dict, Iterator, List, Optional, Sequence
import numpy as np

from planning.mission.spec import ACTION_MASK_Z_MARGIN_M, DEFAULT_ALTITUDE_LEVELS_M, FLIGHT_Z_MAX_M, FLIGHT_Z_MIN_M, planar_distance_xy

_PARALLEL_SAMPLER = {}

DEFAULT_MAX_SAMPLING_ATTEMPTS = 200000
DEFAULT_SAMPLING_ATTEMPT_FACTOR = 4

def resolve_sampling_max_attempts(count: int, max_attempts: int = 0) -> int:
    """Return a scale-safe attempt budget without changing sampling criteria."""
    requested = int(count)
    configured = int(max_attempts)
    if requested <= 0:
        raise ValueError("mission count must be positive")
    if configured < 0:
        raise ValueError("max sampling attempts must be non-negative")
    if configured > 0:
        if configured < requested:
            raise ValueError(
                "max sampling attempts cannot be smaller than mission count: "
                "{} < {}".format(configured, requested)
            )
        return configured
    return max(
        int(DEFAULT_MAX_SAMPLING_ATTEMPTS),
        requested * int(DEFAULT_SAMPLING_ATTEMPT_FACTOR),
    )

def _candidate(
    rng,
    episode_id: int,
    start_x_range: Sequence[float],
    start_y_range: Sequence[float],
    goal_distance: float,
    goal_y_offset_range: Sequence[float],
    altitude_levels: Sequence[float],
) -> Dict:
    x0 = float(rng.uniform(*start_x_range))
    y0 = float(rng.uniform(*start_y_range))
    z0 = float(rng.choice(altitude_levels))
    goal_z = float(rng.choice(altitude_levels))
    lateral = float(rng.uniform(*goal_y_offset_range))
    if goal_distance <= 0.0 or abs(lateral) >= goal_distance:
        raise ValueError("invalid goal distance or lateral offset range")
    goal_x = x0 + math.sqrt(goal_distance * goal_distance - lateral * lateral)
    goal_y = y0 + lateral
    if abs(planar_distance_xy(x0, y0, goal_x, goal_y) - goal_distance) > 1.0e-6:
        raise RuntimeError("sampled planar distance mismatch")
    return {
        "episode_id": int(episode_id),
        "start": [x0, y0, z0, 0.0],
        "goal": [goal_x, goal_y, goal_z],
    }

def _height_global_valid_count(mpl, checker, position: Sequence[float], yaw: float, check_step: int) -> int:
    height = mpl.valid_action_mask(
        float(position[2]), z_min=FLIGHT_Z_MIN_M, z_max=FLIGHT_Z_MAX_M, margin=ACTION_MASK_Z_MARGIN_M
    )
    collision = checker.check_all_actions_array(
        mpl, position[:3], float(yaw), check_step=max(1, int(check_step))
    )
    global_mask = np.asarray(collision["valid"], dtype=np.bool_)
    return int(np.count_nonzero(height & global_mask))

def _point_clear(checker, position: Sequence[float]) -> bool:
    point = np.asarray(position, dtype=np.float32).reshape(1, 3)
    return bool(checker.check_path(point, check_step=1).get("valid", False))

def _inside_map_bounds(checker, position: Sequence[float], margin_m: float) -> bool:
    if not all(hasattr(checker, name) for name in ("origin_ijk", "grid_shape", "voxel_size")):
        return True
    origin = np.asarray(checker.origin_ijk, dtype=np.float64)
    shape = np.asarray(checker.grid_shape, dtype=np.float64)
    voxel = float(checker.voxel_size)
    point = np.asarray(position, dtype=np.float64).reshape(3)
    lower = origin * voxel
    upper = (origin + shape) * voxel

    # Horizontal sampling needs a margin so a 40 m mission and its local
    # primitives stay inside the map. The voxel map is shallow in Z, so using
    # the same 3 m margin vertically would make the 1-3 m flight band empty.
    xy_lower = lower[:2] + float(margin_m)
    xy_upper = upper[:2] - float(margin_m)
    return bool(
        np.all(point[:2] >= xy_lower)
        and np.all(point[:2] <= xy_upper)
        and lower[2] <= point[2] <= upper[2]
    )

def _mission_is_safe(
    mission: Dict,
    mpl,
    checker,
    min_start_valid_actions: int,
    min_goal_valid_actions: int,
    map_margin_m: float,
    check_step: int,
) -> bool:
    start = mission["start"]
    goal = mission["goal"]
    if not _inside_map_bounds(checker, start[:3], map_margin_m):
        return False
    if not _inside_map_bounds(checker, goal, map_margin_m):
        return False
    if not _point_clear(checker, start[:3]) or not _point_clear(checker, goal):
        return False

    start_valid = _height_global_valid_count(mpl, checker, start[:3], math.radians(start[3]), check_step)
    if start_valid < int(min_start_valid_actions):
        return False
    goal_valid = _height_global_valid_count(mpl, checker, goal, 0.0, check_step)
    if goal_valid < int(min_goal_valid_actions):
        return False

    mission["start_valid_action_count"] = int(start_valid)
    mission["goal_valid_action_count"] = int(goal_valid)
    return True

def _parallel_worker_init(config: Dict) -> None:
    """Load immutable map/MPL state once per CPU worker."""
    from planning.native.geometry import NativeGeometryContext
    from planning.primitives.library import MotionPrimitiveLibrary

    _PARALLEL_SAMPLER["mpl"] = MotionPrimitiveLibrary()
    _PARALLEL_SAMPLER["geometry_context"] = NativeGeometryContext.from_voxel_cache(
        config["cache_npz"],
        voxel_size=float(config["voxel_size"]),
    )
    _PARALLEL_SAMPLER["checker"] = _PARALLEL_SAMPLER[
        "geometry_context"
    ].collision_checker(float(config["inflate_radius"]))
    _PARALLEL_SAMPLER["config"] = config

def _parallel_sample_chunk(task):
    start_attempt, attempts, seed, canonical_first = task
    config = _PARALLEL_SAMPLER["config"]
    mpl = _PARALLEL_SAMPLER["mpl"]
    checker = _PARALLEL_SAMPLER["checker"]
    rng = np.random.default_rng(int(seed))
    accepted = []
    for local_attempt in range(int(attempts)):
        if bool(canonical_first) and local_attempt == 0:
            mission = {
                "episode_id": -1,
                "start": [0.0, 0.0, 2.0, 0.0],
                "goal": [float(config["goal_distance"]), 0.0, 2.0],
            }
        else:
            mission = _candidate(
                rng,
                -1,
                config["start_x_range"],
                config["start_y_range"],
                float(config["goal_distance"]),
                config["goal_y_offset_range"],
                config["altitude_levels"],
            )
        if _mission_is_safe(
            mission,
            mpl,
            checker,
            int(config["min_start_valid_actions"]),
            int(config["min_goal_valid_actions"]),
            float(config["map_margin_m"]),
            int(config["check_step"]),
        ):
            mission["sample_attempts"] = int(start_attempt + local_attempt + 1)
            accepted.append(mission)
    return accepted

def _parallel_sample_safe_missions(
    count: int,
    seed: int,
    cache_npz: str,
    voxel_size: float,
    inflate_radius: float,
    start_x_range: Sequence[float],
    start_y_range: Sequence[float],
    goal_distance: float,
    goal_y_offset_range: Sequence[float],
    altitude_levels: Sequence[float],
    min_start_valid_actions: int,
    min_goal_valid_actions: int,
    map_margin_m: float,
    check_step: int,
    max_attempts: int,
    num_workers: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict]:
    config = {
        "cache_npz": str(cache_npz),
        "voxel_size": float(voxel_size),
        "inflate_radius": float(inflate_radius),
        "start_x_range": tuple(float(value) for value in start_x_range),
        "start_y_range": tuple(float(value) for value in start_y_range),
        "goal_distance": float(goal_distance),
        "goal_y_offset_range": tuple(float(value) for value in goal_y_offset_range),
        "altitude_levels": tuple(float(value) for value in altitude_levels),
        "min_start_valid_actions": int(min_start_valid_actions),
        "min_goal_valid_actions": int(min_goal_valid_actions),
        "map_margin_m": float(map_margin_m),
        "check_step": int(check_step),
    }
    worker_count = max(1, min(int(num_workers), int(os.cpu_count() or 1)))
    chunk_attempts = 128
    missions: List[Dict] = []
    attempts = 0
    batch_index = 0
    context = mp.get_context("fork")
    with context.Pool(processes=worker_count, initializer=_parallel_worker_init, initargs=(config,)) as pool:
        while len(missions) < int(count) and attempts < int(max_attempts):
            remaining = int(max_attempts) - attempts
            task_count = min(worker_count, int(math.ceil(remaining / float(chunk_attempts))))
            tasks = []
            for task_index in range(task_count):
                task_attempts = min(chunk_attempts, remaining - task_index * chunk_attempts)
                task_start = attempts + task_index * chunk_attempts
                tasks.append(
                    (
                        task_start,
                        task_attempts,
                        int(seed) + (batch_index * worker_count + task_index) * 1000003,
                        task_start == 0,
                    )
                )
            for accepted in pool.map(_parallel_sample_chunk, tasks):
                for mission in accepted:
                    if len(missions) >= int(count):
                        break
                    mission["episode_id"] = int(len(missions))
                    missions.append(mission)
            attempts += sum(int(task[1]) for task in tasks)
            batch_index += 1
            if progress_callback is not None:
                progress_callback(attempts, len(missions))
    if len(missions) < int(count):
        raise RuntimeError(
            "safe mission sampler exhausted {} attempts after {} missions".format(max_attempts, len(missions))
        )
    return missions

def sample_safe_missions(
    count: int,
    seed: int,
    mpl,
    checker,
    start_x_range: Sequence[float] = (-50.0, 50.0),
    start_y_range: Sequence[float] = (-50.0, 50.0),
    goal_distance: float = 40.0,
    goal_y_offset_range: Sequence[float] = (-1.5, 1.5),
    altitude_levels: Sequence[float] = DEFAULT_ALTITUDE_LEVELS_M,
    min_start_valid_actions: int = 40,
    min_goal_valid_actions: int = 10,
    map_margin_m: float = 3.0,
    check_step: int = 2,
    max_attempts: int = 0,
    num_workers: int = 1,
    cache_npz: str = "",
    voxel_size: float = 0.10,
    inflate_radius: float = 0.35,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict]:
    resolved_max_attempts = resolve_sampling_max_attempts(count, max_attempts)
    if not altitude_levels or any(
        not FLIGHT_Z_MIN_M <= float(z) <= FLIGHT_Z_MAX_M for z in altitude_levels
    ):
        raise ValueError("invalid altitude levels")

    if int(num_workers) > 1:
        if not str(cache_npz).strip():
            raise ValueError("cache_npz is required when num_workers > 1")
        return _parallel_sample_safe_missions(
            count=count,
            seed=seed,
            cache_npz=cache_npz,
            voxel_size=voxel_size,
            inflate_radius=inflate_radius,
            start_x_range=start_x_range,
            start_y_range=start_y_range,
            goal_distance=goal_distance,
            goal_y_offset_range=goal_y_offset_range,
            altitude_levels=altitude_levels,
            min_start_valid_actions=min_start_valid_actions,
            min_goal_valid_actions=min_goal_valid_actions,
            map_margin_m=map_margin_m,
            check_step=check_step,
            max_attempts=resolved_max_attempts,
            num_workers=num_workers,
            progress_callback=progress_callback,
        )

    rng = np.random.default_rng(int(seed))
    missions: List[Dict] = []
    attempts = 0
    while len(missions) < int(count):
        if attempts >= int(resolved_max_attempts):
            raise RuntimeError(
                "safe mission sampler exhausted {} attempts after {} missions".format(
                    resolved_max_attempts, len(missions)
                )
            )
        attempts += 1
        # Try the canonical origin mission once for easy smoke tests. If it is
        # not valid in the current map, fall back to normal random sampling
        # instead of retrying the same invalid mission forever.
        if attempts == 1 and not missions:
            mission = {
                "episode_id": 0,
                "start": [0.0, 0.0, 2.0, 0.0],
                "goal": [float(goal_distance), 0.0, 2.0],
            }
        else:
            mission = _candidate(
                rng,
                len(missions),
                start_x_range,
                start_y_range,
                float(goal_distance),
                goal_y_offset_range,
                altitude_levels,
            )

        safe = _mission_is_safe(
            mission,
            mpl,
            checker,
            min_start_valid_actions,
            min_goal_valid_actions,
            map_margin_m,
            check_step,
        )
        if safe:
            mission["sample_attempts"] = int(attempts)
            missions.append(mission)
        if progress_callback is not None and (
            attempts % 1000 == 0 or len(missions) >= int(count)
        ):
            progress_callback(attempts, len(missions))
    return missions


def iter_raw_mission_candidates(
    *,
    seed: int,
    max_attempts: int = 0,
    max_accepted: Optional[int] = None,
    start_x_range: Sequence[float] = (-50.0, 50.0),
    start_y_range: Sequence[float] = (-50.0, 50.0),
    goal_distance: float = 40.0,
    goal_y_offset_range: Sequence[float] = (-1.5, 1.5),
    altitude_levels: Sequence[float] = DEFAULT_ALTITUDE_LEVELS_M,
    include_canonical_first: bool = True,
) -> Iterator[Dict]:
    """Yield the deterministic candidate stream before safety evaluation.

    This is intentionally only the raw construction half of
    :func:`iter_safe_missions`.  The preparation parent may advance this
    stream, while workers own the safety predicate and all native resources.
    Keeping the construction here preserves the historical RNG draw order.
    """

    if not altitude_levels or any(
        not FLIGHT_Z_MIN_M <= float(z) <= FLIGHT_Z_MAX_M for z in altitude_levels
    ):
        raise ValueError("invalid altitude levels")
    target = max(1, int(max_accepted or 1))
    resolved_max_attempts = resolve_sampling_max_attempts(target, int(max_attempts))
    rng = np.random.default_rng(int(seed))
    for attempt in range(1, int(resolved_max_attempts) + 1):
        # Match the canonical first-attempt smoke mission in the safe sampler.
        if attempt == 1 and bool(include_canonical_first):
            mission = {
                "episode_id": 0,
                "start": [0.0, 0.0, 2.0, 0.0],
                "goal": [float(goal_distance), 0.0, 2.0],
            }
        else:
            mission = _candidate(
                rng,
                -1,
                start_x_range,
                start_y_range,
                float(goal_distance),
                goal_y_offset_range,
                altitude_levels,
            )
        mission["sample_attempts"] = int(attempt)
        yield mission


def iter_safe_missions(
    *,
    seed: int,
    mpl,
    checker,
    start_x_range: Sequence[float] = (-50.0, 50.0),
    start_y_range: Sequence[float] = (-50.0, 50.0),
    goal_distance: float = 40.0,
    goal_y_offset_range: Sequence[float] = (-1.5, 1.5),
    altitude_levels: Sequence[float] = DEFAULT_ALTITUDE_LEVELS_M,
    min_start_valid_actions: int = 40,
    min_goal_valid_actions: int = 10,
    map_margin_m: float = 3.0,
    check_step: int = 2,
    max_attempts: int = 0,
    max_accepted: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    include_canonical_first: bool = True,
) -> Iterator[Dict]:
    """Yield locally safe candidates without materialising the candidate set.

    The candidate RNG and acceptance predicate are the same as the historical
    single-process sampler.  This iterator is the bounded preparation seam;
    callers decide when to stop after route and Teacher audit outcomes.
    """

    if not altitude_levels or any(
        not FLIGHT_Z_MIN_M <= float(z) <= FLIGHT_Z_MAX_M for z in altitude_levels
    ):
        raise ValueError("invalid altitude levels")
    target = None if max_accepted is None else int(max_accepted)
    if target is not None and target < 0:
        raise ValueError("max_accepted must be non-negative")
    resolved_max_attempts = resolve_sampling_max_attempts(
        max(1, target or 1), max_attempts
    ) if not max_attempts else int(max_attempts)
    rng = np.random.default_rng(int(seed))
    attempts = 0
    accepted = 0
    while (target is None or accepted < target) and attempts < resolved_max_attempts:
        attempts += 1
        if attempts == 1 and bool(include_canonical_first):
            mission = {
                "episode_id": 0,
                "start": [0.0, 0.0, 2.0, 0.0],
                "goal": [float(goal_distance), 0.0, 2.0],
            }
        else:
            mission = _candidate(
                rng,
                accepted,
                start_x_range,
                start_y_range,
                float(goal_distance),
                goal_y_offset_range,
                altitude_levels,
            )
        if _mission_is_safe(
            mission,
            mpl,
            checker,
            min_start_valid_actions,
            min_goal_valid_actions,
            map_margin_m,
            check_step,
        ):
            mission["episode_id"] = int(accepted)
            mission["sample_attempts"] = int(attempts)
            accepted += 1
            yield mission
        if progress_callback is not None and (
            attempts % 1000 == 0 or (target is not None and accepted >= target)
        ):
            progress_callback(attempts, accepted)
    if target is not None and accepted < target:
        raise RuntimeError(
            "safe mission sampler exhausted {} attempts after {} missions".format(
                resolved_max_attempts, accepted
            )
        )
