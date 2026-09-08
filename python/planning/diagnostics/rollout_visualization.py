#!/usr/bin/env python3
"""Publication-quality rollout visualization for the planning project.

The script reads the project's native rollout NPZ files directly.  It can
produce:

* a 4-panel diagnostic PNG at 600 DPI by default;
* an optional vector PDF;
* an interactive Plotly 3-D HTML scene;
* a top-view planning animation (MP4 or GIF);
* overlays of additional rollout NPZ files for policy comparison.

No Agile Autonomy CSV conversion is required.

Typical use:

  rosrun planning visualize_rollout.py \
    --rollout-index data/teach/<run_id>/rollout_index.csv \
    --episode-id 123 \
    --labels data/teach/<run_id>/teacher_labels.npz \
    --point-cloud data/map_data/forest_point_cloud.bin \
    --point-cloud-frame unity \
    --mode all \
    --dpi 600 \
    --out-dir data/smoke/visualizations

The static figure is saved at the requested DPI.  Interactive HTML does not
have a DPI because its trajectories remain vector graphics in the browser.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from planning.primitives.library import rotation_z

# Matplotlib is imported lazily enough to keep --help responsive, but it is a
# required dependency for static and animation modes.
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.lines import Line2D


NUM_ACTIONS = 105
DEFAULT_DPI = 600


@dataclass
class EpisodeData:
    path: Path
    episode_id: int
    poses_before: np.ndarray          # [T,4] = x,y,z,yaw in ROS map frame
    velocities_before: np.ndarray     # [T,3]
    behavior_actions: np.ndarray      # [T]
    prev_actions: np.ndarray          # [T]
    execution_masks: np.ndarray       # [T,105]
    height_masks: np.ndarray          # [T,105]
    sensor_skew_ns: np.ndarray        # [T]
    start: np.ndarray                 # [3]
    goal: np.ndarray                  # [3]
    metadata: Dict
    index_row: Dict


@dataclass
class EpisodeLabels:
    soft_targets: np.ndarray          # [T,105]
    global_masks: np.ndarray          # [T,105]
    teacher_argmax: np.ndarray        # [T]
    valid_counts: np.ndarray          # [T]
    teacher_scores: Optional[np.ndarray]
    metadata: Dict


@dataclass
class MplData:
    path: Path
    pos_ref: np.ndarray               # [105,N,3]

    @property
    def num_actions(self) -> int:
        return int(self.pos_ref.shape[0])


@dataclass
class MapData:
    points: np.ndarray                # [N,3] ROS map frame
    source: str


@dataclass
class ComparisonTrajectory:
    name: str
    points: np.ndarray                # [N,3]


def _find_planning_root() -> Path:
    """Find the planning package independently of the command's working directory."""
    from planning.common.paths import planning_package_root

    return planning_package_root()


def _resolve_path(value: str, package_root: Path, required: bool = True) -> Optional[Path]:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        local = Path.cwd() / path
        package_relative = package_root / path
        if local.exists():
            path = local
        else:
            path = package_relative
    path = path.resolve()
    if required and not path.exists():
        raise FileNotFoundError(str(path))
    return path


def _metadata_from_npz(value: np.ndarray) -> Dict:
    if value is None:
        return {}
    raw = value
    if isinstance(raw, np.ndarray):
        raw = raw.item() if raw.ndim == 0 else raw.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not raw:
        return {}
    try:
        return json.loads(str(raw))
    except Exception:
        return {"raw_metadata": str(raw)}


def _read_csv_rows(path: Path) -> List[Dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_dataset_path(row: Dict, index_path: Path) -> Path:
    raw = str(row.get("dataset_npz", "")).strip()
    if not raw:
        raise ValueError("rollout index row does not contain dataset_npz")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = index_path.parent / path
    return path.resolve()


def _episode_id_from_filename(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else -1


def _load_episode_npz(path: Path, index_row: Optional[Dict] = None) -> EpisodeData:
    index_row = dict(index_row or {})
    with np.load(str(path), allow_pickle=False) as data:
        files = set(data.files)
        required = {"poses_before", "behavior_actions", "start", "goal"}
        missing = sorted(required - files)
        if missing:
            raise KeyError("{} missing rollout keys {}".format(path, missing))

        poses = np.asarray(data["poses_before"], dtype=np.float32)
        actions = np.asarray(data["behavior_actions"], dtype=np.int64).reshape(-1)
        transition_count = int(actions.shape[0])
        if poses.shape != (transition_count, 4):
            raise ValueError("poses_before shape {} != ({},4)".format(poses.shape, transition_count))

        velocities = (
            np.asarray(data["velocities_before"], dtype=np.float32)
            if "velocities_before" in files
            else np.zeros((transition_count, 3), dtype=np.float32)
        )
        prev_actions = (
            np.asarray(data["prev_actions"], dtype=np.int64).reshape(-1)
            if "prev_actions" in files
            else np.full((transition_count,), -1, dtype=np.int64)
        )
        execution_masks = (
            np.asarray(data["execution_action_masks"], dtype=np.bool_)
            if "execution_action_masks" in files
            else np.ones((transition_count, NUM_ACTIONS), dtype=np.bool_)
        )
        height_masks = (
            np.asarray(data["height_action_masks"], dtype=np.bool_)
            if "height_action_masks" in files
            else execution_masks.copy()
        )
        sensor_skew_ns = (
            np.asarray(data["sensor_skew_ns"], dtype=np.int64).reshape(-1)
            if "sensor_skew_ns" in files
            else np.zeros((transition_count,), dtype=np.int64)
        )
        metadata = _metadata_from_npz(data["metadata_json"]) if "metadata_json" in files else {}
        start = np.asarray(data["start"], dtype=np.float32).reshape(3)
        goal = np.asarray(data["goal"], dtype=np.float32).reshape(3)

    if velocities.shape != (transition_count, 3):
        raise ValueError("velocities_before shape bad: {}".format(velocities.shape))
    if execution_masks.shape != (transition_count, NUM_ACTIONS):
        raise ValueError("execution_action_masks shape bad: {}".format(execution_masks.shape))

    episode_id = int(float(index_row.get("episode_id", metadata.get("episode_id", -1))))
    if episode_id < 0:
        episode_id = _episode_id_from_filename(path)

    return EpisodeData(
        path=path,
        episode_id=episode_id,
        poses_before=poses,
        velocities_before=velocities,
        behavior_actions=actions,
        prev_actions=prev_actions,
        execution_masks=execution_masks,
        height_masks=height_masks,
        sensor_skew_ns=sensor_skew_ns,
        start=start,
        goal=goal,
        metadata=metadata,
        index_row=index_row,
    )


def load_episode(
    episode_npz: Optional[Path],
    rollout_index: Optional[Path],
    episode_id: Optional[int],
    row_index: int,
) -> EpisodeData:
    if episode_npz is not None:
        return _load_episode_npz(episode_npz)
    if rollout_index is None:
        raise ValueError("provide --episode-npz or --rollout-index")

    rows = _read_csv_rows(rollout_index)
    if not rows:
        raise RuntimeError("empty rollout index: {}".format(rollout_index))

    selected: Optional[Dict] = None
    if episode_id is not None:
        for row in rows:
            try:
                if int(float(row.get("episode_id", -1))) == int(episode_id):
                    selected = row
                    break
            except Exception:
                continue
        if selected is None:
            raise KeyError("episode_id {} not found in {}".format(episode_id, rollout_index))
    else:
        if not 0 <= int(row_index) < len(rows):
            raise IndexError("row-index {} outside [0,{})".format(row_index, len(rows)))
        selected = rows[int(row_index)]

    path = _resolve_dataset_path(selected, rollout_index)
    return _load_episode_npz(path, selected)


def load_episode_labels(labels_path: Optional[Path], episode: EpisodeData) -> Optional[EpisodeLabels]:
    if labels_path is None:
        return None

    with np.load(str(labels_path), allow_pickle=False) as data:
        files = set(data.files)
        required = {
            "soft_targets",
            "global_action_masks",
            "teacher_argmax",
            "valid_counts",
            "episode_npz_paths",
            "episode_offsets",
            "episode_lengths",
        }
        missing = sorted(required - files)
        if missing:
            raise KeyError("{} missing label keys {}".format(labels_path, missing))

        paths = [
            str((labels_path.parent / Path(str(value))).resolve())
            if not Path(str(value)).expanduser().is_absolute()
            else str(Path(str(value)).expanduser().resolve())
            for value in data["episode_npz_paths"].tolist()
        ]
        offsets = np.asarray(data["episode_offsets"], dtype=np.int64)
        lengths = np.asarray(data["episode_lengths"], dtype=np.int64)
        target = str(episode.path.resolve())

        match = None
        for i, value in enumerate(paths):
            if value == target:
                match = i
                break
        if match is None:
            basename_matches = [i for i, value in enumerate(paths) if Path(value).name == episode.path.name]
            if len(basename_matches) == 1:
                match = basename_matches[0]
        if match is None:
            raise KeyError("episode {} not found in label store {}".format(episode.path, labels_path))

        offset = int(offsets[match])
        length = int(lengths[match])
        if length != episode.behavior_actions.shape[0]:
            raise ValueError(
                "label length {} != rollout transitions {}".format(length, episode.behavior_actions.shape[0])
            )
        sl = slice(offset, offset + length)
        soft_targets = np.asarray(data["soft_targets"][sl], dtype=np.float32)
        global_masks = np.asarray(data["global_action_masks"][sl], dtype=np.bool_)
        teacher_argmax = np.asarray(data["teacher_argmax"][sl], dtype=np.int64)
        valid_counts = np.asarray(data["valid_counts"][sl], dtype=np.int64)
        scores = np.asarray(data["teacher_scores"][sl], dtype=np.float32) if "teacher_scores" in files else None
        metadata = _metadata_from_npz(data["metadata_json"]) if "metadata_json" in files else {}

    return EpisodeLabels(
        soft_targets=soft_targets,
        global_masks=global_masks,
        teacher_argmax=teacher_argmax,
        valid_counts=valid_counts,
        teacher_scores=scores,
        metadata=metadata,
    )


def load_mpl(npz_path: Path) -> MplData:
    with np.load(str(npz_path), allow_pickle=False) as data:
        if "pos_ref" not in data:
            raise KeyError("{} does not contain pos_ref".format(npz_path))
        pos_ref = np.asarray(data["pos_ref"], dtype=np.float32)
    if pos_ref.ndim != 3 or pos_ref.shape[0] != NUM_ACTIONS or pos_ref.shape[2] != 3:
        raise ValueError("MPL pos_ref shape bad: {}".format(pos_ref.shape))
    return MplData(path=npz_path, pos_ref=pos_ref)



def transform_primitive(local_path: np.ndarray, pose_xyzyaw: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_xyzyaw, dtype=np.float32).reshape(4)
    return np.asarray(local_path, dtype=np.float32) @ rotation_z(float(pose[3])).T + pose[:3]


def nominal_behavior_segments(episode: EpisodeData, mpl: MplData) -> List[np.ndarray]:
    output: List[np.ndarray] = []
    for pose, action in zip(episode.poses_before, episode.behavior_actions):
        if 0 <= int(action) < mpl.num_actions:
            output.append(transform_primitive(mpl.pos_ref[int(action)], pose))
    return output


def trajectory_points(episode: EpisodeData, mpl: Optional[MplData], append_nominal_endpoint: bool = True) -> np.ndarray:
    points = np.asarray(episode.poses_before[:, :3], dtype=np.float32)
    if append_nominal_endpoint and mpl is not None and points.shape[0] > 0:
        action = int(episode.behavior_actions[-1])
        if 0 <= action < mpl.num_actions:
            endpoint = transform_primitive(mpl.pos_ref[action], episode.poses_before[-1])[-1]
            points = np.vstack([points, endpoint.reshape(1, 3)])
    return points


def _generic_comparison_trajectory(path: Path, name: str, mpl: Optional[MplData]) -> ComparisonTrajectory:
    with np.load(str(path), allow_pickle=False) as data:
        files = set(data.files)
        if "poses_before" in files:
            poses = np.asarray(data["poses_before"], dtype=np.float32)
            points = poses[:, :3]
            if mpl is not None and "behavior_actions" in files and poses.shape[0] > 0:
                actions = np.asarray(data["behavior_actions"], dtype=np.int64).reshape(-1)
                action = int(actions[-1])
                if 0 <= action < mpl.num_actions:
                    points = np.vstack([points, transform_primitive(mpl.pos_ref[action], poses[-1])[-1]])
        else:
            key = None
            for candidate in ("positions", "trajectory", "poses", "path"):
                if candidate in files:
                    key = candidate
                    break
            if key is None:
                raise KeyError("{} has no supported trajectory key".format(path))
            raw = np.asarray(data[key], dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] < 3:
                raise ValueError("{} {} shape bad: {}".format(path, key, raw.shape))
            points = raw[:, :3]
    return ComparisonTrajectory(name=name, points=np.asarray(points, dtype=np.float32))


def _roi_bounds(trajectories: Sequence[np.ndarray], start: np.ndarray, goal: np.ndarray, padding: float) -> Tuple[float, float, float, float]:
    all_xy = [np.asarray(start[:2]).reshape(1, 2), np.asarray(goal[:2]).reshape(1, 2)]
    for trajectory in trajectories:
        if trajectory.size:
            all_xy.append(np.asarray(trajectory[:, :2], dtype=np.float32))
    xy = np.concatenate(all_xy, axis=0)
    return (
        float(np.min(xy[:, 0]) - padding),
        float(np.max(xy[:, 0]) + padding),
        float(np.min(xy[:, 1]) - padding),
        float(np.max(xy[:, 1]) + padding),
    )


def _deterministic_downsample(points: np.ndarray, max_points: int, seed: int = 2026) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(points.shape[0], size=int(max_points), replace=False)
    return points[np.sort(indices)]


def _filter_points(points: np.ndarray, bounds: Tuple[float, float, float, float], z_min: float, z_max: float) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    mask = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
        & (points[:, 2] >= float(z_min))
        & (points[:, 2] <= float(z_max))
    )
    return points[mask]


def _load_forest_bin(
    path: Path,
    frame: str,
    bounds: Tuple[float, float, float, float],
    z_min: float,
    z_max: float,
    max_points: int,
) -> np.ndarray:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(4)
    if len(header) != 4:
        raise ValueError("point-cloud file has no 4-byte header: {}".format(path))
    count = int(struct.unpack("<i", header)[0])
    if count < 0:
        raise ValueError("negative point count in {}".format(path))
    expected = 4 + count * 12
    if expected > file_size:
        raise ValueError("point-cloud payload shorter than header count: {}".format(path))

    raw = np.memmap(str(path), dtype=np.float32, mode="r", offset=4, shape=(count, 3))
    selected: List[np.ndarray] = []
    chunk_size = 1_000_000
    for begin in range(0, count, chunk_size):
        points = np.asarray(raw[begin : min(count, begin + chunk_size)], dtype=np.float32)
        if frame == "unity":
            # Unity world -> ROS map: [x_ros,y_ros,z_ros] = [z_unity,-x_unity,y_unity]
            points = np.column_stack([points[:, 2], -points[:, 0], points[:, 1]]).astype(np.float32)
        elif frame != "ros":
            raise ValueError("point-cloud frame must be unity or ros")
        points = _filter_points(points, bounds, z_min, z_max)
        if points.size:
            selected.append(points)
    if not selected:
        return np.empty((0, 3), dtype=np.float32)
    return _deterministic_downsample(np.concatenate(selected, axis=0), max_points)


def _load_voxel_cache(
    path: Path,
    bounds: Tuple[float, float, float, float],
    z_min: float,
    z_max: float,
    max_points: int,
) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as data:
        keys = np.asarray(data["occupied_keys"], dtype=np.int64)
        origin = np.asarray(data["origin_ijk"], dtype=np.int64).reshape(3)
        shape = np.asarray(data["grid_shape"], dtype=np.int64).reshape(3)
        voxel_size = float(data["voxel_size"])

    stride_x = np.int64(shape[1] * shape[2])
    stride_y = np.int64(shape[2])
    rel_x = keys // stride_x
    remainder = keys - rel_x * stride_x
    rel_y = remainder // stride_y
    rel_z = remainder - rel_y * stride_y
    ijk = np.column_stack([rel_x, rel_y, rel_z]).astype(np.int64) + origin.reshape(1, 3)
    points = (ijk.astype(np.float32) + 0.5) * float(voxel_size)
    points = _filter_points(points, bounds, z_min, z_max)
    return _deterministic_downsample(points, max_points)


def _load_open3d_cloud(
    path: Path,
    bounds: Tuple[float, float, float, float],
    z_min: float,
    z_max: float,
    max_points: int,
) -> np.ndarray:
    try:
        import open3d as o3d  # type: ignore
    except Exception as error:
        raise RuntimeError("Open3D is required to read {}: {}".format(path.suffix, error))
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float32)
    points = _filter_points(points, bounds, z_min, z_max)
    return _deterministic_downsample(points, max_points)


def load_map(
    point_cloud: Optional[Path],
    voxel_cache: Optional[Path],
    point_cloud_frame: str,
    bounds: Tuple[float, float, float, float],
    z_min: float,
    z_max: float,
    max_points: int,
) -> MapData:
    if point_cloud is not None:
        suffix = point_cloud.suffix.lower()
        if suffix == ".bin":
            points = _load_forest_bin(point_cloud, point_cloud_frame, bounds, z_min, z_max, max_points)
        elif suffix in (".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"):
            points = _load_open3d_cloud(point_cloud, bounds, z_min, z_max, max_points)
        elif suffix == ".npy":
            points = _filter_points(np.load(str(point_cloud)).astype(np.float32), bounds, z_min, z_max)
            points = _deterministic_downsample(points, max_points)
        else:
            raise ValueError("unsupported point-cloud format: {}".format(point_cloud))
        return MapData(points=points, source=str(point_cloud))

    if voxel_cache is not None:
        return MapData(
            points=_load_voxel_cache(voxel_cache, bounds, z_min, z_max, max_points),
            source=str(voxel_cache),
        )
    return MapData(points=np.empty((0, 3), dtype=np.float32), source="none")


def make_height_raster(
    points: np.ndarray,
    bounds: Tuple[float, float, float, float],
    resolution: float,
    max_cells: int = 4_000_000,
) -> Tuple[np.ma.MaskedArray, Tuple[float, float, float, float]]:
    xmin, xmax, ymin, ymax = bounds
    width = max(1.0e-6, xmax - xmin)
    height = max(1.0e-6, ymax - ymin)
    resolution = max(0.01, float(resolution))
    nx = max(2, int(math.ceil(width / resolution)))
    ny = max(2, int(math.ceil(height / resolution)))
    if nx * ny > int(max_cells):
        scale = math.sqrt((nx * ny) / float(max_cells))
        nx = max(2, int(nx / scale))
        ny = max(2, int(ny / scale))

    grid = np.full((ny, nx), -np.inf, dtype=np.float32)
    if points.size:
        ix = np.floor((points[:, 0] - xmin) / width * nx).astype(np.int64)
        iy = np.floor((points[:, 1] - ymin) / height * ny).astype(np.int64)
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        np.maximum.at(grid.ravel(), iy * nx + ix, points[:, 2])
    return np.ma.masked_where(~np.isfinite(grid), grid), (xmin, xmax, ymin, ymax)


def _select_step(step: int, episode: EpisodeData, labels: Optional[EpisodeLabels]) -> int:
    transition_count = int(episode.behavior_actions.shape[0])
    if transition_count <= 0:
        return 0
    if step >= 0:
        return min(int(step), transition_count - 1)
    if labels is not None:
        probabilities = np.clip(labels.soft_targets, 1.0e-12, 1.0)
        entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
        finite = entropy[np.isfinite(entropy)]
        if finite.size and float(np.max(finite) - np.min(finite)) > 1.0e-6:
            return int(np.nanargmax(entropy))
    return transition_count // 2


def _candidate_order(labels: Optional[EpisodeLabels], episode: EpisodeData, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if labels is not None:
        mask = labels.global_masks[step].astype(np.bool_)
        probabilities = labels.soft_targets[step].astype(np.float32)
        teacher_action = int(labels.teacher_argmax[step])
    else:
        mask = episode.execution_masks[step].astype(np.bool_)
        probabilities = np.zeros((NUM_ACTIONS,), dtype=np.float32)
        valid = np.flatnonzero(mask)
        if valid.size:
            probabilities[valid] = 1.0 / float(valid.size)
        teacher_action = -1
    behavior_action = int(episode.behavior_actions[step])
    order = np.argsort(probabilities)[::-1]
    return order, mask, probabilities, teacher_action, behavior_action


def _set_publication_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "figure.titlesize": 12.0,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_top_view(
    ax,
    raster: np.ma.MaskedArray,
    extent: Tuple[float, float, float, float],
    episode: EpisodeData,
    trajectory: np.ndarray,
    nominal_segments: Sequence[np.ndarray],
    comparisons: Sequence[ComparisonTrajectory],
    selected_step: int,
) -> None:
    if raster.size and np.any(~raster.mask):
        ax.imshow(
            raster,
            origin="lower",
            extent=extent,
            cmap="inferno",
            interpolation="nearest",
            alpha=0.92,
            aspect="equal",
            rasterized=True,
        )
    else:
        ax.set_facecolor("#f1f1f1")

    for segment in nominal_segments:
        ax.plot(segment[:, 0], segment[:, 1], color="#52d36b", alpha=0.22, linewidth=0.65)

    ax.plot(
        [episode.start[0], episode.goal[0]],
        [episode.start[1], episode.goal[1]],
        linestyle="--",
        color="#e43d30",
        linewidth=1.1,
        alpha=0.95,
        label="Straight reference",
    )
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="#00b944", linewidth=2.0, label="Rollout")
    boundary_points = episode.poses_before[:, :3]
    ax.scatter(
        boundary_points[:, 0],
        boundary_points[:, 1],
        s=7,
        c=np.arange(boundary_points.shape[0]),
        cmap="viridis",
        edgecolors="none",
        zorder=5,
        label="Primitive boundaries",
    )

    comparison_cmaps = ["winter", "cool", "plasma", "cividis"]
    for index, comparison in enumerate(comparisons):
        cmap = plt.get_cmap(comparison_cmaps[index % len(comparison_cmaps)])
        color = cmap(0.75)
        ax.plot(
            comparison.points[:, 0],
            comparison.points[:, 1],
            linewidth=1.6,
            alpha=0.95,
            color=color,
            label=comparison.name,
        )

    ax.scatter([episode.start[0]], [episode.start[1]], s=70, marker="o", color="#18c83e", edgecolor="black", linewidth=0.6, zorder=8, label="Start")
    ax.scatter([episode.goal[0]], [episode.goal[1]], s=75, marker="*", color="#ff2c24", edgecolor="black", linewidth=0.6, zorder=8, label="Goal")
    selected = episode.poses_before[selected_step]
    ax.scatter([selected[0]], [selected[1]], s=55, marker="D", color="#00e5ff", edgecolor="black", linewidth=0.7, zorder=9, label="Selected step")

    ax.set_title("Global top view")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    deduplicated = dict(zip(labels, handles))
    ax.legend(deduplicated.values(), deduplicated.keys(), loc="best", framealpha=0.85, ncol=2)


def _plot_candidate_scene(
    ax,
    map_points: np.ndarray,
    episode: EpisodeData,
    labels: Optional[EpisodeLabels],
    mpl: MplData,
    selected_step: int,
    local_radius: float,
    candidate_top_k: int,
    elevation: float,
    azimuth: float,
) -> None:
    pose = episode.poses_before[selected_step]
    order, mask, probabilities, teacher_action, behavior_action = _candidate_order(labels, episode, selected_step)

    if map_points.size:
        local_mask = (
            (np.abs(map_points[:, 0] - pose[0]) <= local_radius)
            & (np.abs(map_points[:, 1] - pose[1]) <= local_radius)
            & (map_points[:, 2] >= pose[2] - 2.5)
            & (map_points[:, 2] <= pose[2] + 4.0)
        )
        local_points = _deterministic_downsample(map_points[local_mask], 120_000, seed=selected_step + 2026)
        if local_points.size:
            ax.scatter(
                local_points[:, 0],
                local_points[:, 1],
                local_points[:, 2],
                c=local_points[:, 2],
                cmap="inferno",
                s=0.22,
                alpha=0.38,
                linewidths=0.0,
                depthshade=False,
                rasterized=True,
            )

    invalid_actions = np.flatnonzero(~mask)
    for action in invalid_actions:
        world = transform_primitive(mpl.pos_ref[int(action)], pose)
        ax.plot(world[:, 0], world[:, 1], world[:, 2], color="#dc3b38", alpha=0.055, linewidth=0.35)

    valid_actions = np.flatnonzero(mask)
    for action in valid_actions:
        world = transform_primitive(mpl.pos_ref[int(action)], pose)
        ax.plot(world[:, 0], world[:, 1], world[:, 2], color="#386cb0", alpha=0.10, linewidth=0.45)

    probability_norm = Normalize(vmin=0.0, vmax=max(1.0e-6, float(np.max(probabilities))))
    probability_cmap = plt.get_cmap("viridis")
    top_actions = [int(a) for a in order if mask[int(a)] and probabilities[int(a)] > 0.0][: max(1, int(candidate_top_k))]
    for action in reversed(top_actions):
        world = transform_primitive(mpl.pos_ref[action], pose)
        ax.plot(
            world[:, 0],
            world[:, 1],
            world[:, 2],
            color=probability_cmap(probability_norm(float(probabilities[action]))),
            alpha=0.75,
            linewidth=0.8 + 2.2 * probability_norm(float(probabilities[action])),
        )

    if 0 <= behavior_action < mpl.num_actions:
        world = transform_primitive(mpl.pos_ref[behavior_action], pose)
        ax.plot(world[:, 0], world[:, 1], world[:, 2], color="#ff9d00", alpha=1.0, linewidth=3.0)
    if 0 <= teacher_action < mpl.num_actions:
        world = transform_primitive(mpl.pos_ref[teacher_action], pose)
        ax.plot(world[:, 0], world[:, 1], world[:, 2], color="#1be800", alpha=1.0, linewidth=2.5)

    ax.scatter([pose[0]], [pose[1]], [pose[2]], s=35, color="black", depthshade=False)
    ax.scatter([episode.goal[0]], [episode.goal[1]], [episode.goal[2]], s=45, marker="*", color="#ff2c24", depthshade=False)
    ax.set_xlim(pose[0] - local_radius, pose[0] + local_radius)
    ax.set_ylim(pose[1] - local_radius, pose[1] + local_radius)
    ax.set_zlim(max(0.0, pose[2] - 2.0), pose[2] + 3.0)
    try:
        ax.set_box_aspect((2.0, 2.0, 0.8))
    except Exception:
        pass
    ax.view_init(elev=float(elevation), azim=float(azimuth))
    ax.set_xlabel("x [m]", labelpad=1)
    ax.set_ylabel("y [m]", labelpad=1)
    ax.set_zlabel("z [m]", labelpad=1)
    ax.set_title(
        "Candidate primitives at step {}\nbehavior={}  teacher={}  valid={}".format(
            selected_step,
            behavior_action,
            teacher_action if teacher_action >= 0 else "n/a",
            int(np.count_nonzero(mask)),
        )
    )

    legend_handles = [
        Line2D([0], [0], color="#386cb0", linewidth=1.2, alpha=0.6, label="Valid candidates"),
        Line2D([0], [0], color="#dc3b38", linewidth=1.2, alpha=0.6, label="Invalid candidates"),
        Line2D([0], [0], color="#ff9d00", linewidth=2.8, label="Behavior action"),
    ]
    if teacher_action >= 0:
        legend_handles.append(Line2D([0], [0], color="#1be800", linewidth=2.5, label="Teacher argmax"))
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.82)


def _plot_metrics(ax, episode: EpisodeData, labels: Optional[EpisodeLabels], selected_step: int) -> None:
    positions = episode.poses_before[:, :3]
    steps = np.arange(positions.shape[0])
    distance_xy = np.linalg.norm(positions[:, :2] - episode.goal[:2].reshape(1, 2), axis=1)
    speed = np.linalg.norm(episode.velocities_before, axis=1)
    valid_counts = (
        labels.valid_counts.astype(np.float32)
        if labels is not None
        else episode.execution_masks.sum(axis=1).astype(np.float32)
    )

    ax.plot(steps, distance_xy, linewidth=1.7, label="Goal distance [m]")
    ax.plot(steps, speed, linewidth=1.1, linestyle="--", label="Speed [m/s]")
    ax.axvline(selected_step, linewidth=0.9, linestyle=":", color="black")
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Distance / speed")
    ax.grid(True, alpha=0.22)

    ax2 = ax.twinx()
    ax2.plot(steps, valid_counts, linewidth=1.25, alpha=0.85, label="Valid actions")
    ax2.set_ylabel("Valid action count")
    ax2.set_ylim(bottom=0, top=max(NUM_ACTIONS, float(np.max(valid_counts)) * 1.05))

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="best", framealpha=0.85)
    skew_ms = episode.sensor_skew_ns.astype(np.float64) * 1.0e-6
    summary = "final boundary d={:.2f} m | max speed={:.2f} m/s | max sensor skew={:.1f} ms".format(
        float(distance_xy[-1]),
        float(np.max(speed)) if speed.size else 0.0,
        float(np.max(skew_ms)) if skew_ms.size else 0.0,
    )
    ax.set_title("Closed-loop metrics\n" + summary)


def _plot_action_heatmap(ax, episode: EpisodeData, labels: Optional[EpisodeLabels], selected_step: int) -> None:
    transition_count = int(episode.behavior_actions.shape[0])
    if labels is not None:
        matrix = labels.soft_targets.T
        vmax = max(1.0e-5, float(np.max(matrix)))
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            norm=PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax),
            extent=(-0.5, transition_count - 0.5, -0.5, NUM_ACTIONS - 0.5),
            rasterized=True,
        )
        colorbar = plt.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        colorbar.set_label("Soft-target probability")
        ax.scatter(np.arange(transition_count), episode.behavior_actions, s=13, facecolors="none", edgecolors="cyan", linewidths=0.7, label="Behavior")
        valid_teacher = labels.teacher_argmax >= 0
        ax.scatter(
            np.arange(transition_count)[valid_teacher],
            labels.teacher_argmax[valid_teacher],
            s=10,
            marker="x",
            color="lime",
            linewidths=0.75,
            label="Teacher argmax",
        )
        probabilities = np.clip(labels.soft_targets, 1.0e-12, 1.0)
        entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
        title = "Soft-label distribution | mean entropy={:.3f}".format(float(np.mean(entropy)))
    else:
        matrix = episode.execution_masks.T.astype(np.float32)
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            extent=(-0.5, transition_count - 0.5, -0.5, NUM_ACTIONS - 0.5),
            rasterized=True,
        )
        ax.scatter(np.arange(transition_count), episode.behavior_actions, s=12, color="cyan", label="Behavior")
        title = "Execution action masks"

    ax.axvline(selected_step, linewidth=0.9, linestyle=":", color="white")
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Action ID")
    ax.set_ylim(-0.5, NUM_ACTIONS - 0.5)
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.82)


def save_static_figure(
    output_path: Path,
    episode: EpisodeData,
    labels: Optional[EpisodeLabels],
    mpl: MplData,
    map_data: MapData,
    bounds: Tuple[float, float, float, float],
    comparisons: Sequence[ComparisonTrajectory],
    selected_step: int,
    dpi: int,
    fig_width: float,
    fig_height: float,
    map_grid_resolution: float,
    local_radius: float,
    candidate_top_k: int,
    elevation: float,
    azimuth: float,
    also_pdf: bool,
    show: bool,
) -> List[Path]:
    _set_publication_style()
    raster, extent = make_height_raster(map_data.points, bounds, map_grid_resolution)
    trajectory = trajectory_points(episode, mpl, append_nominal_endpoint=True)
    nominal_segments = nominal_behavior_segments(episode, mpl)

    fig = plt.figure(figsize=(float(fig_width), float(fig_height)), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.06, 0.94), height_ratios=(1.08, 0.92), wspace=0.18, hspace=0.28)
    ax_top = fig.add_subplot(grid[0, 0])
    ax_3d = fig.add_subplot(grid[0, 1], projection="3d")
    ax_metrics = fig.add_subplot(grid[1, 0])
    ax_heatmap = fig.add_subplot(grid[1, 1])

    _plot_top_view(ax_top, raster, extent, episode, trajectory, nominal_segments, comparisons, selected_step)
    _plot_candidate_scene(
        ax_3d,
        map_data.points,
        episode,
        labels,
        mpl,
        selected_step,
        local_radius,
        candidate_top_k,
        elevation,
        azimuth,
    )
    _plot_metrics(ax_metrics, episode, labels, selected_step)
    _plot_action_heatmap(ax_heatmap, episode, labels, selected_step)

    status_bits = []
    for key in ("success", "collision", "dead_end", "stop_reason"):
        if key in episode.index_row and str(episode.index_row[key]) != "":
            status_bits.append("{}={}".format(key, episode.index_row[key]))
    status = " | ".join(status_bits)
    title = "Episode {:06d} | {} transitions | selected step {}".format(
        episode.episode_id,
        episode.behavior_actions.shape[0],
        selected_step,
    )
    fig.suptitle(title, y=0.992, fontsize=11.5, fontweight="semibold")
    if status:
        fig.text(0.5, 0.958, status, ha="center", va="top", fontsize=8.2)
    fig.subplots_adjust(left=0.055, right=0.97, bottom=0.065, top=0.895)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(output_path),
        dpi=int(dpi),
        bbox_inches="tight",
        pad_inches=0.035,
        facecolor="white",
    )
    outputs = [output_path]

    if also_pdf:
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(str(pdf_path), bbox_inches="tight", pad_inches=0.035, facecolor="white")
        outputs.append(pdf_path)

    if show:
        plt.show()
    plt.close(fig)

    # Record expected raster dimensions and requested DPI.  Pillow is optional,
    # so verification does not become a runtime dependency.
    metadata = {
        "episode_id": int(episode.episode_id),
        "selected_step": int(selected_step),
        "requested_dpi": int(dpi),
        "figure_width_in": float(fig_width),
        "figure_height_in": float(fig_height),
        "nominal_pixel_width": int(round(float(fig_width) * int(dpi))),
        "nominal_pixel_height": int(round(float(fig_height) * int(dpi))),
        "map_source": (
            os.path.relpath(str(Path(map_data.source).expanduser()), str(output_path.parent))
            if Path(map_data.source).expanduser().is_absolute()
            else map_data.source
        ),
        "rollout_npz": os.path.relpath(str(episode.path), str(output_path.parent)),
        "labels_present": labels is not None,
    }
    try:
        from PIL import Image  # type: ignore

        with Image.open(str(output_path)) as image:
            metadata["actual_pixel_width"] = int(image.size[0])
            metadata["actual_pixel_height"] = int(image.size[1])
            if "dpi" in image.info:
                metadata["png_dpi_metadata"] = [float(value) for value in image.info["dpi"]]
    except Exception:
        pass
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    outputs.append(sidecar)
    return outputs


def _plotly_color(probability: float, maximum: float) -> str:
    try:
        from plotly.colors import sample_colorscale  # type: ignore

        ratio = 0.0 if maximum <= 0.0 else min(1.0, max(0.0, float(probability) / maximum))
        return str(sample_colorscale("Viridis", [ratio])[0])
    except Exception:
        return "rgb(40,120,180)"


def save_interactive_html(
    output_path: Path,
    episode: EpisodeData,
    labels: Optional[EpisodeLabels],
    mpl: MplData,
    map_data: MapData,
    selected_step: int,
    candidate_top_k: int,
    max_map_points: int,
    embed_plotly: bool,
) -> Path:
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as error:
        raise RuntimeError("interactive mode requires plotly: pip install plotly ({})".format(error))

    figure = go.Figure()
    map_points = _deterministic_downsample(map_data.points, max_map_points)
    if map_points.size:
        figure.add_trace(
            go.Scatter3d(
                x=map_points[:, 0],
                y=map_points[:, 1],
                z=map_points[:, 2],
                mode="markers",
                name="Forest map",
                marker={"size": 1.0, "opacity": 0.32, "color": map_points[:, 2], "colorscale": "Inferno"},
                hoverinfo="skip",
            )
        )

    trajectory = trajectory_points(episode, mpl, append_nominal_endpoint=True)
    figure.add_trace(
        go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode="lines+markers",
            name="Rollout",
            line={"width": 7, "color": "#00c843"},
            marker={"size": 2.5, "color": np.arange(trajectory.shape[0]), "colorscale": "Viridis"},
            text=["boundary {}".format(i) for i in range(trajectory.shape[0])],
            hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[episode.start[0], episode.goal[0]],
            y=[episode.start[1], episode.goal[1]],
            z=[episode.start[2], episode.goal[2]],
            mode="markers",
            name="Start / goal",
            marker={"size": 8, "color": ["#00d83b", "#ff2c24"], "symbol": ["circle", "diamond"]},
            text=["Start", "Goal"],
            hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
        )
    )

    pose = episode.poses_before[selected_step]
    order, mask, probabilities, teacher_action, behavior_action = _candidate_order(labels, episode, selected_step)
    maximum = float(np.max(probabilities)) if probabilities.size else 0.0
    top_actions = [int(a) for a in order if mask[int(a)] and probabilities[int(a)] > 0.0][: max(1, candidate_top_k)]
    special = [action for action in (behavior_action, teacher_action) if 0 <= action < mpl.num_actions]
    displayed = []
    for action in top_actions + special:
        if action not in displayed:
            displayed.append(action)

    for action in displayed:
        world = transform_primitive(mpl.pos_ref[action], pose)
        probability = float(probabilities[action])
        if action == teacher_action:
            color = "#20ee00"
            width = 9
            name = "Teacher argmax {}".format(action)
        elif action == behavior_action:
            color = "#ff9d00"
            width = 9
            name = "Behavior action {}".format(action)
        else:
            color = _plotly_color(probability, maximum)
            width = 3 + int(round(5.0 * probability / max(1.0e-9, maximum)))
            name = "Candidate {}".format(action)
        figure.add_trace(
            go.Scatter3d(
                x=world[:, 0],
                y=world[:, 1],
                z=world[:, 2],
                mode="lines",
                name=name,
                line={"width": width, "color": color},
                text=["action={} p={:.5f} valid={}".format(action, probability, bool(mask[action]))] * world.shape[0],
                hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
            )
        )

    figure.update_layout(
        title="Episode {:06d} — step {} candidate primitives".format(episode.episode_id, selected_step),
        template="plotly_white",
        scene={
            "xaxis_title": "x [m]",
            "yaxis_title": "y [m]",
            "zaxis_title": "z [m]",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.55, "y": -1.65, "z": 1.15}},
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        str(output_path),
        include_plotlyjs=True if embed_plotly else "cdn",
        full_html=True,
        auto_open=False,
    )
    return output_path


def save_animation(
    output_path: Path,
    episode: EpisodeData,
    labels: Optional[EpisodeLabels],
    mpl: MplData,
    map_data: MapData,
    bounds: Tuple[float, float, float, float],
    map_grid_resolution: float,
    top_k: int,
    fps: int,
    animation_dpi: int,
) -> Path:
    raster, extent = make_height_raster(map_data.points, bounds, map_grid_resolution)
    _set_publication_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    if raster.size and np.any(~raster.mask):
        ax.imshow(raster, origin="lower", extent=extent, cmap="inferno", interpolation="nearest", alpha=0.92, aspect="equal")
    ax.plot(
        [episode.start[0], episode.goal[0]],
        [episode.start[1], episode.goal[1]],
        linestyle="--",
        color="#e43d30",
        linewidth=1.1,
    )
    ax.scatter([episode.start[0]], [episode.start[1]], s=65, color="#18c83e", edgecolor="black", linewidth=0.6)
    ax.scatter([episode.goal[0]], [episode.goal[1]], s=75, marker="*", color="#ff2c24", edgecolor="black", linewidth=0.6)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    track_line, = ax.plot([], [], color="#00c843", linewidth=2.3)
    current_marker, = ax.plot([], [], marker="o", markersize=6, color="#00e5ff", markeredgecolor="black")
    candidate_lines = [ax.plot([], [], linewidth=1.0, alpha=0.75)[0] for _ in range(max(1, int(top_k)))]
    behavior_line, = ax.plot([], [], color="#ff9d00", linewidth=3.0)
    teacher_line, = ax.plot([], [], color="#1be800", linewidth=2.6)
    title = ax.set_title("")

    positions = episode.poses_before[:, :3]
    distance_xy = np.linalg.norm(positions[:, :2] - episode.goal[:2].reshape(1, 2), axis=1)

    def update(frame: int):
        pose = episode.poses_before[frame]
        track_line.set_data(positions[: frame + 1, 0], positions[: frame + 1, 1])
        current_marker.set_data([pose[0]], [pose[1]])
        order, mask, probabilities, teacher_action, behavior_action = _candidate_order(labels, episode, frame)
        maximum = max(1.0e-9, float(np.max(probabilities)))
        actions = [int(a) for a in order if mask[int(a)] and probabilities[int(a)] > 0.0][: len(candidate_lines)]
        cmap = plt.get_cmap("viridis")
        for index, line in enumerate(candidate_lines):
            if index < len(actions):
                action = actions[index]
                world = transform_primitive(mpl.pos_ref[action], pose)
                line.set_data(world[:, 0], world[:, 1])
                line.set_color(cmap(float(probabilities[action]) / maximum))
                line.set_linewidth(0.7 + 1.8 * float(probabilities[action]) / maximum)
                line.set_visible(True)
            else:
                line.set_data([], [])
                line.set_visible(False)

        if 0 <= behavior_action < mpl.num_actions:
            world = transform_primitive(mpl.pos_ref[behavior_action], pose)
            behavior_line.set_data(world[:, 0], world[:, 1])
        else:
            behavior_line.set_data([], [])
        if 0 <= teacher_action < mpl.num_actions:
            world = transform_primitive(mpl.pos_ref[teacher_action], pose)
            teacher_line.set_data(world[:, 0], world[:, 1])
        else:
            teacher_line.set_data([], [])

        valid_count = int(np.count_nonzero(mask))
        title.set_text(
            "Episode {:06d} | step {}/{} | d_goal={:.2f} m | valid={} | behavior={} | teacher={}".format(
                episode.episode_id,
                frame,
                positions.shape[0] - 1,
                float(distance_xy[frame]),
                valid_count,
                behavior_action,
                teacher_action if teacher_action >= 0 else "n/a",
            )
        )
        return [track_line, current_marker, behavior_line, teacher_line, title] + candidate_lines

    movie = animation.FuncAnimation(fig, update, frames=positions.shape[0], interval=1000.0 / max(1, fps), blit=False, repeat=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=max(1, int(fps)))
        movie.save(str(output_path), writer=writer, dpi=int(animation_dpi))
    else:
        if not animation.writers.is_available("ffmpeg"):
            fallback = output_path.with_suffix(".gif")
            writer = animation.PillowWriter(fps=max(1, int(fps)))
            movie.save(str(fallback), writer=writer, dpi=int(animation_dpi))
            plt.close(fig)
            return fallback
        writer = animation.FFMpegWriter(fps=max(1, int(fps)), bitrate=5000)
        movie.save(str(output_path), writer=writer, dpi=int(animation_dpi))
    plt.close(fig)
    return output_path


def _default_existing(package_root: Path, relative: str) -> str:
    path = package_root / relative
    return str(path) if path.exists() else ""


def build_parser(package_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality rollout visualizations from native planning data."
    )
    source = parser.add_argument_group("rollout source")
    source.add_argument("--episode-npz", default="", help="Direct path to one rollout episode NPZ")
    source.add_argument("--rollout-index", default="", help="rollout_index.csv containing dataset_npz")
    source.add_argument("--episode-id", type=int, default=None, help="Episode ID to select from rollout index")
    source.add_argument("--row-index", type=int, default=0, help="Index row used when --episode-id is omitted")
    source.add_argument("--labels", default="", help="Offline teacher_labels.npz generated from actual states")
    source.add_argument("--compare-npz", action="append", default=[], help="Additional rollout NPZ to overlay; repeatable")
    source.add_argument("--compare-name", action="append", default=[], help="Name for each --compare-npz")

    geometry = parser.add_argument_group("map and primitive geometry")
    geometry.add_argument(
        "--mpl-npz",
        default=_default_existing(package_root, "data/motion_primitives/motion_primitives_105.npz"),
        help="MPL NPZ containing pos_ref",
    )
    geometry.add_argument(
        "--point-cloud",
        default=_default_existing(package_root, "data/map_data/forest_point_cloud.bin"),
        help="Forest .bin/.ply/.pcd/.npy. Empty uses --voxel-cache.",
    )
    geometry.add_argument(
        "--voxel-cache",
        default=_default_existing(package_root, "data/map_data/forest_voxels_10cm.npz"),
        help="Voxel cache fallback when no point cloud is supplied",
    )
    geometry.add_argument("--point-cloud-frame", choices=("unity", "ros"), default="unity")
    geometry.add_argument("--no-map", action="store_true", help="Do not load a point cloud or voxel cache")
    geometry.add_argument("--crop-padding", type=float, default=6.0)
    geometry.add_argument("--z-min", type=float, default=-1.5)
    geometry.add_argument("--z-max", type=float, default=6.0)
    geometry.add_argument("--max-map-points", type=int, default=650000)
    geometry.add_argument("--map-grid-resolution", type=float, default=0.05)

    output = parser.add_argument_group("outputs")
    output.add_argument("--mode", choices=("static", "interactive", "animation", "all"), default="static")
    output.add_argument("--out-dir", default="data/smoke/visualizations")
    output.add_argument("--prefix", default="", help="Output filename prefix")
    output.add_argument("--step", type=int, default=-1, help="Candidate step; -1 selects maximum teacher entropy")
    output.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Static PNG DPI; default 600")
    output.add_argument("--fig-width", type=float, default=12.0, help="Static figure width in inches")
    output.add_argument("--fig-height", type=float, default=8.0, help="Static figure height in inches")
    output.add_argument("--also-pdf", action="store_true", help="Also save a publication PDF")
    output.add_argument("--show", action="store_true", help="Open the static Matplotlib window")

    candidates = parser.add_argument_group("candidate visualization")
    candidates.add_argument("--candidate-top-k", type=int, default=24)
    candidates.add_argument("--local-radius", type=float, default=7.0)
    candidates.add_argument("--view-elevation", type=float, default=57.0)
    candidates.add_argument("--view-azimuth", type=float, default=-72.0)
    candidates.add_argument("--interactive-map-points", type=int, default=90000)
    candidates.add_argument("--embed-plotly", action="store_true", help="Embed Plotly JS; larger but offline HTML")

    video = parser.add_argument_group("animation")
    video.add_argument("--animation-format", choices=("mp4", "gif"), default="mp4")
    video.add_argument("--animation-top-k", type=int, default=12)
    video.add_argument("--fps", type=int, default=4)
    video.add_argument("--animation-dpi", type=int, default=160)
    return parser


def main() -> int:
    package_root = _find_planning_root()
    parser = build_parser(package_root)
    args = parser.parse_args()

    episode_npz = _resolve_path(args.episode_npz, package_root) if args.episode_npz else None
    rollout_index = _resolve_path(args.rollout_index, package_root) if args.rollout_index else None
    labels_path = _resolve_path(args.labels, package_root) if args.labels else None
    mpl_path = _resolve_path(args.mpl_npz, package_root)
    if mpl_path is None:
        raise RuntimeError("--mpl-npz is required")

    episode = load_episode(episode_npz, rollout_index, args.episode_id, args.row_index)
    labels = load_episode_labels(labels_path, episode)
    mpl = load_mpl(mpl_path)
    if episode.behavior_actions.shape[0] == 0:
        raise RuntimeError("rollout contains no transitions")
    selected_step = _select_step(args.step, episode, labels)

    comparisons: List[ComparisonTrajectory] = []
    compare_names = list(args.compare_name)
    for index, raw in enumerate(args.compare_npz):
        path = _resolve_path(raw, package_root)
        if path is None:
            continue
        name = compare_names[index] if index < len(compare_names) else path.stem
        comparisons.append(_generic_comparison_trajectory(path, name, mpl))

    main_trajectory = trajectory_points(episode, mpl, append_nominal_endpoint=True)
    comparison_arrays = [value.points for value in comparisons]
    bounds = _roi_bounds([main_trajectory] + comparison_arrays, episode.start, episode.goal, args.crop_padding)

    if args.no_map:
        point_cloud_path = None
        voxel_cache_path = None
    else:
        point_cloud_path = _resolve_path(args.point_cloud, package_root, required=False) if args.point_cloud else None
        if point_cloud_path is not None and not point_cloud_path.exists():
            point_cloud_path = None
        voxel_cache_path = _resolve_path(args.voxel_cache, package_root, required=False) if args.voxel_cache else None
        if voxel_cache_path is not None and not voxel_cache_path.exists():
            voxel_cache_path = None

    map_data = load_map(
        point_cloud_path,
        voxel_cache_path,
        args.point_cloud_frame,
        bounds,
        args.z_min,
        args.z_max,
        args.max_map_points,
    )

    out_dir = _resolve_path(args.out_dir, package_root, required=False)
    if out_dir is None:
        raise RuntimeError("invalid --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix.strip() or "episode_{:06d}".format(episode.episode_id)

    print("ROLLOUT_VISUALIZATION_START")
    print("  planning_root:", package_root)
    print("  rollout:", episode.path)
    print("  episode_id:", episode.episode_id)
    print("  transitions:", episode.behavior_actions.shape[0])
    print("  selected_step:", selected_step)
    print("  labels:", labels_path if labels_path is not None else "none")
    print("  map_source:", map_data.source)
    print("  map_points_roi:", map_data.points.shape[0])
    print("  static_dpi:", int(args.dpi))

    outputs: List[Path] = []
    if args.mode in ("static", "all"):
        png = out_dir / "{}_diagnostic_{}dpi.png".format(prefix, int(args.dpi))
        outputs.extend(
            save_static_figure(
                png,
                episode,
                labels,
                mpl,
                map_data,
                bounds,
                comparisons,
                selected_step,
                args.dpi,
                args.fig_width,
                args.fig_height,
                args.map_grid_resolution,
                args.local_radius,
                args.candidate_top_k,
                args.view_elevation,
                args.view_azimuth,
                args.also_pdf,
                args.show,
            )
        )

    if args.mode in ("interactive", "all"):
        html = out_dir / "{}_interactive_step_{:03d}.html".format(prefix, selected_step)
        outputs.append(
            save_interactive_html(
                html,
                episode,
                labels,
                mpl,
                map_data,
                selected_step,
                args.candidate_top_k,
                args.interactive_map_points,
                args.embed_plotly,
            )
        )

    if args.mode in ("animation", "all"):
        suffix = ".{}".format(args.animation_format)
        movie = out_dir / "{}_planning_animation{}".format(prefix, suffix)
        outputs.append(
            save_animation(
                movie,
                episode,
                labels,
                mpl,
                map_data,
                bounds,
                args.map_grid_resolution,
                args.animation_top_k,
                args.fps,
                args.animation_dpi,
            )
        )

    print("ROLLOUT_VISUALIZATION_SUMMARY")
    for output in outputs:
        print("  output:", output)
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
