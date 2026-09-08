"""Resolved configuration for parallel teacher rollout collection."""

from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple

from planning.runtime.ports import MANAGED_TRAINING_PORT_DEFAULTS
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.common.config import parse_bool, pre_bc_value
from planning.common.paths import planning_package_root, planning_workspace_root
from planning.mission.spec import DEFAULT_MAX_PRIMITIVE_STEPS
from planning.contracts.task import task_contract_fields
from planning.safety.collision_checker import resolve_collision_backend
from planning.safety.depth_safety import resolve_depth_safety_backend
from planning.teacher.policy import DEFAULT_TEACHER_CONFIG


_PROTECTED_COLLECTOR_OPTIONS = frozenset(
    {
        "--index",
        "--out-dir",
        "--worker-id",
        "--num-workers",
        "--max-episodes",
        "--target-accepted",
        "--stop-file",
        "--max-steps",
        "--stream-horizon",
        "--max-stream-drift-m",
        "--max-endpoint-error-m",
        "--max-actual-path-length-m",
        "--worker-ready-timeout",
        "--max-sensor-skew-ms",
        "--collision-cache",
        "--voxel-size",
        "--inflate-radius",
        "--prefetch-inflate-radius",
        "--candidate-top-k",
        "--current-check-step",
        "--lookahead-check-step",
        "--expected-planar-distance",
        "--compress-episodes",
        "--reliable-exact-runtime-instance-id",
        "--reliable-exact-command-endpoint",
        "--reliable-exact-result-endpoint",
        "--reliable-exact-snapshot-endpoint",
        "--reliable-exact-timeout",
    }
)


def _env_int(environment: Mapping[str, str], name: str, default: int) -> int:
    return int(environment.get(name, str(default)))


def _env_float(environment: Mapping[str, str], name: str, default: float) -> float:
    return float(environment.get(name, str(default)))


def _parse_extra_args(raw: str) -> Tuple[str, ...]:
    tokens = tuple(shlex.split(str(raw))) if str(raw).strip() else ()
    seen = set()
    for token in tokens:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        if option in _PROTECTED_COLLECTOR_OPTIONS:
            raise ValueError(f"COLLECTOR_EXTRA_ARGS may not override managed option: {option}")
        if option in seen:
            raise ValueError(f"COLLECTOR_EXTRA_ARGS contains duplicate option: {option}")
        seen.add(option)
    return tokens


@dataclass(frozen=True)
class ParallelCollectionConfig:
    mission_index: Path
    out_dir: Path
    workers_requested: int
    max_episodes: int
    target_accepted: int
    window_width: int
    window_height: int
    max_steps: int
    stream_horizon: int
    candidate_top_k: int
    collision_threads: int
    current_check_step: int
    lookahead_check_step: int
    expected_planar_distance: float
    voxel_size: float
    inflate_radius: float
    prefetch_inflate_radius: float
    max_stream_drift_m: float
    max_endpoint_error_m: float
    max_actual_path_length_m: float
    max_sensor_skew_ms: float
    worker_ready_timeout_s: float
    startup_timeout_s: float
    master_base: int
    command_base: int
    depth_base: int
    port_stride: int
    compress_episodes: bool
    disable_async_prefetch: bool
    collector_extra_args: Tuple[str, ...]
    collection_run_id: str
    training_run_id: str
    runtime_launch_nonce: str
    reliable_timeout_s: float
    # Runtime and Teacher values are explicit resolved configuration.  The
    # defaults below are compatibility-only; formal CLI construction resolves
    # them from the canonical owners in ``from_environment``.
    unity_bin: Optional[Path] = None
    bridge_binary: Optional[Path] = None
    collision_cache: Optional[Path] = None
    collision_backend: str = "cpp_cpu"
    depth_safety_backend: str = "cpp_native"
    global_route_resolution_m: float = DEFAULT_TEACHER_CONFIG.global_route_resolution_m
    global_route_lookahead_m: float = DEFAULT_TEACHER_CONFIG.global_route_lookahead_m
    global_route_tracking_margin_m: float = DEFAULT_TEACHER_CONFIG.global_route_tracking_margin_m
    beam_depth: int = DEFAULT_TEACHER_CONFIG.beam_depth
    beam_width: int = DEFAULT_TEACHER_CONFIG.beam_width
    beam_branching: int = DEFAULT_TEACHER_CONFIG.beam_branching
    beam_discount: float = DEFAULT_TEACHER_CONFIG.beam_discount
    command_argv: Tuple[str, ...] = ()
    disk_warning_free_gb: float = 30.0
    disk_stop_free_gb: float = 15.0
    disk_check_interval_s: float = 15.0
    disk_safety_margin_gb: float = 10.0
    disk_estimated_episode_bytes: int = 0

    @property
    def active_workers(self) -> int:
        workers = int(self.workers_requested)
        if self.target_accepted > 0:
            workers = min(workers, self.target_accepted)
        if self.max_episodes > 0:
            workers = min(workers, self.max_episodes)
        return workers

    @property
    def observation_contract(self) -> str:
        return EXACT_ENDPOINT_OBSERVATION_CONTRACT

    @property
    def observation_source(self) -> str:
        return EXACT_ENDPOINT_OBSERVATION_CONTRACT

    @property
    def task_contract_fields(self) -> dict:
        return task_contract_fields(int(self.max_steps))

    def validate(self) -> None:
        if not self.mission_index.is_file():
            raise FileNotFoundError(f"mission index not found: {self.mission_index}")
        positive = {
            "workers_requested": self.workers_requested,
            "window_width": self.window_width,
            "window_height": self.window_height,
            "max_steps": self.max_steps,
            "stream_horizon": self.stream_horizon,
            "candidate_top_k": self.candidate_top_k,
            "collision_threads": self.collision_threads,
            "port_stride": self.port_stride,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name, value in {
            "max_episodes": self.max_episodes,
            "target_accepted": self.target_accepted,
        }.items():
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if not self.collection_run_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
            for character in self.collection_run_id
        ):
            raise ValueError(
                f"COLLECTION_RUN_ID contains unsupported characters: {self.collection_run_id}"
            )
        for name, value in (
            ("training_run_id", self.training_run_id),
            ("runtime_launch_nonce", self.runtime_launch_nonce),
        ):
            if not value or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
                for character in str(value)
            ):
                raise ValueError(
                    "{} contains unsupported characters: {}".format(name, value)
                )
        if float(self.reliable_timeout_s) <= 0.0:
            raise ValueError("reliable_timeout_s must be positive")
        if str(self.collision_backend) != "cpp_cpu":
            raise ValueError(
                "formal collection requires collision backend cpp_cpu, got {}".format(
                    self.collision_backend
                )
            )
        if str(self.depth_safety_backend) != "cpp_native":
            raise ValueError(
                "formal collection requires depth safety backend cpp_native, got {}".format(
                    self.depth_safety_backend
                )
            )
        for name, value in (
            ("beam_depth", self.beam_depth),
            ("beam_width", self.beam_width),
            ("beam_branching", self.beam_branching),
        ):
            if int(value) <= 0:
                raise ValueError("{} must be positive, got {}".format(name, value))
        if not 0.0 < float(self.beam_discount) <= 1.0:
            raise ValueError("beam_discount must be in (0, 1]")
        if float(self.disk_stop_free_gb) <= 0.0:
            raise ValueError("disk_stop_free_gb must be positive")
        if float(self.disk_warning_free_gb) < float(self.disk_stop_free_gb):
            raise ValueError("disk_warning_free_gb must be >= disk_stop_free_gb")
        if float(self.disk_check_interval_s) <= 0.0:
            raise ValueError("disk_check_interval_s must be positive")
        if float(self.disk_safety_margin_gb) < 0.0:
            raise ValueError("disk_safety_margin_gb must be non-negative")
        if int(self.disk_estimated_episode_bytes) < 0:
            raise ValueError("disk_estimated_episode_bytes must be non-negative")
        # Resolve the task identity from the same max_steps value that is
        # passed to the collector; callers persist the returned fields in the
        # resolved collection contract.
        self.task_contract_fields

    @classmethod
    def from_environment(
        cls,
        *,
        mission_index: Path,
        out_dir: Path,
        workers: int,
        environment: Mapping[str, str] | None = None,
        default_run_id: str,
        overrides: Mapping[str, Any] | None = None,
    ) -> "ParallelCollectionConfig":
        env = os.environ if environment is None else environment
        explicit = dict(overrides or {})
        deprecated_names = set()

        def pick(
            field: str,
            env_names: Tuple[str, ...],
            default: Any,
            cast: Callable[[Any], Any],
        ) -> Any:
            if field in explicit and explicit[field] is not None:
                return cast(explicit[field])
            for env_name in env_names:
                if env_name in env and str(env[env_name]).strip():
                    deprecated_names.add(env_name)
                    return cast(env[env_name])
            return cast(default)

        workspace = planning_workspace_root().resolve()
        package_root = planning_package_root().resolve()
        teacher = DEFAULT_TEACHER_CONFIG
        canonical_defaults = {
            "unity_bin": workspace / "src" / "unity" / "XMflight.x86_64",
            "bridge_binary": workspace / "devel" / "lib" / "planning" / "unity_bridge_node",
            "collision_cache": package_root / "data" / "map_data" / "forest_voxels_10cm.npz",
            "target_accepted": int(pre_bc_value("collection", "target_accepted")),
            "max_episodes": 0,
            "window_width": 320,
            "window_height": 240,
            "max_steps": DEFAULT_MAX_PRIMITIVE_STEPS,
            "stream_horizon": 1,
            "candidate_top_k": teacher.candidate_top_k,
            "collision_threads": 1,
            "current_check_step": teacher.current_check_step,
            "lookahead_check_step": teacher.lookahead_check_step,
            "expected_planar_distance": 40.0,
            "voxel_size": 0.10,
            "inflate_radius": 0.35,
            "prefetch_inflate_radius": 0.40,
            "max_stream_drift_m": 0.35,
            "max_endpoint_error_m": 0.60,
            "max_actual_path_length_m": 46.0,
            "max_sensor_skew_ms": 80.0,
            "worker_ready_timeout_s": 45.0,
            "startup_timeout_s": 60.0,
            "master_base": MANAGED_TRAINING_PORT_DEFAULTS["master_port_base"],
            "command_base": MANAGED_TRAINING_PORT_DEFAULTS["command_port_base"],
            "depth_base": MANAGED_TRAINING_PORT_DEFAULTS["depth_port_base"],
            "port_stride": MANAGED_TRAINING_PORT_DEFAULTS["port_stride"],
            "compress_episodes": False,
            "disable_async_prefetch": False,
            "reliable_timeout_s": 10.0,
            "global_route_resolution_m": teacher.global_route_resolution_m,
            "global_route_lookahead_m": teacher.global_route_lookahead_m,
            "global_route_tracking_margin_m": teacher.global_route_tracking_margin_m,
            "beam_depth": teacher.beam_depth,
            "beam_width": teacher.beam_width,
            "beam_branching": teacher.beam_branching,
            "beam_discount": teacher.beam_discount,
            "disk_warning_free_gb": 30.0,
            "disk_stop_free_gb": 15.0,
            "disk_check_interval_s": 15.0,
            "disk_safety_margin_gb": 10.0,
            "disk_estimated_episode_bytes": 0,
        }

        def path_value(field: str, env_names: Tuple[str, ...]) -> Path:
            return Path(pick(field, env_names, canonical_defaults[field], str)).expanduser().resolve()

        deprecated_extra_args = _parse_extra_args(env.get("COLLECTOR_EXTRA_ARGS", ""))
        if deprecated_extra_args:
            raise ValueError(
                "COLLECTOR_EXTRA_ARGS is no longer accepted; use typed Teacher flags"
            )
        config = cls(
            mission_index=Path(mission_index).expanduser().resolve(),
            out_dir=Path(out_dir).expanduser().resolve(),
            workers_requested=int(workers),
            max_episodes=pick("max_episodes", ("MAX_EPISODES",), canonical_defaults["max_episodes"], int),
            target_accepted=pick("target_accepted", ("TARGET_ACCEPTED",), canonical_defaults["target_accepted"], int),
            window_width=pick("window_width", ("WINDOW_WIDTH",), canonical_defaults["window_width"], int),
            window_height=pick("window_height", ("WINDOW_HEIGHT",), canonical_defaults["window_height"], int),
            max_steps=pick("max_steps", ("MAX_STEPS",), canonical_defaults["max_steps"], int),
            stream_horizon=pick("stream_horizon", ("STREAM_HORIZON",), canonical_defaults["stream_horizon"], int),
            candidate_top_k=pick("candidate_top_k", ("CANDIDATE_TOP_K",), canonical_defaults["candidate_top_k"], int),
            collision_threads=pick("collision_threads", ("COLLISION_THREADS",), canonical_defaults["collision_threads"], int),
            current_check_step=pick("current_check_step", ("CURRENT_CHECK_STEP",), canonical_defaults["current_check_step"], int),
            lookahead_check_step=pick("lookahead_check_step", ("LOOKAHEAD_CHECK_STEP",), canonical_defaults["lookahead_check_step"], int),
            expected_planar_distance=pick("expected_planar_distance", ("EXPECTED_PLANAR_DISTANCE",), canonical_defaults["expected_planar_distance"], float),
            voxel_size=pick("voxel_size", ("VOXEL_SIZE",), canonical_defaults["voxel_size"], float),
            inflate_radius=pick("inflate_radius", ("INFLATE_RADIUS",), canonical_defaults["inflate_radius"], float),
            prefetch_inflate_radius=pick("prefetch_inflate_radius", ("PREFETCH_INFLATE_RADIUS",), canonical_defaults["prefetch_inflate_radius"], float),
            max_stream_drift_m=pick("max_stream_drift_m", ("MAX_STREAM_DRIFT_M",), canonical_defaults["max_stream_drift_m"], float),
            max_endpoint_error_m=pick("max_endpoint_error_m", ("MAX_ENDPOINT_ERROR_M",), canonical_defaults["max_endpoint_error_m"], float),
            max_actual_path_length_m=pick("max_actual_path_length_m", ("MAX_ACTUAL_PATH_LENGTH_M",), canonical_defaults["max_actual_path_length_m"], float),
            max_sensor_skew_ms=pick("max_sensor_skew_ms", ("MAX_SENSOR_SKEW_MS",), canonical_defaults["max_sensor_skew_ms"], float),
            worker_ready_timeout_s=pick("worker_ready_timeout_s", ("WORKER_READY_TIMEOUT",), canonical_defaults["worker_ready_timeout_s"], float),
            startup_timeout_s=pick("startup_timeout_s", ("STARTUP_TIMEOUT",), canonical_defaults["startup_timeout_s"], float),
            master_base=pick("master_base", ("MASTER_BASE",), canonical_defaults["master_base"], int),
            command_base=pick("command_base", ("CMD_BASE",), canonical_defaults["command_base"], int),
            depth_base=pick("depth_base", ("DEPTH_BASE",), canonical_defaults["depth_base"], int),
            port_stride=pick("port_stride", ("PORT_STRIDE",), canonical_defaults["port_stride"], int),
            compress_episodes=pick("compress_episodes", ("COMPRESS_EPISODES",), canonical_defaults["compress_episodes"], parse_bool),
            disable_async_prefetch=pick("disable_async_prefetch", ("DISABLE_ASYNC_PREFETCH",), canonical_defaults["disable_async_prefetch"], parse_bool),
            # Retained as an empty compatibility field for callers that still
            # inspect the resolved dataclass; formal argv never expands it.
            collector_extra_args=(),
            collection_run_id=pick("collection_run_id", ("COLLECTION_RUN_ID",), default_run_id, str),
            training_run_id=pick("training_run_id", ("TRAINING_RUN_ID", "PLANNING_TRAINING_RUN_ID"), default_run_id, str),
            runtime_launch_nonce=pick("runtime_launch_nonce", ("RUNTIME_LAUNCH_NONCE", "PLANNING_RUNTIME_LAUNCH_NONCE"), default_run_id, str),
            reliable_timeout_s=pick("reliable_timeout_s", ("RELIABLE_EXACT_TIMEOUT",), canonical_defaults["reliable_timeout_s"], float),
            unity_bin=path_value("unity_bin", ("UNITY_BIN",)),
            bridge_binary=path_value("bridge_binary", ("PLANNING_BRIDGE_BINARY", "BRIDGE_BIN")),
            collision_cache=path_value("collision_cache", ("COLLISION_CACHE",)),
            collision_backend=resolve_collision_backend(
                pick("collision_backend", ("PLANNING_COLLISION_BACKEND",), "cpp_cpu", str)
            ),
            depth_safety_backend=resolve_depth_safety_backend(
                pick("depth_safety_backend", ("PLANNING_DEPTH_SAFETY_BACKEND",), "cpp_native", str)
            ),
            global_route_resolution_m=pick("global_route_resolution_m", ("GLOBAL_ROUTE_RESOLUTION_M",), canonical_defaults["global_route_resolution_m"], float),
            global_route_lookahead_m=pick("global_route_lookahead_m", ("GLOBAL_ROUTE_LOOKAHEAD_M",), canonical_defaults["global_route_lookahead_m"], float),
            global_route_tracking_margin_m=pick("global_route_tracking_margin_m", ("GLOBAL_ROUTE_TRACKING_MARGIN_M",), canonical_defaults["global_route_tracking_margin_m"], float),
            beam_depth=pick("beam_depth", ("BEAM_DEPTH",), canonical_defaults["beam_depth"], int),
            beam_width=pick("beam_width", ("BEAM_WIDTH",), canonical_defaults["beam_width"], int),
            beam_branching=pick("beam_branching", ("BEAM_BRANCHING",), canonical_defaults["beam_branching"], int),
            beam_discount=pick("beam_discount", ("BEAM_DISCOUNT",), canonical_defaults["beam_discount"], float),
            disk_warning_free_gb=pick("disk_warning_free_gb", ("DISK_WARNING_FREE_GB",), canonical_defaults["disk_warning_free_gb"], float),
            disk_stop_free_gb=pick("disk_stop_free_gb", ("DISK_STOP_FREE_GB",), canonical_defaults["disk_stop_free_gb"], float),
            disk_check_interval_s=pick("disk_check_interval_s", ("DISK_CHECK_INTERVAL_SEC",), canonical_defaults["disk_check_interval_s"], float),
            disk_safety_margin_gb=pick("disk_safety_margin_gb", ("DISK_SAFETY_MARGIN_GB",), canonical_defaults["disk_safety_margin_gb"], float),
            disk_estimated_episode_bytes=pick("disk_estimated_episode_bytes", ("DISK_ESTIMATED_EPISODE_BYTES",), canonical_defaults["disk_estimated_episode_bytes"], int),
            command_argv=tuple(str(value) for value in explicit.get("command_argv", ())),
        )
        if deprecated_names:
            print(
                "DEPRECATED_COLLECTION_ENV_VARS: use typed CLI flags or canonical config: {}".format(
                    ",".join(sorted(deprecated_names))
                ),
                file=sys.stderr,
            )
        collector_program = str(env.get("COLLECTOR_PROGRAM", "collect_teacher_rollouts.py"))
        if collector_program != "collect_teacher_rollouts.py":
            raise ValueError("COLLECTOR_PROGRAM override is forbidden for formal collection")
        if "COLLECTOR_PROGRAM" in env:
            print(
                "DEPRECATED_COLLECTION_ENV_VAR: COLLECTOR_PROGRAM is fixed to collect_teacher_rollouts.py",
                file=sys.stderr,
            )
        config.validate()
        return config

    def resolved_fields(self) -> dict:
        """Return safe, JSON-ready fields for the parent run manifest."""

        return {
            "mission_index": str(self.mission_index),
            "out_dir": str(self.out_dir),
            "workers_requested": int(self.workers_requested),
            "workers_active": int(self.active_workers),
            "max_episodes": int(self.max_episodes),
            "target_accepted": int(self.target_accepted),
            "window_width": int(self.window_width),
            "window_height": int(self.window_height),
            "max_steps": int(self.max_steps),
            "stream_horizon": int(self.stream_horizon),
            "candidate_top_k": int(self.candidate_top_k),
            "collision_threads": int(self.collision_threads),
            "current_check_step": int(self.current_check_step),
            "lookahead_check_step": int(self.lookahead_check_step),
            "expected_planar_distance": float(self.expected_planar_distance),
            "voxel_size": float(self.voxel_size),
            "inflate_radius": float(self.inflate_radius),
            "prefetch_inflate_radius": float(self.prefetch_inflate_radius),
            "max_stream_drift_m": float(self.max_stream_drift_m),
            "max_endpoint_error_m": float(self.max_endpoint_error_m),
            "max_actual_path_length_m": float(self.max_actual_path_length_m),
            "max_sensor_skew_ms": float(self.max_sensor_skew_ms),
            "worker_ready_timeout_s": float(self.worker_ready_timeout_s),
            "startup_timeout_s": float(self.startup_timeout_s),
            "master_base": int(self.master_base),
            "command_base": int(self.command_base),
            "depth_base": int(self.depth_base),
            "port_stride": int(self.port_stride),
            "compress_episodes": bool(self.compress_episodes),
            "disable_async_prefetch": bool(self.disable_async_prefetch),
            "collection_run_id": str(self.collection_run_id),
            "training_run_id": str(self.training_run_id),
            "runtime_launch_nonce": str(self.runtime_launch_nonce),
            "reliable_timeout_s": float(self.reliable_timeout_s),
            "unity_bin": str(self.unity_bin) if self.unity_bin is not None else "",
            "bridge_binary": str(self.bridge_binary) if self.bridge_binary is not None else "",
            "collision_cache": str(self.collision_cache) if self.collision_cache is not None else "",
            "collision_backend": str(self.collision_backend),
            "depth_safety_backend": str(self.depth_safety_backend),
            "observation_contract": self.observation_contract,
            "observation_source": self.observation_source,
            "global_route_resolution_m": float(self.global_route_resolution_m),
            "global_route_lookahead_m": float(self.global_route_lookahead_m),
            "global_route_tracking_margin_m": float(self.global_route_tracking_margin_m),
            "beam_depth": int(self.beam_depth),
            "beam_width": int(self.beam_width),
            "beam_branching": int(self.beam_branching),
            "beam_discount": float(self.beam_discount),
            "disk_warning_free_gb": float(self.disk_warning_free_gb),
            "disk_stop_free_gb": float(self.disk_stop_free_gb),
            "disk_check_interval_s": float(self.disk_check_interval_s),
            "disk_safety_margin_gb": float(self.disk_safety_margin_gb),
            "disk_estimated_episode_bytes": int(self.disk_estimated_episode_bytes),
        }
