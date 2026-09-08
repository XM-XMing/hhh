"""Unity ROS environment with synchronized state/depth observations and MPL control."""

from __future__ import annotations

import math
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool

from planning.msg import PrimitiveExecution, XMState
from planning.protocol.msgpack import messagepack_unpack
from planning.contracts.primitive_execution import (
    PRIMITIVE_EXECUTION_RESULT_COMPLETED,
    PRIMITIVE_EXECUTION_RESULT_TERMINAL_ABORT,
    PrimitiveExecutionContractError,
    classify_primitive_execution_result,
)
from planning.contracts.observation import exact_endpoint_metadata
from planning.mission.spec import (
    ACTION_MASK_Z_MARGIN_M,
    FLIGHT_Z_MAX_M,
    FLIGHT_Z_MIN_M,
    GOAL_RADIUS_XY_M,
    GOAL_TOLERANCE_Z_M,
    goal_reached,
)

try:
    from planning.primitives.library import MotionPrimitiveLibrary, rotation_z
except Exception:
    MotionPrimitiveLibrary = None

try:
    from planning.safety.collision_checker import VoxelCollisionChecker, VoxelMapConfig
except Exception:
    VoxelCollisionChecker = None
    VoxelMapConfig = None

from planning.safety.depth_safety import DepthSafetyConfig, camera_intrinsics_from_fov, local_depth_action_mask
from planning.contracts.policy_runtime import terminal_done_reason
from planning.contracts.reward import TERMINAL_FAILURE_PENALTY, compute_reward
from planning.runtime.reliable_unity_env_backend import (
    ReliableV4BackendError,
    ReliableV4EnvironmentBackend,
    ReliableV4ResetResult,
)

EXECUTION_TRANSPORT_AUDIT_SCHEMA_VERSION = 2
EXECUTION_TRANSPORT_AUDIT_RECORD_CAPACITY = 4096


class PrimitiveExecutionAbortedError(RuntimeError):
    """A Unity failed acknowledgement plus its classified execution result."""

    def __init__(
        self,
        *,
        execution_id: int,
        frame_count: int,
        frames: Sequence[Dict[str, Any]],
        execution_result: Optional[Dict[str, Any]] = None,
        terminal_state_message: Optional[XMState] = None,
    ):
        self.audit_receipt = {
            "execution_id": int(execution_id),
            "requested_frame_count": int(frame_count),
            "received_frames": [
                {
                    key: value
                    for key, value in frame.items()
                    if key != "state_message"
                }
                for frame in frames
            ],
        }
        self.execution_result = (
            dict(execution_result) if execution_result is not None else None
        )
        self.terminal_state_message = terminal_state_message
        self.previous_depth_seq: Optional[int] = None
        super().__init__(
            "Unity failed primitive execution before completion: execution_id={} "
            "received_frames={}/{}".format(
                execution_id, max(0, len(frames) - 1), frame_count
            )
        )

def apply_post_action_dead_end(
    reward: float,
    done: bool,
    info: Dict[str, Any],
    next_mask_info: Dict[str, Any],
    terminate_on_dead_end: bool,
    reward_dead_end: float,
) -> Tuple[float, bool, Dict[str, Any]]:
    """Apply dead-end termination only when no higher-priority terminal fired."""
    if (
        not bool(done)
        and bool(terminate_on_dead_end)
        and bool(next_mask_info.get("dead_end", False))
    ):
        reward = float(reward + reward_dead_end)
        done = True
        info["dead_end"] = True
        info["done_reason"] = "dead_end"
    return float(reward), bool(done), info

@dataclass
class EnvConfig:
    obs_timeout: float = 2.0
    # A freshly started Unity build may need several seconds before reset
    # commands and the first state frame are both available through ROS.
    reset_timeout: float = 10.0
    reset_tolerance_m: float = 0.10
    reset_settle_s: float = 0.30
    primitive_post_wait_s: float = 0.05
    stop_at_primitive_end: bool = True

    # Depth
    depth_min_m: float = 0.30
    depth_max_m: float = 3.00
    depth_sensor_max_m: float = 6.00
    depth_scale_m_per_unit: float = 0.001
    # Unity authoritative snapshots carry the camera capture resolution;
    # policy observations retain the existing depth_out_* resolution below.
    depth_capture_width: int = 848
    depth_capture_height: int = 480
    depth_out_width: int = 160
    depth_out_height: int = 90
    # Pair state and depth by Unity simulation/capture timestamps, independent
    # of bridge socket and ROS callback delivery order.
    enforce_sensor_sync: bool = True
    max_sensor_skew_s: float = 0.08

    # Optional deployment-safe local collision screen reconstructed from the
    # current depth image. It never accesses the global map or point cloud.
    use_depth_collision_mask: bool = False
    depth_camera_hfov_deg: float = DepthSafetyConfig().horizontal_fov_deg
    depth_camera_vfov_deg: float = DepthSafetyConfig().vertical_fov_deg
    depth_mask_min_forward_m: float = DepthSafetyConfig().min_forward_m
    depth_mask_path_sample_stride: int = DepthSafetyConfig().path_sample_stride
    depth_mask_collision_radius_m: float = DepthSafetyConfig().collision_radius_m
    depth_mask_slack_m: float = DepthSafetyConfig().depth_slack_m
    depth_mask_patch_radius_px: int = DepthSafetyConfig().patch_radius_px
    depth_mask_max_patch_radius_px: int = DepthSafetyConfig().max_patch_radius_px

    # Episode
    max_episode_steps: int = 300
    # Must match expert planning and strict execution acceptance.
    goal_radius_xy: float = GOAL_RADIUS_XY_M
    goal_tolerance_z: float = GOAL_TOLERANCE_Z_M
    far_distance_xy: float = 50.0
    z_min: float = FLIGHT_Z_MIN_M
    z_max: float = FLIGHT_Z_MAX_M
    z_hard_min: float = 0.8
    z_hard_max: float = 3.2
    action_mask_z_margin: float = ACTION_MASK_Z_MARGIN_M

    # Reward
    reward_progress_scale: float = 2.0
    reward_goal_z_progress_scale: float = 1.0
    reward_step: float = -0.02
    reward_clearance_scale: float = -1.0
    clearance_margin_m: float = 0.50
    reward_action_change: float = -0.02
    reward_success: float = 30.0
    reward_collision: float = TERMINAL_FAILURE_PENALTY
    reward_altitude_violation: float = TERMINAL_FAILURE_PENALTY
    reward_timeout: float = TERMINAL_FAILURE_PENALTY
    reward_far: float = TERMINAL_FAILURE_PENALTY

    # Optional privileged global collision mask. Keep disabled for pure depth-only
    # evaluation; enable for privileged-map safety diagnostics.
    use_global_collision_mask: bool = False
    collision_cache_npz: str = "data/map_data/forest_voxels_10cm.npz"
    collision_voxel_size: float = 0.10
    collision_inflate_radius: float = 0.35
    collision_check_step: int = 1
    rebuild_collision_cache: bool = False
    terminate_on_dead_end: bool = True
    reward_dead_end: float = TERMINAL_FAILURE_PENALTY
    terminate_on_invalid_action: bool = True
    reward_invalid_action: float = TERMINAL_FAILURE_PENALTY

def _quat_to_yaw_xyzw(q: Sequence[float]) -> float:
    x, y, z, w = [float(v) for v in q]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

class UnityForestEnv:
    def __init__(
        self,
        start: Sequence[float] = (0.0, 0.0, 2.0),
        goal: Sequence[float] = (40.0, 0.0, 2.0),
        config: Optional[EnvConfig] = None,
        mpl: Optional[Any] = None,
        reliable_v4_backend: Optional[ReliableV4EnvironmentBackend] = None,
        node_name: str = "planning_unity_env",
        anonymous: bool = True,
        legacy_command_path_enabled: Optional[bool] = None,
    ):
        self.config = config or EnvConfig()
        task_values = (
            ("z_min", self.config.z_min, FLIGHT_Z_MIN_M),
            ("z_max", self.config.z_max, FLIGHT_Z_MAX_M),
            ("goal_radius_xy", self.config.goal_radius_xy, GOAL_RADIUS_XY_M),
            ("goal_tolerance_z", self.config.goal_tolerance_z, GOAL_TOLERANCE_Z_M),
            ("action_mask_z_margin", self.config.action_mask_z_margin, ACTION_MASK_Z_MARGIN_M),
        )
        mismatches = [
            "{}={} expected {}".format(name, actual, expected)
            for name, actual, expected in task_values
            if abs(float(actual) - float(expected)) > 1.0e-9
        ]
        if mismatches:
            raise ValueError(
                "EnvConfig overrides hashed task contract: " + "; ".join(mismatches)
            )
        self.start = np.asarray(start, dtype=np.float32)
        self.goal = np.asarray(goal, dtype=np.float32)
        self._reliable_v4_backend = reliable_v4_backend
        self._legacy_command_path_enabled = (
            reliable_v4_backend is None
            if legacy_command_path_enabled is None
            else bool(legacy_command_path_enabled)
        )
        if self._reliable_v4_backend is not None and self._legacy_command_path_enabled:
            raise ValueError(
                "reliable_v4 and legacy command paths cannot both be enabled"
            )
        if self._reliable_v4_backend is None and not self._legacy_command_path_enabled:
            raise ValueError(
                "at least one environment command path must be enabled"
            )
        self.telemetry_lookup_count = 0
        self.episode_id = "episode-0"
        self.reset_id = "reset-0"
        self._reset_generation = 0

        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=anonymous)
        self._ros_use_sim_time = bool(rospy.get_param("/use_sim_time", False))

        self._lock = threading.RLock()
        self._execution_condition = threading.Condition(self._lock)
        self._execution_acknowledgements: Dict[int, list[Dict[str, Any]]] = {}
        self._latest_state: Optional[XMState] = None
        self._latest_state_receive_monotonic_ns = 0
        self._latest_depth: Optional[Image] = None
        self._latest_camera_info: Optional[CameraInfo] = None
        self._state_seq = 0
        self._depth_seq = 0
        self._command_frame_audit_active = False
        self._command_frame_audit_states = []
        # This is a bounded, audit-only in-memory ledger.  It is enabled only
        # by the diagnostic evaluator and is written after the episode; state
        # callbacks never perform filesystem I/O or alter acknowledgement flow.
        self._execution_transport_audit_active = False
        self._execution_transport_audits: Dict[int, Dict[str, Any]] = {}
        self._execution_transport_receive_sequence = 0
        self._execution_transport_callback_records: list[Optional[tuple[Any, ...]]] = []
        self._execution_transport_callback_count = 0
        self._execution_transport_callback_overflow = False
        self._execution_transport_audit_run_id = ""
        self._execution_transport_audit_unity_runtime_identity = ""
        self._pairing_candidate_audit_enabled = False
        self._pairing_candidate_state_history = deque(maxlen=64)
        self._pairing_candidate_depth_history = deque(maxlen=64)
        self._pairing_candidate_boundaries: list[Dict[str, Any]] = []
        self._trajectory_tracking_enabled = False
        self._trajectory_last_position: Optional[np.ndarray] = None
        self._trajectory_length_m = 0.0

        # A reliable endpoint owns an immutable state/depth pair.  Creating
        # PUB/SUB subscribers here would make the formal collector capable of
        # silently consulting latest telemetry, so keep the legacy sensors
        # entirely out of that mode.  Legacy diagnostic callers retain the
        # existing subscriptions unchanged.
        if self._reliable_v4_backend is None:
            self._state_sub = rospy.Subscriber("/xm/state", XMState, self._on_state, queue_size=20)
            self._depth_sub = rospy.Subscriber("/xm/depth/image_raw", Image, self._on_depth, queue_size=2)
            self._camera_info_sub = rospy.Subscriber("/xm/depth/camera_info", CameraInfo, self._on_camera_info, queue_size=1)
        else:
            self._state_sub = None
            self._depth_sub = None
            self._camera_info_sub = None

        if self._legacy_command_path_enabled:
            self._cmd_pub = rospy.Publisher("/xm/cmd_vel", Twist, queue_size=10)
            self._primitive_execution_pub = rospy.Publisher(
                "/xm/primitive_execution", PrimitiveExecution, queue_size=1
            )
            self._reset_pub = rospy.Publisher("/xm/reset_pose", PoseStamped, queue_size=1)
            self._stop_pub = rospy.Publisher("/xm/stop", Bool, queue_size=1)
        else:
            self._cmd_pub = None
            self._primitive_execution_pub = None
            self._reset_pub = None
            self._stop_pub = None

        self.mpl = mpl
        if self.mpl is None and MotionPrimitiveLibrary is not None:
            try:
                self.mpl = MotionPrimitiveLibrary()
            except Exception:
                self.mpl = None

        self.action_space_n = int(self.mpl.num_actions) if self.mpl is not None else 0

        self._collision_checker = None
        self._last_action_mask_info: Dict[str, Any] = {}
        if bool(self.config.use_global_collision_mask):
            if self.mpl is None:
                raise RuntimeError("global collision mask requires MPL library")
            if VoxelCollisionChecker is None or VoxelMapConfig is None:
                raise RuntimeError("global collision mask requires collision.py")
            self._collision_checker = VoxelCollisionChecker.from_config(
                VoxelMapConfig(
                    voxel_size=float(self.config.collision_voxel_size),
                    inflate_radius=float(self.config.collision_inflate_radius),
                    cache_npz=str(self.config.collision_cache_npz),
                ),
                rebuild_cache=bool(self.config.rebuild_collision_cache),
            )

        self.episode_step = 0
        # One environment transition per reliable execution identity.  The
        # transport/backend owns delivery deduplication; this cache prevents a
        # duplicate terminal result from re-running reward/step accounting.
        self._reliable_transition_cache: Dict[int, Tuple[Any, float, bool, Dict[str, Any]]] = {}
        self.prev_distance_xy: Optional[float] = None
        self.prev_abs_goal_dz: Optional[float] = None
        self.prev_action_id = -1
        # Let publishers connect.
        rospy.sleep(0.2)

    # ---------------------------------------------------------------------
    # ROS callbacks
    # ---------------------------------------------------------------------

    @staticmethod
    def _command_audit_state_record(
        msg: XMState,
        *,
        receive_monotonic_ns: int,
        state_sequence: int,
        source: str = "callback",
    ) -> Dict[str, Any]:
        orientation = [
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
        ]
        return {
            "source": str(source),
            "receive_monotonic_ns": int(receive_monotonic_ns),
            "state_sequence": int(state_sequence),
            "state_id": int(msg.state_id),
            "sim_time_ns": int(msg.sim_time_ns),
            "position": [
                float(msg.position.x), float(msg.position.y), float(msg.position.z)
            ],
            "velocity": [
                float(msg.velocity.x), float(msg.velocity.y), float(msg.velocity.z)
            ],
            "yaw": float(_quat_to_yaw_xyzw(orientation)),
            "applied_execution_id": int(getattr(msg, "applied_execution_id", -1)),
            "applied_execution_frame_index": int(
                getattr(msg, "applied_execution_frame_index", -1)
            ),
            "applied_command_id": int(getattr(msg, "applied_command_id", -1)),
            "execution_status": int(getattr(msg, "execution_status", 0)),
        }

    def _on_state(self, msg: XMState) -> None:
        receive_monotonic_ns = time.monotonic_ns()
        with self._lock:
            position = np.asarray(
                [msg.position.x, msg.position.y, msg.position.z], dtype=np.float64
            )
            if self._trajectory_tracking_enabled and np.isfinite(position).all():
                if self._trajectory_last_position is not None:
                    self._trajectory_length_m += float(
                        np.linalg.norm(position - self._trajectory_last_position)
                    )
                self._trajectory_last_position = position
            self._latest_state = msg
            self._latest_state_receive_monotonic_ns = int(receive_monotonic_ns)
            self._state_seq += 1
            if self._pairing_candidate_audit_enabled:
                self._pairing_candidate_state_history.append({
                    "state_sequence": int(self._state_seq),
                    "receive_monotonic_ns": int(receive_monotonic_ns),
                    "message": msg,
                })
            applied_execution_id = int(getattr(msg, "applied_execution_id", -1))
            if applied_execution_id >= 0:
                acknowledgement = {
                    "execution_id": applied_execution_id,
                    "frame_index": int(msg.applied_execution_frame_index),
                    "command_id": int(msg.applied_command_id),
                    "applied_state_id": int(msg.state_id),
                    "sim_time_ns": int(msg.sim_time_ns),
                    "execution_status": int(msg.execution_status),
                    "state_sequence": int(self._state_seq),
                    "position": [
                        float(msg.position.x), float(msg.position.y), float(msg.position.z)
                    ],
                    "velocity": [
                        float(msg.velocity.x), float(msg.velocity.y), float(msg.velocity.z)
                    ],
                    "acceleration": [
                        float(msg.acceleration.x), float(msg.acceleration.y),
                        float(msg.acceleration.z),
                    ],
                    "orientation_xyzw": [
                        float(msg.orientation.x), float(msg.orientation.y),
                        float(msg.orientation.z), float(msg.orientation.w),
                    ],
                    "collided": bool(msg.collided),
                    "altitude_violation": bool(msg.altitude_violation),
                    "min_clearance": float(msg.min_clearance),
                    "front_clearances": [
                        float(value) for value in msg.front_clearances
                    ],
                    "state_message": msg,
                }
                self._execution_acknowledgements.setdefault(
                    applied_execution_id, []
                ).append(acknowledgement)
                if self._execution_transport_audit_active:
                    self._execution_transport_receive_sequence += 1
                    callback_index = self._execution_transport_callback_count
                    self._execution_transport_callback_count += 1
                    callback_record = (
                        int(applied_execution_id),
                        int(msg.applied_execution_frame_index),
                        int(msg.applied_command_id),
                        int(msg.state_id),
                        int(msg.sim_time_ns),
                        int(msg.execution_status),
                        int(self._execution_transport_receive_sequence),
                        int(receive_monotonic_ns),
                    )
                    if callback_index < len(self._execution_transport_callback_records):
                        self._execution_transport_callback_records[callback_index] = (
                            callback_record
                        )
                    else:
                        self._execution_transport_callback_overflow = True
                self._execution_condition.notify_all()
            if self._command_frame_audit_active:
                self._command_frame_audit_states.append(
                    self._command_audit_state_record(
                        msg,
                        receive_monotonic_ns=receive_monotonic_ns,
                        state_sequence=self._state_seq,
                    )
                )

    def _on_depth(self, msg: Image) -> None:
        with self._lock:
            self._latest_depth = msg
            self._depth_seq += 1
            if self._pairing_candidate_audit_enabled:
                self._pairing_candidate_depth_history.append({
                    "depth_sequence": int(self._depth_seq),
                    "receive_monotonic_ns": int(time.monotonic_ns()),
                    "message": msg,
                })

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._latest_camera_info = msg

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def observation_token(self) -> Tuple[int, int]:
        """Return callback sequence counters for state and depth."""
        with self._lock:
            return int(self._state_seq), int(self._depth_seq)

    def begin_execution_transport_audit(
        self,
        *,
        run_id: str,
        unity_runtime_identity: str,
    ) -> None:
        """Capture acknowledgement receipts in memory for one diagnostic episode.

        The audit merely observes callback and receipt-collector events.  It
        neither waits for additional states nor changes execution validation.
        """
        with self._lock:
            if not str(run_id).strip() or not str(unity_runtime_identity).strip():
                raise ValueError("execution transport audit identities must be non-empty")
            self._execution_transport_audits = {}
            self._execution_transport_receive_sequence = 0
            self._execution_transport_callback_records = [None] * (
                EXECUTION_TRANSPORT_AUDIT_RECORD_CAPACITY
            )
            self._execution_transport_callback_count = 0
            self._execution_transport_callback_overflow = False
            self._execution_transport_audit_run_id = str(run_id)
            self._execution_transport_audit_unity_runtime_identity = str(
                unity_runtime_identity
            )
            self._execution_transport_audit_active = True

    def end_execution_transport_audit(self) -> Dict[str, Any]:
        """Return the completed audit snapshot and disable further capture."""
        with self._lock:
            self._execution_transport_audit_active = False
            callback_records: Dict[int, list[Dict[str, Any]]] = {}
            for callback_record in self._execution_transport_callback_records[
                :min(
                    self._execution_transport_callback_count,
                    EXECUTION_TRANSPORT_AUDIT_RECORD_CAPACITY,
                )
            ]:
                if callback_record is None:
                    raise RuntimeError("execution transport audit callback record is missing")
                (
                    execution_id,
                    frame_index,
                    command_id,
                    state_id,
                    sim_time_ns,
                    execution_status,
                    receive_sequence,
                    receive_monotonic_ns,
                ) = callback_record
                callback_records.setdefault(int(execution_id), []).append({
                    "execution_id": int(execution_id),
                    "frame_index": int(frame_index),
                    "command_id": int(command_id),
                    "state_id": int(state_id),
                    "sim_time_ns": int(sim_time_ns),
                    "execution_status": int(execution_status),
                    "received": True,
                    "receive_sequence": int(receive_sequence),
                    "receive_monotonic_ns": int(receive_monotonic_ns),
                })
            executions = []
            for execution_id in sorted(self._execution_transport_audits):
                audit = self._execution_transport_audits[execution_id]
                executions.append({
                    "execution_id": int(execution_id),
                    "requested_frame_count": int(audit.get("requested_frame_count", 0)),
                    "command_ids": [int(value) for value in audit.get("command_ids", [])],
                    "planning_received": list(callback_records.get(execution_id, [])),
                    "collector_received": list(audit.get("collector_received", [])),
                    "execution_accounted": list(audit.get("execution_accounted", [])),
                    "collector_result": dict(audit.get("collector_result", {})),
                    "collector_terminal": str(audit.get("collector_terminal", "")),
                })
            return {
                "contract_id": "[DEBUG-EXEC-TRANSPORT-81660]planning_receipt_audit",
                "audit_schema_version": EXECUTION_TRANSPORT_AUDIT_SCHEMA_VERSION,
                "audit_run_id": str(self._execution_transport_audit_run_id),
                "unity_runtime_identity": str(
                    self._execution_transport_audit_unity_runtime_identity
                ),
                "capture_overflow": bool(self._execution_transport_callback_overflow),
                "executions": executions,
            }

    def _begin_execution_transport_record(
        self,
        *,
        execution_id: int,
        frame_count: int,
        command_ids: Sequence[int],
    ) -> None:
        with self._lock:
            if not self._execution_transport_audit_active:
                return
            record = self._execution_transport_audits.setdefault(int(execution_id), {})
            record.update({
                "requested_frame_count": int(frame_count),
                "command_ids": [int(value) for value in command_ids],
            })
            record.setdefault("collector_received", [])
            record.setdefault("execution_accounted", [])

    def _finish_execution_transport_record(
        self,
        *,
        execution_id: int,
        applied_frames: Sequence[Dict[str, Any]],
        result: Dict[str, Any],
        collector_terminal: str,
    ) -> None:
        with self._lock:
            if not self._execution_transport_audit_active:
                return
            record = self._execution_transport_audits.setdefault(int(execution_id), {})
            record.setdefault("collector_received", [])
            collector_records = [
                {
                    "execution_id": int(frame["execution_id"]),
                    "frame_index": int(frame["frame_index"]),
                    "command_id": int(frame["command_id"]),
                    "state_id": int(frame["applied_state_id"]),
                    "sim_time_ns": int(frame["sim_time_ns"]),
                    "received": True,
                    "collected": True,
                    "execution_status": int(frame["execution_status"]),
                }
                for frame in applied_frames
            ]
            record["collector_received"] = collector_records
            if result.get("kind") in (
                PRIMITIVE_EXECUTION_RESULT_COMPLETED,
                PRIMITIVE_EXECUTION_RESULT_TERMINAL_ABORT,
            ):
                record["execution_accounted"] = [
                    {
                        **frame,
                        "accounted": True,
                    }
                    for frame in collector_records
                ]
            else:
                record["execution_accounted"] = []
            record["collector_result"] = dict(result)
            record["collector_terminal"] = str(collector_terminal)

    def get_observation(self) -> Dict[str, Any]:
        """Return the latest complete observation."""
        return self._get_observation_blocking()

    def actual_trajectory_length_m(self) -> float:
        """Return the 3D polyline length accumulated from Unity state frames."""
        with self._lock:
            return float(self._trajectory_length_m)

    def begin_pairing_candidate_audit(self, history_limit: int = 64) -> None:
        """Enable a bounded read-only sensor history for one diagnostic episode."""
        limit = int(history_limit)
        if limit <= 0:
            raise ValueError("pairing audit history_limit must be positive")
        with self._lock:
            self._pairing_candidate_state_history = deque(maxlen=limit)
            self._pairing_candidate_depth_history = deque(maxlen=limit)
            self._pairing_candidate_boundaries = []
            self._pairing_candidate_audit_enabled = True

    def finish_pairing_candidate_audit(self) -> Dict[str, Any]:
        """Disable capture and return in-memory candidates for offline replay."""
        with self._lock:
            self._pairing_candidate_audit_enabled = False
            boundaries = list(self._pairing_candidate_boundaries)
            unique_states = {
                int(record["state_sequence"]): record
                for boundary in boundaries
                for record in boundary["states"]
            }
            unique_depths = {
                int(record["depth_sequence"]): record
                for boundary in boundaries
                for record in boundary["depths"]
            }
            return {
                "boundaries": boundaries,
                "states": [unique_states[key] for key in sorted(unique_states)],
                "depths": [unique_depths[key] for key in sorted(unique_depths)],
            }

    def build_pairing_candidate_observation(
        self,
        state_record: Dict[str, Any],
        depth_record: Dict[str, Any],
        camera_info: Optional[CameraInfo],
    ) -> Dict[str, Any]:
        """Replay one captured pair through the production observation builder."""
        state = state_record["message"]
        depth = depth_record["message"]
        return self._build_observation(
            state,
            depth,
            state_seq=int(state_record["state_sequence"]),
            depth_seq=int(depth_record["depth_sequence"]),
            state_stamp_ns=int(state.sim_time_ns),
            depth_stamp_ns=self._stamp_ns(depth),
            camera_info=camera_info,
        )

    def _record_pairing_candidate_boundary(
        self,
        *,
        boundary_kind: str,
        previous_token: Tuple[int, int],
        selected_observation: Dict[str, Any],
        endpoint_state: Optional[XMState] = None,
    ) -> None:
        with self._lock:
            if not self._pairing_candidate_audit_enabled:
                return
            state_records = list(self._pairing_candidate_state_history)
            depth_records = list(self._pairing_candidate_depth_history)
            selected_seq = selected_observation.get("sensor_seq", {})
            required_state_sequence = None
            required_state_timestamp_ns = None
            if endpoint_state is not None:
                required_state_timestamp_ns = int(endpoint_state.sim_time_ns)
                for record in reversed(state_records):
                    message = record["message"]
                    if (
                        int(message.state_id) == int(endpoint_state.state_id)
                        and int(message.sim_time_ns) == required_state_timestamp_ns
                    ):
                        required_state_sequence = int(record["state_sequence"])
                        break
                if required_state_sequence is None:
                    raise RuntimeError(
                        "pairing audit lost primitive endpoint state_id={}".format(
                            int(endpoint_state.state_id)
                        )
                    )
            self._pairing_candidate_boundaries.append({
                "boundary_index": len(self._pairing_candidate_boundaries),
                "boundary_kind": str(boundary_kind),
                "previous_state_sequence": int(previous_token[0]),
                "previous_depth_sequence": int(previous_token[1]),
                "selected_state_sequence": int(selected_seq.get("state", -1)),
                "selected_depth_sequence": int(selected_seq.get("depth", -1)),
                "required_state_sequence": required_state_sequence,
                "required_state_timestamp_ns": required_state_timestamp_ns,
                "states": state_records,
                "depths": depth_records,
                "camera_info": self._latest_camera_info,
            })

    def get_observation_after(
        self,
        token: Tuple[int, int],
        require_new_state: bool = True,
        require_new_depth: bool = True,
    ) -> Dict[str, Any]:
        """Wait until state/depth callbacks are newer than ``token``."""
        return self._get_observation_after_token(
            token, require_new_state=require_new_state, require_new_depth=require_new_depth
        )

    def wait_until_ready(self, timeout_s: float = 30.0) -> None:
        """Wait for sensor data and a reset command subscriber before an episode."""
        if self._reliable_v4_backend is not None:
            # Readiness is proven by the reliable reset/execute transaction.
            # Do not turn a formal endpoint run into a ROS latest-topic probe.
            return
        deadline = time.time() + max(0.0, float(timeout_s))
        while not rospy.is_shutdown() and time.time() < deadline:
            with self._lock:
                has_state = self._latest_state is not None
                has_depth = self._latest_depth is not None
            reset_subscribers = (
                int(self._reset_pub.get_num_connections())
                if self._reset_pub is not None else 0
            )
            execution_subscribers = (
                int(self._primitive_execution_pub.get_num_connections())
                if self._primitive_execution_pub is not None else 0
            )
            execution_ready = (
                self._reliable_v4_backend is not None
                or execution_subscribers > 0
            )
            reset_ready = (
                self._reliable_v4_backend is not None
                or reset_subscribers > 0
            )
            if has_state and has_depth and reset_ready and execution_ready:
                return
            rospy.sleep(0.05)

        with self._lock:
            has_state = self._latest_state is not None
            has_depth = self._latest_depth is not None
        raise RuntimeError(
            "Unity ROS worker is not ready within {:.1f}s: state={} depth={} "
            "reset_subscribers={} execution_subscribers={}. Check ROS_MASTER_URI, roslaunch, "
            "and require schema-v3 unity_bridge_node or reliable v4 backend.".format(
                float(timeout_s), has_state, has_depth,
                int(self._reset_pub.get_num_connections()) if self._reset_pub is not None else 0,
                int(self._primitive_execution_pub.get_num_connections())
                if self._primitive_execution_pub is not None else 0,
            )
        )

    def stop_and_wait_until_stable(
        self,
        timeout_s: float = 2.0,
        speed_tolerance_mps: float = 0.03,
        position_tolerance_m: float = 0.01,
        yaw_tolerance_rad: float = 0.01,
        stable_frames: int = 3,
        poll_s: float = 0.05,
    ) -> Dict[str, Any]:
        """Stop Unity and return a fresh observation after motion has settled."""
        self.stop()
        deadline = time.time() + max(0.0, float(timeout_s))
        previous = None
        consecutive = 0
        last_obs = None
        required = max(1, int(stable_frames))
        token = self.observation_token()

        while not rospy.is_shutdown() and time.time() < deadline:
            obs = self._get_observation_after_token(token, require_new_state=True, require_new_depth=True)
            token = self.observation_token()
            last_obs = obs
            position = np.asarray(obs["state"]["position"], dtype=np.float32)
            velocity = np.asarray(obs["state"]["velocity"], dtype=np.float32)
            yaw = float(obs["state"]["yaw"])
            speed_ok = float(np.linalg.norm(velocity)) <= float(speed_tolerance_mps)
            if previous is None:
                delta_ok = False
            else:
                delta_position = float(np.linalg.norm(position - previous[0]))
                delta_yaw = abs(math.atan2(math.sin(yaw - previous[1]), math.cos(yaw - previous[1])))
                delta_ok = delta_position <= float(position_tolerance_m) and delta_yaw <= float(yaw_tolerance_rad)
            consecutive = consecutive + 1 if speed_ok and delta_ok else 0
            if consecutive >= required:
                return obs
            previous = (position, yaw)
            rospy.sleep(max(0.001, float(poll_s)))

        speed = float(np.linalg.norm(last_obs["state"]["velocity"])) if last_obs is not None else float("nan")
        raise TimeoutError(
            "Unity state did not settle within {:.2f}s: speed={:.4f}m/s stable_frames={}/{}".format(
                float(timeout_s), speed, consecutive, required
            )
        )

    def reset(
        self,
        start: Optional[Sequence[float]] = None,
        goal: Optional[Sequence[float]] = None,
        settle: Optional[float] = None,
    ) -> Dict[str, Any]:
        if start is not None:
            self.start = np.asarray(start, dtype=np.float32)
        if goal is not None:
            self.goal = np.asarray(goal, dtype=np.float32)

        if self.start.shape[0] < 3 or self.goal.shape[0] < 3:
            raise ValueError("start and goal must contain xyz")
        if not self.config.z_min <= float(self.start[2]) <= self.config.z_max:
            raise ValueError("reset start altitude outside [{},{}]".format(self.config.z_min, self.config.z_max))
        if not self.config.z_min <= float(self.goal[2]) <= self.config.z_max:
            raise ValueError("reset goal altitude outside [{},{}]".format(self.config.z_min, self.config.z_max))

        reliable_reset = getattr(self._reliable_v4_backend, "reset", None)
        if callable(reliable_reset):
            next_episode_id, next_reset_id = self._next_reset_identity()
            completion = reliable_reset(
                start=tuple(float(value) for value in self.start),
                goal=tuple(float(value) for value in self.goal),
                episode_id=next_episode_id,
                reset_id=next_reset_id,
            )
            if not isinstance(completion, ReliableV4ResetResult):
                raise TypeError("reliable reset backend returned an invalid completion")
            if (
                completion.episode_id != next_episode_id
                or completion.reset_id != next_reset_id
                or completion.observation is None
            ):
                raise RuntimeError("reliable reset completion identity mismatch")
            observation = completion.observation
            if isinstance(observation, dict):
                observation = dict(observation)
                observation.setdefault("episode_id", next_episode_id)
                observation.setdefault("reset_id", next_reset_id)
            else:
                # The real bridge snapshot provider returns the immutable
                # EndpointObservation transport object.  Materialize it
                # through the same exact state/depth decoder used by the
                # reliable primitive endpoint; do not consult telemetry.
                observation = self._build_reliable_v4_observation(observation)
            self._validate_reliable_reset_observation(
                observation, next_episode_id, next_reset_id
            )
            self.episode_id = next_episode_id
            self.reset_id = next_reset_id
            self.episode_step = 0
            self._reliable_transition_cache = {}
            self.prev_distance_xy = float(observation["goal"]["distance_xy"])
            self.prev_abs_goal_dz = abs(float(observation["goal"]["dz"]))
            self.prev_action_id = -1
            self._trajectory_last_position = np.asarray(
                observation["state"]["position"], dtype=np.float64
            ).reshape(3)
            self._trajectory_length_m = 0.0
            self._trajectory_tracking_enabled = True
            return observation

        self.stop()
        with self._lock:
            # Exclude the reset teleport and settling motion from the episode.
            self._trajectory_tracking_enabled = False
            self._trajectory_last_position = None
            self._trajectory_length_m = 0.0

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        pose.pose.position.x = float(self.start[0])
        pose.pose.position.y = float(self.start[1])
        pose.pose.position.z = float(self.start[2])
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        deadline = time.time() + self.config.reset_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            self._reset_pub.publish(pose)
            obs = self._try_get_observation()
            if obs is not None:
                pos = obs["state"]["position"]
                err = float(np.linalg.norm(pos - self.start))
                if err <= self.config.reset_tolerance_m:
                    self.episode_step = 0
                    self.prev_distance_xy = obs["goal"]["distance_xy"]
                    self.prev_abs_goal_dz = abs(float(obs["goal"]["dz"]))
                    self.prev_action_id = -1
                    self.stop()
                    token = self.observation_token()
                    s = self.config.reset_settle_s if settle is None else float(settle)
                    if s > 0.0:
                        rospy.sleep(s)
                    final_observation = self._get_observation_after_token(
                        token, require_new_state=True, require_new_depth=True
                    )
                    self._record_pairing_candidate_boundary(
                        boundary_kind="reset",
                        previous_token=token,
                        selected_observation=final_observation,
                    )
                    with self._lock:
                        self._trajectory_last_position = np.asarray(
                            final_observation["state"]["position"], dtype=np.float64
                        ).reshape(3)
                        self._trajectory_length_m = 0.0
                        self._trajectory_tracking_enabled = True
                    return final_observation
            rospy.sleep(0.05)

        raise TimeoutError("reset failed: target={} timeout={}s".format(
            self.start.tolist(), self.config.reset_timeout
        ))

    def _next_reset_identity(self) -> Tuple[str, str]:
        """Reserve the next reset identity without consulting telemetry.

        Reservation happens before transport submission so a failed reset
        cannot cause the next mission to reuse an identity whose payload is
        already cached by the reliable backend.
        """

        generation = int(self._reset_generation) + 1
        self._reset_generation = generation
        return "episode-{}".format(generation), "reset-{}".format(generation)

    @staticmethod
    def _validate_reliable_reset_observation(
        observation: Dict[str, Any], episode_id: str, reset_id: str
    ) -> None:
        if not isinstance(observation, dict):
            raise RuntimeError("reliable reset observation must be a mapping")
        state = observation.get("state")
        goal = observation.get("goal")
        if not isinstance(state, dict) or not isinstance(goal, dict):
            raise RuntimeError("reliable reset observation is incomplete")
        supplied_episode = observation.get("episode_id", state.get("episode_id"))
        supplied_reset = observation.get("reset_id", state.get("reset_id"))
        if supplied_episode != episode_id or supplied_reset != reset_id:
            raise RuntimeError(
                "reliable reset observation identity mismatch: received=({}, {}) expected=({}, {})".format(
                    supplied_episode, supplied_reset, episode_id, reset_id
                )
            )

    def step(self, action_id: int):
        return self.step_primitive(action_id)

    def _execute_primitive_physics_clock(
        self,
        commands: Sequence[Sequence[float]],
    ) -> Tuple[Dict[str, Any], XMState]:
        """Submit one complete primitive and wait for Unity post-integration acks."""
        if self._reliable_v4_backend is not None or not self._legacy_command_path_enabled:
            raise RuntimeError(
                "legacy primitive execution is disabled when reliable v4 is enabled"
            )
        frame_commands = [np.asarray(command, dtype=np.float32).reshape(4) for command in commands]
        frame_count = len(frame_commands)
        if frame_count <= 0:
            raise ValueError("primitive execution requires at least one command frame")
        if self._primitive_execution_pub.get_num_connections() <= 0:
            raise RuntimeError(
                "deterministic primitive protocol unavailable: /xm/primitive_execution has no "
                "subscriber; schema-v3 unity_bridge_node is required and legacy fallback is disabled"
            )

        execution_id = secrets.randbits(63)
        command_ids = []
        used_command_ids = set()
        while len(command_ids) < frame_count:
            command_id = secrets.randbits(63)
            if command_id not in used_command_ids:
                used_command_ids.add(command_id)
                command_ids.append(command_id)

        execution = PrimitiveExecution()
        execution.execution_id = execution_id
        execution.frame_count = frame_count
        execution.command_ids = command_ids
        execution.commands = [
            self._make_twist(
                float(command[0]), float(command[1]), float(command[2]), float(command[3])
            )
            for command in frame_commands
        ]

        self._begin_execution_transport_record(
            execution_id=execution_id,
            frame_count=frame_count,
            command_ids=command_ids,
        )

        with self._execution_condition:
            self._execution_acknowledgements[execution_id] = []
        self._primitive_execution_pub.publish(execution)

        deadline = time.monotonic() + float(self.config.obs_timeout)
        applied_frames: list[Dict[str, Any]] = []
        with self._execution_condition:
            while not rospy.is_shutdown():
                applied_frames = list(self._execution_acknowledgements.get(execution_id, []))
                if any(int(frame["execution_status"]) in (2, 3) for frame in applied_frames):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._execution_condition.wait(timeout=min(remaining, 0.1))
            self._execution_acknowledgements.pop(execution_id, None)

        failed_frames = [
            frame for frame in applied_frames if int(frame["execution_status"]) == 3
        ]
        if failed_frames:
            partial_receipt = {
                "execution_id": execution_id,
                "requested_frame_count": frame_count,
                "received_frames": [
                    {key: value for key, value in frame.items() if key != "state_message"}
                    for frame in applied_frames
                ],
            }
            result = classify_primitive_execution_result(
                partial_receipt,
                expected_frame_count=frame_count,
                expected_command_ids=command_ids,
            )
            self._finish_execution_transport_record(
                execution_id=execution_id,
                applied_frames=applied_frames,
                result=result,
                collector_terminal="failed_acknowledgement",
            )
            raise PrimitiveExecutionAbortedError(
                execution_id=execution_id,
                frame_count=frame_count,
                frames=applied_frames,
                execution_result=result,
                terminal_state_message=failed_frames[-1].get("state_message"),
            )
        completion_frames = [
            frame for frame in applied_frames if int(frame["execution_status"]) == 2
        ]
        if not completion_frames:
            result = classify_primitive_execution_result(
                {
                    "execution_id": execution_id,
                    "requested_frame_count": frame_count,
                    "received_frames": [
                        {
                            key: value for key, value in frame.items()
                            if key != "state_message"
                        }
                        for frame in applied_frames
                    ],
                },
                expected_frame_count=frame_count,
                expected_command_ids=command_ids,
                timed_out=True,
            )
            self._finish_execution_transport_record(
                execution_id=execution_id,
                applied_frames=applied_frames,
                result=result,
                collector_terminal="timeout_without_completion",
            )
            raise PrimitiveExecutionContractError(
                "primitive execution acknowledgement timeout: execution_id={} "
                "received_frames={}/{} classifier={}".format(
                    execution_id,
                    len(applied_frames),
                    frame_count,
                    result.get("kind"),
                )
            )

        completion = completion_frames[-1]
        endpoint_state = completion["state_message"]
        receipt = {
            "execution_id": execution_id,
            "requested_frame_count": frame_count,
            "applied_frames": [
                {key: value for key, value in frame.items() if key != "state_message"}
                for frame in applied_frames
            ],
            "endpoint_state_id": int(completion["applied_state_id"]),
        }
        result = classify_primitive_execution_result(
            receipt,
            expected_frame_count=frame_count,
            expected_command_ids=command_ids,
        )
        self._finish_execution_transport_record(
            execution_id=execution_id,
            applied_frames=applied_frames,
            result=result,
            collector_terminal="completed_acknowledgement",
        )
        if result["kind"] != PRIMITIVE_EXECUTION_RESULT_COMPLETED:
            raise PrimitiveExecutionContractError(
                "completed primitive execution failed result classification: {}"
                .format(result.get("error", "unknown classifier error"))
            )
        validated = result["receipt"]
        statuses = [int(frame["execution_status"]) for frame in validated["applied_frames"]]
        if statuses[:-1] != [1] * (frame_count - 1) or statuses[-1:] != [2]:
            raise RuntimeError(
                "primitive execution status contract violated: execution_id={} statuses={}"
                .format(execution_id, statuses)
            )
        return validated, endpoint_state

    def _get_observation_for_endpoint_state(
        self,
        endpoint_state: XMState,
        *,
        previous_depth_seq: int,
    ) -> Dict[str, Any]:
        """Pair fresh depth with the exact frame-N-1 post-integration state."""
        self.telemetry_lookup_count = int(getattr(self, "telemetry_lookup_count", 0)) + 1
        deadline = time.monotonic() + float(self.config.obs_timeout)
        state_stamp_ns = int(endpoint_state.sim_time_ns)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self._lock:
                depth = self._latest_depth
                camera_info = self._latest_camera_info
                depth_seq = int(self._depth_seq)
            if depth is not None and depth_seq > int(previous_depth_seq):
                depth_stamp_ns = self._stamp_ns(depth)
                sensor_skew_s = abs(state_stamp_ns - depth_stamp_ns) * 1.0e-9
                is_after_endpoint = state_stamp_ns <= 0 or depth_stamp_ns >= state_stamp_ns
                sync_ok = (
                    not bool(self.config.enforce_sensor_sync)
                    or state_stamp_ns <= 0
                    or depth_stamp_ns <= 0
                    or sensor_skew_s <= float(self.config.max_sensor_skew_s)
                )
                if is_after_endpoint and sync_ok:
                    observation = self._build_observation(
                        endpoint_state,
                        depth,
                        state_seq=-1,
                        depth_seq=depth_seq,
                        state_stamp_ns=state_stamp_ns,
                        depth_stamp_ns=depth_stamp_ns,
                        camera_info=camera_info,
                    )
                    self._record_pairing_candidate_boundary(
                        boundary_kind="primitive_endpoint",
                        previous_token=(-1, int(previous_depth_seq)),
                        selected_observation=observation,
                        endpoint_state=endpoint_state,
                    )
                    return observation
            rospy.sleep(0.002)
        raise TimeoutError(
            "observation timeout waiting for depth captured after primitive endpoint state_id={}"
            .format(int(endpoint_state.state_id))
        )

    def step_primitive(
        self,
        action_id: int,
        obs_before: Optional[Dict[str, Any]] = None,
        precomputed_mask: Optional[np.ndarray] = None,
        precomputed_mask_info: Optional[Dict[str, Any]] = None,
        audit_command_frames: bool = False,
    ):
        """Execute one primitive using the same observation used for selection.

        Passing ``obs_before`` and its precomputed mask avoids a second sensor
        read and collision scan between policy inference and command
        publication. This is important for continuous-flight evaluation.
        """
        if self.mpl is None:
            raise RuntimeError("MPL library is not loaded. Run rosrun planning generate_motion_primitives.py first.")
        if action_id < 0 or action_id >= self.action_space_n:
            raise IndexError("action_id {} out of range [0,{})".format(action_id, self.action_space_n))

        obs0 = obs_before if obs_before is not None else self._get_observation_blocking()
        z0 = float(obs0["state"]["z"])
        altitude_violation0 = bool(
            obs0["safety"].get("altitude_violation", False)
            or z0 < self.config.z_min
            or z0 > self.config.z_max
        )
        if altitude_violation0:
            self.stop()
            mask, mask_info = self.get_action_mask(obs0, return_info=True)
            info = {
                "action_id": int(action_id),
                "action_valid_by_height": False,
                "action_valid_by_mask": False,
                "invalid_action": False,
                "dead_end": False,
                "success": False,
                "collided": bool(obs0["safety"].get("collided", False)),
                "altitude_violation": True,
                "hard_altitude_violation": True,
                "extreme_altitude_violation": bool(
                    z0 < self.config.z_hard_min or z0 > self.config.z_hard_max
                ),
                "timeout": False,
                "far": False,
                "distance_xy": float(obs0["goal"]["distance_xy"]),
                "abs_goal_dz": abs(float(obs0["goal"]["dz"])),
                "min_clearance": float(obs0["safety"]["min_clearance"]),
                "z": z0,
                "episode_step": int(self.episode_step),
                "done_reason": "altitude_violation",
                "action_mask_info": mask_info,
            }
            self.episode_step += 1
            obs0["action_mask"] = mask
            return obs0, float(self.config.reward_altitude_violation), True, info
        if precomputed_mask is None or precomputed_mask_info is None:
            mask, mask_info = self.get_action_mask(obs0, return_info=True)
        else:
            mask = np.asarray(precomputed_mask, dtype=np.bool_).reshape(self.action_space_n)
            mask_info = dict(precomputed_mask_info)
        action_valid_by_mask = bool(mask[action_id]) if mask.size else False

        if bool(self.config.terminate_on_dead_end) and bool(mask_info.get("dead_end", False)):
            reward = float(self.config.reward_step + self.config.reward_dead_end)
            info = {
                "action_id": int(action_id),
                "action_valid_by_height": bool(mask_info.get("height_mask", mask)[action_id]) if mask.size else False,
                "action_valid_by_mask": action_valid_by_mask,
                "invalid_action": False,
                "dead_end": True,
                "success": False,
                "collided": bool(obs0["safety"]["collided"]),
                "altitude_violation": bool(obs0["safety"]["altitude_violation"]),
                "hard_altitude_violation": False,
                "timeout": False,
                "far": False,
                "distance_xy": float(obs0["goal"]["distance_xy"]),
                "min_clearance": float(obs0["safety"]["min_clearance"]),
                "z": float(obs0["state"]["z"]),
                "episode_step": int(self.episode_step),
                "done_reason": "dead_end",
                "action_mask_info": mask_info,
            }
            self.episode_step += 1
            obs0["action_mask"] = mask
            obs0["prev_action_id"] = int(self.prev_action_id)
            obs0["prev_action_onehot"] = self._prev_action_onehot()
            return obs0, reward, True, info

        if bool(self.config.terminate_on_invalid_action) and not action_valid_by_mask:
            reward = float(self.config.reward_step + self.config.reward_invalid_action)
            info = {
                "action_id": int(action_id),
                "action_valid_by_height": bool(mask_info.get("height_mask", mask)[action_id]) if mask.size else False,
                "action_valid_by_mask": False,
                "invalid_action": True,
                "dead_end": False,
                "success": False,
                "collided": bool(obs0["safety"]["collided"]),
                "altitude_violation": bool(obs0["safety"]["altitude_violation"]),
                "hard_altitude_violation": False,
                "timeout": False,
                "far": False,
                "distance_xy": float(obs0["goal"]["distance_xy"]),
                "min_clearance": float(obs0["safety"]["min_clearance"]),
                "z": float(obs0["state"]["z"]),
                "episode_step": int(self.episode_step),
                "done_reason": "invalid_action",
                "action_mask_info": mask_info,
            }
            self.episode_step += 1
            obs0["action_mask"] = mask
            obs0["prev_action_id"] = int(self.prev_action_id)
            obs0["prev_action_onehot"] = self._prev_action_onehot()
            return obs0, reward, True, info

        p0 = np.asarray(obs0["state"]["position"], dtype=np.float32).copy()
        yaw0 = float(obs0["state"]["yaw"])
        token_before_command = self.observation_token()
        primitive_commands = list(self.mpl.command_sequence(action_id))
        command_frame_audit = None
        if bool(audit_command_frames):
            with self._lock:
                self._command_frame_audit_states = []
                if self._latest_state is not None:
                    self._command_frame_audit_states.append(
                        self._command_audit_state_record(
                            self._latest_state,
                            receive_monotonic_ns=self._latest_state_receive_monotonic_ns,
                            state_sequence=self._state_seq,
                            source="initial_snapshot",
                        )
                    )
                self._command_frame_audit_active = True
            command_frame_audit = {
                "contract_id": "[DEBUG-CMD-TICK-c83e]command_frame_audit",
                "ros_use_sim_time": self._ros_use_sim_time,
                "control_dt_s": float(self.mpl.control_dt_s),
                "commands": [],
                "states": None,
            }
        try:
            batch_publish_monotonic_ns = time.monotonic_ns() if command_frame_audit is not None else 0
            batch_publish_ros_time_ns = (
                int(rospy.Time.now().to_nsec()) if command_frame_audit is not None else 0
            )
            if self._reliable_v4_backend is not None:
                reliable_step = self._reliable_v4_backend.execute(
                    action_id=int(action_id),
                    primitive_commands=primitive_commands,
                )
                reliable_status = str(reliable_step.status)
                reliable_result = dict(reliable_step.result or {})
                cached_transition = self._reliable_transition_cache.get(
                    int(reliable_step.execution_id)
                )
                if cached_transition is not None:
                    cached_status = str(cached_transition[3].get("terminal_result_status", ""))
                    if cached_status != reliable_status:
                        raise PrimitiveExecutionContractError(
                            "duplicate reliable result status conflict for execution_id={}".format(
                                reliable_step.execution_id
                            )
                        )
                    return cached_transition

                if reliable_status == "COMPLETE":
                    if reliable_step.observation is None:
                        raise PrimitiveExecutionContractError(
                            "reliable v4 COMPLETE is missing exact endpoint observation"
                        )
                elif reliable_status == "FAILED" and reliable_result.get("reason_code") == "COLLISION":
                    if reliable_step.observation is None:
                        raise ReliableV4BackendError(
                            "reliable v4 COLLISION is missing exact terminal observation"
                        )
                else:
                    raise PrimitiveExecutionContractError(
                        "reliable v4 terminal result is not a legal environment transition: "
                        "status={} reason={}".format(
                            reliable_status, reliable_result.get("reason_code", "")
                        )
                    )
                receipt = reliable_result
                endpoint_state = reliable_step.observation
                reliable_terminal_status = reliable_status
            else:
                receipt, endpoint_state = self._execute_primitive_physics_clock(primitive_commands)
                reliable_terminal_status = "COMPLETE"
            if command_frame_audit is not None:
                for command_index, command in enumerate(primitive_commands):
                    applied_frames = receipt.get("applied_frames", [])
                    if command_index >= len(applied_frames):
                        break
                    command_frame_audit["commands"].append({
                        "command_index": int(command_index),
                        "command": np.asarray(command, dtype=np.float32).astype(float).tolist(),
                        "batch_publish_ros_time_ns": batch_publish_ros_time_ns,
                        "batch_publish_monotonic_ns": batch_publish_monotonic_ns,
                        "applied_frame": receipt["applied_frames"][command_index],
                    })
        except PrimitiveExecutionAbortedError as error:
            # The evaluator must pair the real failed Unity state with a depth
            # frame captured after this primitive was submitted.  Do not turn
            # it into a completed endpoint here.
            error.previous_depth_seq = int(token_before_command[1])
            raise
        finally:
            if command_frame_audit is not None:
                with self._lock:
                    self._command_frame_audit_active = False
                    command_frame_audit["states"] = list(self._command_frame_audit_states)
                    self._command_frame_audit_states = []

        if self._reliable_v4_backend is not None:
            if isinstance(endpoint_state, dict):
                obs = endpoint_state
            else:
                obs = self._build_reliable_v4_observation(endpoint_state)
        else:
            obs = self._get_observation_for_endpoint_state(
                endpoint_state, previous_depth_seq=int(token_before_command[1])
            )
        p1 = np.asarray(obs["state"]["position"], dtype=np.float32).copy()
        delta_world = p1 - p0
        delta_body = rotation_z(-yaw0).dot(delta_world)
        ref_body = self.mpl.endpoint(action_id).astype(np.float32)
        ref_world = rotation_z(yaw0).dot(ref_body)

        reward, done, info = self._compute_reward_done(obs, action_id)
        self.episode_step += 1
        self.prev_action_id = int(action_id)
        info.update({
            "action_id": int(action_id),
            "action_valid_by_height": bool(mask_info.get("height_mask", mask)[action_id]) if mask.size else False,
            "action_valid_by_mask": action_valid_by_mask,
            "invalid_action": False,
            "action_mask_info": mask_info,
            "action_meta": self.mpl.action_metadata(action_id),
            "primitive_endpoint_ref_body": ref_body,
            "primitive_endpoint_ref_world": ref_world,
            "primitive_delta_body": delta_body,
            "primitive_delta_world": delta_world,
            "primitive_endpoint_error_body_m": float(np.linalg.norm(delta_body - ref_body)),
            "primitive_endpoint_error_world_m": float(np.linalg.norm(delta_world - ref_world)),
            "start_yaw": yaw0,
            "end_yaw": float(obs["state"]["yaw"]),
            "primitive_execution": receipt,
            "terminal_result_status": reliable_terminal_status,
            "primitive_completed": reliable_terminal_status == "COMPLETE",
        })
        if self._reliable_v4_backend is not None:
            execution = receipt if isinstance(receipt, dict) else {}
            info.update({
                **exact_endpoint_metadata(),
                "reliable_v4": True,
                "reliable_v4_execution_id": int(reliable_step.execution_id),
                "reliable_v4_runtime_instance_id": str(
                    execution.get("runtime_instance_id", "")
                ),
                "reliable_v4_command_sequence_hash": str(
                    execution.get("command_sequence_hash", "")
                ),
                "reliable_v4_observation_ref": execution.get(
                    "endpoint_observation_ref"
                ),
                "reliable_v4_snapshot_hash": getattr(
                    endpoint_state, "snapshot_hash", None
                ),
                "telemetry_lookup_count": 0,
            })
        if command_frame_audit is not None:
            info["command_frame_audit"] = command_frame_audit

        obs["prev_action_id"] = int(self.prev_action_id)
        obs["prev_action_onehot"] = self._prev_action_onehot()
        next_mask, next_mask_info = self.get_action_mask(obs, return_info=True)
        obs["action_mask"] = next_mask
        info["next_action_mask_info"] = next_mask_info
        reward, done, info = apply_post_action_dead_end(
            reward=reward,
            done=done,
            info=info,
            next_mask_info=next_mask_info,
            terminate_on_dead_end=bool(self.config.terminate_on_dead_end),
            reward_dead_end=float(self.config.reward_dead_end),
        )
        result_tuple = (obs, reward, done, info)
        if self._reliable_v4_backend is not None:
            self._reliable_transition_cache[int(reliable_step.execution_id)] = result_tuple
        return result_tuple

    def materialize_terminal_abort(
        self,
        error: PrimitiveExecutionAbortedError,
        *,
        action_id: int,
        obs_before: Dict[str, Any],
        precomputed_mask: Optional[np.ndarray] = None,
        precomputed_mask_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Turn a classifier-validated collision abort into one terminal step.

        This is intentionally separate from ``step_primitive``: a terminal
        abort is not a completed primitive and never receives an endpoint
        receipt.  The post-action task/reward transition uses Unity's real
        failed state, exactly once.
        """
        result = error.execution_result
        if not isinstance(result, dict) or result.get("kind") != (
            PRIMITIVE_EXECUTION_RESULT_TERMINAL_ABORT
        ):
            raise PrimitiveExecutionContractError(
                "cannot materialize an unclassified primitive execution failure"
            )
        if result.get("terminal_reason") != "collision" or bool(
            result.get("retry_allowed", True)
        ):
            raise PrimitiveExecutionContractError(
                "only non-retryable collision terminal aborts are supported"
            )
        terminal_state = error.terminal_state_message
        if terminal_state is None:
            raise PrimitiveExecutionContractError(
                "terminal abort is missing its real Unity failure state"
            )
        terminal_record = result.get("terminal_state")
        if not isinstance(terminal_record, dict) or int(
            terminal_record.get("applied_state_id", -1)
        ) != int(terminal_state.state_id):
            raise PrimitiveExecutionContractError(
                "terminal abort state does not match the classified failed acknowledgement"
            )
        if error.previous_depth_seq is None:
            raise PrimitiveExecutionContractError(
                "terminal abort is missing its pre-command depth token"
            )

        if precomputed_mask is None or precomputed_mask_info is None:
            mask, mask_info = self.get_action_mask(obs_before, return_info=True)
        else:
            mask = np.asarray(precomputed_mask, dtype=np.bool_).reshape(
                self.action_space_n
            )
            mask_info = dict(precomputed_mask_info)

        obs = self._get_observation_for_endpoint_state(
            terminal_state, previous_depth_seq=int(error.previous_depth_seq)
        )
        if not bool(obs["safety"].get("collided", False)):
            raise PrimitiveExecutionContractError(
                "terminal abort state lost Unity's collision terminal condition"
            )
        p0 = np.asarray(obs_before["state"]["position"], dtype=np.float32)
        p1 = np.asarray(obs["state"]["position"], dtype=np.float32)
        delta_world = p1 - p0
        yaw0 = float(obs_before["state"]["yaw"])
        delta_body = rotation_z(-yaw0).dot(delta_world)

        reward, done, info = self._compute_reward_done(obs, action_id)
        if not bool(done) or not bool(info.get("collided", False)):
            raise PrimitiveExecutionContractError(
                "classified collision terminal abort did not produce collision terminal reward"
            )
        self.episode_step += 1
        self.prev_action_id = int(action_id)
        partial_receipt = result["partial_receipt"]
        applied_frame_count = len(partial_receipt["received_frames"]) - 1
        info.update({
            "action_id": int(action_id),
            "action_valid_by_height": bool(
                mask_info.get("height_mask", mask)[action_id]
            ) if mask.size else False,
            "action_valid_by_mask": bool(mask[action_id]) if mask.size else False,
            "invalid_action": False,
            "action_mask_info": mask_info,
            "action_meta": self.mpl.action_metadata(action_id),
            "primitive_completed": False,
            "terminal_abort": True,
            "terminal_reason": "collision",
            "effective_integration_ticks": int(applied_frame_count),
            "applied_frame_count": int(applied_frame_count),
            "primitive_delta_body": delta_body,
            "primitive_delta_world": delta_world,
            "start_yaw": yaw0,
            "end_yaw": float(obs["state"]["yaw"]),
            "primitive_execution_result": result,
        })
        obs["prev_action_id"] = int(self.prev_action_id)
        obs["prev_action_onehot"] = self._prev_action_onehot()
        next_mask, next_mask_info = self.get_action_mask(obs, return_info=True)
        obs["action_mask"] = next_mask
        info["next_action_mask_info"] = next_mask_info
        reward, done, info = apply_post_action_dead_end(
            reward=reward,
            done=done,
            info=info,
            next_mask_info=next_mask_info,
            terminate_on_dead_end=bool(self.config.terminate_on_dead_end),
            reward_dead_end=float(self.config.reward_dead_end),
        )
        return obs, reward, done, info

    def execute_primitive_stream(
        self,
        action_ids: Sequence[int],
        stop_at_end: bool = False,
        require_current_mask: bool = True,
        expected_end_positions: Optional[Sequence[Sequence[float]]] = None,
        max_drift_m: float = 0.0,
        stop_on_terminal: bool = True,
        stop_on_drift: bool = True,
    ) -> list[Dict[str, Any]]:
        """Execute prefetched primitives continuously with actual-state gates.

        Each action is checked against the latest actual state immediately
        before publication. Optional predicted endpoints allow the caller to
        terminate the stream at the first large tracking drift and replan from
        the actual state without discarding the episode.
        """
        if self._reliable_v4_backend is not None or not self._legacy_command_path_enabled:
            raise RuntimeError(
                "legacy primitive stream is disabled when reliable v4 is enabled"
            )
        if self.mpl is None:
            raise RuntimeError("continuous primitive stream requires an MPL library")
        ids = [int(action_id) for action_id in action_ids]
        if not ids:
            return []
        if any(action_id < 0 or action_id >= self.action_space_n for action_id in ids):
            raise IndexError("continuous stream action outside valid range")
        expected = None
        if expected_end_positions is not None:
            expected = [np.asarray(value, dtype=np.float32).reshape(3) for value in expected_end_positions]
            if len(expected) != len(ids):
                raise ValueError("expected_end_positions length must match action_ids")

        rate = rospy.Rate(1.0 / float(self.mpl.control_dt_s))
        boundaries: list[Dict[str, Any]] = []
        for stream_index, action_id in enumerate(ids):
            obs_before = self._get_observation_blocking()
            z_before = float(obs_before["state"]["z"])
            altitude_violation_before = bool(
                obs_before["safety"].get("altitude_violation", False)
                or z_before < self.config.z_min
                or z_before > self.config.z_max
            )
            if altitude_violation_before:
                self.stop()
                boundaries.append({
                    "action_id": action_id,
                    "stream_index": stream_index,
                    "obs_before": obs_before,
                    "skipped_invalid": True,
                    "dead_end_before": False,
                    "altitude_violation_before": True,
                    "hard_altitude_after": True,
                    "stream_break_reason": "altitude_violation",
                })
                break
            mask = None
            mask_info = None
            if require_current_mask:
                mask, mask_info = self.get_action_mask(obs_before, return_info=True)
                if bool(mask_info.get("dead_end", False)) or not bool(mask[action_id]):
                    self.stop()
                    boundaries.append({
                        "action_id": action_id,
                        "stream_index": stream_index,
                        "obs_before": obs_before,
                        "skipped_invalid": True,
                        "dead_end_before": bool(mask_info.get("dead_end", False)),
                        "action_mask": np.asarray(mask, dtype=np.bool_),
                        "action_mask_info": mask_info,
                        "valid_action_count": int(mask_info.get("combined_valid_count", int(mask.sum()))),
                        "stream_break_reason": "dead_end" if bool(mask_info.get("dead_end", False)) else "invalid_prefetch_action",
                    })
                    break

            token_before_command = self.observation_token()
            position_before = np.asarray(obs_before["state"]["position"], dtype=np.float32).copy()
            yaw_before = float(obs_before["state"]["yaw"])
            for command in self.mpl.command_sequence(action_id):
                self._cmd_pub.publish(
                    self._make_twist(float(command[0]), float(command[1]), float(command[2]), float(command[3]))
                )
                rate.sleep()
            obs_after = self._get_observation_after_token(
                token_before_command, require_new_state=True, require_new_depth=True
            )
            position_after = np.asarray(obs_after["state"]["position"], dtype=np.float32)
            delta_world = position_after - position_before
            delta_body = rotation_z(-yaw_before).dot(delta_world)
            reference_body = self.mpl.endpoint(action_id).astype(np.float32)
            drift = float(np.linalg.norm(position_after - expected[stream_index])) if expected is not None else float("nan")
            collided = bool(obs_after["safety"].get("collided", False))
            z_after = float(obs_after["state"]["z"])
            altitude_violation = bool(
                obs_after["safety"].get("altitude_violation", False)
                or z_after < self.config.z_min
                or z_after > self.config.z_max
            )
            extreme_altitude = bool(z_after < self.config.z_hard_min or z_after > self.config.z_hard_max)
            success = goal_reached(
                position_after,
                obs_after["goal"]["position"],
                radius_xy=self.config.goal_radius_xy,
                tolerance_z=self.config.goal_tolerance_z,
            )
            drift_break = bool(
                expected is not None
                and float(max_drift_m) > 0.0
                and np.isfinite(drift)
                and drift > float(max_drift_m)
            )
            terminal_break = bool(stop_on_terminal and (collided or altitude_violation or success))
            boundary = {
                "action_id": action_id,
                "stream_index": stream_index,
                "obs_before": obs_before,
                "obs_after": obs_after,
                "endpoint_error_body_m": float(np.linalg.norm(delta_body - reference_body)),
                "endpoint_error_world_m": float(
                    np.linalg.norm(delta_world - rotation_z(yaw_before).dot(reference_body))
                ),
                "stream_drift_m": drift,
                "episode_actual_path_length_m": self.actual_trajectory_length_m(),
                "skipped_invalid": False,
                "action_mask": None if mask is None else np.asarray(mask, dtype=np.bool_),
                "action_mask_info": mask_info,
                "collided_after": collided,
                "altitude_violation_after": altitude_violation,
                # Compatibility field: callers historically used this as the
                # altitude terminal flag. It now follows the strict contract.
                "hard_altitude_after": altitude_violation,
                "extreme_altitude_after": extreme_altitude,
                "success_after": success,
                "stream_break_reason": "",
            }
            if terminal_break:
                boundary["stream_break_reason"] = "success" if success else ("collision" if collided else "altitude_violation")
            elif drift_break and stop_on_drift:
                boundary["stream_break_reason"] = "tracking_drift"
            boundaries.append(boundary)
            if terminal_break or (drift_break and stop_on_drift):
                self.stop()
                break

        if stop_at_end:
            self.stop()
        return boundaries

    def stop(self) -> None:
        if self._reliable_v4_backend is not None:
            # Reliable v4 reset owns lifecycle cleanup. Publishing legacy stop
            # or zero-velocity commands would re-enter the v3 command decoder.
            return
        if not self._legacy_command_path_enabled:
            raise RuntimeError("no environment command path is enabled")
        self._stop_pub.publish(Bool(data=True))
        self._cmd_pub.publish(self._make_twist(0.0, 0.0, 0.0, 0.0))

    def get_action_mask(self, obs: Optional[Dict[str, Any]] = None, return_info: bool = False):
        """Return available action mask.

        Default behavior is the original altitude-only mask. If enabled, the
        local depth screen and/or privileged global collision mask are ANDed:

            altitude_mask AND local_depth_mask AND global_voxel_mask

        The extra privileged mask is intended for safety-shielded training or
        diagnostics. Keep it disabled when evaluating pure depth-only policies.
        """
        if self.mpl is None:
            mask = np.zeros((0,), dtype=np.bool_)
            info = {
                "dead_end": True,
                "height_valid_count": 0,
                "depth_valid_count": -1,
                "depth_blocked_count": -1,
                "depth_checked_sample_count": 0,
                "global_valid_count": -1,
                "combined_valid_count": 0,
            }
            self._last_action_mask_info = info
            return (mask, info) if return_info else mask

        if obs is None:
            obs = self._try_get_observation()
        if obs is None:
            # No current state means safety cannot be established. Fail closed
            # instead of advertising every action as executable.
            mask = np.zeros((self.action_space_n,), dtype=np.bool_)
            info = {
                "dead_end": True,
                "height_valid_count": 0,
                "depth_valid_count": -1,
                "depth_blocked_count": -1,
                "depth_checked_sample_count": 0,
                "global_valid_count": -1,
                "combined_valid_count": 0,
            }
            self._last_action_mask_info = info
            return (mask, info) if return_info else mask

        z = float(obs["state"]["z"])
        current_altitude_violation = bool(
            obs["safety"].get("altitude_violation", False)
            or z < self.config.z_min
            or z > self.config.z_max
        )
        if current_altitude_violation:
            height_mask = np.zeros((self.action_space_n,), dtype=np.bool_)
        else:
            height_mask = self.mpl.valid_action_mask(
                z,
                z_min=self.config.z_min,
                z_max=self.config.z_max,
                margin=self.config.action_mask_z_margin,
            ).astype(np.bool_)

        depth_mask = None
        depth_info: Dict[str, Any] = {}
        if bool(self.config.use_depth_collision_mask):
            depth_mask, depth_info = local_depth_action_mask(
                self.mpl,
                np.asarray(obs["depth_m"], dtype=np.float32),
                DepthSafetyConfig(
                    horizontal_fov_deg=float(self.config.depth_camera_hfov_deg),
                    vertical_fov_deg=float(self.config.depth_camera_vfov_deg),
                    min_forward_m=float(self.config.depth_mask_min_forward_m),
                    path_sample_stride=max(1, int(self.config.depth_mask_path_sample_stride)),
                    collision_radius_m=float(self.config.depth_mask_collision_radius_m),
                    depth_slack_m=float(self.config.depth_mask_slack_m),
                    patch_radius_px=max(0, int(self.config.depth_mask_patch_radius_px)),
                    max_patch_radius_px=max(0, int(self.config.depth_mask_max_patch_radius_px)),
                    valid_depth_min_m=float(self.config.depth_min_m),
                    valid_depth_max_m=float(self.config.depth_sensor_max_m) - 0.05,
                ),
                intrinsics=dict(obs["depth_intrinsics"]),
            )
            mask = height_mask & depth_mask
        else:
            mask = height_mask.copy()

        global_mask = None
        global_min_dist = None
        if bool(self.config.use_global_collision_mask):
            if self._collision_checker is None:
                raise RuntimeError("use_global_collision_mask=True but collision checker is not initialized")
            pos = np.asarray(obs["state"]["position"], dtype=np.float32).reshape(3)
            yaw = float(obs["state"].get("yaw", 0.0))
            results = self._collision_checker.check_all_actions(
                self.mpl,
                pos,
                yaw,
                check_step=max(1, int(self.config.collision_check_step)),
                return_path_map=False,
            )
            global_mask = np.zeros((self.action_space_n,), dtype=np.bool_)
            global_min_dist = np.full((self.action_space_n,), np.inf, dtype=np.float32)
            for r in results:
                aid = int(r.get("action_id", -1))
                if 0 <= aid < self.action_space_n:
                    global_mask[aid] = bool(r.get("valid", False))
                    d = float(r.get("min_distance_voxel_center_m", np.inf))
                    global_min_dist[aid] = d if np.isfinite(d) else np.inf
            mask &= global_mask

        info = {
            "dead_end": bool(mask.size > 0 and int(mask.sum()) == 0),
            "height_valid_count": int(height_mask.sum()),
            "depth_valid_count": int(depth_mask.sum()) if depth_mask is not None else -1,
            "depth_blocked_count": int(depth_info.get("depth_blocked_count", 0)) if depth_mask is not None else -1,
            "depth_checked_sample_count": int(depth_info.get("depth_checked_sample_count", 0)) if depth_mask is not None else 0,
            "depth_frame_valid_fraction": float(depth_info.get("depth_frame_valid_fraction", np.nan)),
            "depth_action_checked_sample_count": depth_info.get("depth_action_checked_sample_count"),
            "depth_action_valid_patch_sample_count": depth_info.get("depth_action_valid_patch_sample_count"),
            "depth_action_capped_patch_sample_count": depth_info.get("depth_action_capped_patch_sample_count"),
            "depth_action_max_unclipped_patch_radius_px": depth_info.get("depth_action_max_unclipped_patch_radius_px"),
            "depth_action_mean_invalid_patch_fraction": depth_info.get("depth_action_mean_invalid_patch_fraction"),
            "depth_action_min_ray_clearance_m": depth_info.get("depth_action_min_ray_clearance_m"),
            "global_valid_count": int(global_mask.sum()) if global_mask is not None else -1,
            "combined_valid_count": int(mask.sum()),
            "height_mask": height_mask,
            "depth_mask": depth_mask,
            "global_mask": global_mask,
            "global_min_distance": global_min_dist,
        }
        self._last_action_mask_info = info
        return (mask, info) if return_info else mask

    # ---------------------------------------------------------------------
    # Observation and reward
    # ---------------------------------------------------------------------

    @staticmethod
    def _stamp_ns(message) -> int:
        stamp = getattr(getattr(message, "header", None), "stamp", None)
        if stamp is None:
            return 0
        if hasattr(stamp, "to_nsec"):
            return int(stamp.to_nsec())
        return int(getattr(stamp, "secs", 0)) * 1_000_000_000 + int(getattr(stamp, "nsecs", 0))

    def _try_get_observation(self) -> Optional[Dict[str, Any]]:
        if self._reliable_v4_backend is not None:
            raise RuntimeError(
                "latest ROS telemetry observation is disabled for reliable exact collection"
            )
        with self._lock:
            state = self._latest_state
            depth = self._latest_depth
            camera_info = self._latest_camera_info
            state_seq = int(self._state_seq)
            depth_seq = int(self._depth_seq)
        if state is None or depth is None:
            return None
        state_stamp_ns = int(state.sim_time_ns)
        depth_stamp_ns = self._stamp_ns(depth)
        sensor_skew_s = abs(state_stamp_ns - depth_stamp_ns) * 1.0e-9
        if (
            bool(self.config.enforce_sensor_sync)
            and state_stamp_ns > 0
            and depth_stamp_ns > 0
            and sensor_skew_s > float(self.config.max_sensor_skew_s)
        ):
            return None
        return self._build_observation(
            state,
            depth,
            state_seq=state_seq,
            depth_seq=depth_seq,
            state_stamp_ns=state_stamp_ns,
            depth_stamp_ns=depth_stamp_ns,
            camera_info=camera_info,
        )

    def _get_observation_blocking(self) -> Dict[str, Any]:
        if self._reliable_v4_backend is not None:
            raise RuntimeError(
                "blocking latest ROS observation is disabled for reliable exact collection"
            )
        deadline = time.time() + self.config.obs_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            obs = self._try_get_observation()
            if obs is not None:
                return obs
            rospy.sleep(0.01)
        raise TimeoutError("observation timeout: no /xm/state or /xm/depth/image_raw")

    def _get_observation_after_token(
        self,
        previous_token: Tuple[int, int],
        require_new_state: bool = True,
        require_new_depth: bool = True,
    ) -> Dict[str, Any]:
        if self._reliable_v4_backend is not None:
            raise RuntimeError(
                "post-token latest ROS observation is disabled for reliable exact collection"
            )
        previous_state_seq, previous_depth_seq = [int(value) for value in previous_token]
        deadline = time.time() + self.config.obs_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            obs = self._try_get_observation()
            if obs is not None:
                token = obs["sensor_seq"]
                state_ok = (not require_new_state) or int(token["state"]) > previous_state_seq
                depth_ok = (not require_new_depth) or int(token["depth"]) > previous_depth_seq
                if state_ok and depth_ok:
                    return obs
            rospy.sleep(0.002)
        raise TimeoutError(
            "observation timeout waiting for fresh sensors: state>{} depth>{}".format(
                previous_state_seq if require_new_state else "any",
                previous_depth_seq if require_new_depth else "any",
            )
        )

    def _build_observation(
        self,
        state: XMState,
        depth_msg: Image,
        state_seq: int,
        depth_seq: int,
        state_stamp_ns: int,
        depth_stamp_ns: int,
        camera_info: Optional[CameraInfo],
    ) -> Dict[str, Any]:
        pos = np.asarray(
            [state.position.x, state.position.y, state.position.z], dtype=np.float32
        )
        vel = np.asarray(
            [state.velocity.x, state.velocity.y, state.velocity.z], dtype=np.float32
        )
        acc = np.asarray(
            [state.acceleration.x, state.acceleration.y, state.acceleration.z], dtype=np.float32
        )
        quat_xyzw = np.asarray(
            [
                state.orientation.x,
                state.orientation.y,
                state.orientation.z,
                state.orientation.w,
            ],
            dtype=np.float32,
        )
        yaw = _quat_to_yaw_xyzw(quat_xyzw)

        rel = self.goal - pos
        dist_xy = float(np.linalg.norm(rel[:2]))
        if dist_xy > 1e-6:
            direction_xy = rel[:2] / dist_xy
        else:
            direction_xy = np.zeros(2, dtype=np.float32)

        c = math.cos(-yaw)
        s = math.sin(-yaw)
        direction_body_xy = np.asarray(
            [c * direction_xy[0] - s * direction_xy[1], s * direction_xy[0] + c * direction_xy[1]],
            dtype=np.float32,
        )

        depth, depth_m = self._decode_depth(depth_msg)

        obs: Dict[str, Any] = {
            "depth": depth,
            "depth_m": depth_m,
            "depth_intrinsics": self._depth_intrinsics(depth_msg, camera_info),
            "state": {
                "position": pos,
                "velocity": vel,
                "acceleration": acc,
                "orientation_xyzw": quat_xyzw,
                "yaw": float(yaw),
                "z": float(pos[2]),
                "z_to_min": float(pos[2] - self.config.z_min),
                "z_to_max": float(self.config.z_max - pos[2]),
                "state_id": int(state.state_id),
                "sim_time_ns": int(state.sim_time_ns),
                "flags": int(state.flags),
            },
            "goal": {
                "position": self.goal.copy(),
                "relative": rel.astype(np.float32),
                "direction_xy": direction_xy.astype(np.float32),
                "direction_body_xy": direction_body_xy,
                "distance_xy": dist_xy,
                "distance_xy_norm40": float(dist_xy / 40.0),
                "dz": float(rel[2]),
            },
            "safety": {
                "min_clearance": float(state.min_clearance),
                "front_clearances": np.asarray(state.front_clearances, dtype=np.float32),
                "collided": bool(state.collided),
                "altitude_violation": bool(state.altitude_violation),
            },
            "prev_action_id": int(self.prev_action_id),
            "prev_action_onehot": self._prev_action_onehot(),
            "sensor_seq": {"state": int(state_seq), "depth": int(depth_seq)},
            "sensor_time": {
                "state_stamp_ns": int(state_stamp_ns),
                "depth_stamp_ns": int(depth_stamp_ns),
                "skew_ns": int(abs(int(state_stamp_ns) - int(depth_stamp_ns))),
            },
        }
        # Keep observations cheap and deployment-compatible: attach only the
        # altitude mask. Privileged global collision masks are computed
        # explicitly by get_action_mask() when requested.
        if self.mpl is not None:
            obs["action_mask"] = self.mpl.valid_action_mask(
                float(pos[2]),
                z_min=self.config.z_min,
                z_max=self.config.z_max,
                margin=self.config.action_mask_z_margin,
            ).astype(np.bool_)
        else:
            obs["action_mask"] = np.zeros((0,), dtype=np.bool_)
        return obs

    def _depth_intrinsics(self, depth_msg: Image, camera_info: Optional[CameraInfo]) -> Dict[str, float]:
        """Return intrinsics scaled to the policy/depth-mask resolution."""
        out_w = int(self.config.depth_out_width)
        out_h = int(self.config.depth_out_height)
        if camera_info is None or float(camera_info.K[0]) <= 0.0 or float(camera_info.K[4]) <= 0.0:
            return camera_intrinsics_from_fov(
                out_w, out_h, self.config.depth_camera_hfov_deg, self.config.depth_camera_vfov_deg
            )
        scale_x = (float(out_w) - 1.0) / max(1.0, float(depth_msg.width) - 1.0)
        scale_y = (float(out_h) - 1.0) / max(1.0, float(depth_msg.height) - 1.0)
        return {
            "fx": float(camera_info.K[0]) * scale_x,
            "fy": float(camera_info.K[4]) * scale_y,
            "cx": float(camera_info.K[2]) * scale_x,
            "cy": float(camera_info.K[5]) * scale_y,
        }

    def _build_reliable_v4_observation(self, endpoint: Any) -> Dict[str, Any]:
        """Decode one exact v4 snapshot into the existing observation shape."""
        if not hasattr(endpoint, "state") or not hasattr(endpoint, "depth"):
            raise PrimitiveExecutionContractError(
                "reliable v4 endpoint observation payload is incomplete"
            )
        try:
            values = messagepack_unpack(bytes(endpoint.state))
        except Exception as error:
            raise PrimitiveExecutionContractError(
                "reliable v4 state snapshot is not MessagePack: {}".format(error)
            ) from error
        if not isinstance(values, (list, tuple)) or len(values) < 17:
            raise PrimitiveExecutionContractError(
                "reliable v4 state snapshot has an invalid field count"
            )
        if int(values[1]) != int(endpoint.state_id) or int(values[2]) != int(endpoint.sim_time_ns):
            raise PrimitiveExecutionContractError(
                "reliable v4 state snapshot identity does not match ObservationRef"
            )
        if str(values[14]) != str(endpoint.episode_id) or str(values[15]) != str(endpoint.reset_id):
            raise PrimitiveExecutionContractError(
                "reliable v4 state snapshot episode/reset identity mismatch"
            )

        def _vector(value: Any, size: int) -> list[float]:
            if not isinstance(value, (list, tuple)) or len(value) != size:
                raise PrimitiveExecutionContractError(
                    "reliable v4 state vector has invalid size"
                )
            return [float(item) for item in value]

        position = _vector(values[5], 3)
        orientation = _vector(values[6], 4)
        velocity = _vector(values[7], 3)
        acceleration = _vector(values[8], 3)
        front_clearances = _vector(values[9], 3)
        state = SimpleNamespace(
            state_id=int(values[1]),
            sim_time_ns=int(values[2]),
            flags=int(values[3]),
            min_clearance=float(values[4]),
            position=SimpleNamespace(x=position[0], y=position[1], z=position[2]),
            orientation=SimpleNamespace(
                x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3]
            ),
            velocity=SimpleNamespace(x=velocity[0], y=velocity[1], z=velocity[2]),
            acceleration=SimpleNamespace(
                x=acceleration[0], y=acceleration[1], z=acceleration[2]
            ),
            front_clearances=front_clearances,
            collided=bool(int(values[3]) & 1),
            altitude_violation=bool(int(values[3]) & 2),
        )

        depth_bytes = bytes(endpoint.depth)
        output_width = int(self.config.depth_out_width)
        output_height = int(self.config.depth_out_height)
        capture_width = int(self.config.depth_capture_width)
        capture_height = int(self.config.depth_capture_height)
        output_size = output_width * output_height * 2
        capture_size = capture_width * capture_height * 2
        if len(depth_bytes) == output_size:
            width, height = output_width, output_height
        elif len(depth_bytes) == capture_size:
            width, height = capture_width, capture_height
        else:
            raise PrimitiveExecutionContractError(
                "reliable v4 depth snapshot size mismatch: received={} expected capture={} or output={}".format(
                    len(depth_bytes), capture_size, output_size
                )
            )
        depth_msg = SimpleNamespace(
            width=width,
            height=height,
            encoding="16UC1",
            is_bigendian=False,
            step=width * 2,
            data=depth_bytes,
        )
        observation = self._build_observation(
            state,
            depth_msg,
            state_seq=-1,
            depth_seq=-1,
            state_stamp_ns=int(endpoint.sim_time_ns),
            depth_stamp_ns=int(endpoint.sim_time_ns),
            camera_info=None,
        )
        observation["episode_id"] = str(endpoint.episode_id)
        observation["reset_id"] = str(endpoint.reset_id)
        observation.update(exact_endpoint_metadata())
        observation["telemetry_lookup_count"] = 0
        observation["endpoint_identity"] = {
            "runtime_instance_id": str(endpoint.runtime_instance_id),
            "episode_id": str(endpoint.episode_id),
            "reset_id": str(endpoint.reset_id),
            "state_id": int(endpoint.state_id),
            "depth_id": str(endpoint.depth_id),
            "sim_time_ns": int(endpoint.sim_time_ns),
            "execution_id": int(endpoint.execution_id),
            "snapshot_hash": (
                str(endpoint.snapshot_hash)
                if getattr(endpoint, "snapshot_hash", None) is not None
                else ""
            ),
        }
        return observation

    def _decode_depth(self, msg: Image) -> Tuple[np.ndarray, np.ndarray]:
        if msg.encoding not in ("16UC1", "mono16"):
            raise ValueError("unsupported depth encoding: {}".format(msg.encoding))

        width = int(msg.width)
        height = int(msg.height)
        packed_row_step = width * np.dtype(np.uint16).itemsize
        row_step = int(msg.step) if int(msg.step) > 0 else packed_row_step
        expected = row_step * height
        if width <= 0 or height <= 0 or row_step < packed_row_step:
            raise ValueError(
                "invalid depth layout: width={}, height={}, step={}".format(width, height, row_step)
            )
        if len(msg.data) < expected:
            raise ValueError(
                "depth payload too short: got {}, expected at least {}".format(len(msg.data), expected)
            )

        dtype = np.dtype(">u2" if bool(msg.is_bigendian) else "<u2")
        raw = np.ndarray(
            shape=(height, width), dtype=dtype, buffer=msg.data, strides=(row_step, dtype.itemsize)
        )
        depth_m = raw.astype(np.float32)
        depth_m *= float(self.config.depth_scale_m_per_unit)

        invalid = depth_m <= 0.0
        depth_m[invalid] = self.config.depth_sensor_max_m
        depth_m = np.clip(depth_m, self.config.depth_min_m, self.config.depth_sensor_max_m)

        # Nearest-neighbor downsample.
        out_h = int(self.config.depth_out_height)
        out_w = int(self.config.depth_out_width)
        if depth_m.shape != (out_h, out_w):
            ys = np.linspace(0, depth_m.shape[0] - 1, out_h).astype(np.int32)
            xs = np.linspace(0, depth_m.shape[1] - 1, out_w).astype(np.int32)
            depth_m = depth_m[ys][:, xs]

        policy_depth_m = np.clip(depth_m, self.config.depth_min_m, self.config.depth_max_m)
        norm = (policy_depth_m - self.config.depth_min_m) / max(
            1e-6, self.config.depth_max_m - self.config.depth_min_m
        )
        return np.clip(norm, 0.0, 1.0).astype(np.float32), depth_m.astype(np.float32)

    def _compute_reward_done(self, obs: Dict[str, Any], action_id: int) -> Tuple[float, bool, Dict[str, Any]]:
        dist_xy = float(obs["goal"]["distance_xy"])
        abs_goal_dz = abs(float(obs["goal"]["dz"]))
        if self.prev_distance_xy is None:
            progress = 0.0
        else:
            progress = float(self.prev_distance_xy - dist_xy)
        self.prev_distance_xy = dist_xy
        if self.prev_abs_goal_dz is None:
            z_progress = 0.0
        else:
            z_progress = float(self.prev_abs_goal_dz - abs_goal_dz)
        self.prev_abs_goal_dz = abs_goal_dz

        safety = obs["safety"]
        z = float(obs["state"]["z"])

        collided = bool(safety["collided"])
        altitude_violation = bool(
            safety["altitude_violation"] or z < self.config.z_min or z > self.config.z_max
        )
        extreme_altitude = bool(z < self.config.z_hard_min or z > self.config.z_hard_max)
        reached_goal = goal_reached(
            obs["state"]["position"],
            obs["goal"]["position"],
            radius_xy=self.config.goal_radius_xy,
            tolerance_z=self.config.goal_tolerance_z,
        )
        # Reaching the goal on the same sensor frame as a safety violation is
        # not a successful flight. This keeps collision/altitude penalties from
        # being offset by the success bonus in policy-learning targets.
        success = bool(reached_goal and not collided and not altitude_violation)
        timeout = bool(
            self.episode_step + 1 >= self.config.max_episode_steps
            and not collided
            and not altitude_violation
            and not success
        )
        far = bool(
            dist_xy > self.config.far_distance_xy
            and not collided
            and not altitude_violation
            and not success
            and not timeout
        )
        action_mask = obs.get("action_mask", None)
        dead_end = bool(
            action_mask is not None
            and len(action_mask) > 0
            and np.count_nonzero(action_mask) == 0
            and not collided
            and not altitude_violation
            and not success
            and not timeout
            and not far
        )

        min_clearance = float(safety["min_clearance"])
        clearance_penalty = max(0.0, self.config.clearance_margin_m - min_clearance)

        action_changed = self.prev_action_id >= 0 and int(action_id) != int(self.prev_action_id)

        reward = compute_reward(
            progress=progress,
            z_progress=z_progress,
            min_clearance=min_clearance,
            clearance_margin_m=self.config.clearance_margin_m,
            action_changed=action_changed,
            success=success,
            collided=collided,
            altitude_violation=altitude_violation,
            timeout=timeout,
            far=far,
            dead_end=dead_end,
            reward_progress_scale=self.config.reward_progress_scale,
            reward_goal_z_progress_scale=self.config.reward_goal_z_progress_scale,
            reward_step=self.config.reward_step,
            reward_clearance_scale=self.config.reward_clearance_scale,
            reward_action_change=self.config.reward_action_change,
            reward_success=self.config.reward_success,
            reward_collision=self.config.reward_collision,
            reward_altitude_violation=self.config.reward_altitude_violation,
            reward_timeout=self.config.reward_timeout,
            reward_far=self.config.reward_far,
            reward_dead_end=self.config.reward_dead_end,
        )

        done = bool(success or collided or altitude_violation or timeout or far or dead_end)
        done_reason = terminal_done_reason(
            success=success,
            collision=collided,
            dead_end=dead_end,
            timeout=timeout,
            far=far,
            hard_altitude=altitude_violation,
        )
        info = {
            "progress": progress,
            "goal_z_progress": z_progress,
            "distance_xy": dist_xy,
            "abs_goal_dz": abs_goal_dz,
            "success": success,
            "collided": collided,
            "altitude_violation": altitude_violation,
            "hard_altitude_violation": altitude_violation,
            "extreme_altitude_violation": extreme_altitude,
            "timeout": timeout,
            "far": far,
            "dead_end": dead_end,
            "done_reason": done_reason,
            "min_clearance": min_clearance,
            "z": z,
            "action_changed": action_changed,
            "episode_step": int(self.episode_step),
        }
        return float(reward), done, info

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _prev_action_onehot(self) -> np.ndarray:
        if self.action_space_n <= 0:
            return np.zeros((0,), dtype=np.float32)
        out = np.zeros((self.action_space_n,), dtype=np.float32)
        if 0 <= int(self.prev_action_id) < self.action_space_n:
            out[int(self.prev_action_id)] = 1.0
        return out

    def _make_twist(self, vx: float, vy: float, vz: float, yaw_rate: float) -> Twist:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
        msg.angular.z = float(yaw_rate)
        return msg
