"""Formal online BC-calibration replay production.

The AWAC learner owns Bellman/CQL mathematics and ``AWACReplayBuffer`` owns
the persistent ten-field storage contract.  This module owns only the
parent-side runtime orchestration needed to turn reliable-exact environment
steps into those existing replay transitions.  It is deliberately injectable
at the ``ParallelEnvPool`` and gate seams so the contract can be tested
without starting ROS, ZMQ, Unity, or Bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from planning.awac.calibration import CriticCalibrationConfig, evaluate_calibration_gate
from planning.awac.checkpoint import (
    build_calibration_exact_resume_state,
    restore_calibration_exact_resume_state,
)
from planning.awac.contract import TERMINAL_REASONS
from planning.awac.interaction import BehaviorSource
from planning.awac.phase1 import (
    CALIBRATION_GATE_FAIL_DIVERGED,
    CALIBRATION_GATE_PASS,
    CALIBRATION_GATE_PENDING,
    CALIBRATION_CERTIFICATION_BLOCKED_PENDING,
    Phase1CalibrationController,
    Phase1CalibrationSafetyCap,
)
from planning.common import file_sha256, read_csv
from planning.contracts.feature import (
    CONTINUOUS_DIM,
    INITIAL_PREV_ACTION,
    NUM_ACTIONS,
    POLICY_VECTOR_DIM,
    obs_continuous_vector,
    action_onehot,
)
from planning.contracts.observation import (
    EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    validate_reliable_exact_metadata,
)
from planning.contracts.task import DEFAULT_MAX_PRIMITIVE_STEPS
from planning.mission.spec import mission_id_from_row, row_start_goal, validate_mission_rows
from planning.awac.dev_selection import validate_training_mission_source


CALIBRATION_RUNTIME_SCHEMA_ID = "awac_formal_calibration_runtime_v1"
# Formal Critic calibration deliberately has a different policy contract from
# the deployment evaluator.  The evaluator remains deterministic argmax at
# T=0; calibration data and its independent holdout instead follow the same
# frozen-BC masked categorical distribution used by the Bellman expectation.
# Keep this identity here, beside the only runtime action-selection owner, so
# checkpoints and exact-resume state cannot merely label an old trajectory as
# compatible.
CALIBRATION_BEHAVIOR_POLICY_ID = "frozen_bc_masked_categorical_t1_v1"
CALIBRATION_BEHAVIOR_SELECTION_MODE = "masked_categorical"
CALIBRATION_BEHAVIOR_TEMPERATURE = 1.0
CALIBRATION_BEHAVIOR_RNG_OWNER = "calibration_producer_numpy_randomstate"
CALIBRATION_BEHAVIOR_RNG_SCOPE = "per_episode_per_worker"
CALIBRATION_BEHAVIOR_RNG_DERIVATION = (
    "sha256_base_seed_worker_episode_mission_uint32_v1"
)
_TERMINAL_REASON_ALIASES = {
    "altitude_violation": "hard_altitude",
    "max_steps": "timeout",
}
_TERMINAL_REASONS = frozenset(TERMINAL_REASONS)
_HOLDOUT_UNAVAILABLE_REASONS = frozenset(
    {"holdout_not_ready", "holdout_evaluator_not_configured"}
)


class CalibrationRuntimeError(RuntimeError):
    """A fail-closed runtime/observation error before replay commit."""


def calibration_behavior_policy_identity(
    *,
    checkpoint_sha256: str = "",
    rng_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the immutable behavior/target distribution for new calibration.

    ``rng_seed`` records the owner-visible seed only.  Exact resume persists
    the full ``RandomState`` state separately through
    ``build_calibration_exact_resume_state``; serialising it here would make a
    configuration identity depend on progress.
    """

    result: Dict[str, Any] = {
        "policy_id": CALIBRATION_BEHAVIOR_POLICY_ID,
        "selection_mode": CALIBRATION_BEHAVIOR_SELECTION_MODE,
        "temperature": float(CALIBRATION_BEHAVIOR_TEMPERATURE),
        "mask_contract": "depth_action_mask",
        "rng_owner": CALIBRATION_BEHAVIOR_RNG_OWNER,
        "rng_algorithm": "numpy.random.RandomState",
        "rng_scope": CALIBRATION_BEHAVIOR_RNG_SCOPE,
        "rng_derivation": CALIBRATION_BEHAVIOR_RNG_DERIVATION,
        "checkpoint_sha256": str(checkpoint_sha256),
    }
    if rng_seed is not None:
        result["rng_seed"] = int(rng_seed)
    return result


def calibration_episode_rng(
    *,
    base_seed: int,
    worker_id: int,
    episode_id: str,
    mission_id: str,
) -> Tuple[np.random.RandomState, Dict[str, Any]]:
    """Create one reproducible masked-categorical RNG stream per assignment.

    The producer RNG remains dedicated to replay-batch sampling. Behavior
    actions must not depend on how many Critic batches happened to be sampled
    before an episode, nor may concurrently active workers begin from
    identical random streams.
    """

    seed = int(base_seed)
    if seed < 0 or seed > 0xFFFFFFFF:
        raise ValueError("calibration behavior base seed is outside uint32")
    worker = int(worker_id)
    episode = str(episode_id).strip()
    mission = str(mission_id).strip()
    if not episode or not mission:
        raise ValueError("calibration episode RNG identity is incomplete")
    serialized = "{}\0{}\0{}\0{}".format(seed, worker, episode, mission)
    digest = hashlib.sha256(serialized.encode("utf-8")).digest()
    identity = {
        "rng_algorithm": "numpy.random.RandomState",
        "rng_scope": CALIBRATION_BEHAVIOR_RNG_SCOPE,
        "rng_derivation": CALIBRATION_BEHAVIOR_RNG_DERIVATION,
        "base_seed": seed,
        "worker_id": worker,
        "episode_id": episode,
        "mission_id": mission,
        "derived_seed": int.from_bytes(digest[:4], byteorder="big", signed=False),
        "derivation_sha256": digest.hex(),
    }
    return np.random.RandomState(identity["derived_seed"]), identity


def _is_holdout_unavailable_window(window: Mapping[str, Any]) -> bool:
    """Identify an availability placeholder, not a numerical measurement."""

    return bool(
        window.get("finite") is False
        and str(window.get("reason", "")) in _HOLDOUT_UNAVAILABLE_REASONS
    )


@dataclass(frozen=True)
class CalibrationMission:
    """One validated training mission in source-file order."""

    episode_id: str
    mission_id: str
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
    source_row: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _EpisodeState:
    mission: CalibrationMission
    source_index: int
    observation: Mapping[str, Any]
    action_mask: np.ndarray
    action_mask_info: Mapping[str, Any]
    previous_action: int = INITIAL_PREV_ACTION
    depth_history: List[np.ndarray] = field(default_factory=list)
    pending_transitions: List[Dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    policy_rng: Optional[np.random.RandomState] = None
    policy_rng_identity: Mapping[str, Any] = field(default_factory=dict)


def _hash_ordered(values: Iterable[str]) -> str:
    payload = json.dumps(
        [str(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_final_test_manifest(path: Path) -> bool:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file() or candidate.suffix.lower() != ".json":
        return False
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(payload, Mapping):
        return False
    role = str(payload.get("role", "")).strip().upper()
    return role in {"FINAL_TEST", "FINAL_TEST_ONLY"} or bool(
        payload.get("final_test_only", False)
    )


def _adjacent_final_test_manifest(source: Path) -> Optional[Path]:
    for candidate in (source.parent / "manifest.json", source.with_suffix(".manifest.json")):
        if _is_final_test_manifest(candidate):
            return candidate
    return None


def load_calibration_missions(
    index_path: Path,
    *,
    max_steps: int = DEFAULT_MAX_PRIMITIVE_STEPS,
    final_test_manifest: Optional[Path] = None,
) -> Tuple[CalibrationMission, ...]:
    """Load a V2 training mission source before any runtime is constructed.

    The source is read in CSV order.  It is never silently replaced by a
    replay report or by a final-test index.  Validation deliberately happens
    before the caller constructs ``ParallelEnvPool``.
    """

    source = Path(index_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("training mission source does not exist: {}".format(source))
    manifest = (
        Path(final_test_manifest).expanduser().resolve()
        if final_test_manifest is not None
        else _adjacent_final_test_manifest(source)
    )
    if manifest is not None:
        # The existing owner checks role-bearing manifests and recursively
        # rejects indexes named by a final-test artifact.
        validate_training_mission_source(source, final_test_manifest=manifest)
    rows = [dict(row) for row in read_csv(source)]
    if not rows:
        raise ValueError("training mission source is empty: {}".format(source))
    validate_mission_rows(rows, expected_max_primitive_steps=int(max_steps))

    missions: List[CalibrationMission] = []
    seen_episode_ids: Set[str] = set()
    seen_mission_ids: Set[str] = set()
    for row in rows:
        episode_id = str(row.get("episode_id", "")).strip()
        if not episode_id:
            raise ValueError("training mission source has an empty episode_id")
        mission_id = str(mission_id_from_row(row)).strip()
        if not mission_id:
            raise ValueError("training mission source has an empty mission_id")
        if episode_id in seen_episode_ids:
            raise ValueError("training mission source has duplicate episode_id: {}".format(episode_id))
        if mission_id in seen_mission_ids:
            raise ValueError("training mission source has duplicate mission_id: {}".format(mission_id))
        seen_episode_ids.add(episode_id)
        seen_mission_ids.add(mission_id)
        start, goal = row_start_goal(row)
        missions.append(
            CalibrationMission(
                episode_id=episode_id,
                mission_id=mission_id,
                start=tuple(float(value) for value in start),
                goal=tuple(float(value) for value in goal),
                source_row=row,
            )
        )
    return tuple(missions)


def calibration_mission_source_identity(
    index_path: Path, missions: Sequence[CalibrationMission]
) -> Dict[str, Any]:
    """Return stable identity for the exact mission source and ordering."""

    source = Path(index_path).expanduser().resolve()
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "row_count": int(len(missions)),
        "ordered_episode_ids_sha256": _hash_ordered(mission.episode_id for mission in missions),
        "ordered_mission_ids_sha256": _hash_ordered(mission.mission_id for mission in missions),
    }


def build_calibration_split(
    missions: Sequence[CalibrationMission],
    *,
    seed: int,
    holdout_fraction: float = 0.10,
) -> Dict[str, Any]:
    """Build a deterministic episode-level split for a source index.

    This is used only when the caller did not provide a persisted split.  The
    ranking is independent of Python/NumPy RNG implementation and the output
    ID lists retain source order for auditable dispatch.
    """

    rows = tuple(missions)
    if len(rows) < 2:
        raise ValueError("calibration split requires at least two missions")
    fraction = float(holdout_fraction)
    if not 0.0 < fraction < 1.0 or not math.isfinite(fraction):
        raise ValueError("holdout_fraction must be in (0,1)")
    holdout_count = max(1, int(round(len(rows) * fraction)))
    holdout_count = min(len(rows) - 1, holdout_count)
    ranked = sorted(
        rows,
        key=lambda mission: hashlib.sha256(
            "{}\0{}".format(int(seed), mission.episode_id).encode("utf-8")
        ).hexdigest(),
    )
    holdout_set = {mission.episode_id for mission in ranked[:holdout_count]}
    train_ids = [mission.episode_id for mission in rows if mission.episode_id not in holdout_set]
    holdout_ids = [mission.episode_id for mission in rows if mission.episode_id in holdout_set]
    return {
        "contract_id": "awac_calibration_episode_split_v1",
        "seed": int(seed),
        "eligible_mission_count": int(len(rows)),
        "eligible_ordered_episode_ids_sha256": _hash_ordered(
            mission.episode_id for mission in rows
        ),
        "selection_method": "sha256_seed_episode_id_then_source_order",
        "train_episode_ids": train_ids,
        "holdout_episode_ids": holdout_ids,
    }


def bc_calibration_action(
    *,
    model: Any,
    observation: Mapping[str, Any],
    previous_action: int,
    action_mask: Sequence[bool],
    normalizer: Any,
    torch: Any,
    device: Any,
    depth_history: Optional[Sequence[np.ndarray]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> int:
    """Sample frozen BC from the calibration masked categorical policy.

    This must not call ``choose_action``: that public evaluator correctly
    enforces deterministic argmax/T=0 for formal Dev100, which is not the
    behavior distribution required for calibration replay or its holdout.
    The producer owns the supplied ``RandomState`` and persists its state in
    the exact-resume checkpoint transaction.
    """

    if rng is None:
        raise CalibrationRuntimeError(
            "calibration masked-categorical action requires a producer RNG"
        )
    mask = np.asarray(action_mask, dtype=np.bool_).reshape(-1)
    if mask.shape != (NUM_ACTIONS,) or not bool(mask.any()):
        raise CalibrationRuntimeError("calibration action mask is invalid")

    # Keep the evaluator tensor construction import lazy: importing this
    # module must not import ROS/ZMQ in parent-side or CPU-only tests.
    from planning.evaluation.policy_evaluator import observation_tensors
    from planning.awac.model import masked_policy

    depth, vector = observation_tensors(
        dict(observation),
        int(previous_action),
        normalizer,
        torch,
        device,
        depth_history=depth_history,
    )
    mask_tensor = torch.from_numpy(mask.reshape(1, -1)).to(
        device=device, dtype=torch.bool
    )
    with torch.no_grad():
        logits = model(depth, vector)
        probabilities, _, _ = masked_policy(
            logits,
            mask_tensor,
            torch,
            temperature=float(CALIBRATION_BEHAVIOR_TEMPERATURE),
        )
    values = probabilities[0].detach().cpu().numpy().astype(np.float64, copy=False)
    valid_indices = np.flatnonzero(mask)
    valid_probabilities = values[valid_indices]
    if (
        valid_probabilities.shape != valid_indices.shape
        or not np.isfinite(valid_probabilities).all()
        or float(valid_probabilities.sum()) <= 0.0
    ):
        raise CalibrationRuntimeError("calibration BC policy probabilities are invalid")
    valid_probabilities = valid_probabilities / float(valid_probabilities.sum())
    return int(rng.choice(valid_indices, p=valid_probabilities))


def _depth_frame(observation: Mapping[str, Any]) -> np.ndarray:
    try:
        depth = np.asarray(observation["depth"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationRuntimeError("observation depth is missing or invalid") from exc
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise CalibrationRuntimeError("observation depth must be finite HxW")
    return np.ascontiguousarray(depth, dtype=np.float32)


def _action_mask(mask: Any, *, allow_empty: bool = False) -> np.ndarray:
    try:
        value = np.asarray(mask, dtype=np.bool_).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise CalibrationRuntimeError("action mask is invalid") from exc
    if value.size != NUM_ACTIONS:
        raise CalibrationRuntimeError(
            "action mask dimension {} != {}".format(value.size, NUM_ACTIONS)
        )
    if not bool(value.any()) and not bool(allow_empty):
        raise CalibrationRuntimeError("action mask has no valid action")
    return value


def _validate_observation(
    observation: Any,
    action_mask: Any,
    *,
    path: str,
    allow_empty_action_mask: bool = False,
) -> Tuple[Mapping[str, Any], np.ndarray]:
    if not isinstance(observation, Mapping):
        raise CalibrationRuntimeError("{} observation is not a mapping".format(path))
    try:
        validate_reliable_exact_metadata(observation, path=path)
        _depth_frame(observation)
        continuous = obs_continuous_vector(dict(observation))
    except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
        raise CalibrationRuntimeError(
            "{} failed reliable-exact observation validation: {}".format(path, exc)
        ) from exc
    if continuous.shape != (CONTINUOUS_DIM,) or not np.isfinite(continuous).all():
        raise CalibrationRuntimeError("{} continuous feature contract failed".format(path))
    return observation, _action_mask(
        action_mask,
        allow_empty=bool(allow_empty_action_mask),
    )


def _validate_runtime_info(info: Any, *, path: str) -> str:
    if info is None:
        raise CalibrationRuntimeError("{} runtime info is missing".format(path))
    if not isinstance(info, Mapping):
        raise CalibrationRuntimeError("{} runtime info is not a mapping".format(path))
    try:
        validate_reliable_exact_metadata(info, path=path)
    except (TypeError, ValueError) as exc:
        raise CalibrationRuntimeError(
            "{} failed exact runtime metadata validation: {}".format(path, exc)
        ) from exc
    for key in (
        "telemetry_lookup_count",
        "snapshot_missing_count",
        "frame_contract_failures",
        "state_depth_skew_max_ns",
    ):
        try:
            if int(info.get(key, 0)) != 0:
                raise CalibrationRuntimeError("{} has non-zero {}".format(path, key))
        except (TypeError, ValueError) as exc:
            raise CalibrationRuntimeError("{} has invalid {}".format(path, key)) from exc
    runtime_id = str(
        info.get("reliable_v4_runtime_instance_id", info.get("runtime_instance_id", ""))
    ).strip()
    endpoint = info.get("endpoint_identity")
    if not runtime_id and isinstance(endpoint, Mapping):
        runtime_id = str(endpoint.get("runtime_instance_id", "")).strip()
    if not runtime_id:
        raise CalibrationRuntimeError("{} endpoint runtime identity is missing".format(path))
    return runtime_id


def _episode_identity(observation: Mapping[str, Any], *, path: str) -> Tuple[str, str]:
    episode_id = str(observation.get("episode_id", "")).strip()
    reset_id = str(observation.get("reset_id", "")).strip()
    if not episode_id or not reset_id:
        raise CalibrationRuntimeError(
            "{} endpoint episode/reset identity is missing".format(path)
        )
    return episode_id, reset_id


def _validate_episode_continuity(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    info: Mapping[str, Any],
    *,
    path: str,
) -> None:
    previous_identity = _episode_identity(previous, path="{} previous".format(path))
    current_identity = _episode_identity(current, path="{} next".format(path))
    if previous_identity != current_identity:
        raise CalibrationRuntimeError(
            "{} endpoint episode/reset identity changed".format(path)
        )
    endpoint = info.get("endpoint_identity")
    if isinstance(endpoint, Mapping):
        for key, expected in zip(("episode_id", "reset_id"), current_identity):
            supplied = str(endpoint.get(key, "")).strip()
            if supplied and supplied != expected:
                raise CalibrationRuntimeError(
                    "{} endpoint {} identity mismatch".format(path, key)
                )


def _terminal_reason(info: Mapping[str, Any], done: bool) -> str:
    if not bool(done):
        return ""
    reason = str(info.get("done_reason", "")).strip().lower()
    reason = _TERMINAL_REASON_ALIASES.get(reason, reason)
    if reason not in _TERMINAL_REASONS:
        raise CalibrationRuntimeError(
            "terminal contract rejected done_reason={!r}".format(reason)
        )
    return reason


class CalibrationReplayProducer:
    """Collect reliable-exact BC behavior and update only the Critics."""

    def __init__(
        self,
        *,
        missions: Sequence[CalibrationMission],
        pool: Any,
        replay: Any,
        learner: Any,
        train_episode_ids: Optional[Iterable[str]] = None,
        holdout_episode_ids: Optional[Iterable[str]] = None,
        action_selector: Optional[Callable[..., int]] = None,
        gate_evaluator: Optional[Callable[..., Mapping[str, Any]]] = None,
        holdout_evaluator: Optional[Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]]] = None,
        on_checkpoint: Optional[Callable[["CalibrationReplayProducer", Mapping[str, Any]], None]] = None,
        calibration_config: Optional[CriticCalibrationConfig] = None,
        controller: Optional[Phase1CalibrationController] = None,
        torch: Any = None,
        normalizer: Any = None,
        device: Any = None,
        calibration_window_interval_steps: int = 32,
        min_replay_transitions: int = 128,
        min_completed_episodes: int = 16,
        min_critic_updates: int = 32,
        min_holdout_episodes: int = 8,
        min_stability_windows: int = 3,
        holdout_fraction: float = 0.10,
        batch_size: int = 128,
        learning_starts: int = 5000,
        updates_per_step: float = 0.50,
        max_transitions: int = 30000,
        max_episodes: int = 1000,
        ready_timeout_s: float = 30.0,
        reset_timeout_s: float = 10.0,
        step_timeout_s: float = 30.0,
        reset_settle: float = 0.30,
        rng: Optional[np.random.RandomState] = None,
        behavior_policy: Optional[Mapping[str, Any]] = None,
        mission_source_identity: Optional[Mapping[str, Any]] = None,
        expected_runtime_ids: Optional[Mapping[int, str]] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.missions = tuple(missions)
        if not self.missions:
            raise ValueError("calibration missions must not be empty")
        self.pool = pool
        self.replay = replay
        self.learner = learner
        worker_ids = tuple(int(value) for value in getattr(pool, "worker_ids", ()))
        if not worker_ids:
            raise ValueError("calibration runtime pool has no workers")
        self.worker_ids = worker_ids
        self.torch = torch
        self.normalizer = normalizer
        self.device = device if device is not None else getattr(learner, "device", None)
        self.action_selector = action_selector
        self.holdout_evaluator = holdout_evaluator
        self.on_checkpoint = on_checkpoint
        self.gate_evaluator = gate_evaluator or evaluate_calibration_gate
        self.calibration_config = calibration_config or CriticCalibrationConfig(
            min_replay_transitions=int(min_replay_transitions),
            min_completed_episodes=int(min_completed_episodes),
            min_critic_updates=int(min_critic_updates),
            min_holdout_episodes=int(min_holdout_episodes),
            min_stability_windows=int(min_stability_windows),
            holdout_fraction=float(holdout_fraction),
        )
        self.controller = controller or Phase1CalibrationController(
            safety_cap=Phase1CalibrationSafetyCap(
                max_transitions=int(max_transitions), max_episodes=int(max_episodes)
            )
        )
        self.calibration_window_interval_steps = int(calibration_window_interval_steps)
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.updates_per_step = float(updates_per_step)
        self.ready_timeout_s = float(ready_timeout_s)
        self.reset_timeout_s = float(reset_timeout_s)
        self.step_timeout_s = float(step_timeout_s)
        self.reset_settle = float(reset_settle)
        self.rng = rng or np.random.RandomState(55)
        supplied_behavior_policy = dict(behavior_policy or {})
        expected_behavior_policy = calibration_behavior_policy_identity(
            checkpoint_sha256=str(supplied_behavior_policy.get("checkpoint_sha256", "")),
            rng_seed=supplied_behavior_policy.get("rng_seed"),
        )
        if supplied_behavior_policy and supplied_behavior_policy != expected_behavior_policy:
            raise ValueError("calibration behavior policy identity is invalid")
        self.behavior_policy = expected_behavior_policy
        self._behavior_rng_seed = self.behavior_policy.get("rng_seed")
        if self._behavior_rng_seed is not None:
            self._behavior_rng_seed = int(self._behavior_rng_seed)
        self.mission_source_identity = dict(mission_source_identity or {})
        self.expected_runtime_ids = {
            int(key): str(value)
            for key, value in dict(expected_runtime_ids or {}).items()
        }
        self.output_dir = None if output_dir is None else Path(output_dir).expanduser().resolve()
        if self.calibration_window_interval_steps <= 0:
            raise ValueError("calibration window interval must be positive")
        if self.batch_size <= 0 or self.learning_starts < self.batch_size:
            raise ValueError("calibration batch/warmup schedule is invalid")
        if self.updates_per_step <= 0.0 or not math.isfinite(self.updates_per_step):
            raise ValueError("updates_per_step must be finite and positive")

        self.train_episode_ids = {
            str(value) for value in train_episode_ids
        } if train_episode_ids is not None else set()
        self.holdout_episode_ids = {
            str(value) for value in holdout_episode_ids
        } if holdout_episode_ids is not None else set()
        if self.train_episode_ids.intersection(self.holdout_episode_ids):
            raise ValueError("calibration train/holdout episode IDs overlap")
        self._validate_split_coverage()

        self._next_unassigned_index = 0
        self._completed_mission_ids: Set[str] = set()
        self._environment_step_count = 0
        self._completed_episode_count = 0
        self._gate_history: List[Dict[str, Any]] = []
        self._holdout_records: List[Dict[str, Any]] = []
        self._holdout_completed_ids: Set[str] = set()
        self._runtime_ids: Dict[int, str] = {}
        self._episode_rng_assignments: Dict[str, Dict[str, Any]] = {}
        self._active: Dict[int, _EpisodeState] = {}
        self._stop_requested = False
        self._stop_reason = ""
        self._actor_before = self._actor_fingerprint()
        self.metrics: Dict[str, Any] = {
            "runtime_failed_attempt_count": 0,
            "runtime_failure_reasons": [],
            "behavior_source": "BC_CALIBRATION",
            "behavior_policy": dict(self.behavior_policy),
            "behavior_rng_scope": CALIBRATION_BEHAVIOR_RNG_SCOPE,
            "behavior_rng_derivation": CALIBRATION_BEHAVIOR_RNG_DERIVATION,
            "reward_scale_applied_by": "learner_bellman_target",
            "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
            "replay_final_flush": "NOT_ATTEMPTED",
            "runtime_close": "NOT_ATTEMPTED",
        }

    @property
    def pool_closed(self) -> bool:
        return bool(getattr(self.pool, "closed", getattr(self.pool, "is_closed", False)))

    @property
    def progress(self) -> Dict[str, Any]:
        return self._progress_snapshot()

    def checkpoint_snapshot(self, status: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Return durable producer state at a gate/checkpoint boundary."""

        gate_status = dict(status or {})
        state = str(
            gate_status.get(
                "calibration_certification",
                gate_status.get("state", self.controller.certification),
            )
        )
        return {
            "status": state,
            "stop_reason": str(self._stop_reason),
            "environment_step_count": int(self._environment_step_count),
            "completed_episode_count": int(self._completed_episode_count),
            "replay_size": int(getattr(self.replay, "size", 0)),
            "critic_update_count": int(getattr(self.learner, "critic_update_count", 0)),
            "gate_history": [dict(value) for value in self._gate_history],
            "progress": self._progress_snapshot(),
            "runtime_identity": dict(self._runtime_ids),
            "actor_unchanged": self._actor_before == self._actor_fingerprint(),
            "actor_update_count": int(getattr(self.learner, "actor_update_count", 0)),
            "actor_optimizer_step_count": int(
                getattr(self.learner, "actor_optimizer_step_count", 0)
            ),
            "holdout_episode_count": int(len(self._holdout_completed_ids)),
            "holdout_transition_count": int(len(self._holdout_records)),
            "behavior_policy": dict(self.behavior_policy),
            "episode_rng_assignments": self._episode_rng_assignment_snapshot(),
            "latest_gate_status": gate_status,
            "exact_resume_state": build_calibration_exact_resume_state(
                raw_holdout_records=self._holdout_records,
                producer_rng=self.rng,
                torch=self.torch,
            ),
        }

    def _validate_split_coverage(self) -> None:
        if not self.train_episode_ids and not self.holdout_episode_ids:
            return
        known = self.train_episode_ids | self.holdout_episode_ids
        missing = [
            mission.episode_id
            for mission in self.missions
            if mission.episode_id not in known and mission.mission_id not in known
        ]
        if missing:
            raise ValueError(
                "calibration split does not cover training mission source: {}".format(
                    ",".join(missing[:5])
                )
            )

    def _mission_is_holdout(self, mission: CalibrationMission) -> bool:
        return mission.episode_id in self.holdout_episode_ids or mission.mission_id in self.holdout_episode_ids

    def _mission_is_train(self, mission: CalibrationMission) -> bool:
        if not self.train_episode_ids and not self.holdout_episode_ids:
            return True
        return mission.episode_id in self.train_episode_ids or mission.mission_id in self.train_episode_ids

    def _actor_fingerprint(self) -> str:
        actor = getattr(self.learner, "actor", None)
        state = actor.state_dict() if actor is not None else None
        if state is None:
            return ""
        fingerprint = getattr(self.learner, "_state_fingerprint", None)
        if callable(fingerprint):
            return str(fingerprint(state))
        digest = hashlib.sha256()
        for name in sorted(state):
            value = state[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().contiguous().numpy()
            array = np.ascontiguousarray(value)
            digest.update(str(name).encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
            digest.update(array.tobytes())
        return digest.hexdigest()

    def _episode_rng_assignment_snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {
            mission_id: dict(identity)
            for mission_id, identity in sorted(self._episode_rng_assignments.items())
        }

    def _episode_policy_rng(
        self, *, worker_id: int, mission: CalibrationMission
    ) -> Tuple[np.random.RandomState, Dict[str, Any]]:
        if self._behavior_rng_seed is None:
            raise CalibrationRuntimeError(
                "formal calibration behavior policy has no per-episode RNG seed"
            )
        return calibration_episode_rng(
            base_seed=int(self._behavior_rng_seed),
            worker_id=int(worker_id),
            episode_id=mission.episode_id,
            mission_id=mission.mission_id,
        )

    def restore_progress(
        self,
        progress: Mapping[str, Any],
        *,
        exact_resume_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Restore only a completed-boundary progress snapshot.

        In-flight episode buffers are never durable replay rows.  A resume
        therefore starts the first uncommitted mission from its reset, without
        duplicating a committed transition.
        """

        if not isinstance(progress, Mapping):
            raise ValueError("calibration progress must be a mapping")
        if progress.get("schema_id") != CALIBRATION_RUNTIME_SCHEMA_ID:
            raise ValueError("calibration progress schema mismatch")
        expected_source = progress.get("mission_source_identity")
        if self.mission_source_identity and expected_source != self.mission_source_identity:
            raise ValueError("calibration mission source identity mismatch")
        persisted_behavior_policy = progress.get("behavior_policy")
        if (
            persisted_behavior_policy is not None
            and dict(persisted_behavior_policy) != self.behavior_policy
        ):
            raise ValueError("calibration behavior policy identity mismatch")
        completed_values = progress.get("completed_mission_ids", [])
        if not isinstance(completed_values, (list, tuple)):
            raise ValueError("calibration progress completed missions are invalid")
        completed = {str(value) for value in completed_values}
        if len(completed) != len(completed_values):
            raise ValueError("calibration progress contains duplicate missions")
        known = {mission.mission_id for mission in self.missions}
        if not completed.issubset(known):
            raise ValueError("calibration progress contains an unknown mission")
        next_index = int(progress.get("next_mission_index", 0))
        if next_index < 0 or next_index > len(self.missions):
            raise ValueError("calibration progress next mission index is invalid")
        expected_next_index = len(self.missions)
        for index, mission in enumerate(self.missions):
            if mission.mission_id not in completed:
                expected_next_index = index
                break
        if next_index != expected_next_index:
            raise ValueError("calibration progress mission boundary is inconsistent")
        self._completed_mission_ids = completed
        self._next_unassigned_index = next_index
        self._environment_step_count = int(progress.get("environment_step_count", 0))
        self._completed_episode_count = int(progress.get("completed_episode_count", 0))
        if self._completed_episode_count != len(completed):
            raise ValueError("calibration progress episode count is inconsistent")
        self._gate_history = [dict(value) for value in progress.get("gate_history", [])]
        self._holdout_completed_ids = {
            str(value) for value in progress.get("holdout_completed_ids", [])
        }
        if not self._holdout_completed_ids.issubset(self.holdout_episode_ids):
            raise ValueError("calibration progress contains an unknown holdout mission")
        raw_assignments = progress.get("episode_rng_assignments", {})
        if raw_assignments is None:
            raw_assignments = {}
        if not isinstance(raw_assignments, Mapping):
            raise ValueError("calibration progress episode RNG assignments are invalid")
        mission_by_id = {mission.mission_id: mission for mission in self.missions}
        assignments: Dict[str, Dict[str, Any]] = {}
        for mission_id, raw_identity in raw_assignments.items():
            mission_key = str(mission_id)
            mission = mission_by_id.get(mission_key)
            if mission is None or not isinstance(raw_identity, Mapping):
                raise ValueError("calibration progress episode RNG assignment is invalid")
            if self._behavior_rng_seed is None:
                raise ValueError(
                    "calibration progress has RNG assignments without a behavior seed"
                )
            _, expected_identity = self._episode_policy_rng(
                worker_id=int(raw_identity.get("worker_id", -1)),
                mission=mission,
            )
            if dict(raw_identity) != expected_identity:
                raise ValueError("calibration progress episode RNG identity mismatch")
            assignments[mission_key] = dict(expected_identity)
        self._episode_rng_assignments = assignments
        if exact_resume_state is not None:
            restored = restore_calibration_exact_resume_state(
                exact_resume_state,
                producer_rng=self.rng,
                torch=self.torch,
            )
            records = list(restored["raw_holdout_records"])
            record_episode_ids = {
                str(record.get("episode_id", ""))
                for record in records
                if str(record.get("episode_id", ""))
            }
            if not record_episode_ids.issubset(self._holdout_completed_ids):
                raise ValueError(
                    "calibration exact resume holdout records exceed completed episodes"
                )
            self._holdout_records = records
        if self._environment_step_count < 0 or self._completed_episode_count < 0:
            raise ValueError("calibration progress counts are invalid")
        # Rebuild the controller's small finite state from the durable gate
        # history.  This does not replay environment transitions or optimizer
        # updates; it only restores the control decision already observed.
        for status in self._gate_history:
            state = str(
                status.get(
                    "calibration_gate_state", status.get("state", CALIBRATION_GATE_PENDING)
                )
            )
            if state == CALIBRATION_CERTIFICATION_BLOCKED_PENDING:
                state = CALIBRATION_GATE_PENDING
            self.controller.observe_gate(
                {"state": state},
                transitions=int(status.get("environment_step_count", 0)),
                episodes=int(status.get("episodes", 0)),
            )

    def _progress_snapshot(self) -> Dict[str, Any]:
        return {
            "schema_id": CALIBRATION_RUNTIME_SCHEMA_ID,
            "mission_source_identity": dict(self.mission_source_identity),
            "next_mission_index": int(self._durable_next_mission_index()),
            "completed_mission_ids": [
                mission.mission_id
                for mission in self.missions
                if mission.mission_id in self._completed_mission_ids
            ],
            "environment_step_count": int(self._environment_step_count),
            "completed_episode_count": int(self._completed_episode_count),
            "critic_update_count": int(getattr(self.learner, "critic_update_count", 0)),
            "gate_state": str(self.controller.certification),
            "gate_history": [dict(value) for value in self._gate_history],
            "holdout_completed_ids": sorted(self._holdout_completed_ids),
            "behavior_policy": dict(self.behavior_policy),
            "episode_rng_assignments": self._episode_rng_assignment_snapshot(),
        }

    def _durable_next_mission_index(self) -> int:
        for index, mission in enumerate(self.missions):
            if mission.mission_id not in self._completed_mission_ids:
                return index
        return len(self.missions)

    def _ready(self) -> None:
        try:
            result = self.pool.ready(
                worker_ready_timeout_s=self.ready_timeout_s,
                timeout_s=self.ready_timeout_s + 1.0,
            )
        except TypeError:
            result = self.pool.ready()
        if not isinstance(result, Mapping) or set(int(k) for k in result) != set(self.worker_ids):
            raise CalibrationRuntimeError("runtime pool did not ready all requested workers")

    def _assign_missions(self) -> None:
        if self._stop_requested:
            return
        episode_slots = (
            int(self.controller.safety_cap.max_episodes)
            - int(self._completed_episode_count)
            - len(self._active)
        )
        if episode_slots <= 0:
            return
        payloads: Dict[int, Dict[str, Any]] = {}
        assignments: Dict[int, Tuple[int, CalibrationMission]] = {}
        for worker_id in self.worker_ids:
            if worker_id in self._active or episode_slots <= 0:
                continue
            while self._next_unassigned_index < len(self.missions):
                index = self._next_unassigned_index
                mission = self.missions[index]
                self._next_unassigned_index += 1
                if mission.mission_id in self._completed_mission_ids:
                    continue
                assignments[worker_id] = (index, mission)
                episode_slots -= 1
                payloads[worker_id] = {
                    "start": list(mission.start),
                    "goal": list(mission.goal),
                    "settle": self.reset_settle,
                }
                break
        if not payloads:
            return
        try:
            responses = self.pool.reset(payloads, timeout_s=self.reset_timeout_s)
        except TypeError:
            responses = self.pool.reset(payloads)
        if not isinstance(responses, Mapping):
            raise CalibrationRuntimeError("runtime reset result is not a mapping")
        for worker_id, (source_index, mission) in assignments.items():
            response = responses.get(worker_id)
            if not isinstance(response, Mapping):
                raise CalibrationRuntimeError("worker {} reset response is missing".format(worker_id))
            observation, mask = _validate_observation(
                response.get("observation"), response.get("action_mask"),
                path="worker {} reset".format(worker_id),
            )
            _episode_identity(
                observation, path="worker {} reset".format(worker_id)
            )
            depth = _depth_frame(observation)
            policy_rng = None
            policy_rng_identity: Dict[str, Any] = {}
            if self._behavior_rng_seed is not None:
                policy_rng, policy_rng_identity = self._episode_policy_rng(
                    worker_id=int(worker_id), mission=mission
                )
                previous = self._episode_rng_assignments.get(mission.mission_id)
                if previous is not None and previous != policy_rng_identity:
                    raise CalibrationRuntimeError(
                        "calibration episode RNG assignment changed across resume"
                    )
                self._episode_rng_assignments[mission.mission_id] = dict(
                    policy_rng_identity
                )
            self._active[worker_id] = _EpisodeState(
                mission=mission,
                source_index=source_index,
                observation=observation,
                action_mask=mask,
                action_mask_info=dict(response.get("action_mask_info", {})),
                previous_action=int(INITIAL_PREV_ACTION),
                depth_history=[depth],
                policy_rng=policy_rng,
                policy_rng_identity=policy_rng_identity,
            )

    def _policy_action(self, state: _EpisodeState) -> int:
        if self.action_selector is not None:
            action = self.action_selector(
                observation=state.observation,
                previous_action=int(state.previous_action),
                action_mask=state.action_mask,
                depth_history=tuple(state.depth_history),
                learner=self.learner,
                normalizer=self.normalizer,
                torch=self.torch,
                device=self.device,
            )
        else:
            if self.normalizer is None or self.torch is None or self.device is None:
                raise CalibrationRuntimeError("formal BC action selector is not configured")
            if state.policy_rng is None:
                raise CalibrationRuntimeError(
                    "formal calibration action has no per-episode RNG assignment"
                )
            action = bc_calibration_action(
                model=self.learner.actor,
                observation=state.observation,
                previous_action=state.previous_action,
                action_mask=state.action_mask,
                normalizer=self.normalizer,
                torch=self.torch,
                device=self.device,
                depth_history=state.depth_history,
                rng=state.policy_rng,
            )
        try:
            value = int(action)
        except (TypeError, ValueError) as exc:
            raise CalibrationRuntimeError("BC action is not an integer") from exc
        if value < 0 or value >= NUM_ACTIONS or not bool(state.action_mask[value]):
            raise CalibrationRuntimeError("BC action is outside the current action mask")
        return value

    def _append_episode(self, state: _EpisodeState) -> None:
        if not state.pending_transitions:
            raise CalibrationRuntimeError("terminal episode has no transitions")
        add_batch = getattr(self.replay, "add_batch", None)
        if callable(add_batch):
            add_batch(state.pending_transitions)
        else:
            for transition in state.pending_transitions:
                self.replay.add(**transition)

    def _update_critics(self) -> None:
        replay_size = int(getattr(self.replay, "size", 0))
        if replay_size < self.batch_size or replay_size < self.learning_starts:
            return
        target_updates = int(
            math.floor(max(0, replay_size - self.learning_starts + 1) * self.updates_per_step)
        )
        current = int(getattr(self.learner, "critic_update_count", 0))
        while current < target_updates:
            sample = getattr(self.replay, "sample", None)
            if not callable(sample):
                raise CalibrationRuntimeError("replay has no sample transaction")
            batch = sample(
                self.batch_size,
                rng=self.rng,
                torch=self.torch,
                device=self.device,
            )
            self.learner.calibration_update(batch)
            current = int(getattr(self.learner, "critic_update_count", current + 1))

    def _policy_continuous_vector(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Build the normalized policy vector used by BC and the replay learner."""

        continuous = obs_continuous_vector(dict(observation))
        if self.normalizer is not None:
            transform = getattr(self.normalizer, "transform_continuous", None)
            if not callable(transform):
                raise CalibrationRuntimeError(
                    "calibration normalizer has no transform_continuous method"
                )
            try:
                continuous = transform(continuous)
            except (TypeError, ValueError, FloatingPointError) as exc:
                raise CalibrationRuntimeError(
                    "calibration normalizer rejected continuous features"
                ) from exc
        value = np.asarray(continuous, dtype=np.float32).reshape(-1)
        if value.shape != (CONTINUOUS_DIM,) or not np.isfinite(value).all():
            raise CalibrationRuntimeError(
                "normalized policy continuous feature contract failed"
            )
        return value

    def _holdout_window(self) -> Dict[str, Any]:
        if self.holdout_evaluator is None:
            return {
                "finite": False,
                "reason": "holdout_evaluator_not_configured",
            }
        if not self._holdout_records:
            return {
                "finite": False,
                "reason": "holdout_not_ready",
            }
        return dict(self.holdout_evaluator(tuple(self._holdout_records)))

    def _observe_gate(self, *, force: bool = False) -> Optional[Dict[str, Any]]:
        if not force and self._environment_step_count % self.calibration_window_interval_steps != 0:
            return None
        holdout_window = self._holdout_window()
        if (
            _is_holdout_unavailable_window(holdout_window)
            and self.gate_evaluator is evaluate_calibration_gate
        ):
            gate = {
                "state": CALIBRATION_GATE_PENDING,
                "reason": str(holdout_window.get("reason", "holdout_not_ready")),
            }
        else:
            gate = dict(
                self.gate_evaluator(
                    windows=self._gate_windows_for_evaluator(holdout_window),
                    replay_transitions=int(getattr(self.replay, "size", 0)),
                    completed_episodes=int(self._completed_episode_count),
                    critic_updates=int(getattr(self.learner, "critic_update_count", 0)),
                    holdout_episodes=int(len(self._holdout_completed_ids)),
                    config=self.calibration_config,
                )
            )
        status = self.controller.observe_gate(
            gate,
            transitions=int(self._environment_step_count),
            episodes=int(self._completed_episode_count),
        )
        status["window"] = dict(holdout_window)
        status["environment_step_count"] = int(self._environment_step_count)
        status["replay_transitions"] = int(getattr(self.replay, "size", 0))
        status["critic_updates"] = int(getattr(self.learner, "critic_update_count", 0))
        self._persist_gate_reason_fields(
            status=status,
            gate=gate,
            holdout_window=holdout_window,
        )
        self._gate_history.append(dict(status))
        if self.on_checkpoint is not None:
            self.on_checkpoint(self, status)
        certification = str(status.get("calibration_certification", status.get("state", "")))
        if certification == CALIBRATION_GATE_PASS:
            self._stop_requested = True
            self._stop_reason = "CALIBRATION_GATE_PASS"
        elif certification == CALIBRATION_GATE_FAIL_DIVERGED:
            self._stop_requested = True
            self._stop_reason = "CALIBRATION_DIVERGED"
        elif certification == CALIBRATION_CERTIFICATION_BLOCKED_PENDING:
            self._stop_requested = True
            self._stop_reason = "CALIBRATION_SAFETY_CAP_REACHED"
        return status

    def _maturity_ready(self) -> bool:
        return bool(
            int(getattr(self.replay, "size", 0))
            >= int(self.calibration_config.min_replay_transitions)
            and int(self._completed_episode_count)
            >= int(self.calibration_config.min_completed_episodes)
            and int(getattr(self.learner, "critic_update_count", 0))
            >= int(self.calibration_config.min_critic_updates)
            and int(len(self._holdout_completed_ids))
            >= int(self.calibration_config.min_holdout_episodes)
        )

    def _persist_gate_reason_fields(
        self,
        *,
        status: Dict[str, Any],
        gate: Mapping[str, Any],
        holdout_window: Mapping[str, Any],
    ) -> None:
        """Persist gate evidence needed to distinguish pending from divergence."""

        unavailable = _is_holdout_unavailable_window(holdout_window)
        status["gate_state"] = str(
            gate.get(
                "gate_state",
                gate.get("state", status.get("calibration_gate_state", "")),
            )
        )
        status["gate_reason"] = str(
            gate.get("gate_reason", gate.get("reason", ""))
        )
        status["maturity_ready"] = bool(
            gate.get("maturity_ready", self._maturity_ready())
        )
        status["replay_transitions"] = int(getattr(self.replay, "size", 0))
        status["completed_episodes"] = int(self._completed_episode_count)
        status["critic_updates"] = int(getattr(self.learner, "critic_update_count", 0))
        status["holdout_episodes"] = int(len(self._holdout_completed_ids))
        status["stability_windows"] = int(
            gate.get(
                "stability_windows",
                0
                if unavailable
                else len(self._gate_windows_for_evaluator(holdout_window)),
            )
        )
        status["hard_divergence_reason"] = str(
            gate.get("hard_divergence_reason", "")
        )
        # Persist the value-measurement contract alongside the legacy state so
        # a later reader can distinguish a historical calibration PASS from a
        # V2 MC-return certification without rewriting the old artifact.
        for field in (
            "calibration_gate_contract_id",
            "calibration_gate_contract_version",
            "holdout_return_semantics",
            "holdout_measurement_status",
            "value_policy_alignment",
            "holdout_q_return_rank_correlation",
            "q_mc_rank_correlation_threshold",
            "q_mc_rank_correlation_comparison",
            "evidence_status",
        ):
            if field in gate:
                status[field] = gate[field]
            elif field in holdout_window:
                status[field] = holdout_window[field]

    def _gate_windows_for_evaluator(self, current: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        if self.gate_evaluator is evaluate_calibration_gate:
            historical = []
            for value in self._gate_history:
                window = value.get("window", value)
                if isinstance(window, Mapping) and not _is_holdout_unavailable_window(
                    window
                ):
                    historical.append(dict(window))
            if _is_holdout_unavailable_window(current):
                return historical
            return historical + [dict(current)]
        if current.get("finite") is False:
            return [dict(current)] if self._gate_history else []
        return [
            dict(value.get("window", value))
            for value in self._gate_history
            if isinstance(value.get("window", value), Mapping)
        ] + [dict(current)]

    def _record_failure(self, error: BaseException) -> None:
        self.metrics["runtime_failed_attempt_count"] = int(
            self.metrics.get("runtime_failed_attempt_count", 0)
        ) + 1
        self.metrics.setdefault("runtime_failure_reasons", []).append(
            "{}: {}".format(type(error).__name__, str(error))
        )

    def _step_once(self) -> None:
        actions = {
            worker_id: self._policy_action(state)
            for worker_id, state in self._active.items()
        }
        try:
            results = self.pool.step(actions, timeout_s=self.step_timeout_s)
        except TypeError:
            results = self.pool.step(actions)
        if not isinstance(results, Mapping):
            raise CalibrationRuntimeError("runtime step result is not a mapping")
        runtime_ids_before = dict(self._runtime_ids)
        staged = []
        try:
            for worker_id, state in list(self._active.items()):
                result = results.get(worker_id)
                if not isinstance(result, Mapping):
                    raise CalibrationRuntimeError(
                        "worker {} step response is missing".format(worker_id)
                    )
                transition, next_observation, next_mask, reason = (
                    self._make_transition_for_worker(
                        worker_id, state, result, actions[worker_id]
                    )
                )
                action_mask_info = result.get("action_mask_info", {})
                if not isinstance(action_mask_info, Mapping):
                    raise CalibrationRuntimeError(
                        "worker {} action mask info is invalid".format(worker_id)
                    )
                staged.append(
                    (
                        worker_id,
                        state,
                        transition,
                        next_observation,
                        next_mask,
                        dict(action_mask_info),
                        reason,
                    )
                )
        except Exception:
            # A parallel response is one logical environment step.  Do not
            # retain an identity learned from a partially invalid response.
            self._runtime_ids = runtime_ids_before
            raise

        # Validate every worker response before mutating episode buffers or
        # committing a terminal episode.  A failure in one worker therefore
        # cannot publish a sibling worker's transition from the same wave.
        for (
            worker_id,
            state,
            transition,
            next_observation,
            next_mask,
            action_mask_info,
            reason,
        ) in staged:
            state.pending_transitions.append(transition)
            self._environment_step_count += 1
            state.step_count += 1
            if self._mission_is_holdout(state.mission):
                if bool(transition["done"]):
                    for episode_transition_index, pending in enumerate(
                        state.pending_transitions
                    ):
                        diagnostic = dict(pending)
                        diagnostic["terminal_reason"] = reason
                        diagnostic["episode_id"] = state.mission.episode_id
                        diagnostic["mission_id"] = state.mission.mission_id
                        diagnostic["episode_transition_index"] = int(
                            episode_transition_index
                        )
                        diagnostic["holdout_record_index"] = int(
                            len(self._holdout_records)
                        )
                        self._holdout_records.append(diagnostic)
                # Holdout transitions are never passed to Replay.
            if bool(transition["done"]):
                if self._mission_is_train(state.mission):
                    self._append_episode(state)
                elif not self._mission_is_holdout(state.mission):
                    raise CalibrationRuntimeError("mission is outside calibration split")
                self._completed_mission_ids.add(state.mission.mission_id)
                self._completed_episode_count += 1
                if self._mission_is_holdout(state.mission):
                    self._holdout_completed_ids.add(state.mission.episode_id)
                self._active.pop(worker_id, None)
            else:
                state.observation = next_observation
                state.action_mask = next_mask
                state.action_mask_info = action_mask_info
                state.previous_action = int(transition["action"])
                state.depth_history = [_depth_frame(next_observation)]
        self._update_critics()

    def _make_transition_for_worker(
        self,
        worker_id: int,
        state: _EpisodeState,
        result: Mapping[str, Any],
        action: int,
    ) -> Tuple[Dict[str, Any], Mapping[str, Any], np.ndarray, str]:
        info = result.get("info")
        runtime_id = _validate_runtime_info(info, path="worker {} runtime".format(worker_id))
        done = bool(result.get("done", False))
        reason = _terminal_reason(info, done)
        next_observation, next_mask = _validate_observation(
            result.get("observation"),
            result.get("action_mask"),
            path="worker {} step".format(worker_id),
            allow_empty_action_mask=done,
        )
        _validate_episode_continuity(
            state.observation,
            next_observation,
            info,
            path="worker {} step".format(worker_id),
        )
        previous_runtime_id = self._runtime_ids.get(int(worker_id))
        if previous_runtime_id is not None and previous_runtime_id != runtime_id:
            raise CalibrationRuntimeError("worker {} runtime identity changed".format(worker_id))
        expected_runtime_id = self.expected_runtime_ids.get(int(worker_id))
        if expected_runtime_id is not None and expected_runtime_id != runtime_id:
            raise CalibrationRuntimeError(
                "worker {} runtime identity differs from worker spec".format(worker_id)
            )
        self._runtime_ids[int(worker_id)] = runtime_id
        try:
            reward = float(result["reward"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationRuntimeError("runtime reward is missing or invalid") from exc
        if not math.isfinite(reward):
            raise CalibrationRuntimeError("runtime reward is non-finite")
        if not done and not bool(next_mask.any()):
            raise CalibrationRuntimeError("non-terminal next action mask is empty")
        depth = _depth_frame(state.observation)[None, ...]
        next_depth = _depth_frame(next_observation)[None, ...]
        vector = np.concatenate(
            (self._policy_continuous_vector(state.observation), action_onehot(state.previous_action)),
            axis=0,
        ).astype(np.float32)
        next_vector = np.concatenate(
            (self._policy_continuous_vector(next_observation), action_onehot(int(action))),
            axis=0,
        ).astype(np.float32)
        action = int(action)
        transition = {
            "depth": depth,
            "vector": vector,
            "action_mask": np.asarray(state.action_mask, dtype=np.bool_),
            "action": action,
            "reward": reward,
            "next_depth": next_depth,
            "next_vector": next_vector,
            "next_action_mask": np.asarray(next_mask, dtype=np.bool_),
            "done": done,
            "behavior_source": int(BehaviorSource.BC_CALIBRATION),
        }
        if vector.shape != (POLICY_VECTOR_DIM,) or next_vector.shape != (POLICY_VECTOR_DIM,):
            raise CalibrationRuntimeError("policy vector dimension contract failed")
        if not np.isfinite(vector).all() or not np.isfinite(next_vector).all():
            raise CalibrationRuntimeError("policy vector contains non-finite values")
        return transition, next_observation, next_mask, reason

    def run(self) -> Dict[str, Any]:
        """Run until PASS, safety-cap stop, divergence, or source exhaustion."""

        try:
            self._ready()
            freeze = getattr(self.learner, "freeze_actor_for_calibration", None)
            if callable(freeze):
                freeze()
            while True:
                if not self._stop_requested and self.controller.safety_cap.reached(
                    transitions=int(self._environment_step_count),
                    episodes=int(self._completed_episode_count),
                ):
                    self._stop_requested = True
                    self._stop_reason = "CALIBRATION_SAFETY_CAP_REACHED"
                    self._active.clear()
                    break
                self._assign_missions()
                if not self._active:
                    if self._stop_requested or self._next_unassigned_index >= len(self.missions):
                        self._stop_reason = self._stop_reason or "MISSION_SOURCE_EXHAUSTED"
                        break
                    continue
                # One parallel step contributes one transition per active
                # worker.  Do not cross the typed transition cap merely
                # because a whole worker wave was ready; the current in-flight
                # episodes remain uncommitted and are safely discarded.
                if (
                    self._environment_step_count + len(self._active)
                    > self.controller.safety_cap.max_transitions
                ):
                    self._stop_requested = True
                    self._stop_reason = "CALIBRATION_SAFETY_CAP_REACHED"
                    self._active.clear()
                    break
                try:
                    # Avoid starting a fresh mission after a PASS/DIVERGED
                    # decision, but safely finish already-reset episodes.
                    self._step_once()
                except Exception as error:
                    self._record_failure(error)
                    raise CalibrationRuntimeError(str(error)) from error
                if not self._stop_requested:
                    self._observe_gate()
                if self._stop_reason == "CALIBRATION_DIVERGED":
                    # Divergence is an immediate stop.  The current episode
                    # has not reached a legitimate terminal boundary, so its
                    # staged transitions must not become replay rows.
                    self._active.clear()
                cap_reached = self.controller.safety_cap.reached(
                    transitions=int(self._environment_step_count),
                    episodes=int(self._completed_episode_count),
                )
                if cap_reached and not self._stop_requested:
                    self._stop_requested = True
                    self._stop_reason = "CALIBRATION_SAFETY_CAP_REACHED"
                if self._stop_requested and not self._active:
                    break
                if self._stop_requested and cap_reached:
                    # Any uncommitted current episode is deliberately dropped;
                    # its transitions never entered the persistent replay.
                    self._active.clear()
                    break
            if self._stop_reason == "CALIBRATION_GATE_PASS":
                status = CALIBRATION_GATE_PASS
            elif self._stop_reason == "CALIBRATION_DIVERGED":
                status = CALIBRATION_GATE_FAIL_DIVERGED
            elif self._stop_reason == "CALIBRATION_SAFETY_CAP_REACHED":
                status = CALIBRATION_CERTIFICATION_BLOCKED_PENDING
            else:
                status = str(self.controller.certification or CALIBRATION_GATE_PENDING)
            actor_after = self._actor_fingerprint()
            result = {
                "status": status,
                "stop_reason": self._stop_reason,
                "environment_step_count": int(self._environment_step_count),
                "completed_episode_count": int(self._completed_episode_count),
                "next_mission_index": int(self._durable_next_mission_index()),
                "replay_size": int(getattr(self.replay, "size", 0)),
                "critic_update_count": int(getattr(self.learner, "critic_update_count", 0)),
                "holdout_transition_count": int(len(self._holdout_records)),
                "holdout_episode_count": int(len(self._holdout_completed_ids)),
                "gate_history": [dict(value) for value in self._gate_history],
                "gate_state": status,
                "progress": self._progress_snapshot(),
                "runtime_identity": dict(self._runtime_ids),
                "actor_state_before": self._actor_before,
                "actor_state_after": actor_after,
                "actor_unchanged": self._actor_before == actor_after,
                "actor_update_count": int(getattr(self.learner, "actor_update_count", 0)),
                "actor_optimizer_step_count": int(
                    getattr(self.learner, "actor_optimizer_step_count", 0)
                ),
                "pass_checkpoint_allowed": status == CALIBRATION_GATE_PASS,
                "metrics": dict(self.metrics),
            }
            return result
        except CalibrationRuntimeError:
            raise
        except Exception as error:
            self._record_failure(error)
            raise CalibrationRuntimeError(str(error)) from error
        finally:
            self._safe_stop_and_close()

    def _safe_stop_and_close(self) -> None:
        flush = getattr(self.replay, "flush", None)
        if callable(flush):
            try:
                flush()
                self.metrics["replay_final_flush"] = "PASS"
            except Exception:
                self.metrics["replay_final_flush"] = "FAIL"
        if self._stop_requested:
            stop = getattr(self.pool, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    pass
        close = getattr(self.pool, "close", None)
        if callable(close):
            try:
                close()
                self.metrics["runtime_close"] = "PASS"
            except Exception:
                self.metrics["runtime_close"] = "FAIL"


__all__ = [
    "CALIBRATION_BEHAVIOR_POLICY_ID",
    "CALIBRATION_BEHAVIOR_RNG_DERIVATION",
    "CALIBRATION_BEHAVIOR_RNG_OWNER",
    "CALIBRATION_BEHAVIOR_RNG_SCOPE",
    "CALIBRATION_BEHAVIOR_SELECTION_MODE",
    "CALIBRATION_BEHAVIOR_TEMPERATURE",
    "CALIBRATION_RUNTIME_SCHEMA_ID",
    "CalibrationMission",
    "CalibrationReplayProducer",
    "CalibrationRuntimeError",
    "bc_calibration_action",
    "build_calibration_split",
    "calibration_behavior_policy_identity",
    "calibration_episode_rng",
    "calibration_mission_source_identity",
    "load_calibration_missions",
]
