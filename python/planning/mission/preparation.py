"""Bounded, resumable mission preparation for the formal Pre-BC pipeline.

The preparation stage is deliberately an orchestration seam.  Sampling,
global routing, and Teacher auditing remain owned by their domain modules; this
module only enforces their order and the durable artifact boundaries:

    candidate -> one route plan -> route append -> immediate Teacher audit

Only bounded journal state is held in the process.  Final CSV artifacts are
compacted from the journals at the explicit publication boundary.
"""

from __future__ import annotations

import inspect
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np

from planning.common import (
    DEFAULT_SEED,
    file_sha256,
    write_csv_atomic,
    write_json_atomic,
)
from planning.common.config import pre_bc_value
from planning.data.csv_journal import CsvJournal
from planning.data.mission_routes import route_store_provenance
from planning.contracts.teacher_path import TEACHER_PATH_LENGTH_CONTRACT_ID
from planning.mission.sampling import (
    _mission_is_safe,
    iter_raw_mission_candidates,
    resolve_sampling_max_attempts,
)
from planning.mission.spec import DEFAULT_ALTITUDE_LEVELS_M
from planning.common.progress import (
    ProgressRateTracker,
    format_duration,
    format_progress,
    progress_metrics,
)


PREPARATION_CONTRACT_ID = "formal_teacher_mission_preparation_v1"
DEFAULT_PREPARATION_WORKERS = int(pre_bc_value("mission", "route_workers"))
MAX_INFLIGHT_RESULTS = DEFAULT_PREPARATION_WORKERS * 2
AUDIT_CONTRACT_ID = "ideal_kinematics_teacher_feasibility_path_lower_bound"
AUDIT_JOURNAL_FIELDS = (
    "candidate_number",
    "mission_id",
    "global_route_index",
    "result",
    "steps",
    "action_sequence",
    "final_distance_xy_m",
    "final_error_z_m",
    "route_length_m",
    "executed_path_length_m",
    "remaining_path_lower_bound_m",
    "straight_line_distance_m",
    "path_stretch",
    "path_length_contract_id",
    "plan_path_max_m",
)


def build_passing_mission_rows(
    audit_rows: Iterable[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    required_passing: int,
    route_provenance: Optional[Mapping[str, Any]] = None,
) -> list[Dict[str, Any]]:
    """Project successful audit rows onto the canonical mission artifact.

    This is a small publication projection, not a second audit implementation.
    Routing and Teacher feasibility remain owned by the preparation workers.
    """

    passing: list[Dict[str, Any]] = []
    for audit_row in audit_rows:
        if str(audit_row.get("result", "")) != "success":
            continue
        row = candidate_rows[int(audit_row["candidate_number"])]
        selected = dict(row)
        selected["episode_id"] = len(passing)
        selected["teacher_audit_contract_id"] = AUDIT_CONTRACT_ID
        selected["teacher_path_length_contract_id"] = TEACHER_PATH_LENGTH_CONTRACT_ID
        selected["teacher_plan_path_length_m"] = float(
            audit_row["executed_path_length_m"]
        )
        selected["teacher_plan_path_stretch"] = float(audit_row["path_stretch"])
        if route_provenance is not None:
            selected.update(dict(route_provenance))
        passing.append(selected)
        if len(passing) >= int(required_passing):
            break
    return passing


@dataclass(frozen=True)
class MissionPreparationConfig:
    """Boundaries for one bounded preparation run."""

    required_passing: int = 100_000
    max_candidates: int = 2_000_000
    max_sampling_attempts: int = 0
    max_route_stretch: float = 1.15
    seed: int = DEFAULT_SEED
    workers: int = DEFAULT_PREPARATION_WORKERS
    max_inflight_results: int = MAX_INFLIGHT_RESULTS
    checkpoint_interval: int = 1000
    collision_threads: int = 1
    progress_interval_sec: float = 10.0
    resume: bool = False

    def resolved_sampling_attempts(self) -> int:
        return resolve_sampling_max_attempts(
            max(1, int(self.max_candidates)), int(self.max_sampling_attempts)
        )

    def validate(self) -> None:
        if int(self.required_passing) <= 0:
            raise ValueError("required_passing must be positive")
        if int(self.max_candidates) <= 0:
            raise ValueError("max_candidates must be positive")
        if int(self.workers) <= 0:
            raise ValueError("workers must be positive")
        if int(self.max_inflight_results) <= 0:
            raise ValueError("max_inflight_results must be positive")
        if int(self.checkpoint_interval) <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if int(self.collision_threads) <= 0:
            raise ValueError("collision_threads must be positive")
        if float(self.progress_interval_sec) <= 0.0:
            raise ValueError("progress_interval_sec must be positive")
        if float(self.max_route_stretch) < 1.0:
            raise ValueError("max_route_stretch must be >= 1")


@dataclass(frozen=True)
class MissionPreparationResult:
    candidate_count: int
    routed_count: int
    passing_count: int
    rejected_route_count: int
    sampling_attempts: int
    generation_astar_calls: int
    audit_astar_calls: int = 0
    collection_astar_calls: int = 0
    relabel_astar_calls: int = 0
    resumed_candidate_count: int = 0
    replanned_committed_route_count: int = 0
    candidate_sha256: str = ""
    mission_sha256: str = ""


def _goal_from_candidate(candidate: Dict[str, Any]) -> Optional[Sequence[float]]:
    if isinstance(candidate.get("goal"), (list, tuple, np.ndarray)):
        return [float(value) for value in candidate["goal"][:3]]
    keys = ("goal_x", "goal_y", "goal_z")
    if all(key in candidate for key in keys):
        return [float(candidate[key]) for key in keys]
    return None


def _route_result(value: Any) -> Tuple[str, float, float, Optional[np.ndarray]]:
    if value is None:
        return "no_route", 0.0, 0.0, None
    if isinstance(value, tuple) and len(value) == 4 and isinstance(value[0], str):
        status, length, stretch, route = value
        return (
            str(status),
            float(length),
            float(stretch),
            None if route is None else np.asarray(route, dtype=np.float32),
        )
    route = np.asarray(value, dtype=np.float32)
    if route.ndim != 2 or route.shape[0] < 2 or route.shape[1] != 3:
        raise ValueError("route planner must return a finite Nx3 route")
    length = float(np.sum(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)))
    return "ok", length, 0.0, route


def _invoke_auditor(auditor: Callable, candidate: Dict[str, Any], route: np.ndarray) -> Dict[str, Any]:
    """Call either the two-argument preparation adapter or a keyword adapter."""

    result = auditor(candidate, route)
    if isinstance(result, dict):
        return dict(result)
    if isinstance(result, bool):
        return {"result": "success" if result else "dead_end"}
    return {"result": str(result)}


def _normalise_audit_journal_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Keep rejection and success records on one recoverable journal schema."""

    return {field: row.get(field, "") for field in AUDIT_JOURNAL_FIELDS}


def _teacher_path_contract_fields(audit: Mapping[str, Any]) -> Dict[str, Any]:
    """Project internal audit names onto the published collector contract."""

    contract_id = str(
        audit.get(
            "teacher_path_length_contract_id",
            audit.get("path_length_contract_id", ""),
        )
    ).strip()
    path_length = audit.get(
        "teacher_plan_path_length_m",
        audit.get("executed_path_length_m"),
    )
    path_stretch = audit.get(
        "teacher_plan_path_stretch",
        audit.get("path_stretch"),
    )
    if not contract_id or path_length is None or path_stretch is None:
        return {}
    fields: Dict[str, Any] = {
        "teacher_path_length_contract_id": contract_id,
        "teacher_plan_path_length_m": float(path_length),
        "teacher_plan_path_stretch": float(path_stretch),
    }
    teacher_audit_contract_id = str(
        audit.get("teacher_audit_contract_id", "")
    ).strip()
    if teacher_audit_contract_id:
        fields["teacher_audit_contract_id"] = teacher_audit_contract_id
    return fields


_PREPARATION_ROUTE_CONTEXT = None
_PREPARATION_ROUTE_PLANNER = None
_PREPARATION_WORKER_STATE = None
_PREPARATION_WORKER_INIT_COUNT = 0


class _SingleRouteView:
    """Minimal route-store view for worker-local Teacher audit reuse."""

    def __init__(self, route: np.ndarray):
        self.route = np.asarray(route, dtype=np.float32)

    def validate(self, _index: int, goal=None) -> np.ndarray:
        if goal is not None and not np.allclose(
            self.route[-1], np.asarray(goal, dtype=np.float32), atol=1.0e-3
        ):
            raise ValueError("route endpoint mismatch")
        return self.route


def _prepare_preparation_worker(payload: Mapping[str, Any]) -> None:
    """Initialize all expensive preparation resources once in one process."""

    global _PREPARATION_ROUTE_CONTEXT
    global _PREPARATION_ROUTE_PLANNER
    global _PREPARATION_WORKER_STATE
    global _PREPARATION_WORKER_INIT_COUNT

    from planning.contracts.teacher_path import primitive_reference_path_lengths_m
    from planning.mission.audit_worker import _audit_mission_row_with_state
    from planning.mission.global_route import GlobalRouteConfig
    from planning.native.geometry import NativeGeometryContext
    from planning.primitives.library import MotionPrimitiveLibrary
    from planning.teacher.policy import ReachabilityTeacher, TeacherConfig

    if _PREPARATION_WORKER_STATE is not None:
        previous_context = _PREPARATION_WORKER_STATE.get("geometry_context")
        if previous_context is not None:
            previous_context.close()

    os.environ["PLANNING_COLLISION_BACKEND"] = str(
        payload.get("collision_backend", "cpp_cpu")
    )
    os.environ["PLANNING_COLLISION_THREADS"] = str(
        int(payload.get("collision_threads", 1))
    )
    route_config = GlobalRouteConfig(
        resolution_m=float(payload["route_resolution_m"]),
        flight_z_min_m=float(payload["flight_z_min_m"]),
        flight_z_max_m=float(payload["flight_z_max_m"]),
        lookahead_m=float(payload["route_lookahead_m"]),
        tracking_margin_m=float(payload["route_tracking_margin_m"]),
        nearest_free_radius_cells=int(payload["nearest_free_radius_cells"]),
    )
    geometry_context = NativeGeometryContext.from_voxel_cache(
        Path(str(payload["cache_path"])),
        voxel_size=float(payload["voxel_size"]),
        route_config=route_config,
    )
    mpl = MotionPrimitiveLibrary()
    checker = geometry_context.collision_checker(float(payload["inflate_radius"]))
    planner = geometry_context.global_route_planner(checker, route_config)
    teacher_config = TeacherConfig(**dict(payload["teacher_config"]))
    teacher = ReachabilityTeacher(mpl, checker, teacher_config)
    _PREPARATION_WORKER_STATE = {
        "geometry_context": geometry_context,
        "mpl": mpl,
        "checker": checker,
        "route_planner": planner,
        "teacher": teacher,
        "max_steps": int(payload["max_steps"]),
        "max_route_stretch": float(payload["max_route_stretch"]),
        "start_x_range": tuple(float(value) for value in payload["start_x_range"]),
        "start_y_range": tuple(float(value) for value in payload["start_y_range"]),
        "goal_distance": float(payload["goal_distance"]),
        "goal_y_offset_range": tuple(
            float(value) for value in payload["goal_y_offset_range"]
        ),
        "altitude_levels": tuple(float(value) for value in payload["altitude_levels"]),
        "min_start_valid_actions": int(payload["min_start_valid_actions"]),
        "min_goal_valid_actions": int(payload["min_goal_valid_actions"]),
        "map_margin_m": float(payload["map_margin_m"]),
        "check_step": int(payload["check_step"]),
        "action_path_lengths_m": primitive_reference_path_lengths_m(mpl),
        "audit": _audit_mission_row_with_state,
    }
    _PREPARATION_WORKER_INIT_COUNT += 1
    _PREPARATION_ROUTE_CONTEXT = geometry_context
    _PREPARATION_ROUTE_PLANNER = planner


def _prepare_route_worker(payload: Mapping[str, Any]) -> None:
    """Compatibility alias for the complete preparation initializer."""

    _prepare_preparation_worker(payload)


def _evaluate_preparation_candidate(task):
    """Run safety, one route plan, route gates, and Teacher audit in a worker."""

    if _PREPARATION_WORKER_STATE is None:
        raise RuntimeError("preparation worker is not initialized")
    attempt, raw_candidate = task
    state = _PREPARATION_WORKER_STATE
    mission = dict(raw_candidate)
    mission["sample_attempts"] = int(attempt)
    if not _mission_is_safe(
        mission,
        state["mpl"],
        state["checker"],
        state["min_start_valid_actions"],
        state["min_goal_valid_actions"],
        state["map_margin_m"],
        state["check_step"],
    ):
        return {
            "safe": False,
            "sampling_attempts": int(attempt),
        }

    candidate = _formal_candidate_row(
        mission,
        state["max_steps"],
        route_resolution_m=float(state["teacher"].config.global_route_resolution_m),
        route_tracking_margin_m=float(
            state["teacher"].config.global_route_tracking_margin_m
        ),
    )
    route_value = _route_candidate_with_planner(
        candidate, state["route_planner"]
    )
    route_status, route_length, route_stretch, route = _route_result(route_value)
    result = {
        "safe": True,
        "candidate": candidate,
        "route_value": (route_status, route_length, route_stretch, route),
        "sampling_attempts": int(attempt),
    }
    if route_status != "ok" or route is None:
        return result
    if route_stretch and route_stretch > state["max_route_stretch"]:
        return result

    audit_candidate = dict(candidate)
    audit_candidate["candidate_number"] = int(attempt)
    # The historical parent adapter supplied a synthetic route-store index
    # while validating the already-computed route.  Preserve that audit input
    # contract without causing another route lookup or A* invocation.
    audit_candidate["global_route_index"] = 0
    audit = state["audit"](
        (int(attempt), audit_candidate),
        state["teacher"],
        state["mpl"],
        state["max_steps"],
        state["action_path_lengths_m"],
        _SingleRouteView(route),
    )
    audit = dict(audit)
    audit.pop("candidate_number", None)
    audit.pop("global_route_index", None)
    result["audit"] = audit
    return result


def _route_candidate_with_planner(candidate: Mapping[str, Any], planner: Any):
    start = [candidate["start_x"], candidate["start_y"], candidate["start_z"]]
    goal = [candidate["goal_x"], candidate["goal_y"], candidate["goal_z"]]
    try:
        route = planner.plan(start, goal)
    except RuntimeError:
        return "no_route", 0.0, 0.0, None
    route = np.asarray(route, dtype=np.float32)
    route_length = float(np.sum(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)))
    direct = float(np.linalg.norm(np.asarray(goal[:2]) - np.asarray(start[:2])))
    return "ok", route_length, route_length / max(1.0e-6, direct), route


def _preparation_route_worker(candidate: Dict[str, Any]):
    if _PREPARATION_ROUTE_PLANNER is None:
        raise RuntimeError("preparation route worker is not initialized")
    return _route_candidate_with_planner(candidate, _PREPARATION_ROUTE_PLANNER)


class _BoundedPreparationRouteResults:
    """Bounded, input-ordered complete preparation results."""

    def __init__(
        self,
        *,
        candidates: Iterator[Dict[str, Any]],
        start_candidate_number: int,
        start_sampling_attempts: int,
        max_candidates: int,
        max_sampling_attempts: int,
        workers: int,
        max_inflight_results: int,
        route_payload: Mapping[str, Any],
        max_accepted: int,
        worker: Optional[Callable] = None,
        initializer: Optional[Callable] = None,
    ) -> None:
        if int(workers) <= 1:
            raise ValueError("bounded route results require workers > 1")
        self._candidates = iter(candidates)
        self._next_submission_attempt = int(start_sampling_attempts) + 1
        self._next_result_attempt = int(start_sampling_attempts) + 1
        self._max_sampling_attempts = int(max_sampling_attempts)
        self._max_candidates = int(max_candidates)
        self._safe_candidates_seen = int(start_candidate_number)
        self._max_accepted = int(max_accepted)
        self._max_inflight_results = int(max_inflight_results)
        self._worker = worker or _evaluate_preparation_candidate
        self._initializer = initializer
        self._pending: Dict[int, Tuple[Future, Dict[str, Any]]] = {}
        self._exhausted = False
        self._closed = False
        context = mp.get_context("spawn")
        initargs = () if initializer is None else (dict(route_payload),)
        self._executor = ProcessPoolExecutor(
            max_workers=min(int(workers), self._max_inflight_results),
            mp_context=context,
            initializer=initializer,
            initargs=initargs,
        )

    def _fill(self) -> None:
        while (
            not self._exhausted
            and len(self._pending) < self._max_inflight_results
            and self._next_submission_attempt <= self._max_sampling_attempts
            and self._safe_candidates_seen < self._max_accepted
        ):
            try:
                candidate = dict(next(self._candidates))
            except StopIteration:
                self._exhausted = True
                break
            attempt = int(self._next_submission_attempt)
            self._next_submission_attempt += 1
            self._pending[attempt] = (
                self._executor.submit(self._worker, (attempt, candidate)),
                candidate,
            )

    def __iter__(self):
        self._fill()
        try:
            while self._pending:
                attempt = int(self._next_result_attempt)
                future, candidate = self._pending.pop(attempt)
                result = future.result()
                if not isinstance(result, Mapping):
                    raise TypeError("preparation worker returned a non-mapping result")
                if bool(result.get("safe", False)):
                    self._safe_candidates_seen += 1
                yield attempt, candidate, dict(result)
                self._next_result_attempt += 1
                self._fill()
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Results submitted ahead of the required passing boundary are
        # speculative, but they still need to be drained before closing the
        # Python 3.8 process-pool IPC handles.  The window is bounded, so this
        # cleanup cannot grow with the formal candidate count.
        for future, _candidate in self._pending.values():
            try:
                future.result()
            except Exception:
                pass
        self._pending.clear()
        self._executor.shutdown(wait=True)

    @property
    def pending_count(self) -> int:
        """Current bounded speculative/reorder occupancy."""

        return int(len(self._pending))


def _compact_with_extra_fields(
    journal: CsvJournal,
    output_path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> int:
    """Compact a streaming journal once, enriching rows with final identity."""

    journal.flush()
    base_fields = list(journal.fieldnames)
    extra_fields = list((extra or {}).keys())
    fields = base_fields + [field for field in extra_fields if field not in base_fields]
    count = 0

    def enriched_rows() -> Iterator[Dict[str, Any]]:
        nonlocal count
        for value in journal.iter_rows():
            row = dict(value)
            row.update(extra or {})
            count += 1
            yield row

    write_csv_atomic(Path(output_path), enriched_rows(), fieldnames=fields)
    return count


class MissionPreparationRun:
    """Execute one streaming preparation run with explicit artifact seams."""

    def __init__(
        self,
        *,
        candidates: Iterable[Dict[str, Any]],
        route_planner: Callable[[Dict[str, Any]], Any],
        mission_auditor: Callable[[Dict[str, Any], np.ndarray], Any],
        route_store_writer: Any,
        candidate_output: Path,
        mission_output: Path,
        config: MissionPreparationConfig,
        candidate_journal_path: Optional[Path] = None,
        audit_journal_path: Optional[Path] = None,
        mission_journal_path: Optional[Path] = None,
        progress_path: Optional[Path] = None,
        parallel_route_payload: Optional[Mapping[str, Any]] = None,
        parallel_worker: Optional[Callable] = None,
        parallel_worker_initializer: Optional[Callable] = None,
    ) -> None:
        config.validate()
        self.candidates = iter(candidates)
        self.route_planner = route_planner
        self.mission_auditor = mission_auditor
        self.route_store_writer = route_store_writer
        self.candidate_output = Path(candidate_output).expanduser().resolve()
        self.mission_output = Path(mission_output).expanduser().resolve()
        self.config = config
        self.parallel_route_payload = (
            None
            if parallel_route_payload is None
            else dict(parallel_route_payload)
        )
        self.parallel_worker = parallel_worker
        self.parallel_worker_initializer = parallel_worker_initializer
        self._resume_sampling_attempts = 0
        self.candidate_journal_path = Path(
            candidate_journal_path
            or self.candidate_output.with_suffix(".candidate.journal.jsonl")
        ).expanduser().resolve()
        self.audit_journal_path = Path(
            audit_journal_path
            or self.mission_output.with_suffix(".audit.journal.jsonl")
        ).expanduser().resolve()
        self.mission_journal_path = Path(
            mission_journal_path
            or self.mission_output.with_suffix(".passing.journal.jsonl")
        ).expanduser().resolve()
        self.progress_path = Path(
            progress_path
            or self.mission_output.with_suffix(".preparation.progress.json")
        ).expanduser().resolve()

    def _journals(self) -> Tuple[CsvJournal, CsvJournal, CsvJournal, CsvJournal]:
        # Streaming mode is important here: the final 2M candidate history is
        # on disk, while the engine retains only counters and the current row.
        return (
            CsvJournal(
                self.candidate_journal_path,
                key_field="candidate_number",
                retain_rows=False,
            ),
            CsvJournal(
                self.audit_journal_path,
                key_field="candidate_number",
                retain_rows=False,
            ),
            CsvJournal(
                self.candidate_output.with_suffix(".routed.candidate.journal.jsonl"),
                key_field="candidate_number",
                retain_rows=False,
            ),
            CsvJournal(
                self.mission_journal_path,
                key_field="candidate_number",
                retain_rows=False,
            ),
        )

    @staticmethod
    def _existing_count(journal: CsvJournal) -> int:
        return int(journal.row_count)

    def _skip_candidates(self, count: int) -> None:
        for _ in range(int(count)):
            try:
                next(self.candidates)
            except StopIteration as error:
                raise RuntimeError(
                    "resume journal has more candidates than input iterator"
                ) from error

    def _write_progress(
        self,
        *,
        status: str,
        candidate_count: int,
        routed_count: int,
        passing_count: int,
        sampling_attempts: int,
        started: float,
        rejected_route_count: int = 0,
        inflight_results: int = 0,
    ) -> None:
        elapsed_s = max(0.0, time.monotonic() - started)
        estimated_stop = int(self.config.resolved_sampling_attempts())
        trackers = getattr(self, "_progress_trackers", {})
        attempt_rate = trackers.get("attempts")
        candidate_rate = trackers.get("candidates")
        audit_rate = trackers.get("audits")
        passing_rate = trackers.get("passing")
        attempt_snapshot = (
            attempt_rate.update(int(sampling_attempts))
            if attempt_rate is not None
            else {}
        )
        candidate_snapshot = (
            candidate_rate.update(int(candidate_count))
            if candidate_rate is not None
            else {}
        )
        audit_snapshot = (
            audit_rate.update(int(routed_count)) if audit_rate is not None else {}
        )
        passing_snapshot = (
            passing_rate.update(int(passing_count))
            if passing_rate is not None
            else {}
        )
        remaining_passing = max(
            0, int(self.config.required_passing) - int(passing_count)
        )
        candidate_pass_rate = (
            float(passing_count) / float(candidate_count)
            if int(candidate_count) > 0
            else 0.0
        )
        attempt_pass_rate = (
            float(passing_count) / float(sampling_attempts)
            if int(sampling_attempts) > 0
            else 0.0
        )
        if remaining_passing == 0:
            estimated_candidates_required = int(candidate_count)
            estimated_attempts_required = int(sampling_attempts)
            target_at_risk = "NO"
            attempt_budget_at_risk = "NO"
        else:
            estimated_candidates_required = (
                int(candidate_count)
                + int(math.ceil(remaining_passing / candidate_pass_rate))
                if candidate_pass_rate > 0.0
                else None
            )
            estimated_attempts_required = (
                int(sampling_attempts)
                + int(math.ceil(remaining_passing / attempt_pass_rate))
                if attempt_pass_rate > 0.0
                else None
            )
            target_at_risk = (
                "YES"
                if estimated_candidates_required is not None
                and estimated_candidates_required > int(self.config.max_candidates)
                else ("NO" if estimated_candidates_required is not None else "UNKNOWN")
            )
            attempt_budget_at_risk = (
                "YES"
                if estimated_attempts_required is not None
                and estimated_attempts_required
                > int(self.config.resolved_sampling_attempts())
                else (
                    "NO"
                    if estimated_attempts_required is not None
                    else "UNKNOWN"
                )
            )
        remaining_candidate_budget = max(
            0, int(self.config.max_candidates) - int(candidate_count)
        )
        remaining_attempt_budget = max(
            0,
            int(self.config.resolved_sampling_attempts())
            - int(sampling_attempts),
        )
        metrics = progress_metrics(
            processed=int(sampling_attempts),
            passing=int(passing_count),
            estimated_stop=estimated_stop,
            elapsed_s=elapsed_s,
            baseline_processed=int(self._resume_sampling_attempts),
            eta_processed=int(passing_count),
            eta_baseline_processed=int(self._resume_passing_count),
            eta_target=int(self.config.required_passing),
            rolling_throughput_per_s=passing_snapshot.get("rolling_rate_per_s"),
            ewma_throughput_per_s=passing_snapshot.get("ewma_rate_per_s"),
            minimum_calibration_s=min(10.0, float(self.config.progress_interval_sec)),
            minimum_calibration_items=5,
        )
        progress_line = format_progress(
            component="teacher_mission_preparation",
            processed=int(sampling_attempts),
            passing=int(passing_count),
            estimated_stop=estimated_stop,
            elapsed_s=elapsed_s,
            baseline_processed=int(self._resume_sampling_attempts),
            eta_processed=int(passing_count),
            eta_baseline_processed=int(self._resume_passing_count),
            eta_target=int(self.config.required_passing),
            rolling_throughput_per_s=passing_snapshot.get("rolling_rate_per_s"),
            ewma_throughput_per_s=passing_snapshot.get("ewma_rate_per_s"),
            minimum_calibration_s=min(10.0, float(self.config.progress_interval_sec)),
            minimum_calibration_items=5,
            candidate_count=int(candidate_count),
            routed_count=int(routed_count),
            audited_count=int(routed_count),
            rejected=int(rejected_route_count),
            required_passing=int(self.config.required_passing),
            worker_count=int(self.config.workers),
            inflight="{}/{}".format(
                int(inflight_results), int(self.config.max_inflight_results)
            ),
            reorder_buffer=int(inflight_results),
            attempts_per_s=round(float(attempt_snapshot.get("stable_rate_per_s", 0.0)), 4),
            candidates_per_s=round(
                float(candidate_snapshot.get("stable_rate_per_s", 0.0)), 4
            ),
            audits_per_s=round(float(audit_snapshot.get("stable_rate_per_s", 0.0)), 4),
            passing_per_s=round(float(metrics["throughput_per_s"]), 4),
            elapsed=format_duration(elapsed_s),
            TARGET_AT_RISK=target_at_risk,
            ATTEMPT_BUDGET_AT_RISK=attempt_budget_at_risk,
            ESTIMATED_CANDIDATES_REQUIRED=estimated_candidates_required,
            ESTIMATED_ATTEMPTS_REQUIRED=estimated_attempts_required,
        )
        write_json_atomic(
            self.progress_path,
            {
                "contract_id": PREPARATION_CONTRACT_ID,
                "status": str(status),
                "candidate_count": int(candidate_count),
                "routed_count": int(routed_count),
                "passing_count": int(passing_count),
                "sampling_attempts": int(sampling_attempts),
                "last_sequence": max(-1, int(candidate_count) - 1),
                "elapsed_s": elapsed_s,
                "estimated_stop": estimated_stop,
                "throughput_per_s": float(metrics["throughput_per_s"]),
                "rolling_throughput_per_s": float(
                    metrics["rolling_throughput_per_s"]
                ),
                "ewma_throughput_per_s": float(metrics["ewma_throughput_per_s"]),
                "eta_h": (
                    float(metrics["eta_h"])
                    if metrics["eta_h"] is not None
                    and np.isfinite(float(metrics["eta_h"]))
                    else None
                ),
                "eta_status": metrics["eta_status"],
                "eta_target": int(self.config.required_passing),
                "attempts_per_s": float(
                    attempt_snapshot.get("stable_rate_per_s", 0.0)
                ),
                "candidates_per_s": float(
                    candidate_snapshot.get("stable_rate_per_s", 0.0)
                ),
                "audits_per_s": float(audit_snapshot.get("stable_rate_per_s", 0.0)),
                "passing_per_s": float(metrics["throughput_per_s"]),
                "rejected_route_count": int(rejected_route_count),
                "audited_count": int(routed_count),
                "required_passing": int(self.config.required_passing),
                "inflight_results": int(inflight_results),
                "progress": progress_line,
                "max_inflight_results": int(self.config.max_inflight_results),
                "max_reorder_buffer_results": int(self.config.max_inflight_results),
                "reorder_buffer_results": int(inflight_results),
                "remaining_passing": int(remaining_passing),
                "candidate_pass_rate": float(candidate_pass_rate),
                "attempt_pass_rate": float(attempt_pass_rate),
                "remaining_candidate_budget": int(remaining_candidate_budget),
                "remaining_attempt_budget": int(remaining_attempt_budget),
                "estimated_candidates_required": (
                    int(estimated_candidates_required)
                    if estimated_candidates_required is not None
                    else None
                ),
                "estimated_attempts_required": (
                    int(estimated_attempts_required)
                    if estimated_attempts_required is not None
                    else None
                ),
                "TARGET_AT_RISK": target_at_risk,
                "ATTEMPT_BUDGET_AT_RISK": attempt_budget_at_risk,
            },
        )
        print(progress_line, flush=True)

    def execute(self) -> MissionPreparationResult:
        started = time.monotonic()
        candidate_journal, audit_journal, routed_journal, mission_journal = self._journals()
        resumed = self._existing_count(candidate_journal) if self.config.resume else 0
        sampling_attempts = int(
            (candidate_journal.last_row or {}).get("sample_attempts", 0)
        )
        if self.config.resume:
            if self._existing_count(audit_journal) != resumed:
                raise ValueError("candidate/audit journal resume counts differ")
            # The parallel stream contains raw attempts, while the legacy
            # sequential stream contains only safe candidates.  Replaying
            # only the cheap RNG construction keeps resume deterministic
            # without repeating native safety work in the parent.
            self._skip_candidates(
                sampling_attempts
                if self.parallel_route_payload is not None
                else resumed
            )
        elif any(
            journal.row_count
            for journal in (candidate_journal, audit_journal, routed_journal, mission_journal)
        ):
            raise FileExistsError(
                "preparation journal exists; pass resume=True or remove the incomplete run"
            )

        candidate_count = resumed
        routed_count = int(routed_journal.row_count)
        if self.config.resume and hasattr(self.route_store_writer, "mission_count"):
            if int(self.route_store_writer.mission_count) != routed_count:
                raise ValueError(
                    "route-store/journal resume counts differ: {} != {}".format(
                        self.route_store_writer.mission_count, routed_count
                    )
                )
        passing_count = int(mission_journal.row_count)
        rejected_route_count = 0
        self._resume_sampling_attempts = int(sampling_attempts)
        self._resume_passing_count = int(passing_count)
        self._progress_trackers = {
            "attempts": ProgressRateTracker(initial_completed=sampling_attempts),
            "candidates": ProgressRateTracker(initial_completed=candidate_count),
            "audits": ProgressRateTracker(initial_completed=routed_count),
            "passing": ProgressRateTracker(initial_completed=passing_count),
        }
        generation_astar_calls = 0

        route_results = None

        def handle_candidate(
            candidate_number: int,
            candidate: Dict[str, Any],
            route_value: Any,
            audit_result: Optional[Mapping[str, Any]] = None,
        ) -> None:
            nonlocal routed_count, passing_count, rejected_route_count
            nonlocal generation_astar_calls
            route_status, route_length, route_stretch, route = _route_result(route_value)
            generation_astar_calls += 1
            candidate_row = {
                **candidate,
                "candidate_number": int(candidate_number),
                "route_status": route_status,
                "global_route_length_m": route_length,
                "global_route_stretch": route_stretch,
            }
            if route_status != "ok" or route is None:
                rejected_route_count += 1
                audit_row = {
                    "candidate_number": int(candidate_number),
                    "result": "no_route",
                }
                candidate_row["global_route_index"] = ""
                audit_journal.append(_normalise_audit_journal_row(audit_row))
                candidate_journal.append(candidate_row)
            elif route_stretch and route_stretch > float(self.config.max_route_stretch):
                rejected_route_count += 1
                candidate_row["route_status"] = "rejected_stretch"
                audit_row = {
                    "candidate_number": int(candidate_number),
                    "result": "route_stretch",
                }
                candidate_row["global_route_index"] = ""
                audit_journal.append(_normalise_audit_journal_row(audit_row))
                candidate_journal.append(candidate_row)
            else:
                candidate_row["episode_id"] = int(routed_count)
                route_index = self.route_store_writer.append(
                    route, goal=_goal_from_candidate(candidate)
                )
                routed_count += 1
                candidate_row["global_route_index"] = int(route_index)
                audit_candidate = dict(candidate)
                audit_candidate["candidate_number"] = int(candidate_number)
                audit = (
                    dict(audit_result)
                    if audit_result is not None
                    else _invoke_auditor(self.mission_auditor, audit_candidate, route)
                )
                audit_row = {
                    "candidate_number": int(candidate_number),
                    **audit,
                    "global_route_index": int(route_index),
                }
                audit_journal.append(_normalise_audit_journal_row(audit_row))
                candidate_journal.append(candidate_row)
                routed_journal.append(candidate_row)
                if str(audit.get("result", "")) == "success":
                    selected = {
                        **candidate_row,
                        **audit,
                        "global_route_index": int(route_index),
                        "episode_id": int(passing_count),
                    }
                    selected.update(_teacher_path_contract_fields(audit))
                    mission_journal.append(selected)
                    passing_count += 1

        last_progress_attempts = int(sampling_attempts)
        last_progress_time = started

        def maybe_write_progress(status: str = "checkpoint", force: bool = False) -> None:
            nonlocal last_progress_attempts, last_progress_time
            now = time.monotonic()
            due = (
                bool(force)
                or int(sampling_attempts) - last_progress_attempts
                >= int(self.config.checkpoint_interval)
                or now - last_progress_time >= float(self.config.progress_interval_sec)
            )
            if not due:
                return
            self._write_progress(
                status=status,
                candidate_count=candidate_count,
                routed_count=routed_count,
                passing_count=passing_count,
                sampling_attempts=sampling_attempts,
                started=started,
                rejected_route_count=rejected_route_count,
                inflight_results=(
                    route_results.pending_count if route_results is not None else 0
                ),
            )
            last_progress_attempts = int(sampling_attempts)
            last_progress_time = now

        try:
            if self.parallel_route_payload is not None and self.config.workers > 1:
                route_results = _BoundedPreparationRouteResults(
                    candidates=self.candidates,
                    start_candidate_number=candidate_count,
                    start_sampling_attempts=sampling_attempts,
                    max_candidates=int(self.config.max_candidates),
                    max_sampling_attempts=self.config.resolved_sampling_attempts(),
                    workers=int(self.config.workers),
                    max_inflight_results=int(self.config.max_inflight_results),
                    route_payload=self.parallel_route_payload,
                    max_accepted=int(self.config.max_candidates),
                    worker=self.parallel_worker,
                    initializer=self.parallel_worker_initializer,
                )
                for raw_attempt, raw_candidate, evaluation in route_results:
                    sampling_attempts = max(
                        int(sampling_attempts),
                        int(evaluation.get("sampling_attempts", raw_attempt)),
                    )
                    if not bool(evaluation.get("safe", False)):
                        maybe_write_progress()
                        continue
                    candidate = dict(evaluation.get("candidate") or raw_candidate)
                    candidate_number = int(candidate_count)
                    candidate["episode_id"] = candidate_number
                    candidate_count += 1
                    handle_candidate(
                        candidate_number,
                        candidate,
                        evaluation.get("route_value"),
                        evaluation.get("audit"),
                    )
                    maybe_write_progress()
                    if passing_count >= int(self.config.required_passing):
                        break
            else:
                while (
                    candidate_count < int(self.config.max_candidates)
                    and sampling_attempts < self.config.resolved_sampling_attempts()
                    and passing_count < int(self.config.required_passing)
                ):
                    try:
                        candidate = dict(next(self.candidates))
                    except StopIteration:
                        break
                    candidate_number = int(candidate_count)
                    candidate_count += 1
                    sampling_attempts = max(
                        sampling_attempts + 1,
                        int(candidate.get("sample_attempts", sampling_attempts + 1)),
                    )
                    handle_candidate(
                        candidate_number, candidate, self.route_planner(candidate)
                    )
                    maybe_write_progress()

            if passing_count < int(self.config.required_passing):
                maybe_write_progress(status="insufficient_passing", force=True)
                raise RuntimeError(
                    "only {} of {} required missions passed after {} candidates".format(
                        passing_count,
                        int(self.config.required_passing),
                        candidate_count,
                    )
                )

            routed_journal.commit(self.candidate_output)
            audit_journal.flush(fsync=True)
            if hasattr(self.route_store_writer, "set_candidate_index_sha256"):
                self.route_store_writer.set_candidate_index_sha256(
                    file_sha256(self.candidate_output)
                )
            route_store = self.route_store_writer.commit()
            route_extra = {}
            if (
                hasattr(route_store, "meta_path")
                and isinstance(getattr(route_store, "metadata", None), dict)
                and "global_route_contract_id" in route_store.metadata
            ):
                route_extra = route_store_provenance(
                    route_store, artifact_path=self.mission_output
                )
            _compact_with_extra_fields(
                mission_journal, self.mission_output, route_extra
            )
            if hasattr(route_store, "close"):
                route_store.close()
            candidate_sha = file_sha256(self.candidate_output)
            mission_sha = file_sha256(self.mission_output)
            self._write_progress(
                status="completed",
                candidate_count=candidate_count,
                routed_count=routed_count,
                passing_count=passing_count,
                sampling_attempts=sampling_attempts,
                started=started,
                rejected_route_count=rejected_route_count,
                inflight_results=0,
            )
            return MissionPreparationResult(
                candidate_count=candidate_count,
                routed_count=routed_count,
                passing_count=passing_count,
                rejected_route_count=rejected_route_count,
                sampling_attempts=sampling_attempts,
                generation_astar_calls=generation_astar_calls,
                resumed_candidate_count=resumed,
                replanned_committed_route_count=0,
                candidate_sha256=candidate_sha,
                mission_sha256=mission_sha,
            )
        except Exception:
            if hasattr(self.route_store_writer, "close"):
                self.route_store_writer.close()
            raise
        finally:
            if route_results is not None:
                route_results.close()
            candidate_journal.close()
            audit_journal.close()
            routed_journal.close()
            mission_journal.close()


def iter_preparation_candidates(
    *,
    sampler: Iterable[Dict[str, Any]],
    max_candidates: int,
) -> Iterator[Dict[str, Any]]:
    """Apply the candidate bound without converting an iterable to a list."""

    for index, candidate in enumerate(sampler):
        if index >= int(max_candidates):
            break
        yield dict(candidate)


__all__ = [
    "MAX_INFLIGHT_RESULTS",
    "MissionPreparationConfig",
    "MissionPreparationResult",
    "MissionPreparationRun",
    "PREPARATION_CONTRACT_ID",
    "iter_raw_mission_candidates",
    "iter_preparation_candidates",
]


def _formal_candidate_row(
    mission: Dict[str, Any],
    max_steps: int,
    *,
    route_resolution_m: float = 0.25,
    route_tracking_margin_m: float = 0.0,
) -> Dict[str, Any]:
    from planning.mission.spec import (
        mission_id_from_values,
        planar_distance_xy,
        task_contract_fields,
    )
    from planning.mission.global_route import GLOBAL_ROUTE_CONTRACT_ID

    start = mission["start"]
    goal = mission["goal"]
    return {
        "episode_id": int(mission["episode_id"]),
        "mission_id": mission_id_from_values(
            start[0], start[1], start[2], start[3], goal[0], goal[1], goal[2]
        ),
        **task_contract_fields(int(max_steps)),
        "start_x": float(start[0]),
        "start_y": float(start[1]),
        "start_z": float(start[2]),
        "start_yaw_deg": float(start[3]),
        "goal_x": float(goal[0]),
        "goal_y": float(goal[1]),
        "goal_z": float(goal[2]),
        "planar_distance_m": planar_distance_xy(
            start[0], start[1], goal[0], goal[1]
        ),
        "global_route_contract_id": GLOBAL_ROUTE_CONTRACT_ID,
        "global_route_resolution_m": float(route_resolution_m),
        "global_route_tracking_margin_m": float(route_tracking_margin_m),
        "start_valid_action_count": int(mission["start_valid_action_count"]),
        "goal_valid_action_count": int(mission["goal_valid_action_count"]),
        "sample_attempts": int(mission["sample_attempts"]),
    }


def build_argument_parser():
    """Build the canonical bounded-preparation CLI parser."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare routed and Teacher-audited formal Pre-BC missions."
    )
    parser.add_argument(
        "--candidate-index",
        default="data/teach/flight_reliable_exact_teacher_20260829_v2/mission_candidates.csv",
    )
    parser.add_argument(
        "--missions",
        default="data/teach/flight_reliable_exact_teacher_20260829_v2/missions.csv",
    )
    parser.add_argument("--route-store-prefix", default="")
    parser.add_argument("--required-passing", type=int, default=int(pre_bc_value("mission", "required_passing")))
    parser.add_argument("--max-candidates", type=int, default=int(pre_bc_value("mission", "candidate_count")))
    parser.add_argument("--max-sampling-attempts", type=int, default=int(pre_bc_value("mission", "max_sampling_attempts")))
    parser.add_argument("--seed", type=int, default=int(pre_bc_value("mission", "seed")))
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--collision-cache", default="data/map_data/forest_voxels_10cm.npz")
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--inflate-radius", type=float, default=0.35)
    parser.add_argument("--goal-distance", type=float, default=40.0)
    parser.add_argument("--start-x-range", type=float, nargs=2, default=(-50.0, 50.0))
    parser.add_argument("--start-y-range", type=float, nargs=2, default=(-50.0, 50.0))
    parser.add_argument("--goal-y-offset-range", type=float, nargs=2, default=(-1.5, 1.5))
    parser.add_argument("--min-start-valid-actions", type=int, default=40)
    parser.add_argument("--min-goal-valid-actions", type=int, default=10)
    parser.add_argument("--map-margin-m", type=float, default=3.0)
    parser.add_argument("--check-step", type=int, default=2)
    parser.add_argument("--global-route-resolution-m", type=float, default=0.25)
    parser.add_argument("--global-route-lookahead-m", type=float, default=3.0)
    parser.add_argument("--global-route-tracking-margin-m", type=float, default=0.0)
    parser.add_argument("--max-route-stretch", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=DEFAULT_PREPARATION_WORKERS)
    parser.add_argument("--max-inflight-results", type=int, default=MAX_INFLIGHT_RESULTS)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--collision-threads", type=int, default=1)
    parser.add_argument("--progress-interval-sec", type=float, default=10.0)
    parser.add_argument(
        "--skip-canonical-smoke-mission",
        action="store_true",
        help="omit the fixed origin smoke candidate for an independent holdout",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    """Run the formal preparation owner against the current cache."""

    args = build_argument_parser().parse_args()
    from planning.contracts.teacher_path import primitive_reference_path_lengths_m
    from planning.mission.audit_worker import _audit_mission_row_with_state
    from planning.mission.global_route import GlobalRouteConfig, global_route_identity
    from planning.mission.sampling import iter_safe_missions
    from planning.native.geometry import NativeGeometryContext
    from planning.primitives.library import MotionPrimitiveLibrary, resolve_package_path
    from planning.teacher.policy import ReachabilityTeacher, TeacherConfig
    from planning.data.mission_routes import MissionRouteStoreWriter

    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    candidate_output = resolve_package_path(args.candidate_index)
    mission_output = resolve_package_path(args.missions)
    if not args.overwrite and not args.resume:
        for path in (candidate_output, mission_output):
            if path.exists():
                raise FileExistsError("output exists; pass --overwrite or --resume: {}".format(path))
    if args.overwrite:
        for path in (
            candidate_output,
            mission_output,
            candidate_output.with_suffix(".candidate.journal.jsonl"),
            candidate_output.with_suffix(".routed.candidate.journal.jsonl"),
            mission_output.with_suffix(".audit.journal.jsonl"),
            mission_output.with_suffix(".passing.journal.jsonl"),
            mission_output.with_suffix(".preparation.progress.json"),
        ):
            path.unlink(missing_ok=True)

    route_config = GlobalRouteConfig(
        resolution_m=float(args.global_route_resolution_m),
        lookahead_m=float(args.global_route_lookahead_m),
        tracking_margin_m=float(args.global_route_tracking_margin_m),
    )
    cache_path = resolve_package_path(args.collision_cache)
    teacher_config = TeacherConfig(
        global_route_resolution_m=float(args.global_route_resolution_m),
        global_route_lookahead_m=float(args.global_route_lookahead_m),
        global_route_tracking_margin_m=float(args.global_route_tracking_margin_m),
    )
    # Parallel preparation owns all native/Teacher state in the worker
    # initializer.  The parent is limited to deterministic raw construction,
    # bounded scheduling, and journal publication.
    geometry = None
    mpl = None
    checker = None
    route_planner = None
    teacher = None
    action_path_lengths = None
    if int(args.workers) <= 1:
        geometry = NativeGeometryContext.from_voxel_cache(
            cache_path,
            voxel_size=float(args.voxel_size),
            route_config=route_config,
        )
        mpl = MotionPrimitiveLibrary()
        checker = geometry.collision_checker(float(args.inflate_radius))
        route_planner = geometry.global_route_planner(checker, route_config)
        teacher = ReachabilityTeacher(mpl, checker, teacher_config)
        action_path_lengths = primitive_reference_path_lengths_m(mpl)

    route_identity = global_route_identity(route_config)
    route_identity["global_route_map_identity"] = {
        "cache_path": str(args.collision_cache),
        "cache_sha256": file_sha256(cache_path),
        "voxel_size_m": float(args.voxel_size),
        "inflate_radius_m": float(args.inflate_radius),
    }
    route_store_prefix = (
        Path(args.route_store_prefix).expanduser().resolve()
        if str(args.route_store_prefix).strip()
        else candidate_output.parent / "mission_routes"
    )
    source_config_identity = {
        "preparation": "prepare_teacher_missions.py",
        "seed": int(args.seed),
        "max_candidates": int(args.max_candidates),
        "required_passing": int(args.required_passing),
        "include_canonical_smoke_mission": not bool(
            args.skip_canonical_smoke_mission
        ),
    }
    if args.resume:
        writer = MissionRouteStoreWriter.resume(
            route_store_prefix,
            global_route_contract_id=route_identity["global_route_contract_id"],
            route_resolution_m=float(args.global_route_resolution_m),
            tracking_margin_m=float(args.global_route_tracking_margin_m),
            collision_map_identity=route_identity["global_route_map_identity"],
            source_config_identity=source_config_identity,
        )
    else:
        writer = MissionRouteStoreWriter(
            route_store_prefix,
            global_route_contract_id=route_identity["global_route_contract_id"],
            route_resolution_m=float(args.global_route_resolution_m),
            tracking_margin_m=float(args.global_route_tracking_margin_m),
            collision_map_identity=route_identity["global_route_map_identity"],
            source_config_identity=source_config_identity,
            expected_mission_count=None,
            overwrite=bool(args.overwrite),
        )

    def candidate_stream():
        if int(args.workers) > 1:
            yield from iter_raw_mission_candidates(
                seed=int(args.seed),
                max_attempts=int(args.max_sampling_attempts),
                max_accepted=int(args.max_candidates),
                start_x_range=args.start_x_range,
                start_y_range=args.start_y_range,
                goal_distance=float(args.goal_distance),
                goal_y_offset_range=args.goal_y_offset_range,
                altitude_levels=DEFAULT_ALTITUDE_LEVELS_M,
                include_canonical_first=not bool(args.skip_canonical_smoke_mission),
            )
            return
        missions = iter_safe_missions(
            seed=int(args.seed),
            mpl=mpl,
            checker=checker,
            start_x_range=args.start_x_range,
            start_y_range=args.start_y_range,
            goal_distance=float(args.goal_distance),
            goal_y_offset_range=args.goal_y_offset_range,
            min_start_valid_actions=int(args.min_start_valid_actions),
            min_goal_valid_actions=int(args.min_goal_valid_actions),
            map_margin_m=float(args.map_margin_m),
            check_step=int(args.check_step),
            max_attempts=int(args.max_sampling_attempts),
            max_accepted=int(args.max_candidates),
            include_canonical_first=not bool(args.skip_canonical_smoke_mission),
        )
        for mission in missions:
            yield _formal_candidate_row(
                mission,
                int(args.max_steps),
                route_resolution_m=float(args.global_route_resolution_m),
                route_tracking_margin_m=float(args.global_route_tracking_margin_m),
            )

    def route(candidate):
        if route_planner is None:
            raise RuntimeError("parallel preparation route planner is not local")
        start = [candidate["start_x"], candidate["start_y"], candidate["start_z"]]
        goal = [candidate["goal_x"], candidate["goal_y"], candidate["goal_z"]]
        try:
            route = route_planner.plan(start, goal)
        except RuntimeError:
            return "no_route", 0.0, 0.0, None
        route = np.asarray(route, dtype=np.float32)
        route_length = float(np.sum(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)))
        direct = float(np.linalg.norm(np.asarray(goal[:2]) - np.asarray(start[:2])))
        return "ok", route_length, route_length / max(1.0e-6, direct), route

    def audit(candidate, route_value):
        if teacher is None or mpl is None or action_path_lengths is None:
            raise RuntimeError("parallel preparation Teacher audit is not local")
        row = dict(candidate)
        row["global_route_index"] = 0
        result = _audit_mission_row_with_state(
            (int(candidate.get("candidate_number", 0)), row),
            teacher,
            mpl,
            int(args.max_steps),
            action_path_lengths,
            _SingleRouteView(route_value),
        )
        return result

    config = MissionPreparationConfig(
        required_passing=int(args.required_passing),
        max_candidates=int(args.max_candidates),
        max_sampling_attempts=int(args.max_sampling_attempts),
        max_route_stretch=float(args.max_route_stretch),
        seed=int(args.seed),
        workers=int(args.workers),
        max_inflight_results=int(args.max_inflight_results),
        checkpoint_interval=int(args.checkpoint_interval),
        collision_threads=int(args.collision_threads),
        progress_interval_sec=float(args.progress_interval_sec),
        resume=bool(args.resume),
    )
    parallel_route_payload = None
    if int(args.workers) > 1:
        parallel_route_payload = {
            "cache_path": str(cache_path),
            "voxel_size": float(args.voxel_size),
            "inflate_radius": float(args.inflate_radius),
            "route_resolution_m": float(route_config.resolution_m),
            "flight_z_min_m": float(route_config.flight_z_min_m),
            "flight_z_max_m": float(route_config.flight_z_max_m),
            "route_lookahead_m": float(route_config.lookahead_m),
            "route_tracking_margin_m": float(route_config.tracking_margin_m),
            "nearest_free_radius_cells": int(route_config.nearest_free_radius_cells),
            "collision_backend": "cpp_cpu",
            "collision_threads": int(args.collision_threads),
            "max_steps": int(args.max_steps),
            "max_route_stretch": float(args.max_route_stretch),
            "teacher_config": dict(teacher_config.__dict__),
            "start_x_range": tuple(float(value) for value in args.start_x_range),
            "start_y_range": tuple(float(value) for value in args.start_y_range),
            "goal_distance": float(args.goal_distance),
            "goal_y_offset_range": tuple(
                float(value) for value in args.goal_y_offset_range
            ),
            "altitude_levels": tuple(float(value) for value in DEFAULT_ALTITUDE_LEVELS_M),
            "min_start_valid_actions": int(args.min_start_valid_actions),
            "min_goal_valid_actions": int(args.min_goal_valid_actions),
            "map_margin_m": float(args.map_margin_m),
            "check_step": int(args.check_step),
        }
    try:
        result = MissionPreparationRun(
            candidates=candidate_stream(),
            route_planner=route,
            mission_auditor=audit,
            route_store_writer=writer,
            candidate_output=candidate_output,
            mission_output=mission_output,
            config=config,
            parallel_route_payload=parallel_route_payload,
            parallel_worker=(
                _evaluate_preparation_candidate if int(args.workers) > 1 else None
            ),
            parallel_worker_initializer=(
                _prepare_preparation_worker if int(args.workers) > 1 else None
            ),
        ).execute()
    finally:
        if geometry is not None:
            geometry.close()
    metadata = {
        "preparation_contract_id": PREPARATION_CONTRACT_ID,
        "candidate_output": str(candidate_output),
        "mission_output": str(mission_output),
        "route_store_prefix": str(route_store_prefix),
        "cache_sha256": file_sha256(cache_path),
        "seed": int(args.seed),
        "max_candidates": int(args.max_candidates),
        "required_passing": int(args.required_passing),
        "max_sampling_attempts": int(config.resolved_sampling_attempts()),
        "include_canonical_smoke_mission": not bool(
            args.skip_canonical_smoke_mission
        ),
        "max_inflight_results": int(args.max_inflight_results),
        "max_reorder_buffer_routes": (
            int(args.max_inflight_results) if int(args.workers) > 1 else 0
        ),
        "generation_astar_calls": int(result.generation_astar_calls),
        "audit_astar_call_count": 0,
        "collection_astar_call_count": 0,
        "relabel_astar_call_count": 0,
        **result.__dict__,
        **route_identity,
    }
    write_json_atomic(candidate_output.with_suffix(".preparation.json"), metadata)
    print("MISSION_PREPARATION")
    print("  candidates:", result.candidate_count)
    print("  routed:", result.routed_count)
    print("  passing:", result.passing_count)
    print("  sampling_attempts:", result.sampling_attempts)
    print("  generation_astar_calls:", result.generation_astar_calls)
    print("RESULT=PASS")
    return 0


__all__.append("build_argument_parser")
__all__.append("main")
