"""Single-owner scalar TensorBoard logging for the AWAC experiment lineage.

This module is deliberately an observability boundary.  It does not own an
optimizer, replay sampling, environment interaction, or any AWAC calculation.
The Standard runner supplies already-computed diagnostics and this class only
serializes finite scalar values under the frozen step contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional


TENSORBOARD_SCHEMA_ID = "awac_tensorboard_logging"
TENSORBOARD_GLOBAL_STEP_OWNER = "phase_local_online_environment_steps"
TENSORBOARD_LOG_INTERVAL_STEPS = 100


_TRAINING_FIELDS = {
    "progress": (
        "online_env_steps",
        "online_transitions_committed",
        "replay_size",
        "completed_episodes",
        "actor_update_count",
        "critic_update_count",
        "target_update_count",
    ),
    "actor": (
        "loss",
        "entropy",
        "bc_kl_mean",
        "bc_top1_disagreement_rate",
        "parameter_delta_norm",
        "grad_norm",
        "depth_grad_norm",
        "vector_grad_norm",
        "head_grad_norm",
    ),
    "awac": (
        "advantage_mean",
        "advantage_std",
        "weight_mean",
        "weight_p50",
        "weight_p95",
        "weight_max",
        "weight_clip_fraction",
        "raw_weight_ess_fraction",
        "normalized_clipped_weight_ess_fraction",
        "ess_batch_size",
        "ess_aggregation_window_updates",
    ),
    "critic": (
        "td_loss_mean",
        "td_loss_p95",
        "q_mean",
        "q_std",
        "q_p01",
        "q_p99",
        "twin_q_disagreement_mean",
        "twin_q_disagreement_p95",
        "grad_norm",
        "cql_loss",
        "bellman_loss",
    ),
    "episode": (
        "return_mean",
        "length_mean",
        "success_rate_window",
        "collision_rate_window",
        "dead_end_rate_window",
        "timeout_rate_window",
    ),
    "terminal": (
        "success_count",
        "collision_count",
        "dead_end_count",
        "timeout_count",
        "hard_altitude_count",
        "runtime_drop_count",
    ),
    "replay": (
        "size",
        "bc_calibration_rows",
        "awac_online_rows",
        "bc_calibration_fraction",
        "awac_online_fraction",
    ),
    "runtime": (
        "env_steps_per_sec",
        "committed_transitions_per_sec",
        "episodes_per_min",
        "actor_updates_per_sec",
        "critic_updates_per_sec",
        "transition_drop_count",
        "runtime_failure_count",
    ),
    "lr": (
        "actor_depth",
        "actor_vector",
        "actor_head",
        "critic_depth",
        "critic_vector",
        "critic_head",
    ),
    "health": (
        "nan_count",
        "inf_count",
        "nonfinite_metric_count",
        "q_divergence",
        "actor_divergence",
    ),
    "confidence": (
        "mean",
        "p05",
        "p50",
        "p95",
        "delta_q_mean",
        "delta_q_norm_mean",
        "uncertainty_mean",
        "uncertainty_norm_mean",
        "q_margin_norm_mean",
        "twin_disagreement_norm_mean",
    ),
    "adaptive_bc_kl": (
        "beta_mean",
        "beta_p05",
        "beta_p50",
        "beta_p95",
        "low_confidence_beta_mean",
        "high_confidence_beta_mean",
    ),
}
_OPTIONAL_SCALAR_TAGS = frozenset(
    {
        "critic/cql_loss",
        "critic/bellman_loss",
        "confidence/mean",
        "confidence/p05",
        "confidence/p50",
        "confidence/p95",
        "confidence/delta_q_mean",
        "confidence/delta_q_norm_mean",
        "confidence/uncertainty_mean",
        "confidence/uncertainty_norm_mean",
        "confidence/q_margin_norm_mean",
        "confidence/twin_disagreement_norm_mean",
        "adaptive_bc_kl/beta_mean",
        "adaptive_bc_kl/beta_p05",
        "adaptive_bc_kl/beta_p50",
        "adaptive_bc_kl/beta_p95",
        "adaptive_bc_kl/low_confidence_beta_mean",
        "adaptive_bc_kl/high_confidence_beta_mean",
    }
)


class AWACTensorBoardLogger:
    """Write one AWAC experiment lineage using one SummaryWriter owner."""

    def __init__(
        self,
        log_dir: Path,
        *,
        log_interval_steps: int = TENSORBOARD_LOG_INTERVAL_STEPS,
    ) -> None:
        if int(log_interval_steps) <= 0:
            raise ValueError("TensorBoard log interval must be positive")
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard logging requested but torch.utils.tensorboard is unavailable"
            ) from exc
        self._writer = SummaryWriter(log_dir=str(self.log_dir))
        self.log_interval_steps = int(log_interval_steps)
        self._last_step = -1
        self._closed = False
        self._logged_steps = []
        self._started_at = time.monotonic()

    @classmethod
    def required_scalar_tags(cls):
        """Return the training tags required by the formal observability gate."""

        return {
            "{}/{}".format(namespace, field)
            for namespace, fields in _TRAINING_FIELDS.items()
            for field in fields
        } - set(_OPTIONAL_SCALAR_TAGS)

    @property
    def last_step(self) -> int:
        return int(self._last_step)

    @property
    def closed(self) -> bool:
        return bool(self._closed)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("TensorBoard logger is already closed")

    def _check_step(self, global_step: int) -> int:
        self._check_open()
        step = int(global_step)
        if step < 0:
            raise ValueError("TensorBoard global step must be non-negative")
        if step < self._last_step:
            raise ValueError(
                "TensorBoard global step regressed: {} < {}".format(
                    step, self._last_step
                )
            )
        self._last_step = step
        if not self._logged_steps or self._logged_steps[-1] != step:
            self._logged_steps.append(step)
        return step

    def _scalar(self, tag: str, value: Any, step: int) -> bool:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(numeric):
            return False
        self._writer.add_scalar(str(tag), numeric, step)
        return True

    def log_snapshot(self, snapshot: Mapping[str, Mapping[str, Any]], *, global_step: int) -> None:
        """Serialize one already-computed training snapshot."""

        if not isinstance(snapshot, Mapping):
            raise TypeError("TensorBoard snapshot must be a mapping")
        step = self._check_step(global_step)
        for namespace, fields in _TRAINING_FIELDS.items():
            values = snapshot.get(namespace, {})
            if not isinstance(values, Mapping):
                continue
            for field in fields:
                if field in values:
                    self._scalar("{}/{}".format(namespace, field), values[field], step)
        self._writer.flush()

    def log_runner(self, runner: Any, *, force: bool = False) -> None:
        """Serialize the runner-owned snapshot at its phase-local step."""

        step = int(runner.online_env_steps)
        if not force and step < int(getattr(runner, "next_tensorboard_step", step)):
            return
        self.log_snapshot(runner.tensorboard_snapshot(), global_step=step)

    def log_dev100(
        self,
        summary: Mapping[str, Any],
        *,
        global_step: int,
        bc_baseline: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Append the fixed Dev100 point to the same experiment lineage."""

        if not isinstance(summary, Mapping):
            raise TypeError("Dev100 summary must be a mapping")
        step = self._check_step(global_step)
        baseline = {
            "success_rate": 0.69,
            "collision_rate": 0.09,
            "dead_end_rate": 0.22,
        }
        if bc_baseline is not None:
            baseline.update({key: float(value) for key, value in bc_baseline.items()})
        mapping = {
            "success_rate": summary.get("success_rate"),
            "collision_rate": summary.get("collision_rate"),
            "dead_end_rate": summary.get("dead_end_rate"),
            "timeout_rate": summary.get("timeout_rate"),
            "mean_return": summary.get("episode_return_mean"),
            "mean_steps": summary.get(
                "episode_steps_mean", summary.get("mean_steps")
            ),
            "delta_success_vs_bc": (
                None
                if summary.get("success_rate") is None
                else float(summary["success_rate"])
                - float(baseline["success_rate"])
            ),
        }
        for field, value in mapping.items():
            if value is not None:
                self._scalar("dev100/{}".format(field), value, step)
        self._scalar("baseline/bc_dev_success_rate", baseline["success_rate"], step)
        self._scalar("baseline/bc_dev_collision_rate", baseline["collision_rate"], step)
        self._scalar("baseline/bc_dev_dead_end_rate", baseline["dead_end_rate"], step)
        self._scalar("evaluation/success_rate", summary.get("success_rate"), step)
        self._scalar("evaluation/collision_rate", summary.get("collision_rate"), step)
        self._scalar("evaluation/dead_end_rate", summary.get("dead_end_rate"), step)
        self._scalar("evaluation/timeout_rate", summary.get("timeout_rate"), step)
        self._scalar(
            "evaluation/episode_return_mean", summary.get("episode_return_mean"), step
        )
        self._writer.add_text(
            "evaluation/contract",
            json.dumps(
                {
                    "episodes": summary.get("episodes"),
                    "checkpoint_sha256": summary.get("checkpoint_sha256", ""),
                    "mission_index_sha256": summary.get("mission_index_sha256", ""),
                },
                sort_keys=True,
            ),
            step,
        )
        self._writer.flush()

    def manifest(self) -> dict:
        return {
            "schema_id": TENSORBOARD_SCHEMA_ID,
            "owner": "planning.awac.tensorboard.AWACTensorBoardLogger",
            "log_dir": str(self.log_dir),
            "global_step_owner": TENSORBOARD_GLOBAL_STEP_OWNER,
            "log_interval_steps": self.log_interval_steps,
            "scalars_only": True,
            "last_step": int(self._last_step),
            "logged_steps": list(self._logged_steps),
            "required_scalar_tags": sorted(self.required_scalar_tags()),
            "awac_ess_contract": {
                "raw_weight_ess_fraction": (
                    "mean of learner awac_raw_weight_ess_fraction over the "
                    "current runner update window"
                ),
                "normalized_clipped_weight_ess_fraction": (
                    "mean of learner awac_weight_ess_fraction over the current "
                    "runner update window; weights are normalized and capped"
                ),
                "batch_size_owner": "StandardAWACOnlineRunner.batch_size",
                "aggregation_window_owner": "StandardAWACOnlineRunner._online_metric_rows",
                "clip_fraction_is_separate": True,
            },
        }

    def physical_size_mb(self) -> float:
        total = sum(
            path.stat().st_size
            for path in self.log_dir.glob("*")
            if path.is_file()
        )
        return float(total) / (1024.0 * 1024.0)

    def close(self) -> None:
        if self._closed:
            return
        # Mark closed before touching the backend so a second cleanup attempt
        # is a no-op even when flush/close raises.  Try both operations and
        # re-raise the first backend error; trainer cleanup catches it without
        # replacing the owning runtime/checkpoint exception.
        self._closed = True
        first_error = None
        try:
            self._writer.flush()
        except BaseException as error:
            first_error = error
        try:
            self._writer.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error


__all__ = [
    "AWACTensorBoardLogger",
    "TENSORBOARD_GLOBAL_STEP_OWNER",
    "TENSORBOARD_LOG_INTERVAL_STEPS",
    "TENSORBOARD_SCHEMA_ID",
]
