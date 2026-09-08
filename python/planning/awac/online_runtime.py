"""Standard masked-discrete AWAC online runtime.

This module owns only the Standard phase orchestration.  Environment process
lifecycle, request/reply behavior, observation validation, terminal handling,
and replay field encoding remain owned by the existing calibration/runtime
modules.  Learner mathematics remains in :mod:`planning.awac.learner`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from pathlib import Path
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from planning.awac.calibration_runtime import (
    CalibrationMission,
    CalibrationReplayProducer,
    CalibrationRuntimeError,
    _EpisodeState,
    _validate_observation,
)
from planning.awac.checkpoint import build_standard_online_exact_resume_state
from planning.awac.contract import TERMINAL_REASONS
from planning.awac.interaction import BehaviorSource
from planning.awac.optimization import (
    actor_update_due,
    critic_update_target,
)
from planning.contracts.feature import (
    INITIAL_PREV_ACTION,
    NUM_ACTIONS,
    POLICY_VECTOR_DIM,
    action_onehot,
)


STANDARD_ONLINE_RUNTIME_SCHEMA_ID = "awac_standard_online_runtime_v1"
STANDARD_ONLINE_SUMMARY_SCHEMA_ID = "awac_standard_online_summary_v1"


def _state_fingerprint(learner: Any) -> str:
    actor = getattr(learner, "actor", None)
    state_dict = actor.state_dict() if actor is not None else None
    if not isinstance(state_dict, Mapping):
        return ""
    fingerprint = getattr(learner, "_state_fingerprint", None)
    if callable(fingerprint):
        return str(fingerprint(state_dict))
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach().cpu().contiguous().numpy()
        array = np.ascontiguousarray(value)
        digest.update(str(name).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _finite_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize numeric learner diagnostics without changing their owner."""

    keys = (
        "critic_td_loss",
        "q_mean",
        "q_std",
        "q_p01",
        "q_p99",
        "critic_td_loss_p95",
        "critic_loss",
        "critic_cql_loss",
        "critic1_updated",
        "critic2_updated",
        "target_critic1_updated",
        "target_critic2_updated",
        "critic_parameter_delta_norm",
        "critic1_parameter_delta_norm",
        "critic2_parameter_delta_norm",
        "awac_advantage_mean",
        "awac_advantage_std",
        "awac_weight_mean",
        "awac_weight_p50",
        "awac_weight_p95",
        "awac_weight_max",
        "awac_weight_clip_fraction",
        "awac_raw_weight_ess_fraction",
        "awac_weight_ess_fraction",
        "bc_kl",
        "twin_q_disagreement_mean",
        "twin_disagreement_mean",
        "confidence_mean",
        "confidence_p05",
        "confidence_p50",
        "confidence_p95",
        "confidence_delta_q_mean",
        "confidence_delta_q_norm_mean",
        "confidence_uncertainty_mean",
        "confidence_uncertainty_norm_mean",
        "confidence_q_margin_norm_mean",
        "confidence_twin_disagreement_norm_mean",
        "adaptive_bc_kl_beta_mean",
        "adaptive_bc_kl_beta_p05",
        "adaptive_bc_kl_beta_p50",
        "adaptive_bc_kl_beta_p95",
        "adaptive_bc_kl_low_confidence_beta_mean",
        "adaptive_bc_kl_high_confidence_beta_mean",
        "actor_entropy",
        "bc_awac_top1_disagreement_rate",
        "actor_gradient_norm",
        "actor_parameter_delta_norm",
        "actor_depth_gradient_norm",
        "actor_vector_gradient_norm",
        "actor_head_gradient_norm",
        "actor_depth_parameter_delta_norm",
        "actor_vector_parameter_delta_norm",
        "actor_head_parameter_delta_norm",
        "actor_depth_update_norm",
        "actor_vector_update_norm",
        "actor_head_update_norm",
        "actor_depth_update",
        "actor_vector_update",
        "actor_head_update",
    )
    result: Dict[str, Any] = {}
    for key in keys:
        values = []
        for row in rows:
            if key not in row:
                continue
            value = float(row[key])
            if not math.isfinite(value):
                raise FloatingPointError(
                    "Standard AWAC metric {} is non-finite".format(key)
                )
            values.append(value)
        if values:
            array = np.asarray(values, dtype=np.float64)
            result["{}_mean".format(key)] = float(array.mean())
            result["{}_p95".format(key)] = float(np.quantile(array, 0.95))
            result["{}_max".format(key)] = float(array.max())
    return result


class StandardAWACOnlineRunner(CalibrationReplayProducer):
    """Run the continuous Standard AWAC phase on an injected environment pool.

    The inherited producer supplies mission assignment, exact observation and
    endpoint validation, staged terminal episodes, and cleanup.  This class
    changes only the behavior source, policy action owner, schedule, and
    phase-local progress accounting.
    """

    def __init__(
        self,
        *,
        missions: Sequence[CalibrationMission],
        pool: Any,
        replay: Any,
        learner: Any,
        action_selector: Optional[Callable[..., int]] = None,
        normalizer: Any = None,
        torch: Any = None,
        device: Any = None,
        online_env_steps: int,
        batch_size: int = 128,
        learning_starts: int = 5000,
        actor_learning_starts: int = 8000,
        critic_burnin_updates: int = 2000,
        actor_update_interval: int = 4,
        updates_per_step: float = 0.50,
        behavior_temperature: float = 1.0,
        max_episodes: Optional[int] = None,
        ready_timeout_s: float = 30.0,
        reset_timeout_s: float = 10.0,
        step_timeout_s: float = 30.0,
        reset_settle: float = 0.30,
        rng: Optional[np.random.RandomState] = None,
        mission_source_identity: Optional[Mapping[str, Any]] = None,
        expected_runtime_ids: Optional[Mapping[int, str]] = None,
        replay_identity: Optional[Mapping[str, Any]] = None,
        output_dir: Optional[Path] = None,
        phase1_state: Any = None,
        on_checkpoint: Optional[Callable[["StandardAWACOnlineRunner", Mapping[str, Any]], Any]] = None,
        checkpoint_interval_steps: int = 5000,
        starting_replay_size: Optional[int] = None,
        starting_replay_total_added: Optional[int] = None,
        tensorboard_logger: Any = None,
        tensorboard_log_interval_steps: int = 100,
    ) -> None:
        budget = int(online_env_steps)
        if budget <= 0:
            raise ValueError("online_env_steps must be positive")
        if float(behavior_temperature) <= 0.0 or not math.isfinite(float(behavior_temperature)):
            raise ValueError("behavior_temperature must be finite and positive")
        if int(actor_learning_starts) < int(learning_starts):
            raise ValueError("actor_learning_starts must be at least learning_starts")
        if int(critic_burnin_updates) < 0:
            raise ValueError("critic_burnin_updates must be non-negative")
        if int(actor_update_interval) <= 0:
            raise ValueError("actor_update_interval must be positive")
        if int(checkpoint_interval_steps) <= 0:
            raise ValueError("checkpoint_interval_steps must be positive")

        all_episode_ids = {mission.episode_id for mission in missions}
        super().__init__(
            missions=missions,
            pool=pool,
            replay=replay,
            learner=learner,
            train_episode_ids=all_episode_ids,
            holdout_episode_ids=set(),
            action_selector=action_selector,
            normalizer=normalizer,
            torch=torch,
            device=device,
            batch_size=int(batch_size),
            learning_starts=int(learning_starts),
            updates_per_step=float(updates_per_step),
            max_transitions=budget,
            max_episodes=int(max_episodes or len(tuple(missions))),
            ready_timeout_s=float(ready_timeout_s),
            reset_timeout_s=float(reset_timeout_s),
            step_timeout_s=float(step_timeout_s),
            reset_settle=float(reset_settle),
            rng=rng,
            mission_source_identity=mission_source_identity,
            expected_runtime_ids=expected_runtime_ids,
            output_dir=output_dir,
        )
        self.online_env_steps_budget = budget
        self.actor_learning_starts = int(actor_learning_starts)
        self.critic_burnin_updates = int(critic_burnin_updates)
        self.actor_update_interval = int(actor_update_interval)
        self.behavior_temperature = float(behavior_temperature)
        self.checkpoint_interval_steps = int(checkpoint_interval_steps)
        self.phase1_state = phase1_state
        self.replay_identity = dict(replay_identity or {})
        self.on_checkpoint = on_checkpoint
        self._next_checkpoint_step = self.checkpoint_interval_steps
        if int(tensorboard_log_interval_steps) <= 0:
            raise ValueError("tensorboard log interval must be positive")
        self.tensorboard_logger = tensorboard_logger
        self.tensorboard_log_interval_steps = int(tensorboard_log_interval_steps)
        self._next_tensorboard_step = self.tensorboard_log_interval_steps
        self._episode_window = deque(maxlen=100)
        self._terminal_counts = {
            str(reason): 0 for reason in TERMINAL_REASONS
        }
        self._tensorboard_started_at = time.monotonic()
        self._starting_replay_size = int(
            getattr(replay, "size", 0)
            if starting_replay_size is None
            else starting_replay_size
        )
        self._starting_replay_total_added = int(
            getattr(replay, "total_added", self._starting_replay_size)
            if starting_replay_total_added is None
            else starting_replay_total_added
        )
        if self._starting_replay_size < 0 or self._starting_replay_total_added < 0:
            raise ValueError("starting replay counters must be non-negative")
        self._online_metric_rows = []
        self._online_transitions_committed = 0
        self.metrics.update(
            {
                "behavior_source": "AWAC_ONLINE",
                "behavior_policy_action_mode": "masked_categorical",
                "behavior_policy_temperature": float(self.behavior_temperature),
                "starting_replay_size": self._starting_replay_size,
                "starting_replay_total_added": self._starting_replay_total_added,
                "online_env_steps_budget": self.online_env_steps_budget,
                "online_budget_counter_owner": "phase_local_online_environment_steps",
                "learning_starts_counter_owner": "replay_total_added",
                "actor_learning_starts_counter_owner": "replay_total_added",
                "critic_burnin_counter_owner": "learner.update_step",
            }
        )

    @property
    def online_env_steps(self) -> int:
        return int(self._environment_step_count)

    @property
    def online_transitions_committed(self) -> int:
        return int(self._online_transitions_committed)

    @property
    def starting_replay_size(self) -> int:
        return int(self._starting_replay_size)

    @property
    def starting_replay_total_added(self) -> int:
        return int(self._starting_replay_total_added)

    @property
    def next_tensorboard_step(self) -> int:
        return int(self._next_tensorboard_step)

    @property
    def online_metric_rows(self):
        return tuple(dict(row) for row in self._online_metric_rows)

    def _phase1_snapshot(self) -> Dict[str, Any]:
        if self.phase1_state is None:
            return {
                "phase": "ACTOR_ENABLED_STANDARD_AWAC",
                "actor_update_enabled": True,
                "stop_reason": "",
                "transition_records": [],
                "processed_milestones": [],
            }
        return {
            "phase": str(getattr(self.phase1_state, "phase", "")),
            "actor_update_enabled": bool(
                getattr(self.phase1_state, "actor_update_enabled", False)
            ),
            "stop_reason": str(getattr(self.phase1_state, "stop_reason", "")),
            "transition_records": [
                dict(value)
                for value in getattr(self.phase1_state, "transition_records", ())
            ],
            "processed_milestones": [
                int(value)
                for value in getattr(self.phase1_state, "processed_milestones", ())
            ],
        }

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
                raise CalibrationRuntimeError(
                    "formal AWAC action selector is not configured"
                )
            depth = np.stack(tuple(state.depth_history), axis=0).astype(
                np.float32, copy=False
            )[None, ...]
            vector = np.concatenate(
                (
                    self._policy_continuous_vector(state.observation),
                    action_onehot(int(state.previous_action)),
                ),
                axis=0,
            ).astype(np.float32, copy=False)[None, ...]
            if vector.shape != (1, POLICY_VECTOR_DIM):
                raise CalibrationRuntimeError("AWAC policy vector dimension failed")
            depth_tensor = self.torch.from_numpy(depth).to(
                device=self.device, dtype=self.torch.float32
            )
            vector_tensor = self.torch.from_numpy(vector).to(
                device=self.device, dtype=self.torch.float32
            )
            mask_tensor = self.torch.from_numpy(
                np.asarray(state.action_mask, dtype=np.bool_)[None, ...]
            ).to(device=self.device, dtype=self.torch.bool)
            try:
                act_kwargs = {
                    "deterministic": False,
                    "temperature": self.behavior_temperature,
                }
                learner_config = getattr(self.learner, "config", None)
                if bool(
                    getattr(
                        learner_config,
                        "enable_primitive_neighbor_exploration",
                        False,
                    )
                ):
                    act_kwargs["anchor_primitive"] = int(state.previous_action)
                selected = self.learner.act(
                    depth_tensor,
                    vector_tensor,
                    mask_tensor,
                    **act_kwargs,
                )
            except TypeError:
                # Narrow test doubles from pre-Standard code do not expose
                # temperature yet; the production learner does.
                selected = self.learner.act(
                    depth_tensor,
                    vector_tensor,
                    mask_tensor,
                    deterministic=False,
                )
            action = selected[0] if isinstance(selected, (tuple, list)) else selected
            if hasattr(action, "detach"):
                action = action.detach().cpu().reshape(-1)[0].item()
        try:
            value = int(action)
        except (TypeError, ValueError) as exc:
            raise CalibrationRuntimeError("AWAC action is not an integer") from exc
        if value < 0 or value >= NUM_ACTIONS or not bool(state.action_mask[value]):
            raise CalibrationRuntimeError("AWAC action is outside the current action mask")
        return value

    def _make_transition_for_worker(
        self,
        worker_id: int,
        state: _EpisodeState,
        result: Mapping[str, Any],
        action: int,
    ):
        transition, next_observation, next_mask, reason = super()._make_transition_for_worker(
            worker_id, state, result, action
        )
        # The base producer owns terminal validation.  Retain its already
        # validated reason only as observability state; it is not serialized
        # into the replay row and cannot affect action/reward semantics.
        state._last_terminal_reason = str(reason)
        transition["behavior_source"] = int(BehaviorSource.AWAC_ONLINE)
        return transition, next_observation, next_mask, reason

    def _append_episode(self, state: _EpisodeState) -> None:
        """Commit one complete episode and account only durable rows."""

        super()._append_episode(state)
        self._online_transitions_committed += len(state.pending_transitions)
        reason = str(getattr(state, "_last_terminal_reason", ""))
        if reason not in self._terminal_counts:
            raise CalibrationRuntimeError(
                "Standard AWAC terminal reason is not observable: {}".format(reason)
            )
        rewards = [float(row["reward"]) for row in state.pending_transitions]
        record = {
            "return": float(sum(rewards)),
            "length": int(len(state.pending_transitions)),
            "terminal_reason": reason,
        }
        self._episode_window.append(record)
        self._terminal_counts[reason] += 1

    def _update_critics(self) -> None:
        total_added = int(
            getattr(self.replay, "total_added", getattr(self.replay, "size", 0))
        )
        target_updates = critic_update_target(
            replay_total_added=total_added,
            learning_starts=self.learning_starts,
            updates_per_step=self.updates_per_step,
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
            update_actor = actor_update_due(
                replay_total_added=total_added,
                learner_update_step=int(getattr(self.learner, "update_step", 0)),
                actor_learning_starts=self.actor_learning_starts,
                critic_burnin_updates=self.critic_burnin_updates,
                actor_update_interval=self.actor_update_interval,
            )
            update = getattr(self.learner, "update", None)
            if not callable(update):
                raise CalibrationRuntimeError("Standard AWAC learner has no update")
            metrics = update(batch, update_actor=bool(update_actor))
            if not isinstance(metrics, Mapping):
                raise CalibrationRuntimeError("AWAC learner metrics are not a mapping")
            metrics = dict(metrics)
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)) and not math.isfinite(float(value)):
                    raise FloatingPointError(
                        "Standard AWAC metric {} is non-finite".format(key)
                    )
            self._online_metric_rows.append(metrics)
            next_count = int(getattr(self.learner, "critic_update_count", current))
            if next_count <= current:
                raise CalibrationRuntimeError(
                    "Standard AWAC learner did not advance Critic update count"
                )
            current = next_count

    def tensorboard_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Build scalar-only diagnostics from existing runtime/learner owners."""

        learner_metrics = _finite_metrics(self._online_metric_rows)

        def metric(name: str, default: Optional[float] = None) -> Optional[float]:
            value = learner_metrics.get("{}_mean".format(name), default)
            if value is None:
                return None
            return float(value)

        replay_size = int(getattr(self.replay, "size", 0))
        source_counts = {"BC_CALIBRATION": 0, "AWAC_ONLINE": 0}
        arrays = getattr(self.replay, "arrays", {})
        if isinstance(arrays, Mapping) and "behavior_source" in arrays:
            values = np.asarray(arrays["behavior_source"][:replay_size], dtype=np.int64)
            source_counts["BC_CALIBRATION"] = int(np.count_nonzero(values == 0))
            source_counts["AWAC_ONLINE"] = int(np.count_nonzero(values == 1))
        total_rows = max(1, sum(source_counts.values()))
        elapsed = max(time.monotonic() - self._tensorboard_started_at, 1.0e-9)
        episode_values = tuple(self._episode_window)
        episode_count = len(episode_values)
        episode_return_mean = (
            float(np.mean([row["return"] for row in episode_values]))
            if episode_count
            else None
        )
        episode_length_mean = (
            float(np.mean([row["length"] for row in episode_values]))
            if episode_count
            else None
        )
        window_rates = {
            "success_rate_window": 0.0,
            "collision_rate_window": 0.0,
            "dead_end_rate_window": 0.0,
            "timeout_rate_window": 0.0,
        }
        if episode_count:
            for reason, count in self._terminal_counts.items():
                window_count = sum(
                    1
                    for row in episode_values
                    if row["terminal_reason"] == reason
                )
                key = "{}_rate_window".format(reason)
                if key in window_rates:
                    window_rates[key] = float(window_count) / float(episode_count)
        nonfinite_metric_count = int(self.metrics.get("nonfinite_metric_count", 0))
        snapshot = {
            "progress": {
                "online_env_steps": float(self.online_env_steps),
                "online_transitions_committed": float(
                    self.online_transitions_committed
                ),
                "replay_size": float(replay_size),
                "completed_episodes": float(self._completed_episode_count),
                "actor_update_count": float(
                    getattr(self.learner, "actor_update_count", 0)
                ),
                "critic_update_count": float(
                    getattr(self.learner, "critic_update_count", 0)
                ),
                # The learner performs one soft target update per Critic
                # update; update_step is the existing durable owner.
                "target_update_count": float(
                    getattr(self.learner, "update_step", 0)
                ),
            },
            "actor": {
                "loss": metric("actor_loss"),
                "entropy": metric("actor_entropy"),
                "bc_kl_mean": metric("bc_kl"),
                "bc_top1_disagreement_rate": metric(
                    "bc_awac_top1_disagreement_rate"
                ),
                "parameter_delta_norm": metric("actor_parameter_delta_norm"),
                "grad_norm": metric("actor_gradient_norm"),
                "depth_grad_norm": metric("actor_depth_gradient_norm"),
                "vector_grad_norm": metric("actor_vector_gradient_norm"),
                "head_grad_norm": metric("actor_head_gradient_norm"),
            },
            "awac": {
                "advantage_mean": metric("awac_advantage_mean"),
                "advantage_std": metric("awac_advantage_std"),
                "weight_mean": metric("awac_weight_mean"),
                "weight_p50": metric("awac_weight_p50"),
                "weight_p95": metric("awac_weight_p95"),
                "weight_max": metric("awac_weight_max"),
                "weight_clip_fraction": metric("awac_weight_clip_fraction"),
                "raw_weight_ess_fraction": metric(
                    "awac_raw_weight_ess_fraction"
                ),
                # ``awac_weight_ess_fraction`` is calculated from the
                # normalized, capped weights actually used by the actor loss.
                # Keep it distinct from the raw-weight ESS and clip fraction.
                "normalized_clipped_weight_ess_fraction": metric(
                    "awac_weight_ess_fraction"
                ),
                "ess_batch_size": float(self.batch_size),
                "ess_aggregation_window_updates": float(
                    len(self._online_metric_rows)
                ),
            },
            "critic": {
                "td_loss_mean": metric("critic_td_loss"),
                "td_loss_p95": metric("critic_td_loss_p95"),
                "q_mean": metric("q_mean"),
                "q_std": metric("q_std"),
                "q_p01": metric("q_p01"),
                "q_p99": metric("q_p99"),
                "twin_q_disagreement_mean": metric(
                    "twin_q_disagreement_mean"
                ),
                "twin_q_disagreement_p95": metric(
                    "twin_q_disagreement_p95"
                ),
                "grad_norm": metric("critic_gradient_norm"),
                "cql_loss": metric("critic_cql_loss"),
                "bellman_loss": metric("critic_td_loss"),
            },
            "episode": {
                "return_mean": episode_return_mean,
                "length_mean": episode_length_mean,
                **window_rates,
            },
            "terminal": {
                "success_count": float(self._terminal_counts["success"]),
                "collision_count": float(self._terminal_counts["collision"]),
                "dead_end_count": float(self._terminal_counts["dead_end"]),
                "timeout_count": float(self._terminal_counts["timeout"]),
                "hard_altitude_count": float(
                    self._terminal_counts["hard_altitude"]
                ),
                "runtime_drop_count": float(
                    self.metrics.get("transition_drop_count", 0)
                ),
            },
            "replay": {
                "size": float(replay_size),
                "bc_calibration_rows": float(source_counts["BC_CALIBRATION"]),
                "awac_online_rows": float(source_counts["AWAC_ONLINE"]),
                "bc_calibration_fraction": float(
                    source_counts["BC_CALIBRATION"] / total_rows
                ),
                "awac_online_fraction": float(source_counts["AWAC_ONLINE"] / total_rows),
            },
            "runtime": {
                "env_steps_per_sec": float(self.online_env_steps / elapsed),
                "committed_transitions_per_sec": float(
                    self.online_transitions_committed / elapsed
                ),
                "episodes_per_min": float(self._completed_episode_count * 60.0 / elapsed),
                "actor_updates_per_sec": float(
                    getattr(self.learner, "actor_update_count", 0) / elapsed
                ),
                "critic_updates_per_sec": float(
                    getattr(self.learner, "critic_update_count", 0) / elapsed
                ),
                "transition_drop_count": float(
                    self.metrics.get("transition_drop_count", 0)
                ),
                "runtime_failure_count": float(
                    self.metrics.get("runtime_failed_attempt_count", 0)
                ),
            },
            "lr": self._tensorboard_learning_rates(),
            "health": {
                "nan_count": float(self.metrics.get("nan_count", 0)),
                "inf_count": float(self.metrics.get("inf_count", 0)),
                "nonfinite_metric_count": float(nonfinite_metric_count),
                "q_divergence": float(self.metrics.get("q_divergence", 0)),
                "actor_divergence": float(self.metrics.get("actor_divergence", 0)),
            },
            "confidence": {
                "mean": metric("confidence_mean"),
                "p05": metric("confidence_p05"),
                "p50": metric("confidence_p50"),
                "p95": metric("confidence_p95"),
                "delta_q_mean": metric("confidence_delta_q_mean"),
                "delta_q_norm_mean": metric("confidence_delta_q_norm_mean"),
                "uncertainty_mean": metric("confidence_uncertainty_mean"),
                "uncertainty_norm_mean": metric(
                    "confidence_uncertainty_norm_mean"
                ),
                "q_margin_norm_mean": metric("confidence_q_margin_norm_mean"),
                "twin_disagreement_norm_mean": metric(
                    "confidence_twin_disagreement_norm_mean"
                ),
            },
            "adaptive_bc_kl": {
                "beta_mean": metric("adaptive_bc_kl_beta_mean"),
                "beta_p05": metric("adaptive_bc_kl_beta_p05"),
                "beta_p50": metric("adaptive_bc_kl_beta_p50"),
                "beta_p95": metric("adaptive_bc_kl_beta_p95"),
                "low_confidence_beta_mean": metric(
                    "adaptive_bc_kl_low_confidence_beta_mean"
                ),
                "high_confidence_beta_mean": metric(
                    "adaptive_bc_kl_high_confidence_beta_mean"
                ),
            },
        }
        return snapshot

    def _tensorboard_learning_rates(self) -> Dict[str, float]:
        result = {
            "actor_depth": 0.0,
            "actor_vector": 0.0,
            "actor_head": 0.0,
            "critic_depth": 0.0,
            "critic_vector": 0.0,
            "critic_head": 0.0,
        }
        for prefix, optimizer in (
            ("actor", getattr(self.learner, "actor_optimizer", None)),
            ("critic", getattr(self.learner, "critic_optimizer", None)),
        ):
            for index, group in enumerate(getattr(optimizer, "param_groups", ())):
                name = str(group.get("name", "group_{}".format(index)))
                component = (
                    "depth"
                    if "depth" in name
                    else "vector"
                    if "vector" in name
                    else "head"
                )
                result["{}_{}".format(prefix, component)] = float(group["lr"])
        return result

    def _progress_snapshot(self) -> Dict[str, Any]:
        ordered_mission_ids = [mission.mission_id for mission in self.missions]
        ordered_episode_ids = [mission.episode_id for mission in self.missions]
        return {
            "schema_id": STANDARD_ONLINE_RUNTIME_SCHEMA_ID,
            "mission_source_identity": dict(self.mission_source_identity),
            "ordered_mission_ids": ordered_mission_ids,
            "ordered_episode_ids": ordered_episode_ids,
            "next_mission_index": int(self._durable_next_mission_index()),
            "completed_mission_ids": [
                mission.mission_id
                for mission in self.missions
                if mission.mission_id in self._completed_mission_ids
            ],
            "completed_episode_ids": [
                mission.episode_id
                for mission in self.missions
                if mission.mission_id in self._completed_mission_ids
            ],
            "environment_step_count": int(self._environment_step_count),
            "online_env_steps": int(self._environment_step_count),
            "online_env_steps_budget": int(self.online_env_steps_budget),
            "online_transitions_committed": int(self._online_transitions_committed),
            "completed_episode_count": int(self._completed_episode_count),
            "critic_update_count": int(getattr(self.learner, "critic_update_count", 0)),
            "actor_update_count": int(getattr(self.learner, "actor_update_count", 0)),
            "actor_optimizer_step_count": int(
                getattr(self.learner, "actor_optimizer_step_count", 0)
            ),
            "update_step": int(getattr(self.learner, "update_step", 0)),
            "phase1_state": self._phase1_snapshot(),
        }

    def _restore_mission_progress(
        self,
        progress: Mapping[str, Any],
        *,
        expected_schema: str,
        restore_online_counters: bool,
    ) -> None:
        if not isinstance(progress, Mapping):
            raise ValueError("Standard AWAC mission progress must be a mapping")
        if progress.get("mission_source_identity") != dict(self.mission_source_identity):
            raise ValueError("Standard AWAC mission source identity mismatch")
        if str(progress.get("schema_id", "")) != str(expected_schema):
            raise ValueError("Standard AWAC mission progress schema mismatch")
        completed_values = progress.get("completed_mission_ids", [])
        if not isinstance(completed_values, (list, tuple)):
            raise ValueError("Standard AWAC completed missions are invalid")
        completed = {str(value) for value in completed_values}
        if len(completed) != len(completed_values):
            raise ValueError("Standard AWAC completed missions contain duplicates")
        known = {mission.mission_id for mission in self.missions}
        ordered = progress.get("ordered_mission_ids")
        if ordered is not None and list(ordered) != [
            mission.mission_id for mission in self.missions
        ]:
            raise ValueError("Standard AWAC mission ordering identity mismatch")
        ordered_episodes = progress.get("ordered_episode_ids")
        if ordered_episodes is not None and list(ordered_episodes) != [
            mission.episode_id for mission in self.missions
        ]:
            raise ValueError("Standard AWAC episode ordering identity mismatch")
        completed_episodes = progress.get("completed_episode_ids")
        if completed_episodes is not None:
            expected_episodes = [
                mission.episode_id
                for mission in self.missions
                if mission.mission_id in completed
            ]
            if list(completed_episodes) != expected_episodes:
                raise ValueError("Standard AWAC completed episode ordering mismatch")
        if not completed.issubset(known):
            raise ValueError("Standard AWAC progress contains an unknown mission")
        next_index = int(progress.get("next_mission_index", 0))
        expected_next = len(self.missions)
        for index, mission in enumerate(self.missions):
            if mission.mission_id not in completed:
                expected_next = index
                break
        if next_index != expected_next:
            raise ValueError("Standard AWAC mission cursor is inconsistent")
        completed_count = int(progress.get("completed_episode_count", len(completed)))
        if completed_count != len(completed):
            raise ValueError("Standard AWAC completed episode count is inconsistent")
        self._completed_mission_ids = completed
        self._next_unassigned_index = next_index
        self._completed_episode_count = completed_count
        if "online_transitions_committed" not in progress:
            raise ValueError(
                "Standard AWAC progress is missing online_transitions_committed"
            )
        committed = int(progress["online_transitions_committed"])
        if committed < 0 or committed > int(progress.get("online_env_steps", self._environment_step_count)):
            raise ValueError("Standard AWAC committed transition count is invalid")
        self._online_transitions_committed = committed
        phase1 = progress.get("phase1_state")
        if isinstance(phase1, Mapping) and self.phase1_state is not None:
            if phase1.get("phase"):
                setattr(self.phase1_state, "_phase", str(phase1["phase"]))
            if "actor_update_enabled" in phase1:
                setattr(
                    self.phase1_state,
                    "_actor_update_enabled",
                    bool(phase1["actor_update_enabled"]),
                )
            if "stop_reason" in phase1:
                setattr(self.phase1_state, "_stop_reason", str(phase1["stop_reason"]))
            records = phase1.get("transition_records")
            if isinstance(records, (list, tuple)):
                setattr(self.phase1_state, "_transition_records", [dict(row) for row in records])
            milestones = phase1.get("processed_milestones")
            if isinstance(milestones, (list, tuple)):
                setattr(self.phase1_state, "_processed_milestones", set(int(value) for value in milestones))
        if restore_online_counters:
            online_steps = int(
                progress.get("online_env_steps", progress.get("environment_step_count", 0))
            )
            if online_steps < 0 or online_steps > self.online_env_steps_budget:
                raise ValueError("Standard AWAC online progress is outside its budget")
            self._environment_step_count = online_steps

    def restore_from_calibration_handoff(
        self,
        progress: Mapping[str, Any],
        *,
        exact_resume_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Resume mission cursor/RNG from the committed Calibration PASS."""

        # The source checkpoint uses the existing calibration progress schema;
        # online phase-local counters intentionally begin at zero.
        if not isinstance(progress, Mapping):
            raise ValueError("Calibration handoff mission progress is missing")
        if progress.get("mission_source_identity") != dict(self.mission_source_identity):
            raise ValueError("Calibration handoff mission source identity mismatch")
        completed_values = progress.get("completed_mission_ids", [])
        completed = {str(value) for value in completed_values}
        known = {mission.mission_id for mission in self.missions}
        if len(completed) != len(completed_values) or not completed.issubset(known):
            raise ValueError("Calibration handoff mission cursor is invalid")
        next_index = int(progress.get("next_mission_index", 0))
        expected_next = len(self.missions)
        for index, mission in enumerate(self.missions):
            if mission.mission_id not in completed:
                expected_next = index
                break
        if next_index != expected_next:
            raise ValueError("Calibration handoff mission cursor is inconsistent")
        if int(progress.get("completed_episode_count", len(completed))) != len(completed):
            raise ValueError("Calibration handoff episode count is inconsistent")
        self._completed_mission_ids = completed
        self._next_unassigned_index = next_index
        self._completed_episode_count = len(completed)
        self._environment_step_count = 0
        self._online_transitions_committed = 0
        if exact_resume_state is not None:
            from planning.awac.checkpoint import restore_calibration_exact_resume_state

            restore_calibration_exact_resume_state(
                exact_resume_state,
                producer_rng=self.rng,
                torch=self.torch,
            )

    def restore_progress(
        self,
        progress: Mapping[str, Any],
        *,
        exact_resume_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._restore_mission_progress(
            progress,
            expected_schema=STANDARD_ONLINE_RUNTIME_SCHEMA_ID,
            restore_online_counters=True,
        )
        if exact_resume_state is not None:
            if exact_resume_state.get("mission_progress") != progress:
                raise ValueError("Standard AWAC exact resume progress mismatch")
            from planning.awac.checkpoint import restore_standard_online_exact_resume_state

            restore_standard_online_exact_resume_state(
                exact_resume_state,
                producer_rng=self.rng,
                torch=self.torch,
            )
        self._next_tensorboard_step = (
            (self.online_env_steps // self.tensorboard_log_interval_steps) + 1
        ) * self.tensorboard_log_interval_steps

    def _current_replay_identity(self) -> Dict[str, Any]:
        """Return the replay identity that a checkpoint transaction will bind.

        ``calibration_replay_identity`` includes the hash of serialized replay
        metadata. The handoff identity is stale after an online append unless
        metadata is flushed at the checkpoint boundary. Small injected test
        doubles do not expose persistent replay metadata, so they retain the
        counter-only identity used by the runtime unit tests.
        """

        current = {
            **dict(self.replay_identity),
            "replay_size": int(getattr(self.replay, "size", 0)),
            "replay_total_added": int(
                getattr(self.replay, "total_added", getattr(self.replay, "size", 0))
            ),
        }
        if hasattr(self.replay, "metadata") and hasattr(self.replay, "directory"):
            flush = getattr(self.replay, "flush", None)
            if not callable(flush):
                raise CalibrationRuntimeError(
                    "persistent Standard AWAC replay has no flush owner"
                )
            flush()
            from planning.awac.checkpoint import calibration_replay_identity

            return calibration_replay_identity(self.replay)
        return current

    def checkpoint_snapshot(self, status: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        state = dict(status or {})
        progress = self._progress_snapshot()
        replay_identity = self._current_replay_identity()
        exact = build_standard_online_exact_resume_state(
            learner_state=self.learner.state_dict(),
            producer_rng=self.rng,
            torch=self.torch,
            mission_source_identity=self.mission_source_identity,
            mission_progress=progress,
            runtime_identity=dict(self._runtime_ids),
            phase1_state=self._phase1_snapshot(),
            replay_identity=replay_identity,
        )
        return {
            "status": str(state.get("status", state.get("state", "RUNNING"))),
            "stop_reason": str(self._stop_reason),
            "environment_step_count": int(self._environment_step_count),
            "online_env_steps": int(self._environment_step_count),
            "online_transitions_committed": int(self._online_transitions_committed),
            "completed_episode_count": int(self._completed_episode_count),
            "replay_size": int(getattr(self.replay, "size", 0)),
            "replay_total_added": int(
                getattr(self.replay, "total_added", getattr(self.replay, "size", 0))
            ),
            "starting_replay_size": int(self._starting_replay_size),
            "starting_replay_total_added": int(self._starting_replay_total_added),
            "critic_update_count": int(getattr(self.learner, "critic_update_count", 0)),
            "actor_update_count": int(getattr(self.learner, "actor_update_count", 0)),
            "actor_optimizer_step_count": int(
                getattr(self.learner, "actor_optimizer_step_count", 0)
            ),
            "update_step": int(getattr(self.learner, "update_step", 0)),
            "progress": progress,
            "runtime_identity": dict(self._runtime_ids),
            "replay_identity": replay_identity,
            "phase1_state": self._phase1_snapshot(),
            "actor_state_before": self._actor_before,
            "actor_state_after": _state_fingerprint(self.learner),
            "standard_online_resume_state": exact,
            "learner_metrics": _finite_metrics(self._online_metric_rows),
            "runtime_metrics": dict(self.metrics),
        }

    def _maybe_checkpoint(self) -> None:
        if self.on_checkpoint is None:
            return
        if int(self._environment_step_count) < int(self._next_checkpoint_step):
            return
        snapshot = self.checkpoint_snapshot({"status": "RUNNING"})
        snapshot["checkpoint_reason"] = "interval"
        self.on_checkpoint(self, snapshot)
        while self._next_checkpoint_step <= int(self._environment_step_count):
            self._next_checkpoint_step += self.checkpoint_interval_steps

    def _maybe_tensorboard(self, *, force: bool = False) -> None:
        if self.tensorboard_logger is None:
            return
        if not force and int(self.online_env_steps) < int(self._next_tensorboard_step):
            return
        try:
            self.tensorboard_logger.log_runner(self, force=True)
            while self._next_tensorboard_step <= int(self.online_env_steps):
                self._next_tensorboard_step += self.tensorboard_log_interval_steps
        except Exception as error:
            # Observability failure must be explicit but must not turn an
            # already committed replay/model update into a different result.
            self.metrics["tensorboard_error"] = "{}: {}".format(
                type(error).__name__, str(error)
            )
            self.tensorboard_logger = None

    def run(self) -> Dict[str, Any]:
        """Collect until the explicit online budget or mission exhaustion."""

        try:
            self._ready()
            while True:
                if int(self._environment_step_count) >= self.online_env_steps_budget:
                    self._stop_requested = True
                    self._stop_reason = "ONLINE_ENV_STEP_BUDGET_REACHED"
                    self._active.clear()
                    break
                self._assign_missions()
                if not self._active:
                    if self._next_unassigned_index >= len(self.missions):
                        self._stop_reason = self._stop_reason or "MISSION_SOURCE_EXHAUSTED"
                        break
                    continue
                if (
                    int(self._environment_step_count) + len(self._active)
                    > self.online_env_steps_budget
                ):
                    self._stop_requested = True
                    self._stop_reason = "ONLINE_ENV_STEP_BUDGET_REACHED"
                    self._active.clear()
                    break
                try:
                    self._step_once()
                except Exception as error:
                    self._record_failure(error)
                    raise CalibrationRuntimeError(str(error)) from error
                self._maybe_checkpoint()
                self._maybe_tensorboard()
            if self._stop_reason == "ONLINE_ENV_STEP_BUDGET_REACHED":
                status = "ONLINE_ENV_STEP_BUDGET_REACHED"
            else:
                status = "MISSION_SOURCE_EXHAUSTED"
            actor_after = _state_fingerprint(self.learner)
            self._maybe_tensorboard(force=True)
            replay_identity = self._current_replay_identity()
            result = {
                "status": status,
                "stop_reason": self._stop_reason,
                "online_env_steps": int(self._environment_step_count),
                "online_env_steps_budget": int(self.online_env_steps_budget),
                "environment_step_count": int(self._environment_step_count),
                "completed_episode_count": int(self._completed_episode_count),
                "next_mission_index": int(self._durable_next_mission_index()),
                "replay_size": int(getattr(self.replay, "size", 0)),
                "replay_total_added": int(
                    getattr(self.replay, "total_added", getattr(self.replay, "size", 0))
                ),
                "online_transitions_committed": int(self._online_transitions_committed),
                "critic_update_count": int(getattr(self.learner, "critic_update_count", 0)),
                "actor_update_count": int(getattr(self.learner, "actor_update_count", 0)),
                "actor_optimizer_step_count": int(
                    getattr(self.learner, "actor_optimizer_step_count", 0)
                ),
                "actor_state_before": self._actor_before,
                "actor_state_after": actor_after,
                "actor_changed": self._actor_before != actor_after,
                "progress": self._progress_snapshot(),
                "runtime_identity": dict(self._runtime_ids),
                "replay_identity": replay_identity,
                "starting_replay_size": int(self._starting_replay_size),
                "starting_replay_total_added": int(
                    self._starting_replay_total_added
                ),
                "metrics": {
                    **dict(self.metrics),
                    "learner": _finite_metrics(self._online_metric_rows),
                },
            }
            return result
        except CalibrationRuntimeError:
            raise
        except Exception as error:
            self._record_failure(error)
            raise CalibrationRuntimeError(str(error)) from error
        finally:
            self._stop_requested = True
            self._safe_stop_and_close()


def build_standard_online_summary(
    *,
    result: Mapping[str, Any],
    phase: str,
    start_checkpoint_identity: Mapping[str, Any],
    starting_replay_size: int,
    ending_replay_size: int,
    behavior_source_counts: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    checkpoint_paths: Sequence[str],
) -> Dict[str, Any]:
    """Build the canonical typed Standard online summary artifact."""

    if not isinstance(result, Mapping):
        raise TypeError("Standard AWAC result must be a mapping")
    summary = {
        "schema_id": STANDARD_ONLINE_SUMMARY_SCHEMA_ID,
        "phase": str(phase),
        "start_checkpoint_identity": dict(start_checkpoint_identity),
        "starting_replay_size": int(starting_replay_size),
        "ending_replay_size": int(ending_replay_size),
        "online_transitions": int(result["online_transitions_committed"]),
        "online_transitions_committed": int(result["online_transitions_committed"]),
        "online_env_steps": int(result.get("online_env_steps", 0)),
        "online_env_steps_budget": int(result.get("online_env_steps_budget", 0)),
        "completed_episodes": int(result.get("completed_episode_count", 0)),
        "bc_calibration_rows": int(behavior_source_counts.get("BC_CALIBRATION", 0)),
        "awac_online_rows": int(behavior_source_counts.get("AWAC_ONLINE", 0)),
        "behavior_source_counts": dict(behavior_source_counts),
        "reliable_exact_observation_contract": "reliable_exact_endpoint_snapshot",
        "nan_inf_count": int(
            result.get("metrics", {}).get("nan_inf_count", 0)
        ),
        "nonfinite_metric_count": int(
            result.get("metrics", {}).get("nonfinite_metric_count", 0)
        ),
        "online_budget_counter_owner": "phase_local_online_environment_steps",
        "checkpoint_retention": 2,
        "actor_updates": int(result.get("actor_update_count", 0)),
        "critic_updates": int(result.get("critic_update_count", 0)),
        "actor_state_before_sha256": str(result.get("actor_state_before", "")),
        "actor_state_after_sha256": str(result.get("actor_state_after", "")),
        "learner_metrics": dict(result.get("metrics", {}).get("learner", {})),
        "tensorboard": dict(result.get("tensorboard", {})),
        "runtime_failure_summary": {
            "count": int(
                result.get("metrics", {}).get("runtime_failed_attempt_count", 0)
            ),
            "reasons": list(
                result.get("metrics", {}).get("runtime_failure_reasons", [])
            ),
        },
        "runtime_identity": dict(runtime_identity),
        "checkpoint_paths": [str(value) for value in checkpoint_paths],
        "cleanup": {
            "replay_final_flush": str(
                result.get("metrics", {}).get("replay_final_flush", "UNKNOWN")
            ),
            "runtime_close": str(
                result.get("metrics", {}).get("runtime_close", "UNKNOWN")
            ),
        },
        "stop_reason": str(result.get("stop_reason", "")),
    }
    validate_standard_online_summary(summary)
    return summary


def validate_standard_online_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the stable fields consumed by replay-audit/report tooling."""

    if not isinstance(summary, Mapping):
        raise ValueError("Standard AWAC summary must be a mapping")
    if summary.get("schema_id") != STANDARD_ONLINE_SUMMARY_SCHEMA_ID:
        raise ValueError("Standard AWAC summary schema mismatch")
    if summary.get("phase") != "awac_training":
        raise ValueError("Standard AWAC summary phase mismatch")
    for name in (
        "phase",
        "start_checkpoint_identity",
        "starting_replay_size",
        "ending_replay_size",
        "online_transitions",
        "online_transitions_committed",
        "online_env_steps",
        "online_env_steps_budget",
        "completed_episodes",
        "bc_calibration_rows",
        "awac_online_rows",
        "actor_updates",
        "critic_updates",
        "actor_state_before_sha256",
        "actor_state_after_sha256",
        "learner_metrics",
        "runtime_failure_summary",
        "runtime_identity",
        "checkpoint_paths",
        "cleanup",
        "stop_reason",
        "reliable_exact_observation_contract",
        "nan_inf_count",
        "nonfinite_metric_count",
        "online_budget_counter_owner",
        "checkpoint_retention",
    ):
        if name not in summary:
            raise ValueError("Standard AWAC summary missing {}".format(name))
    for name in (
        "starting_replay_size",
        "ending_replay_size",
        "online_transitions",
        "online_transitions_committed",
        "online_env_steps",
        "completed_episodes",
        "bc_calibration_rows",
        "awac_online_rows",
        "actor_updates",
        "critic_updates",
        "nan_inf_count",
        "nonfinite_metric_count",
        "checkpoint_retention",
    ):
        if int(summary[name]) < 0:
            raise ValueError("Standard AWAC summary {} is negative".format(name))
    if int(summary["awac_online_rows"]) != int(summary["online_transitions"]):
        raise ValueError("Standard AWAC summary online row count mismatch")
    if int(summary["online_transitions_committed"]) != int(
        summary["online_transitions"]
    ):
        raise ValueError("Standard AWAC summary committed transition count mismatch")
    if int(summary["online_env_steps_budget"]) <= 0:
        raise ValueError("Standard AWAC summary online budget is invalid")
    if int(summary["online_transitions"]) > int(summary["online_env_steps"]):
        raise ValueError("Standard AWAC summary online transitions exceed steps")
    if int(summary["online_env_steps"]) > int(summary["online_env_steps_budget"]):
        raise ValueError("Standard AWAC summary exceeded its online budget")
    if summary["reliable_exact_observation_contract"] != (
        "reliable_exact_endpoint_snapshot"
    ):
        raise ValueError("Standard AWAC summary observation contract mismatch")
    if int(summary["nan_inf_count"]) != 0 or int(summary["nonfinite_metric_count"]) != 0:
        raise ValueError("Standard AWAC summary contains non-finite diagnostics")
    if summary["online_budget_counter_owner"] != (
        "phase_local_online_environment_steps"
    ):
        raise ValueError("Standard AWAC summary online budget owner mismatch")
    if int(summary["checkpoint_retention"]) != 2:
        raise ValueError("Standard AWAC summary checkpoint retention mismatch")
    start_identity = summary["start_checkpoint_identity"]
    if not isinstance(start_identity, Mapping) or not start_identity:
        raise ValueError("Standard AWAC summary checkpoint identity is missing")
    if not str(start_identity.get("path", "")).strip():
        raise ValueError("Standard AWAC summary checkpoint path is missing")
    if len(str(start_identity.get("sha256", ""))) != 64:
        raise ValueError("Standard AWAC summary checkpoint SHA is invalid")
    if not isinstance(summary["behavior_source_counts"], Mapping):
        raise ValueError("Standard AWAC summary behavior counts are invalid")
    for source in ("BC_CALIBRATION", "AWAC_ONLINE"):
        if source not in summary["behavior_source_counts"]:
            raise ValueError(
                "Standard AWAC summary behavior source {} is missing".format(source)
            )
        if int(summary["behavior_source_counts"][source]) < 0:
            raise ValueError(
                "Standard AWAC summary behavior source {} is negative".format(source)
            )
    if int(summary["behavior_source_counts"]["BC_CALIBRATION"]) != int(
        summary["bc_calibration_rows"]
    ):
        raise ValueError("Standard AWAC summary BC row count mismatch")
    if int(summary["behavior_source_counts"]["AWAC_ONLINE"]) != int(
        summary["awac_online_rows"]
    ):
        raise ValueError("Standard AWAC summary online source count mismatch")
    if not isinstance(summary["runtime_identity"], Mapping) or not summary[
        "runtime_identity"
    ]:
        raise ValueError("Standard AWAC summary runtime identity is missing")
    if not isinstance(summary["checkpoint_paths"], (list, tuple)):
        raise ValueError("Standard AWAC summary checkpoint paths are invalid")
    if not isinstance(summary["learner_metrics"], Mapping):
        raise ValueError("Standard AWAC summary learner metrics are invalid")
    if not isinstance(summary["runtime_failure_summary"], Mapping):
        raise ValueError("Standard AWAC summary runtime failure summary is invalid")
    if not isinstance(summary["cleanup"], Mapping):
        raise ValueError("Standard AWAC summary cleanup is invalid")
    return dict(summary)


__all__ = [
    "STANDARD_ONLINE_RUNTIME_SCHEMA_ID",
    "STANDARD_ONLINE_SUMMARY_SCHEMA_ID",
    "StandardAWACOnlineRunner",
    "build_standard_online_summary",
    "validate_standard_online_summary",
]
