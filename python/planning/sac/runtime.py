"""Bounded online runtime for the BC-initialized discrete SAC mainline.

The reliable-exact environment lifecycle is intentionally reused from the
existing runtime owner.  This module owns only SAC action selection, episode
commit, critic/actor scheduling, and SAC-specific diagnostics; it does not
call the AWAC learner or any historical calibration replay.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from planning.awac.calibration_runtime import (
    CalibrationReplayProducer,
    calibration_episode_rng,
)
from planning.evaluation.policy_evaluator import observation_tensors
from planning.sac.contract import SAC_POLICY_ID
from planning.sac.contract import SAC_KL_BACKTRACKING_FACTORS
from planning.common.atomic import write_json_atomic
from planning.sac.diagnostics import (
    SACUpdateJournal,
    action_mask_sha256,
    batch_indices_sha256,
    capture_rng_state,
    restore_rng_state,
    rng_fingerprint,
    stable_fingerprint,
)


class SACSafetyStop(RuntimeError):
    """Fail-closed stop caused by a non-finite or collapsed SAC signal."""


class SACOnlineRunner(CalibrationReplayProducer):
    """Run reliable-exact episodes and train a fresh online SAC agent."""

    def __init__(
        self,
        *,
        missions: Sequence[Any],
        pool: Any,
        replay: Any,
        agent: Any,
        normalizer: Any,
        torch: Any,
        device: Any,
        seed: int,
        batch_size: int,
        learning_starts: int,
        critic_updates_per_transition: float,
        actor_updates_per_transition: float,
        max_env_steps: int,
        max_episodes: int,
        expected_runtime_ids: Optional[Mapping[int, str]] = None,
        output_dir: Optional[Path] = None,
        snapshot_callback: Optional[Callable[["SACOnlineRunner"], None]] = None,
        bc_kl_hard_stop: float = 1.0,
        entropy_floor: float = 0.02,
        journal: Optional[SACUpdateJournal] = None,
    ) -> None:
        super().__init__(
            missions=missions,
            pool=pool,
            replay=replay,
            learner=agent,
            train_episode_ids=None,
            holdout_episode_ids=None,
            torch=torch,
            normalizer=normalizer,
            device=device,
            batch_size=int(batch_size),
            learning_starts=int(learning_starts),
            updates_per_step=1.0,
            max_transitions=int(max_env_steps),
            max_episodes=int(max_episodes),
            ready_timeout_s=60.0,
            reset_timeout_s=30.0,
            step_timeout_s=30.0,
            reset_settle=0.30,
            rng=np.random.RandomState(int(seed)),
            behavior_policy=None,
            mission_source_identity={},
            expected_runtime_ids=expected_runtime_ids,
            output_dir=output_dir,
        )
        self.agent = agent
        self.seed = int(seed)
        self.critic_updates_per_transition = float(critic_updates_per_transition)
        self.actor_updates_per_transition = float(actor_updates_per_transition)
        self.bc_kl_hard_stop = float(bc_kl_hard_stop)
        self.entropy_floor = float(entropy_floor)
        if self.critic_updates_per_transition <= 0.0 or not math.isfinite(
            self.critic_updates_per_transition
        ):
            raise ValueError("critic_updates_per_transition must be positive and finite")
        if self.actor_updates_per_transition <= 0.0 or not math.isfinite(
            self.actor_updates_per_transition
        ):
            raise ValueError("actor_updates_per_transition must be positive and finite")
        if self.bc_kl_hard_stop <= 0.0 or not math.isfinite(self.bc_kl_hard_stop):
            raise ValueError("bc_kl_hard_stop must be positive and finite")
        if self.entropy_floor < 0.0 or not math.isfinite(self.entropy_floor):
            raise ValueError("entropy_floor must be finite and non-negative")
        self.behavior_policy = {
            "policy_id": str(agent.actor_policy_id),
            "selection_mode": "masked_categorical",
            "temperature": 1.0,
            "mask_contract": "depth_action_mask",
            "rng_owner": "sac_online_runner_numpy_randomstate",
            "rng_scope": "per_episode_per_worker",
            "rng_algorithm": "numpy.random.RandomState",
            "rng_derivation": "sha256_base_seed_worker_episode_mission_uint32_v1",
            "checkpoint_sha256": str(
                replay.metadata.get("source_bc_checkpoint_sha256", "")
            ),
            "warmup_policy": "frozen_bc_until_learning_starts",
        }
        self._behavior_rng_seed = None
        self.training_rows = []
        self.episode_summaries = []
        self.snapshot_callback = snapshot_callback
        self.first_actor_update_at_transition: Optional[int] = None
        self.stop_reason = ""
        self._last_snapshot_environment_steps = -1
        self.update_journal = journal
        self.transition_commit_order = []
        self.sentinel_frozen = False
        self.sentinel_manifest = {}
        self.sentinel_manifest_sha256 = ""
        self.max_sentinel_kl = 0.0
        self.trust_region_factor_counts = {
            str(factor): 0 for factor in SAC_KL_BACKTRACKING_FACTORS
        }
        self.agent.set_actor_rng_hooks(
            lambda: capture_rng_state(
                self.torch, self.rng, worker_state=self.worker_state_snapshot()
            ),
            lambda state: restore_rng_state(self.torch, self.rng, state),
        )

    def worker_state_snapshot(self) -> Mapping[str, Any]:
        result = {}
        for worker_id, state in sorted(self._active.items()):
            policy_rng = getattr(state, "policy_rng", None)
            result[str(worker_id)] = {
                "worker_id": int(worker_id),
                "mission_id": str(state.mission.mission_id),
                "episode_id": str(state.mission.episode_id),
                "step_count": int(state.step_count),
                "previous_action": int(state.previous_action),
                "action_mask": np.asarray(state.action_mask, dtype=np.uint8).copy(),
                "policy_rng_state": policy_rng.get_state() if policy_rng is not None else None,
                "policy_rng_identity": dict(getattr(state, "policy_rng_identity", {}) or {}),
                "pending_transition_ids": [
                    str(row.get("transition_id", "")) for row in state.pending_transitions
                ],
            }
        return result

    def runtime_state_snapshot(self) -> Mapping[str, Any]:
        return {
            "schema_id": "bc_initialized_discrete_sac_runtime_state_v2",
            "environment_step_count": int(self._environment_step_count),
            "completed_episode_count": int(self._completed_episode_count),
            "next_unassigned_index": int(getattr(self, "_next_unassigned_index", 0)),
            "completed_mission_ids": sorted(str(value) for value in self._completed_mission_ids),
            "runtime_ids": {str(key): str(value) for key, value in self._runtime_ids.items()},
            "episode_rng_assignments": self._episode_rng_assignment_snapshot(),
            "active_workers": self.worker_state_snapshot(),
            "transition_commit_order_count": len(self.transition_commit_order),
            "transition_commit_order_sha256": stable_fingerprint(self.transition_commit_order),
            "stop_reason": str(self.stop_reason),
            "sentinel_frozen": bool(self.sentinel_frozen),
            "sentinel_manifest_sha256": str(self.sentinel_manifest_sha256),
            "sentinel_state_count": int(self.sentinel_manifest.get("state_count", 0)),
            "max_sentinel_kl": float(self.max_sentinel_kl),
            "trust_region_factor_counts": dict(self.trust_region_factor_counts),
        }

    def transition_order_snapshot(self) -> Mapping[str, Any]:
        return {
            "schema_id": "bc_initialized_discrete_sac_transition_order_v2",
            "count": len(self.transition_commit_order),
            "transition_ids": list(self.transition_commit_order),
            "sha256": stable_fingerprint(self.transition_commit_order),
        }

    def _journal_update(
        self,
        *,
        update_kind: str,
        update_index: int,
        replay_size: int,
        indices: np.ndarray,
        batch: Mapping[str, Any],
        rng_before: Mapping[str, Any],
        rng_after: Mapping[str, Any],
        state_before: str,
        state_after: str,
        optimizer_before: str,
        optimizer_after: str,
        metrics: Mapping[str, Any],
    ) -> None:
        if self.update_journal is None:
            return
        values = np.asarray(indices, dtype=np.int64).reshape(-1)
        max_position = None
        if "bc_kl_post_max" in metrics:
            max_position = int(metrics.get("bc_kl_max_position", 0))
        transition_id = None
        if max_position is not None and 0 <= max_position < values.size:
            transition_id = str(
                self.replay.identity_records[int(values[max_position])]["transition_id"]
            )
        record = {
            "update_kind": str(update_kind),
            "update_index": int(update_index),
            "environment_steps": int(self._environment_step_count),
            "completed_episodes": int(self._completed_episode_count),
            "replay_size": int(replay_size),
            "batch_indices": values.tolist(),
            "batch_indices_sha256": batch_indices_sha256(values),
            "action_mask_sha256": action_mask_sha256(batch["action_mask"].detach().cpu().numpy()),
            "max_kl_batch_position": max_position,
            "max_kl_transition_id": transition_id,
            "model_state_before_sha256": state_before,
            "model_state_after_sha256": state_after,
            "optimizer_state_before_sha256": optimizer_before,
            "optimizer_state_after_sha256": optimizer_after,
            "rng_before_fingerprint": rng_fingerprint(rng_before),
            "rng_after_fingerprint": rng_fingerprint(rng_after),
            "replayable_rng_before_fingerprint": rng_fingerprint(
                {key: value for key, value in rng_before.items() if key != "workers"}
            ),
            "replayable_rng_after_fingerprint": rng_fingerprint(
                {key: value for key, value in rng_after.items() if key != "workers"}
            ),
            "metrics": dict(metrics),
            "pre_step_kl_used": bool(metrics.get("pre_step_kl_used", False)),
            "post_step_hard_stop_checked": bool(metrics.get("post_step_hard_stop_checked", False)),
            "rollback_performed": bool(metrics.get("rollback_performed", False)),
        }
        self.update_journal.append(record)

    def _assign_missions(self) -> None:
        before = set(self._active)
        super()._assign_missions()
        for worker_id, state in self._active.items():
            if worker_id in before or state.policy_rng is not None:
                continue
            policy_rng, identity = calibration_episode_rng(
                base_seed=self.seed,
                worker_id=int(worker_id),
                episode_id=state.mission.episode_id,
                mission_id=state.mission.mission_id,
            )
            state.policy_rng = policy_rng
            state.policy_rng_identity = identity
            self._episode_rng_assignments[state.mission.mission_id] = dict(identity)

    def _policy_action(self, state: Any) -> int:
        if state.policy_rng is None:
            raise SACSafetyStop("SAC episode RNG was not assigned")
        depth, vector = observation_tensors(
            dict(state.observation),
            int(state.previous_action),
            self.normalizer,
            self.torch,
            self.device,
            state.depth_history,
        )
        mask = self.torch.from_numpy(
            np.asarray(state.action_mask, dtype=np.bool_).reshape(1, -1)
        ).to(device=self.device, dtype=self.torch.bool)
        warmup = int(self.replay.total_added) < int(self.learning_starts)
        diagnostic = self.agent.select_action(
            depth,
            vector,
            mask,
            state.policy_rng,
            warmup=warmup,
        )
        state.sac_action_diagnostics = dict(diagnostic)
        return int(diagnostic["action"])

    def _make_transition_for_worker(
        self, worker_id: int, state: Any, result: Mapping[str, Any], action: int
    ):
        transition, next_observation, next_mask, reason = super()._make_transition_for_worker(
            worker_id, state, result, action
        )
        diagnostic = dict(getattr(state, "sac_action_diagnostics", {}))
        runtime_id = str(self._runtime_ids[int(worker_id)])
        step_id = int(state.step_count)
        transition.update(
            {
                "mission_id": str(state.mission.mission_id),
                "episode_id": str(state.mission.episode_id),
                "step_id": step_id,
                "behavior_policy_version": str(
                    diagnostic.get("behavior_policy_version", SAC_POLICY_ID)
                ),
                "termination_reason": str(reason) if bool(transition["done"]) else "",
                "runtime_instance_id": runtime_id,
                "transition_id": "{}:{}:{}".format(
                    runtime_id, state.mission.episode_id, step_id
                ),
                "actor_log_prob": float(diagnostic.get("log_prob", 0.0)),
                "policy_entropy": float(diagnostic.get("entropy", 0.0)),
                "bc_kl": float(diagnostic.get("bc_kl", 0.0)),
                "realized_return": 0.0,
                "argmax_flip": bool(diagnostic.get("argmax_flip", 0)),
                "valid_action_count": int(
                    diagnostic.get("valid_action_count", np.asarray(state.action_mask).sum())
                ),
                "action_mask_valid": True,
            }
        )
        return transition, next_observation, next_mask, reason

    def _append_episode(self, state: Any) -> None:
        rows = list(state.pending_transitions)
        if not rows:
            raise SACSafetyStop("terminal SAC episode has no transitions")
        future = 0.0
        for row in reversed(rows):
            reward = float(row["reward"])
            if bool(row["done"]):
                future = reward
            else:
                future = reward + float(self.agent.gamma) * future
            row["realized_return"] = float(future)
        self.replay.add_batch(rows)
        self.transition_commit_order.extend(
            str(row["transition_id"]) for row in rows
        )
        terminal_reason = str(rows[-1].get("termination_reason", ""))
        episode_return = float(sum(float(row["reward"]) for row in rows))
        self.episode_summaries.append(
            {
                "mission_id": str(state.mission.mission_id),
                "episode_id": str(state.mission.episode_id),
                "steps": int(len(rows)),
                "episode_return": episode_return,
                "termination_reason": terminal_reason,
                "success": terminal_reason == "success",
                "collision": terminal_reason == "collision",
                "dead_end": terminal_reason == "dead_end",
                "timeout": terminal_reason == "timeout",
            }
        )

    def _check_metrics(self, metrics: Mapping[str, Any], *, actor: bool) -> None:
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise SACSafetyStop("non-finite SAC {} metric: {}".format("actor" if actor else "critic", name))
        if actor:
            safety_kl = float(
                metrics.get("sentinel_kl_max", metrics.get("bc_kl_max", 0.0))
            )
            # Keep this helper usable in focused tests that construct a runner
            # without invoking __init__, while preserving the normal runtime
            # accumulator initialized by SACOnlineRunner.__init__.
            self.max_sentinel_kl = max(
                getattr(self, "max_sentinel_kl", 0.0),
                safety_kl,
            )
            if safety_kl >= self.bc_kl_hard_stop:
                raise SACSafetyStop("BC KL hard stop exceeded")
            entropy_checks = int(metrics.get("entropy_floor_eligible_count", 0))
            if entropy_checks > 0 and float(metrics.get("entropy_min", self.entropy_floor)) < self.entropy_floor:
                raise SACSafetyStop("policy entropy floor violated")

    def _record_update(self, kind: str, metrics: Mapping[str, Any]) -> None:
        row = {
            "environment_steps": int(self._environment_step_count),
            "completed_episodes": int(self._completed_episode_count),
            "replay_size": int(self.replay.size),
            "actor_optimizer_steps": int(self.agent.actor_optimizer_step_count),
            "critic_optimizer_steps": int(self.agent.critic_optimizer_step_count),
            "actor_update_count": int(self.agent.actor_update_count),
            "critic_update_count": int(self.agent.critic_update_count),
            "update_kind": str(kind),
        }
        row.update({str(key): value for key, value in metrics.items()})
        self.training_rows.append(row)
        if kind.startswith("actor"):
            for result in metrics.get("factor_results", ()):
                factor = str(result.get("factor"))
                if factor in self.trust_region_factor_counts:
                    self.trust_region_factor_counts[factor] += 1
            value = metrics.get("sentinel_kl_max")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                self.max_sentinel_kl = max(self.max_sentinel_kl, float(value))

    @staticmethod
    def _state_sha256(replay: Any, row: int) -> str:
        digest = hashlib.sha256()
        for name in ("depth", "vector", "action_mask"):
            value = np.ascontiguousarray(replay.arrays[name][int(row)])
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(value.shape).encode("utf-8"))
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()

    @staticmethod
    def _mask_sha256(mask: Any) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(np.asarray(mask, dtype=np.uint8)).tobytes(order="C")
        ).hexdigest()

    def _freeze_kl_sentinel(self, committed: int) -> None:
        if self.sentinel_frozen:
            return
        if committed < int(self.learning_starts):
            return
        rows = []
        manifest_rows = []
        seen = set()
        for row in range(int(committed)):
            state_sha = self._state_sha256(self.replay, row)
            if state_sha in seen:
                continue
            seen.add(state_sha)
            identity = dict(self.replay.identity_records[row])
            rows.append(int(row))
            manifest_rows.append(
                {
                    "replay_row": int(row),
                    "mission_id": str(identity.get("mission_id", "")),
                    "episode_id": str(identity.get("episode_id", "")),
                    "step_id": int(identity.get("step_id", -1)),
                    "transition_id": str(identity.get("transition_id", "")),
                    "state_sha256": state_sha,
                    "action_mask_sha256": self._mask_sha256(self.replay.arrays["action_mask"][row]),
                }
            )
        if not rows:
            raise SACSafetyStop("cannot freeze an empty KL sentinel")
        indices = np.asarray(rows, dtype=np.int64)
        batch = self.replay.batch_from_indices(indices, torch=self.torch, device=self.device)
        self.agent.set_kl_sentinel(batch)
        self.sentinel_manifest = {
            "schema_id": "bc_initialized_discrete_sac_sentinel_manifest_v4",
            "selection": "unique_pre_actor_committed_observation_states",
            "return_or_q_selection": False,
            "fixed_after_first_actor": True,
            "source_replay_size": int(committed),
            "state_count": len(manifest_rows),
            "rows": manifest_rows,
        }
        encoded = json.dumps(
            self.sentinel_manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.sentinel_manifest_sha256 = hashlib.sha256(encoded).hexdigest()
        if self.output_dir is not None:
            write_json_atomic(
                Path(self.output_dir) / "sentinel_manifest.json",
                self.sentinel_manifest,
                trailing_newline=True,
            )
        self.sentinel_frozen = True

    def _update_critics(self) -> None:
        committed = int(self.replay.size)
        if committed < int(self.learning_starts) or committed < int(self.batch_size):
            return
        self._freeze_kl_sentinel(committed)
        eligible = max(0, committed - int(self.learning_starts) + 1)
        target_critic_updates = int(
            math.floor(eligible * self.critic_updates_per_transition)
        )
        while int(self.agent.critic_update_count) < target_critic_updates:
            indices = self.replay.sample_indices(int(self.batch_size), rng=self.rng)
            batch = self.replay.batch_from_indices(indices, torch=self.torch, device=self.device)
            rng_before = capture_rng_state(
                self.torch, self.rng, worker_state=self.worker_state_snapshot()
            )
            state_before = stable_fingerprint(self.agent.state_payload())
            optimizer_before = stable_fingerprint(self.agent.critic_optimizer.state_dict())
            metrics = self.agent.update_critic(batch)
            self._check_metrics(metrics, actor=False)
            self._journal_update(
                update_kind="critic",
                update_index=int(self.agent.critic_update_count),
                replay_size=committed,
                indices=indices,
                batch=batch,
                rng_before=rng_before,
                rng_after=capture_rng_state(
                    self.torch, self.rng, worker_state=self.worker_state_snapshot()
                ),
                state_before=state_before,
                state_after=stable_fingerprint(self.agent.state_payload()),
                optimizer_before=optimizer_before,
                optimizer_after=stable_fingerprint(self.agent.critic_optimizer.state_dict()),
                metrics=metrics,
            )
            self._record_update("critic", metrics)
        target_actor_updates = int(
            math.floor(eligible * self.actor_updates_per_transition)
        )
        while int(self.agent.actor_update_count) < target_actor_updates:
            indices = self.replay.sample_indices(int(self.batch_size), rng=self.rng)
            batch = self.replay.batch_from_indices(indices, torch=self.torch, device=self.device)
            rng_before = capture_rng_state(
                self.torch, self.rng, worker_state=self.worker_state_snapshot()
            )
            state_before = stable_fingerprint(self.agent.state_payload())
            optimizer_before = stable_fingerprint(self.agent.actor_optimizer.state_dict())
            metrics = self.agent.update_actor_transactional(
                batch,
                hard_stop=self.bc_kl_hard_stop,
                entropy_floor=self.entropy_floor,
                trust_region_factors=SAC_KL_BACKTRACKING_FACTORS,
            )
            metrics = dict(metrics)
            metrics.update(self.agent.residual_statistics(batch))
            self._journal_update(
                update_kind="actor" if metrics.get("accepted") else "actor_rejected",
                update_index=int(self.agent.actor_proposal_count),
                replay_size=committed,
                indices=indices,
                batch=batch,
                rng_before=rng_before,
                rng_after=capture_rng_state(
                    self.torch, self.rng, worker_state=self.worker_state_snapshot()
                ),
                state_before=state_before,
                state_after=stable_fingerprint(self.agent.state_payload()),
                optimizer_before=optimizer_before,
                optimizer_after=stable_fingerprint(self.agent.actor_optimizer.state_dict()),
                metrics=metrics,
            )
            self._record_update(
                "actor" if metrics.get("accepted") else "actor_rejected", metrics
            )
            if not bool(metrics.get("accepted")):
                raise SACSafetyStop(
                    "SAC Actor proposal rejected: {}".format(
                        metrics.get("rejection_reason", "hard_stop")
                    )
                )
            self._check_metrics(metrics, actor=True)
            if self.first_actor_update_at_transition is None:
                self.first_actor_update_at_transition = committed

    def _observe_gate(self, *, force: bool = False):
        del force
        if self.snapshot_callback is not None:
            self.snapshot_callback(self)
        return None

    def run(self) -> Dict[str, Any]:
        try:
            result = super().run()
            self.stop_reason = str(result.get("stop_reason", ""))
            return {
                **dict(result),
                "sac_status": "BOUNDED_COMPLETE",
                "training_rows": len(self.training_rows),
                "episode_summaries": len(self.episode_summaries),
                "first_actor_update_at_transition": self.first_actor_update_at_transition,
                "actor_optimizer_step_count": int(self.agent.actor_optimizer_step_count),
                "critic_optimizer_step_count": int(self.agent.critic_optimizer_step_count),
                "actor_update_count": int(self.agent.actor_update_count),
                "critic_update_count": int(self.agent.critic_update_count),
                "nan_count": int(self.agent.nan_count),
                "invalid_action_count": int(self.agent.invalid_action_count),
                "sentinel_frozen": bool(self.sentinel_frozen),
                "sentinel_manifest_sha256": str(self.sentinel_manifest_sha256),
                "sentinel_state_count": int(self.sentinel_manifest.get("state_count", 0)),
                "max_sentinel_kl": float(self.max_sentinel_kl),
                "trust_region_factor_counts": dict(self.trust_region_factor_counts),
            }
        except Exception as error:
            self.stop_reason = "{}: {}".format(type(error).__name__, error)
            raise

    def _safe_stop_and_close(self) -> None:
        self._stop_requested = True
        super()._safe_stop_and_close()


__all__ = ["SACOnlineRunner", "SACSafetyStop"]
