"""Fail-closed Phase 1 AWAC readiness and orchestration contracts.

This module owns Phase 1 control-flow plumbing only.  It does not implement a
new policy objective, collect Unity data, or update an Actor by itself.  The
existing learner remains the owner of masked-discrete AWAC math and the
existing calibration gate remains the owner of calibration metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from planning.common.hashing import canonical_json_sha256


PHASE_CRITIC_CALIBRATION = "CRITIC_CALIBRATION"
PHASE_ACTOR_ENABLED_STANDARD_AWAC = "ACTOR_ENABLED_STANDARD_AWAC"
PHASE_STOPPED = "STOPPED"
PHASE_DIVERGED = "DIVERGED"

CALIBRATION_GATE_PENDING = "PENDING"
CALIBRATION_GATE_PASS = "PASS"
CALIBRATION_GATE_FAIL_DIVERGED = "FAIL_DIVERGED"
CALIBRATION_CERTIFICATION_BLOCKED_PENDING = "BLOCKED_PENDING"

PHASE1_MILESTONE_TRANSITIONS = (10_000, 25_000, 50_000)
PHASE1_DEFAULT_ONLINE_TRANSITION_CAP = 50_000
PHASE1_COLLAPSE_MARGIN = 0.10
PHASE1_CONTRACT_ID = "awac_phase1_readiness_v1"
PHASE1_CONTRACT_SCHEMA_VERSION = 1


class Phase1ReadinessError(ValueError):
    """Raised when a Phase 1 safety or identity gate cannot be satisfied."""


@dataclass(frozen=True)
class Phase1CalibrationSafetyCap:
    """Typed upper bounds for the critic-only calibration phase.

    The cap is a fail-closed maximum, not a required amount of training.  A
    gate that reaches PASS before either bound may certify; a PENDING gate at
    either bound is stopped and reported as BLOCKED_PENDING.
    """

    max_transitions: int = 30_000
    max_episodes: int = 1_000

    def __post_init__(self) -> None:
        for name in ("max_transitions", "max_episodes"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError("{} must be positive".format(name))
            object.__setattr__(self, name, value)

    def as_dict(self) -> Dict[str, int]:
        return {
            "max_transitions": int(self.max_transitions),
            "max_episodes": int(self.max_episodes),
        }

    def reached(self, *, transitions: int, episodes: int) -> bool:
        transition_count = int(transitions)
        episode_count = int(episodes)
        if transition_count < 0 or episode_count < 0:
            raise ValueError("calibration progress counts must be non-negative")
        return (
            transition_count >= int(self.max_transitions)
            or episode_count >= int(self.max_episodes)
        )


class Phase1CalibrationController:
    """Apply the existing gate result under a bounded, fail-closed policy.

    This controller does not enable the Actor.  A PASS result is handed to
    :class:`Phase1StateMachine` by the runtime owner; this seam only decides
    whether calibration may remain pending, has certified, or must stop.
    """

    def __init__(
        self,
        *,
        safety_cap: Optional[Phase1CalibrationSafetyCap] = None,
    ) -> None:
        if safety_cap is not None and not isinstance(
            safety_cap, Phase1CalibrationSafetyCap
        ):
            raise TypeError("safety_cap must be Phase1CalibrationSafetyCap")
        self._safety_cap = safety_cap or Phase1CalibrationSafetyCap()
        self._phase = PHASE_CRITIC_CALIBRATION
        self._certification = CALIBRATION_GATE_PENDING
        self._stop_reason = ""
        self._last_status: Dict[str, Any] = {}

    @property
    def safety_cap(self) -> Phase1CalibrationSafetyCap:
        return self._safety_cap

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def certification(self) -> str:
        return self._certification

    @property
    def actor_update_enabled(self) -> bool:
        return False

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    @property
    def status(self) -> Dict[str, Any]:
        return dict(self._last_status)

    def observe_gate(
        self,
        gate_result: Mapping[str, Any],
        *,
        transitions: int,
        episodes: int,
    ) -> Dict[str, Any]:
        """Record one existing gate result without changing its metrics."""

        if not isinstance(gate_result, Mapping):
            raise TypeError("calibration gate result must be a mapping")
        transition_count = int(transitions)
        episode_count = int(episodes)
        if transition_count < 0 or episode_count < 0:
            raise ValueError("calibration progress counts must be non-negative")
        state = str(
            gate_result.get("state", gate_result.get("calibration_gate_state", ""))
        ).strip().upper()
        if state not in {
            CALIBRATION_GATE_PENDING,
            CALIBRATION_GATE_PASS,
            CALIBRATION_GATE_FAIL_DIVERGED,
        }:
            raise Phase1ReadinessError(
                "unknown calibration gate state: {}".format(state or "<missing>")
            )
        cap_reached = self._safety_cap.reached(
            transitions=transition_count, episodes=episode_count
        )
        if state == CALIBRATION_GATE_FAIL_DIVERGED:
            self._phase = PHASE_DIVERGED
            self._certification = CALIBRATION_GATE_FAIL_DIVERGED
            self._stop_reason = "CALIBRATION_DIVERGED"
        elif state == CALIBRATION_GATE_PASS:
            if self._phase != PHASE_CRITIC_CALIBRATION:
                raise Phase1ReadinessError(
                    "calibration controller cannot reopen after stopping"
                )
            self._certification = CALIBRATION_GATE_PASS
            self._stop_reason = ""
        elif cap_reached:
            self._phase = PHASE_STOPPED
            self._certification = CALIBRATION_CERTIFICATION_BLOCKED_PENDING
            self._stop_reason = "CALIBRATION_SAFETY_CAP_REACHED"
        else:
            self._phase = PHASE_CRITIC_CALIBRATION
            self._certification = CALIBRATION_GATE_PENDING
            self._stop_reason = ""
        status = dict(gate_result)
        status.update(
            {
                "calibration_gate_state": state,
                "calibration_certification": self._certification,
                "phase": self._phase,
                "actor_update_enabled": False,
                "transitions": transition_count,
                "episodes": episode_count,
                "calibration_safety_cap": self._safety_cap.as_dict(),
                "calibration_safety_cap_reached": cap_reached,
                "stop_reason": self._stop_reason,
            }
        )
        self._last_status = dict(status)
        return dict(status)


def _finite(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Phase1ReadinessError("{} must be numeric".format(name)) from exc
    if not math.isfinite(result):
        raise Phase1ReadinessError("{} must be finite".format(name))
    return result


def validate_phase1_depth_learning_rates(
    *, actor_depth_lr: float, critic_depth_lr: float
) -> Dict[str, Any]:
    """Require both depth encoders to be trainable for Actor-enabled Phase 1."""

    actor = _finite(actor_depth_lr, name="actor_depth_lr")
    critic = _finite(critic_depth_lr, name="critic_depth_lr")
    if actor <= 0.0:
        raise Phase1ReadinessError(
            "actor_depth_lr must be > 0 for Phase 1 Actor enablement"
        )
    if critic <= 0.0:
        raise Phase1ReadinessError(
            "critic_depth_lr must be > 0 for Phase 1 Actor enablement"
        )
    return {
        "actor_depth_lr": actor,
        "critic_depth_lr": critic,
        "actor_depth_lr_positive": True,
        "critic_depth_lr_positive": True,
    }


def _optimizer_groups(optimizer: Any, *, name: str) -> Tuple[Mapping[str, Any], ...]:
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, (list, tuple)) or not groups:
        raise Phase1ReadinessError("{} has no optimizer parameter groups".format(name))
    result = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise Phase1ReadinessError(
                "{} optimizer group {} is not a mapping".format(name, index)
            )
        result.append(group)
    return tuple(result)


def _require_named_group(
    groups: Sequence[Mapping[str, Any]],
    *,
    expected_names: Iterable[str],
    expected_lr: float,
    label: str,
) -> Mapping[str, Any]:
    expected = set(str(value) for value in expected_names)
    matches = [group for group in groups if str(group.get("name", "")) in expected]
    if not matches:
        raise Phase1ReadinessError(
            "{} optimizer depth_encoder group is missing".format(label)
        )
    for group in matches:
        params = group.get("params")
        if not isinstance(params, (list, tuple)) or not params:
            raise Phase1ReadinessError(
                "{} optimizer depth_encoder group has no parameters".format(label)
            )
        if not math.isclose(
            _finite(group.get("lr"), name="{} depth group lr".format(label)),
            float(expected_lr),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise Phase1ReadinessError(
                "{} depth optimizer learning rate does not match resolved config".format(
                    label
                )
            )
        if not any(bool(getattr(parameter, "requires_grad", False)) for parameter in params):
            raise Phase1ReadinessError(
                "{} depth optimizer group is not trainable".format(label)
            )
    return matches[0]


def validate_phase1_optimizer_groups(
    actor_optimizer: Any,
    critic_optimizer: Any,
    *,
    actor_depth_lr: float,
    critic_depth_lr: float,
) -> Dict[str, Any]:
    """Verify positive resolved depth LRs are represented by real groups."""

    rates = validate_phase1_depth_learning_rates(
        actor_depth_lr=actor_depth_lr,
        critic_depth_lr=critic_depth_lr,
    )
    actor_groups = _optimizer_groups(actor_optimizer, name="Actor")
    critic_groups = _optimizer_groups(critic_optimizer, name="Critic")
    actor_depth = _require_named_group(
        actor_groups,
        expected_names=("depth_encoder",),
        expected_lr=rates["actor_depth_lr"],
        label="Actor",
    )
    critic_depth_names = tuple(
        str(group.get("name", ""))
        for group in critic_groups
        if str(group.get("name", "")).endswith("_depth_encoder")
    )
    critic_depth = _require_named_group(
        critic_groups,
        expected_names=critic_depth_names,
        expected_lr=rates["critic_depth_lr"],
        label="Critic",
    )
    return {
        **rates,
        "actor_group_names": [str(group.get("name", "")) for group in actor_groups],
        "critic_group_names": [str(group.get("name", "")) for group in critic_groups],
        "actor_depth_group": str(actor_depth.get("name")),
        "critic_depth_group": str(critic_depth.get("name")),
        "critic_depth_groups": list(critic_depth_names),
        "actor_depth_parameter_count": len(actor_depth.get("params", ())),
        "critic_depth_parameter_count": sum(
            len(group.get("params", ()))
            for group in critic_groups
            if str(group.get("name", "")).endswith("_depth_encoder")
        ),
    }


def require_calibration_pass(gate_state: str) -> bool:
    """Permit Phase 1 transition/checkpoint publication only after gate PASS."""

    state = str(gate_state)
    if state != CALIBRATION_GATE_PASS:
        raise Phase1ReadinessError(
            "calibration gate must be PASS before Phase 1 enablement; received {}".format(
                state
            )
        )
    return True


def _checkpoint_name(transition_count: int) -> str:
    if transition_count % 1000 == 0:
        suffix = "{}k".format(transition_count // 1000)
    else:
        suffix = str(transition_count)
    return "checkpoint_awac_{}.pt".format(suffix)


@dataclass(frozen=True)
class Phase1Milestone:
    """One standard-AWAC checkpoint and Dev evaluation boundary."""

    transition_count: int
    checkpoint_name: str
    evaluation_role: str = "DEV"
    evaluation_mission_count: int = 100

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_phase1_milestones(
    *, online_transition_cap: int = PHASE1_DEFAULT_ONLINE_TRANSITION_CAP
) -> Tuple[Phase1Milestone, ...]:
    """Build deterministic 10K/25K/50K milestones clipped to the safety cap."""

    cap = int(online_transition_cap)
    if cap <= 0:
        raise ValueError("online_transition_cap must be positive")
    selected = tuple(
        Phase1Milestone(
            transition_count=count,
            checkpoint_name=_checkpoint_name(count),
        )
        for count in PHASE1_MILESTONE_TRANSITIONS
        if count <= cap
    )
    if not selected:
        raise ValueError(
            "online_transition_cap must reach the first Phase 1 milestone"
        )
    return selected


def _timestamp_utc(value: Optional[str]) -> str:
    if value is not None and str(value).strip():
        return str(value)
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


class Phase1StateMachine:
    """Explicit Phase 1 state machine with no implicit Actor enablement."""

    def __init__(
        self,
        *,
        online_transition_cap: int = PHASE1_DEFAULT_ONLINE_TRANSITION_CAP,
    ) -> None:
        self._phase = PHASE_CRITIC_CALIBRATION
        self._actor_update_enabled = False
        self._stop_reason = ""
        self._last_gate_state = CALIBRATION_GATE_PENDING
        self._transition_records = []
        self._processed_milestones = set()
        self._milestones = build_phase1_milestones(
            online_transition_cap=int(online_transition_cap)
        )
        self._online_transition_cap = int(online_transition_cap)

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def actor_update_enabled(self) -> bool:
        return bool(self._actor_update_enabled)

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    @property
    def transition_records(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(dict(record) for record in self._transition_records)

    @property
    def processed_milestones(self) -> Tuple[int, ...]:
        """Return milestone counters already handled by the online phase."""

        return tuple(sorted(int(value) for value in self._processed_milestones))

    @property
    def milestones(self) -> Tuple[Phase1Milestone, ...]:
        return self._milestones

    def transition_from_calibration(
        self,
        *,
        gate_state: str,
        actor_depth_lr: float,
        critic_depth_lr: float,
        env_step: int,
        replay_size: int,
        critic_update_count: int,
        actor_update_count: int,
        source_checkpoint: str,
        timestamp_utc: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Apply the sole legal calibration-to-standard transition."""

        if self._phase != PHASE_CRITIC_CALIBRATION:
            raise Phase1ReadinessError(
                "calibration transition is only valid from CRITIC_CALIBRATION"
            )
        state = str(gate_state)
        self._last_gate_state = state
        if state == CALIBRATION_GATE_PENDING:
            return None
        if state == CALIBRATION_GATE_FAIL_DIVERGED:
            self._phase = PHASE_DIVERGED
            self._actor_update_enabled = False
            self._stop_reason = "CALIBRATION_DIVERGED"
            raise Phase1ReadinessError(
                "calibration gate diverged; Actor enablement is stopped"
            )
        require_calibration_pass(state)
        validate_phase1_depth_learning_rates(
            actor_depth_lr=actor_depth_lr,
            critic_depth_lr=critic_depth_lr,
        )
        counters = {
            "env_step": int(env_step),
            "replay_size": int(replay_size),
            "critic_update_count": int(critic_update_count),
            "actor_update_count": int(actor_update_count),
        }
        if any(value < 0 for value in counters.values()):
            raise Phase1ReadinessError("Phase 1 transition counters must be non-negative")
        if counters["actor_update_count"] != 0:
            raise Phase1ReadinessError(
                "calibration-to-standard transition requires actor_update_count=0"
            )
        checkpoint = str(source_checkpoint).strip()
        if not checkpoint:
            raise Phase1ReadinessError("calibration source checkpoint is required")
        record = {
            "from_phase": PHASE_CRITIC_CALIBRATION,
            "to_phase": PHASE_ACTOR_ENABLED_STANDARD_AWAC,
            "gate_state": CALIBRATION_GATE_PASS,
            "timestamp_utc": _timestamp_utc(timestamp_utc),
            "actor_depth_lr": float(actor_depth_lr),
            "critic_depth_lr": float(critic_depth_lr),
            "env_step": counters["env_step"],
            "replay_size": counters["replay_size"],
            "critic_update_count": counters["critic_update_count"],
            "actor_update_count": counters["actor_update_count"],
            "source_checkpoint": checkpoint,
        }
        self._phase = PHASE_ACTOR_ENABLED_STANDARD_AWAC
        self._actor_update_enabled = True
        self._transition_records.append(record)
        return dict(record)

    def stop(self, reason: str) -> None:
        """Stop the phase without rollback or an implicit restart."""

        if self._phase == PHASE_DIVERGED:
            return
        self._phase = PHASE_STOPPED
        self._actor_update_enabled = False
        self._stop_reason = str(reason)

    def observe_online_progress(
        self,
        *,
        online_transitions: int,
        save_checkpoint: Callable[[Phase1Milestone], Any],
        evaluate_dev: Callable[[Phase1Milestone], Mapping[str, Any]],
        bc_dev_success_rate: float,
        collapse_margin: float = PHASE1_COLLAPSE_MARGIN,
    ) -> Dict[str, Any]:
        """Run due checkpoint/evaluation hooks and enforce collapse stop."""

        if self._phase != PHASE_ACTOR_ENABLED_STANDARD_AWAC:
            raise Phase1ReadinessError(
                "online milestone loop requires ACTOR_ENABLED_STANDARD_AWAC"
            )
        count = int(online_transitions)
        if count < 0 or count > self._online_transition_cap:
            raise ValueError("online_transitions is outside the Phase 1 safety cap")
        baseline = _finite(bc_dev_success_rate, name="bc_dev_success_rate")
        margin = _finite(collapse_margin, name="collapse_margin")
        if not 0.0 <= baseline <= 1.0 or margin < 0.0:
            raise ValueError("invalid Phase 1 Dev collapse configuration")
        processed = []
        evaluations = []
        for milestone in self._milestones:
            if milestone.transition_count > count:
                continue
            if milestone.transition_count in self._processed_milestones:
                continue
            save_checkpoint(milestone)
            result = evaluate_dev(milestone)
            if not isinstance(result, Mapping):
                raise Phase1ReadinessError("Dev evaluation hook must return a mapping")
            success_rate = _finite(
                result.get("success_rate"), name="Dev success_rate"
            )
            if not 0.0 <= success_rate <= 1.0:
                raise Phase1ReadinessError("Dev success_rate is outside [0,1]")
            evaluations.append(
                {
                    "transition_count": milestone.transition_count,
                    "success_rate": success_rate,
                    "result": dict(result),
                }
            )
            self._processed_milestones.add(milestone.transition_count)
            processed.append(milestone.transition_count)
            if success_rate < baseline - margin:
                self.stop("COLLAPSED")
                break
        if (
            self._phase == PHASE_ACTOR_ENABLED_STANDARD_AWAC
            and count >= self._online_transition_cap
        ):
            self.stop("ONLINE_TRANSITION_CAP_REACHED")
        return {
            "phase": self._phase,
            "actor_update_enabled": self.actor_update_enabled,
            "online_transitions": count,
            "milestones_processed": processed,
            "evaluations": evaluations,
            "stop_reason": self._stop_reason,
        }


def validate_phase1_dev_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Accept only the independent DEV artifact, never the final holdout."""

    if not isinstance(manifest, Mapping):
        raise ValueError("Phase 1 Dev manifest must be a mapping")
    role = str(manifest.get("role", "")).strip().upper()
    if role in {"FINAL_TEST", "FINAL_TEST_ONLY"} or bool(
        manifest.get("final_test_only", False)
    ):
        raise ValueError("FINAL_TEST_ONLY artifact is forbidden for Phase 1 Dev")
    if role != "DEV":
        raise ValueError("Phase 1 evaluation requires a DEV manifest")
    mission_count = int(manifest.get("mission_count", manifest.get("requested_count", 0)))
    if mission_count <= 0:
        raise ValueError("Phase 1 DEV manifest must contain missions")
    return {
        "role": "DEV",
        "mission_count": mission_count,
        "final_test_only": False,
    }


def build_phase1_training_contract(
    *,
    actor_depth_lr: float,
    critic_depth_lr: float,
    online_transition_cap: int = PHASE1_DEFAULT_ONLINE_TRANSITION_CAP,
    collapse_margin: float = PHASE1_COLLAPSE_MARGIN,
    calibration_safety_cap: Optional[Phase1CalibrationSafetyCap] = None,
) -> Dict[str, Any]:
    """Build the Phase 1-only config projection without changing Phase 0 SHA."""

    rates = validate_phase1_depth_learning_rates(
        actor_depth_lr=actor_depth_lr,
        critic_depth_lr=critic_depth_lr,
    )
    margin = _finite(collapse_margin, name="collapse_margin")
    if margin < 0.0:
        raise ValueError("collapse_margin must be non-negative")
    safety_cap = calibration_safety_cap or Phase1CalibrationSafetyCap()
    if not isinstance(safety_cap, Phase1CalibrationSafetyCap):
        raise TypeError(
            "calibration_safety_cap must be Phase1CalibrationSafetyCap"
        )
    contract = {
        "schema_version": PHASE1_CONTRACT_SCHEMA_VERSION,
        "contract_id": PHASE1_CONTRACT_ID,
        "initial_phase": PHASE_CRITIC_CALIBRATION,
        "actor_enabled_phase": PHASE_ACTOR_ENABLED_STANDARD_AWAC,
        "actor_depth_lr": rates["actor_depth_lr"],
        "critic_depth_lr": rates["critic_depth_lr"],
        "online_transition_cap": int(online_transition_cap),
        "milestones": [
            milestone.as_dict()
            for milestone in build_phase1_milestones(
                online_transition_cap=int(online_transition_cap)
            )
        ],
        "dev_evaluation_role": "DEV",
        "dev_evaluation_mission_count": 100,
        "collapse_margin": margin,
        "calibration_safety_cap": safety_cap.as_dict(),
        "actor_update_enabled_before_transition": False,
        "replay_behavior_sources": ["BC_CALIBRATION", "AWAC_ONLINE"],
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def phase1_training_contract_sha256(contract: Optional[Mapping[str, Any]] = None) -> str:
    value = (
        build_phase1_training_contract(actor_depth_lr=1.0e-6, critic_depth_lr=1.0e-5)
        if contract is None
        else dict(contract)
    )
    value.pop("contract_sha256", None)
    return canonical_json_sha256(value)


def build_phase1_readiness_dry_run(
    *,
    learner: Any,
    dev_manifest: Mapping[str, Any],
    actor_update_count: int = 0,
    calibration_safety_cap: Optional[Phase1CalibrationSafetyCap] = None,
) -> Dict[str, Any]:
    """Validate construction seams without stepping an optimizer or runtime."""

    config = getattr(learner, "config", None)
    if config is None:
        raise Phase1ReadinessError("learner optimization config is missing")
    optimizer = validate_phase1_optimizer_groups(
        learner.actor_optimizer,
        learner.critic_optimizer,
        actor_depth_lr=float(config.actor_depth_lr),
        critic_depth_lr=float(config.critic_depth_lr),
    )
    dev = validate_phase1_dev_manifest(dev_manifest)
    if int(actor_update_count) != 0:
        raise Phase1ReadinessError("readiness dry-run requires actor_update_count=0")
    contract = build_phase1_training_contract(
        actor_depth_lr=float(config.actor_depth_lr),
        critic_depth_lr=float(config.critic_depth_lr),
        calibration_safety_cap=calibration_safety_cap,
    )
    return {
        "phase": PHASE_CRITIC_CALIBRATION,
        "actor_update_enabled": False,
        "actor_update_count": 0,
        "optimizer": optimizer,
        "dev_manifest": dev,
        "milestones": [milestone.as_dict() for milestone in build_phase1_milestones()],
        "phase1_training_contract": contract,
        "phase1_training_contract_sha256": phase1_training_contract_sha256(contract),
        "final_test_protected": True,
    }


__all__ = [
    "CALIBRATION_CERTIFICATION_BLOCKED_PENDING",
    "CALIBRATION_GATE_FAIL_DIVERGED",
    "CALIBRATION_GATE_PASS",
    "CALIBRATION_GATE_PENDING",
    "PHASE1_COLLAPSE_MARGIN",
    "PHASE1_CONTRACT_ID",
    "PHASE1_DEFAULT_ONLINE_TRANSITION_CAP",
    "PHASE1_MILESTONE_TRANSITIONS",
    "PHASE_ACTOR_ENABLED_STANDARD_AWAC",
    "PHASE_CRITIC_CALIBRATION",
    "PHASE_DIVERGED",
    "PHASE_STOPPED",
    "Phase1Milestone",
    "Phase1CalibrationController",
    "Phase1CalibrationSafetyCap",
    "Phase1ReadinessError",
    "Phase1StateMachine",
    "build_phase1_milestones",
    "build_phase1_readiness_dry_run",
    "build_phase1_training_contract",
    "phase1_training_contract_sha256",
    "require_calibration_pass",
    "validate_phase1_depth_learning_rates",
    "validate_phase1_dev_manifest",
    "validate_phase1_optimizer_groups",
]
