"""Critic-only AWAC calibration contracts and diagnostics.

Phase 0 deliberately keeps the BC Actor as a frozen behavior policy.  This
module owns the masked discrete Bellman projection, calibration metrics, and
relative stability gate; it does not open Actor updates or start an
environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from planning.awac.contract import (
    AWAC_PHASE_CRITIC_CALIBRATION,
    AWAC_REWARD_SCALE,
    AWAC_REWARD_SCALE_OWNER,
    TERMINAL_REASONS,
    build_awac_training_contract,
    awac_training_contract_sha256,
    replay_contract_sha256,
)
from planning.awac.checkpoint import build_awac_checkpoint_identity
from planning.awac.model import masked_policy
from planning.awac.phase1 import (
    PHASE_CRITIC_CALIBRATION,
    build_phase1_training_contract,
    require_calibration_pass,
)
from planning.common import canonical_json_sha256, file_sha256
from planning.contracts.policy_checkpoint_fingerprint import actor_state_sha256


CALIBRATION_GATE_CONTRACT_ID = "awac_calibration_gate_v2"
CALIBRATION_GATE_CONTRACT_VERSION = 2
HOLDOUT_RETURN_SEMANTICS = "episode_monte_carlo_return_v1"
RHO_COMPARISON = "strict_greater_than"


class HoldoutEpisodeValidationError(ValueError):
    """Raised when holdout rows cannot support a complete Monte Carlo return."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        reasons = [
            str(item.get("reason", "unknown"))
            for item in self.report.get("invalid_episodes", ())
            if isinstance(item, Mapping)
        ]
        super().__init__(
            "calibration holdout does not contain complete episodes: {}".format(
                ", ".join(reasons) or "unknown"
            )
        )


@dataclass(frozen=True)
class CriticCalibrationConfig:
    """Small, relative calibration gate configuration."""

    min_replay_transitions: int = 128
    min_completed_episodes: int = 16
    min_critic_updates: int = 32
    min_holdout_episodes: int = 8
    min_stability_windows: int = 3
    max_recent_td_relative_change: float = 0.25
    max_recent_disagreement_relative_change: float = 0.25
    q_explosion_multiplier: float = 4.0
    robust_scale_epsilon: float = 1.0e-6
    holdout_fraction: float = 0.10
    min_q_mc_rank_correlation: float = 0.40

    def __post_init__(self) -> None:
        for name in (
            "min_replay_transitions",
            "min_completed_episodes",
            "min_critic_updates",
            "min_holdout_episodes",
            "min_stability_windows",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError("{} must be positive".format(name))
        for name in (
            "max_recent_td_relative_change",
            "max_recent_disagreement_relative_change",
            "q_explosion_multiplier",
            "robust_scale_epsilon",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be finite and positive".format(name))
        if not 0.0 < float(self.holdout_fraction) < 1.0:
            raise ValueError("holdout_fraction must be in (0,1)")
        rho = float(self.min_q_mc_rank_correlation)
        if not math.isfinite(rho) or rho < -1.0 or rho > 1.0:
            raise ValueError("min_q_mc_rank_correlation must be finite in [-1,1]")

    def as_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "calibration_gate_contract_id": CALIBRATION_GATE_CONTRACT_ID,
                "calibration_gate_contract_version": CALIBRATION_GATE_CONTRACT_VERSION,
                "holdout_return_semantics": HOLDOUT_RETURN_SEMANTICS,
                "q_mc_rank_correlation_comparison": RHO_COMPARISON,
                "value_policy_alignment_required": True,
            }
        )
        return value


def calibration_gate_contract(config: CriticCalibrationConfig) -> Dict[str, Any]:
    """Return the versioned, future-only value-certification contract."""

    if not isinstance(config, CriticCalibrationConfig):
        raise TypeError("config must be CriticCalibrationConfig")
    return {
        "contract_id": CALIBRATION_GATE_CONTRACT_ID,
        "contract_version": CALIBRATION_GATE_CONTRACT_VERSION,
        "holdout_return_semantics": HOLDOUT_RETURN_SEMANTICS,
        "min_q_mc_rank_correlation": float(config.min_q_mc_rank_correlation),
        "q_mc_rank_correlation_comparison": RHO_COMPARISON,
        "value_policy_alignment_required": True,
    }


def calibration_gate_contract_from_config(
    value: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Project persisted config into a readable gate-contract boundary.

    Old checkpoints predate the V2 MC-return gate.  They remain readable, but
    this projection makes their legacy status explicit rather than silently
    treating an old ``calibration_gate_state=PASS`` as new certification.
    """

    config = dict(value or {})
    if config.get("calibration_gate_contract_id") != CALIBRATION_GATE_CONTRACT_ID:
        return {
            "contract_id": "awac_calibration_gate_legacy_v1",
            "contract_version": 1,
            "legacy": True,
            "new_value_certification_available": False,
        }
    fields = (
        "calibration_gate_contract_id",
        "calibration_gate_contract_version",
        "holdout_return_semantics",
        "min_q_mc_rank_correlation",
        "q_mc_rank_correlation_comparison",
        "value_policy_alignment_required",
    )
    missing = [field for field in fields if field not in config]
    if missing:
        raise ValueError(
            "V2 calibration gate config is incomplete: {}".format(
                ", ".join(missing)
            )
        )
    return {
        "contract_id": str(config["calibration_gate_contract_id"]),
        "contract_version": int(config["calibration_gate_contract_version"]),
        "holdout_return_semantics": str(config["holdout_return_semantics"]),
        "min_q_mc_rank_correlation": float(config["min_q_mc_rank_correlation"]),
        "q_mc_rank_correlation_comparison": str(
            config["q_mc_rank_correlation_comparison"]
        ),
        "value_policy_alignment_required": bool(
            config["value_policy_alignment_required"]
        ),
        "legacy": False,
        "new_value_certification_available": True,
    }


def _finite_array(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("{} must not be empty".format(name))
    if not np.isfinite(array).all():
        raise ValueError("{} contains non-finite values".format(name))
    return array


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = 0.5 * float(index + end - 1) + 1.0
        index = end
    return ranks


def spearman_rank_correlation(
    left: Sequence[float], right: Sequence[float]
) -> Optional[float]:
    """Compute Spearman correlation without adding a SciPy dependency."""

    x = _finite_array(left, name="left")
    y = _finite_array(right, name="right")
    if x.size != y.size:
        raise ValueError("rank-correlation arrays must have equal length")
    if x.size < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
    if denominator <= 0.0:
        return None
    return float(np.dot(rx, ry) / denominator)


def _holdout_done(value: Any) -> bool:
    """Decode the persisted numeric/bool terminal flag without truthiness bugs."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if numeric in (0.0, 1.0):
            return bool(numeric)
    raise ValueError("done must be an exact boolean or 0/1")


def _holdout_transition_index(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("episode_transition_index must be an integer")
    if isinstance(value, (int, np.integer)):
        index = int(value)
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        index = int(value)
    else:
        raise ValueError("episode_transition_index must be an integer")
    if index < 0:
        raise ValueError("episode_transition_index must be non-negative")
    return index


def _holdout_episode_identity(row: Mapping[str, Any]) -> Tuple[str, str]:
    mission_id = str(row.get("mission_id", "")).strip()
    episode_id = str(row.get("episode_id", "")).strip()
    if not mission_id or not episode_id:
        raise ValueError("mission_id and episode_id are both required")
    return mission_id, episode_id


def _runtime_abort_marked(row: Mapping[str, Any]) -> bool:
    """Treat persisted runtime-abort/incomplete markers as non-MC evidence."""

    if any(
        bool(row.get(name, False))
        for name in ("runtime_abort", "runtime_aborted", "episode_aborted")
    ):
        return True
    return row.get("episode_complete", True) is False


def episode_monte_carlo_returns(
    records: Sequence[Mapping[str, Any]], *, gamma: float, reward_scale: float
) -> Dict[str, Any]:
    """Compute complete-episode MC returns while preserving input row order.

    Holdout rows are allowed to arrive worker-interleaved.  The returned values
    are restored to their original record order, but every episode must prove a
    contiguous transition sequence with one legal terminal row.  Incomplete or
    aborted episodes are deliberately rejected rather than given a fabricated
    zero bootstrap value.
    """

    if not 0.0 <= float(gamma) < 1.0:
        raise ValueError("Monte Carlo gamma must be in [0,1)")
    if not math.isfinite(float(reward_scale)) or float(reward_scale) <= 0.0:
        raise ValueError("Monte Carlo reward_scale must be finite and positive")
    if not records:
        raise ValueError("calibration holdout requires at least one record")

    grouped: Dict[Tuple[str, str], list] = {}
    invalid: list = []
    for row_index, row in enumerate(records):
        if not isinstance(row, Mapping):
            invalid.append(
                {
                    "mission_id": "",
                    "episode_id": "",
                    "record_indices": [int(row_index)],
                    "reason": "record_not_mapping",
                }
            )
            continue
        try:
            key = _holdout_episode_identity(row)
        except ValueError:
            invalid.append(
                {
                    "mission_id": str(row.get("mission_id", "")).strip(),
                    "episode_id": str(row.get("episode_id", "")).strip(),
                    "record_indices": [int(row_index)],
                    "reason": "missing_episode_identity",
                }
            )
            continue
        grouped.setdefault(key, []).append((int(row_index), row))

    returns: list = [None] * len(records)
    complete_keys: list = []
    usable_row_count = 0
    legal_terminal_reasons = set(TERMINAL_REASONS)
    for key, rows in grouped.items():
        mission_id, episode_id = key
        indices = [row_index for row_index, _ in rows]
        try:
            parsed = [
                (row_index, row, _holdout_transition_index(row.get("episode_transition_index")))
                for row_index, row in rows
            ]
        except ValueError as error:
            invalid.append(
                {
                    "mission_id": mission_id,
                    "episode_id": episode_id,
                    "record_indices": indices,
                    "reason": "invalid_episode_transition_index",
                    "detail": str(error),
                }
            )
            continue
        transition_indices = [value[2] for value in parsed]
        if len(set(transition_indices)) != len(transition_indices):
            invalid.append(
                {
                    "mission_id": mission_id,
                    "episode_id": episode_id,
                    "record_indices": indices,
                    "reason": "duplicate_episode_transition_index",
                }
            )
            continue
        parsed.sort(key=lambda value: value[2])
        if [value[2] for value in parsed] != list(range(len(parsed))):
            invalid.append(
                {
                    "mission_id": mission_id,
                    "episode_id": episode_id,
                    "record_indices": indices,
                    "reason": "episode_transition_index_not_contiguous",
                }
            )
            continue
        if any(_runtime_abort_marked(row) for _, row, _ in parsed):
            invalid.append(
                {
                    "mission_id": mission_id,
                    "episode_id": episode_id,
                    "record_indices": indices,
                    "reason": "runtime_abort",
                }
            )
            continue
        try:
            done_values = [_holdout_done(row.get("done")) for _, row, _ in parsed]
        except ValueError as error:
            invalid.append(
                {
                    "mission_id": mission_id,
                    "episode_id": episode_id,
                    "record_indices": indices,
                    "reason": "invalid_done_flag",
                    "detail": str(error),
                }
            )
            continue
        if not done_values[-1]:
            invalid.append(
                {
                    "mission_id": mission_id,
                    "episode_id": episode_id,
                    "record_indices": indices,
                    "reason": "episode_not_terminal",
                }
            )
            continue
        if any(done_values[:-1]):
            invalid.append(
                {
                    "mission_id": mission_id,
                    "episode_id": episode_id,
                    "record_indices": indices,
                    "reason": "terminal_before_final_transition",
                }
            )
            continue
        terminal_reason = str(parsed[-1][1].get("terminal_reason", "")).strip()
        if terminal_reason not in legal_terminal_reasons:
            invalid.append(
                {
                    "mission_id": mission_id,
                    "episode_id": episode_id,
                    "record_indices": indices,
                    "reason": "illegal_terminal_reason",
                    "detail": terminal_reason,
                }
            )
            continue
        try:
            rewards = [float(row["reward"]) for _, row, _ in parsed]
        except (KeyError, TypeError, ValueError) as error:
            invalid.append(
                {
                    "mission_id": mission_id,
                    "episode_id": episode_id,
                    "record_indices": indices,
                    "reason": "invalid_reward",
                    "detail": str(error),
                }
            )
            continue
        if not np.isfinite(np.asarray(rewards, dtype=np.float64)).all():
            invalid.append(
                {
                    "mission_id": mission_id,
                    "episode_id": episode_id,
                    "record_indices": indices,
                    "reason": "nonfinite_reward",
                }
            )
            continue
        value = 0.0
        episode_returns = [0.0] * len(parsed)
        for position in range(len(parsed) - 1, -1, -1):
            value = float(reward_scale) * rewards[position] + float(gamma) * value
            episode_returns[position] = float(value)
        for (row_index, _, _), value in zip(parsed, episode_returns):
            returns[row_index] = float(value)
        complete_keys.append([mission_id, episode_id])
        usable_row_count += len(parsed)

    report = {
        "return_semantics": HOLDOUT_RETURN_SEMANTICS,
        "gamma": float(gamma),
        "reward_scale": float(reward_scale),
        "episode_keys": complete_keys,
        "complete_episode_count": int(len(complete_keys)),
        "usable_row_count": int(usable_row_count),
        "invalid_episode_count": int(len(invalid)),
        "invalid_episodes": invalid,
        "returns": returns,
    }
    if invalid:
        raise HoldoutEpisodeValidationError(report)
    return report


_VALUE_POLICY_IDENTITY_FIELDS = (
    "policy_id",
    "checkpoint_sha256",
    "mask_contract",
    "selection_mode",
    "temperature",
)


def assess_value_policy_alignment(
    *,
    behavior_policy: Optional[Mapping[str, Any]],
    target_policy: Optional[Mapping[str, Any]],
    return_policy: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """State exactly whether a Q/MC comparison certifies the target policy.

    A Q-versus-behavior-MC association is still useful when policies differ,
    but it is explicitly a cross-policy diagnostic and cannot certify the
    target Bellman policy's values.
    """

    values = {
        "behavior_policy": behavior_policy,
        "target_policy": target_policy,
        "return_policy": return_policy,
    }
    normalized: Dict[str, Dict[str, Any]] = {}
    missing = {}
    for name, value in values.items():
        if not isinstance(value, Mapping):
            missing[name] = list(_VALUE_POLICY_IDENTITY_FIELDS)
            continue
        item = {field: value.get(field) for field in _VALUE_POLICY_IDENTITY_FIELDS}
        absent = [
            field
            for field, field_value in item.items()
            if field_value is None or str(field_value) == ""
        ]
        if absent:
            missing[name] = absent
        else:
            normalized[name] = item
    if missing:
        return {
            "status": "UNKNOWN",
            "evidence": "policy_identity_missing",
            "missing_fields": missing,
            "behavior_policy": dict(behavior_policy or {}),
            "target_policy": dict(target_policy or {}),
            "return_policy": dict(return_policy or {}),
        }

    behavior_matches_return = (
        normalized["behavior_policy"] == normalized["return_policy"]
    )
    target_matches_return = normalized["target_policy"] == normalized["return_policy"]
    status = "MATCH" if behavior_matches_return and target_matches_return else "MISMATCH"
    return {
        "status": status,
        "evidence": "full_policy_identity_comparison",
        "behavior_matches_return_policy": behavior_matches_return,
        "target_matches_return_policy": target_matches_return,
        "behavior_policy": dict(behavior_policy or {}),
        "target_policy": dict(target_policy or {}),
        "return_policy": dict(return_policy or {}),
    }


def _robust_scale(values: np.ndarray, epsilon: float) -> float:
    median_abs = float(np.median(np.abs(values)))
    q25, q75 = np.percentile(values, [25.0, 75.0])
    iqr = float(q75 - q25)
    return max(median_abs, iqr, float(epsilon))


def _median_or_none(values: np.ndarray) -> Optional[float]:
    return None if values.size == 0 else float(np.median(values))


def _derived_return_bound(
    rewards: np.ndarray,
    returns: np.ndarray,
    *,
    gamma: float,
    reward_scale: float,
    multiplier: float,
    epsilon: float,
) -> float:
    if not 0.0 <= float(gamma) < 1.0:
        raise ValueError("calibration q bound requires gamma in [0,1)")
    reward_bound = (
        float(np.max(np.abs(rewards))) * float(reward_scale) / max(
            float(epsilon), 1.0 - float(gamma)
        )
    )
    observed_bound = float(np.max(np.abs(returns))) * float(multiplier)
    return max(reward_bound, observed_bound, float(epsilon))


def summarize_calibration_window(
    *,
    critic1_td_loss: float,
    critic2_td_loss: float,
    holdout_td_loss: float,
    q1: Sequence[float],
    q2: Sequence[float],
    rewards: Sequence[float],
    returns: Sequence[float],
    terminal_reasons: Iterable[str],
    gamma: float = 0.99,
    reward_scale: float = AWAC_REWARD_SCALE,
    q_explosion_multiplier: float = 4.0,
    robust_scale_epsilon: float = 1.0e-6,
    holdout_return_semantics: Optional[str] = None,
    holdout_measurement_status: str = "PASS",
    value_policy_alignment: str = "UNKNOWN",
    value_policy_alignment_evidence: Optional[Mapping[str, Any]] = None,
    complete_episode_count: Optional[int] = None,
    usable_row_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Summarize one holdout window with finite and relative diagnostics."""

    q1_array = _finite_array(q1, name="q1")
    q2_array = _finite_array(q2, name="q2")
    reward_array = _finite_array(rewards, name="rewards")
    return_array = _finite_array(returns, name="returns")
    if not (q1_array.size == q2_array.size == reward_array.size == return_array.size):
        raise ValueError("calibration window arrays must have equal length")
    losses = np.asarray(
        [critic1_td_loss, critic2_td_loss, holdout_td_loss], dtype=np.float64
    )
    if not np.isfinite(losses).all():
        return {"finite": False, "reason": "nonfinite_td_loss"}
    q_min = np.minimum(q1_array, q2_array)
    disagreement = np.abs(q1_array - q2_array)
    q_scale = _robust_scale(
        np.concatenate((q_min, return_array)),
        robust_scale_epsilon,
    )
    normalized = disagreement / (q_scale + float(robust_scale_epsilon))
    reasons = [str(value).strip() for value in terminal_reasons]
    if len(reasons) != int(q_min.size):
        raise ValueError("terminal_reasons length differs from Q arrays")
    success_mask = np.asarray([value == "success" for value in reasons], dtype=bool)
    failure_mask = ~success_mask
    success_q = q_min[success_mask]
    failure_q = q_min[failure_mask]
    success_median = _median_or_none(success_q)
    failure_median = _median_or_none(failure_q)
    difference = (
        None
        if success_median is None or failure_median is None
        else float(success_median - failure_median)
    )
    counts = {
        reason: int(sum(value == reason for value in reasons))
        for reason in (
            "success",
            "collision",
            "dead_end",
            "timeout",
            "max_steps",
            "hard_altitude",
        )
    }
    return {
        "finite": True,
        "critic1_td_loss": float(losses[0]),
        "critic2_td_loss": float(losses[1]),
        "holdout_td_loss": float(losses[2]),
        "q_mean": float(np.mean(q_min)),
        "q_std": float(np.std(q_min)),
        "q_min": float(np.min(q_min)),
        "q_max": float(np.max(q_min)),
        "q_p01": float(np.percentile(q_min, 1.0)),
        "q_p99": float(np.percentile(q_min, 99.0)),
        "twin_disagreement_mean": float(np.mean(disagreement)),
        "twin_disagreement_p50": float(np.percentile(disagreement, 50.0)),
        "twin_disagreement_p95": float(np.percentile(disagreement, 95.0)),
        "q_scale": float(q_scale),
        "normalized_twin_disagreement": float(np.mean(normalized)),
        "normalized_twin_disagreement_p95": float(np.percentile(normalized, 95.0)),
        "reward_mean": float(np.mean(reward_array)),
        "reward_std": float(np.std(reward_array)),
        "return_mean": float(np.mean(return_array)),
        "return_std": float(np.std(return_array)),
        "return_p99": float(np.percentile(return_array, 99.0)),
        "q_explosion_bound": float(
            _derived_return_bound(
                reward_array,
                return_array,
                gamma=gamma,
                reward_scale=reward_scale,
                multiplier=q_explosion_multiplier,
                epsilon=robust_scale_epsilon,
            )
        ),
        "success_q_median": success_median,
        "failure_q_median": failure_median,
        "success_minus_failure_q_median": difference,
        "success_sample_count": int(success_q.size),
        "failure_sample_count": int(failure_q.size),
        "value_ordering_status": (
            "PASS"
            if difference is not None and difference >= 0.0
            else "PENDING_INSUFFICIENT_DATA"
            if difference is None
            else "NOT_ORDERED"
        ),
        "holdout_q_return_rank_correlation": spearman_rank_correlation(
            q_min, return_array
        ),
        "calibration_gate_contract_id": CALIBRATION_GATE_CONTRACT_ID,
        "calibration_gate_contract_version": CALIBRATION_GATE_CONTRACT_VERSION,
        "holdout_return_semantics": (
            None
            if holdout_return_semantics is None
            else str(holdout_return_semantics)
        ),
        "holdout_measurement_status": str(holdout_measurement_status),
        "value_policy_alignment": str(value_policy_alignment),
        "value_policy_alignment_evidence": dict(
            value_policy_alignment_evidence or {}
        ),
        "complete_episode_count": (
            None if complete_episode_count is None else int(complete_episode_count)
        ),
        "usable_row_count": (
            None if usable_row_count is None else int(usable_row_count)
        ),
        "terminal_counts": counts,
        "terminal_success_count": counts["success"],
        "terminal_collision_count": counts["collision"],
        "terminal_dead_end_count": counts["dead_end"],
        "terminal_timeout_count": counts["timeout"],
        "terminal_max_steps_count": counts["max_steps"],
        "terminal_hard_altitude_count": counts["hard_altitude"],
    }


def _relative_change(previous: float, current: float, epsilon: float) -> float:
    return abs(float(current) - float(previous)) / (
        abs(float(previous)) + float(epsilon)
    )


def _directional_relative_increase(previous: float, current: float, epsilon: float) -> float:
    """Return signed growth; a large decrease is not a worsening signal."""

    return (float(current) - float(previous)) / (
        abs(float(previous)) + float(epsilon)
    )


def _window_values(windows: Sequence[Mapping[str, Any]], field: str) -> list:
    return [float(window[field]) for window in windows]


def evaluate_calibration_gate(
    *,
    windows: Sequence[Mapping[str, Any]],
    replay_transitions: int,
    completed_episodes: int,
    critic_updates: int,
    holdout_episodes: int,
    config: CriticCalibrationConfig,
) -> Dict[str, Any]:
    """Evaluate the versioned value-certification gate fail-closed.

    ``FAIL_DIVERGED`` is reserved for actual non-finite/numeric safety failures
    and directional sustained worsening.  Missing or incompatible holdout
    evidence remains a non-opening ``PENDING`` state instead of being mislabeled
    as a numerical divergence.
    """

    if not isinstance(config, CriticCalibrationConfig):
        raise TypeError("config must be CriticCalibrationConfig")
    rows = list(windows)
    replay_count = int(replay_transitions)
    episode_count = int(completed_episodes)
    update_count = int(critic_updates)
    holdout_count = int(holdout_episodes)
    minimums_met = (
        replay_count >= int(config.min_replay_transitions)
        and episode_count >= int(config.min_completed_episodes)
        and update_count >= int(config.min_critic_updates)
        and holdout_count >= int(config.min_holdout_episodes)
    )
    result_context = {
        "minimums_met": bool(minimums_met),
        "maturity_ready": bool(minimums_met),
        "replay_transitions": replay_count,
        "completed_episodes": episode_count,
        "critic_updates": update_count,
        "holdout_episodes": holdout_count,
        "stability_windows": int(len(rows)),
        "calibration_gate_contract_id": CALIBRATION_GATE_CONTRACT_ID,
        "calibration_gate_contract_version": CALIBRATION_GATE_CONTRACT_VERSION,
        "q_mc_rank_correlation_threshold": float(
            config.min_q_mc_rank_correlation
        ),
        "q_mc_rank_correlation_comparison": RHO_COMPARISON,
    }
    finite_fields = (
        "critic1_td_loss",
        "critic2_td_loss",
        "holdout_td_loss",
        "q_mean",
        "q_std",
        "q_min",
        "q_max",
        "q_p99",
        "normalized_twin_disagreement",
        "normalized_twin_disagreement_p95",
    )
    nonfinite = []
    missing_core_metrics = []
    for index, window in enumerate(rows):
        if window.get("finite") is False:
            # A producer marks a window ``finite=False`` only after observing
            # an actual numerical failure.  Contract/evidence failures use an
            # explicit finite placeholder plus ``CONTRACT_ERROR`` so they stay
            # pending instead of being mislabeled as divergence.
            if str(window.get("holdout_measurement_status", "")).upper() in (
                "CONTRACT_ERROR",
                "PENDING",
            ):
                missing_core_metrics.append(index)
            else:
                nonfinite.append(index)
            continue
        for field in finite_fields:
            try:
                raw_value = window[field]
            except KeyError:
                missing_core_metrics.append(index)
                break
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                missing_core_metrics.append(index)
                break
            if not math.isfinite(value):
                nonfinite.append(index)
                break
    if nonfinite:
        reason = "nonfinite_calibration_metrics"
        return {
            **result_context,
            "state": "FAIL_DIVERGED",
            "gate_state": "FAIL_DIVERGED",
            "would_open": False,
            "reason": reason,
            "gate_reason": reason,
            "hard_divergence_reason": reason,
            "nonfinite_window_indices": nonfinite,
            "evidence_status": "HARD_NUMERICAL_FAILURE",
        }

    enough_windows = len(rows) >= int(config.min_stability_windows)
    stability = False
    td_changes = []
    disagreement_changes = []
    td_directional_changes = []
    disagreement_directional_changes = []
    if enough_windows:
        recent = rows[-int(config.min_stability_windows) :]
        td_values = _window_values(recent, "holdout_td_loss")
        disagreement_values = _window_values(
            recent, "normalized_twin_disagreement"
        )
        for previous, current in zip(td_values, td_values[1:]):
            td_changes.append(
                _relative_change(previous, current, config.robust_scale_epsilon)
            )
            td_directional_changes.append(
                _directional_relative_increase(
                    previous, current, config.robust_scale_epsilon
                )
            )
        for previous, current in zip(
            disagreement_values, disagreement_values[1:]
        ):
            disagreement_changes.append(
                _relative_change(previous, current, config.robust_scale_epsilon)
            )
            disagreement_directional_changes.append(
                _directional_relative_increase(
                    previous, current, config.robust_scale_epsilon
                )
            )
        stability = bool(
            all(
                change <= float(config.max_recent_td_relative_change)
                for change in td_changes
            )
            and all(
                change <= float(config.max_recent_disagreement_relative_change)
                for change in disagreement_changes
            )
        )

    sustained_td_worsening = bool(
        len(td_changes) >= 2
        and all(
            change > float(config.max_recent_td_relative_change)
            for change in td_directional_changes
        )
    )
    sustained_disagreement_worsening = bool(
        len(disagreement_changes) >= 2
        and all(
            change > float(config.max_recent_disagreement_relative_change)
            for change in disagreement_directional_changes
        )
    )

    last = rows[-1] if rows else {}
    q_bound = float(last.get("q_explosion_bound", 0.0) or 0.0)
    q_min = float(last.get("q_min", 0.0) or 0.0)
    q_max = float(last.get("q_max", 0.0) or 0.0)
    q_explosion = bool(
        q_bound > 0.0
        and max(abs(q_min), abs(q_max)) > q_bound
    )
    failure_count = int(last.get("failure_sample_count", 0) or 0)
    success_count = int(last.get("success_sample_count", 0) or 0)
    ordering_status = str(last.get("value_ordering_status", "PENDING_INSUFFICIENT_DATA"))
    ordering_ready = bool(
        success_count > 0
        and failure_count > 0
        and ordering_status == "PASS"
    )
    measurement_status = str(last.get("holdout_measurement_status", "UNKNOWN"))
    return_semantics = last.get("holdout_return_semantics")
    policy_alignment = str(last.get("value_policy_alignment", "UNKNOWN"))
    rho_value = last.get("holdout_q_return_rank_correlation")
    try:
        rho = None if rho_value is None else float(rho_value)
    except (TypeError, ValueError):
        rho = None
    rho_available = rho is not None and math.isfinite(rho)
    hard_divergence_reason = ""
    if q_explosion:
        state = "FAIL_DIVERGED"
        reason = "q_explosion_relative_to_observed_return_scale"
        hard_divergence_reason = reason
        evidence_status = "HARD_NUMERICAL_FAILURE"
    elif missing_core_metrics:
        state = "PENDING"
        reason = "contract_error_missing_calibration_metrics"
        evidence_status = "CONTRACT_ERROR"
    elif not minimums_met or not enough_windows:
        state = "PENDING"
        reason = "insufficient_data"
        evidence_status = "PENDING_INSUFFICIENT_DATA"
    elif measurement_status != "PASS":
        state = "PENDING"
        reason = "contract_error_holdout_measurement_{}".format(
            measurement_status.lower() or "unknown"
        )
        evidence_status = "CONTRACT_ERROR"
    elif return_semantics != HOLDOUT_RETURN_SEMANTICS:
        state = "PENDING"
        reason = "holdout_return_semantics_unverified"
        evidence_status = "PENDING_EVIDENCE"
    elif policy_alignment != "MATCH":
        state = "PENDING"
        reason = "value_policy_alignment_{}".format(policy_alignment.lower())
        evidence_status = "PENDING_EVIDENCE"
    elif not rho_available:
        state = "PENDING"
        reason = "q_mc_rank_correlation_unavailable"
        evidence_status = "PENDING_EVIDENCE"
    elif not rho > float(config.min_q_mc_rank_correlation):
        state = "PENDING"
        reason = "q_mc_rank_correlation_below_threshold"
        evidence_status = "PENDING_EVIDENCE"
    elif sustained_td_worsening or sustained_disagreement_worsening:
        state = "FAIL_DIVERGED"
        reason = "sustained_relative_calibration_worsening"
        evidence_status = "HARD_NUMERICAL_FAILURE"
    elif stability and ordering_ready:
        state = "PASS"
        reason = "minimums_stable_finite_ordered_and_q_mc_ranked"
        evidence_status = "PASS"
    else:
        state = "PENDING"
        reason = "relative_stability_or_value_ordering_pending"
        evidence_status = "PENDING_EVIDENCE"
    return {
        **result_context,
        "state": state,
        "gate_state": state,
        "would_open": bool(state == "PASS"),
        "reason": reason,
        "gate_reason": reason,
        "hard_divergence_reason": hard_divergence_reason,
        "evidence_status": evidence_status,
        "missing_core_metric_window_indices": missing_core_metrics,
        "stability_windows_met": bool(enough_windows),
        "relative_stability": bool(stability),
        "sustained_td_worsening": sustained_td_worsening,
        "sustained_disagreement_worsening": sustained_disagreement_worsening,
        "holdout_td_relative_changes": td_changes,
        "normalized_disagreement_relative_changes": disagreement_changes,
        "holdout_td_directional_relative_changes": td_directional_changes,
        "normalized_disagreement_directional_relative_changes": disagreement_directional_changes,
        "q_explosion": q_explosion,
        "q_absolute_extreme": max(abs(q_min), abs(q_max)),
        "value_ordering_status": ordering_status,
        "value_ordering_ready": ordering_ready,
        "holdout_return_semantics": return_semantics,
        "holdout_measurement_status": measurement_status,
        "value_policy_alignment": policy_alignment,
        "holdout_q_return_rank_correlation": rho,
        "q_mc_rank_correlation_ready": bool(
            rho_available and rho > float(config.min_q_mc_rank_correlation)
        ),
    }


def masked_bellman_target(
    *,
    bc_logits,
    target_q1,
    target_q2,
    next_action_mask,
    reward,
    done,
    gamma: float,
    reward_scale: float,
    torch,
) -> Dict[str, Any]:
    """Compute the BC-distribution masked target exactly once."""

    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must be in (0,1]")
    if not math.isfinite(float(reward_scale)) or float(reward_scale) <= 0.0:
        raise ValueError("reward_scale must be finite and positive")
    if bc_logits.ndim != 2 or target_q1.shape != bc_logits.shape or target_q2.shape != bc_logits.shape:
        raise ValueError("Bellman policy/Q tensors must have identical [B,A] shape")
    if next_action_mask.shape != bc_logits.shape:
        raise ValueError("next_action_mask shape differs from policy/Q tensors")
    if reward.ndim != 1 or done.ndim != 1 or int(reward.shape[0]) != int(bc_logits.shape[0]) or done.shape != reward.shape:
        raise ValueError("reward/done must have shape [B]")
    masks = next_action_mask.bool()
    terminal = done.bool()
    nonterminal = ~terminal
    empty = ~masks.any(dim=1)
    if bool((nonterminal & empty).any()):
        raise ValueError("non-terminal next action mask is empty")

    # A terminal row has no bootstrap term.  Keep its next-policy diagnostics
    # at zero rather than manufacturing a valid action to make masking work.
    probabilities = torch.zeros_like(bc_logits)
    q_min = torch.zeros_like(target_q1)
    next_value = torch.zeros_like(reward)
    if bool(nonterminal.any()):
        active_masks = masks[nonterminal]
        active_logits = bc_logits[nonterminal]
        active_q_min = torch.minimum(
            target_q1[nonterminal], target_q2[nonterminal]
        )
        active_logits = active_logits.masked_fill(~active_masks, -1.0e4)
        active_probabilities = torch.softmax(active_logits, dim=1)
        active_probabilities = active_probabilities * active_masks.to(
            dtype=bc_logits.dtype
        )
        active_probabilities = active_probabilities / active_probabilities.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-12)
        probabilities[nonterminal] = active_probabilities
        q_min[nonterminal] = active_q_min
        next_value[nonterminal] = (
            active_probabilities * active_q_min
        ).sum(dim=1)
    effective_done = terminal
    target = reward * float(reward_scale) + float(gamma) * (
        ~terminal
    ).to(dtype=next_value.dtype) * next_value
    if not bool(torch.isfinite(target).all()):
        raise FloatingPointError("masked Bellman target is non-finite")
    next_value = torch.where(empty, torch.zeros_like(next_value), next_value)
    return {
        "target": target,
        "next_value": next_value,
        "q_min": q_min,
        "normalized_probabilities": probabilities,
        "empty_next_mask": empty,
        "effective_done": effective_done,
    }


def calibration_state_sha256(state_dict: Mapping, torch) -> str:
    """Fingerprint a model state without introducing a second hash format."""

    return actor_state_sha256({"actor_state_dict": state_dict})


def build_calibration_checkpoint_payload(
    *,
    learner,
    bc_checkpoint: Mapping[str, Any],
    bc_checkpoint_sha256: str,
    mpl_contract_sha256: str,
    training_contract_sha256: str,
    calibration_split: Mapping[str, Any],
    replay_identity: Mapping[str, Any],
    environment_step_count: int,
    calibration_metrics: Mapping[str, Any],
    mission_source_identity: Optional[Mapping[str, Any]] = None,
    mission_progress: Optional[Mapping[str, Any]] = None,
    completed_episode_count: Optional[int] = None,
    runtime_identity: Optional[Mapping[str, Any]] = None,
    exact_resume_state: Optional[Mapping[str, Any]] = None,
    source_code_sha256: str = "",
    training_contract: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a phase-bound checkpoint without enabling Actor updates."""

    if not isinstance(calibration_split, Mapping) or not calibration_split:
        raise ValueError("calibration split identity is missing")
    if not isinstance(replay_identity, Mapping) or not replay_identity:
        raise ValueError("replay identity is missing")
    if int(environment_step_count) < 0:
        raise ValueError("environment_step_count must be non-negative")
    if str(training_contract_sha256) == "" or len(str(training_contract_sha256)) != 64:
        raise ValueError("training contract SHA is missing")
    config = getattr(learner, "config", None)
    gamma = float(getattr(config, "gamma", 0.99))
    tau = float(getattr(config, "tau", 0.005))
    training_contract = dict(
        training_contract
        if training_contract is not None
        else build_awac_training_contract(
            phase=AWAC_PHASE_CRITIC_CALIBRATION,
            gamma=gamma,
            tau=tau,
            calibration={},
        )
    )
    expected_training_contract_sha256 = awac_training_contract_sha256(
        training_contract
    )
    if str(training_contract_sha256) != expected_training_contract_sha256:
        raise ValueError("training contract SHA does not match the calibration contract")
    resolved_config = dict(training_contract)
    resolved_config["optimization_config"] = (
        dict(config.__dict__) if config is not None else {}
    )
    gate_contract = calibration_gate_contract_from_config(
        training_contract.get("calibration")
    )
    max_primitive_steps = int(training_contract.get("max_primitive_steps", 45))
    split_hash = canonical_json_sha256(dict(calibration_split))
    state = learner.state_dict()
    if not source_code_sha256:
        root = Path(__file__).resolve().parents[3]
        source_code_sha256 = awac_calibration_source_sha256(root)
    payload = {
        **build_awac_checkpoint_identity(
            max_primitive_steps=max_primitive_steps
        ),
        "mpl_contract_sha256": str(mpl_contract_sha256),
        "resolved_training_config": resolved_config,
        "resolved_training_config_sha256": canonical_json_sha256(resolved_config),
        "training_contract": training_contract,
        "training_contract_sha256": str(training_contract_sha256),
        "source_code_sha256": str(source_code_sha256),
        "split_manifest_sha256": split_hash,
        "source_bc_checkpoint_sha256": str(bc_checkpoint_sha256),
        "bc_checkpoint_sha256": str(bc_checkpoint_sha256),
        "bc_reference_fingerprint": actor_state_sha256(bc_checkpoint),
        "global_step": int(getattr(learner, "update_step", 0)),
        "replay_size": int(replay_identity.get("replay_size", 0)),
        "update_step": int(getattr(learner, "update_step", 0)),
        "phase": AWAC_PHASE_CRITIC_CALIBRATION,
        "actor_update_enabled": False,
        "actor_optimizer_status": "unused_frozen",
        "actor_optimizer_step_count": int(getattr(learner, "actor_optimizer_step_count", 0)),
        "critic_update_count": int(getattr(learner, "critic_update_count", 0)),
        "environment_step_count": int(environment_step_count),
        "calibration_split": dict(calibration_split),
        "calibration_split_sha256": split_hash,
        "replay_identity": dict(replay_identity),
        "replay_contract_sha256": replay_contract_sha256(),
        "calibration_metrics": dict(calibration_metrics),
        "calibration_gate_state": str(calibration_metrics.get("state", "PENDING")),
        "calibration_gate_contract": gate_contract,
        "calibration_gate_contract_sha256": canonical_json_sha256(gate_contract),
        "reward_scale": AWAC_REWARD_SCALE,
        "reward_scale_owner": AWAC_REWARD_SCALE_OWNER,
        "actor_state_sha256": calibration_state_sha256(
            learner.actor.state_dict(), learner.torch
        ),
        "critic1_state_sha256": calibration_state_sha256(
            learner.critic1.state_dict(), learner.torch
        ),
        "critic2_state_sha256": calibration_state_sha256(
            learner.critic2.state_dict(), learner.torch
        ),
        "calibration_runtime_schema_id": "awac_formal_calibration_runtime_v1",
        "mission_source_identity": dict(mission_source_identity or {}),
        "mission_progress": dict(mission_progress or {}),
        "completed_episode_count": int(
            0 if completed_episode_count is None else completed_episode_count
        ),
        "runtime_identity": dict(runtime_identity or {}),
        "exact_resume_state": dict(exact_resume_state or {}),
    }
    payload.update(state)
    return payload


def build_calibration_pass_checkpoint_payload(
    *,
    learner,
    bc_checkpoint: Mapping[str, Any],
    bc_checkpoint_sha256: str,
    mpl_contract_sha256: str,
    training_contract_sha256: str,
    calibration_split: Mapping[str, Any],
    replay_identity: Mapping[str, Any],
    environment_step_count: int,
    calibration_metrics: Mapping[str, Any],
    actor_depth_lr: float,
    critic_depth_lr: float,
    mission_source_identity: Optional[Mapping[str, Any]] = None,
    mission_progress: Optional[Mapping[str, Any]] = None,
    completed_episode_count: Optional[int] = None,
    runtime_identity: Optional[Mapping[str, Any]] = None,
    exact_resume_state: Optional[Mapping[str, Any]] = None,
    phase1_training_contract: Optional[Mapping[str, Any]] = None,
    source_code_sha256: str = "",
    training_contract: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the publishable checkpoint only after a real gate PASS.

    The legacy ``build_calibration_checkpoint_payload`` remains available for
    Phase-0 progress snapshots, including ``PENDING``.  This separate owner is
    the only builder used for the Phase-1 pass artifact, so a pending or
    divergent calibration cannot be mislabeled as ready.
    """

    if not isinstance(calibration_metrics, Mapping):
        raise ValueError("calibration metrics are required for a pass checkpoint")
    require_calibration_pass(calibration_metrics.get("state", ""))
    if phase1_training_contract is None:
        phase1_training_contract = build_phase1_training_contract(
            actor_depth_lr=float(actor_depth_lr),
            critic_depth_lr=float(critic_depth_lr),
        )
    else:
        phase1_training_contract = dict(phase1_training_contract)
        if phase1_training_contract.get("actor_depth_lr") != float(actor_depth_lr):
            raise ValueError("Phase-1 Actor depth LR does not match its contract")
        if phase1_training_contract.get("critic_depth_lr") != float(critic_depth_lr):
            raise ValueError("Phase-1 Critic depth LR does not match its contract")

    payload = build_calibration_checkpoint_payload(
        learner=learner,
        bc_checkpoint=bc_checkpoint,
        bc_checkpoint_sha256=bc_checkpoint_sha256,
        mpl_contract_sha256=mpl_contract_sha256,
        training_contract_sha256=training_contract_sha256,
        calibration_split=calibration_split,
        replay_identity=replay_identity,
        environment_step_count=environment_step_count,
        calibration_metrics=calibration_metrics,
        mission_source_identity=mission_source_identity,
        mission_progress=mission_progress,
        completed_episode_count=completed_episode_count,
        runtime_identity=runtime_identity,
        exact_resume_state=exact_resume_state,
        source_code_sha256=source_code_sha256,
        training_contract=training_contract,
    )
    state = learner.state_dict()
    if int(payload.get("actor_optimizer_step_count", 0)) != 0:
        raise ValueError("calibration pass checkpoint Actor optimizer was used")
    if int(state.get("actor_update_count", 0)) != 0:
        raise ValueError("calibration pass checkpoint Actor update count is non-zero")
    if int(payload.get("critic_update_count", 0)) <= 0:
        raise ValueError("calibration pass checkpoint has no Critic updates")
    payload.update(
        {
            "calibration_pass_checkpoint": True,
            "phase1_state": PHASE_CRITIC_CALIBRATION,
            "phase1_training_contract": dict(phase1_training_contract),
            "phase1_training_contract_sha256": str(
                phase1_training_contract.get("contract_sha256", "")
            ),
            "phase1_actor_depth_lr": float(actor_depth_lr),
            "phase1_critic_depth_lr": float(critic_depth_lr),
            "calibration_actor_unchanged": True,
            "calibration_actor_update_count": 0,
        }
    )
    return payload


def awac_calibration_source_sha256(package_root: Path) -> str:
    """Hash the Phase-0 implementation files for checkpoint provenance."""

    root = Path(package_root).expanduser().resolve()
    relative_files = (
        "python/planning/awac/calibration.py",
        "python/planning/awac/calibration_runtime.py",
        "python/planning/awac/checkpoint.py",
        "python/planning/awac/contract.py",
        "python/planning/awac/interaction.py",
        "python/planning/awac/learner.py",
        "python/planning/awac/model.py",
        "python/planning/awac/optimization.py",
        "python/planning/awac/replay.py",
    )
    digest = hashlib.sha256()
    for relative in sorted(relative_files):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError("calibration source file missing: {}".format(path))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "CALIBRATION_GATE_CONTRACT_ID",
    "CALIBRATION_GATE_CONTRACT_VERSION",
    "CriticCalibrationConfig",
    "HOLDOUT_RETURN_SEMANTICS",
    "HoldoutEpisodeValidationError",
    "RHO_COMPARISON",
    "assess_value_policy_alignment",
    "awac_calibration_source_sha256",
    "build_calibration_checkpoint_payload",
    "build_calibration_pass_checkpoint_payload",
    "calibration_gate_contract",
    "calibration_gate_contract_from_config",
    "calibration_state_sha256",
    "episode_monte_carlo_returns",
    "evaluate_calibration_gate",
    "masked_bellman_target",
    "spearman_rank_correlation",
    "summarize_calibration_window",
]
