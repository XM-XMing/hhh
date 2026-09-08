"""Pure, production-independent helpers for the V6 entropy-floor audit.

The SAC runtime owns the update transaction and its gate.  This module is
intentionally limited to reference calculations and journal checks so that a
diagnostic cannot silently change the production policy or replay contract.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Tuple

import numpy as np


def _as_2d(name: str, value: Any, *, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2:
        raise ValueError("{} must be a rank-2 array".format(name))
    return array


def _validate_logits_and_mask(logits: Any, mask: Any) -> Tuple[np.ndarray, np.ndarray]:
    values = _as_2d("logits", logits, dtype=np.float64)
    valid = _as_2d("action mask", mask, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("logits and action mask must have the same shape")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    counts = valid.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("empty action mask")
    return values, valid


def masked_entropy_reference(logits: Any, mask: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a masked categorical distribution without torch.

    Invalid actions have probability and returned log-probability zero.  The
    valid logits use a row-wise max subtraction, and the entropy calculation
    therefore implements the same ``0 * log(0)`` convention as the runtime.
    """

    values, valid = _validate_logits_and_mask(logits, mask)
    masked = np.where(valid, values, -np.inf)
    row_max = np.max(masked, axis=1, keepdims=True)
    exp_values = np.where(valid, np.exp(masked - row_max), 0.0)
    normalizer = exp_values.sum(axis=1, keepdims=True)
    probabilities = exp_values / normalizer
    log_probabilities = np.where(
        valid,
        np.log(np.maximum(probabilities, np.finfo(np.float64).tiny)),
        0.0,
    )
    entropy = -(probabilities * log_probabilities).sum(axis=1)
    return probabilities, log_probabilities, entropy


def valid_action_entropy_max(mask: Any) -> np.ndarray:
    """Return ``log(n_valid)``; one-valid-action states are not normalizable."""

    valid = _as_2d("action mask", mask, dtype=bool)
    counts = valid.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("empty action mask")
    result = np.full(counts.shape, np.nan, dtype=np.float64)
    multi = counts > 1
    result[multi] = np.log(counts[multi].astype(np.float64))
    return result


def normalized_entropy(entropy: Any, valid_action_count: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize entropy by ``log(n_valid)`` and mark single-action rows."""

    values = np.asarray(entropy, dtype=np.float64)
    counts = np.asarray(valid_action_count, dtype=np.int64)
    if values.shape != counts.shape:
        raise ValueError("entropy and valid_action_count must have the same shape")
    if np.any(counts <= 0):
        raise ValueError("valid_action_count must be positive")
    single = counts == 1
    result = np.full(values.shape, np.nan, dtype=np.float64)
    multi = counts > 1
    result[multi] = values[multi] / np.log(counts[multi].astype(np.float64))
    return result, single


def gate_statistic(
    entropy: Any,
    mask: Any,
    bc_entropy: Any,
    *,
    floor: float,
) -> Tuple[float, np.ndarray]:
    """Mirror the production eligible/minimum entropy gate contract."""

    values = np.asarray(entropy, dtype=np.float64)
    reference = np.asarray(bc_entropy, dtype=np.float64)
    valid = _as_2d("action mask", mask, dtype=bool)
    if values.ndim != 1 or reference.ndim != 1 or values.shape != reference.shape:
        raise ValueError("entropy, bc_entropy must be rank-1 and aligned")
    if values.shape[0] != valid.shape[0]:
        raise ValueError("entropy and action mask rows must align")
    if not np.isfinite(float(floor)) or float(floor) < 0.0:
        raise ValueError("floor must be finite and non-negative")
    counts = valid.sum(axis=1)
    eligible = (counts > 1) & (reference >= float(floor))
    statistic = float(np.min(values[eligible])) if np.any(eligible) else 0.0
    return statistic, eligible


def first_entropy_crossing(
    before: Sequence[float], after: Sequence[float], floor: float
) -> Any:
    """Return the one-based index of the first strict floor crossing."""

    left = np.asarray(list(before), dtype=np.float64)
    right = np.asarray(list(after), dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("before and after must have the same shape")
    crossing = (left >= float(floor)) & (right < float(floor))
    indices = np.flatnonzero(crossing)
    return int(indices[0] + 1) if indices.size else None


def residual_saturation_metrics(
    residual: Any, *, cap: float, threshold: float = 0.99
) -> Mapping[str, float]:
    """Summarize bounded residual magnitudes without clipping the evidence."""

    values = np.abs(np.asarray(residual, dtype=np.float64).reshape(-1))
    if values.size == 0:
        raise ValueError("residual cannot be empty")
    if not np.isfinite(values).all():
        raise ValueError("residual must be finite")
    cap_value = float(cap)
    if not np.isfinite(cap_value) or cap_value <= 0.0:
        raise ValueError("cap must be finite and positive")
    threshold_value = float(threshold) * cap_value
    return {
        "abs_mean": float(np.mean(values)),
        "abs_p90": float(np.percentile(values, 90.0)),
        "abs_p99": float(np.percentile(values, 99.0)),
        "abs_max": float(np.max(values)),
        "saturation_fraction": float(np.mean(values >= threshold_value)),
    }


def audit_transaction_rows(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Check the safety/rollback invariants represented by journal rows."""

    failures = []
    accepted_before = 0
    accepted_after = 0
    for index, row in enumerate(rows):
        pre_safe = bool(row.get("pre_gate_safe", True))
        optimizer_step = bool(row.get("optimizer_step", row.get("optimizer_step_performed", False)))
        accepted = bool(row.get("accepted", False))
        rollback = bool(row.get("rollback", row.get("rollback_performed", False)))
        if not pre_safe and optimizer_step:
            failures.append("row {} performed optimizer step after unsafe pre-gate".format(index))
        if not accepted and optimizer_step and not rollback:
            failures.append("row {} rejected without rollback".format(index))
        if accepted:
            accepted_after += 1
        if bool(row.get("accepted_count_changed", accepted)):
            accepted_before += 1
        if not accepted and bool(row.get("accepted_count_incremented", False)):
            failures.append("row {} incremented accepted count while rejected".format(index))
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "row_count": int(index + 1) if 'index' in locals() else 0,
        "accepted_count_rollback_safe": not any("accepted count" in item for item in failures),
        "accepted_rows": int(accepted_after),
    }


__all__ = [
    "audit_transaction_rows",
    "first_entropy_crossing",
    "gate_statistic",
    "masked_entropy_reference",
    "normalized_entropy",
    "residual_saturation_metrics",
    "valid_action_entropy_max",
]
