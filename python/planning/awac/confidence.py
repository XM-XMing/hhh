"""Mask-aware Twin-Q confidence for the AWAC innovation seam.

The production definition is intentionally explicit:

``a_BC = argmax(masked_pi_BC)``
``a_RL = argmax(masked_pi_AWAC)``
``DeltaQ = min(Q1,Q2)[a_RL] - min(Q1,Q2)[a_BC]``
``U = abs(Q1[a_RL] - Q2[a_RL])``

The closed form below is an implementation choice, not a claim that the
research plan specified a unique function ``f``.  It is affine-invariant for
positive common Q scaling and common translation, and it is detached from the
Actor/Critic graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Protocol

import numpy as np


CONFIDENCE_CONTRACT_ID = "awac_twin_q_confidence_v2"
CONFIDENCE_FORMULA_VERSION = "delta_q_uncertainty_exp_v2"
CONFIDENCE_SCALE_VERSION = "valid_qmin_half_range_or_mad_plus_mean_u_v1"


@dataclass(frozen=True)
class ConfidenceConfig:
    """Bounded mapping parameters; this is not a tuned production claim."""

    margin_gain: float = 2.0
    disagreement_gain: float = 1.0
    aggregation: str = "valid_action_mean"
    formula_version: str = CONFIDENCE_FORMULA_VERSION

    def __post_init__(self) -> None:
        for name in ("margin_gain", "disagreement_gain"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 100.0:
                raise ValueError("{} must be finite in (0,100]".format(name))
        if self.aggregation != "valid_action_mean":
            raise ValueError("unsupported confidence aggregation")
        if str(self.formula_version) != CONFIDENCE_FORMULA_VERSION:
            raise ValueError("unsupported confidence formula version")


@dataclass(frozen=True)
class ConfidenceSignal:
    available: bool = False
    value: Optional[float] = None
    source: str = "unavailable"


class ConfidenceProvider(Protocol):
    def confidence(self, observation) -> ConfidenceSignal:
        ...


def _numpy_row_median(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median with an explicit average-of-two convention for even rows."""

    result = []
    for row, mask in zip(values, valid):
        selected = np.sort(np.asarray(row[mask], dtype=np.float64))
        if selected.size == 0:
            raise ValueError("each confidence row needs one valid action")
        middle = selected.size // 2
        if selected.size % 2:
            result.append(selected[middle])
        else:
            result.append(0.5 * (selected[middle - 1] + selected[middle]))
    return np.asarray(result, dtype=np.float64)


class TwinQConfidenceEstimator:
    """Compute the plan-defined Twin-Q confidence without privileged inputs."""

    def __init__(self, config: Optional[ConfidenceConfig] = None) -> None:
        self.config = config or ConfidenceConfig()

    @staticmethod
    def _validate_numpy(q1, q2, mask, bc_policy, rl_policy):
        first = np.asarray(q1)
        second = np.asarray(q2)
        valid = np.asarray(mask, dtype=bool)
        bc = np.asarray(bc_policy)
        rl = np.asarray(rl_policy)
        arrays = (first, second, valid, bc, rl)
        arrays = tuple(value[None, :] if value.ndim == 1 else value for value in arrays)
        first, second, valid, bc, rl = arrays
        if first.shape != second.shape or first.shape != valid.shape:
            raise ValueError("Twin-Q inputs must have matching [B,A] shapes")
        if bc.shape != valid.shape or rl.shape != valid.shape:
            raise ValueError("BC/RL policies must match the [B,A] confidence shape")
        if first.ndim != 2 or first.shape[0] == 0 or first.shape[1] == 0:
            raise ValueError("Twin-Q inputs must be non-empty [B,A]")
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError("Twin-Q inputs contain non-finite values")
        if not np.isfinite(bc).all() or not np.isfinite(rl).all():
            raise ValueError("BC/RL policies contain non-finite values")
        if (bc < 0.0).any() or (rl < 0.0).any():
            raise ValueError("BC/RL policies must be non-negative probabilities")
        if not valid.any(axis=1).all():
            raise ValueError("each confidence row needs one valid action")
        return first, second, valid, bc, rl

    @staticmethod
    def _numpy_result(q1, q2, valid, bc_policy, rl_policy, config):
        qmin = np.minimum(q1, q2)
        valid_float = valid.astype(np.float64)
        counts = valid_float.sum(axis=1)
        masked_bc = np.where(valid, bc_policy, -np.inf)
        masked_rl = np.where(valid, rl_policy, -np.inf)
        a_bc = np.argmax(masked_bc, axis=1)
        a_rl = np.argmax(masked_rl, axis=1)
        rows = np.arange(qmin.shape[0])
        delta_q = qmin[rows, a_rl] - qmin[rows, a_bc]
        uncertainty = np.abs(q1[rows, a_rl] - q2[rows, a_rl])

        valid_qmin = np.where(valid, qmin, np.nan)
        qmin_range = np.nanmax(valid_qmin, axis=1) - np.nanmin(valid_qmin, axis=1)
        median = _numpy_row_median(qmin, valid)
        mad = _numpy_row_median(np.abs(qmin - median[:, None]), valid)
        disagreement_mean = (np.abs(q1 - q2) * valid_float).sum(axis=1) / counts
        q_scale = np.maximum.reduce(
            (0.5 * qmin_range, 1.4826 * mad, disagreement_mean)
        )
        q_scale = np.where(q_scale > 0.0, q_scale, 1.0)
        normalized_delta_q = delta_q / q_scale
        normalized_uncertainty = uncertainty / q_scale
        positive_signal = 1.0 - np.exp(
            -float(config.margin_gain) * np.maximum(normalized_delta_q, 0.0)
        )
        confidence = positive_signal * np.exp(
            -float(config.disagreement_gain) * np.maximum(normalized_uncertainty, 0.0)
        )
        confidence = np.clip(confidence, 0.0, 1.0)
        return {
            "confidence": confidence.astype(np.float64, copy=False),
            "qmin": qmin.astype(np.float64, copy=False),
            "a_bc": a_bc.astype(np.int64, copy=False),
            "a_rl": a_rl.astype(np.int64, copy=False),
            "delta_q": delta_q.astype(np.float64, copy=False),
            "uncertainty": uncertainty.astype(np.float64, copy=False),
            "q_scale": q_scale.astype(np.float64, copy=False),
            "normalized_delta_q": normalized_delta_q.astype(np.float64, copy=False),
            "normalized_uncertainty": normalized_uncertainty.astype(np.float64, copy=False),
            # Compatibility aliases now carry the corrected semantics.
            "margin": delta_q.astype(np.float64, copy=False),
            "normalized_margin": normalized_delta_q.astype(np.float64, copy=False),
            "twin_disagreement": uncertainty.astype(np.float64, copy=False),
            "normalized_twin_disagreement": normalized_uncertainty.astype(np.float64, copy=False),
            "confidence_contract_id": CONFIDENCE_CONTRACT_ID,
            "confidence_formula_version": CONFIDENCE_FORMULA_VERSION,
            "confidence_scale_version": CONFIDENCE_SCALE_VERSION,
            "aggregation": config.aggregation,
        }

    def estimate(
        self,
        q1: Any,
        q2: Any,
        valid_action_mask: Any,
        current_policy: Optional[Any] = None,
        *,
        bc_policy: Optional[Any] = None,
        rl_policy: Optional[Any] = None,
    ) -> Mapping[str, Any]:
        """Return detached confidence diagnostics for one or more states.

        ``current_policy`` is retained as a source-compatible alias for the
        AWAC policy only.  It never changes the aggregation rule.  Both BC and
        AWAC policies are required so the action identities cannot silently
        fall back to a replay/previous action.
        """

        if rl_policy is None and current_policy is not None:
            rl_policy = current_policy
        if bc_policy is None or rl_policy is None:
            raise ValueError("confidence requires both masked BC and AWAC policies")
        config = self.config
        if hasattr(q1, "detach") and hasattr(q2, "detach"):
            import torch

            first = q1.detach()
            second = q2.detach()
            valid = valid_action_mask.detach().bool()
            bc = bc_policy.detach() if hasattr(bc_policy, "detach") else torch.as_tensor(bc_policy, device=first.device)
            rl = rl_policy.detach() if hasattr(rl_policy, "detach") else torch.as_tensor(rl_policy, device=first.device)
            values = (first, second, valid, bc, rl)
            values = tuple(value.unsqueeze(0) if value.ndim == 1 else value for value in values)
            first, second, valid, bc, rl = values
            if first.shape != second.shape or first.shape != valid.shape:
                raise ValueError("Twin-Q inputs must have matching [B,A] shapes")
            if bc.shape != valid.shape or rl.shape != valid.shape:
                raise ValueError("BC/RL policies must match the [B,A] confidence shape")
            if first.ndim != 2 or int(first.shape[0]) == 0 or int(first.shape[1]) == 0:
                raise ValueError("Twin-Q inputs must be non-empty [B,A]")
            if not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all()):
                raise ValueError("Twin-Q inputs contain non-finite values")
            if not bool(torch.isfinite(bc).all()) or not bool(torch.isfinite(rl).all()):
                raise ValueError("BC/RL policies contain non-finite values")
            if bool((bc < 0.0).any()) or bool((rl < 0.0).any()):
                raise ValueError("BC/RL policies must be non-negative probabilities")
            if not bool(valid.any(dim=1).all()):
                raise ValueError("each confidence row needs one valid action")
            qmin = torch.minimum(first, second)
            masked_bc = torch.where(valid, bc, torch.full_like(bc, float("-inf")))
            masked_rl = torch.where(valid, rl, torch.full_like(rl, float("-inf")))
            a_bc = masked_bc.argmax(dim=1)
            a_rl = masked_rl.argmax(dim=1)
            rows = torch.arange(int(qmin.shape[0]), device=qmin.device)
            delta_q = qmin[rows, a_rl] - qmin[rows, a_bc]
            uncertainty = (first[rows, a_rl] - second[rows, a_rl]).abs()
            counts = valid.sum(dim=1).to(first.dtype)

            def row_median(value):
                ordered = torch.sort(
                    torch.where(valid, value, torch.full_like(value, float("inf"))), dim=1
                ).values
                upper = counts.to(torch.long) // 2
                lower = (counts.to(torch.long) - 1) // 2
                return 0.5 * (
                    ordered.gather(1, lower[:, None]).squeeze(1)
                    + ordered.gather(1, upper[:, None]).squeeze(1)
                )

            valid_qmin = torch.where(valid, qmin, torch.full_like(qmin, float("nan")))
            qmin_range = torch.nan_to_num(
                torch.nan_to_num(valid_qmin, nan=float("-inf")).amax(dim=1)
                - torch.nan_to_num(valid_qmin, nan=float("inf")).amin(dim=1),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            median = row_median(qmin)
            mad = row_median((qmin - median[:, None]).abs())
            disagreement_mean = (
                (first - second).abs() * valid.to(first.dtype)
            ).sum(dim=1) / counts
            q_scale = torch.stack(
                (0.5 * qmin_range, 1.4826 * mad, disagreement_mean), dim=0
            ).amax(dim=0)
            q_scale = torch.where(q_scale > 0.0, q_scale, torch.ones_like(q_scale))
            normalized_delta_q = delta_q / q_scale
            normalized_uncertainty = uncertainty / q_scale
            positive_signal = 1.0 - torch.exp(
                -float(config.margin_gain) * normalized_delta_q.clamp_min(0.0)
            )
            confidence = positive_signal * torch.exp(
                -float(config.disagreement_gain) * normalized_uncertainty.clamp_min(0.0)
            )
            confidence = confidence.clamp(0.0, 1.0).detach()
            return {
                "confidence": confidence,
                "qmin": qmin.detach(),
                "a_bc": a_bc.detach(),
                "a_rl": a_rl.detach(),
                "delta_q": delta_q.detach(),
                "uncertainty": uncertainty.detach(),
                "q_scale": q_scale.detach(),
                "normalized_delta_q": normalized_delta_q.detach(),
                "normalized_uncertainty": normalized_uncertainty.detach(),
                "margin": delta_q.detach(),
                "normalized_margin": normalized_delta_q.detach(),
                "twin_disagreement": uncertainty.detach(),
                "normalized_twin_disagreement": normalized_uncertainty.detach(),
                "confidence_contract_id": CONFIDENCE_CONTRACT_ID,
                "confidence_formula_version": CONFIDENCE_FORMULA_VERSION,
                "confidence_scale_version": CONFIDENCE_SCALE_VERSION,
                "aggregation": config.aggregation,
            }

        first, second, valid, bc, rl = self._validate_numpy(
            q1, q2, valid_action_mask, bc_policy, rl_policy
        )
        return self._numpy_result(first, second, valid, bc, rl, config)

    __call__ = estimate


def unavailable_confidence() -> ConfidenceSignal:
    return ConfidenceSignal()


__all__ = [
    "CONFIDENCE_CONTRACT_ID",
    "CONFIDENCE_FORMULA_VERSION",
    "CONFIDENCE_SCALE_VERSION",
    "ConfidenceConfig",
    "ConfidenceProvider",
    "ConfidenceSignal",
    "TwinQConfidenceEstimator",
    "unavailable_confidence",
]
