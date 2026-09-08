"""Masked local-to-global primitive-space exploration distribution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np


PRIMITIVE_EXPLORATION_CONTRACT_ID = "awac_primitive_exploration_v1"


@dataclass(frozen=True)
class PrimitiveExplorationConfig:
    enabled: bool = False
    neighbor_top_k: int = 8
    neighbor_radius: Optional[float] = None
    local_global_mix: float = 0.20
    confidence_coupling: float = 0.0
    progress_coupling: float = 0.0

    def __post_init__(self) -> None:
        if int(self.neighbor_top_k) <= 0:
            raise ValueError("neighbor_top_k must be positive")
        if self.neighbor_radius is not None and (
            not math.isfinite(float(self.neighbor_radius))
            or float(self.neighbor_radius) < 0.0
        ):
            raise ValueError("neighbor_radius must be finite and non-negative")
        for name in ("local_global_mix", "confidence_coupling", "progress_coupling"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("{} must be in [0,1]".format(name))


def _matrix_from_neighborhood(neighborhood: Any) -> np.ndarray:
    matrix = getattr(neighborhood, "distance_matrix", neighborhood)
    matrix = np.asarray(matrix, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("neighborhood must provide a finite square distance matrix")
    return matrix


def _one_distribution(
    policy,
    valid,
    anchor,
    matrix,
    config,
    confidence=None,
    training_progress=None,
):
    values = np.asarray(policy, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or mask.shape != values.shape:
        raise ValueError("policy and valid mask must have shape [A]")
    if matrix.shape != (values.shape[0], values.shape[0]):
        raise ValueError("neighborhood action count differs from policy")
    if not np.isfinite(values).all() or not mask.any():
        raise ValueError("policy must be finite and have one valid action")
    base = np.where(mask, np.maximum(values, 0.0), 0.0)
    base_sum = float(base.sum())
    if base_sum <= 0.0:
        base = mask.astype(np.float64)
        base_sum = float(base.sum())
    global_distribution = base / base_sum
    if not config.enabled:
        return global_distribution
    anchor = int(anchor)
    if anchor < 0 or anchor >= values.shape[0]:
        raise ValueError("anchor primitive is outside the action space")
    if config.neighbor_radius is not None:
        support = np.flatnonzero(matrix[anchor] <= float(config.neighbor_radius))
    else:
        order = np.argsort(matrix[anchor], kind="stable")
        support = order[: int(config.neighbor_top_k) + 1]
    support = support[mask[support]]
    if support.size == 0:
        support = np.flatnonzero(mask)
    local = np.zeros_like(global_distribution)
    local[support] = base[support]
    local_sum = float(local.sum())
    if local_sum <= 0.0:
        local[support] = 1.0
        local_sum = float(local.sum())
    local /= local_sum
    mix = float(config.local_global_mix)
    if confidence is not None and config.confidence_coupling:
        mix = min(
            1.0,
            mix
            + float(config.confidence_coupling)
            * (1.0 - float(np.clip(confidence, 0.0, 1.0))),
        )
    if training_progress is not None and config.progress_coupling:
        mix = min(
            1.0,
            mix
            + float(config.progress_coupling)
            * float(np.clip(training_progress, 0.0, 1.0)),
        )
    result = (1.0 - mix) * local + mix * global_distribution
    result[~mask] = 0.0
    total = float(result.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise FloatingPointError("primitive exploration produced an invalid distribution")
    return result / total


def primitive_behavior_distribution(
    policy_distribution: Any,
    valid_action_mask: Any,
    *,
    anchor_primitive: int,
    neighborhood: Any,
    config: Optional[PrimitiveExplorationConfig] = None,
    confidence: Optional[Any] = None,
    training_progress: Optional[float] = None,
) -> np.ndarray:
    """Return a finite masked distribution with explicit local/global support."""

    config = config or PrimitiveExplorationConfig()
    policy = np.asarray(policy_distribution, dtype=np.float64)
    mask = np.asarray(valid_action_mask, dtype=bool)
    matrix = _matrix_from_neighborhood(neighborhood)
    if policy.ndim == 1:
        result = _one_distribution(
            policy, mask, anchor_primitive, matrix, config, confidence, training_progress
        )
    elif policy.ndim == 2 and mask.shape == policy.shape:
        confidence_values = None if confidence is None else np.asarray(confidence).reshape(-1)
        result = np.stack(
            [
                _one_distribution(
                    policy[index],
                    mask[index],
                    anchor_primitive,
                    matrix,
                    config,
                    None if confidence_values is None else confidence_values[index],
                    training_progress,
                )
                for index in range(policy.shape[0])
            ],
            axis=0,
        )
    else:
        raise ValueError("policy and valid mask must both be [A] or [B,A]")
    if not np.isfinite(result).all():
        raise FloatingPointError("primitive exploration distribution is non-finite")
    if result.ndim == 1:
        if not np.isclose(result[mask].sum(), 1.0) or np.any(result[~mask] != 0.0):
            raise FloatingPointError("primitive exploration mask invariant failed")
    else:
        if not np.allclose(result.sum(axis=1), 1.0) or np.any(result[~mask] != 0.0):
            raise FloatingPointError("primitive exploration batch mask invariant failed")
    return result


__all__ = [
    "PRIMITIVE_EXPLORATION_CONTRACT_ID",
    "PrimitiveExplorationConfig",
    "primitive_behavior_distribution",
]
