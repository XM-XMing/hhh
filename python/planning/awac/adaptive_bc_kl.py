"""State-adaptive BC KL weighting for the AWAC innovation seam."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


ADAPTIVE_BC_KL_CONTRACT_ID = "awac_adaptive_bc_kl_v1"


@dataclass(frozen=True)
class AdaptiveBCKLConfig:
    """Bounded beta mapping; hard-KL recovery remains a separate gate."""

    beta_min: float = 0.02
    beta_max: float = 0.10
    base_weight: float = 0.05

    def __post_init__(self) -> None:
        values = {
            name: float(getattr(self, name))
            for name in ("beta_min", "beta_max", "base_weight")
        }
        if not all(math.isfinite(value) and value >= 0.0 for value in values.values()):
            raise ValueError("adaptive BC KL weights must be finite and non-negative")
        if values["beta_min"] > values["beta_max"]:
            raise ValueError("beta_min must not exceed beta_max")
        if values["base_weight"] <= 0.0:
            raise ValueError("base_weight must be positive")


def adaptive_bc_kl_weight(
    confidence: Any,
    *,
    config: AdaptiveBCKLConfig,
    torch=None,
    enabled: bool,
) -> Any:
    """Map low confidence to high beta; disabled mode is exactly base weight."""

    if torch is not None and hasattr(confidence, "detach"):
        value = confidence.detach().clamp(0.0, 1.0)
        if not enabled:
            return torch.full_like(value, float(config.base_weight))
        return (
            float(config.beta_max)
            - (float(config.beta_max) - float(config.beta_min)) * value
        ).clamp(float(config.beta_min), float(config.beta_max)).detach()
    value = np.asarray(confidence, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("confidence contains non-finite values")
    if not enabled:
        return np.full_like(value, float(config.base_weight), dtype=np.float64)
    clipped = np.clip(value, 0.0, 1.0)
    return np.clip(
        float(config.beta_max) - (float(config.beta_max) - float(config.beta_min)) * clipped,
        float(config.beta_min),
        float(config.beta_max),
    )


def adaptive_bc_kl_loss(
    per_row_kl: Any,
    confidence: Any,
    *,
    config: AdaptiveBCKLConfig,
    torch=None,
    enabled: bool,
) -> Any:
    """Reduce masked BC->Actor KL; hard-budget recovery is applied by learner."""

    if torch is not None and hasattr(per_row_kl, "detach"):
        if per_row_kl.ndim != 1 or confidence.shape != per_row_kl.shape:
            raise ValueError("per-row KL and confidence must have shape [B]")
        beta = adaptive_bc_kl_weight(
            confidence, config=config, torch=torch, enabled=enabled
        )
        return (beta * per_row_kl).mean(), beta
    kl = np.asarray(per_row_kl, dtype=np.float64)
    beta = adaptive_bc_kl_weight(
        confidence, config=config, torch=None, enabled=enabled
    )
    if kl.shape != np.asarray(beta).shape or not np.isfinite(kl).all():
        raise ValueError("per-row KL and confidence must be finite and shape-matched")
    return float(np.mean(kl * beta)), beta


__all__ = [
    "ADAPTIVE_BC_KL_CONTRACT_ID",
    "AdaptiveBCKLConfig",
    "adaptive_bc_kl_loss",
    "adaptive_bc_kl_weight",
]
