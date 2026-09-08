"""Read-only P3-C evidence validation for the first accepted Actor update.

This module is intentionally outside the off-policy source manifest: it
validates emitted metrics after an update and cannot alter collection, replay,
or learner behavior.  That preserves exact-resume identity for the frozen
P3-B checkpoint.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional


_FIRST_ACTOR_METRICS = (
    "actor_loss",
    "awac_advantage_mean",
    "awac_advantage_std",
    "awac_advantage_min",
    "awac_advantage_max",
    "awac_weight_mean",
    "awac_weight_std",
    "awac_weight_p50",
    "awac_weight_p95",
    "awac_weight_max",
    "awac_weight_cap_fraction",
    "awac_data_q_mean",
    "awac_data_q_std",
    "bc_kl",
    "bc_kl_p95",
    "bc_kl_max_after_update",
    "bc_kl_hard_budget",
    "bc_kl_budget_exceeded",
    "actor_gradient_norm",
    "actor_parameter_delta_norm",
    "actor_optimizer_step",
    "actor_trust_region_rejection_count",
    "parameters_finite",
)


def require_reference_fingerprint(
    *, reference_fingerprint: str, expected_fingerprint: str
) -> str:
    """Reject a trainable actor accidentally used as the frozen BC reference."""

    if not reference_fingerprint or not expected_fingerprint:
        raise ValueError("BC reference fingerprint is required")
    if str(reference_fingerprint) != str(expected_fingerprint):
        raise ValueError("BC reference fingerprint changed")
    return str(reference_fingerprint)


def first_actor_update_evidence(
    *,
    global_step: int,
    actor_update_step: int,
    metrics: Mapping[str, float],
    actor_fingerprint_before: str,
    actor_fingerprint_after: str,
) -> Optional[dict]:
    """Validate and materialize evidence for one accepted Actor update."""

    if int(actor_update_step) <= 0:
        return None
    if not actor_fingerprint_before or not actor_fingerprint_after:
        raise ValueError("actor fingerprint is required")
    if str(actor_fingerprint_before) == str(actor_fingerprint_after):
        raise ValueError("accepted Actor update did not change actor fingerprint")

    evidence = {
        "global_step": int(global_step),
        "actor_update_step": int(actor_update_step),
        "actor_fingerprint_before": str(actor_fingerprint_before),
        "actor_fingerprint_after": str(actor_fingerprint_after),
    }
    for name in _FIRST_ACTOR_METRICS:
        if name not in metrics:
            raise KeyError("missing first actor metric: {}".format(name))
        value = float(metrics[name])
        if not math.isfinite(value):
            raise FloatingPointError("non-finite first actor metric: {}".format(name))
        evidence[name] = value
    if evidence["actor_optimizer_step"] != 1.0:
        raise ValueError("first actor evidence requires an accepted optimizer step")
    if evidence["actor_parameter_delta_norm"] <= 0.0:
        raise ValueError("accepted Actor update has zero parameter delta")
    if evidence["parameters_finite"] != 1.0:
        raise FloatingPointError("accepted Actor update has non-finite parameters")
    return evidence


__all__ = ["first_actor_update_evidence", "require_reference_fingerprint"]
