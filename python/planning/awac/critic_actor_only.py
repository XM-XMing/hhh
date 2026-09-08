"""Small, fail-closed helpers for fixed-Critic Actor-only diagnostics."""

from __future__ import annotations

import copy
from typing import Any, Mapping, MutableMapping, Tuple


def adapt_h0_diagnostic_state_dict(
    state_dict: Mapping[str, Any],
    *,
    torch,
    policy_vector_dim: int = 127,
) -> Tuple[MutableMapping[str, Any], dict]:
    """Adapt a zero-budget diagnostic Critic to the production input shape.

    The diagnostic ``BudgetCritic`` stores a nested ``base.`` state dict and
    appends one budget column to the production vector.  H0 is valid only when
    that column is exactly zero.  The adapter strips the wrapper and drops the
    column; it never silently truncates a non-zero budget feature.
    """

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("diagnostic Critic state_dict is missing or empty")
    expected = int(policy_vector_dim)
    if expected <= 0:
        raise ValueError("policy_vector_dim must be positive")

    if "base.vector_encoder.0.weight" in state_dict:
        prefix = "base."
        weight_key = "base.vector_encoder.0.weight"
    elif "vector_encoder.0.weight" in state_dict:
        prefix = ""
        weight_key = "vector_encoder.0.weight"
    else:
        raise KeyError("diagnostic Critic has no vector encoder first-layer weight")

    weight = state_dict[weight_key]
    if not hasattr(weight, "ndim") or int(weight.ndim) != 2:
        raise ValueError("diagnostic Critic vector dimension is not a matrix")
    source_dim = int(weight.shape[1])
    if source_dim != expected + 1:
        raise ValueError(
            "diagnostic Critic vector dimension must be {} (got {})".format(
                expected + 1, source_dim
            )
        )
    budget_column = weight[:, -1].detach()
    budget_max_abs = float(budget_column.abs().max().cpu().item())
    if budget_max_abs != 0.0:
        raise ValueError(
            "diagnostic Critic H0 budget column is non-zero (max abs {})".format(
                budget_max_abs
            )
        )

    adapted = {}
    for name, value in state_dict.items():
        source_name = str(name)
        if prefix:
            if not source_name.startswith(prefix):
                raise ValueError(
                    "diagnostic Critic state contains an unwrapped key: {}".format(
                        source_name
                    )
                )
            target_name = source_name[len(prefix) :]
        else:
            target_name = source_name
        adapted[target_name] = (
            value.detach().clone()
            if hasattr(value, "detach")
            else copy.deepcopy(value)
        )
    adapted["vector_encoder.0.weight"] = weight[:, :expected].detach().clone()
    return adapted, {
        "source_prefix": prefix,
        "source_vector_dim": source_dim,
        "target_vector_dim": expected,
        "budget_column_max_abs": budget_max_abs,
        "budget_column_policy": "strict_zero_required",
    }
