"""Runtime and terminal-outcome contracts shared by policy evaluation."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Tuple

from planning.common.config import parse_bool


POLICY_RUNTIME_CONTRACT_ID = "policy_runtime_mask_execution"
POLICY_UNITY_EVALUATION_CONTRACT_ID = "policy_unity_fixed_holdout"
POLICY_UNITY_OUTCOME_CONTRACT_ID = "policy_unity_terminal_outcomes"
MIN_FORMAL_HOLDOUT_EPISODES = 100
SAFETY_MASKS = ("height", "depth", "global")
EXECUTION_MODES = ("continuous", "stop")
DEPTH_MASK_NUMERIC_DEFAULTS = {
    "depth_mask_collision_radius_m": 0.40,
    "depth_mask_slack_m": 0.08,
    "depth_mask_sample_stride": 4,
    "depth_mask_max_patch_radius_px": 14,
}
_DEPTH_MASK_ARG_ALIASES = {
    "depth_mask_collision_radius_m": "depth_mask_collision_radius",
    "depth_mask_slack_m": "depth_mask_slack",
    "depth_mask_sample_stride": "depth_mask_sample_stride",
    "depth_mask_max_patch_radius_px": "depth_mask_max_patch_radius_px",
}
_DEPTH_MASK_NESTED_ALIASES = {
    "depth_mask_collision_radius_m": "collision_radius_m",
    "depth_mask_slack_m": "slack_m",
    "depth_mask_sample_stride": "sample_stride",
    "depth_mask_max_patch_radius_px": "max_patch_radius_px",
}
POLICY_TERMINAL_OUTCOMES = (
    "success",
    "collision",
    "dead_end",
    "timeout",
    "far",
    "hard_altitude",
)

def terminal_done_reason(
    *,
    success: bool,
    collision: bool,
    dead_end: bool,
    timeout: bool,
    far: bool,
    hard_altitude: bool,
) -> str:
    """Return the canonical terminal reason, or an empty string."""
    ordered = (
        ("success", success),
        ("collision", collision),
        ("altitude_violation", hard_altitude),
        ("timeout", timeout),
        ("far", far),
        ("dead_end", dead_end),
    )
    return next((name for name, enabled in ordered if bool(enabled)), "")


def summarize_policy_outcomes(
    rows: Iterable[Mapping],
    *,
    allow_legacy_done_timeout: bool = False,
) -> Dict:
    """Audit a rollout table as an exhaustive, mutually exclusive partition."""
    values = list(rows)
    counts = {name: 0 for name in POLICY_TERMINAL_OUTCOMES}
    unclassified_count = 0
    ambiguous_count = 0
    legacy_done_timeout_count = 0
    for row in values:
        flags = {
            name: parse_bool(row.get(name, False))
            for name in POLICY_TERMINAL_OUTCOMES
        }
        if (
            allow_legacy_done_timeout
            and not any(flags.values())
            and str(row.get("stop_reason", "")).strip() in ("done", "max_steps")
        ):
            flags["timeout"] = True
            legacy_done_timeout_count += 1
        active = [name for name, enabled in flags.items() if enabled]
        if not active:
            unclassified_count += 1
            continue
        if len(active) > 1:
            ambiguous_count += 1
            continue
        counts[active[0]] += 1

    episodes = len(values)
    summary = {
        "policy_unity_outcome_contract_id": POLICY_UNITY_OUTCOME_CONTRACT_ID,
        "terminal_outcome_total": int(sum(counts.values())),
        "unclassified_count": int(unclassified_count),
        "ambiguous_outcome_count": int(ambiguous_count),
        "legacy_done_timeout_count": int(legacy_done_timeout_count),
        "outcome_partition_valid": bool(
            sum(counts.values()) == episodes
            and unclassified_count == 0
            and ambiguous_count == 0
        ),
    }
    for name, count in counts.items():
        summary_name = (
            "altitude_violation" if name == "hard_altitude" else name
        )
        summary["{}_count".format(summary_name)] = int(count)
        summary["{}_rate".format(summary_name)] = float(count) / max(1, episodes)
    return summary


def holdout_score(metrics: Dict) -> Tuple[float, ...]:
    """Higher is better: success first, then fewer safety failures/dead ends."""
    return (
        float(metrics["success_rate"]),
        -float(metrics["collision_rate"]),
        -float(metrics["dead_end_rate"]),
        -float(metrics.get("timeout_rate", 0.0)),
        -float(metrics.get("far_rate", 0.0)),
        -float(metrics["altitude_violation_rate"]),
    )


def checkpoint_runtime_contract(checkpoint: Dict) -> Tuple[str, str]:
    """Return a checkpoint's audited mask and execution contract."""
    model_type = str(checkpoint.get("model_type", ""))
    args = checkpoint.get("args", {})
    args = args if isinstance(args, dict) else {}
    safety_mask = str(
        checkpoint.get("safety_mask", args.get("safety_mask", ""))
    )
    execution_mode = str(
        checkpoint.get("execution_mode", args.get("execution_mode", ""))
    )
    contract_id = str(checkpoint.get("policy_runtime_contract_id", ""))
    if contract_id:
        if safety_mask not in SAFETY_MASKS or execution_mode not in EXECUTION_MODES:
            raise ValueError(
                "policy checkpoint has no valid mask/execution runtime contract"
            )
        if contract_id != POLICY_RUNTIME_CONTRACT_ID:
            raise ValueError(
                "policy checkpoint runtime contract mismatch: {}".format(
                    contract_id
                )
            )
        return safety_mask, execution_mode
    # Existing BC artifacts already persist both resolved runtime fields even
    # though they predate the policy-neutral contract identifier.
    if (
        model_type.startswith("bc_")
        and safety_mask in SAFETY_MASKS
        and execution_mode in EXECUTION_MODES
    ):
        return safety_mask, execution_mode
    # Older BC checkpoints without resolved runtime fields used continuous
    # execution and the non-privileged height mask.
    return "height", "continuous"


def _normalize_depth_mask_numeric_config(values: Mapping) -> Dict:
    """Return the canonical, validated numeric depth-mask configuration."""
    config = {
        "depth_mask_collision_radius_m": float(
            values["depth_mask_collision_radius_m"]
        ),
        "depth_mask_slack_m": float(values["depth_mask_slack_m"]),
        "depth_mask_sample_stride": int(values["depth_mask_sample_stride"]),
        "depth_mask_max_patch_radius_px": int(
            values["depth_mask_max_patch_radius_px"]
        ),
    }
    if (
        not math.isfinite(config["depth_mask_collision_radius_m"])
        or config["depth_mask_collision_radius_m"] <= 0.0
    ):
        raise ValueError("depth-mask collision radius must be finite and positive")
    if (
        not math.isfinite(config["depth_mask_slack_m"])
        or config["depth_mask_slack_m"] < 0.0
    ):
        raise ValueError("depth-mask slack must be finite and non-negative")
    if config["depth_mask_sample_stride"] < 1:
        raise ValueError("depth-mask sample stride must be positive")
    if config["depth_mask_max_patch_radius_px"] < 0:
        raise ValueError("depth-mask max patch radius must be non-negative")
    return config


def checkpoint_depth_mask_numeric_config(checkpoint: Mapping) -> Dict:
    """Read resolved depth-mask numbers from a policy checkpoint.

    Current AWAC checkpoints persist the canonical values under
    ``depth_mask_config``. The older resolved CLI representation under ``args``
    and individual top-level fields remain readable for artifact compatibility.
    BC artifacts predating this numeric contract inherit the historical values
    used by both training and fixed-holdout evaluation.
    """
    args = checkpoint.get("args", {})
    args = args if isinstance(args, Mapping) else {}
    nested = checkpoint.get("depth_mask_config", {})
    nested = nested if isinstance(nested, Mapping) else {}
    values = {}
    for canonical_name, argument_name in _DEPTH_MASK_ARG_ALIASES.items():
        value = nested.get(_DEPTH_MASK_NESTED_ALIASES[canonical_name])
        if value is None:
            value = checkpoint.get(canonical_name)
        if value is None:
            value = args.get(argument_name)
        if value is None:
            value = DEPTH_MASK_NUMERIC_DEFAULTS[canonical_name]
        values[canonical_name] = value
    return _normalize_depth_mask_numeric_config(values)


def resolve_evaluation_depth_mask_numeric_config(
    checkpoint: Mapping,
    requested: Mapping,
    allow_override: bool,
) -> Tuple[Dict, bool]:
    """Inherit depth-mask numbers and reject unlabelled evaluation changes."""
    checkpoint_config = checkpoint_depth_mask_numeric_config(checkpoint)
    resolved = {
        name: checkpoint_config[name]
        if requested.get(name) is None
        else requested[name]
        for name in DEPTH_MASK_NUMERIC_DEFAULTS
    }
    resolved = _normalize_depth_mask_numeric_config(resolved)
    mismatch = any(
        resolved[name] != checkpoint_config[name]
        for name in DEPTH_MASK_NUMERIC_DEFAULTS
    )
    if mismatch and not bool(allow_override):
        differences = [
            "{}:{}->{}".format(
                name, checkpoint_config[name], resolved[name]
            )
            for name in DEPTH_MASK_NUMERIC_DEFAULTS
            if resolved[name] != checkpoint_config[name]
        ]
        raise ValueError(
            "evaluation depth-mask numeric config differs from checkpoint: {}; "
            "pass --audit-allow-runtime-override only for a labeled audit".format(
                ", ".join(differences)
            )
        )
    return resolved, bool(mismatch)


def resolve_evaluation_runtime(
    checkpoint: Dict,
    requested_mask: str,
    requested_execution: str,
    allow_override: bool,
) -> Tuple[str, str, bool]:
    checkpoint_mask, checkpoint_execution = checkpoint_runtime_contract(checkpoint)
    safety_mask = (
        checkpoint_mask if requested_mask == "inherit" else str(requested_mask)
    )
    execution_mode = (
        checkpoint_execution
        if requested_execution == "inherit"
        else str(requested_execution)
    )
    if safety_mask not in SAFETY_MASKS or execution_mode not in EXECUTION_MODES:
        raise ValueError("invalid evaluation runtime contract")
    mismatch = (
        safety_mask != checkpoint_mask or execution_mode != checkpoint_execution
    )
    if mismatch and not bool(allow_override):
        raise ValueError(
            "evaluation runtime differs from checkpoint: mask {}->{}, "
            "execution {}->{}; pass --audit-allow-runtime-override only for "
            "a labeled audit".format(
                checkpoint_mask,
                safety_mask,
                checkpoint_execution,
                execution_mode,
            )
        )
    return safety_mask, execution_mode, mismatch
