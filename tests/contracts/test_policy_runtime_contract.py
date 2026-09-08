#!/usr/bin/env python3
"""Test strict inheritance and audited overrides for policy runtime behavior."""

from planning.contracts.policy_runtime import (
    DEPTH_MASK_NUMERIC_DEFAULTS,
    POLICY_RUNTIME_CONTRACT_ID,
    POLICY_UNITY_OUTCOME_CONTRACT_ID,
    checkpoint_depth_mask_numeric_config,
    holdout_score,
    resolve_evaluation_depth_mask_numeric_config,
    resolve_evaluation_runtime,
    summarize_policy_outcomes,
    terminal_done_reason,
)


def main() -> int:
    checkpoint = {
        "model_type": "bc_soft",
        "policy_runtime_contract_id": POLICY_RUNTIME_CONTRACT_ID,
        "safety_mask": "depth",
        "execution_mode": "continuous",
    }
    assert resolve_evaluation_runtime(checkpoint, "inherit", "inherit", False) == (
        "depth", "continuous", False
    )
    try:
        resolve_evaluation_runtime(checkpoint, "height", "inherit", False)
    except ValueError:
        pass
    else:
        raise AssertionError("unapproved mask override was accepted")
    assert resolve_evaluation_runtime(checkpoint, "height", "stop", True) == (
        "height", "stop", True
    )

    awac_checkpoint = {
        **checkpoint,
        "args": {
            "depth_mask_collision_radius": 0.44,
            "depth_mask_slack": 0.06,
            "depth_mask_sample_stride": 3,
            "depth_mask_max_patch_radius_px": 18,
        },
    }
    inherited_depth, depth_override = resolve_evaluation_depth_mask_numeric_config(
        awac_checkpoint,
        {name: None for name in DEPTH_MASK_NUMERIC_DEFAULTS},
        False,
    )
    assert inherited_depth == {
        "depth_mask_collision_radius_m": 0.44,
        "depth_mask_slack_m": 0.06,
        "depth_mask_sample_stride": 3,
        "depth_mask_max_patch_radius_px": 18,
    }
    assert not depth_override
    changed = dict(inherited_depth, depth_mask_max_patch_radius_px=20)
    try:
        resolve_evaluation_depth_mask_numeric_config(
            awac_checkpoint, changed, False
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unapproved depth-mask numeric override was accepted")
    overridden_depth, depth_override = (
        resolve_evaluation_depth_mask_numeric_config(
            awac_checkpoint, changed, True
        )
    )
    assert overridden_depth["depth_mask_max_patch_radius_px"] == 20
    assert depth_override
    assert checkpoint_depth_mask_numeric_config({}) == DEPTH_MASK_NUMERIC_DEFAULTS
    nested_depth = checkpoint_depth_mask_numeric_config({
        "depth_mask_config": {
            "collision_radius_m": 0.41,
            "slack_m": 0.07,
            "sample_stride": 2,
            "max_patch_radius_px": 16,
        },
        "args": {"depth_mask_collision_radius": 0.99},
    })
    assert nested_depth == {
        "depth_mask_collision_radius_m": 0.41,
        "depth_mask_slack_m": 0.07,
        "depth_mask_sample_stride": 2,
        "depth_mask_max_patch_radius_px": 16,
    }

    uncontracted_bc = {"model_type": "bc_soft"}
    assert resolve_evaluation_runtime(
        uncontracted_bc, "inherit", "inherit", False
    ) == ("height", "continuous", False)

    # The current BC artifact predates the neutral identifier but already
    # stores both resolved runtime fields.
    legacy_bc = {
        "model_type": "bc_soft",
        "safety_mask": "depth",
        "execution_mode": "continuous",
    }
    assert resolve_evaluation_runtime(legacy_bc, "inherit", "inherit", False) == (
        "depth", "continuous", False
    )

    safe = {
        "success_rate": 0.5,
        "collision_rate": 0.0,
        "dead_end_rate": 0.2,
        "timeout_rate": 0.0,
        "far_rate": 0.0,
        "altitude_violation_rate": 0.0,
    }
    unsafe = dict(safe, collision_rate=0.1)
    assert holdout_score(safe) > holdout_score(unsafe)
    assert terminal_done_reason(
        success=False,
        collision=False,
        dead_end=False,
        timeout=True,
        far=False,
        hard_altitude=False,
    ) == "timeout"
    outcomes = summarize_policy_outcomes([
        {"success": True},
        {"collision": True},
        {"dead_end": True},
        {"timeout": True},
        {"far": True},
        {"hard_altitude": True},
    ])
    assert (
        outcomes["policy_unity_outcome_contract_id"]
        == POLICY_UNITY_OUTCOME_CONTRACT_ID
    )
    assert outcomes["outcome_partition_valid"]
    assert outcomes["terminal_outcome_total"] == 6
    assert outcomes["timeout_count"] == 1
    assert outcomes["far_count"] == 1
    incomplete = summarize_policy_outcomes([{"stop_reason": "done"}])
    assert not incomplete["outcome_partition_valid"]
    migrated = summarize_policy_outcomes(
        [{"stop_reason": "done"}], allow_legacy_done_timeout=True
    )
    assert migrated["outcome_partition_valid"]
    assert migrated["timeout_count"] == 1
    print("POLICY_RUNTIME_CONTRACT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
