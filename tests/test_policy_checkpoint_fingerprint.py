from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from planning.contracts.policy_checkpoint_fingerprint import (
    actor_state_sha256,
    deployable_policy_sha256,
)
from planning.contracts.policy_runtime import POLICY_RUNTIME_CONTRACT_ID
from planning.contracts.task import TASK_CONTRACT_ID


def _checkpoint(torch, *, awac: bool = False):
    state = OrderedDict([
        ("linear.weight", torch.arange(12, dtype=torch.float32).reshape(3, 4)),
        ("linear.bias", torch.tensor([0.5, -0.5, 1.0], dtype=torch.float32)),
    ])
    payload = {
        "model_type": "awac_discrete_masked" if awac else "bc_soft",
        "feature_contract_id": "depth_goal_state_prev_action",
        "policy_input_contract_sha256": "1" * 64,
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": "2" * 64,
        "mpl_contract_sha256": "3" * 64,
        "vec_dim": 127,
        "num_actions": 105,
        "depth_history_frames": 1,
        "initial_prev_action": -1,
        "safety_mask": "depth",
        "execution_mode": "continuous",
        "privileged_runtime_inputs": [],
        "feature_mean": np.arange(22, dtype=np.float32),
        "feature_std": np.linspace(0.5, 1.5, 22, dtype=np.float32),
    }
    if awac:
        payload.update({
            "actor_state_dict": state,
            "policy_runtime_contract_id": POLICY_RUNTIME_CONTRACT_ID,
            "depth_mask_config": {
                "collision_radius_m": 0.40,
                "slack_m": 0.08,
                "sample_stride": 4,
                "max_patch_radius_px": 14,
            },
            "critic1_state_dict": {"unrelated": torch.tensor([99.0])},
            "global_step": 6000,
        })
    else:
        payload["model_state_dict"] = state
    return payload


@pytest.mark.unit
def test_actor_fingerprint_matches_bc_and_awac_wrappers_and_ignores_metadata():
    torch = pytest.importorskip("torch")
    bc = _checkpoint(torch, awac=False)
    awac = _checkpoint(torch, awac=True)
    expected = actor_state_sha256(bc)
    assert actor_state_sha256(awac) == expected

    awac["global_step"] = 9000
    awac["critic1_state_dict"]["unrelated"] = torch.tensor([-123.0])
    awac["actor_state_dict"] = OrderedDict(
        reversed(list(awac["actor_state_dict"].items()))
    )
    assert actor_state_sha256(awac) == expected


@pytest.mark.unit
def test_actor_fingerprint_changes_with_deployable_tensor_and_rejects_bad_state():
    torch = pytest.importorskip("torch")
    checkpoint = _checkpoint(torch, awac=True)
    expected = actor_state_sha256(checkpoint)
    changed = _checkpoint(torch, awac=True)
    changed["actor_state_dict"]["linear.bias"][0] += 0.25
    assert actor_state_sha256(changed) != expected

    checkpoint["actor_state_dict"] = []
    checkpoint["model_state_dict"] = _checkpoint(
        torch, awac=False
    )["model_state_dict"]
    with pytest.raises(TypeError, match="actor_state_dict"):
        actor_state_sha256(checkpoint)


@pytest.mark.unit
def test_deployable_fingerprint_normalizes_bc_and_awac_runtime_depth_contracts():
    torch = pytest.importorskip("torch")
    bc = _checkpoint(torch, awac=False)
    awac = _checkpoint(torch, awac=True)
    assert deployable_policy_sha256(bc) == deployable_policy_sha256(awac)

    awac["feature_mean"] = np.asarray(awac["feature_mean"]).copy()
    awac["feature_mean"][0] += 0.01
    assert deployable_policy_sha256(bc) != deployable_policy_sha256(awac)

    awac = _checkpoint(torch, awac=True)
    awac["depth_mask_config"]["max_patch_radius_px"] = 28
    assert deployable_policy_sha256(bc) != deployable_policy_sha256(awac)
