from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from planning.bc.model import build_model
from planning.contracts.offpolicy import SAC_OFFPOLICY_SPEC, get_offpolicy_algorithm_spec
from planning.sac.network import (
    DiscreteSACAgent,
    discrete_value_from_policy,
    masked_categorical,
)
from planning.sac.replay import SACReplayBuffer
from planning.sac.runtime import SACOnlineRunner, SACSafetyStop


def _bc_checkpoint():
    torch.manual_seed(7)
    model = build_model(nn, depth_channels=1, vec_dim=127, num_actions=105)
    return {
        "model_state_dict": deepcopy(model.state_dict()),
        "depth_history_frames": 1,
    }


@pytest.fixture()
def agent():
    torch.set_num_threads(1)
    return DiscreteSACAgent(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_checkpoint=_bc_checkpoint(),
        actor_lr=1.0e-5,
        critic_lr=1.0e-4,
    )


def _batch(size=4, *, done=False):
    mask = torch.ones((size, 105), dtype=torch.bool)
    return {
        "depth": torch.randn((size, 1, 90, 160)),
        "vector": torch.randn((size, 127)),
        "action_mask": mask,
        "action": torch.zeros((size,), dtype=torch.long),
        "reward": torch.zeros((size,)),
        "next_depth": torch.randn((size, 1, 90, 160)),
        "next_vector": torch.randn((size, 127)),
        "next_action_mask": mask.clone(),
        "done": torch.full((size,), float(done)),
    }


def _transition(step, *, done=False):
    mask = np.ones((105,), dtype=np.bool_)
    return {
        "depth": np.zeros((1, 90, 160), dtype=np.float16),
        "vector": np.zeros((127,), dtype=np.float32),
        "action_mask": mask,
        "action": 0,
        "reward": 1.0,
        "next_depth": np.zeros((1, 90, 160), dtype=np.float16),
        "next_vector": np.zeros((127,), dtype=np.float32),
        "next_action_mask": np.zeros((105,), dtype=np.bool_) if done else mask,
        "done": done,
        "mission_id": "m0",
        "episode_id": "e0",
        "step_id": step,
        "behavior_policy_version": "sac_masked_categorical_v1",
        "termination_reason": "timeout" if done else "",
        "runtime_instance_id": "runtime-0",
        "transition_id": "runtime-0:e0:{}".format(step),
        "actor_log_prob": -1.0,
        "policy_entropy": 1.0,
        "bc_kl": 0.0,
        "realized_return": 1.0,
        "argmax_flip": False,
        "valid_action_count": 105,
        "action_mask_valid": True,
    }


def test_masked_categorical_has_zero_probability_for_invalid_actions():
    logits = torch.zeros((1, 105))
    logits[0, 0] = 1.0
    logits[0, 2] = -4.0
    mask = torch.zeros((1, 105), dtype=torch.bool)
    mask[0, 0] = True
    mask[0, 2] = True
    probabilities, log_probs, valid = masked_categorical(logits, mask, torch)
    assert bool(valid[0, 0]) and bool(valid[0, 2])
    assert probabilities[0, 1].item() == 0.0
    assert torch.isfinite(log_probs).all()
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))


def test_discrete_value_uses_masked_distribution():
    q = torch.tensor([[2.0, 1000.0, 4.0]])
    probabilities = torch.tensor([[0.25, 0.0, 0.75]])
    log_probs = torch.log(torch.tensor([[0.25, 1.0, 0.75]]))
    value = discrete_value_from_policy(q, probabilities, log_probs, 0.0, torch)
    assert value.item() == pytest.approx(3.5)


def test_sac_registers_as_distinct_offpolicy_algorithm():
    assert get_offpolicy_algorithm_spec(SAC_OFFPOLICY_SPEC.algorithm_id) == SAC_OFFPOLICY_SPEC
    assert SAC_OFFPOLICY_SPEC.model_type == "sac_discrete"


def test_actor_is_initialized_from_bc_and_reference_is_frozen(agent):
    assert agent.actor_fingerprint() == agent.bc_reference_fingerprint()
    assert all(not parameter.requires_grad for parameter in agent.bc_reference.parameters())
    assert agent.critic_update_count == 0
    assert agent.actor_update_count == 0


def test_masked_action_selection_respects_mask(agent):
    depth = torch.zeros((1, 1, 90, 160))
    vector = torch.zeros((1, 127))
    mask = torch.zeros((1, 105), dtype=torch.bool)
    mask[0, 7] = True
    mask[0, 44] = True
    result = agent.select_action(
        depth, vector, mask, np.random.RandomState(3), warmup=True
    )
    assert result["action"] in (7, 44)
    assert result["valid_action_count"] == 2


def test_actor_update_does_not_step_critics_or_targets(agent):
    before_critic = {key: value.detach().clone() for key, value in agent.critic1.state_dict().items()}
    before_critic2 = {key: value.detach().clone() for key, value in agent.critic2.state_dict().items()}
    before_target = {key: value.detach().clone() for key, value in agent.target_critic1.state_dict().items()}
    before_critic_optimizer = deepcopy(agent.critic_optimizer.state_dict())
    result = agent.update_actor(_batch())
    assert np.isfinite(result["actor_loss"])
    for key, value in agent.critic1.state_dict().items():
        assert torch.equal(value, before_critic[key])
    for key, value in agent.critic2.state_dict().items():
        assert torch.equal(value, before_critic2[key])
    for key, value in agent.target_critic1.state_dict().items():
        assert torch.equal(value, before_target[key])
    assert agent.critic_update_count == 0
    assert agent.critic_optimizer.state_dict() == before_critic_optimizer


def test_terminal_critic_target_skips_empty_successor_mask(agent):
    result = agent.update_critic(_batch(done=True))
    assert np.isfinite(result["critic1_loss"])
    assert agent.critic_update_count == 1


def test_entropy_floor_ignores_structurally_single_action_rows():
    runner = object.__new__(SACOnlineRunner)
    runner.bc_kl_hard_stop = 1.0
    runner.entropy_floor = 0.02
    runner._check_metrics(
        {"bc_kl_max": 0.0, "entropy_min": 0.0, "entropy_floor_eligible_count": 0},
        actor=True,
    )


def test_entropy_floor_still_rejects_collapsed_choice_state():
    runner = object.__new__(SACOnlineRunner)
    runner.bc_kl_hard_stop = 1.0
    runner.entropy_floor = 0.02
    with pytest.raises(SACSafetyStop, match="entropy floor"):
        runner._check_metrics(
            {"bc_kl_max": 0.0, "entropy_min": 0.01, "entropy_floor_eligible_count": 1},
            actor=True,
        )


def test_actor_entropy_diagnostic_excludes_single_action_state(agent):
    batch = _batch(size=2)
    batch["action_mask"][0] = False
    batch["action_mask"][0, 0] = True
    result = agent.update_actor(batch)
    assert result["entropy_floor_eligible_count"] == 1


def test_replay_persists_identity_and_rejects_duplicate(tmp_path):
    replay = SACReplayBuffer.create(
        tmp_path / "replay",
        capacity=8,
        depth_shape=(90, 160),
        vector_dim=127,
        action_dim=105,
        provenance={"source_bc_checkpoint_sha256": "a" * 64},
    )
    replay.add_batch([_transition(0), _transition(1, done=True)])
    assert replay.audit()["identity_complete"] is True
    assert replay.audit()["rows"] == 2
    with pytest.raises(ValueError, match="duplicate SAC transition_id"):
        replay.add_batch([_transition(1, done=True)])
    replay.close()


def test_replay_rejects_action_outside_mask(tmp_path):
    replay = SACReplayBuffer.create(
        tmp_path / "replay",
        capacity=2,
        depth_shape=(90, 160),
        vector_dim=127,
        action_dim=105,
        provenance={},
    )
    bad = _transition(0)
    bad["action"] = 4
    bad["action_mask"] = np.zeros((105,), dtype=np.bool_)
    with pytest.raises(ValueError, match="outside current mask"):
        replay.add_batch([bad])
    replay.close()
