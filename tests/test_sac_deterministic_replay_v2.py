from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from planning.sac.diagnostics import SACUpdateJournal, stable_fingerprint
from planning.sac.network import DiscreteSACAgent
from planning.sac.replay import SACReplayBuffer


def _bc_checkpoint():
    from planning.bc.model import build_model

    torch.manual_seed(7)
    model = build_model(nn, depth_channels=1, vec_dim=127, num_actions=105)
    return {"model_state_dict": deepcopy(model.state_dict()), "depth_history_frames": 1}


def _agent():
    return DiscreteSACAgent(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_checkpoint=_bc_checkpoint(),
        actor_lr=1.0e-5,
        critic_lr=1.0e-4,
    )


def _batch(size=2, seed=17):
    torch.manual_seed(seed)
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
        "done": torch.zeros((size,)),
    }


def test_replay_indices_are_reconstructible(tmp_path):
    replay = SACReplayBuffer.create(
        tmp_path / "replay",
        capacity=4,
        depth_shape=(90, 160),
        vector_dim=127,
        action_dim=105,
        provenance={},
    )
    mask = np.ones((105,), dtype=np.bool_)
    rows = []
    for step in range(2):
        rows.append(
            {
                "depth": np.zeros((1, 90, 160), dtype=np.float16),
                "vector": np.zeros((127,), dtype=np.float32),
                "action_mask": mask,
                "action": 0,
                "reward": 0.0,
                "next_depth": np.zeros((1, 90, 160), dtype=np.float16),
                "next_vector": np.zeros((127,), dtype=np.float32),
                "next_action_mask": mask,
                "done": False,
                "mission_id": "m",
                "episode_id": "e",
                "step_id": step,
                "behavior_policy_version": "v",
                "termination_reason": "",
                "runtime_instance_id": "r",
                "transition_id": "r:e:{}".format(step),
                "actor_log_prob": 0.0,
                "policy_entropy": 1.0,
                "bc_kl": 0.0,
                "realized_return": 0.0,
                "argmax_flip": False,
                "valid_action_count": 105,
                "action_mask_valid": True,
            }
        )
    replay.add_batch(rows)
    rng = np.random.RandomState(9)
    indices = replay.sample_indices(2, rng=rng)
    batch = replay.batch_from_indices(indices, torch=torch, device=torch.device("cpu"))
    assert indices.dtype == np.int64
    assert batch["action"].shape == (2,)
    replay.close()


def test_rejected_actor_proposal_restores_parameters_and_optimizer():
    torch.set_num_threads(1)
    agent = _agent()
    batch = _batch(seed=18)
    actor_before = {key: value.detach().clone() for key, value in agent.actor.state_dict().items()}
    optimizer_before = deepcopy(agent.actor_optimizer.state_dict())
    critic_before = stable_fingerprint(agent.critic1.state_dict())
    result = agent.update_actor_transactional(batch, hard_stop=-1.0, entropy_floor=0.0)
    assert result["accepted"] is False
    assert result["rollback_performed"] is False
    for key, value in agent.actor.state_dict().items():
        assert torch.equal(value, actor_before[key])
    assert stable_fingerprint(agent.actor_optimizer.state_dict()) == stable_fingerprint(optimizer_before)
    assert stable_fingerprint(agent.critic1.state_dict()) == critic_before
    assert agent.actor_update_count == 0
    assert agent.actor_rejection_count == 1
    assert result["pre_step_kl_used"] is True
    assert result["post_step_hard_stop_checked"] is False


def test_pre_step_unsafe_proposal_never_calls_optimizer_step():
    torch.set_num_threads(1)
    agent = _agent()
    batch = _batch()
    calls = []
    original_step = agent.actor_optimizer.step

    def wrapped_step(*args, **kwargs):
        calls.append(True)
        return original_step(*args, **kwargs)

    agent.actor_optimizer.step = wrapped_step
    result = agent.update_actor_transactional(batch, hard_stop=-1.0, entropy_floor=0.0)
    assert result["accepted"] is False
    assert result["hard_stop_result"] == "REJECT_PRE_KL"
    assert result["pre_step_hard_stop"] is True
    assert result["post_step_hard_stop_checked"] is False
    assert calls == []
    assert result["rollback_performed"] is False
    assert agent.actor_update_count == 0
    assert agent.actor_rejection_count == 1


def test_post_step_unsafe_proposal_rolls_back_actor_and_optimizer():
    torch.set_num_threads(1)
    agent = _agent()
    batch = _batch(seed=18)
    initial = deepcopy(agent.state_payload())
    probe = agent.update_actor_transactional(batch, hard_stop=float("inf"), entropy_floor=0.0)
    assert probe["bc_kl_post_max"] > probe["bc_kl_max"]
    agent.actor.load_state_dict(initial["actor_state_dict"], strict=True)
    agent.bc_reference.load_state_dict(initial["bc_reference_state_dict"], strict=True)
    agent.critic1.load_state_dict(initial["critic1_state_dict"], strict=True)
    agent.critic2.load_state_dict(initial["critic2_state_dict"], strict=True)
    agent.target_critic1.load_state_dict(initial["target_critic1_state_dict"], strict=True)
    agent.target_critic2.load_state_dict(initial["target_critic2_state_dict"], strict=True)
    agent.actor_optimizer.load_state_dict(initial["actor_optimizer_state_dict"])
    agent.critic_optimizer.load_state_dict(initial["critic_optimizer_state_dict"])
    for name in (
        "actor_optimizer_step_count", "critic_optimizer_step_count", "actor_update_count",
        "critic_update_count", "actor_proposal_count", "actor_rejection_count",
        "actor_recovery_update_count", "nan_count", "invalid_action_count",
    ):
        setattr(agent, name, int(initial[name]))
    actor_before = stable_fingerprint(agent.actor.state_dict())
    optimizer_before = stable_fingerprint(agent.actor_optimizer.state_dict())
    threshold = (probe["bc_kl_max"] + probe["bc_kl_post_max"]) / 2.0
    result = agent.update_actor_transactional(batch, hard_stop=threshold, entropy_floor=0.0)
    assert result["accepted"] is False
    assert result["hard_stop_result"] == "REJECT_POST_KL"
    assert result["pre_step_hard_stop"] is False
    assert result["post_step_hard_stop_checked"] is True
    assert result["rollback_performed"] is True
    assert stable_fingerprint(agent.actor.state_dict()) == actor_before
    assert stable_fingerprint(agent.actor_optimizer.state_dict()) == optimizer_before
    assert agent.actor_update_count == 0
    assert agent.actor_rejection_count == 1


def test_journal_is_append_only_and_ignores_partial_tail(tmp_path):
    path = tmp_path / "training_step_journal.jsonl"
    journal = SACUpdateJournal(path)
    journal.append({"update_kind": "critic", "update_index": 1, "batch_indices": [1, 2]})
    journal.close()
    with path.open("ab") as stream:
        stream.write(b'{"schema_id":"bc_initialized_discrete_sac_update_journal_v2"')
    reopened = SACUpdateJournal(path)
    assert reopened.record_count == 1
    reopened.close()


def test_fixed_cpu_critic_actor_sequence_is_exact_for_100_updates():
    torch.set_num_threads(1)
    batch = _batch(size=2)
    left = _agent()
    right = _agent()
    right.actor.load_state_dict(deepcopy(left.actor.state_dict()))
    right.bc_reference.load_state_dict(deepcopy(left.bc_reference.state_dict()))
    right.critic1.load_state_dict(deepcopy(left.critic1.state_dict()))
    right.critic2.load_state_dict(deepcopy(left.critic2.state_dict()))
    right.target_critic1.load_state_dict(deepcopy(left.target_critic1.state_dict()))
    right.target_critic2.load_state_dict(deepcopy(left.target_critic2.state_dict()))
    right.actor_optimizer.load_state_dict(deepcopy(left.actor_optimizer.state_dict()))
    right.critic_optimizer.load_state_dict(deepcopy(left.critic_optimizer.state_dict()))
    for _ in range(100):
        left.update_critic(batch)
        left.update_actor(batch)
        right.update_critic(batch)
        right.update_actor(batch)
    assert stable_fingerprint(left.state_payload()) == stable_fingerprint(right.state_payload())
