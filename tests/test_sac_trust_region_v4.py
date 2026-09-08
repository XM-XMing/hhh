from __future__ import annotations

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from planning.sac.diagnostics import stable_fingerprint
from planning.sac.network import DiscreteSACAgent


FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)


def _agent():
    from planning.bc.model import build_model

    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = build_model(nn, depth_channels=1, vec_dim=127, num_actions=105)
    checkpoint = {"model_state_dict": deepcopy(model.state_dict()), "depth_history_frames": 1}
    return DiscreteSACAgent(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_checkpoint=checkpoint,
        actor_lr=1.0e-5,
        critic_lr=1.0e-4,
    )


def _batch(seed=18, size=2):
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


def test_trust_region_uses_fixed_factors_and_accepts_safe_factor():
    agent = _agent()
    batch = _batch()
    agent.set_kl_sentinel(batch)
    probe = agent.update_actor_transactional(
        batch, hard_stop=float("inf"), entropy_floor=0.0, trust_region_factors=FACTORS
    )
    assert probe["raw_factor_1_kl_max"] > probe["pre_sentinel_kl_max"]
    threshold = float(probe["raw_factor_1_kl_max"]) * 0.9

    agent = _agent()
    agent.set_kl_sentinel(batch)
    before_actor = stable_fingerprint(agent.actor.state_dict())
    result = agent.update_actor_transactional(
        batch, hard_stop=threshold, entropy_floor=0.0, trust_region_factors=FACTORS
    )
    assert result["accepted"] is True
    safe_result = next(item for item in result["factor_results"] if item["safe"])
    assert result["accepted_lr_factor"] == pytest.approx(safe_result["factor"])
    assert result["backtrack_count"] == FACTORS.index(safe_result["factor"])
    assert result["sentinel_kl_max"] < threshold
    assert result["raw_factor_1_kl_max"] >= threshold
    assert result["gradient_source_sha256"]
    assert stable_fingerprint(agent.actor.state_dict()) != before_actor
    assert all(group["lr"] == pytest.approx(1.0e-5) for group in agent.actor_optimizer.param_groups)


def test_failed_factors_restore_actor_optimizer_and_rng_bitwise():
    agent = _agent()
    batch = _batch()
    agent.set_kl_sentinel(batch)
    initial_actor = stable_fingerprint(agent.actor.state_dict())
    initial_optimizer = stable_fingerprint(agent.actor_optimizer.state_dict())
    rng = {"counter": 0}

    def capture():
        return dict(rng)

    def restore(value):
        rng.clear()
        rng.update(value)

    agent.set_actor_rng_hooks(capture, restore)
    result = agent.update_actor_transactional(
        batch, hard_stop=-1.0, entropy_floor=0.0, trust_region_factors=FACTORS
    )
    assert result["accepted"] is False
    assert result["hard_stop_result"] == "REJECT_PRE_KL"
    assert result["optimizer_step_performed"] is False
    assert stable_fingerprint(agent.actor.state_dict()) == initial_actor
    assert stable_fingerprint(agent.actor_optimizer.state_dict()) == initial_optimizer
    assert rng == {"counter": 0}


def test_all_backtracking_factors_fail_and_stop_without_update():
    agent = _agent()
    batch = _batch()
    agent.set_kl_sentinel(batch)
    initial_actor = stable_fingerprint(agent.actor.state_dict())
    initial_optimizer = stable_fingerprint(agent.actor_optimizer.state_dict())
    result = agent.update_actor_transactional(
        batch, hard_stop=0.0, entropy_floor=0.0, trust_region_factors=FACTORS
    )
    assert result["accepted"] is False
    assert result["hard_stop_result"] == "REJECT_PRE_KL"
    assert result["accepted_lr_factor"] is None
    assert stable_fingerprint(agent.actor.state_dict()) == initial_actor
    assert stable_fingerprint(agent.actor_optimizer.state_dict()) == initial_optimizer


def test_all_factors_fail_after_safe_pre_state_and_restore_transaction():
    agent = _agent()
    batch = _batch()
    agent.set_kl_sentinel(batch)
    initial_actor = stable_fingerprint(agent.actor.state_dict())
    initial_optimizer = stable_fingerprint(agent.actor_optimizer.state_dict())
    result = agent.update_actor_transactional(
        batch, hard_stop=1.0e-12, entropy_floor=0.0, trust_region_factors=FACTORS
    )
    assert result["accepted"] is False
    assert result["hard_stop_result"] == "REJECT_BACKTRACK_KL"
    assert len(result["factor_results"]) == len(FACTORS)
    assert result["optimizer_step_performed"] is True
    assert result["rollback_performed"] is True
    assert stable_fingerprint(agent.actor.state_dict()) == initial_actor
    assert stable_fingerprint(agent.actor_optimizer.state_dict()) == initial_optimizer
    assert all(group["lr"] == pytest.approx(1.0e-5) for group in agent.actor_optimizer.param_groups)


def test_each_factor_reuses_one_gradient_source_and_sentinel_is_fixed():
    agent = _agent()
    batch = _batch()
    agent.set_kl_sentinel(batch)
    sentinel_before = stable_fingerprint(agent._kl_sentinel)
    result = agent.update_actor_transactional(
        batch, hard_stop=1.0e-12, entropy_floor=0.0, trust_region_factors=FACTORS
    )
    assert result["gradient_source_sha256"]
    assert result["actor_proposal_id"] == 1
    assert stable_fingerprint(agent._kl_sentinel) == sentinel_before
    assert [item["factor"] for item in result["factor_results"]] == list(FACTORS)


def test_sentinel_is_copied_and_bc_reference_remains_frozen():
    agent = _agent()
    batch = _batch(size=2)
    original = batch["vector"].clone()
    manifest = agent.set_kl_sentinel(batch)
    batch["vector"].zero_()
    stats = agent.kl_sentinel_stats()
    assert manifest["state_count"] == 2
    assert stats["count"] == 2
    assert torch.equal(agent._kl_sentinel["vector"], original)
    assert all(not parameter.requires_grad for parameter in agent.bc_reference.parameters())
