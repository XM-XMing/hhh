"""Public Actor-only AWAC update contracts for fixed-Critic experiments."""

from __future__ import annotations

import copy

import pytest

from planning.awac.interaction import BehaviorSource
from planning.awac.learner import (
    AWACOptimizationConfig,
    DiscreteAWACLearner,
    awac_advantage_weights,
)
from planning.awac.model import masked_policy
from planning.bc.model import build_model, require_torch
from planning.contracts.feature import NUM_ACTIONS, POLICY_VECTOR_DIM


def _batch(torch, *, batch_size: int = 6, valid_actions: int = 4):
    mask = torch.zeros((batch_size, NUM_ACTIONS), dtype=torch.bool)
    mask[:, :valid_actions] = True
    return {
        "depth": torch.rand((batch_size, 1, 32, 32)),
        "vector": torch.randn((batch_size, POLICY_VECTOR_DIM)),
        "action_mask": mask,
        "action": torch.arange(batch_size, dtype=torch.long) % valid_actions,
        "reward": torch.linspace(-1.0, 1.0, batch_size),
        "next_depth": torch.rand((batch_size, 1, 32, 32)),
        "next_vector": torch.randn((batch_size, POLICY_VECTOR_DIM)),
        "next_action_mask": mask.clone(),
        "done": torch.zeros((batch_size,), dtype=torch.float32),
        "behavior_source": torch.full(
            (batch_size,), int(BehaviorSource.BC_CALIBRATION), dtype=torch.long
        ),
    }


def _nested_equal(torch, actual, expected):
    if torch.is_tensor(expected):
        return torch.is_tensor(actual) and torch.equal(actual, expected)
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _nested_equal(torch, actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, (list, tuple)):
        return type(actual) is type(expected) and len(actual) == len(expected) and all(
            _nested_equal(torch, left, right)
            for left, right in zip(actual, expected)
        )
    return actual == expected


def _learner(torch, nn, **overrides):
    torch.manual_seed(7202)
    bc = build_model(nn, depth_channels=1)
    config = {
        "actor_vector_lr": 0.0,
        "actor_depth_lr": 0.0,
        "critic_vector_lr": 0.0,
        "critic_depth_lr": 0.0,
    }
    config.update(overrides)
    return DiscreteAWACLearner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_state_dict=bc.state_dict(),
        depth_channels=1,
        config=AWACOptimizationConfig(**config),
    )


@pytest.mark.unit
def test_actor_only_update_preserves_all_critic_and_reference_state():
    """The controlled A/B/C path must not use a disguised Critic update."""

    torch, nn, _, _, _ = require_torch()
    learner = _learner(torch, nn)
    batch = _batch(torch)
    trust = {
        name: value.clone()
        for name, value in batch.items()
        if name in {"depth", "vector", "action_mask"}
    }
    before = {
        "critic1": copy.deepcopy(learner.critic1.state_dict()),
        "critic2": copy.deepcopy(learner.critic2.state_dict()),
        "target1": copy.deepcopy(learner.target_critic1.state_dict()),
        "target2": copy.deepcopy(learner.target_critic2.state_dict()),
        "bc": copy.deepcopy(learner.bc_reference.state_dict()),
        "critic_optimizer": copy.deepcopy(learner.critic_optimizer.state_dict()),
        "critic_updates": int(learner.critic_update_count),
        "update_step": int(learner.update_step),
    }

    metrics = learner.actor_only_update(
        batch,
        actor_trust_batch=trust,
        weight_mode="uniform",
    )

    assert metrics["actor_only_update"] == 1.0
    assert metrics["actor_weight_uniform"] == 1.0
    assert learner.critic_update_count == before["critic_updates"]
    assert learner.update_step == before["update_step"]
    assert _nested_equal(torch, learner.critic1.state_dict(), before["critic1"])
    assert _nested_equal(torch, learner.critic2.state_dict(), before["critic2"])
    assert _nested_equal(torch, learner.target_critic1.state_dict(), before["target1"])
    assert _nested_equal(torch, learner.target_critic2.state_dict(), before["target2"])
    assert _nested_equal(torch, learner.bc_reference.state_dict(), before["bc"])
    assert _nested_equal(
        torch, learner.critic_optimizer.state_dict(), before["critic_optimizer"]
    )


@pytest.mark.unit
def test_actor_only_update_rejects_unknown_weight_mode_without_side_effects():
    torch, nn, _, _, _ = require_torch()
    learner = _learner(torch, nn)
    before = copy.deepcopy(learner.actor.state_dict())
    with pytest.raises(ValueError, match="weight_mode"):
        learner.actor_only_update(_batch(torch), weight_mode="not-a-mode")
    assert _nested_equal(torch, learner.actor.state_dict(), before)


@pytest.mark.unit
def test_actor_only_uniform_metrics_are_exact_ones_and_include_required_quantiles():
    """The B branch must record actual uniform weights, not a label only."""

    torch, nn, _, _, _ = require_torch()
    learner = _learner(torch, nn)
    metrics = learner.actor_only_update(_batch(torch), weight_mode="uniform")

    for prefix in ("awac_raw_weight", "awac_weight"):
        for suffix in ("mean", "p05", "p50", "p95", "p99"):
            assert metrics["{}_{}".format(prefix, suffix)] == pytest.approx(1.0)
        assert metrics["{}_ess_fraction".format(prefix)] == pytest.approx(1.0)
    assert "awac_advantage_p99" in metrics
    assert metrics["bc_top1_top2_margin_valid_count"] == 6.0
    assert metrics["bc_top1_top2_margin_na_count"] == 0.0


@pytest.mark.unit
def test_actor_only_awac_weights_use_production_function_and_no_update_measurement_is_pure():
    """C keeps production weighting while snapshot diagnostics mutate nothing."""

    torch, nn, _, _, _ = require_torch()
    learner = _learner(torch, nn)
    batch = _batch(torch)
    actor_before = copy.deepcopy(learner.actor.state_dict())
    optimizer_before = copy.deepcopy(learner.actor_optimizer.state_dict())
    with torch.no_grad():
        logits = learner.actor(batch["depth"], batch["vector"])
        probabilities, _, _ = masked_policy(logits, batch["action_mask"], torch)
        minimum_q = torch.minimum(
            learner.critic1(batch["depth"], batch["vector"]),
            learner.critic2(batch["depth"], batch["vector"]),
        )
        data_q = minimum_q.gather(1, batch["action"][:, None]).squeeze(1)
        advantage = data_q - (probabilities * minimum_q).sum(dim=1)
        expected = awac_advantage_weights(
            advantage,
            temperature=learner.config.awac_temperature,
            weight_max=learner.config.awac_weight_max,
            torch=torch,
        )
    metrics = learner.actor_only_update(
        batch,
        weight_mode="awac",
        update_actor=False,
    )

    assert metrics["actor_only_update"] == 0.0
    assert metrics["awac_raw_weight_p99"] == pytest.approx(
        float(torch.quantile(expected["raw"], 0.99).item())
    )
    assert metrics["awac_weight_p99"] == pytest.approx(
        float(torch.quantile(expected["normalized"], 0.99).item())
    )
    assert _nested_equal(torch, learner.actor.state_dict(), actor_before)
    assert _nested_equal(torch, learner.actor_optimizer.state_dict(), optimizer_before)


@pytest.mark.unit
def test_actor_only_rejection_preserves_actor_optimizer_and_fixed_critic_state():
    """A rejected proposal uses the production rollback without touching Critic."""

    torch, nn, _, _, _ = require_torch()
    learner = _learner(torch, nn, actor_head_lr=1.0e-2)
    batch = _batch(torch, batch_size=3)
    accepted = learner.actor_only_update(batch, weight_mode="uniform")
    assert accepted["actor_optimizer_step"] == 1.0
    actor_before = copy.deepcopy(learner.actor.state_dict())
    optimizer_before = copy.deepcopy(learner.actor_optimizer.state_dict())
    fixed_before = {
        "critic1": copy.deepcopy(learner.critic1.state_dict()),
        "critic2": copy.deepcopy(learner.critic2.state_dict()),
        "target1": copy.deepcopy(learner.target_critic1.state_dict()),
        "target2": copy.deepcopy(learner.target_critic2.state_dict()),
        "critic_optimizer": copy.deepcopy(learner.critic_optimizer.state_dict()),
    }
    original = learner._batch_bc_kl
    learner._batch_bc_kl = lambda unused: (torch.tensor(1000.0), torch.tensor(1000.0))
    try:
        metrics = learner.actor_only_update(batch, weight_mode="uniform")
    finally:
        learner._batch_bc_kl = original

    assert metrics["actor_trust_region_rejection"] == 1.0
    assert metrics["actor_optimizer_step"] == 0.0
    assert _nested_equal(torch, learner.actor.state_dict(), actor_before)
    assert _nested_equal(torch, learner.actor_optimizer.state_dict(), optimizer_before)
    assert _nested_equal(torch, learner.critic1.state_dict(), fixed_before["critic1"])
    assert _nested_equal(torch, learner.critic2.state_dict(), fixed_before["critic2"])
    assert _nested_equal(torch, learner.target_critic1.state_dict(), fixed_before["target1"])
    assert _nested_equal(torch, learner.target_critic2.state_dict(), fixed_before["target2"])
    assert _nested_equal(
        torch, learner.critic_optimizer.state_dict(), fixed_before["critic_optimizer"]
    )


@pytest.mark.unit
def test_actor_only_recovery_keeps_critic_frozen_but_uses_production_recovery():
    torch, nn, _, _, _ = require_torch()
    learner = _learner(
        torch,
        nn,
        actor_head_lr=1.0e-2,
        bc_kl_hard_budget=0.01,
        trust_tail_top_k=2,
    )
    batch = _batch(torch, batch_size=4, valid_actions=1)
    trust = _batch(torch, batch_size=4, valid_actions=4)
    with torch.no_grad():
        learner.actor.head[-1].bias[0] += 2.0
    _, before = learner._batch_bc_kl(trust)
    metrics = learner.actor_only_update(
        batch, actor_trust_batch={key: trust[key] for key in ("depth", "vector", "action_mask")}, weight_mode="uniform"
    )
    _, after = learner._batch_bc_kl(trust)

    assert metrics["bc_kl_budget_exceeded"] == 1.0
    assert metrics["actor_recovery_optimizer_step"] == 1.0
    assert metrics["actor_trust_region_rejection"] == 0.0
    assert float(after.item()) < float(before.item())
    assert learner.critic_update_count == 0
    assert learner.update_step == 0


@pytest.mark.unit
def test_normal_update_still_mutates_critic_and_target_after_actor_only_extraction():
    """The diagnostic seam cannot accidentally make the production update inert."""

    torch, nn, _, _, _ = require_torch()
    learner = _learner(torch, nn)
    critic_before = copy.deepcopy(learner.critic1.state_dict())
    target_before = copy.deepcopy(learner.target_critic1.state_dict())
    metrics = learner.update(_batch(torch), update_actor=False)

    assert metrics["critic1_updated"] == 1.0
    assert learner.critic_update_count == 1
    assert learner.update_step == 1
    assert not _nested_equal(torch, learner.critic1.state_dict(), critic_before)
    assert not _nested_equal(torch, learner.target_critic1.state_dict(), target_before)
