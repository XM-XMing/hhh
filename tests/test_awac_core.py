"""CPU-only contracts for the independent masked discrete AWAC learner."""

from __future__ import annotations

import copy
import math

import pytest

from planning.awac.learner import (
    AWAC_LEARNER_STATE_SCHEMA_ID,
    AWACOptimizationConfig,
    DiscreteAWACLearner,
    awac_advantage_weights,
)
from planning.bc.model import build_model, require_torch
from planning.contracts.feature import NUM_ACTIONS, POLICY_VECTOR_DIM
from planning.awac.interaction import BehaviorSource
from planning.awac.model import masked_policy


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
        "behavior_source": torch.tensor(
            [
                int(BehaviorSource.BC_WARMUP),
                int(BehaviorSource.ACCEPTED_COLLECTION_POLICY),
            ]
            * ((batch_size + 1) // 2),
            dtype=torch.int64,
        )[:batch_size],
    }


def _learner(torch, nn, **overrides):
    torch.manual_seed(112)
    bc = build_model(nn, depth_channels=1)
    defaults = {
        "actor_vector_lr": 0.0,
        "actor_depth_lr": 0.0,
        "critic_vector_lr": 0.0,
        "critic_depth_lr": 0.0,
    }
    defaults.update(overrides)
    return bc, DiscreteAWACLearner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_state_dict=bc.state_dict(),
        depth_channels=1,
        config=AWACOptimizationConfig(**defaults),
    )


@pytest.mark.unit
def test_awac_initializes_from_bc_and_copies_frozen_encoder():
    torch, nn, _, _, _ = require_torch()
    bc, learner = _learner(torch, nn)
    for name, value in bc.state_dict().items():
        assert torch.equal(value, learner.actor.state_dict()[name])
    assert not any(parameter.requires_grad for parameter in learner.bc_reference.parameters())
    for prefix in ("depth_encoder", "vector_encoder"):
        source = getattr(learner.actor, prefix).state_dict()
        for critic in (learner.critic1, learner.critic2):
            assert all(
                torch.equal(value, getattr(critic, prefix).state_dict()[name])
                for name, value in source.items()
            )
    assert not hasattr(learner, "alpha")
    assert not hasattr(learner, "log_alpha")


@pytest.mark.unit
def test_awac_bellman_target_is_full_mask_and_has_no_entropy_term():
    torch, nn, _, _, _ = require_torch()
    _, learner = _learner(torch, nn, gamma=0.5)
    batch = _batch(torch, batch_size=2, valid_actions=4)
    batch["reward"][:] = torch.tensor([1.0, -2.0])
    with torch.no_grad():
        for parameter in learner.actor.parameters():
            parameter.zero_()
        action_values = torch.arange(NUM_ACTIONS, dtype=torch.float32)
        for critic in (learner.target_critic1, learner.target_critic2):
            for parameter in critic.parameters():
                parameter.zero_()
            critic.head[-1].bias.copy_(action_values)
    # Uniform Actor over all four valid actions gives E[Q]=1.5.  A top-3
    # backup would be 1.0 and an entropy-regularized target would be larger.
    metrics = learner.update(batch, update_actor=False)
    # The formal reward contract stores raw environment rewards and applies
    # the configured 0.10 scale exactly once in the Bellman target.
    expected = ((0.1 * 1.0 + 0.5 * 1.5) + (0.1 * -2.0 + 0.5 * 1.5)) / 2.0
    assert metrics["target_policy_support_size_mean"] == 4.0
    assert math.isclose(metrics["next_value_mean"], 1.5, abs_tol=1.0e-6)
    assert math.isclose(metrics["target_q_mean"], expected, abs_tol=1.0e-6)


@pytest.mark.unit
def test_awac_loss_uses_replay_data_action_and_terminal_empty_mask_is_safe():
    torch, nn, _, _, _ = require_torch()
    _, learner = _learner(torch, nn)
    batch = _batch(torch, batch_size=3, valid_actions=4)
    batch["done"][2] = 1.0
    batch["next_action_mask"][2] = False
    with torch.no_grad():
        logits = learner.actor(batch["depth"], batch["vector"])
        _, log_probabilities, _ = masked_policy(logits, batch["action_mask"], torch)
        expected_data_logp = log_probabilities.gather(
            1, batch["action"][:, None]
        ).mean()
    metrics = learner.update(batch, update_actor=False)
    assert math.isclose(
        metrics["awac_data_log_probability_mean"],
        float(expected_data_logp.item()),
        abs_tol=1.0e-6,
    )
    assert metrics["target_policy_empty_mask_rate"] == pytest.approx(1.0 / 3.0)
    batch["done"][2] = 0.0
    with pytest.raises(ValueError, match="non-terminal.*no valid next action"):
        learner.update(batch, update_actor=False)


@pytest.mark.unit
def test_awac_terminal_zero_mask_does_not_evaluate_next_state_models(monkeypatch):
    torch, nn, _, _, _ = require_torch()
    _, learner = _learner(torch, nn)
    batch = _batch(torch, batch_size=2, valid_actions=4)
    batch["done"][:] = 1.0
    batch["next_action_mask"][:] = False
    sentinel = 12345.0
    batch["next_vector"][:, 0] = sentinel

    def reject_terminal_next_input(forward):
        def guarded(depth, vector):
            if bool((vector[:, 0] == sentinel).any()):
                raise AssertionError("terminal next-state model evaluation is forbidden")
            return forward(depth, vector)

        return guarded

    monkeypatch.setattr(
        learner.actor,
        "forward",
        reject_terminal_next_input(learner.actor.forward),
    )
    for critic in (learner.target_critic1, learner.target_critic2):
        monkeypatch.setattr(
            critic,
            "forward",
            reject_terminal_next_input(critic.forward),
        )

    metrics = learner.update(batch, update_actor=False)

    assert metrics["next_value_mean"] == pytest.approx(0.0)
    assert metrics["target_q_mean"] == pytest.approx(
        float((batch["reward"] * 0.10).mean().item())
    )
    assert metrics["target_policy_empty_mask_rate"] == pytest.approx(1.0)


@pytest.mark.unit
def test_awac_weights_are_stable_positive_capped_and_observable():
    torch, _, _, _, _ = require_torch()
    advantage = torch.tensor([-1.0e6, -2.0, 0.0, 2.0, 1.0e6])
    result = awac_advantage_weights(
        advantage, temperature=0.5, weight_max=7.0, torch=torch
    )
    raw = result["raw"]
    normalized = result["normalized"]
    assert torch.isfinite(raw).all() and torch.isfinite(normalized).all()
    assert bool((raw > 0).all()) and bool((normalized > 0).all())
    assert float(raw.max()) <= 7.0 + 1.0e-6
    assert float(normalized.max()) <= 7.0 + 1.0e-6
    assert bool((raw[1:] >= raw[:-1]).all())
    assert bool(result["raw_high_clip"][-1])
    assert bool(result["raw_low_floor"][0])
    with pytest.raises(ValueError, match="temperature"):
        awac_advantage_weights(advantage, temperature=0.0, weight_max=7.0, torch=torch)
    with pytest.raises(ValueError, match="weight_max"):
        awac_advantage_weights(advantage, temperature=1.0, weight_max=0.9, torch=torch)


@pytest.mark.unit
def test_awac_stratified_metrics_and_effective_sample_size_are_complete():
    torch, nn, _, _, _ = require_torch()
    _, learner = _learner(torch, nn)
    batch = _batch(torch, batch_size=6)
    batch["done"][:] = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.float32)
    batch["reward"][:] = torch.tensor([0, -1, 2, -2, 3, 0], dtype=torch.float32)
    metrics = learner.update(batch, update_actor=False)
    assert metrics["awac_stratum_nonterminal_count"] == 3.0
    assert metrics["awac_stratum_terminal_positive_reward_count"] == 2.0
    assert metrics["awac_stratum_terminal_nonpositive_reward_count"] == 1.0
    assert metrics["awac_stratum_behavior_bc_warmup_count"] == 3.0
    assert metrics["awac_stratum_behavior_accepted_collection_policy_count"] == 3.0
    terminal_mass = sum(
        metrics["awac_stratum_{}_weight_mass_fraction".format(name)]
        for name in (
            "nonterminal",
            "terminal_positive_reward",
            "terminal_nonpositive_reward",
        )
    )
    behavior_mass = sum(
        metrics["awac_stratum_behavior_{}_weight_mass_fraction".format(name)]
        for name in ("bc_warmup", "accepted_collection_policy")
    )
    assert terminal_mass == pytest.approx(1.0, abs=1.0e-6)
    assert behavior_mass == pytest.approx(1.0, abs=1.0e-6)
    assert 0.0 < metrics["awac_weight_ess_fraction"] <= 1.0
    assert all(math.isfinite(float(value)) for value in metrics.values())


@pytest.mark.unit
def test_awac_kl_rejection_restores_actor_and_optimizer_exactly():
    torch, nn, _, _, _ = require_torch()
    _, learner = _learner(
        torch,
        nn,
        actor_head_lr=1.0e-2,
        bc_kl_hard_budget=100.0,
    )
    batch = _batch(torch, batch_size=3)
    warmup = learner.update(batch, update_actor=True)
    assert warmup["actor_awac_optimizer_step"] == 1.0
    assert learner.actor_optimizer.state
    actor_before = copy.deepcopy(learner.actor.state_dict())
    optimizer_before = copy.deepcopy(learner.actor_optimizer.state_dict())
    original = learner._batch_bc_kl
    learner._batch_bc_kl = lambda unused: (
        torch.tensor(1000.0),
        torch.tensor(1000.0),
    )
    try:
        metrics = learner.update(batch, update_actor=True)
    finally:
        learner._batch_bc_kl = original
    assert metrics["actor_optimizer_step"] == 0.0
    assert metrics["actor_trust_region_rejection"] == 1.0
    assert learner.actor_awac_update_count == 1
    assert learner.actor_trust_region_rejection_count == 1
    assert all(
        torch.equal(actor_before[name], value)
        for name, value in learner.actor.state_dict().items()
    )

    def assert_nested_equal(actual, expected):
        if torch.is_tensor(expected):
            assert torch.is_tensor(actual)
            assert torch.equal(actual, expected)
            return
        if isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                assert_nested_equal(actual[key], expected[key])
            return
        if isinstance(expected, (list, tuple)):
            assert type(actual) is type(expected)
            assert len(actual) == len(expected)
            for actual_item, expected_item in zip(actual, expected):
                assert_nested_equal(actual_item, expected_item)
            return
        assert actual == expected

    assert_nested_equal(learner.actor_optimizer.state_dict(), optimizer_before)


@pytest.mark.unit
def test_awac_recovery_backpropagates_through_violating_trust_tail():
    torch, nn, _, _, _ = require_torch()
    _, learner = _learner(
        torch,
        nn,
        actor_head_lr=1.0e-2,
        bc_kl_hard_budget=0.01,
        trust_tail_top_k=2,
    )
    # The ordinary batch has a single valid action and therefore zero KL.
    # Only the independent trust states expose the deliberately perturbed
    # Actor distribution. Recovery must optimize those trust rows directly.
    batch = _batch(torch, batch_size=4, valid_actions=1)
    trust_batch = _batch(torch, batch_size=4, valid_actions=4)
    with torch.no_grad():
        learner.actor.head[-1].bias[0] += 2.0
    _, trust_max_before = learner._batch_bc_kl(trust_batch)
    assert float(trust_max_before.item()) > learner.config.bc_kl_hard_budget

    metrics = learner.update(
        batch,
        update_actor=True,
        actor_trust_batch=trust_batch,
    )
    _, trust_max_after = learner._batch_bc_kl(trust_batch)

    assert metrics["bc_kl_batch_max_before_update"] == pytest.approx(0.0)
    assert metrics["bc_kl_budget_exceeded"] == 1.0
    assert metrics["trust_tail_gradient_enabled"] == 1.0
    assert metrics["trust_tail_selected_count"] == 2.0
    assert metrics["trust_tail_bc_kl_max_before_update"] > 0.01
    assert metrics["actor_recovery_optimizer_step"] == 1.0
    assert metrics["actor_trust_region_rejection"] == 0.0
    assert float(trust_max_after.item()) < float(trust_max_before.item())


@pytest.mark.unit
def test_awac_state_schema_roundtrip_is_independent_and_strict():
    torch, nn, _, _, _ = require_torch()
    bc, learner = _learner(torch, nn)
    learner.update(_batch(torch, batch_size=3), update_actor=False)
    state = learner.state_dict()
    assert state["awac_learner_state_schema_id"] == AWAC_LEARNER_STATE_SCHEMA_ID
    assert "log_alpha" not in state
    assert "alpha_optimizer_state_dict" not in state
    restored = DiscreteAWACLearner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_state_dict=bc.state_dict(),
        depth_channels=1,
        config=learner.config,
    )
    restored.load_state_dict(state)
    assert restored.update_step == learner.update_step
    assert restored.actor_awac_update_count == learner.actor_awac_update_count
    assert restored.policy_update_count == learner.actor_awac_update_count
    invalid = copy.deepcopy(state)
    invalid["awac_learner_state_schema_id"] = "foreign_state"
    with pytest.raises(ValueError, match="schema"):
        restored.load_state_dict(invalid)


@pytest.mark.unit
def test_awac_rejects_invalid_config_and_batch():
    torch, nn, _, _, _ = require_torch()
    with pytest.raises(ValueError, match="awac_temperature"):
        AWACOptimizationConfig(awac_temperature=float("nan"))
    with pytest.raises(ValueError, match="awac_weight_max"):
        AWACOptimizationConfig(awac_weight_max=0.5)
    with pytest.raises(ValueError, match="trust_tail_top_k"):
        AWACOptimizationConfig(trust_tail_top_k=0)
    _, learner = _learner(torch, nn)
    batch = _batch(torch, batch_size=3)
    invalid = copy.deepcopy(batch)
    invalid["reward"][0] = float("nan")
    with pytest.raises(ValueError, match="reward contains non-finite"):
        learner.update(invalid, update_actor=False)
    invalid = copy.deepcopy(batch)
    invalid["action_mask"][0] = False
    with pytest.raises(ValueError, match="no valid current action"):
        learner.update(invalid, update_actor=False)
    invalid = copy.deepcopy(batch)
    invalid["behavior_source"][0] = 99
    with pytest.raises(ValueError, match="unknown"):
        learner.update(invalid, update_actor=False)
