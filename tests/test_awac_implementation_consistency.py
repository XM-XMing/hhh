"""Regression tests for the implementation-consistency recertification seam."""

from __future__ import annotations

import copy

import numpy as np
import pytest


pytestmark = pytest.mark.unit


def _learner(torch, *, critic_scale: float = 1.0, adaptive: bool = False, budget=0.10):
    from planning.awac.learner import AWACOptimizationConfig, DiscreteAWACLearner
    from planning.bc.model import build_model

    torch.manual_seed(1201)
    bc = build_model(torch.nn, depth_channels=1)
    return DiscreteAWACLearner(
        torch=torch,
        nn=torch.nn,
        device=torch.device("cpu"),
        bc_state_dict=bc.state_dict(),
        depth_channels=1,
        config=AWACOptimizationConfig(
            critic_head_lr=1.0e-4 * critic_scale,
            critic_vector_lr=1.0e-5 * critic_scale,
            critic_depth_lr=1.0e-5 * critic_scale,
            enable_twin_q_confidence=bool(adaptive),
            enable_adaptive_bc_kl=bool(adaptive),
            adaptive_bc_kl_beta_min=0.05 if adaptive else 0.02,
            adaptive_bc_kl_beta_max=0.05 if adaptive else 0.10,
            bc_kl_hard_budget=float(budget),
            actor_vector_lr=0.0,
            actor_depth_lr=0.0,
        ),
    )


def _batch(torch):
    from planning.awac.interaction import BehaviorSource
    from planning.contracts.feature import NUM_ACTIONS, POLICY_VECTOR_DIM

    mask = torch.zeros((4, NUM_ACTIONS), dtype=torch.bool)
    mask[:, :4] = True
    return {
        "depth": torch.rand((4, 1, 32, 32)),
        "vector": torch.randn((4, POLICY_VECTOR_DIM)),
        "action_mask": mask,
        "action": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "reward": torch.tensor([0.5, -0.25, 0.1, -0.2]),
        "next_depth": torch.rand((4, 1, 32, 32)),
        "next_vector": torch.randn((4, POLICY_VECTOR_DIM)),
        "next_action_mask": mask.clone(),
        "done": torch.zeros((4,)),
        "behavior_source": torch.tensor(
            [
                int(BehaviorSource.BC_WARMUP),
                int(BehaviorSource.ACCEPTED_COLLECTION_POLICY),
                int(BehaviorSource.BC_WARMUP),
                int(BehaviorSource.ACCEPTED_COLLECTION_POLICY),
            ],
            dtype=torch.long,
        ),
    }


def _nested_equal(left, right, torch):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key], torch) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_equal(a, b, torch) for a, b in zip(left, right)
        )
    if torch.is_tensor(left):
        return torch.equal(left, right)
    return left == right


def test_exact_resume_preserves_optimizer_lrs_moments_and_step():
    torch = pytest.importorskip("torch")
    torch.manual_seed(1202)
    original = _learner(torch)
    original.update(_batch(torch), update_actor=True)
    payload = original.state_dict()
    resumed = _learner(torch)
    resumed.load_state_dict(payload)
    assert resumed.optimizer_lr_inventory() == original.optimizer_lr_inventory()
    assert _nested_equal(
        resumed.actor_optimizer.state_dict(),
        original.actor_optimizer.state_dict(),
        torch,
    )
    assert _nested_equal(
        resumed.critic_optimizer.state_dict(),
        original.critic_optimizer.state_dict(),
        torch,
    )
    assert resumed.update_step == original.update_step


def test_named_point_three_critic_lr_fork_changes_only_six_q_groups():
    torch = pytest.importorskip("torch")
    source = _learner(torch)
    source.update(_batch(torch), update_actor=False)
    payload = source.state_dict()
    candidate = _learner(torch, critic_scale=0.3)
    candidate.load_state_dict(payload)
    critic_state_before = copy.deepcopy(candidate.critic_optimizer.state_dict())
    actor_lrs_before = {
        name: value
        for name, value in candidate.optimizer_lr_inventory().items()
        if name.startswith("actor_")
    }
    overrides = {
        "critic_critic1_head": 3.0e-5,
        "critic_critic1_vector_encoder": 3.0e-6,
        "critic_critic1_depth_encoder": 3.0e-6,
        "critic_critic2_head": 3.0e-5,
        "critic_critic2_vector_encoder": 3.0e-6,
        "critic_critic2_depth_encoder": 3.0e-6,
    }
    candidate.apply_optimizer_lr_overrides(
        overrides, parent_checkpoint_sha256="a" * 64
    )
    candidate.assert_optimizer_lrs_match_config()
    actual = candidate.optimizer_lr_inventory()
    assert {name: actual[name] for name in overrides} == overrides
    assert {
        name: value
        for name, value in actual.items()
        if name.startswith("actor_")
    } == actor_lrs_before
    after = candidate.critic_optimizer.state_dict()
    assert _nested_equal(
        {key: value for key, value in after.items() if key != "param_groups"},
        {key: value for key, value in critic_state_before.items() if key != "param_groups"},
        torch,
    )
    assert candidate.optimizer_lr_provenance["parent_checkpoint_sha256"] == "a" * 64


def test_lr_override_rejects_unknown_and_illegal_names():
    torch = pytest.importorskip("torch")
    learner = _learner(torch)
    with pytest.raises(ValueError, match="unknown"):
        learner.apply_optimizer_lr_overrides(
            {"critic_missing": 1.0e-5}, parent_checkpoint_sha256="b" * 64
        )
    with pytest.raises(ValueError, match="may not change"):
        learner.apply_optimizer_lr_overrides(
            {"actor_head": 2.0e-5}, parent_checkpoint_sha256="b" * 64
        )


@pytest.mark.parametrize("hard_recovery", [False, True])
def test_constant_beta_matches_standard_actor_path(hard_recovery):
    torch = pytest.importorskip("torch")
    torch.manual_seed(1203)
    standard = _learner(torch, adaptive=False, budget=1.0e-12 if hard_recovery else 0.10)
    adaptive = _learner(torch, adaptive=True, budget=1.0e-12 if hard_recovery else 0.10)
    with torch.no_grad():
        for parameter in standard.actor.parameters():
            parameter.add_(0.02)
        for target, source in zip(adaptive.actor.parameters(), standard.actor.parameters()):
            target.copy_(source)
    batch = _batch(torch)
    left = standard.update(batch, update_actor=True)
    right = adaptive.update(batch, update_actor=True)
    assert right["bc_kl_budget_exceeded"] == left["bc_kl_budget_exceeded"]
    assert right["actor_trust_region_rejection"] == left["actor_trust_region_rejection"]
    assert right["actor_optimization_loss"] == pytest.approx(
        left["actor_optimization_loss"], abs=1.0e-6
    )
    assert right["actor_update_count"] == left["actor_update_count"]
    for left_parameter, right_parameter in zip(
        standard.actor.parameters(), adaptive.actor.parameters()
    ):
        torch.testing.assert_close(left_parameter, right_parameter, rtol=1.0e-5, atol=1.0e-6)


def test_nonconstant_beta_does_not_change_critic_when_actor_disabled():
    torch = pytest.importorskip("torch")
    torch.manual_seed(1204)
    standard = _learner(torch, adaptive=False)
    adaptive = _learner(torch, adaptive=True)
    batch = _batch(torch)
    left = standard.update(batch, update_actor=False)
    right = adaptive.update(batch, update_actor=False)
    for key in ("critic_loss", "critic_td_loss", "critic_cql_loss", "target_q_mean"):
        assert right[key] == pytest.approx(left[key], abs=1.0e-6)
    for left_parameter, right_parameter in zip(
        standard.critic1.parameters(), adaptive.critic1.parameters()
    ):
        torch.testing.assert_close(left_parameter, right_parameter, rtol=1.0e-5, atol=1.0e-6)


def test_online_metric_summary_persists_confidence_v2_fields():
    from planning.awac.online_runtime import _finite_metrics

    summary = _finite_metrics(
        [
            {
                "confidence_delta_q_mean": 0.25,
                "confidence_delta_q_norm_mean": 0.50,
                "confidence_uncertainty_mean": 0.10,
                "confidence_uncertainty_norm_mean": 0.20,
            }
        ]
    )
    assert summary["confidence_delta_q_mean_mean"] == pytest.approx(0.25)
    assert summary["confidence_delta_q_norm_mean_mean"] == pytest.approx(0.50)
    assert summary["confidence_uncertainty_mean_mean"] == pytest.approx(0.10)
    assert summary["confidence_uncertainty_norm_mean_mean"] == pytest.approx(0.20)


@pytest.mark.parametrize(
    "perturbation,budget,expected_recovery,expected_rejection",
    [
        (0.0, 0.10, False, False),
        (0.0001, 1.0e-8, True, False),
        (0.02, 0.10, True, True),
    ],
)
def test_constant_beta_parity_covers_normal_recovery_accept_and_reject(
    perturbation, budget, expected_recovery, expected_rejection
):
    """Constant beta preserves every actor proposal outcome, not just loss."""

    torch = pytest.importorskip("torch")
    torch.manual_seed(1300)
    standard = _learner(torch, adaptive=False, budget=budget)
    adaptive = _learner(torch, adaptive=True, budget=budget)
    with torch.no_grad():
        for parameter in standard.actor.parameters():
            parameter.add_(float(perturbation))
        for target, source in zip(adaptive.actor.parameters(), standard.actor.parameters()):
            target.copy_(source)
    batch = _batch(torch)
    left = standard.update(batch, update_actor=True)
    right = adaptive.update(batch, update_actor=True)
    assert bool(left["bc_kl_budget_exceeded"]) is expected_recovery
    assert bool(left["actor_trust_region_rejection"]) is expected_rejection
    for key in (
        "actor_optimization_loss",
        "actor_update_count",
        "actor_awac_optimizer_step",
        "actor_recovery_optimizer_step",
        "actor_trust_region_rejection",
    ):
        assert right[key] == pytest.approx(left[key], abs=1.0e-6)
    for left_parameter, right_parameter in zip(
        standard.actor.parameters(), adaptive.actor.parameters()
    ):
        torch.testing.assert_close(left_parameter, right_parameter, rtol=1.0e-5, atol=1.0e-6)
    for left_parameter, right_parameter in zip(
        standard.critic1.parameters(), adaptive.critic1.parameters()
    ):
        torch.testing.assert_close(left_parameter, right_parameter, rtol=1.0e-5, atol=1.0e-6)
