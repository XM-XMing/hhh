"""TDD contracts for the Phase-2 AWAC innovation foundation seams."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.unit


def test_twin_q_confidence_is_masked_normalized_and_monotone():
    from planning.awac.confidence import TwinQConfidenceEstimator

    estimator = TwinQConfidenceEstimator()
    q1 = np.asarray([[1.0, 0.0, -1.0, 99.0]], dtype=np.float32)
    q2 = np.asarray([[0.8, 0.1, -1.2, -99.0]], dtype=np.float32)
    mask = np.asarray([[1, 1, 1, 0]], dtype=bool)
    bc_policy = np.asarray([[0.1, 0.8, 0.1, 0.0]], dtype=np.float32)
    rl_policy = np.asarray([[0.8, 0.1, 0.1, 0.0]], dtype=np.float32)
    result = estimator.estimate(q1, q2, mask, bc_policy=bc_policy, rl_policy=rl_policy)
    assert np.all(np.isfinite(result["confidence"]))
    assert np.all((result["confidence"] >= 0.0) & (result["confidence"] <= 1.0))
    assert np.all(result["margin"] >= 0.0)
    assert np.all(result["normalized_margin"] >= 0.0)

    larger_margin = estimator.estimate(
        np.asarray([[3.0, 0.0, -1.0, 99.0]], dtype=np.float32),
        np.asarray([[2.8, 0.1, -1.2, -99.0]], dtype=np.float32),
        mask,
        bc_policy=bc_policy,
        rl_policy=rl_policy,
    )
    assert larger_margin["confidence"][0] >= result["confidence"][0]

    larger_disagreement = estimator.estimate(
        np.asarray([[1.0, 0.0, -1.0, 99.0]], dtype=np.float32),
        np.asarray([[-1.0, 0.1, -1.2, -99.0]], dtype=np.float32),
        mask,
        bc_policy=bc_policy,
        rl_policy=rl_policy,
    )
    assert larger_disagreement["confidence"][0] <= result["confidence"][0]


def test_twin_q_confidence_positive_affine_invariance_and_single_action():
    from planning.awac.confidence import TwinQConfidenceEstimator

    estimator = TwinQConfidenceEstimator()
    q1 = np.asarray([[2.0, 0.5, -1.0]], dtype=np.float64)
    q2 = np.asarray([[1.5, 0.0, -1.5]], dtype=np.float64)
    mask = np.ones_like(q1, dtype=bool)
    bc_policy = np.asarray([[0.1, 0.8, 0.1]], dtype=np.float64)
    rl_policy = np.asarray([[0.8, 0.1, 0.1]], dtype=np.float64)
    baseline = estimator.estimate(q1, q2, mask, bc_policy=bc_policy, rl_policy=rl_policy)
    for scale, offset in ((10.0, 0.0), (0.1, 0.0), (1.0, 100.0), (1.0, -100.0)):
        transformed = estimator.estimate(
            q1 * scale + offset,
            q2 * scale + offset,
            mask,
            bc_policy=bc_policy,
            rl_policy=rl_policy,
        )
        np.testing.assert_allclose(
            transformed["confidence"], baseline["confidence"], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            transformed["normalized_margin"], baseline["normalized_margin"], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            transformed["normalized_twin_disagreement"],
            baseline["normalized_twin_disagreement"],
            rtol=1e-6,
            atol=1e-6,
        )
    single = estimator.estimate(
        np.asarray([[2.0, 100.0]], dtype=np.float32),
        np.asarray([[1.0, -100.0]], dtype=np.float32),
        np.asarray([[1, 0]], dtype=bool),
        bc_policy=np.asarray([[1.0, 0.0]], dtype=np.float32),
        rl_policy=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    assert single["margin"][0] == pytest.approx(0.0)
    assert np.isfinite(single["confidence"]).all()


def test_twin_q_confidence_torch_result_is_detached():
    torch = pytest.importorskip("torch")
    from planning.awac.confidence import TwinQConfidenceEstimator

    q1 = torch.tensor([[2.0, 1.0]], requires_grad=True)
    q2 = torch.tensor([[1.0, 0.0]], requires_grad=True)
    result = TwinQConfidenceEstimator().estimate(
        q1,
        q2,
        torch.tensor([[True, True]]),
        bc_policy=torch.tensor([[1.0, 0.0]]),
        rl_policy=torch.tensor([[0.0, 1.0]]),
    )
    assert result["confidence"].requires_grad is False
    assert result["normalized_margin"].requires_grad is False


def test_twin_q_confidence_uses_policy_actions_and_rejects_negative_delta():
    from planning.awac.confidence import (
        CONFIDENCE_CONTRACT_ID,
        CONFIDENCE_FORMULA_VERSION,
        TwinQConfidenceEstimator,
    )

    result = TwinQConfidenceEstimator().estimate(
        np.asarray([[10.0, 0.0, -1.0]], dtype=np.float64),
        np.asarray([[10.0, 0.0, -1.0]], dtype=np.float64),
        np.ones((1, 3), dtype=bool),
        bc_policy=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        rl_policy=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
    )
    assert result["a_bc"].tolist() == [0]
    assert result["a_rl"].tolist() == [2]
    assert result["delta_q"].tolist() == pytest.approx([-11.0])
    assert result["confidence"].tolist() == pytest.approx([0.0])
    assert result["confidence_contract_id"] == CONFIDENCE_CONTRACT_ID
    assert result["confidence_formula_version"] == CONFIDENCE_FORMULA_VERSION


def test_twin_q_confidence_numpy_torch_parity_and_no_policy_weighted_switch():
    torch = pytest.importorskip("torch")
    from planning.awac.confidence import TwinQConfidenceEstimator

    q1 = np.asarray([[4.0, 2.0, -1.0, 8.0], [1.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    q2 = np.asarray([[3.0, 3.0, -2.0, 99.0], [1.0, 0.5, 0.0, 0.0]], dtype=np.float64)
    mask = np.asarray([[1, 1, 1, 0], [1, 0, 1, 1]], dtype=bool)
    bc = np.asarray([[0.1, 0.8, 0.1, 0.0], [0.7, 0.0, 0.2, 0.1]], dtype=np.float64)
    rl = np.asarray([[0.8, 0.1, 0.1, 0.0], [0.1, 0.0, 0.8, 0.1]], dtype=np.float64)
    estimator = TwinQConfidenceEstimator()
    numpy_result = estimator.estimate(q1, q2, mask, bc_policy=bc, rl_policy=rl)
    torch_result = estimator.estimate(
        torch.from_numpy(q1),
        torch.from_numpy(q2),
        torch.from_numpy(mask),
        bc_policy=torch.from_numpy(bc),
        rl_policy=torch.from_numpy(rl),
    )
    for name in ("confidence", "a_bc", "a_rl", "delta_q", "uncertainty", "q_scale"):
        np.testing.assert_allclose(
            np.asarray(numpy_result[name]),
            torch_result[name].cpu().numpy(),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
    alias_result = estimator.estimate(
        q1,
        q2,
        mask,
        current_policy=rl,
        bc_policy=bc,
    )
    np.testing.assert_allclose(alias_result["confidence"], numpy_result["confidence"])


def test_adaptive_bc_kl_is_bounded_and_disabled_is_exact_base_weight():
    import torch

    from planning.awac.adaptive_bc_kl import (
        AdaptiveBCKLConfig,
        adaptive_bc_kl_weight,
    )

    confidence = torch.tensor([0.0, 0.5, 1.0])
    config = AdaptiveBCKLConfig(beta_min=0.02, beta_max=0.10, base_weight=0.05)
    weights = adaptive_bc_kl_weight(confidence, config=config, torch=torch, enabled=True)
    assert torch.all(weights[:-1] >= weights[1:])
    assert float(weights.min()) >= 0.02
    assert float(weights.max()) <= 0.10 + 1.0e-6
    disabled = adaptive_bc_kl_weight(confidence, config=config, torch=torch, enabled=False)
    assert torch.equal(disabled, torch.full_like(confidence, 0.05))


def test_primitive_neighborhood_is_geometric_and_artifact_roundtrips(tmp_path: Path):
    from planning.awac.primitive_neighborhood import (
        build_primitive_neighborhood,
        load_primitive_neighborhood,
        write_primitive_neighborhood_artifact,
    )

    trajectories = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    neighborhood = build_primitive_neighborhood(trajectories)
    assert neighborhood.distance_matrix.shape == (3, 3)
    assert np.isfinite(neighborhood.distance_matrix).all()
    np.testing.assert_array_equal(np.diag(neighborhood.distance_matrix), 0.0)
    np.testing.assert_allclose(
        neighborhood.distance_matrix,
        neighborhood.distance_matrix.T,
        rtol=0.0,
        atol=1e-7,
    )
    assert neighborhood.top_k(0, 1)[0] != 0
    np.testing.assert_array_equal(neighborhood.radius(0, 0.0), np.asarray([0]))
    npz_path = tmp_path / "neighborhood.npz"
    json_path = tmp_path / "neighborhood.json"
    write_primitive_neighborhood_artifact(
        neighborhood,
        npz_path=npz_path,
        metadata_path=json_path,
        source_path="fixture.npz",
        source_sha256="a" * 64,
        mpl_contract_sha256="b" * 64,
    )
    loaded = load_primitive_neighborhood(npz_path, metadata_path=json_path)
    np.testing.assert_allclose(loaded.distance_matrix, neighborhood.distance_matrix, atol=2e-7)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["action_count"] == 3
    assert metadata["source_sha256"] == "a" * 64
    assert metadata["artifact_sha256"]


def test_primitive_exploration_preserves_masks_and_support_modes():
    from planning.awac.primitive_exploration import (
        PrimitiveExplorationConfig,
        primitive_behavior_distribution,
    )

    policy = np.asarray([0.6, 0.2, 0.1, 0.1], dtype=np.float64)
    valid = np.asarray([1, 1, 1, 0], dtype=bool)
    matrix = np.asarray(
        [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.0, 3.0], [2.0, 1.0, 0.0, 1.0], [3.0, 3.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    local = primitive_behavior_distribution(
        policy,
        valid,
        anchor_primitive=0,
        neighborhood=matrix,
        config=PrimitiveExplorationConfig(enabled=True, neighbor_top_k=1, local_global_mix=0.0),
    )
    assert local[2] == pytest.approx(0.0)
    assert local[3] == pytest.approx(0.0)
    assert local.sum() == pytest.approx(1.0)
    global_mix = primitive_behavior_distribution(
        policy,
        valid,
        anchor_primitive=0,
        neighborhood=matrix,
        config=PrimitiveExplorationConfig(enabled=True, neighbor_top_k=1, local_global_mix=1.0),
    )
    assert global_mix[2] > 0.0
    assert global_mix[3] == pytest.approx(0.0)
    assert global_mix.sum() == pytest.approx(1.0)


def test_innovation_flags_are_default_off_and_serialized():
    from planning.awac.learner import AWACOptimizationConfig

    config = AWACOptimizationConfig()
    assert config.enable_twin_q_confidence is False
    assert config.enable_adaptive_bc_kl is False
    assert config.enable_primitive_neighbor_exploration is False
    assert config.bc_kl_weight == pytest.approx(0.05)


def test_resolved_config_records_innovation_identity():
    from planning.awac import trainer

    args = trainer.build_parser().parse_args(
        ["--bc-checkpoint", "bc.pt", "--out-dir", "out"]
    )
    resolved = trainer._resolved_training_config(
        args, source_bc_sha256="a" * 64, mpl_sha256="b" * 64
    )
    innovation = resolved["innovation_config"]
    assert innovation["enable_twin_q_confidence"] is False
    assert innovation["enable_adaptive_bc_kl"] is False
    assert innovation["enable_primitive_neighbor_exploration"] is False
    assert innovation["default_enabled"] is False


@pytest.mark.parametrize("updates_per_step", [0.50, 0.25])
def test_disabled_innovation_path_preserves_synthetic_awac_update(
    updates_per_step: float,
):
    """The two frozen development schedules share the disabled math path."""

    del updates_per_step  # schedule ownership is outside the learner update.
    torch = pytest.importorskip("torch")
    from planning.awac.interaction import BehaviorSource
    from planning.awac.learner import AWACOptimizationConfig, DiscreteAWACLearner
    from planning.bc.model import build_model
    from planning.contracts.feature import NUM_ACTIONS, POLICY_VECTOR_DIM

    def make_learner():
        torch.manual_seed(711)
        bc = build_model(torch.nn, depth_channels=1)
        return DiscreteAWACLearner(
            torch=torch,
            nn=torch.nn,
            device=torch.device("cpu"),
            bc_state_dict=bc.state_dict(),
            depth_channels=1,
            config=AWACOptimizationConfig(
                actor_vector_lr=0.0,
                actor_depth_lr=0.0,
                critic_vector_lr=0.0,
                critic_depth_lr=0.0,
            ),
        )

    torch.manual_seed(712)
    batch_size = 2
    mask = torch.zeros((batch_size, NUM_ACTIONS), dtype=torch.bool)
    mask[:, :4] = True
    batch = {
        "depth": torch.rand((batch_size, 1, 32, 32)),
        "vector": torch.randn((batch_size, POLICY_VECTOR_DIM)),
        "action_mask": mask,
        "action": torch.tensor([0, 1], dtype=torch.long),
        "reward": torch.tensor([0.5, -0.25]),
        "next_depth": torch.rand((batch_size, 1, 32, 32)),
        "next_vector": torch.randn((batch_size, POLICY_VECTOR_DIM)),
        "next_action_mask": mask.clone(),
        "done": torch.zeros((batch_size,)),
        "behavior_source": torch.tensor(
            [int(BehaviorSource.BC_WARMUP), int(BehaviorSource.ACCEPTED_COLLECTION_POLICY)],
            dtype=torch.long,
        ),
    }
    first = make_learner()
    second = make_learner()
    metrics_first = first.update(batch, update_actor=False)
    metrics_second = second.update(batch, update_actor=False)
    for key in (
        "critic_loss",
        "critic_td_loss",
        "critic_cql_loss",
        "awac_advantage_mean",
        "awac_weight_mean",
        "bc_kl",
        "target_q_mean",
        "actor_loss",
    ):
        assert metrics_first[key] == pytest.approx(metrics_second[key], abs=1.0e-7)


def test_enabled_confidence_and_adaptive_kl_path_is_finite_and_observable():
    torch = pytest.importorskip("torch")
    from planning.awac.interaction import BehaviorSource
    from planning.awac.learner import AWACOptimizationConfig, DiscreteAWACLearner
    from planning.bc.model import build_model
    from planning.contracts.feature import NUM_ACTIONS, POLICY_VECTOR_DIM

    torch.manual_seed(913)
    bc = build_model(torch.nn, depth_channels=1)
    learner = DiscreteAWACLearner(
        torch=torch,
        nn=torch.nn,
        device=torch.device("cpu"),
        bc_state_dict=bc.state_dict(),
        depth_channels=1,
        config=AWACOptimizationConfig(
            enable_twin_q_confidence=True,
            enable_adaptive_bc_kl=True,
            actor_vector_lr=0.0,
            actor_depth_lr=0.0,
            critic_vector_lr=0.0,
            critic_depth_lr=0.0,
        ),
    )
    mask = torch.zeros((2, NUM_ACTIONS), dtype=torch.bool)
    mask[:, :4] = True
    batch = {
        "depth": torch.rand((2, 1, 32, 32)),
        "vector": torch.randn((2, POLICY_VECTOR_DIM)),
        "action_mask": mask,
        "action": torch.tensor([0, 1]),
        "reward": torch.tensor([0.5, -0.25]),
        "next_depth": torch.rand((2, 1, 32, 32)),
        "next_vector": torch.randn((2, POLICY_VECTOR_DIM)),
        "next_action_mask": mask.clone(),
        "done": torch.zeros((2,)),
        "behavior_source": torch.tensor(
            [
                int(BehaviorSource.BC_WARMUP),
                int(BehaviorSource.ACCEPTED_COLLECTION_POLICY),
            ]
        ),
    }
    metrics = learner.update(batch, update_actor=False)
    for key in (
        "confidence_mean",
        "confidence_p05",
        "confidence_p95",
        "adaptive_bc_kl_beta_mean",
        "adaptive_bc_kl_beta_p50",
        "adaptive_bc_kl_low_confidence_beta_mean",
        "adaptive_bc_kl_high_confidence_beta_mean",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])
    assert metrics["adaptive_bc_kl_enabled"] == pytest.approx(1.0)


def test_exp1_confidence_diagnostic_only_preserves_awac_update_values():
    torch = pytest.importorskip("torch")
    from planning.awac.interaction import BehaviorSource
    from planning.awac.learner import AWACOptimizationConfig, DiscreteAWACLearner
    from planning.bc.model import build_model
    from planning.contracts.feature import NUM_ACTIONS, POLICY_VECTOR_DIM

    def make_learner(*, confidence_enabled):
        torch.manual_seed(914)
        bc = build_model(torch.nn, depth_channels=1)
        return DiscreteAWACLearner(
            torch=torch,
            nn=torch.nn,
            device=torch.device("cpu"),
            bc_state_dict=bc.state_dict(),
            depth_channels=1,
            config=AWACOptimizationConfig(
                enable_twin_q_confidence=bool(confidence_enabled),
                enable_adaptive_bc_kl=False,
                actor_vector_lr=0.0,
                actor_depth_lr=0.0,
                critic_vector_lr=0.0,
                critic_depth_lr=0.0,
            ),
        )

    torch.manual_seed(915)
    mask = torch.zeros((2, NUM_ACTIONS), dtype=torch.bool)
    mask[:, :4] = True
    batch = {
        "depth": torch.rand((2, 1, 32, 32)),
        "vector": torch.randn((2, POLICY_VECTOR_DIM)),
        "action_mask": mask,
        "action": torch.tensor([0, 1]),
        "reward": torch.tensor([0.5, -0.25]),
        "next_depth": torch.rand((2, 1, 32, 32)),
        "next_vector": torch.randn((2, POLICY_VECTOR_DIM)),
        "next_action_mask": mask.clone(),
        "done": torch.zeros((2,)),
        "behavior_source": torch.tensor(
            [
                int(BehaviorSource.BC_WARMUP),
                int(BehaviorSource.ACCEPTED_COLLECTION_POLICY),
            ]
        ),
    }
    baseline = make_learner(confidence_enabled=False).update(batch, update_actor=False)
    diagnostic = make_learner(confidence_enabled=True).update(batch, update_actor=False)
    for key in (
        "critic_loss",
        "critic_td_loss",
        "critic_cql_loss",
        "awac_advantage_mean",
        "awac_weight_mean",
        "bc_kl",
        "target_q_mean",
        "actor_loss",
    ):
        assert diagnostic[key] == pytest.approx(baseline[key], abs=1.0e-7)
    assert diagnostic["confidence_mean"] >= 0.0
    assert diagnostic["adaptive_bc_kl_enabled"] == pytest.approx(0.0)
