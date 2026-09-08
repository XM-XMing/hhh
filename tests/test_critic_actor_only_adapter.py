"""Focused contracts for the diagnostic Critic-to-production adapter."""

import pytest

from planning.awac.critic_actor_only import adapt_h0_diagnostic_state_dict


@pytest.mark.unit
def test_h0_diagnostic_state_is_unwrapped_and_budget_column_is_removed():
    torch = pytest.importorskip("torch")
    weight = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    weight[:, -1] = 0.0
    source = {
        "base.vector_encoder.0.weight": weight,
        "base.vector_encoder.0.bias": torch.ones(6),
    }

    adapted, report = adapt_h0_diagnostic_state_dict(
        source, torch=torch, policy_vector_dim=3
    )

    assert tuple(adapted["vector_encoder.0.weight"].shape) == (6, 3)
    assert torch.equal(adapted["vector_encoder.0.weight"], weight[:, :3])
    assert torch.equal(adapted["vector_encoder.0.bias"], source["base.vector_encoder.0.bias"])
    assert report["source_prefix"] == "base."
    assert report["source_vector_dim"] == 4
    assert report["target_vector_dim"] == 3
    assert report["budget_column_max_abs"] == 0.0


@pytest.mark.unit
def test_h0_diagnostic_state_rejects_nonzero_budget_column():
    torch = pytest.importorskip("torch")
    weight = torch.zeros((2, 4), dtype=torch.float32)
    weight[0, -1] = 1.0e-4
    with pytest.raises(ValueError, match="budget column"):
        adapt_h0_diagnostic_state_dict(
            {"base.vector_encoder.0.weight": weight},
            torch=torch,
            policy_vector_dim=3,
        )


@pytest.mark.unit
def test_h0_diagnostic_state_rejects_unexpected_dimension():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="vector dimension"):
        adapt_h0_diagnostic_state_dict(
            {"base.vector_encoder.0.weight": torch.zeros((2, 3))},
            torch=torch,
            policy_vector_dim=3,
        )
