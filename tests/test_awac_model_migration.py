"""Public model seams for the AWAC-only architecture migration."""

from __future__ import annotations

import pytest

from planning.bc.model import require_torch


@pytest.mark.unit
def test_awac_model_owner_is_independent_of_historical_rl_package():
    from planning.awac.model import build_actor, build_critic

    torch, nn, _, _, _ = require_torch()
    actor = build_actor(nn, depth_channels=1)
    critic = build_critic(nn, depth_channels=1)
    assert actor.depth_channels == 1
    assert critic.depth_channels == 1
    assert set(actor.state_dict()) == set(critic.state_dict())


@pytest.mark.unit
def test_awac_depth_encoders_can_be_trainable_when_configured():
    from planning.awac.model import optimizer_parameter_groups

    torch, nn, _, _, _ = require_torch()
    model = __import__("planning.bc.model", fromlist=["build_model"]).build_model(
        nn, depth_channels=1
    )
    groups = optimizer_parameter_groups(
        model, head_lr=1.0e-4, vector_lr=1.0e-5, depth_lr=1.0e-5
    )
    assert any(group["name"] == "depth_encoder" for group in groups)
    assert any(parameter.requires_grad for parameter in model.depth_encoder.parameters())

