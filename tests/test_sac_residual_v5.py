from __future__ import annotations

from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from planning.sac.network import DiscreteSACAgent, state_dict_sha256
from planning.sac.diagnostics import stable_fingerprint


CAP = 0.49


def _agent():
    from planning.bc.model import build_model

    torch.set_num_threads(1)
    torch.manual_seed(31)
    model = build_model(nn, depth_channels=1, vec_dim=127, num_actions=105)
    checkpoint = {"model_state_dict": deepcopy(model.state_dict()), "depth_history_frames": 1}
    return DiscreteSACAgent(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_checkpoint=checkpoint,
        actor_architecture="residual",
        residual_logit_cap=CAP,
        actor_lr=1.0e-5,
        critic_lr=1.0e-4,
    )


def _batch(seed=41, size=3):
    torch.manual_seed(seed)
    mask = torch.ones((size, 105), dtype=torch.bool)
    mask[:, -3:] = False
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


def test_residual_step_zero_matches_frozen_bc_exactly():
    agent = _agent()
    batch = _batch()
    with torch.no_grad():
        residual_raw = agent.actor.residual_raw(batch["depth"], batch["vector"])
        residual_logits = agent.actor.residual_logits(batch["depth"], batch["vector"])
        actor_logits = agent.actor(batch["depth"], batch["vector"])
        bc_logits = agent.bc_reference(batch["depth"], batch["vector"])
        actor_prob, actor_log, mask = agent.action_distribution(
            batch["depth"], batch["vector"], batch["action_mask"]
        )
        bc_prob, bc_log, _ = agent.action_distribution(
            batch["depth"], batch["vector"], batch["action_mask"], use_bc_reference=True
        )
        kl = (actor_prob * (actor_log - bc_log)).sum(dim=1)
    assert torch.equal(residual_raw, torch.zeros_like(residual_raw))
    assert torch.equal(residual_logits, torch.zeros_like(residual_logits))
    assert torch.equal(actor_logits, bc_logits)
    assert float(kl.abs().max()) <= 1.0e-7
    assert torch.equal(
        torch.argmax(actor_prob.masked_fill(~mask, -1.0), dim=1),
        torch.argmax(bc_prob.masked_fill(~mask, -1.0), dim=1),
    )


def test_residual_logits_are_hard_capped_and_invalid_actions_stay_zero():
    agent = _agent()
    batch = _batch()
    final = agent.actor.residual_model.head[-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(100.0)
        base = agent.bc_reference(batch["depth"], batch["vector"])
        residual_logits = agent.actor.residual_logits(batch["depth"], batch["vector"])
        logits = agent.actor(batch["depth"], batch["vector"])
        probabilities, _, mask = agent.action_distribution(
            batch["depth"], batch["vector"], batch["action_mask"]
        )
    assert float(residual_logits.abs().max()) <= CAP + 1.0e-6
    assert torch.allclose(logits - base, torch.full_like(logits, CAP), atol=1.0e-6)
    assert torch.equal(probabilities[~mask], torch.zeros_like(probabilities[~mask]))


def test_residual_optimizer_excludes_bc_parameters_and_update_leaves_bc_frozen():
    agent = _agent()
    batch = _batch()
    bc_before = state_dict_sha256(agent.actor.bc_base.state_dict())
    reference_before = state_dict_sha256(agent.bc_reference.state_dict())
    bc_ids = {id(parameter) for parameter in agent.actor.bc_base.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in agent.actor_optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_ids
    assert not optimizer_ids.intersection(bc_ids)
    assert all(parameter.requires_grad for parameter in agent.actor_trainable_parameters())
    assert all(not parameter.requires_grad for parameter in agent.actor.bc_base.parameters())
    agent.update_actor(batch)
    assert state_dict_sha256(agent.actor.bc_base.state_dict()) == bc_before
    assert state_dict_sha256(agent.bc_reference.state_dict()) == reference_before
    assert agent.actor_optimizer_step_count == 1


def test_residual_contract_exposes_fixed_cap_and_theoretical_kl_bound():
    agent = _agent()
    payload = agent.state_payload()
    assert payload["actor_architecture"] == "residual"
    assert payload["residual_logit_cap"] == pytest.approx(CAP)
    assert payload["theoretical_kl_bound"] == pytest.approx(2.0 * CAP)
    assert 2.0 * CAP == pytest.approx(0.98)


def test_residual_state_payload_restores_actor_and_optimizer_exactly():
    source = _agent()
    source.update_actor(_batch())
    payload = deepcopy(source.state_payload())
    restored = _agent()
    restored.load_state_payload(payload)
    assert state_dict_sha256(restored.actor.state_dict()) == state_dict_sha256(source.actor.state_dict())
    assert state_dict_sha256(restored.bc_reference.state_dict()) == state_dict_sha256(source.bc_reference.state_dict())
    assert state_dict_sha256(restored.critic1.state_dict()) == state_dict_sha256(source.critic1.state_dict())
    assert state_dict_sha256(restored.target_critic1.state_dict()) == state_dict_sha256(source.target_critic1.state_dict())
    assert stable_fingerprint(restored.actor_optimizer.state_dict()) == stable_fingerprint(
        source.actor_optimizer.state_dict()
    )
    assert restored.actor_optimizer_step_count == source.actor_optimizer_step_count
