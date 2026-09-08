"""BC-compatible Actor and twin-Q helpers for discrete AWAC."""

from __future__ import annotations

import math
from typing import Mapping, Tuple

from planning.bc.model import build_model
from planning.contracts.feature import NUM_ACTIONS, POLICY_VECTOR_DIM


def build_actor(nn, *, depth_channels: int, vec_dim: int = POLICY_VECTOR_DIM, num_actions: int = NUM_ACTIONS):
    """Build the exact BC policy architecture used by fixed-holdout evaluation."""
    return build_model(
        nn,
        vec_dim=int(vec_dim),
        num_actions=int(num_actions),
        depth_channels=int(depth_channels),
    )


def build_critic(nn, *, depth_channels: int, vec_dim: int = POLICY_VECTOR_DIM, num_actions: int = NUM_ACTIONS):
    """Build a Q network with the BC depth/vector encoder architecture."""
    return build_model(
        nn,
        vec_dim=int(vec_dim),
        num_actions=int(num_actions),
        depth_channels=int(depth_channels),
    )


def load_actor_state_dict_strict(actor, state_dict: Mapping) -> None:
    """Load deployable Actor tensors without permitting partial handoff."""

    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("BC Actor state_dict is missing or empty")
    actor.load_state_dict(state_dict, strict=True)


def copy_encoder_state(source, destination) -> None:
    """Copy only deployable BC encoders; critic Q heads stay independently initialized."""
    destination.depth_encoder.load_state_dict(source.depth_encoder.state_dict())
    destination.vector_encoder.load_state_dict(source.vector_encoder.state_dict())


def reset_head(module) -> None:
    """Reset all learnable layers in a policy/Q head."""
    for child in module.head.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def set_encoder_trainability(model, *, depth_trainable: bool, vector_trainable: bool) -> None:
    for parameter in model.depth_encoder.parameters():
        parameter.requires_grad_(bool(depth_trainable))
    for parameter in model.vector_encoder.parameters():
        parameter.requires_grad_(bool(vector_trainable))
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)


def optimizer_parameter_groups(
    model,
    *,
    head_lr: float,
    vector_lr: float,
    depth_lr: float,
) -> list:
    """Create explicit optimizer groups and freeze zero-learning-rate encoders."""
    if float(head_lr) <= 0.0:
        raise ValueError("head_lr must be positive")
    set_encoder_trainability(
        model,
        depth_trainable=float(depth_lr) > 0.0,
        vector_trainable=float(vector_lr) > 0.0,
    )
    groups = [{"params": list(model.head.parameters()), "lr": float(head_lr), "name": "head"}]
    if float(vector_lr) > 0.0:
        groups.append({
            "params": list(model.vector_encoder.parameters()),
            "lr": float(vector_lr),
            "name": "vector_encoder",
        })
    if float(depth_lr) > 0.0:
        groups.append({
            "params": list(model.depth_encoder.parameters()),
            "lr": float(depth_lr),
            "name": "depth_encoder",
        })
    return groups


def _sanitized_masks(action_masks, torch):
    masks = action_masks.bool()
    if masks.ndim != 2:
        raise ValueError("action_masks must be [B,A]")
    no_valid = ~masks.any(dim=1)
    if bool(no_valid.any()):
        masks = masks.clone()
        masks[no_valid, 0] = True
    return masks, no_valid


def masked_policy(logits, action_masks, torch, *, temperature: float = 1.0) -> Tuple:
    """Return finite masked probabilities/log-probabilities and empty-mask flags."""
    value = float(temperature)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("policy temperature must be finite and positive")
    masks, no_valid = _sanitized_masks(action_masks, torch)
    scaled_logits = logits if value == 1.0 else logits / value
    masked_logits = scaled_logits.masked_fill(~masks, -1.0e4)
    log_probabilities = torch.log_softmax(masked_logits, dim=1)
    probabilities = torch.exp(log_probabilities) * masks.to(dtype=logits.dtype)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    log_probabilities = torch.log(probabilities.clamp_min(1.0e-12))
    return probabilities, log_probabilities, no_valid


def sample_masked_action(
    logits,
    action_masks,
    torch,
    *,
    deterministic: bool = False,
    temperature: float = 1.0,
):
    probabilities, log_probabilities, no_valid = masked_policy(
        logits, action_masks, torch, temperature=temperature
    )
    if deterministic:
        action = torch.argmax(probabilities, dim=1)
    else:
        action = torch.multinomial(probabilities, num_samples=1).squeeze(1)
    selected_log_probability = log_probabilities.gather(1, action[:, None]).squeeze(1)
    return action, selected_log_probability, probabilities, no_valid


def soft_update(target, source, tau: float) -> None:
    value = float(tau)
    if not 0.0 < value <= 1.0:
        raise ValueError("tau must be in (0,1]")
    with __import__("torch").no_grad():
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            target_parameter.mul_(1.0 - value).add_(source_parameter, alpha=value)
