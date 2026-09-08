"""Masked discrete AWAC learner initialized from the deployable BC policy.

This module deliberately contains no environment-step or update-frequency
schedule.  The caller decides when Critic and Actor updates are allowed.  The
learner owns one Bellman/CQL Critic update followed, when requested, by an
advantage-weighted data-action Actor proposal protected by the BC KL contract.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from planning.awac.interaction import BehaviorSource
from planning.awac.adaptive_bc_kl import AdaptiveBCKLConfig, adaptive_bc_kl_loss
from planning.awac.calibration import masked_bellman_target
from planning.awac.confidence import (
    CONFIDENCE_CONTRACT_ID,
    CONFIDENCE_FORMULA_VERSION,
    CONFIDENCE_SCALE_VERSION,
    ConfidenceConfig,
    TwinQConfidenceEstimator,
)
from planning.awac.optimization import (
    kl_budget_exceeded,
    kl_proposal_is_acceptable,
    masked_discrete_cql_loss,
)
from planning.awac.model import (
    build_actor,
    build_critic,
    copy_encoder_state,
    load_actor_state_dict_strict,
    masked_policy,
    optimizer_parameter_groups,
    reset_head,
    sample_masked_action,
    soft_update,
)


AWAC_LEARNER_STATE_SCHEMA_ID = "masked_discrete_awac_learner_state"


def resolve_torch_device(torch, device):
    """Resolve an AWAC compute device to one concrete PyTorch device.

    ``torch.device("cuda")`` denotes the process current CUDA device, whereas
    model parameters report that device with an explicit index.  Resolve the
    unspecified form once at learner construction so every model and replay
    batch uses the same concrete device identity.
    """

    resolved = torch.device(device)
    if resolved.type != "cuda" or resolved.index is not None:
        return resolved
    return torch.device("cuda:{}".format(int(torch.cuda.current_device())))


def _finite_number(value, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


@dataclass(frozen=True)
class AWACOptimizationConfig:
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 0.10
    actor_head_lr: float = 1.0e-5
    actor_vector_lr: float = 3.0e-6
    actor_depth_lr: float = 0.0
    critic_head_lr: float = 1.0e-4
    critic_vector_lr: float = 1.0e-5
    critic_depth_lr: float = 1.0e-5
    critic_cql_weight: float = 0.05
    awac_temperature: float = 2.0
    awac_weight_max: float = 20.0
    bc_kl_weight: float = 0.05
    trust_tail_top_k: int = 16
    bc_kl_hard_budget: float = 0.10
    bc_kl_recovery_weight: float = 1.0
    gradient_clip_norm: float = 5.0
    # Phase-2 innovation flags are deliberately default-off.  The fields are
    # part of future checkpoint/config identity, but the disabled learner path
    # below retains the established Standard AWAC arithmetic exactly.
    enable_twin_q_confidence: bool = False
    enable_adaptive_bc_kl: bool = False
    enable_primitive_neighbor_exploration: bool = False
    confidence_margin_gain: float = 2.0
    confidence_disagreement_gain: float = 1.0
    adaptive_bc_kl_beta_min: float = 0.02
    adaptive_bc_kl_beta_max: float = 0.10
    primitive_neighborhood_artifact_path: str = ""
    primitive_neighborhood_artifact_sha256: str = ""
    primitive_neighbor_top_k: int = 8
    primitive_neighbor_radius: float = -1.0
    primitive_local_global_mix: float = 0.20

    def __post_init__(self) -> None:
        values = {
            name: _finite_number(getattr(self, name), name=name)
            for name in (
                "gamma",
                "tau",
                "reward_scale",
                "actor_head_lr",
                "actor_vector_lr",
                "actor_depth_lr",
                "critic_head_lr",
                "critic_vector_lr",
                "critic_depth_lr",
                "critic_cql_weight",
                "awac_temperature",
                "awac_weight_max",
                "bc_kl_weight",
                "bc_kl_hard_budget",
                "bc_kl_recovery_weight",
                "gradient_clip_norm",
                "confidence_margin_gain",
                "confidence_disagreement_gain",
                "adaptive_bc_kl_beta_min",
                "adaptive_bc_kl_beta_max",
                "primitive_local_global_mix",
            )
        }
        if not 0.0 <= values["gamma"] <= 1.0:
            raise ValueError("gamma must be in [0,1]")
        if not 0.0 < values["tau"] <= 1.0:
            raise ValueError("tau must be in (0,1]")
        if values["reward_scale"] <= 0.0:
            raise ValueError("reward_scale must be positive")
        if values["actor_head_lr"] <= 0.0 or values["critic_head_lr"] <= 0.0:
            raise ValueError("Actor/Critic head learning rates must be positive")
        for name in (
            "actor_vector_lr",
            "actor_depth_lr",
            "critic_vector_lr",
            "critic_depth_lr",
            "critic_cql_weight",
            "bc_kl_weight",
        ):
            if values[name] < 0.0:
                raise ValueError("{} must be non-negative".format(name))
        if values["awac_temperature"] <= 0.0:
            raise ValueError("awac_temperature must be positive")
        if values["awac_weight_max"] < 1.0:
            raise ValueError("awac_weight_max must be at least one")
        if int(self.trust_tail_top_k) <= 0:
            raise ValueError("trust_tail_top_k must be positive")
        if values["bc_kl_hard_budget"] <= 0.0:
            raise ValueError("bc_kl_hard_budget must be positive")
        if values["bc_kl_recovery_weight"] <= 0.0:
            raise ValueError("bc_kl_recovery_weight must be positive")
        if values["gradient_clip_norm"] <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        ConfidenceConfig(
            margin_gain=values["confidence_margin_gain"],
            disagreement_gain=values["confidence_disagreement_gain"],
        )
        AdaptiveBCKLConfig(
            beta_min=values["adaptive_bc_kl_beta_min"],
            beta_max=values["adaptive_bc_kl_beta_max"],
            base_weight=values["bc_kl_weight"],
        )
        if int(self.primitive_neighbor_top_k) <= 0:
            raise ValueError("primitive_neighbor_top_k must be positive")
        primitive_radius = float(self.primitive_neighbor_radius)
        if primitive_radius < 0.0 and primitive_radius != -1.0:
            raise ValueError("primitive_neighbor_radius must be -1 or non-negative")
        if not 0.0 <= values["primitive_local_global_mix"] <= 1.0:
            raise ValueError("primitive_local_global_mix must be in [0,1]")
        if not isinstance(self.enable_twin_q_confidence, bool):
            raise ValueError("enable_twin_q_confidence must be bool")
        if not isinstance(self.enable_adaptive_bc_kl, bool):
            raise ValueError("enable_adaptive_bc_kl must be bool")
        if not isinstance(self.enable_primitive_neighbor_exploration, bool):
            raise ValueError("enable_primitive_neighbor_exploration must be bool")


def awac_advantage_weights(
    advantage,
    *,
    temperature: float,
    weight_max: float,
    torch,
):
    """Return stable detached raw and mean-normalized AWAC weights.

    Both outputs are capped.  The normalized output is used by the Actor loss;
    the raw output is retained because its scale and effective sample size are
    important evidence when choosing temperature/cap outside this learner.
    """

    temperature = _finite_number(temperature, name="temperature")
    weight_max = _finite_number(weight_max, name="weight_max")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if weight_max < 1.0:
        raise ValueError("weight_max must be at least one")
    if advantage.ndim != 1 or int(advantage.numel()) <= 0:
        raise ValueError("advantage must be a non-empty [B] tensor")
    if not bool(torch.isfinite(advantage).all()):
        raise ValueError("advantage contains non-finite values")

    detached = advantage.detach()
    log_cap = math.log(weight_max)
    # finfo.tiny is representable and strictly positive, retaining non-zero
    # supervision even for strongly negative-advantage failure actions.
    log_floor = math.log(float(torch.finfo(detached.dtype).tiny))
    log_raw_unclipped = detached / temperature
    log_raw = log_raw_unclipped.clamp(min=log_floor, max=log_cap)
    raw = torch.exp(log_raw)

    log_raw_mean = torch.logsumexp(log_raw, dim=0) - math.log(float(raw.numel()))
    normalized_log_unclipped = log_raw - log_raw_mean
    normalized_log = normalized_log_unclipped.clamp(min=log_floor, max=log_cap)
    normalized = torch.exp(normalized_log)
    if not bool(torch.isfinite(raw).all() and torch.isfinite(normalized).all()):
        raise FloatingPointError("AWAC weights became non-finite")
    return {
        "raw": raw.detach(),
        "normalized": normalized.detach(),
        "log_raw_unclipped": log_raw_unclipped.detach(),
        "normalized_log_unclipped": normalized_log_unclipped.detach(),
        "raw_high_clip": (log_raw_unclipped > log_cap).detach(),
        "raw_low_floor": (log_raw_unclipped < log_floor).detach(),
        "normalized_high_clip": (normalized_log_unclipped > log_cap).detach(),
    }


def _effective_sample_size(weights, torch):
    numerator = weights.sum().square()
    denominator = weights.square().sum().clamp_min(torch.finfo(weights.dtype).tiny)
    return numerator / denominator


class DiscreteAWACLearner:
    """Twin-Q discrete AWAC with a BC-compatible deployable Actor."""

    def __init__(
        self,
        *,
        torch,
        nn,
        device,
        bc_state_dict: Dict,
        depth_channels: int,
        config: AWACOptimizationConfig,
    ):
        if not isinstance(config, AWACOptimizationConfig):
            raise TypeError("config must be AWACOptimizationConfig")
        self.torch = torch
        self.nn = nn
        self.device = resolve_torch_device(torch, device)
        self.config = config
        self.depth_channels = int(depth_channels)
        if self.depth_channels <= 0:
            raise ValueError("depth_channels must be positive")

        self.actor = build_actor(nn, depth_channels=self.depth_channels).to(self.device)
        load_actor_state_dict_strict(self.actor, bc_state_dict)
        self.bc_reference = build_actor(nn, depth_channels=self.depth_channels).to(
            self.device
        )
        load_actor_state_dict_strict(self.bc_reference, bc_state_dict)
        self.bc_reference.eval()
        for parameter in self.bc_reference.parameters():
            parameter.requires_grad_(False)

        self.critic1 = build_critic(nn, depth_channels=self.depth_channels).to(
            self.device
        )
        self.critic2 = build_critic(nn, depth_channels=self.depth_channels).to(
            self.device
        )
        copy_encoder_state(self.actor, self.critic1)
        copy_encoder_state(self.actor, self.critic2)
        reset_head(self.critic1)
        reset_head(self.critic2)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device).eval()
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device).eval()
        for target in (self.target_critic1, self.target_critic2):
            for parameter in target.parameters():
                parameter.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            optimizer_parameter_groups(
                self.actor,
                head_lr=config.actor_head_lr,
                vector_lr=config.actor_vector_lr,
                depth_lr=config.actor_depth_lr,
            )
        )
        critic_groups = []
        for prefix, critic in (("critic1", self.critic1), ("critic2", self.critic2)):
            for group in optimizer_parameter_groups(
                critic,
                head_lr=config.critic_head_lr,
                vector_lr=config.critic_vector_lr,
                depth_lr=config.critic_depth_lr,
            ):
                copied = dict(group)
                copied["name"] = "{}_{}".format(prefix, group["name"])
                critic_groups.append(copied)
        self.critic_optimizer = torch.optim.Adam(critic_groups)
        self.optimizer_lr_provenance = {
            "mode": "constructed_from_resolved_config",
            "parent_checkpoint_sha256": "",
            "overrides": {},
            "actual_group_lrs": self.optimizer_lr_inventory(),
        }

        self.confidence_estimator = TwinQConfidenceEstimator(
            ConfidenceConfig(
                margin_gain=float(config.confidence_margin_gain),
                disagreement_gain=float(config.confidence_disagreement_gain),
            )
        )
        self.adaptive_bc_kl_config = AdaptiveBCKLConfig(
            beta_min=float(config.adaptive_bc_kl_beta_min),
            beta_max=float(config.adaptive_bc_kl_beta_max),
            base_weight=float(config.bc_kl_weight),
        )
        # The neighborhood is injected by the future behavior-policy owner;
        # no artifact is loaded when the production flag is disabled.
        self.primitive_neighborhood = None

        self.update_step = 0
        self.actor_update_count = 0
        self.actor_awac_update_count = 0
        self.actor_recovery_update_count = 0
        self.actor_trust_region_rejection_count = 0
        self.actor_optimizer_step_count = 0
        self.critic_update_count = 0
        self.actor_frozen_for_calibration = False

    def innovation_config(self) -> Dict[str, Any]:
        """Return serializable Phase-2 identity without optimizer state."""

        config = self.config
        return {
            "enable_twin_q_confidence": bool(config.enable_twin_q_confidence),
            "enable_adaptive_bc_kl": bool(config.enable_adaptive_bc_kl),
            "enable_primitive_neighbor_exploration": bool(
                config.enable_primitive_neighbor_exploration
            ),
            "confidence": {
                "contract_id": CONFIDENCE_CONTRACT_ID,
                "formula_version": CONFIDENCE_FORMULA_VERSION,
                "scale_version": CONFIDENCE_SCALE_VERSION,
                "margin_gain": float(config.confidence_margin_gain),
                "disagreement_gain": float(config.confidence_disagreement_gain),
                "aggregation": "valid_action_mean",
                "action_source": "masked_bc_argmax_vs_masked_awac_argmax",
                "delta_q": "qmin(a_rl)-qmin(a_bc)",
                "uncertainty": "abs(q1(a_rl)-q2(a_rl))",
            },
            "adaptive_bc_kl": {
                "contract_id": "awac_adaptive_bc_kl_v1",
                "beta_min": float(config.adaptive_bc_kl_beta_min),
                "beta_max": float(config.adaptive_bc_kl_beta_max),
                "base_weight": float(config.bc_kl_weight),
                "hard_budget": float(config.bc_kl_hard_budget),
                "recovery_weight": float(config.bc_kl_recovery_weight),
                "order": "state_beta_then_hard_budget_recovery",
            },
            "primitive_exploration": {
                "enabled": bool(config.enable_primitive_neighbor_exploration),
                "artifact_path": str(config.primitive_neighborhood_artifact_path),
                "artifact_sha256": str(config.primitive_neighborhood_artifact_sha256),
                "neighbor_top_k": int(config.primitive_neighbor_top_k),
                "neighbor_radius": (
                    None
                    if float(config.primitive_neighbor_radius) < 0.0
                    else float(config.primitive_neighbor_radius)
                ),
                "local_global_mix": float(config.primitive_local_global_mix),
            },
            "default_enabled": False,
        }

    @property
    def policy_update_count(self) -> int:
        """Accepted task-policy updates (KL recovery steps are excluded)."""

        return int(self.actor_awac_update_count)

    def act(
        self,
        depth,
        vector,
        action_mask,
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        anchor_primitive: Optional[int] = None,
    ):
        was_training = bool(self.actor.training)
        self.actor.eval()
        with self.torch.no_grad():
            logits = self.actor(depth, vector)
            probabilities, log_probabilities, no_valid = masked_policy(
                logits,
                action_mask,
                self.torch,
                temperature=float(temperature),
            )
            if bool(self.config.enable_primitive_neighbor_exploration):
                if self.primitive_neighborhood is None:
                    raise RuntimeError(
                        "primitive exploration is enabled but no neighborhood is attached"
                    )
                if anchor_primitive is None:
                    raise ValueError("primitive exploration requires anchor_primitive")
                from planning.awac.primitive_exploration import (
                    PrimitiveExplorationConfig,
                    primitive_behavior_distribution,
                )

                behavior = primitive_behavior_distribution(
                    probabilities.detach().cpu().numpy(),
                    action_mask.detach().cpu().numpy(),
                    anchor_primitive=int(anchor_primitive),
                    neighborhood=self.primitive_neighborhood,
                    config=PrimitiveExplorationConfig(
                        enabled=True,
                        neighbor_top_k=int(self.config.primitive_neighbor_top_k),
                        neighbor_radius=(
                            None
                            if float(self.config.primitive_neighbor_radius) < 0.0
                            else float(self.config.primitive_neighbor_radius)
                        ),
                        local_global_mix=float(self.config.primitive_local_global_mix),
                    ),
                )
                probabilities = self.torch.from_numpy(np.asarray(behavior)).to(
                    device=logits.device, dtype=logits.dtype
                )
                log_probabilities = probabilities.clamp_min(
                    self.torch.finfo(probabilities.dtype).tiny
                ).log()
            action = (
                self.torch.argmax(probabilities, dim=1)
                if bool(deterministic)
                else self.torch.multinomial(probabilities, num_samples=1).squeeze(1)
            )
            result = (
                action,
                log_probabilities.gather(1, action[:, None]).squeeze(1),
                probabilities,
                no_valid,
            )
        self.actor.train(was_training)
        return result

    def freeze_actor_for_calibration(self) -> None:
        """Hard-freeze all BC Actor parameters for the calibration phase."""

        self.actor_frozen_for_calibration = True
        self.actor.eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        self.actor_optimizer.zero_grad(set_to_none=True)

    def enable_actor_after_calibration_pass(self, handoff: Mapping) -> None:
        """Enable Actor gradients only for a validated Phase-1 handoff."""

        if not isinstance(handoff, Mapping) or handoff.get("handoff_allowed") is not True:
            raise RuntimeError(
                "Actor enablement requires a validated Calibration PASS handoff"
            )
        if handoff.get("actor_update_enabled") is not True:
            raise RuntimeError("Calibration handoff does not enable Actor updates")
        if int(self.actor_update_count) != 0 or int(self.actor_optimizer_step_count) != 0:
            raise RuntimeError(
                "Actor enablement requires zero pre-handoff Actor updates"
            )
        self.actor_frozen_for_calibration = False
        self.actor.train()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(True)
        self.actor_optimizer.zero_grad(set_to_none=True)

    def calibration_update(self, batch: Dict) -> Dict[str, float]:
        """Update only Twin Critics using BC-calibration behavior data."""

        self.freeze_actor_for_calibration()
        behavior = batch.get("behavior_source")
        if behavior is None:
            raise ValueError("calibration batch requires behavior_source")
        if not bool((behavior == int(BehaviorSource.BC_CALIBRATION)).all()):
            raise ValueError("calibration replay must use BC_CALIBRATION behavior")
        actor_before = self._state_fingerprint(self.actor.state_dict())
        actor_steps_before = int(self.actor_optimizer_step_count)
        metrics = self.update(batch, update_actor=False)
        actor_after = self._state_fingerprint(self.actor.state_dict())
        if actor_before != actor_after:
            raise RuntimeError("calibration changed the frozen Actor")
        if int(self.actor_optimizer_step_count) != actor_steps_before:
            raise RuntimeError("calibration stepped the frozen Actor optimizer")
        metrics.update(
            {
                "calibration_actor_update_enabled": 0.0,
                "critic_gradients_finite": True,
                "critic_update_count": float(self.critic_update_count),
                "actor_optimizer_step_count": float(
                    self.actor_optimizer_step_count
                ),
            }
        )
        return metrics

    def _state_fingerprint(self, state_dict: Mapping) -> str:
        import hashlib

        digest = hashlib.sha256()
        for name in sorted(state_dict):
            value = state_dict[name].detach().cpu().contiguous()
            digest.update(str(name).encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(value.numpy().tobytes(order="C"))
        return digest.hexdigest()

    def _validate_batch(self, batch: Mapping, *, require_behavior: bool = True) -> int:
        torch = self.torch
        required = {
            "depth",
            "vector",
            "action_mask",
            "action",
            "reward",
            "next_depth",
            "next_vector",
            "next_action_mask",
            "done",
        }
        if require_behavior:
            required.add("behavior_source")
        missing = sorted(required.difference(batch))
        if missing:
            raise ValueError("AWAC batch missing fields: {}".format(", ".join(missing)))
        depth = batch["depth"]
        if depth.ndim != 4 or int(depth.shape[1]) != self.depth_channels:
            raise ValueError("depth must have shape [B,C,H,W]")
        size = int(depth.shape[0])
        if size <= 0:
            raise ValueError("AWAC batch must be non-empty")
        if batch["next_depth"].shape != depth.shape:
            raise ValueError("next_depth shape differs from depth")
        if batch["vector"].ndim != 2 or int(batch["vector"].shape[0]) != size:
            raise ValueError("vector must have shape [B,V]")
        if batch["next_vector"].shape != batch["vector"].shape:
            raise ValueError("next_vector shape differs from vector")
        masks = batch["action_mask"]
        next_masks = batch["next_action_mask"]
        if masks.ndim != 2 or int(masks.shape[0]) != size:
            raise ValueError("action_mask must have shape [B,A]")
        if next_masks.shape != masks.shape:
            raise ValueError("next_action_mask shape differs from action_mask")
        if not bool(masks.bool().any(dim=1).all()):
            raise ValueError("AWAC batch row has no valid current action")

        actions = batch["action"]
        if actions.ndim != 1 or int(actions.shape[0]) != size:
            raise ValueError("action must have shape [B]")
        if bool(actions.dtype.is_floating_point) or bool(actions.dtype.is_complex):
            raise ValueError("action must have an integer dtype")
        if bool(((actions < 0) | (actions >= int(masks.shape[1]))).any()):
            raise ValueError("action is outside the action space")
        if not bool(masks.bool().gather(1, actions[:, None]).all()):
            raise ValueError("executed replay action is outside its action mask")

        for name in ("reward", "done"):
            value = batch[name]
            if value.ndim != 1 or int(value.shape[0]) != size:
                raise ValueError("{} must have shape [B]".format(name))
            if not bool(torch.isfinite(value).all()):
                raise ValueError("{} contains non-finite values".format(name))
        done = batch["done"]
        if not bool(((done == 0) | (done == 1)).all()):
            raise ValueError("done must contain only zero or one")
        if bool(((done < 0.5) & (~next_masks.bool().any(dim=1))).any()):
            raise ValueError("non-terminal batch row has no valid next action")
        for name in ("depth", "vector", "next_depth", "next_vector"):
            if not bool(torch.isfinite(batch[name]).all()):
                raise ValueError("{} contains non-finite values".format(name))

        if require_behavior:
            behavior = batch["behavior_source"]
            if behavior.ndim != 1 or int(behavior.shape[0]) != size:
                raise ValueError("behavior_source must have shape [B]")
            if bool(behavior.dtype.is_floating_point) or bool(behavior.dtype.is_complex):
                raise ValueError("behavior_source must have an integer dtype")
            allowed = torch.zeros_like(behavior, dtype=torch.bool)
            for source in BehaviorSource:
                allowed |= behavior == int(source)
            if not bool(allowed.all()):
                raise ValueError("behavior_source contains an unknown value")
        return size

    def _validate_trust_batch(self, batch: Mapping) -> None:
        required = {"depth", "vector", "action_mask"}
        missing = sorted(required.difference(batch))
        if missing:
            raise ValueError("AWAC trust batch missing fields: {}".format(", ".join(missing)))
        size = int(batch["depth"].shape[0])
        if batch["depth"].ndim != 4 or size <= 0:
            raise ValueError("trust depth must have shape [B,C,H,W]")
        if int(batch["depth"].shape[1]) != self.depth_channels:
            raise ValueError("trust depth channel count differs")
        if batch["vector"].ndim != 2 or int(batch["vector"].shape[0]) != size:
            raise ValueError("trust vector must have shape [B,V]")
        if batch["action_mask"].ndim != 2 or int(batch["action_mask"].shape[0]) != size:
            raise ValueError("trust action_mask must have shape [B,A]")
        if not bool(batch["action_mask"].bool().any(dim=1).all()):
            raise ValueError("trust batch row has no valid action")
        for name in ("depth", "vector"):
            if not bool(self.torch.isfinite(batch[name]).all()):
                raise ValueError("trust {} contains non-finite values".format(name))

    def _batch_bc_kl(self, batch: Mapping):
        torch = self.torch
        with torch.no_grad():
            actor_logits = self.actor(batch["depth"], batch["vector"])
            _, actor_log_probabilities, _ = masked_policy(
                actor_logits, batch["action_mask"], torch
            )
            bc_logits = self.bc_reference(batch["depth"], batch["vector"])
            bc_probabilities, bc_log_probabilities, _ = masked_policy(
                bc_logits, batch["action_mask"], torch
            )
            per_row = bc_probabilities * (
                bc_log_probabilities - actor_log_probabilities
            )
            per_row = per_row.sum(dim=1)
        return per_row.mean(), per_row.max()

    def _trust_tail_bc_kl(self, batch: Mapping):
        """Return observed trust KL plus a differentiable high-risk tail.

        Full trust batches are intentionally evaluated without gradients.  We
        then select only the highest-KL rows and recompute those rows with
        Actor gradients enabled.  This keeps recovery memory bounded while
        making the exact states that activated recovery part of its loss.
        """

        torch = self.torch
        with torch.no_grad():
            actor_logits = self.actor(batch["depth"], batch["vector"])
            _, actor_log_probabilities, _ = masked_policy(
                actor_logits, batch["action_mask"], torch
            )
            bc_logits = self.bc_reference(batch["depth"], batch["vector"])
            bc_probabilities, bc_log_probabilities, _ = masked_policy(
                bc_logits, batch["action_mask"], torch
            )
            observed_per_row = (
                bc_probabilities
                * (bc_log_probabilities - actor_log_probabilities)
            ).sum(dim=1)
            selected_count = min(
                int(self.config.trust_tail_top_k), int(observed_per_row.numel())
            )
            selected_indices = torch.topk(
                observed_per_row,
                k=selected_count,
                largest=True,
                sorted=False,
            ).indices

        selected_depth = batch["depth"].index_select(0, selected_indices)
        selected_vector = batch["vector"].index_select(0, selected_indices)
        selected_mask = batch["action_mask"].index_select(0, selected_indices)
        selected_actor_logits = self.actor(selected_depth, selected_vector)
        _, selected_actor_log_probabilities, _ = masked_policy(
            selected_actor_logits, selected_mask, torch
        )
        with torch.no_grad():
            selected_bc_logits = self.bc_reference(
                selected_depth, selected_vector
            )
            selected_bc_probabilities, selected_bc_log_probabilities, _ = masked_policy(
                selected_bc_logits, selected_mask, torch
            )
        differentiable_per_row = (
            selected_bc_probabilities
            * (
                selected_bc_log_probabilities
                - selected_actor_log_probabilities
            )
        ).sum(dim=1)
        return {
            "observed_mean": observed_per_row.mean(),
            "observed_max": observed_per_row.max(),
            "tail_mean": differentiable_per_row.mean(),
            "tail_max": differentiable_per_row.max(),
            "selected_count": selected_count,
        }

    def _snapshot_actor_update_state(self):
        return (
            [parameter.detach().clone() for parameter in self.actor.parameters()],
            copy.deepcopy(self.actor_optimizer.state_dict()),
        )

    def _restore_actor_update_state(self, snapshot) -> None:
        parameters, optimizer_state = snapshot
        with self.torch.no_grad():
            for parameter, value in zip(self.actor.parameters(), parameters):
                parameter.copy_(value)
        self.actor_optimizer.load_state_dict(optimizer_state)
        self.actor_optimizer.zero_grad(set_to_none=True)

    def _assert_finite_loss_and_gradients(self, loss, parameters, *, name: str) -> None:
        torch = self.torch
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError("{} loss is non-finite".format(name))
        for parameter in parameters:
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("{} gradient is non-finite".format(name))

    def _group_gradient_norms(self, optimizer) -> Dict[str, float]:
        """Return post-clipping gradient norms by the optimizer's named groups."""

        torch = self.torch
        result = {}
        for index, group in enumerate(optimizer.param_groups):
            name = str(group.get("name", "group_{}".format(index)))
            squared = None
            for parameter in group.get("params", ()):
                if parameter.grad is None:
                    continue
                value = (parameter.grad.detach() ** 2).sum()
                squared = value if squared is None else squared + value
            result[name] = (
                0.0
                if squared is None
                else float(torch.sqrt(squared).detach().cpu().item())
            )
        return result

    def _group_parameter_delta_norms(
        self, optimizer, parameters_before: Mapping[int, Any]
    ) -> Dict[str, float]:
        """Return accepted parameter deltas by the optimizer's named groups."""

        torch = self.torch
        result = {}
        for index, group in enumerate(optimizer.param_groups):
            name = str(group.get("name", "group_{}".format(index)))
            squared = None
            for parameter in group.get("params", ()):
                previous = parameters_before.get(id(parameter))
                if previous is None:
                    continue
                value = (parameter.detach() - previous).square().sum()
                squared = value if squared is None else squared + value
            result[name] = (
                0.0
                if squared is None
                else float(torch.sqrt(squared).detach().cpu().item())
            )
        return result

    def _parameter_delta_norm(
        self, parameters, parameters_before: Mapping[int, Any]
    ) -> float:
        """Return the L2 delta for one model parameter collection."""

        squared = None
        for parameter in parameters:
            previous = parameters_before.get(id(parameter))
            if previous is None:
                continue
            value = (parameter.detach() - previous).square().sum()
            squared = value if squared is None else squared + value
        return (
            0.0
            if squared is None
            else float(self.torch.sqrt(squared).detach().cpu().item())
        )

    def _apply_actor_proposal(
        self,
        *,
        batch: Mapping,
        actor_trust_batch: Optional[Mapping],
        actor_optimization_loss,
        recovery_active: bool,
        kl_before: float,
        kl_max_before: float,
        bc_kl,
        bc_kl_max,
        update_actor: bool,
    ) -> Dict[str, Any]:
        """Run the production KL-guarded Actor proposal transaction.

        Both normal ``update`` and controlled fixed-Critic experiments use
        this one owner.  In particular, recovery, rollback, named optimizer
        groups, gradient clipping, and accepted/rejected counters cannot
        silently diverge between the production and actor-only paths.
        """

        torch = self.torch
        config = self.config
        actor_optimizer_step = False
        actor_awac_optimizer_step = False
        actor_recovery_optimizer_step = False
        actor_trust_region_rejection = False
        recovery_mean_improved = False
        recovery_max_improved = False
        kl_after = bc_kl.detach()
        kl_max_after = bc_kl_max.detach()
        actor_gradient_norm = torch.zeros((), device=self.device)
        actor_group_gradient_norms: Dict[str, float] = {}
        actor_group_parameter_delta_norms: Dict[str, float] = {}
        actor_parameter_delta_norm = 0.0

        if bool(update_actor):
            self.actor_optimizer.zero_grad(set_to_none=True)
            snapshot = self._snapshot_actor_update_state()
            actor_parameters = [
                parameter for parameter in self.actor.parameters() if parameter.requires_grad
            ]
            actor_parameters_before = {
                id(parameter): parameter.detach().clone()
                for parameter in actor_parameters
            }
            self._assert_finite_loss_and_gradients(
                actor_optimization_loss, actor_parameters, name="Actor"
            )
            actor_optimization_loss.backward()
            self._assert_finite_loss_and_gradients(
                actor_optimization_loss, actor_parameters, name="Actor"
            )
            actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                actor_parameters, float(config.gradient_clip_norm)
            )
            actor_group_gradient_norms = self._group_gradient_norms(
                self.actor_optimizer
            )
            self.actor_optimizer.step()
            proposed_batch_kl, proposed_batch_kl_max = self._batch_bc_kl(batch)
            proposed_trust_kl = proposed_batch_kl
            proposed_trust_kl_max = proposed_batch_kl_max
            if actor_trust_batch is not None:
                proposed_trust_kl, proposed_trust_kl_max = self._batch_bc_kl(
                    actor_trust_batch
                )
            proposed_mean = max(
                float(proposed_batch_kl.detach().cpu().item()),
                float(proposed_trust_kl.detach().cpu().item()),
            )
            proposed_max = max(
                float(proposed_batch_kl_max.detach().cpu().item()),
                float(proposed_trust_kl_max.detach().cpu().item()),
            )
            recovery_mean_improved = proposed_mean < kl_before - 1.0e-8
            recovery_max_improved = proposed_max < kl_max_before - 1.0e-8
            accepted = kl_proposal_is_acceptable(
                recovery_active=bool(recovery_active),
                budget=float(config.bc_kl_hard_budget),
                before_mean=float(kl_before),
                before_max=float(kl_max_before),
                proposed_mean=proposed_mean,
                proposed_max=proposed_max,
            )
            if accepted:
                actor_optimizer_step = True
                self.actor_optimizer_step_count += 1
                actor_recovery_optimizer_step = bool(recovery_active)
                actor_awac_optimizer_step = not bool(recovery_active)
                self.actor_update_count += 1
                if recovery_active:
                    self.actor_recovery_update_count += 1
                else:
                    self.actor_awac_update_count += 1
                kl_after = torch.maximum(proposed_batch_kl, proposed_trust_kl)
                kl_max_after = torch.maximum(
                    proposed_batch_kl_max, proposed_trust_kl_max
                )
            else:
                self._restore_actor_update_state(snapshot)
                actor_trust_region_rejection = True
                self.actor_trust_region_rejection_count += 1
            actor_group_parameter_delta_norms = self._group_parameter_delta_norms(
                self.actor_optimizer, actor_parameters_before
            )
            actor_parameter_delta_norm = self._parameter_delta_norm(
                actor_parameters, actor_parameters_before
            )

        return {
            "actor_optimizer_step": bool(actor_optimizer_step),
            "actor_awac_optimizer_step": bool(actor_awac_optimizer_step),
            "actor_recovery_optimizer_step": bool(actor_recovery_optimizer_step),
            "actor_trust_region_rejection": bool(actor_trust_region_rejection),
            "recovery_mean_improved": bool(recovery_mean_improved),
            "recovery_max_improved": bool(recovery_max_improved),
            "kl_after": kl_after,
            "kl_max_after": kl_max_after,
            "actor_gradient_norm": actor_gradient_norm,
            "actor_group_gradient_norms": actor_group_gradient_norms,
            "actor_group_parameter_delta_norms": actor_group_parameter_delta_norms,
            "actor_parameter_delta_norm": float(actor_parameter_delta_norm),
        }

    def _stratum_metrics(self, *, advantage, raw_weight, weight, done, reward, behavior):
        torch = self.torch
        total = float(advantage.numel())
        strata = {
            "nonterminal": done < 0.5,
            "terminal_positive_reward": (done >= 0.5) & (reward > 0.0),
            "terminal_nonpositive_reward": (done >= 0.5) & (reward <= 0.0),
            "behavior_bc_warmup": behavior == int(BehaviorSource.BC_WARMUP),
            "behavior_accepted_collection_policy": behavior
            == int(BehaviorSource.ACCEPTED_COLLECTION_POLICY),
        }
        result = {}
        total_mass = weight.sum().clamp_min(torch.finfo(weight.dtype).tiny)
        for name, mask in strata.items():
            count = int(mask.sum().detach().cpu().item())
            prefix = "awac_stratum_{}_".format(name)
            result[prefix + "count"] = float(count)
            result[prefix + "fraction"] = float(count) / total
            if count <= 0:
                result[prefix + "advantage_mean"] = 0.0
                result[prefix + "positive_advantage_rate"] = 0.0
                result[prefix + "raw_weight_mean"] = 0.0
                result[prefix + "weight_mean"] = 0.0
                result[prefix + "weight_mass_fraction"] = 0.0
                continue
            result[prefix + "advantage_mean"] = float(
                advantage[mask].mean().detach().cpu().item()
            )
            result[prefix + "positive_advantage_rate"] = float(
                (advantage[mask] > 0.0).float().mean().detach().cpu().item()
            )
            result[prefix + "raw_weight_mean"] = float(
                raw_weight[mask].mean().detach().cpu().item()
            )
            result[prefix + "weight_mean"] = float(
                weight[mask].mean().detach().cpu().item()
            )
            result[prefix + "weight_mass_fraction"] = float(
                (weight[mask].sum() / total_mass).detach().cpu().item()
            )
        return result

    def update(
        self,
        batch: Dict,
        *,
        update_actor: bool = True,
        actor_trust_batch: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Run one Critic update and an optional KL-guarded AWAC Actor update."""
        if self.actor_frozen_for_calibration and bool(update_actor):
            raise RuntimeError(
                "Actor updates are disabled while critic calibration is active"
            )
        torch = self.torch
        config = self.config
        batch_size = self._validate_batch(batch, require_behavior=True)
        if actor_trust_batch is not None:
            self._validate_trust_batch(actor_trust_batch)

        with torch.no_grad():
            terminal = batch["done"].bool()
            nonterminal = ~terminal
            next_empty = ~batch["next_action_mask"].bool().any(dim=1)
            next_value = torch.zeros_like(batch["reward"])
            target = batch["reward"] * float(config.reward_scale)

            # A terminal transition has a reward-only Bellman target.  In
            # particular, terminal observations may intentionally carry the
            # all-zero action mask used by the runtime for success, collision,
            # dead-end, timeout, and hard-altitude outcomes.  Do not evaluate
            # the Actor or target Critics for those rows before multiplying a
            # bootstrap term by zero: that would require inventing a valid
            # action distribution for a state that has no continuation.
            if bool(nonterminal.any().item()):
                next_logits = self.actor(
                    batch["next_depth"][nonterminal],
                    batch["next_vector"][nonterminal],
                )
                target_q = torch.minimum(
                    self.target_critic1(
                        batch["next_depth"][nonterminal],
                        batch["next_vector"][nonterminal],
                    ),
                    self.target_critic2(
                        batch["next_depth"][nonterminal],
                        batch["next_vector"][nonterminal],
                    ),
                )
                target_info = masked_bellman_target(
                    bc_logits=next_logits,
                    target_q1=target_q,
                    target_q2=target_q,
                    next_action_mask=batch["next_action_mask"][nonterminal],
                    reward=batch["reward"][nonterminal],
                    done=batch["done"][nonterminal],
                    gamma=float(config.gamma),
                    reward_scale=float(config.reward_scale),
                    torch=torch,
                )
                next_value[nonterminal] = target_info["next_value"]
                target[nonterminal] = target_info["target"]

        q1_all = self.critic1(batch["depth"], batch["vector"])
        q2_all = self.critic2(batch["depth"], batch["vector"])
        q1 = q1_all.gather(1, batch["action"][:, None]).squeeze(1)
        q2 = q2_all.gather(1, batch["action"][:, None]).squeeze(1)
        critic1_loss = torch.nn.functional.mse_loss(q1, target)
        critic2_loss = torch.nn.functional.mse_loss(q2, target)
        critic_td_loss = critic1_loss + critic2_loss
        critic1_cql_loss = masked_discrete_cql_loss(
            q1_all, batch["action"], batch["action_mask"], torch
        )
        critic2_cql_loss = masked_discrete_cql_loss(
            q2_all, batch["action"], batch["action_mask"], torch
        )
        critic_cql_loss = critic1_cql_loss + critic2_cql_loss
        critic_loss = critic_td_loss + float(config.critic_cql_weight) * critic_cql_loss
        critic_parameters = [
            parameter
            for critic in (self.critic1, self.critic2)
            for parameter in critic.parameters()
            if parameter.requires_grad
        ]
        critic_parameters_before = {
            id(parameter): parameter.detach().clone()
            for parameter in critic_parameters
        }
        self.critic_optimizer.zero_grad(set_to_none=True)
        self._assert_finite_loss_and_gradients(
            critic_loss, critic_parameters, name="Critic"
        )
        critic_loss.backward()
        self._assert_finite_loss_and_gradients(
            critic_loss, critic_parameters, name="Critic"
        )
        torch.nn.utils.clip_grad_norm_(critic_parameters, float(config.gradient_clip_norm))
        critic_gradient_norm = torch.sqrt(
            sum(
                (parameter.grad.detach() ** 2).sum()
                for parameter in critic_parameters
                if parameter.grad is not None
            )
        )
        self.critic_optimizer.step()
        self.critic_update_count += 1
        critic_group_gradient_norms = self._group_gradient_norms(self.critic_optimizer)
        critic_group_parameter_delta_norms = self._group_parameter_delta_norms(
            self.critic_optimizer, critic_parameters_before
        )
        critic_parameter_delta_norm = self._parameter_delta_norm(
            critic_parameters, critic_parameters_before
        )
        critic1_parameters = [
            parameter for parameter in self.critic1.parameters() if parameter.requires_grad
        ]
        critic2_parameters = [
            parameter for parameter in self.critic2.parameters() if parameter.requires_grad
        ]
        critic1_parameter_delta_norm = self._parameter_delta_norm(
            critic1_parameters, critic_parameters_before
        )
        critic2_parameter_delta_norm = self._parameter_delta_norm(
            critic2_parameters, critic_parameters_before
        )

        logits = self.actor(batch["depth"], batch["vector"])
        probabilities, log_probabilities, _ = masked_policy(
            logits, batch["action_mask"], torch
        )
        data_log_probability = log_probabilities.gather(
            1, batch["action"][:, None]
        ).squeeze(1)
        with torch.no_grad():
            minimum_q = torch.minimum(
                self.critic1(batch["depth"], batch["vector"]),
                self.critic2(batch["depth"], batch["vector"]),
            )
            data_q = minimum_q.gather(1, batch["action"][:, None]).squeeze(1)
            state_value = (probabilities.detach() * minimum_q).sum(dim=1)
            advantage = data_q - state_value
            weights = awac_advantage_weights(
                advantage,
                temperature=float(config.awac_temperature),
                weight_max=float(config.awac_weight_max),
                torch=torch,
            )
            raw_weight = weights["raw"]
            normalized_weight = weights["normalized"]
            bc_logits = self.bc_reference(batch["depth"], batch["vector"])
            bc_probabilities, bc_log_probabilities, _ = masked_policy(
                bc_logits, batch["action_mask"], torch
            )
            confidence_result = None
            if bool(config.enable_twin_q_confidence) or bool(
                config.enable_adaptive_bc_kl
            ):
                confidence_result = self.confidence_estimator.estimate(
                    self.critic1(batch["depth"], batch["vector"]),
                    self.critic2(batch["depth"], batch["vector"]),
                    batch["action_mask"],
                    bc_policy=bc_probabilities.detach(),
                    rl_policy=probabilities.detach(),
                )
        awac_actor_loss = -(normalized_weight * data_log_probability).mean()
        bc_kl_per_row = (
            bc_probabilities * (bc_log_probabilities - log_probabilities)
        ).sum(dim=1)
        bc_kl = bc_kl_per_row.mean()
        bc_kl_max = bc_kl_per_row.max()
        trust_tail_mean = bc_kl.new_zeros(())
        trust_tail_max = bc_kl.new_zeros(())
        trust_tail_selected_count = 0

        batch_kl_before = float(bc_kl.detach().cpu().item())
        batch_kl_max_before = float(bc_kl_max.detach().cpu().item())
        trust_kl_before = bc_kl.detach()
        trust_kl_max_before = bc_kl_max.detach()
        if actor_trust_batch is not None:
            if bool(update_actor):
                trust_tail = self._trust_tail_bc_kl(actor_trust_batch)
                trust_kl_before = trust_tail["observed_mean"]
                trust_kl_max_before = trust_tail["observed_max"]
                trust_tail_mean = trust_tail["tail_mean"]
                trust_tail_max = trust_tail["tail_max"]
                trust_tail_selected_count = int(trust_tail["selected_count"])
            else:
                trust_kl_before, trust_kl_max_before = self._batch_bc_kl(
                    actor_trust_batch
                )
        trust_tail_penalty = trust_tail_mean + trust_tail_max
        if bool(config.enable_adaptive_bc_kl):
            if confidence_result is None:
                raise RuntimeError("adaptive BC KL requires a confidence result")
            adaptive_bc_kl, beta_per_row = adaptive_bc_kl_loss(
                bc_kl_per_row,
                confidence_result["confidence"],
                config=self.adaptive_bc_kl_config,
                torch=torch,
                enabled=True,
            )
            beta_mean = beta_per_row.mean()
            adaptive_trust_tail_penalty = beta_mean * trust_tail_penalty
            actor_loss = (
                awac_actor_loss + adaptive_bc_kl + adaptive_trust_tail_penalty
            )
        else:
            # Keep the established scalar arithmetic byte-for-byte in spirit
            # when every innovation feature is disabled.
            beta_per_row = torch.full_like(bc_kl_per_row, float(config.bc_kl_weight))
            adaptive_bc_kl = float(config.bc_kl_weight) * bc_kl
            adaptive_trust_tail_penalty = float(config.bc_kl_weight) * trust_tail_penalty
            beta_mean = beta_per_row.mean()
            actor_loss = awac_actor_loss + float(config.bc_kl_weight) * (
                bc_kl + trust_tail_penalty
            )
        trust_kl_before_value = float(trust_kl_before.detach().cpu().item())
        trust_kl_max_before_value = float(trust_kl_max_before.detach().cpu().item())
        kl_before = max(batch_kl_before, trust_kl_before_value)
        kl_max_before = max(batch_kl_max_before, trust_kl_max_before_value)
        recovery_active = kl_budget_exceeded(
            mean_kl=kl_before,
            max_kl=kl_max_before,
            budget=float(config.bc_kl_hard_budget),
        )
        actor_optimization_loss = (
            float(config.bc_kl_recovery_weight)
            * (bc_kl + trust_tail_penalty)
            if recovery_active
            else actor_loss
        )
        actor_proposal = self._apply_actor_proposal(
            batch=batch,
            actor_trust_batch=actor_trust_batch,
            actor_optimization_loss=actor_optimization_loss,
            recovery_active=recovery_active,
            kl_before=kl_before,
            kl_max_before=kl_max_before,
            bc_kl=bc_kl,
            bc_kl_max=bc_kl_max,
            update_actor=bool(update_actor),
        )
        actor_optimizer_step = actor_proposal["actor_optimizer_step"]
        actor_awac_optimizer_step = actor_proposal["actor_awac_optimizer_step"]
        actor_recovery_optimizer_step = actor_proposal[
            "actor_recovery_optimizer_step"
        ]
        actor_trust_region_rejection = actor_proposal[
            "actor_trust_region_rejection"
        ]
        recovery_mean_improved = actor_proposal["recovery_mean_improved"]
        recovery_max_improved = actor_proposal["recovery_max_improved"]
        kl_after = actor_proposal["kl_after"]
        kl_max_after = actor_proposal["kl_max_after"]
        actor_gradient_norm = actor_proposal["actor_gradient_norm"]
        actor_group_gradient_norms = actor_proposal["actor_group_gradient_norms"]
        actor_group_parameter_delta_norms = actor_proposal[
            "actor_group_parameter_delta_norms"
        ]
        actor_parameter_delta_norm = actor_proposal["actor_parameter_delta_norm"]

        soft_update(self.target_critic1, self.critic1, float(config.tau))
        soft_update(self.target_critic2, self.critic2, float(config.tau))
        self.update_step += 1

        raw_ess = _effective_sample_size(raw_weight, torch)
        normalized_ess = _effective_sample_size(normalized_weight, torch)
        quantiles = torch.quantile(
            advantage.detach(),
            torch.tensor([0.05, 0.50, 0.95], device=advantage.device),
        )
        unexecuted_mask = batch["action_mask"].bool().clone()
        unexecuted_mask.scatter_(1, batch["action"][:, None], False)
        unexecuted_count = int(unexecuted_mask.sum().detach().cpu().item())
        minimum_q_all = torch.minimum(q1_all, q2_all).detach()
        executed_q = torch.minimum(q1, q2).detach()
        q_quantiles = torch.quantile(
            executed_q,
            torch.tensor([0.01, 0.99], device=executed_q.device),
        )
        twin_q_disagreement = (q1.detach() - q2.detach()).abs()
        twin_q_disagreement_mean = float(
            twin_q_disagreement.mean().detach().cpu().item()
        )
        twin_q_disagreement_p95 = float(
            torch.quantile(
                twin_q_disagreement,
                torch.tensor(0.95, device=twin_q_disagreement.device),
            )
            .detach()
            .cpu()
            .item()
        )
        actor_entropy = -(
            probabilities.detach()
            * log_probabilities.detach()
        ).sum(dim=1)
        bc_awac_top1_disagreement_rate = float(
            (
                probabilities.detach().argmax(dim=1)
                != bc_probabilities.detach().argmax(dim=1)
            )
            .float()
            .mean()
            .cpu()
            .item()
        )
        critic_td_per_row = ((q1.detach() - target.detach()).square() +
                             (q2.detach() - target.detach()).square())
        if confidence_result is not None:
            confidence_values = confidence_result["confidence"].detach()
            beta_values = beta_per_row.detach()
            confidence_p25, confidence_p75 = torch.quantile(
                confidence_values,
                torch.tensor([0.25, 0.75], device=confidence_values.device),
            )
            low_confidence_mask = confidence_values <= confidence_p25
            high_confidence_mask = confidence_values >= confidence_p75
            low_confidence_beta_mean = beta_values[low_confidence_mask].mean()
            high_confidence_beta_mean = beta_values[high_confidence_mask].mean()
        else:
            low_confidence_beta_mean = beta_mean.detach()
            high_confidence_beta_mean = beta_mean.detach()
        unexecuted_q_mean = (
            float(minimum_q_all[unexecuted_mask].mean().cpu().item())
            if unexecuted_count > 0
            else 0.0
        )
        metrics = {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "critic_td_loss": float(critic_td_loss.detach().cpu().item()),
            "critic_cql_loss": float(critic_cql_loss.detach().cpu().item()),
            "critic1_loss": float(critic1_loss.detach().cpu().item()),
            "critic2_loss": float(critic2_loss.detach().cpu().item()),
            "critic1_cql_loss": float(critic1_cql_loss.detach().cpu().item()),
            "critic2_cql_loss": float(critic2_cql_loss.detach().cpu().item()),
            "critic_cql_weight": float(config.critic_cql_weight),
            "reward_scale": float(config.reward_scale),
            "cql_valid_unexecuted_count_mean": float(unexecuted_count) / float(batch_size),
            "cql_unexecuted_q_mean": unexecuted_q_mean,
            "cql_executed_q_mean": float(executed_q.mean().cpu().item()),
            "q_mean": float(executed_q.mean().cpu().item()),
            "target_q_mean": float(target.mean().detach().cpu().item()),
            "target_q_std": float(target.std(unbiased=False).detach().cpu().item()),
            "q_std": float(executed_q.std(unbiased=False).cpu().item()),
            "q_p01": float(q_quantiles[0].detach().cpu().item()),
            "q_p99": float(q_quantiles[1].detach().cpu().item()),
            "critic_td_loss_p95": float(
                torch.quantile(
                    critic_td_per_row,
                    torch.tensor(0.95, device=critic_td_per_row.device),
                )
                .detach()
                .cpu()
                .item()
            ),
            "twin_q_disagreement_mean": twin_q_disagreement_mean,
            "twin_q_disagreement_p95": twin_q_disagreement_p95,
            "twin_disagreement_mean": twin_q_disagreement_mean,
            "twin_disagreement_p95": twin_q_disagreement_p95,
            "critic_gradient_norm": float(critic_gradient_norm.detach().cpu().item()),
            "critic_parameter_delta_norm": critic_parameter_delta_norm,
            "critic1_parameter_delta_norm": critic1_parameter_delta_norm,
            "critic2_parameter_delta_norm": critic2_parameter_delta_norm,
            "critic1_updated": 1.0,
            "critic2_updated": 1.0,
            "target_critic1_updated": 1.0,
            "target_critic2_updated": 1.0,
            "actor_gradient_norm": float(
                actor_gradient_norm.detach().cpu().item()
                if bool(update_actor)
                else 0.0
            ),
            "actor_parameter_delta_norm": float(actor_parameter_delta_norm),
            "next_value_mean": float(next_value.mean().detach().cpu().item()),
            "target_policy_support_size_mean": float(
                batch["next_action_mask"].sum(dim=1).float().mean().detach().cpu().item()
            ),
            "target_policy_empty_mask_rate": float(next_empty.float().mean().detach().cpu().item()),
            "actor_optimization_support_size_mean": float(
                batch["action_mask"].sum(dim=1).float().mean().detach().cpu().item()
            ),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "actor_optimization_loss": float(actor_optimization_loss.detach().cpu().item()),
            "awac_actor_loss": float(awac_actor_loss.detach().cpu().item()),
            "awac_temperature": float(config.awac_temperature),
            "awac_weight_max_config": float(config.awac_weight_max),
            "awac_data_log_probability_mean": float(data_log_probability.mean().detach().cpu().item()),
            "awac_data_q_mean": float(data_q.mean().detach().cpu().item()),
            "awac_state_value_mean": float(state_value.mean().detach().cpu().item()),
            "awac_advantage_mean": float(advantage.mean().detach().cpu().item()),
            "awac_advantage_std": float(
                advantage.std(unbiased=False).detach().cpu().item()
            ),
            "awac_advantage_min": float(advantage.min().detach().cpu().item()),
            "awac_advantage_max": float(advantage.max().detach().cpu().item()),
            "awac_advantage_positive_rate": float((advantage > 0).float().mean().detach().cpu().item()),
            "awac_advantage_p05": float(quantiles[0].detach().cpu().item()),
            "awac_advantage_p50": float(quantiles[1].detach().cpu().item()),
            "awac_advantage_p95": float(quantiles[2].detach().cpu().item()),
            "awac_raw_weight_mean": float(raw_weight.mean().detach().cpu().item()),
            "awac_raw_weight_min": float(raw_weight.min().detach().cpu().item()),
            "awac_raw_weight_max": float(raw_weight.max().detach().cpu().item()),
            "awac_raw_weight_high_clip_rate": float(weights["raw_high_clip"].float().mean().detach().cpu().item()),
            "awac_raw_weight_low_floor_rate": float(weights["raw_low_floor"].float().mean().detach().cpu().item()),
            "awac_weight_mean": float(normalized_weight.mean().detach().cpu().item()),
            "awac_weight_p50": float(
                torch.quantile(
                    normalized_weight.detach(),
                    torch.tensor(0.50, device=normalized_weight.device),
                )
                .cpu()
                .item()
            ),
            "awac_weight_p95": float(
                torch.quantile(
                    normalized_weight.detach(),
                    torch.tensor(0.95, device=normalized_weight.device),
                )
                .cpu()
                .item()
            ),
            "awac_weight_max": float(normalized_weight.max().detach().cpu().item()),
            "awac_weight_high_clip_rate": float(weights["normalized_high_clip"].float().mean().detach().cpu().item()),
            "awac_weight_clip_fraction": float(
                weights["normalized_high_clip"].float().mean().detach().cpu().item()
            ),
            "awac_raw_weight_ess": float(raw_ess.detach().cpu().item()),
            "awac_raw_weight_ess_fraction": float((raw_ess / float(batch_size)).detach().cpu().item()),
            "awac_weight_ess": float(normalized_ess.detach().cpu().item()),
            "awac_weight_ess_fraction": float((normalized_ess / float(batch_size)).detach().cpu().item()),
            "bc_kl": float(kl_after.detach().cpu().item()),
            "bc_kl_before_update": float(kl_before),
            "bc_kl_max_before_update": float(kl_max_before),
            "bc_kl_batch_before_update": float(batch_kl_before),
            "bc_kl_batch_max_before_update": float(batch_kl_max_before),
            "bc_kl_trust_before_update": float(trust_kl_before_value),
            "bc_kl_trust_max_before_update": float(trust_kl_max_before_value),
            "bc_kl_max_after_update": float(kl_max_after.detach().cpu().item()),
            "actor_entropy": float(actor_entropy.mean().detach().cpu().item()),
            "actor_entropy_p95": float(
                torch.quantile(
                    actor_entropy.detach(),
                    torch.tensor(0.95, device=actor_entropy.device),
                )
                .cpu()
                .item()
            ),
            "bc_awac_top1_disagreement_rate": bc_awac_top1_disagreement_rate,
            "bc_kl_hard_budget": float(config.bc_kl_hard_budget),
            "bc_kl_budget_exceeded": float(recovery_active),
            "bc_kl_recovery_mean_improved": float(recovery_mean_improved),
            "bc_kl_recovery_max_improved": float(recovery_max_improved),
            "bc_kl_weight": float(config.bc_kl_weight),
            "confidence_mean": float(
                confidence_result["confidence"].mean().detach().cpu().item()
                if confidence_result is not None
                else 0.0
            ),
            "confidence_p05": float(
                torch.quantile(
                    confidence_result["confidence"].detach(),
                    torch.tensor(0.05, device=confidence_result["confidence"].device),
                ).cpu().item()
                if confidence_result is not None
                else 0.0
            ),
            "confidence_p50": float(
                torch.quantile(
                    confidence_result["confidence"].detach(),
                    torch.tensor(0.50, device=confidence_result["confidence"].device),
                ).cpu().item()
                if confidence_result is not None
                else 0.0
            ),
            "confidence_p95": float(
                torch.quantile(
                    confidence_result["confidence"].detach(),
                    torch.tensor(0.95, device=confidence_result["confidence"].device),
                ).cpu().item()
                if confidence_result is not None
                else 0.0
            ),
            "confidence_q_margin_norm_mean": float(
                confidence_result["normalized_margin"].mean().detach().cpu().item()
                if confidence_result is not None
                else 0.0
            ),
            "confidence_delta_q_mean": float(
                confidence_result["delta_q"].mean().detach().cpu().item()
                if confidence_result is not None
                else 0.0
            ),
            "confidence_delta_q_norm_mean": float(
                confidence_result["normalized_delta_q"].mean().detach().cpu().item()
                if confidence_result is not None
                else 0.0
            ),
            "confidence_uncertainty_mean": float(
                confidence_result["uncertainty"].mean().detach().cpu().item()
                if confidence_result is not None
                else 0.0
            ),
            "confidence_uncertainty_norm_mean": float(
                confidence_result["normalized_uncertainty"].mean().detach().cpu().item()
                if confidence_result is not None
                else 0.0
            ),
            "confidence_twin_disagreement_norm_mean": float(
                confidence_result["normalized_twin_disagreement"].mean().detach().cpu().item()
                if confidence_result is not None
                else 0.0
            ),
            "adaptive_bc_kl_beta_mean": float(beta_mean.detach().cpu().item()),
            "adaptive_bc_kl_beta_p05": float(
                torch.quantile(
                    beta_per_row.detach(),
                    torch.tensor(0.05, device=beta_per_row.device),
                ).cpu().item()
            ),
            "adaptive_bc_kl_beta_p95": float(
                torch.quantile(
                    beta_per_row.detach(),
                    torch.tensor(0.95, device=beta_per_row.device),
                ).cpu().item()
            ),
            "adaptive_bc_kl_beta_p50": float(
                torch.quantile(
                    beta_per_row.detach(),
                    torch.tensor(0.50, device=beta_per_row.device),
                ).cpu().item()
            ),
            "adaptive_bc_kl_low_confidence_beta_mean": float(
                low_confidence_beta_mean.cpu().item()
            ),
            "adaptive_bc_kl_high_confidence_beta_mean": float(
                high_confidence_beta_mean.cpu().item()
            ),
            "adaptive_bc_kl_enabled": float(bool(config.enable_adaptive_bc_kl)),
            "trust_tail_top_k_config": float(config.trust_tail_top_k),
            "trust_tail_selected_count": float(trust_tail_selected_count),
            "trust_tail_bc_kl_mean_before_update": float(
                trust_tail_mean.detach().cpu().item()
            ),
            "trust_tail_bc_kl_max_before_update": float(
                trust_tail_max.detach().cpu().item()
            ),
            "trust_tail_penalty": float(
                trust_tail_penalty.detach().cpu().item()
            ),
            "trust_tail_gradient_enabled": float(
                actor_trust_batch is not None and bool(update_actor)
            ),
            "actor_trust_batch_enabled": float(actor_trust_batch is not None),
            "actor_update_enabled": float(bool(update_actor)),
            "actor_optimizer_step": float(actor_optimizer_step),
            "actor_awac_optimizer_step": float(actor_awac_optimizer_step),
            "policy_optimizer_step": float(actor_awac_optimizer_step),
            "actor_recovery_optimizer_step": float(actor_recovery_optimizer_step),
            "actor_trust_region_rejection": float(actor_trust_region_rejection),
            "actor_update_count": float(self.actor_update_count),
            "actor_awac_update_count": float(self.actor_awac_update_count),
            "policy_update_count": float(self.policy_update_count),
            "actor_recovery_update_count": float(self.actor_recovery_update_count),
            "actor_trust_region_rejection_count": float(self.actor_trust_region_rejection_count),
        }
        for group_name, value in critic_group_gradient_norms.items():
            metrics["{}_gradient_norm".format(group_name)] = float(value)
        for group_name, value in critic_group_parameter_delta_norms.items():
            metrics["{}_parameter_delta_norm".format(group_name)] = float(value)
        for group_name, value in actor_group_gradient_norms.items():
            metrics["actor_{}_gradient_norm".format(group_name)] = float(value)
        for group_name, value in actor_group_parameter_delta_norms.items():
            metrics["actor_{}_parameter_delta_norm".format(group_name)] = float(value)
        actor_group_aliases = {
            "depth": "depth_encoder",
            "vector": "vector_encoder",
            "head": "head",
        }
        for component, group_name in actor_group_aliases.items():
            gradient = float(actor_group_gradient_norms.get(group_name, 0.0))
            delta = float(actor_group_parameter_delta_norms.get(group_name, 0.0))
            metrics["actor_{}_gradient_norm".format(component)] = gradient
            metrics["actor_{}_parameter_delta_norm".format(component)] = delta
            metrics["actor_{}_update_norm".format(component)] = delta
            metrics["actor_{}_update".format(component)] = float(
                bool(actor_optimizer_step) and delta > 0.0
            )
        metrics.update(
            self._stratum_metrics(
                advantage=advantage,
                raw_weight=raw_weight,
                weight=normalized_weight,
                done=batch["done"],
                reward=batch["reward"],
                behavior=batch["behavior_source"],
            )
        )
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise FloatingPointError("AWAC metrics contain non-finite values")
        return metrics

    def actor_only_update(
        self,
        batch: Dict,
        *,
        actor_trust_batch: Optional[Dict] = None,
        weight_mode: str = "awac",
        update_actor: bool = True,
    ) -> Dict[str, float]:
        """Apply one production Actor proposal without mutating any Critic.

        This is intentionally not ``update(..., critic_lr=0)``.  No Critic
        optimizer operation, target soft update, Critic counter, or global
        learner update counter is touched.  The Actor proposal itself delegates
        to ``_apply_actor_proposal`` so it retains the production KL recovery,
        rejection rollback, and gradient-clipping semantics exactly.  Passing
        ``update_actor=False`` is an explicitly side-effect-free diagnostic
        measurement for a fixed validation batch; it does not even clear
        Actor gradients or optimizer state.
        """

        if self.actor_frozen_for_calibration:
            raise RuntimeError(
                "Actor-only updates require a validated Calibration PASS handoff"
            )
        mode = str(weight_mode).strip().lower()
        if mode not in {"awac", "uniform"}:
            raise ValueError("actor-only weight_mode must be 'awac' or 'uniform'")

        torch = self.torch
        config = self.config
        batch_size = self._validate_batch(batch, require_behavior=True)
        if actor_trust_batch is not None:
            self._validate_trust_batch(actor_trust_batch)

        # Freeze all non-Actor modules at the module boundary, not merely by
        # omitting ``optimizer.step``.  This preserves BatchNorm-like buffers
        # too, and makes the fixed-Critic control a true state-isolation path.
        frozen_modules = (
            self.critic1,
            self.critic2,
            self.target_critic1,
            self.target_critic2,
            self.bc_reference,
        )
        training_modes = [bool(module.training) for module in frozen_modules]
        for module in frozen_modules:
            module.eval()
        try:
            logits = self.actor(batch["depth"], batch["vector"])
            probabilities, log_probabilities, _ = masked_policy(
                logits, batch["action_mask"], torch
            )
            data_log_probability = log_probabilities.gather(
                1, batch["action"][:, None]
            ).squeeze(1)
            with torch.no_grad():
                minimum_q = torch.minimum(
                    self.critic1(batch["depth"], batch["vector"]),
                    self.critic2(batch["depth"], batch["vector"]),
                )
                data_q = minimum_q.gather(1, batch["action"][:, None]).squeeze(1)
                state_value = (probabilities.detach() * minimum_q).sum(dim=1)
                advantage = data_q - state_value
                if mode == "awac":
                    weights = awac_advantage_weights(
                        advantage,
                        temperature=float(config.awac_temperature),
                        weight_max=float(config.awac_weight_max),
                        torch=torch,
                    )
                    raw_weight = weights["raw"]
                    normalized_weight = weights["normalized"]
                    raw_high_clip = weights["raw_high_clip"]
                    raw_low_floor = weights["raw_low_floor"]
                    normalized_high_clip = weights["normalized_high_clip"]
                else:
                    raw_weight = torch.ones_like(advantage)
                    normalized_weight = torch.ones_like(advantage)
                    raw_high_clip = torch.zeros_like(advantage, dtype=torch.bool)
                    raw_low_floor = torch.zeros_like(advantage, dtype=torch.bool)
                    normalized_high_clip = torch.zeros_like(
                        advantage, dtype=torch.bool
                    )
                bc_logits = self.bc_reference(batch["depth"], batch["vector"])
                bc_probabilities, bc_log_probabilities, _ = masked_policy(
                    bc_logits, batch["action_mask"], torch
                )
                confidence_result = None
                if bool(config.enable_twin_q_confidence) or bool(
                    config.enable_adaptive_bc_kl
                ):
                    confidence_result = self.confidence_estimator.estimate(
                        self.critic1(batch["depth"], batch["vector"]),
                        self.critic2(batch["depth"], batch["vector"]),
                        batch["action_mask"],
                        bc_policy=bc_probabilities.detach(),
                        rl_policy=probabilities.detach(),
                    )

            awac_actor_loss = -(normalized_weight * data_log_probability).mean()
            bc_kl_per_row = (
                bc_probabilities * (bc_log_probabilities - log_probabilities)
            ).sum(dim=1)
            bc_kl = bc_kl_per_row.mean()
            bc_kl_max = bc_kl_per_row.max()
            trust_tail_mean = bc_kl.new_zeros(())
            trust_tail_max = bc_kl.new_zeros(())
            trust_tail_selected_count = 0
            batch_kl_before = float(bc_kl.detach().cpu().item())
            batch_kl_max_before = float(bc_kl_max.detach().cpu().item())
            trust_kl_before = bc_kl.detach()
            trust_kl_max_before = bc_kl_max.detach()
            if actor_trust_batch is not None:
                trust_tail = self._trust_tail_bc_kl(actor_trust_batch)
                trust_kl_before = trust_tail["observed_mean"]
                trust_kl_max_before = trust_tail["observed_max"]
                trust_tail_mean = trust_tail["tail_mean"]
                trust_tail_max = trust_tail["tail_max"]
                trust_tail_selected_count = int(trust_tail["selected_count"])
            trust_tail_penalty = trust_tail_mean + trust_tail_max
            if bool(config.enable_adaptive_bc_kl):
                if confidence_result is None:
                    raise RuntimeError("adaptive BC KL requires a confidence result")
                adaptive_bc_kl, beta_per_row = adaptive_bc_kl_loss(
                    bc_kl_per_row,
                    confidence_result["confidence"],
                    config=self.adaptive_bc_kl_config,
                    torch=torch,
                    enabled=True,
                )
                beta_mean = beta_per_row.mean()
                adaptive_trust_tail_penalty = beta_mean * trust_tail_penalty
                actor_loss = (
                    awac_actor_loss + adaptive_bc_kl + adaptive_trust_tail_penalty
                )
            else:
                beta_per_row = torch.full_like(
                    bc_kl_per_row, float(config.bc_kl_weight)
                )
                adaptive_bc_kl = float(config.bc_kl_weight) * bc_kl
                adaptive_trust_tail_penalty = float(config.bc_kl_weight) * (
                    trust_tail_penalty
                )
                beta_mean = beta_per_row.mean()
                actor_loss = awac_actor_loss + float(config.bc_kl_weight) * (
                    bc_kl + trust_tail_penalty
                )
            trust_kl_before_value = float(trust_kl_before.detach().cpu().item())
            trust_kl_max_before_value = float(
                trust_kl_max_before.detach().cpu().item()
            )
            kl_before = max(batch_kl_before, trust_kl_before_value)
            kl_max_before = max(batch_kl_max_before, trust_kl_max_before_value)
            recovery_active = kl_budget_exceeded(
                mean_kl=kl_before,
                max_kl=kl_max_before,
                budget=float(config.bc_kl_hard_budget),
            )
            actor_optimization_loss = (
                float(config.bc_kl_recovery_weight)
                * (bc_kl + trust_tail_penalty)
                if recovery_active
                else actor_loss
            )
            proposal = self._apply_actor_proposal(
                batch=batch,
                actor_trust_batch=actor_trust_batch,
                actor_optimization_loss=actor_optimization_loss,
                recovery_active=recovery_active,
                kl_before=kl_before,
                kl_max_before=kl_max_before,
                bc_kl=bc_kl,
                bc_kl_max=bc_kl_max,
                update_actor=bool(update_actor),
            )
        finally:
            for module, was_training in zip(frozen_modules, training_modes):
                module.train(was_training)

        raw_ess = _effective_sample_size(raw_weight, torch)
        normalized_ess = _effective_sample_size(normalized_weight, torch)
        quantile_levels = torch.tensor(
            [0.05, 0.50, 0.95, 0.99], device=advantage.device
        )
        advantage_quantiles = torch.quantile(
            advantage.detach(),
            quantile_levels,
        )
        raw_weight_quantiles = torch.quantile(
            raw_weight.detach(),
            torch.tensor([0.05, 0.50, 0.95, 0.99], device=raw_weight.device),
        )
        weight_quantiles = torch.quantile(
            normalized_weight.detach(),
            torch.tensor(
                [0.05, 0.50, 0.95, 0.99], device=normalized_weight.device
            ),
        )
        actor_entropy = -(
            probabilities.detach() * log_probabilities.detach()
        ).sum(dim=1)
        top1_flip_rate = float(
            (
                probabilities.detach().argmax(dim=1)
                != bc_probabilities.detach().argmax(dim=1)
            )
            .float()
            .mean()
            .cpu()
            .item()
        )
        valid_action_count = batch["action_mask"].bool().sum(dim=1)
        multiple_valid_actions = valid_action_count >= 2
        masked_bc_probabilities = bc_probabilities.detach().masked_fill(
            ~batch["action_mask"].bool(), float("-inf")
        )
        top_two_probabilities = torch.topk(
            masked_bc_probabilities, k=2, dim=1
        ).values
        top1_top2_margin = (
            top_two_probabilities[:, 0] - top_two_probabilities[:, 1]
        )[multiple_valid_actions]
        top1_top2_margin_count = int(multiple_valid_actions.sum().cpu().item())
        if top1_top2_margin_count > 0:
            top1_top2_margin_quantiles = torch.quantile(
                top1_top2_margin,
                torch.tensor(
                    [0.05, 0.50, 0.95, 0.99],
                    device=top1_top2_margin.device,
                ),
            )
            top1_top2_margin_mean = float(
                top1_top2_margin.mean().detach().cpu().item()
            )
        else:
            top1_top2_margin_quantiles = torch.zeros(
                4, device=bc_probabilities.device
            )
            top1_top2_margin_mean = 0.0
        metrics: Dict[str, float] = {
            "actor_only_update": float(bool(update_actor)),
            "actor_weight_uniform": float(mode == "uniform"),
            "actor_weight_awac": float(mode == "awac"),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "actor_optimization_loss": float(
                actor_optimization_loss.detach().cpu().item()
            ),
            "awac_actor_loss": float(awac_actor_loss.detach().cpu().item()),
            "awac_advantage_mean": float(advantage.mean().detach().cpu().item()),
            "awac_advantage_p05": float(advantage_quantiles[0].cpu().item()),
            "awac_advantage_p50": float(advantage_quantiles[1].cpu().item()),
            "awac_advantage_p95": float(advantage_quantiles[2].cpu().item()),
            "awac_advantage_p99": float(advantage_quantiles[3].cpu().item()),
            "awac_raw_weight_mean": float(raw_weight.mean().detach().cpu().item()),
            "awac_raw_weight_p05": float(raw_weight_quantiles[0].cpu().item()),
            "awac_raw_weight_p50": float(raw_weight_quantiles[1].cpu().item()),
            "awac_raw_weight_p95": float(raw_weight_quantiles[2].cpu().item()),
            "awac_raw_weight_p99": float(raw_weight_quantiles[3].cpu().item()),
            "awac_raw_weight_high_clip_rate": float(
                raw_high_clip.float().mean().detach().cpu().item()
            ),
            "awac_raw_weight_low_floor_rate": float(
                raw_low_floor.float().mean().detach().cpu().item()
            ),
            "awac_weight_mean": float(
                normalized_weight.mean().detach().cpu().item()
            ),
            "awac_weight_p05": float(weight_quantiles[0].cpu().item()),
            "awac_weight_p50": float(weight_quantiles[1].cpu().item()),
            "awac_weight_p95": float(weight_quantiles[2].cpu().item()),
            "awac_weight_p99": float(weight_quantiles[3].cpu().item()),
            "awac_weight_high_clip_rate": float(
                normalized_high_clip.float().mean().detach().cpu().item()
            ),
            "awac_raw_weight_ess": float(raw_ess.detach().cpu().item()),
            "awac_raw_weight_ess_fraction": float(
                (raw_ess / float(batch_size)).detach().cpu().item()
            ),
            "awac_weight_ess": float(normalized_ess.detach().cpu().item()),
            "awac_weight_ess_fraction": float(
                (normalized_ess / float(batch_size)).detach().cpu().item()
            ),
            "bc_kl": float(proposal["kl_after"].detach().cpu().item()),
            "bc_kl_before_update": float(kl_before),
            "bc_kl_max_before_update": float(kl_max_before),
            "bc_kl_batch_before_update": float(batch_kl_before),
            "bc_kl_batch_max_before_update": float(batch_kl_max_before),
            "bc_kl_trust_before_update": float(trust_kl_before_value),
            "bc_kl_trust_max_before_update": float(trust_kl_max_before_value),
            "bc_kl_max_after_update": float(
                proposal["kl_max_after"].detach().cpu().item()
            ),
            "bc_kl_budget_exceeded": float(recovery_active),
            "bc_kl_recovery_mean_improved": float(
                proposal["recovery_mean_improved"]
            ),
            "bc_kl_recovery_max_improved": float(
                proposal["recovery_max_improved"]
            ),
            "adaptive_bc_kl_beta_mean": float(beta_mean.detach().cpu().item()),
            "trust_tail_selected_count": float(trust_tail_selected_count),
            "trust_tail_bc_kl_mean_before_update": float(
                trust_tail_mean.detach().cpu().item()
            ),
            "trust_tail_bc_kl_max_before_update": float(
                trust_tail_max.detach().cpu().item()
            ),
            "trust_tail_penalty": float(trust_tail_penalty.detach().cpu().item()),
            "actor_gradient_norm": float(
                proposal["actor_gradient_norm"].detach().cpu().item()
            ),
            "actor_parameter_delta_norm": float(
                proposal["actor_parameter_delta_norm"]
            ),
            "actor_entropy": float(actor_entropy.mean().detach().cpu().item()),
            "bc_awac_top1_disagreement_rate": top1_flip_rate,
            "bc_awac_top1_disagreement_count": float(
                (
                    probabilities.detach().argmax(dim=1)
                    != bc_probabilities.detach().argmax(dim=1)
                )
                .sum()
                .cpu()
                .item()
            ),
            "bc_awac_top1_disagreement_denominator": float(batch_size),
            "bc_top1_top2_margin_valid_count": float(top1_top2_margin_count),
            "bc_top1_top2_margin_na_count": float(
                batch_size - top1_top2_margin_count
            ),
            "bc_top1_top2_margin_mean": top1_top2_margin_mean,
            "bc_top1_top2_margin_p05": float(
                top1_top2_margin_quantiles[0].cpu().item()
            ),
            "bc_top1_top2_margin_p50": float(
                top1_top2_margin_quantiles[1].cpu().item()
            ),
            "bc_top1_top2_margin_p95": float(
                top1_top2_margin_quantiles[2].cpu().item()
            ),
            "bc_top1_top2_margin_p99": float(
                top1_top2_margin_quantiles[3].cpu().item()
            ),
            "actor_optimizer_step": float(proposal["actor_optimizer_step"]),
            "actor_awac_optimizer_step": float(
                proposal["actor_awac_optimizer_step"]
            ),
            "actor_recovery_optimizer_step": float(
                proposal["actor_recovery_optimizer_step"]
            ),
            "actor_trust_region_rejection": float(
                proposal["actor_trust_region_rejection"]
            ),
            "actor_update_count": float(self.actor_update_count),
            "actor_awac_update_count": float(self.actor_awac_update_count),
            "actor_recovery_update_count": float(self.actor_recovery_update_count),
            "actor_trust_region_rejection_count": float(
                self.actor_trust_region_rejection_count
            ),
            "actor_optimizer_step_count": float(self.actor_optimizer_step_count),
            "critic_update_count": float(self.critic_update_count),
            "fixed_critic_update_step": float(self.update_step),
            "fixed_critic1_updated": 0.0,
            "fixed_critic2_updated": 0.0,
            "fixed_target_critic1_updated": 0.0,
            "fixed_target_critic2_updated": 0.0,
            "fixed_bc_reference_updated": 0.0,
        }
        for group_name, value in proposal["actor_group_gradient_norms"].items():
            metrics["actor_{}_gradient_norm".format(group_name)] = float(value)
        for group_name, value in proposal[
            "actor_group_parameter_delta_norms"
        ].items():
            metrics["actor_{}_parameter_delta_norm".format(group_name)] = float(value)
        metrics.update(
            self._stratum_metrics(
                advantage=advantage,
                raw_weight=raw_weight,
                weight=normalized_weight,
                done=batch["done"],
                reward=batch["reward"],
                behavior=batch["behavior_source"],
            )
        )
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise FloatingPointError("Actor-only AWAC metrics contain non-finite values")
        return metrics

    def state_dict(self) -> Dict:
        return {
            "awac_learner_state_schema_id": AWAC_LEARNER_STATE_SCHEMA_ID,
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "target_critic1_state_dict": self.target_critic1.state_dict(),
            "target_critic2_state_dict": self.target_critic2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "update_step": int(self.update_step),
            "actor_update_count": int(self.actor_update_count),
            "actor_awac_update_count": int(self.actor_awac_update_count),
            "actor_recovery_update_count": int(self.actor_recovery_update_count),
            "actor_trust_region_rejection_count": int(
                self.actor_trust_region_rejection_count
            ),
            "actor_optimizer_step_count": int(self.actor_optimizer_step_count),
            "critic_update_count": int(self.critic_update_count),
            "actor_frozen_for_calibration": bool(
                self.actor_frozen_for_calibration
            ),
            "optimizer_lr_provenance": copy.deepcopy(self.optimizer_lr_provenance),
        }

    def optimizer_lr_inventory(self) -> Dict[str, float]:
        """Return actual optimizer learning rates keyed by stable group names."""

        result = {}
        for optimizer_name, optimizer in (
            ("actor", self.actor_optimizer),
            ("critic", self.critic_optimizer),
        ):
            names = []
            for index, group in enumerate(optimizer.param_groups):
                name = str(group.get("name", ""))
                if not name:
                    raise ValueError(
                        "{} optimizer group {} has no stable name".format(
                            optimizer_name, index
                        )
                    )
                if name in names:
                    raise ValueError(
                        "{} optimizer has duplicate group name {}".format(
                            optimizer_name, name
                        )
                    )
                names.append(name)
                value = float(group.get("lr", float("nan")))
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        "{} optimizer group {} has invalid learning rate".format(
                            optimizer_name, name
                        )
                    )
                result["{}_{}".format(optimizer_name, name)] = value
        return result

    def expected_optimizer_lr_inventory(self) -> Dict[str, float]:
        """Return config-derived rates for the groups that actually exist."""

        expected = {
            "actor_head": float(self.config.actor_head_lr),
            "actor_vector_encoder": float(self.config.actor_vector_lr),
            "actor_depth_encoder": float(self.config.actor_depth_lr),
            "critic_critic1_head": float(self.config.critic_head_lr),
            "critic_critic1_vector_encoder": float(self.config.critic_vector_lr),
            "critic_critic1_depth_encoder": float(self.config.critic_depth_lr),
            "critic_critic2_head": float(self.config.critic_head_lr),
            "critic_critic2_vector_encoder": float(self.config.critic_vector_lr),
            "critic_critic2_depth_encoder": float(self.config.critic_depth_lr),
        }
        actual = self.optimizer_lr_inventory()
        return {name: expected[name] for name in actual if name in expected}

    def assert_optimizer_lrs_match_config(self, *, context: str = "AWAC") -> None:
        """Fail closed before an update if config and optimizer state diverge."""

        actual = self.optimizer_lr_inventory()
        expected = self.expected_optimizer_lr_inventory()
        if set(actual) != set(expected):
            raise ValueError(
                "{} optimizer group inventory mismatch: actual={} expected={}".format(
                    context, sorted(actual), sorted(expected)
                )
            )
        mismatches = {
            name: {"actual": actual[name], "expected": expected[name]}
            for name in actual
            if not math.isclose(actual[name], expected[name], rel_tol=0.0, abs_tol=1.0e-15)
        }
        if mismatches:
            raise ValueError(
                "{} optimizer learning-rate mismatch: {}".format(
                    context, mismatches
                )
            )

    def apply_optimizer_lr_overrides(
        self,
        overrides: Mapping[str, Any],
        *,
        parent_checkpoint_sha256: str,
        allowed_prefix: str = "critic_",
    ) -> Dict[str, float]:
        """Apply an explicit named handoff override after state restoration.

        Adam moments and per-parameter steps remain untouched.  Only the
        selected group's ``lr`` fields are changed, and unknown/duplicate or
        invalid names fail closed.
        """

        if not isinstance(overrides, Mapping) or not overrides:
            raise ValueError("optimizer LR override must be a non-empty mapping")
        if len(set(str(name) for name in overrides)) != len(overrides):
            raise ValueError("optimizer LR override contains duplicate names")
        actual_names = set(self.optimizer_lr_inventory())
        names = {str(name) for name in overrides}
        unknown = names.difference(actual_names)
        if unknown:
            raise ValueError("optimizer LR override names are unknown: {}".format(sorted(unknown)))
        illegal = {name for name in names if not name.startswith(str(allowed_prefix))}
        if illegal:
            raise ValueError("optimizer LR override may not change: {}".format(sorted(illegal)))
        parent = str(parent_checkpoint_sha256)
        if len(parent) != 64 or any(char not in "0123456789abcdef" for char in parent.lower()):
            raise ValueError("optimizer LR override parent checkpoint SHA is invalid")
        normalized = {}
        for name, value in overrides.items():
            rate = float(value)
            if not math.isfinite(rate) or rate <= 0.0:
                raise ValueError("optimizer LR override {} is invalid".format(name))
            normalized[str(name)] = rate
        for optimizer in (self.actor_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                key = (
                    "actor_{}".format(group["name"])
                    if optimizer is self.actor_optimizer
                    else "critic_{}".format(group["name"])
                )
                if key in normalized:
                    group["lr"] = normalized[key]
        self.optimizer_lr_provenance = {
            "mode": "explicit_named_handoff_override",
            "parent_checkpoint_sha256": parent,
            "overrides": dict(sorted(normalized.items())),
            "actual_group_lrs": self.optimizer_lr_inventory(),
        }
        return dict(self.optimizer_lr_provenance["actual_group_lrs"])

    def load_state_dict(self, payload: Dict) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("AWAC learner state must be a mapping")
        if payload.get("awac_learner_state_schema_id") != AWAC_LEARNER_STATE_SCHEMA_ID:
            raise ValueError("AWAC learner state schema is missing or incompatible")
        required = {
            "actor_state_dict",
            "critic1_state_dict",
            "critic2_state_dict",
            "target_critic1_state_dict",
            "target_critic2_state_dict",
            "actor_optimizer_state_dict",
            "critic_optimizer_state_dict",
            "update_step",
            "actor_update_count",
            "actor_awac_update_count",
            "actor_recovery_update_count",
            "actor_trust_region_rejection_count",
            "actor_optimizer_step_count",
            "critic_update_count",
            "actor_frozen_for_calibration",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError("AWAC learner state missing fields: {}".format(", ".join(missing)))
        counters = {
            name: int(payload[name])
            for name in (
                "update_step",
                "actor_update_count",
                "actor_awac_update_count",
                "actor_recovery_update_count",
                "actor_trust_region_rejection_count",
                "actor_optimizer_step_count",
                "critic_update_count",
            )
        }
        if any(value < 0 for value in counters.values()):
            raise ValueError("AWAC learner counters must be non-negative")
        if counters["actor_update_count"] != (
            counters["actor_awac_update_count"]
            + counters["actor_recovery_update_count"]
        ):
            raise ValueError("AWAC accepted Actor counters are inconsistent")
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.critic1.load_state_dict(payload["critic1_state_dict"])
        self.critic2.load_state_dict(payload["critic2_state_dict"])
        self.target_critic1.load_state_dict(payload["target_critic1_state_dict"])
        self.target_critic2.load_state_dict(payload["target_critic2_state_dict"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        self.update_step = counters["update_step"]
        self.actor_update_count = counters["actor_update_count"]
        self.actor_awac_update_count = counters["actor_awac_update_count"]
        self.actor_recovery_update_count = counters["actor_recovery_update_count"]
        self.actor_trust_region_rejection_count = counters[
            "actor_trust_region_rejection_count"
        ]
        self.actor_optimizer_step_count = counters["actor_optimizer_step_count"]
        self.critic_update_count = counters["critic_update_count"]
        self.actor_frozen_for_calibration = bool(
            payload["actor_frozen_for_calibration"]
        )
        provenance = payload.get("optimizer_lr_provenance")
        self.optimizer_lr_provenance = (
            copy.deepcopy(provenance)
            if isinstance(provenance, Mapping)
            else {
                "mode": "checkpoint_state_restored",
                "parent_checkpoint_sha256": "",
                "overrides": {},
                "actual_group_lrs": self.optimizer_lr_inventory(),
            }
        )
        if self.actor_frozen_for_calibration:
            self.freeze_actor_for_calibration()
