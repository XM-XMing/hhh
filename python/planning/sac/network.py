"""Masked discrete SAC networks and update mathematics."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Tuple

import numpy as np

from planning.bc.model import build_model
from planning.contracts.feature import NUM_ACTIONS, POLICY_VECTOR_DIM
from planning.sac.contract import (
    SAC_BC_REFERENCE_ID,
    SAC_POLICY_ID,
    SAC_RESIDUAL_LOGIT_CAP,
    SAC_RESIDUAL_POLICY_ID,
)
from planning.sac.diagnostics import optimizer_state_sha256, stable_fingerprint


def state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash a torch state dict without depending on pickle serialization."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().contiguous().numpy()
        array = np.ascontiguousarray(value)
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _deterministic_nn_proxy(nn, torch):
    """Build an nn namespace with a deterministic adaptive-pool module."""

    class DeterministicAdaptiveAvgPool2d(nn.Module):
        """Adaptive average pooling expressed without CUDA adaptive-pool backward."""

        def __init__(self, output_size):
            super().__init__()
            self.output_size = tuple(int(value) for value in output_size)

        def forward(self, value):
            height, width = int(value.shape[-2]), int(value.shape[-1])
            out_h, out_w = self.output_size
            rows = []
            for row in range(out_h):
                h0 = (row * height) // out_h
                h1 = ((row + 1) * height + out_h - 1) // out_h
                columns = []
                for column in range(out_w):
                    w0 = (column * width) // out_w
                    w1 = ((column + 1) * width + out_w - 1) // out_w
                    columns.append(value[..., h0:h1, w0:w1].mean(dim=(-2, -1)))
                rows.append(torch.stack(columns, dim=-1))
            return torch.stack(rows, dim=-2)

    class Proxy:
        AdaptiveAvgPool2d = DeterministicAdaptiveAvgPool2d

        def __getattr__(self, name):
            return getattr(nn, name)

    return Proxy()


def build_policy(
    nn,
    *,
    depth_channels: int = 1,
    vec_dim: int = POLICY_VECTOR_DIM,
    deterministic_pool: bool = False,
    torch=None,
):
    """Build the exact BC architecture used by deployment evaluation."""

    model_nn = _deterministic_nn_proxy(nn, torch) if deterministic_pool else nn
    return build_model(
        model_nn,
        depth_channels=int(depth_channels),
        vec_dim=int(vec_dim),
        num_actions=NUM_ACTIONS,
    )


def build_q_network(
    nn,
    *,
    depth_channels: int = 1,
    vec_dim: int = POLICY_VECTOR_DIM,
    deterministic_pool: bool = False,
    torch=None,
):
    """Build one fresh 105-action Q network.

    The architecture is intentionally independent of the Actor instance.  A
    caller may use the same encoder shape, but no Actor or historical Critic
    parameters are copied into this network.
    """

    model_nn = _deterministic_nn_proxy(nn, torch) if deterministic_pool else nn
    return build_model(
        model_nn,
        depth_channels=int(depth_channels),
        vec_dim=int(vec_dim),
        num_actions=NUM_ACTIONS,
    )


def _validate_mask(action_mask, torch):
    mask = action_mask.bool()
    if mask.ndim != 2 or int(mask.shape[1]) != NUM_ACTIONS:
        raise ValueError("action_mask must have shape [B,105]")
    if bool((~mask.any(dim=1)).any()):
        raise ValueError("masked categorical received an empty action mask")
    return mask


def masked_categorical(logits, action_mask, torch) -> Tuple[Any, Any, Any]:
    """Return probabilities, finite masked log-probabilities, and mask.

    Invalid actions have exactly zero probability.  Their log-probability is
    represented as zero in the returned tensor so ``0 * log_prob`` never
    creates a NaN in the discrete expectation.
    """

    mask = _validate_mask(action_mask, torch)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    log_probs = torch.log_softmax(masked_logits, dim=1)
    probabilities = torch.where(mask, torch.exp(log_probs), torch.zeros_like(log_probs))
    normalizer = probabilities.sum(dim=1, keepdim=True)
    probabilities = probabilities / normalizer.clamp_min(torch.finfo(logits.dtype).eps)
    finite_log_probs = torch.where(
        mask,
        torch.log(probabilities.clamp_min(torch.finfo(logits.dtype).eps)),
        torch.zeros_like(probabilities),
    )
    return probabilities, finite_log_probs, mask


def discrete_value_from_policy(q_values, probabilities, log_probs, alpha, torch):
    """Compute ``sum pi * (Q - alpha log pi)`` over valid actions."""

    value = probabilities * (q_values - float(alpha) * log_probs)
    return value.sum(dim=1)


def masked_kl(probabilities, log_probs, reference_probabilities, reference_log_probs, torch):
    """Compute KL(pi || pi_BC) only over actions admitted by the mask."""

    del torch
    return (probabilities * (log_probs - reference_log_probs)).sum(dim=1)


def _polyak_update(target, source, tau: float, torch) -> None:
    value = float(tau)
    if not 0.0 < value <= 1.0 or not math.isfinite(value):
        raise ValueError("tau must be finite and in (0,1]")
    with torch.no_grad():
        target_state = target.state_dict()
        source_state = source.state_dict()
        for name in target_state:
            target_value = target_state[name]
            source_value = source_state[name]
            if target_value.is_floating_point():
                target_value.mul_(1.0 - value).add_(source_value, alpha=value)
            else:
                target_value.copy_(source_value)
        target.load_state_dict(target_state)


class ResidualPolicy:
    """Frozen BC logits plus a bounded trainable residual logit module.

    The wrapper is intentionally a small ``nn.Module``-compatible object built
    with the caller's ``nn`` namespace.  The BC branch is detached and frozen;
    only the copied residual branch participates in autograd.  Bounding the
    residual with ``cap * tanh`` gives the V5 analytic per-state KL bound
    ``KL(pi_residual || pi_BC) <= 2 * cap`` after the common action mask.
    """

    def __init__(self, nn, bc_base, residual_model, *, cap: float, torch) -> None:
        super().__init__()
        self._module = nn.Module()
        self._module.add_module("bc_base", bc_base)
        self._module.add_module("residual_model", residual_model)
        self.logit_cap = float(cap)
        if not math.isfinite(self.logit_cap) or self.logit_cap <= 0.0:
            raise ValueError("residual logit cap must be finite and positive")
        self.torch = torch
        for parameter in self.bc_base.parameters():
            parameter.requires_grad_(False)
        self.bc_base.eval()

    @property
    def bc_base(self):
        return self._module.bc_base

    @property
    def residual_model(self):
        return self._module.residual_model

    def parameters(self, recurse: bool = True):
        return self._module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self._module.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self._module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self._module.load_state_dict(*args, **kwargs)

    def to(self, *args, **kwargs):
        self._module.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self._module.train(mode)
        self.bc_base.eval()
        return self

    def eval(self):
        self._module.eval()
        return self

    def residual_raw(self, depth, vector):
        return self.residual_model(depth, vector)

    def residual_logits(self, depth, vector):
        return self.logit_cap * self.torch.tanh(self.residual_raw(depth, vector))

    def forward(self, depth, vector):
        with self.torch.no_grad():
            base = self.bc_base(depth, vector)
        return base.detach() + self.residual_logits(depth, vector)

    def __call__(self, depth, vector):
        return self.forward(depth, vector)

    def trainable_parameters(self):
        return [parameter for parameter in self.residual_model.parameters() if parameter.requires_grad]


class DiscreteSACAgent:
    """BC-initialized Actor plus fresh twin Critics and fixed-alpha SAC."""

    def __init__(
        self,
        *,
        torch,
        nn,
        device,
        bc_checkpoint: Mapping[str, Any],
        actor_lr: float = 1.0e-5,
        critic_lr: float = 1.0e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.20,
        beta_bc: float = 0.05,
        reward_scale: float = 0.10,
        gradient_clip_norm: float = 5.0,
        entropy_floor: float = 0.02,
        deterministic_pool: bool = False,
        actor_architecture: str = "direct",
        residual_logit_cap: float = SAC_RESIDUAL_LOGIT_CAP,
    ) -> None:
        self.torch = torch
        self.nn = nn
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.beta_bc = float(beta_bc)
        self.reward_scale = float(reward_scale)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.entropy_floor = float(entropy_floor)
        for name, value in (
            ("gamma", self.gamma),
            ("tau", self.tau),
            ("alpha", self.alpha),
            ("beta_bc", self.beta_bc),
            ("reward_scale", self.reward_scale),
            ("gradient_clip_norm", self.gradient_clip_norm),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be finite and positive".format(name))
        if not math.isfinite(self.entropy_floor) or self.entropy_floor < 0.0:
            raise ValueError("entropy_floor must be finite and non-negative")
        depth_channels = int(bc_checkpoint.get("depth_history_frames", 1))
        self.deterministic_pool = bool(deterministic_pool)
        self.actor_architecture = str(actor_architecture)
        if self.actor_architecture not in {"direct", "residual"}:
            raise ValueError("actor_architecture must be direct or residual")
        self.residual_logit_cap = float(residual_logit_cap)
        if self.actor_architecture == "residual" and self.residual_logit_cap != float(SAC_RESIDUAL_LOGIT_CAP):
            raise ValueError("V5 residual_logit_cap is fixed at {}".format(SAC_RESIDUAL_LOGIT_CAP))
        self.actor = build_policy(
            nn, depth_channels=depth_channels, deterministic_pool=self.deterministic_pool, torch=torch
        ).to(device)
        self.bc_reference = build_policy(
            nn, depth_channels=depth_channels, deterministic_pool=self.deterministic_pool, torch=torch
        ).to(device)
        state = bc_checkpoint.get("model_state_dict")
        if not isinstance(state, Mapping) or not state:
            raise ValueError("BC checkpoint model_state_dict is missing")
        self.actor.load_state_dict(state, strict=True)
        self.bc_reference.load_state_dict(state, strict=True)
        self.bc_reference.eval()
        for parameter in self.bc_reference.parameters():
            parameter.requires_grad_(False)

        if self.actor_architecture == "residual":
            residual_model = build_policy(
                nn,
                depth_channels=depth_channels,
                deterministic_pool=self.deterministic_pool,
                torch=torch,
            ).to(device)
            residual_model.load_state_dict(state, strict=True)
            final_head = residual_model.head[-1]
            with torch.no_grad():
                final_head.weight.zero_()
                final_head.bias.zero_()
            bc_base = self.actor
            self.actor = ResidualPolicy(
                nn,
                bc_base,
                residual_model,
                cap=self.residual_logit_cap,
                torch=torch,
            ).to(device)

        # These are intentionally fresh.  No historical AWAC/N1/N5/ranking
        # checkpoint is accepted by this constructor.
        self.critic1 = build_q_network(
            nn, depth_channels=depth_channels, deterministic_pool=self.deterministic_pool, torch=torch
        ).to(device)
        self.critic2 = build_q_network(
            nn, depth_channels=depth_channels, deterministic_pool=self.deterministic_pool, torch=torch
        ).to(device)
        self.target_critic1 = deepcopy(self.critic1).to(device)
        self.target_critic2 = deepcopy(self.critic2).to(device)
        self.target_critic1.eval()
        self.target_critic2.eval()

        self.actor_optimizer = torch.optim.Adam(
            self.actor_trainable_parameters(), lr=float(actor_lr)
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=float(critic_lr),
        )
        self.actor_optimizer_step_count = 0
        self.critic_optimizer_step_count = 0
        self.actor_update_count = 0
        self.critic_update_count = 0
        self.actor_proposal_count = 0
        self.actor_rejection_count = 0
        self.actor_recovery_update_count = 0
        self.nan_count = 0
        self.invalid_action_count = 0
        self._kl_sentinel = None
        self._kl_sentinel_manifest = None
        self._actor_rng_capture = None
        self._actor_rng_restore = None

    @property
    def actor_policy_id(self) -> str:
        return SAC_RESIDUAL_POLICY_ID if self.actor_architecture == "residual" else SAC_POLICY_ID

    @property
    def bc_reference_id(self) -> str:
        return SAC_BC_REFERENCE_ID

    def actor_fingerprint(self) -> str:
        return state_dict_sha256(self.actor.state_dict())

    def bc_reference_fingerprint(self) -> str:
        return state_dict_sha256(self.bc_reference.state_dict())

    def actor_trainable_parameters(self):
        """Return only parameters allowed to receive Actor gradients."""

        return [parameter for parameter in self.actor.parameters() if parameter.requires_grad]

    def residual_statistics(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return finite residual magnitude diagnostics for a batch."""

        if self.actor_architecture != "residual":
            return {
                "architecture": "direct",
            "residual_raw_mean": 0.0,
            "residual_logit_mean": 0.0,
            "residual_logit_std": 0.0,
            "residual_abs_mean": 0.0,
                "residual_abs_p90": 0.0,
                "residual_abs_p99": 0.0,
                "residual_abs_max": 0.0,
                "residual_saturation_fraction": 0.0,
            }
        with self.torch.no_grad():
            raw = self.actor.residual_raw(batch["depth"], batch["vector"])
            residual = self.actor.residual_logits(batch["depth"], batch["vector"])
            absolute = residual.abs().reshape(-1)
            saturated = absolute >= (self.residual_logit_cap * 0.99)
        values = {
            "architecture": "residual",
            "residual_raw_mean": float(raw.mean().item()),
            "residual_logit_mean": float(residual.mean().item()),
            "residual_logit_std": float(residual.std(unbiased=False).item()),
            "residual_abs_mean": float(absolute.mean().item()),
            "residual_abs_p90": float(self.torch.quantile(absolute, 0.90).item()),
            "residual_abs_p99": float(self.torch.quantile(absolute, 0.99).item()),
            "residual_abs_max": float(absolute.max().item()),
            "residual_saturation_fraction": float(saturated.float().mean().item()),
        }
        if not all(math.isfinite(float(value)) for key, value in values.items() if key != "architecture"):
            self.nan_count += 1
            raise FloatingPointError("non-finite residual statistics")
        return values

    def set_actor_rng_hooks(self, capture, restore) -> None:
        """Install the runtime-owned RNG hooks used by Actor transactions."""

        if not callable(capture) or not callable(restore):
            raise TypeError("Actor RNG hooks must be callable")
        self._actor_rng_capture = capture
        self._actor_rng_restore = restore

    def set_kl_sentinel(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Freeze a copied pre-Actor state set for the trust-region gate."""

        required = ("depth", "vector", "action_mask")
        missing = [name for name in required if name not in batch]
        if missing:
            raise ValueError("KL sentinel batch is missing: {}".format(", ".join(missing)))
        copied = {}
        for name in required:
            value = batch[name]
            if hasattr(value, "detach"):
                copied[name] = value.detach().clone().to(device=self.device)
            else:
                copied[name] = deepcopy(value)
        copied["action_mask"] = copied["action_mask"].bool()
        if copied["depth"].ndim < 2 or copied["vector"].ndim != 2:
            raise ValueError("KL sentinel observation shapes are invalid")
        if int(copied["vector"].shape[0]) != int(copied["action_mask"].shape[0]):
            raise ValueError("KL sentinel rows do not align")
        if bool((~copied["action_mask"].any(dim=1)).any()):
            raise ValueError("KL sentinel contains an empty action mask")
        self._kl_sentinel = copied
        manifest = {
            "schema_id": "bc_initialized_discrete_sac_kl_sentinel_v4",
            "state_count": int(copied["vector"].shape[0]),
            "selection": "unique_pre_actor_committed_observation_states",
            "return_or_q_selection": False,
            "fixed_after_first_actor": True,
            "state_fingerprint": stable_fingerprint(
                {name: copied[name] for name in ("depth", "vector", "action_mask")}
            ),
        }
        self._kl_sentinel_manifest = manifest
        return dict(manifest)

    def kl_sentinel_manifest(self) -> Mapping[str, Any]:
        if self._kl_sentinel_manifest is None:
            return {"state_count": 0, "fixed_after_first_actor": False}
        return dict(self._kl_sentinel_manifest)

    def _policy_metrics(self, batch: Mapping[str, Any], *, entropy_floor=None) -> Mapping[str, Any]:
        torch = self.torch
        floor = self.entropy_floor if entropy_floor is None else float(entropy_floor)
        with torch.no_grad():
            probabilities, log_probs, mask = self._distribution(
                self.actor, batch["depth"], batch["vector"], batch["action_mask"].bool()
            )
            bc_prob, bc_log, _ = self._distribution(
                self.bc_reference, batch["depth"], batch["vector"], batch["action_mask"].bool()
            )
            kl = masked_kl(probabilities, log_probs, bc_prob, bc_log, torch)
            entropy = -(probabilities * log_probs).sum(dim=1)
            bc_entropy = -(bc_prob * bc_log).sum(dim=1)
            eligible = (mask.sum(dim=1) > 1) & (bc_entropy >= floor)
            flips = (
                torch.argmax(probabilities.masked_fill(~mask, -1.0), dim=1)
                != torch.argmax(bc_prob.masked_fill(~mask, -1.0), dim=1)
            ).float()
        return {
            "count": int(kl.numel()),
            "kl_mean": float(kl.mean().item()) if kl.numel() else 0.0,
            "kl_p99": float(torch.quantile(kl, 0.99).item()) if kl.numel() else 0.0,
            "kl_max": float(kl.max().item()) if kl.numel() else 0.0,
            "entropy_mean": float(entropy.mean().item()) if entropy.numel() else 0.0,
            "entropy_min": float(entropy[eligible].min().item()) if bool(eligible.any()) else 0.0,
            "entropy_floor_eligible_count": int(eligible.sum().item()),
            "argmax_flip_rate": float(flips.mean().item()) if flips.numel() else 0.0,
        }

    def kl_sentinel_stats(self, *, entropy_floor=None) -> Mapping[str, Any]:
        if self._kl_sentinel is None:
            return {"count": 0, "kl_mean": 0.0, "kl_p99": 0.0, "kl_max": 0.0}
        return dict(self._policy_metrics(self._kl_sentinel, entropy_floor=entropy_floor))

    @staticmethod
    def _moment_norms(optimizer, torch) -> Mapping[str, float]:
        exp_avg = []
        exp_avg_sq = []
        for state in optimizer.state.values():
            if "exp_avg" in state:
                exp_avg.append(state["exp_avg"].detach().reshape(-1))
            if "exp_avg_sq" in state:
                exp_avg_sq.append(state["exp_avg_sq"].detach().reshape(-1))
        return {
            "exp_avg_norm": float(torch.linalg.vector_norm(torch.cat(exp_avg)).item()) if exp_avg else 0.0,
            "exp_avg_sq_norm": float(torch.linalg.vector_norm(torch.cat(exp_avg_sq)).item()) if exp_avg_sq else 0.0,
        }

    @staticmethod
    def _cosine(left, right, torch):
        if left.numel() == 0 or right.numel() == 0:
            return None
        left_norm = torch.linalg.vector_norm(left)
        right_norm = torch.linalg.vector_norm(right)
        if float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0:
            return None
        return float((torch.dot(left, right) / (left_norm * right_norm)).item())

    def _restore_actor_transaction(self, actor_before, optimizer_before, rng_before):
        self.actor.load_state_dict(actor_before, strict=True)
        self.actor_optimizer.load_state_dict(optimizer_before)
        if self._actor_rng_restore is not None and rng_before is not None:
            self._actor_rng_restore(rng_before)

    def _update_actor_transactional_trust(
        self,
        batch: Mapping[str, Any],
        *,
        hard_stop: float,
        entropy_floor: float,
        trust_region_factors,
    ) -> Mapping[str, Any]:
        """Run one fixed-gradient, fixed-sentinel backtracking transaction."""

        torch = self.torch
        factors = tuple(float(value) for value in trust_region_factors)
        expected_factors = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
        if factors != expected_factors:
            raise ValueError("V4 trust-region factors are a fixed contract")
        parameters = self.actor_trainable_parameters()
        parameter_before = [parameter.detach().clone() for parameter in parameters]
        actor_before = deepcopy(self.actor.state_dict())
        optimizer_before = deepcopy(self.actor_optimizer.state_dict())
        actor_before_sha = state_dict_sha256(actor_before)
        optimizer_before_sha = optimizer_state_sha256(self.actor_optimizer)
        rng_before = self._actor_rng_capture() if self._actor_rng_capture is not None else None
        base_lrs = [float(group["lr"]) for group in self.actor_optimizer.param_groups]
        sentinel_pre = self.kl_sentinel_stats(entropy_floor=entropy_floor)
        self.actor_proposal_count += 1
        for module in (self.critic1, self.critic2):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        try:
            if float(sentinel_pre["kl_max"]) >= float(hard_stop):
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.actor_rejection_count += 1
                return {
                    "accepted": False,
                    "rejected": True,
                    "hard_stop_result": "REJECT_PRE_KL",
                    "rejection_reason": "pre_step_sentinel_kl_hard_stop",
                    "pre_sentinel_kl_mean": sentinel_pre["kl_mean"],
                    "pre_sentinel_kl_p99": sentinel_pre["kl_p99"],
                    "pre_sentinel_kl_max": sentinel_pre["kl_max"],
                    "sentinel_kl_max": sentinel_pre["kl_max"],
                    "accepted_lr_factor": None,
                    "backtrack_count": 0,
                    "raw_factor_1_kl_max": None,
                    "factor_results": [],
                    "optimizer_step_performed": False,
                    "rollback_performed": False,
                    "actor_before_sha256": actor_before_sha,
                    "actor_after_sha256": self.actor_fingerprint(),
                    "optimizer_before_sha256": optimizer_before_sha,
                    "optimizer_after_sha256": optimizer_state_sha256(self.actor_optimizer),
                    "critic_optimizer_step_count": int(self.critic_optimizer_step_count),
                    "trust_region_enabled": True,
                    "actor_proposal_id": int(self.actor_proposal_count),
                }

            depth = batch["depth"]
            vector = batch["vector"]
            action_mask = batch["action_mask"].bool()
            probabilities, log_probs, mask = self._distribution(
                self.actor, depth, vector, action_mask
            )
            with torch.no_grad():
                bc_prob, bc_log, _ = self._distribution(
                    self.bc_reference, depth, vector, action_mask
                )
                bc_entropy = -(bc_prob * bc_log).sum(dim=1)
            q_min = torch.minimum(self.critic1(depth, vector), self.critic2(depth, vector))
            q_term = (probabilities * (-q_min)).sum(dim=1)
            entropy_term = (probabilities * log_probs).sum(dim=1) * self.alpha
            kl = masked_kl(probabilities, log_probs, bc_prob, bc_log, torch)
            bc_kl_term = self.beta_bc * kl
            actor_loss = (q_term + entropy_term + bc_kl_term).mean()
            if not bool(torch.isfinite(actor_loss).item()):
                self.nan_count += 1
                raise FloatingPointError("non-finite SAC actor loss")
            component_norms = {}
            for name, component in (("q", q_term), ("entropy", entropy_term), ("bc_kl", bc_kl_term)):
                grads = torch.autograd.grad(component.mean(), parameters, retain_graph=True, allow_unused=True)
                component_norms[name] = self._gradient_norm(grads, torch)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_norm).item())
            if not math.isfinite(pre_clip_norm):
                self.nan_count += 1
                raise FloatingPointError("non-finite SAC actor gradient")
            gradients = [parameter.grad.detach().clone() if parameter.grad is not None else None for parameter in parameters]
            gradient_flat = torch.cat([value.reshape(-1) for value in gradients if value is not None])
            post_clip_norm = float(torch.linalg.vector_norm(gradient_flat).item()) if gradient_flat.numel() else 0.0
            gradient_sha = stable_fingerprint(gradients)
            pre_entropy = (-(probabilities * log_probs).sum(dim=1)).detach()
            pre_kl = kl.detach()
            factor_results = []
            raw_factor_1_kl_max = None
            accepted = None
            accepted_stats = None
            accepted_factor = None
            any_step = False
            for factor_index, factor in enumerate(factors):
                self._restore_actor_transaction(actor_before, optimizer_before, rng_before)
                for group, base_lr in zip(self.actor_optimizer.param_groups, base_lrs):
                    group["lr"] = float(base_lr) * factor
                self.actor_optimizer.zero_grad(set_to_none=True)
                for parameter, gradient in zip(parameters, gradients):
                    parameter.grad = gradient.detach().clone() if gradient is not None else None
                self.actor_optimizer.step()
                any_step = True
                stats = self.kl_sentinel_stats(entropy_floor=entropy_floor)
                if factor_index == 0:
                    raw_factor_1_kl_max = float(stats["kl_max"])
                entropy_violation = (
                    int(stats.get("entropy_floor_eligible_count", 0)) > 0
                    and float(stats.get("entropy_min", entropy_floor)) < float(entropy_floor)
                )
                safe = float(stats["kl_max"]) < float(hard_stop) and not entropy_violation
                factor_result = {
                    "factor": factor,
                    "post_sentinel_kl_mean": float(stats["kl_mean"]),
                    "post_sentinel_kl_p99": float(stats["kl_p99"]),
                    "post_sentinel_kl_max": float(stats["kl_max"]),
                    "entropy_min": float(stats.get("entropy_min", 0.0)),
                    "safe": bool(safe),
                    "entropy_violation": bool(entropy_violation),
                }
                factor_results.append(factor_result)
                if safe:
                    accepted = True
                    accepted_stats = stats
                    accepted_factor = factor
                    break
            if accepted is not True:
                self._restore_actor_transaction(actor_before, optimizer_before, rng_before)
                for group, base_lr in zip(self.actor_optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.actor_rejection_count += 1
                return {
                    "accepted": False,
                    "rejected": True,
                    "hard_stop_result": "REJECT_BACKTRACK_KL",
                    "rejection_reason": "all_trust_region_factors_failed",
                    "pre_sentinel_kl_mean": sentinel_pre["kl_mean"],
                    "pre_sentinel_kl_p99": sentinel_pre["kl_p99"],
                    "pre_sentinel_kl_max": sentinel_pre["kl_max"],
                    "raw_factor_1_kl_max": raw_factor_1_kl_max,
                    "sentinel_kl_max": float(factor_results[-1]["post_sentinel_kl_max"]) if factor_results else sentinel_pre["kl_max"],
                    "accepted_lr_factor": None,
                    "accepted_sentinel_kl_mean": None,
                    "accepted_sentinel_kl_p99": None,
                    "accepted_sentinel_kl_max": None,
                    "backtrack_count": max(0, len(factor_results) - 1),
                    "factor_results": factor_results,
                    "optimizer_step_performed": bool(any_step),
                    "rollback_performed": bool(any_step),
                    "gradient_source_sha256": gradient_sha,
                    "gradient_norm": post_clip_norm,
                    "actor_parameter_delta_norm": 0.0,
                    "optimizer_before_sha256": optimizer_before_sha,
                    "optimizer_after_sha256": optimizer_state_sha256(self.actor_optimizer),
                    "actor_before_sha256": actor_before_sha,
                    "actor_after_sha256": self.actor_fingerprint(),
                    "critic_optimizer_step_count": int(self.critic_optimizer_step_count),
                    "trust_region_enabled": True,
                    "actor_proposal_id": int(self.actor_proposal_count),
                }

            for group, base_lr in zip(self.actor_optimizer.param_groups, base_lrs):
                group["lr"] = base_lr
            self.actor_optimizer_step_count += 1
            self.actor_update_count += 1
            if float(accepted_factor) != 1.0:
                self.actor_recovery_update_count += 1
            delta_terms = []
            for current, before in zip(parameters, parameter_before):
                delta_terms.append((current.detach() - before.to(device=current.device)).reshape(-1))
            delta = torch.cat(delta_terms) if delta_terms else torch.zeros((0,), device=self.device)
            moments = self._moment_norms(self.actor_optimizer, torch)
            moment_values = []
            for state in self.actor_optimizer.state.values():
                if "exp_avg" in state:
                    moment_values.append(state["exp_avg"].detach().reshape(-1))
            moment_flat = torch.cat(moment_values) if moment_values else torch.zeros((0,), device=self.device)
            cosine = self._cosine(gradient_flat, moment_flat, torch)
            post_batch = self._policy_metrics(
                {"depth": depth, "vector": vector, "action_mask": action_mask},
                entropy_floor=entropy_floor,
            )
            batch_pre = {
                "mean": float(pre_kl.mean().item()),
                "p99": float(torch.quantile(pre_kl, 0.99).item()),
                "max": float(pre_kl.max().item()),
            }
            post_flip = post_batch.get("argmax_flip_rate", 0.0)
            return {
                "accepted": True,
                "rejected": False,
                "hard_stop_result": "PASS",
                "rejection_reason": "",
                "actor_loss": float(actor_loss.item()),
                "q_term_mean": float(q_term.mean().item()),
                "entropy_term_mean": float(entropy_term.mean().item()),
                "bc_kl_term_mean": float(bc_kl_term.mean().item()),
                "bc_kl_mean": batch_pre["mean"],
                "bc_kl_p99": batch_pre["p99"],
                "bc_kl_max": batch_pre["max"],
                "bc_kl_post_mean": post_batch["kl_mean"],
                "bc_kl_post_max": post_batch["kl_max"],
                "entropy_mean": float(pre_entropy.mean().item()),
                "entropy_min": float(accepted_stats.get("entropy_min", 0.0)),
                "entropy_floor_eligible_count": int(accepted_stats.get("entropy_floor_eligible_count", 0)),
                "action_flip_rate": float(post_flip),
                "actor_grad_norm": pre_clip_norm,
                "actor_grad_norm_post_clip": post_clip_norm,
                "gradient_norm_q": component_norms["q"],
                "gradient_norm_entropy": component_norms["entropy"],
                "gradient_norm_bc_kl": component_norms["bc_kl"],
                "gradient_norm_total": pre_clip_norm,
                "gradient_norm": post_clip_norm,
                "gradient_source_sha256": gradient_sha,
                "gradient_adam_exp_avg_cosine": cosine,
                "actor_before_sha256": actor_before_sha,
                "actor_after_sha256": self.actor_fingerprint(),
                "optimizer_before_sha256": optimizer_before_sha,
                "optimizer_after_sha256": optimizer_state_sha256(self.actor_optimizer),
                "pre_step_kl_used": True,
                "pre_step_hard_stop": False,
                "post_step_hard_stop_checked": True,
                "post_step_hard_stop": False,
                "optimizer_step_performed": True,
                "rollback_performed": bool(accepted_factor != factors[0]),
                "replay_unchanged": True,
                "critic_optimizer_step_count": int(self.critic_optimizer_step_count),
                "alpha": float(self.alpha),
                "q_min_mean": float(q_min.mean().item()),
                "q_min_std": float(q_min.std(unbiased=False).item()),
                "q_min_min": float(q_min.min().item()),
                "q_min_max": float(q_min.max().item()),
                "pre_sentinel_kl_mean": sentinel_pre["kl_mean"],
                "pre_sentinel_kl_p99": sentinel_pre["kl_p99"],
                "pre_sentinel_kl_max": sentinel_pre["kl_max"],
                "raw_factor_1_kl_max": raw_factor_1_kl_max,
                "accepted_lr_factor": accepted_factor,
                "accepted_sentinel_kl_mean": accepted_stats["kl_mean"],
                "accepted_sentinel_kl_p99": accepted_stats["kl_p99"],
                "accepted_sentinel_kl_max": accepted_stats["kl_max"],
                "sentinel_kl_mean": accepted_stats["kl_mean"],
                "sentinel_kl_p99": accepted_stats["kl_p99"],
                "sentinel_kl_max": accepted_stats["kl_max"],
                "backtrack_count": int(factors.index(accepted_factor)),
                "accepted_actor_update_id": int(self.actor_update_count),
                "actor_proposal_id": int(self.actor_proposal_count),
                "actor_parameter_delta_norm": float(torch.linalg.vector_norm(delta).item()),
                "optimizer_moment_norm": moments["exp_avg_norm"],
                "optimizer_exp_avg_norm": moments["exp_avg_norm"],
                "optimizer_exp_avg_sq_norm": moments["exp_avg_sq_norm"],
                "trust_region_enabled": True,
                "factor_results": factor_results,
            }
        finally:
            for module in (self.critic1, self.critic2):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def _distribution(self, model, depth, vector, action_mask):
        probabilities, log_probs, mask = masked_categorical(
            model(depth, vector), action_mask, self.torch
        )
        return probabilities, log_probs, mask

    def action_distribution(self, depth, vector, action_mask, *, use_bc_reference=False):
        model = self.bc_reference if use_bc_reference else self.actor
        with self.torch.no_grad():
            return self._distribution(model, depth, vector, action_mask)

    def select_action(self, depth, vector, action_mask, rng, *, warmup: bool):
        probabilities, log_probs, mask = self.action_distribution(
            depth, vector, action_mask, use_bc_reference=bool(warmup)
        )
        values = probabilities[0].detach().cpu().numpy().astype(np.float64)
        valid = np.flatnonzero(mask[0].detach().cpu().numpy().astype(bool))
        if valid.size == 0 or not np.isfinite(values[valid]).all():
            self.invalid_action_count += 1
            raise FloatingPointError("SAC masked policy has no finite valid action")
        values = values[valid]
        values /= float(values.sum())
        action = int(rng.choice(valid, p=values))
        bc_prob, bc_log, _ = self.action_distribution(
            depth, vector, action_mask, use_bc_reference=True
        )
        actor_argmax = int(torch_argmax_valid(probabilities[0], mask[0], self.torch))
        bc_argmax = int(torch_argmax_valid(bc_prob[0], mask[0], self.torch))
        return {
            "action": action,
            "log_prob": float(log_probs[0, action].item()),
            "entropy": float(-(probabilities[0] * log_probs[0]).sum().item()),
            "bc_kl": float(masked_kl(probabilities, log_probs, bc_prob, bc_log, self.torch)[0].item()),
            "argmax_flip": int(actor_argmax != bc_argmax),
            "behavior_policy_version": "bc_masked_categorical_t1_warmup_v1" if warmup else self.actor_policy_id,
            "valid_action_count": int(valid.size),
        }

    def update_critic(self, batch: Mapping[str, Any]) -> Mapping[str, float]:
        torch = self.torch
        depth = batch["depth"]
        vector = batch["vector"]
        action = batch["action"].long()
        reward = batch["reward"].float()
        next_depth = batch["next_depth"]
        next_vector = batch["next_vector"]
        next_mask = batch["next_action_mask"].bool()
        done = batch["done"].float()
        with torch.no_grad():
            # Terminal transitions have no successor state under the episode
            # contract, so their next mask is intentionally empty.  Do not
            # construct a policy distribution for those rows: they must use
            # the zero bootstrap below.  Non-terminal rows still go through
            # the strict non-empty-mask validation.
            nonterminal = done < 0.5
            next_value = torch.zeros_like(reward)
            if bool(nonterminal.any().item()):
                next_prob, next_log, _ = self._distribution(
                    self.actor,
                    next_depth[nonterminal],
                    next_vector[nonterminal],
                    next_mask[nonterminal],
                )
                target_q1 = self.target_critic1(next_depth[nonterminal], next_vector[nonterminal])
                target_q2 = self.target_critic2(next_depth[nonterminal], next_vector[nonterminal])
                target_q = torch.minimum(target_q1, target_q2)
                next_value[nonterminal] = discrete_value_from_policy(
                    target_q, next_prob, next_log, self.alpha, torch
                )
            target = self.reward_scale * reward + self.gamma * (1.0 - done) * next_value
        q1_all = self.critic1(depth, vector)
        q2_all = self.critic2(depth, vector)
        q1 = q1_all.gather(1, action[:, None]).squeeze(1)
        q2 = q2_all.gather(1, action[:, None]).squeeze(1)
        loss1 = torch.nn.functional.mse_loss(q1, target)
        loss2 = torch.nn.functional.mse_loss(q2, target)
        loss = loss1 + loss2
        if not bool(torch.isfinite(loss).item()):
            self.nan_count += 1
            raise FloatingPointError("non-finite SAC critic loss")
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            self.gradient_clip_norm,
        )
        if not bool(torch.isfinite(grad_norm).item()):
            self.nan_count += 1
            raise FloatingPointError("non-finite SAC critic gradient")
        self.critic_optimizer.step()
        _polyak_update(self.target_critic1, self.critic1, self.tau, torch)
        _polyak_update(self.target_critic2, self.critic2, self.tau, torch)
        self.critic_optimizer_step_count += 1
        self.critic_update_count += 1
        with torch.no_grad():
            q_min = torch.minimum(q1, q2)
            td_error = 0.5 * (q1 + q2) - target
        return {
            "critic1_loss": float(loss1.item()),
            "critic2_loss": float(loss2.item()),
            "q_mean": float(q_min.mean().item()),
            "q_std": float(q_min.std(unbiased=False).item()),
            "td_target_mean": float(target.mean().item()),
            "td_target_std": float(target.std(unbiased=False).item()),
            "td_error_mean": float(td_error.mean().item()),
            "td_error_abs_mean": float(td_error.abs().mean().item()),
            "critic_grad_norm": float(grad_norm.item()),
            "terminal_fraction": float(done.mean().item()),
        }

    def update_actor(self, batch: Mapping[str, Any]) -> Mapping[str, float]:
        torch = self.torch
        depth = batch["depth"]
        vector = batch["vector"]
        action_mask = batch["action_mask"].bool()
        for module in (self.critic1, self.critic2):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        try:
            probabilities, log_probs, mask = self._distribution(
                self.actor, depth, vector, action_mask
            )
            with torch.no_grad():
                bc_prob, bc_log, _ = self._distribution(
                    self.bc_reference, depth, vector, action_mask
                )
                bc_entropy = -(bc_prob * bc_log).sum(dim=1)
            q_min = torch.minimum(self.critic1(depth, vector), self.critic2(depth, vector))
            kl = masked_kl(probabilities, log_probs, bc_prob, bc_log, torch)
            actor_loss_per_state = (
                probabilities * (self.alpha * log_probs - q_min)
            ).sum(dim=1) + self.beta_bc * kl
            actor_loss = actor_loss_per_state.mean()
            if not bool(torch.isfinite(actor_loss).item()):
                self.nan_count += 1
                raise FloatingPointError("non-finite SAC actor loss")
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_trainable_parameters(), self.gradient_clip_norm
            )
            if not bool(torch.isfinite(grad_norm).item()):
                self.nan_count += 1
                raise FloatingPointError("non-finite SAC actor gradient")
            self.actor_optimizer.step()
        finally:
            for module in (self.critic1, self.critic2):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        self.actor_optimizer_step_count += 1
        self.actor_update_count += 1
        with torch.no_grad():
            entropy = -(probabilities * log_probs).sum(dim=1)
            flip = (
                torch.argmax(probabilities.masked_fill(~mask, -1.0), dim=1)
                != torch.argmax(bc_prob.masked_fill(~mask, -1.0), dim=1)
            ).float()
            # A state with one valid action is structurally deterministic and
            # has zero entropy even when the policy has not collapsed.  Keep
            # those rows out of the entropy-floor safety diagnostic while
            # retaining the minimum over genuinely choice-bearing states.
            entropy_floor_eligible = (mask.sum(dim=1) > 1) & (
                bc_entropy >= self.entropy_floor
            )
            entropy_floor_eligible_count = int(entropy_floor_eligible.sum().item())
            if entropy_floor_eligible_count:
                entropy_min = float(entropy[entropy_floor_eligible].min().item())
            else:
                entropy_min = 0.0
        return {
            "actor_loss": float(actor_loss.item()),
            "bc_kl_mean": float(kl.mean().item()),
            "bc_kl_max": float(kl.max().item()),
            "entropy_mean": float(entropy.mean().item()),
            "entropy_min": entropy_min,
            "entropy_floor_eligible_count": entropy_floor_eligible_count,
            "action_flip_rate": float(flip.mean().item()),
            "actor_grad_norm": float(grad_norm.item()),
        }

    @staticmethod
    def _gradient_norm(values, torch) -> float:
        terms = [value.detach().reshape(-1) for value in values if value is not None]
        if not terms:
            return 0.0
        return float(torch.linalg.vector_norm(torch.cat(terms)).item())

    def update_actor_transactional(
        self,
        batch: Mapping[str, Any],
        *,
        hard_stop: float,
        entropy_floor: float,
        trust_region_factors=None,
    ) -> Mapping[str, Any]:
        """Propose one Actor update and commit it only after post-step gates.

        This is SAC-local V2 instrumentation.  The mathematical loss and all
        frozen inputs match ``update_actor``.  A rejected proposal restores
        both Actor parameters and Adam state byte-for-byte; Critic and target
        modules are never stepped.
        """
        if trust_region_factors is not None:
            if self._kl_sentinel is None:
                raise RuntimeError("trust-region Actor update requires a frozen KL sentinel")
            return self._update_actor_transactional_trust(
                batch,
                hard_stop=hard_stop,
                entropy_floor=entropy_floor,
                trust_region_factors=trust_region_factors,
            )

        torch = self.torch
        parameters = self.actor_trainable_parameters()
        actor_before = deepcopy(self.actor.state_dict())
        optimizer_before = deepcopy(self.actor_optimizer.state_dict())
        actor_before_sha = state_dict_sha256(actor_before)
        optimizer_before_sha = optimizer_state_sha256(self.actor_optimizer)
        for module in (self.critic1, self.critic2):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        try:
            self.actor_proposal_count += 1
            depth = batch["depth"]
            vector = batch["vector"]
            action_mask = batch["action_mask"].bool()
            probabilities, log_probs, mask = self._distribution(
                self.actor, depth, vector, action_mask
            )
            with torch.no_grad():
                bc_prob, bc_log, _ = self._distribution(
                    self.bc_reference, depth, vector, action_mask
                )
                bc_entropy = -(bc_prob * bc_log).sum(dim=1)
            q_min = torch.minimum(self.critic1(depth, vector), self.critic2(depth, vector))
            q_term = (probabilities * (-q_min)).sum(dim=1)
            entropy_term = (probabilities * log_probs).sum(dim=1) * self.alpha
            kl = masked_kl(probabilities, log_probs, bc_prob, bc_log, torch)
            bc_kl_term = self.beta_bc * kl
            actor_loss_per_state = q_term + entropy_term + bc_kl_term
            actor_loss = actor_loss_per_state.mean()
            if not bool(torch.isfinite(actor_loss).item()):
                self.nan_count += 1
                raise FloatingPointError("non-finite SAC actor loss")
            component_norms = {}
            for name, component in (
                ("q", q_term),
                ("entropy", entropy_term),
                ("bc_kl", bc_kl_term),
            ):
                grads = torch.autograd.grad(
                    component.mean(), parameters, retain_graph=True, allow_unused=True
                )
                component_norms[name] = self._gradient_norm(grads, torch)
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            pre_clip_norm = float(
                torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_norm).item()
            )
            if not math.isfinite(pre_clip_norm):
                self.nan_count += 1
                raise FloatingPointError("non-finite SAC actor gradient")
            post_clip_norm = self._gradient_norm(
                [parameter.grad for parameter in parameters], torch
            )
            pre_kl = kl.detach()
            pre_entropy = (-(probabilities * log_probs).sum(dim=1)).detach()
            pre_max = float(pre_kl.max().item())
            pre_max_position = int(torch.argmax(pre_kl).item())
            pre_entropy_eligible = (mask.sum(dim=1) > 1) & (bc_entropy >= entropy_floor)
            pre_entropy_eligible_count = int(pre_entropy_eligible.sum().item())
            pre_entropy_min = (
                float(pre_entropy[pre_entropy_eligible].min().item())
                if pre_entropy_eligible_count
                else 0.0
            )
            pre_flip = (
                torch.argmax(probabilities.masked_fill(~mask, -1.0), dim=1)
                != torch.argmax(bc_prob.masked_fill(~mask, -1.0), dim=1)
            ).float()

            # A pre-step hard stop is fail-closed: the proposal is journaled
            # as rejected, but Adam must never see the unsafe proposal.  This
            # ordering is intentionally distinct from the historical V2
            # behavior, which stepped first and rolled back afterward.
            if pre_max >= float(hard_stop):
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.actor_rejection_count += 1
                return {
                    "accepted": False,
                    "rejected": True,
                    "hard_stop_result": "REJECT_PRE_KL",
                    "rejection_reason": "pre_step_bc_kl_hard_stop",
                    "actor_loss": float(actor_loss.item()),
                    "q_term_mean": float(q_term.mean().item()),
                    "entropy_term_mean": float(entropy_term.mean().item()),
                    "bc_kl_term_mean": float(bc_kl_term.mean().item()),
                    "bc_kl_mean": float(pre_kl.mean().item()),
                    "bc_kl_p90": float(torch.quantile(pre_kl, 0.90).item()),
                    "bc_kl_p99": float(torch.quantile(pre_kl, 0.99).item()),
                    "bc_kl_max": pre_max,
                    "bc_kl_max_position": pre_max_position,
                    "bc_kl_post_mean": float(pre_kl.mean().item()),
                    "bc_kl_post_max": pre_max,
                    "bc_kl_post_max_position": pre_max_position,
                    "entropy_mean": float(pre_entropy.mean().item()),
                    "entropy_p90": float(torch.quantile(pre_entropy, 0.90).item()),
                    "entropy_p99": float(torch.quantile(pre_entropy, 0.99).item()),
                    "entropy_min": pre_entropy_min,
                    "entropy_floor_eligible_count": pre_entropy_eligible_count,
                    "action_flip_rate": float(pre_flip.mean().item()),
                    "actor_grad_norm": pre_clip_norm,
                    "actor_grad_norm_post_clip": post_clip_norm,
                    "gradient_norm_q": component_norms["q"],
                    "gradient_norm_entropy": component_norms["entropy"],
                    "gradient_norm_bc_kl": component_norms["bc_kl"],
                    "gradient_norm_total": pre_clip_norm,
                    "actor_before_sha256": actor_before_sha,
                    "actor_after_proposal_sha256": actor_before_sha,
                    "actor_after_sha256": self.actor_fingerprint(),
                    "optimizer_before_sha256": optimizer_before_sha,
                    "optimizer_after_proposal_sha256": optimizer_before_sha,
                    "optimizer_after_sha256": optimizer_state_sha256(self.actor_optimizer),
                    "pre_step_kl_used": True,
                    "pre_step_hard_stop": True,
                    "post_step_hard_stop_checked": False,
                    "post_step_hard_stop": False,
                    "optimizer_step_performed": False,
                    "rollback_performed": False,
                    "replay_unchanged": True,
                    "critic_optimizer_step_count": int(self.critic_optimizer_step_count),
                    "alpha": float(self.alpha),
                    "q_min_mean": float(q_min.mean().item()),
                    "q_min_std": float(q_min.std(unbiased=False).item()),
                    "q_min_min": float(q_min.min().item()),
                    "q_min_max": float(q_min.max().item()),
                }

            self.actor_optimizer.step()
            with torch.no_grad():
                post_prob, post_log, post_mask = self._distribution(
                    self.actor, depth, vector, action_mask
                )
                post_kl = masked_kl(post_prob, post_log, bc_prob, bc_log, torch)
                post_entropy = -(post_prob * post_log).sum(dim=1)
                flip = (
                    torch.argmax(post_prob.masked_fill(~post_mask, -1.0), dim=1)
                    != torch.argmax(bc_prob.masked_fill(~post_mask, -1.0), dim=1)
                ).float()
                eligible = (post_mask.sum(dim=1) > 1) & (bc_entropy >= entropy_floor)
                eligible_count = int(eligible.sum().item())
                entropy_min = float(post_entropy[eligible].min().item()) if eligible_count else 0.0
            post_max = float(post_kl.max().item())
            post_max_position = int(torch.argmax(post_kl).item())
            entropy_violation = bool(eligible_count and entropy_min < float(entropy_floor))
            hard_stop_result = "PASS"
            rejection_reason = ""
            if post_max >= float(hard_stop):
                hard_stop_result = "REJECT_POST_KL"
                rejection_reason = "post_step_bc_kl_hard_stop"
            elif entropy_violation:
                hard_stop_result = "REJECT_POST_ENTROPY"
                rejection_reason = "post_step_entropy_floor"
            accepted = hard_stop_result == "PASS"
            actor_after_proposal_sha = self.actor_fingerprint()
            optimizer_after_proposal_sha = optimizer_state_sha256(self.actor_optimizer)
            if not accepted:
                self.actor.load_state_dict(actor_before, strict=True)
                self.actor_optimizer.load_state_dict(optimizer_before)
                self.actor_rejection_count += 1
            else:
                self.actor_optimizer_step_count += 1
                self.actor_update_count += 1
            return {
                "accepted": bool(accepted),
                "rejected": not bool(accepted),
                "hard_stop_result": hard_stop_result,
                "rejection_reason": rejection_reason,
                "actor_loss": float(actor_loss.item()),
                "q_term_mean": float(q_term.mean().item()),
                "entropy_term_mean": float(entropy_term.mean().item()),
                "bc_kl_term_mean": float(bc_kl_term.mean().item()),
                "bc_kl_mean": float(pre_kl.mean().item()),
                "bc_kl_p90": float(torch.quantile(pre_kl, 0.90).item()),
                "bc_kl_p99": float(torch.quantile(pre_kl, 0.99).item()),
                "bc_kl_max": pre_max,
                "bc_kl_max_position": pre_max_position,
                "bc_kl_post_mean": float(post_kl.mean().item()),
                "bc_kl_post_max": post_max,
                "bc_kl_post_max_position": post_max_position,
                "entropy_mean": float(pre_entropy.mean().item()),
                "entropy_p90": float(torch.quantile(pre_entropy, 0.90).item()),
                "entropy_p99": float(torch.quantile(pre_entropy, 0.99).item()),
                "entropy_min": entropy_min,
                "entropy_floor_eligible_count": eligible_count,
                "action_flip_rate": float(flip.mean().item()),
                "actor_grad_norm": pre_clip_norm,
                "actor_grad_norm_post_clip": post_clip_norm,
                "gradient_norm_q": component_norms["q"],
                "gradient_norm_entropy": component_norms["entropy"],
                "gradient_norm_bc_kl": component_norms["bc_kl"],
                "gradient_norm_total": pre_clip_norm,
                "actor_before_sha256": actor_before_sha,
                "actor_after_proposal_sha256": actor_after_proposal_sha,
                "actor_after_sha256": self.actor_fingerprint(),
                "optimizer_before_sha256": optimizer_before_sha,
                "optimizer_after_proposal_sha256": optimizer_after_proposal_sha,
                "optimizer_after_sha256": optimizer_state_sha256(self.actor_optimizer),
                "pre_step_kl_used": True,
                "pre_step_hard_stop": False,
                "post_step_hard_stop_checked": True,
                "post_step_hard_stop": bool(post_max >= float(hard_stop)),
                "optimizer_step_performed": True,
                "rollback_performed": not bool(accepted),
                "replay_unchanged": True,
                "critic_optimizer_step_count": int(self.critic_optimizer_step_count),
                "alpha": float(self.alpha),
                "q_min_mean": float(q_min.mean().item()),
                "q_min_std": float(q_min.std(unbiased=False).item()),
                "q_min_min": float(q_min.min().item()),
                "q_min_max": float(q_min.max().item()),
            }
        finally:
            for module in (self.critic1, self.critic2):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def state_payload(self) -> Mapping[str, Any]:
        return {
            "actor_architecture": self.actor_architecture,
            "actor_policy_id": self.actor_policy_id,
            "residual_logit_cap": float(self.residual_logit_cap),
            "theoretical_kl_bound": float(2.0 * self.residual_logit_cap)
            if self.actor_architecture == "residual"
            else None,
            "actor_state_dict": self.actor.state_dict(),
            "bc_reference_state_dict": self.bc_reference.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "target_critic1_state_dict": self.target_critic1.state_dict(),
            "target_critic2_state_dict": self.target_critic2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "actor_optimizer_step_count": int(self.actor_optimizer_step_count),
            "critic_optimizer_step_count": int(self.critic_optimizer_step_count),
            "actor_update_count": int(self.actor_update_count),
            "critic_update_count": int(self.critic_update_count),
            "actor_proposal_count": int(self.actor_proposal_count),
            "actor_rejection_count": int(self.actor_rejection_count),
            "actor_recovery_update_count": int(self.actor_recovery_update_count),
            "nan_count": int(self.nan_count),
            "invalid_action_count": int(self.invalid_action_count),
        }

    def load_state_payload(self, payload: Mapping[str, Any]) -> None:
        """Restore model, optimizer and counters from a compatible checkpoint."""

        architecture = str(payload.get("actor_architecture", "direct"))
        if architecture != self.actor_architecture:
            raise ValueError(
                "SAC Actor architecture mismatch: {} != {}".format(
                    architecture, self.actor_architecture
                )
            )
        if architecture == "residual":
            cap = float(payload.get("residual_logit_cap", -1.0))
            if cap != self.residual_logit_cap:
                raise ValueError("SAC residual logit cap mismatch")
        self.actor.load_state_dict(payload["actor_state_dict"], strict=True)
        self.bc_reference.load_state_dict(payload["bc_reference_state_dict"], strict=True)
        self.critic1.load_state_dict(payload["critic1_state_dict"], strict=True)
        self.critic2.load_state_dict(payload["critic2_state_dict"], strict=True)
        self.target_critic1.load_state_dict(payload["target_critic1_state_dict"], strict=True)
        self.target_critic2.load_state_dict(payload["target_critic2_state_dict"], strict=True)
        self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        for name in (
            "actor_optimizer_step_count",
            "critic_optimizer_step_count",
            "actor_update_count",
            "critic_update_count",
            "actor_proposal_count",
            "actor_rejection_count",
            "actor_recovery_update_count",
            "nan_count",
            "invalid_action_count",
        ):
            if name in payload:
                setattr(self, name, int(payload[name]))


def torch_argmax_valid(values, mask, torch) -> int:
    masked = values.masked_fill(~mask.bool(), torch.finfo(values.dtype).min)
    return int(torch.argmax(masked).item())


__all__ = [
    "DiscreteSACAgent",
    "build_policy",
    "build_q_network",
    "masked_categorical",
    "masked_kl",
    "discrete_value_from_policy",
    "state_dict_sha256",
]
