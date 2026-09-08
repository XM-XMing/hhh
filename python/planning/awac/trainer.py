"""Independent discrete AWAC trainer entrypoint.

This module is the sole formal RL training owner.  It accepts a strict BC V2
handoff, constructs the existing AWAC learner and replay, and persists the
AWAC checkpoint schema.  ``--dry-run`` and ``--synthetic-smoke`` exercise the
construction seams without starting Unity or collecting data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback
from typing import Any, Dict, Mapping, Optional

import numpy as np

from planning.awac.calibration import (
    HOLDOUT_RETURN_SEMANTICS,
    CriticCalibrationConfig,
    HoldoutEpisodeValidationError,
    assess_value_policy_alignment,
    build_calibration_checkpoint_payload,
    build_calibration_pass_checkpoint_payload,
    episode_monte_carlo_returns,
    evaluate_calibration_gate,
    masked_bellman_target,
    summarize_calibration_window,
)
from planning.awac.calibration_runtime import (
    CalibrationReplayProducer,
    build_calibration_split,
    calibration_behavior_policy_identity,
    calibration_mission_source_identity,
    load_calibration_missions,
)
from planning.awac.checkpoint import (
    build_awac_checkpoint_identity,
    calibration_replay_identity,
    committed_checkpoint_kind,
    commit_calibration_checkpoint_transaction,
    commit_standard_awac_checkpoint_transaction,
    calibration_checkpoint_transaction_diagnostics_path,
    load_calibration_checkpoint,
    load_calibration_pass_checkpoint,
    load_committed_calibration_resume_checkpoint,
    load_committed_standard_resume_checkpoint,
    recover_calibration_replay_from_checkpoint_transaction,
    save_awac_checkpoint,
    save_calibration_checkpoint,
    save_calibration_pass_checkpoint,
    validate_standard_awac_handoff_payload,
    validate_standard_awac_checkpoint_payload,
    validate_standard_online_replay_counter,
    write_calibration_transaction_json,
)
from planning.awac.contract import (
    AWAC_ALGORITHM_ID,
    AWAC_CHECKPOINT_CONTRACT_ID,
    AWAC_PHASE_CRITIC_CALIBRATION,
    AWAC_PHASE_STANDARD_TRAINING,
    AWAC_TRAINING_CONFIG_CONTRACT_ID,
    awac_source_sha256,
    awac_training_contract_sha256,
    build_awac_training_contract,
)
from planning.awac.config_identity import (
    TRAINING_SEMANTIC,
    UNKNOWN,
    compare_awac_config_identity,
    project_awac_training_config,
)
from planning.awac.interaction import BehaviorSource
from planning.awac.confidence import (
    CONFIDENCE_CONTRACT_ID,
    CONFIDENCE_FORMULA_VERSION,
    CONFIDENCE_SCALE_VERSION,
)
from planning.awac.learner import (
    AWACOptimizationConfig,
    DiscreteAWACLearner,
    resolve_torch_device,
)
from planning.awac.phase1 import (
    CALIBRATION_GATE_PENDING,
    PHASE1_DEFAULT_ONLINE_TRANSITION_CAP,
    Phase1CalibrationController,
    Phase1CalibrationSafetyCap,
    Phase1ReadinessError,
    Phase1StateMachine,
    build_phase1_readiness_dry_run,
    build_phase1_training_contract,
    phase1_training_contract_sha256,
    validate_phase1_depth_learning_rates,
)
from planning.awac.replay import AWACReplayBuffer, move_awac_batch_to_device
from planning.awac.online_runtime import (
    STANDARD_ONLINE_RUNTIME_SCHEMA_ID,
    StandardAWACOnlineRunner,
    build_standard_online_summary,
)
from planning.awac.replay_audit import audit_awac_replay
from planning.bc.model import require_torch
from planning.bc.model import VectorNormalizer
from planning.common.checkpoint import load_torch
from planning.common.hashing import file_sha256
from planning.common.io import write_json_atomic
from planning.contracts.awac_handoff import (
    ACTION_ORDERING,
    EXPECTED_OBSERVATION_CONTRACT,
    EXPECTED_OBSERVATION_SOURCE,
    validate_awac_bc_checkpoint,
)
from planning.contracts.collection import canonical_sha256
from planning.contracts.feature import (
    FEATURE_CONTRACT_ID,
    INITIAL_PREV_ACTION,
    NUM_ACTIONS,
    POLICY_VECTOR_DIM,
    policy_input_contract_sha256,
)
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.contracts.policy_runtime import POLICY_RUNTIME_CONTRACT_ID
from planning.contracts.policy_checkpoint_fingerprint import actor_state_sha256
from planning.contracts.reward import REWARD_CONTRACT_ID, reward_contract_sha256
from planning.contracts.task import (
    DEFAULT_MAX_PRIMITIVE_STEPS,
    TASK_CONTRACT_ID,
    task_contract_sha256,
)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_awac.py",
        allow_abbrev=False,
        description="Train the formal masked-discrete AWAC learner from a strict BC V2 checkpoint.",
    )
    parser.add_argument("--bc-checkpoint", required=True)
    parser.add_argument(
        "--phase",
        choices=(AWAC_PHASE_CRITIC_CALIBRATION, AWAC_PHASE_STANDARD_TRAINING),
        default=AWAC_PHASE_CRITIC_CALIBRATION,
    )
    parser.add_argument("--train-index", default="")
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--calibration-split", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--replay-dir", default="")
    parser.add_argument("--mpl-contract-sha256", default="")
    parser.add_argument(
        "--total-env-steps",
        type=int,
        default=6000,
        help="legacy calibration budget; Standard phase uses it only when --online-env-steps is omitted",
    )
    parser.add_argument(
        "--online-env-steps",
        type=int,
        default=None,
        help="additional phase-local environment steps for Standard AWAC online training",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_PRIMITIVE_STEPS)
    parser.add_argument("--env-workers", type=int, default=1)
    parser.add_argument("--worker-spec-file", default="")
    parser.add_argument("--reliable-v4", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--actor-learning-starts", type=int, default=8000)
    parser.add_argument("--critic-burnin-updates", type=int, default=2000)
    parser.add_argument("--actor-update-interval", type=int, default=4)
    parser.add_argument("--updates-per-step", type=float, default=0.50)
    parser.add_argument("--warmup-temperature", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-head-lr", type=float, default=1.0e-5)
    parser.add_argument("--actor-vector-lr", type=float, default=3.0e-6)
    parser.add_argument("--actor-depth-lr", type=float, default=1.0e-6)
    parser.add_argument("--critic-head-lr", type=float, default=1.0e-4)
    parser.add_argument("--critic-vector-lr", type=float, default=1.0e-5)
    parser.add_argument("--critic-depth-lr", type=float, default=1.0e-5)
    parser.add_argument("--critic-cql-weight", type=float, default=0.05)
    parser.add_argument("--awac-temperature", type=float, default=2.0)
    parser.add_argument("--awac-weight-max", type=float, default=20.0)
    parser.add_argument("--awac-bc-kl-weight", type=float, default=0.05)
    parser.add_argument("--awac-trust-tail-top-k", type=int, default=16)
    parser.add_argument("--bc-kl-hard-budget", type=float, default=0.10)
    parser.add_argument("--bc-kl-recovery-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--enable-twin-q-confidence", action="store_true")
    parser.add_argument("--enable-adaptive-bc-kl", action="store_true")
    parser.add_argument(
        "--enable-primitive-neighbor-exploration", action="store_true"
    )
    parser.add_argument("--confidence-margin-gain", type=float, default=2.0)
    parser.add_argument("--confidence-disagreement-gain", type=float, default=1.0)
    parser.add_argument("--adaptive-bc-kl-beta-min", type=float, default=0.02)
    parser.add_argument("--adaptive-bc-kl-beta-max", type=float, default=0.10)
    parser.add_argument(
        "--primitive-neighborhood-artifact",
        "--primitive-neighborhood-artifact-path",
        dest="primitive_neighborhood_artifact_path",
        default="",
    )
    parser.add_argument("--primitive-neighborhood-artifact-sha256", default="")
    parser.add_argument("--primitive-neighbor-top-k", type=int, default=8)
    parser.add_argument("--primitive-neighbor-radius", type=float, default=-1.0)
    parser.add_argument("--primitive-local-global-mix", type=float, default=0.20)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=5000)
    parser.add_argument("--log-interval-steps", type=int, default=100)
    parser.add_argument(
        "--tensorboard-log-dir",
        default="",
        help="override the Standard AWAC TensorBoard directory",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="disable the Standard AWAC scalar TensorBoard logger",
    )
    parser.add_argument("--reset-settle", type=float, default=0.30)
    parser.add_argument("--reset-timeout", type=float, default=10.0)
    parser.add_argument("--worker-ready-timeout", type=float, default=30.0)
    parser.add_argument("--post-wait", type=float, default=0.0)
    parser.add_argument("--max-sensor-skew-ms", type=float, default=80.0)
    parser.add_argument("--depth-mask-collision-radius", type=float, default=0.40)
    parser.add_argument("--depth-mask-slack", type=float, default=0.08)
    parser.add_argument("--depth-mask-sample-stride", type=int, default=4)
    parser.add_argument("--depth-mask-max-patch-radius-px", type=int, default=14)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--calibration-smoke", action="store_true")
    parser.add_argument("--calibration-smoke-episodes", type=int, default=20)
    parser.add_argument("--calibration-smoke-transitions", type=int, default=160)
    parser.add_argument("--calibration-window-interval-steps", type=int, default=32)
    parser.add_argument("--calibration-min-replay-transitions", type=int, default=128)
    parser.add_argument("--calibration-min-completed-episodes", type=int, default=16)
    parser.add_argument("--calibration-min-critic-updates", type=int, default=32)
    parser.add_argument("--calibration-min-holdout-episodes", type=int, default=8)
    parser.add_argument("--calibration-min-stability-windows", type=int, default=3)
    parser.add_argument("--calibration-holdout-fraction", type=float, default=0.10)
    parser.add_argument("--calibration-max-transitions", type=int, default=30_000)
    parser.add_argument("--calibration-max-episodes", type=int, default=1_000)
    parser.add_argument(
        "--phase1-readiness-dry-run",
        action="store_true",
        help="validate Phase 1 construction seams without runtime or optimizer steps",
    )
    parser.add_argument(
        "--phase1-dev-manifest",
        default="data/test/awac_dev_seed4026_100/manifest.json",
    )
    parser.add_argument(
        "--phase1-online-transition-cap",
        type=int,
        default=PHASE1_DEFAULT_ONLINE_TRANSITION_CAP,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.total_env_steps) <= 0:
        raise ValueError("total-env-steps must be positive")
    if args.online_env_steps is not None and int(args.online_env_steps) <= 0:
        raise ValueError("online-env-steps must be positive")
    if int(args.max_steps) <= 0:
        raise ValueError("max-steps must be positive")
    if not 1 <= int(args.env_workers) <= 20:
        raise ValueError("env-workers must be in [1,20]")
    if int(args.env_workers) > 1 and not str(args.worker_spec_file).strip():
        raise ValueError("worker-spec-file is required when env-workers > 1")
    if (
        str(args.phase) == AWAC_PHASE_CRITIC_CALIBRATION
        and str(args.train_index).strip()
        and not bool(args.reliable_v4)
    ):
        raise ValueError(
            "online critic calibration requires --reliable-v4 exact endpoint runtime"
        )
    if (
        str(args.phase) == AWAC_PHASE_CRITIC_CALIBRATION
        and str(args.train_index).strip()
        and not Path(str(args.train_index)).expanduser().is_file()
    ):
        raise FileNotFoundError(
            "training mission source does not exist: {}".format(
                Path(str(args.train_index)).expanduser().resolve()
            )
        )
    if int(args.batch_size) <= 0 or int(args.replay_capacity) < int(args.batch_size):
        raise ValueError("replay capacity must be at least batch size")
    if int(args.learning_starts) < int(args.batch_size):
        raise ValueError("learning-starts must be at least batch-size")
    if int(args.actor_learning_starts) < int(args.learning_starts):
        raise ValueError("actor-learning-starts must be at least learning-starts")
    if int(args.actor_update_interval) <= 0 or float(args.updates_per_step) <= 0.0:
        raise ValueError("AWAC update schedule is invalid")
    if float(args.reward_scale) <= 0.0:
        raise ValueError("reward-scale must be positive")
    if not math.isclose(float(args.reward_scale), 0.10, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("formal AWAC reward-scale is fixed at 0.10")
    if not 0.0 < float(args.gamma) <= 1.0 or not 0.0 < float(args.tau) <= 1.0:
        raise ValueError("gamma/tau are invalid")
    if float(args.actor_depth_lr) < 0.0 or float(args.critic_depth_lr) < 0.0:
        raise ValueError("depth learning rates must be non-negative")
    if str(args.phase) == AWAC_PHASE_STANDARD_TRAINING:
        if not str(args.resume_checkpoint).strip():
            raise Phase1ReadinessError(
                "Standard AWAC requires a committed calibration PASS resume checkpoint"
            )
        if not str(args.replay_dir).strip():
            raise Phase1ReadinessError(
                "Standard AWAC requires the calibration replay for handoff validation"
            )
        if not str(args.train_index).strip():
            raise Phase1ReadinessError(
                "Standard AWAC requires a mission source for online interaction"
            )
        if not Path(str(args.train_index)).expanduser().is_file():
            raise FileNotFoundError(
                "Standard AWAC mission source does not exist: {}".format(
                    Path(str(args.train_index)).expanduser().resolve()
                )
            )
        if not str(args.worker_spec_file).strip():
            raise Phase1ReadinessError(
                "Standard AWAC requires --worker-spec-file for managed runtime"
            )
        if not bool(args.reliable_v4):
            raise Phase1ReadinessError(
                "Standard AWAC requires --reliable-v4 exact endpoint runtime"
            )
        validate_phase1_depth_learning_rates(
            actor_depth_lr=float(args.actor_depth_lr),
            critic_depth_lr=float(args.critic_depth_lr),
        )
    if int(args.phase1_online_transition_cap) <= 0:
        raise ValueError("phase1-online-transition-cap must be positive")
    _calibration_safety_cap(args)
    if int(args.depth_mask_sample_stride) <= 0 or int(args.depth_mask_max_patch_radius_px) < 0:
        raise ValueError("depth mask geometry is invalid")
    if float(args.depth_mask_collision_radius) <= 0.0 or float(args.depth_mask_slack) < 0.0:
        raise ValueError("depth mask configuration is invalid")
    if int(args.checkpoint_interval_steps) <= 0 or int(args.log_interval_steps) <= 0:
        raise ValueError("checkpoint/log intervals must be positive")
    if int(args.calibration_smoke_episodes) <= 0:
        raise ValueError("calibration-smoke-episodes must be positive")
    if int(args.calibration_smoke_transitions) <= 0:
        raise ValueError("calibration-smoke-transitions must be positive")
    if int(args.calibration_window_interval_steps) <= 0:
        raise ValueError("calibration-window-interval-steps must be positive")
    _calibration_config(args)


def _online_env_steps(args: argparse.Namespace) -> int:
    """Resolve the unambiguous Standard budget as phase-local steps."""

    value = args.online_env_steps
    return int(args.total_env_steps if value is None else value)


def _optimization_config(args: argparse.Namespace) -> AWACOptimizationConfig:
    return AWACOptimizationConfig(
        gamma=float(args.gamma),
        tau=float(args.tau),
        actor_head_lr=float(args.actor_head_lr),
        actor_vector_lr=float(args.actor_vector_lr),
        actor_depth_lr=float(args.actor_depth_lr),
        critic_head_lr=float(args.critic_head_lr),
        critic_vector_lr=float(args.critic_vector_lr),
        critic_depth_lr=float(args.critic_depth_lr),
        critic_cql_weight=float(args.critic_cql_weight),
        awac_temperature=float(args.awac_temperature),
        awac_weight_max=float(args.awac_weight_max),
        bc_kl_weight=float(args.awac_bc_kl_weight),
        trust_tail_top_k=int(args.awac_trust_tail_top_k),
        bc_kl_hard_budget=float(args.bc_kl_hard_budget),
        bc_kl_recovery_weight=float(args.bc_kl_recovery_weight),
        gradient_clip_norm=float(args.gradient_clip_norm),
        enable_twin_q_confidence=bool(args.enable_twin_q_confidence),
        enable_adaptive_bc_kl=bool(args.enable_adaptive_bc_kl),
        enable_primitive_neighbor_exploration=bool(
            args.enable_primitive_neighbor_exploration
        ),
        confidence_margin_gain=float(args.confidence_margin_gain),
        confidence_disagreement_gain=float(args.confidence_disagreement_gain),
        adaptive_bc_kl_beta_min=float(args.adaptive_bc_kl_beta_min),
        adaptive_bc_kl_beta_max=float(args.adaptive_bc_kl_beta_max),
        primitive_neighborhood_artifact_path=str(
            args.primitive_neighborhood_artifact_path
        ),
        primitive_neighborhood_artifact_sha256=str(
            args.primitive_neighborhood_artifact_sha256
        ),
        primitive_neighbor_top_k=int(args.primitive_neighbor_top_k),
        primitive_neighbor_radius=float(args.primitive_neighbor_radius),
        primitive_local_global_mix=float(args.primitive_local_global_mix),
    )


def _calibration_config(args: argparse.Namespace) -> CriticCalibrationConfig:
    """Resolve the relative Phase-0 gate thresholds from the CLI."""

    return CriticCalibrationConfig(
        min_replay_transitions=int(args.calibration_min_replay_transitions),
        min_completed_episodes=int(args.calibration_min_completed_episodes),
        min_critic_updates=int(args.calibration_min_critic_updates),
        min_holdout_episodes=int(args.calibration_min_holdout_episodes),
        min_stability_windows=int(args.calibration_min_stability_windows),
        holdout_fraction=float(args.calibration_holdout_fraction),
    )


def _calibration_safety_cap(
    args: argparse.Namespace,
) -> Phase1CalibrationSafetyCap:
    """Resolve the typed maximum calibration bounds without changing gate rules."""

    return Phase1CalibrationSafetyCap(
        max_transitions=int(args.calibration_max_transitions),
        max_episodes=int(args.calibration_max_episodes),
    )


def _innovation_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve Phase-2 flags as an explicit future checkpoint identity."""

    return {
        "contract_id": "awac_innovation_foundation_v1",
        "enable_twin_q_confidence": bool(args.enable_twin_q_confidence),
        "enable_adaptive_bc_kl": bool(args.enable_adaptive_bc_kl),
        "enable_primitive_neighbor_exploration": bool(
            args.enable_primitive_neighbor_exploration
        ),
        "confidence": {
            "contract_id": CONFIDENCE_CONTRACT_ID,
            "formula_version": CONFIDENCE_FORMULA_VERSION,
            "scale_version": CONFIDENCE_SCALE_VERSION,
            "margin_gain": float(args.confidence_margin_gain),
            "disagreement_gain": float(args.confidence_disagreement_gain),
            "aggregation": "valid_action_mean",
            "q_scale": CONFIDENCE_SCALE_VERSION,
            "action_source": "masked_bc_argmax_vs_masked_awac_argmax",
            "delta_q": "qmin(a_rl)-qmin(a_bc)",
            "uncertainty": "abs(q1(a_rl)-q2(a_rl))",
        },
        "adaptive_bc_kl": {
            "contract_id": "awac_adaptive_bc_kl_v1",
            "beta_min": float(args.adaptive_bc_kl_beta_min),
            "beta_max": float(args.adaptive_bc_kl_beta_max),
            "base_weight": float(args.awac_bc_kl_weight),
            "hard_budget": float(args.bc_kl_hard_budget),
            "recovery_weight": float(args.bc_kl_recovery_weight),
            "order": "state_beta_then_hard_budget_recovery",
        },
        "primitive_exploration": {
            "contract_id": "awac_primitive_exploration_v1",
            "enabled": bool(args.enable_primitive_neighbor_exploration),
            "artifact_path": str(args.primitive_neighborhood_artifact_path),
            "artifact_sha256": str(args.primitive_neighborhood_artifact_sha256),
            "neighbor_top_k": int(args.primitive_neighbor_top_k),
            "neighbor_radius": (
                None
                if float(args.primitive_neighbor_radius) < 0.0
                else float(args.primitive_neighbor_radius)
            ),
            "local_global_mix": float(args.primitive_local_global_mix),
        },
        "default_enabled": False,
    }


def _training_contract(args) -> Dict[str, Any]:
    return build_awac_training_contract(
        phase=str(args.phase),
        gamma=float(args.gamma),
        tau=float(args.tau),
        max_primitive_steps=int(args.max_steps),
        calibration=_calibration_config(args).as_dict(),
    )


def _load_bc_checkpoint(path: Path, *, torch) -> Dict:
    payload = load_torch(path, torch=torch, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("BC checkpoint must be a mapping")
    if not isinstance(payload.get("model_state_dict"), Mapping):
        raise ValueError("BC checkpoint lacks model_state_dict")
    return dict(payload)


def validate_bc_checkpoint_for_awac(
    checkpoint: Mapping[str, Any], *, mpl_contract_sha256: str
) -> Dict[str, Any]:
    """Strictly validate the deployable BC V2 handoff used by AWAC."""
    if not str(mpl_contract_sha256):
        raise ValueError("MPL contract SHA is required for the AWAC handoff")
    fields = validate_awac_bc_checkpoint(
        checkpoint,
        expected_mpl_contract_sha256=str(mpl_contract_sha256),
        expected_task_contract_sha256=task_contract_sha256(int(checkpoint.get("max_primitive_steps", DEFAULT_MAX_PRIMITIVE_STEPS))),
    )
    if int(fields["vec_dim"]) != POLICY_VECTOR_DIM or int(fields["num_actions"]) != NUM_ACTIONS:
        raise ValueError("BC/AWAC policy dimensions are incompatible")
    if int(fields["depth_history_frames"]) != 1:
        raise ValueError("formal AWAC requires one depth history frame")
    return fields


def build_learner(*, torch, nn, device, bc_checkpoint: Mapping[str, Any], config: AWACOptimizationConfig):
    depth_channels = int(bc_checkpoint.get("depth_history_frames", 1))
    return DiscreteAWACLearner(
        torch=torch,
        nn=nn,
        device=device,
        bc_state_dict=dict(bc_checkpoint["model_state_dict"]),
        depth_channels=depth_channels,
        config=config,
    )


def _split_manifest_sha256(path_value: str) -> str:
    value = str(path_value).strip()
    return file_sha256(Path(value).expanduser().resolve()) if value else "0" * 64


_STANDARD_HANDOFF_CRITIC_LR_OVERRIDE_PATHS = frozenset(
    {
        "optimization_config.critic_depth_lr",
        "optimization_config.critic_vector_lr",
        "optimization_config.critic_head_lr",
    }
)

_STANDARD_HANDOFF_INNOVATION_OVERRIDE_PATHS = frozenset(
    {
        "optimization_config.enable_twin_q_confidence",
        "optimization_config.enable_adaptive_bc_kl",
        "optimization_config.enable_primitive_neighbor_exploration",
        "optimization_config.confidence_margin_gain",
        "optimization_config.confidence_disagreement_gain",
        "optimization_config.adaptive_bc_kl_beta_min",
        "optimization_config.adaptive_bc_kl_beta_max",
        "optimization_config.primitive_neighborhood_artifact_path",
        "optimization_config.primitive_neighborhood_artifact_sha256",
        "optimization_config.primitive_neighbor_top_k",
        "optimization_config.primitive_neighbor_radius",
        "optimization_config.primitive_local_global_mix",
    }
)


def _standard_handoff_config_identity_is_valid(
    report: Mapping[str, Any],
) -> bool:
    """Keep the Calibration handoff fail-closed while allowing critic-LR trials.

    Calibration PASS is the source of the model/replay handoff, but a Standard
    AWAC controlled experiment may intentionally override the three Critic
    optimizer learning rates.  All other training-semantic or unknown config
    differences remain rejected.  The override is deliberately path-scoped so
    this cannot become a general handoff config bypass.
    """

    for difference in report.get("differences", ()):
        if not isinstance(difference, Mapping):
            return False
        classification = str(difference.get("classification", ""))
        path = str(difference.get("path", ""))
        if (
            classification == TRAINING_SEMANTIC
            and path in _STANDARD_HANDOFF_CRITIC_LR_OVERRIDE_PATHS
        ):
            calibration_value = difference.get("calibration")
            handoff_value = difference.get("handoff")
            if calibration_value == "<missing>" or handoff_value == "<missing>":
                return False
            try:
                if not math.isfinite(float(calibration_value)):
                    return False
                if not math.isfinite(float(handoff_value)):
                    return False
                if float(handoff_value) <= 0.0:
                    return False
            except (TypeError, ValueError):
                return False
            continue
        if path in _STANDARD_HANDOFF_INNOVATION_OVERRIDE_PATHS:
            continue
        if classification in {TRAINING_SEMANTIC, UNKNOWN}:
            return False
    return True


def _load_resume_checkpoint(
    path: Path,
    *,
    torch,
    bc_checkpoint: Mapping[str, Any],
    bc_checkpoint_path: Path,
    args: argparse.Namespace,
    mpl_sha256: str,
    resolved_training_config: Mapping[str, Any],
    expected_replay_identity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load only a checkpoint bound to the current BC, config, and contracts."""
    if str(args.phase) == AWAC_PHASE_CRITIC_CALIBRATION:
        payload = load_committed_calibration_resume_checkpoint(
            path, torch=torch, map_location="cpu"
        )
    else:
        # The first Standard invocation consumes a committed Calibration PASS;
        # later invocations consume a committed Standard checkpoint.  Both are
        # selected by the same transaction marker.  Select the loader from the
        # committed kind before loading model tensors; a malformed Standard
        # checkpoint must fail closed instead of being reinterpreted as a
        # Calibration artifact.
        if committed_checkpoint_kind(path) == "checkpoint_calibration_pass":
            payload = load_committed_calibration_resume_checkpoint(
                path, torch=torch, map_location="cpu"
            )
        else:
            payload = load_committed_standard_resume_checkpoint(
                path, torch=torch, map_location="cpu"
            )
    expected_bc_sha256 = file_sha256(bc_checkpoint_path)
    expected_bc_actor_sha256 = actor_state_sha256(bc_checkpoint)
    if str(args.phase) == AWAC_PHASE_STANDARD_TRAINING:
        if payload.get("phase") == AWAC_PHASE_STANDARD_TRAINING:
            validate_standard_awac_checkpoint_payload(payload)
            if expected_replay_identity is not None:
                received = payload.get("replay_identity")
                if not isinstance(received, Mapping):
                    raise Phase1ReadinessError(
                        "Standard AWAC resume replay identity is missing"
                    )
                for field, expected in expected_replay_identity.items():
                    if received.get(field) != expected:
                        raise Phase1ReadinessError(
                            "Standard AWAC resume replay identity {} mismatch".format(
                                field
                            )
                        )
            return payload
        config_report = compare_awac_config_identity(
            project_awac_training_config(payload["resolved_training_config"]),
            project_awac_training_config(resolved_training_config),
        )
        if (
            not _standard_handoff_config_identity_is_valid(config_report)
            or config_report["expected_phase_diff_count"] != 1
            or config_report["expected_actor_enable_diff_count"] < 1
        ):
            raise Phase1ReadinessError(
                "Standard AWAC handoff has unexpected resolved training config differences: {}".format(
                    json.dumps(config_report, sort_keys=True)
                )
            )
        validate_standard_awac_handoff_payload(
            payload,
            expected_bc_checkpoint_sha256=expected_bc_sha256,
            expected_bc_reference_fingerprint=expected_bc_actor_sha256,
            expected_mpl_contract_sha256=str(mpl_sha256),
            expected_task_contract_sha256=task_contract_sha256(int(args.max_steps)),
            expected_replay_identity=expected_replay_identity,
        )
        return payload
    checks = {
        "source_bc_checkpoint_sha256": expected_bc_sha256,
        "mpl_contract_sha256": str(mpl_sha256),
        "observation_contract": EXPECTED_OBSERVATION_CONTRACT,
        "observation_source": EXPECTED_OBSERVATION_SOURCE,
        "task_contract_sha256": task_contract_sha256(int(args.max_steps)),
        "bc_reference_fingerprint": expected_bc_actor_sha256,
        "training_contract_sha256": str(
            resolved_training_config["training_contract_sha256"]
        ),
    }
    if str(args.phase) != AWAC_PHASE_CRITIC_CALIBRATION:
        checks["split_manifest_sha256"] = _split_manifest_sha256(args.split_manifest)
        checks["resolved_training_config_sha256"] = canonical_sha256(
            resolved_training_config
        )
    for field, expected in checks.items():
        if payload.get(field) != expected:
            raise ValueError(
                "AWAC resume {} mismatch: received={} expected={}".format(
                    field, payload.get(field), expected
                )
            )
    if str(args.phase) == AWAC_PHASE_CRITIC_CALIBRATION:
        if payload.get("phase") != AWAC_PHASE_CRITIC_CALIBRATION:
            raise ValueError("AWAC resume calibration phase mismatch")
        if payload.get("actor_update_enabled") is not False:
            raise ValueError("AWAC resume calibration Actor updates are enabled")
    return payload


def _load_worker_specs(
    path: Path,
    *,
    worker_count: Optional[int] = None,
    reliable_v4: bool = False,
    max_steps: int = DEFAULT_MAX_PRIMITIVE_STEPS,
    max_sensor_skew_ms: float = 80.0,
    depth_mask_collision_radius: float = 0.40,
    depth_mask_slack: float = 0.08,
    depth_mask_sample_stride: int = 4,
    depth_mask_max_patch_radius_px: int = 14,
):
    """Read serializable worker specs without importing ROS in the parent.

    ``worker_count`` is intentionally applied here, at the owner boundary,
    so the number requested by ``--env-workers`` is the number actually handed
    to ``ParallelEnvPool``.  A reliable-v4 run also gets the same depth-mask
    and continuous-execution EnvConfig used by formal evaluation.
    """
    from planning.runtime.parallel_env import EnvWorkerSpec
    from planning.runtime.reliable_training import build_reliable_v4_runtime_config

    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    workers = payload.get("workers") if isinstance(payload, Mapping) else None
    if not isinstance(workers, list) or not workers:
        raise ValueError("worker spec file must contain a non-empty workers list")
    workers = sorted(workers, key=lambda value: int(value.get("worker_id", -1)))
    requested = len(workers) if worker_count is None else int(worker_count)
    if requested <= 0:
        raise ValueError("worker_count must be positive")
    if requested > len(workers):
        raise ValueError(
            "worker spec file contains {} workers, requested {}".format(
                len(workers), requested
            )
        )
    workers = workers[:requested]
    specs = []
    for expected_worker_id, worker in enumerate(workers):
        if not isinstance(worker, Mapping):
            raise ValueError("worker spec must be a mapping")
        if int(worker["worker_id"]) != expected_worker_id:
            raise ValueError("worker specs must start at contiguous worker_id 0")
        env_kwargs = dict(worker.get("env_kwargs", {}))
        if bool(reliable_v4):
            endpoint_fields = {
                "runtime_instance_id": worker.get("runtime_instance_id", ""),
                "command_endpoint": worker.get(
                    "python_command_endpoint", worker.get("reliable_command_endpoint", "")
                ),
                "result_endpoint": worker.get(
                    "python_result_endpoint", worker.get("reliable_result_endpoint", "")
                ),
                "snapshot_endpoint": worker.get(
                    "python_snapshot_endpoint", worker.get("reliable_snapshot_endpoint", "")
                ),
            }
            env_kwargs["reliable_v4_runtime"] = build_reliable_v4_runtime_config(
                runtime_instance_id=str(endpoint_fields["runtime_instance_id"]),
                command_endpoint=str(endpoint_fields["command_endpoint"]),
                result_endpoint=str(endpoint_fields["result_endpoint"]),
                snapshot_endpoint=str(endpoint_fields["snapshot_endpoint"]),
                timeout_s=30.0,
            )
            if "config" in env_kwargs:
                raise ValueError(
                    "reliable-v4 worker specs must use serializable config_kwargs"
                )
            raw_config_kwargs = env_kwargs.get("config_kwargs", {})
            if raw_config_kwargs is None:
                raw_config_kwargs = {}
            if not isinstance(raw_config_kwargs, Mapping):
                raise ValueError("worker config_kwargs must be a mapping")
            config_kwargs = dict(raw_config_kwargs)
            config_kwargs.update(
                {
                    "max_episode_steps": int(max_steps),
                    "stop_at_primitive_end": False,
                    "use_depth_collision_mask": True,
                    "enforce_sensor_sync": True,
                    "max_sensor_skew_s": float(max_sensor_skew_ms) / 1000.0,
                    "depth_mask_collision_radius_m": float(depth_mask_collision_radius),
                    "depth_mask_slack_m": float(depth_mask_slack),
                    "depth_mask_path_sample_stride": int(depth_mask_sample_stride),
                    "depth_mask_max_patch_radius_px": int(depth_mask_max_patch_radius_px),
                }
            )
            env_kwargs["config_kwargs"] = config_kwargs
        extra_environment = {
            str(key): str(value)
            for key, value in dict(worker.get("extra_environment", {})).items()
        }
        if bool(reliable_v4):
            for key, value in {
                "PLANNING_RUNTIME_INSTANCE_ID": str(
                    worker.get("runtime_instance_id", "")
                ),
                "PLANNING_TRAINING_RUN_ID": str(
                    worker.get("training_run_id", "legacy-unspecified")
                ),
                "PLANNING_RUNTIME_LAUNCH_NONCE": str(
                    worker.get("runtime_launch_nonce", "legacy-unspecified")
                ),
            }.items():
                if not value:
                    raise ValueError("worker spec is missing {}".format(key))
                if key in extra_environment and extra_environment[key] != value:
                    raise ValueError("worker spec {} identity mismatch".format(key))
                extra_environment[key] = value
        specs.append(
            EnvWorkerSpec(
                worker_id=expected_worker_id,
                ros_master_uri=str(worker["ros_master_uri"]),
                ros_home=str(worker["ros_home"]),
                env_kwargs=env_kwargs,
                extra_environment=extra_environment,
            )
        )
    return tuple(specs)


def build_runtime_pool(
    worker_spec_file: Path,
    *,
    worker_count: Optional[int] = None,
    reliable_v4: bool = False,
    max_steps: int = DEFAULT_MAX_PRIMITIVE_STEPS,
    max_sensor_skew_ms: float = 80.0,
    depth_mask_collision_radius: float = 0.40,
    depth_mask_slack: float = 0.08,
    depth_mask_sample_stride: int = 4,
    depth_mask_max_patch_radius_px: int = 14,
    startup_timeout_s: float = 30.0,
    request_timeout_s: float = 30.0,
    pool_factory=None,
):
    """Construct the shared runtime owner on explicit caller request."""
    from planning.runtime.parallel_env import ParallelEnvPool

    specs = _load_worker_specs(
        Path(worker_spec_file),
        worker_count=worker_count,
        reliable_v4=bool(reliable_v4),
        max_steps=int(max_steps),
        max_sensor_skew_ms=float(max_sensor_skew_ms),
        depth_mask_collision_radius=float(depth_mask_collision_radius),
        depth_mask_slack=float(depth_mask_slack),
        depth_mask_sample_stride=int(depth_mask_sample_stride),
        depth_mask_max_patch_radius_px=int(depth_mask_max_patch_radius_px),
    )
    constructor = pool_factory or ParallelEnvPool
    return constructor(
        specs,
        startup_timeout_s=float(startup_timeout_s),
        request_timeout_s=float(request_timeout_s),
    )


def build_managed_runtime_pool(
    worker_spec_file: Path,
    *,
    worker_count: int,
    output_dir: Path,
    task_contract_sha256: str,
    mpl_contract_sha256: str,
    max_steps: int = DEFAULT_MAX_PRIMITIVE_STEPS,
    startup_timeout_s: float = 60.0,
    runtime_factory=None,
):
    """Construct the one shared owner for a calibration runtime.

    The canonical serialized ``WorkerRuntimeSpec`` is loaded here, rather
    than re-derived from raw JSON or from a second set of port defaults.  The
    returned owner is not started until the caller has created the complete
    calibration run boundary.
    """

    from planning.common.paths import planning_workspace_root
    from planning.runtime.managed_runtime import ManagedRuntimePool
    from planning.runtime.ports import validate_worker_runtime_specs
    from planning.runtime.worker import load_worker_runtime_spec_file

    requested = int(worker_count)
    if requested <= 0:
        raise ValueError("worker_count must be positive")
    all_specs = load_worker_runtime_spec_file(
        Path(worker_spec_file).expanduser().resolve()
    )
    if requested > len(all_specs):
        raise ValueError(
            "worker spec file contains {} workers, requested {}".format(
                len(all_specs), requested
            )
        )
    specs = validate_worker_runtime_specs(tuple(all_specs[:requested]))
    constructor = runtime_factory or ManagedRuntimePool
    return constructor(
        specs,
        workspace=planning_workspace_root(),
        log_dir=Path(output_dir).expanduser().resolve() / "runtime",
        task_contract_sha256=str(task_contract_sha256),
        mpl_contract_sha256=str(mpl_contract_sha256),
        observation_contract=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        max_steps=int(max_steps),
        startup_timeout_s=float(startup_timeout_s),
        reliable_v4=True,
    )


def start_managed_calibration_runtime(
    worker_spec_file: Path,
    *,
    worker_count: int,
    output_dir: Path,
    task_contract_sha256: str,
    mpl_contract_sha256: str,
    max_steps: int = DEFAULT_MAX_PRIMITIVE_STEPS,
    max_sensor_skew_ms: float = 80.0,
    depth_mask_collision_radius: float = 0.40,
    depth_mask_slack: float = 0.08,
    depth_mask_sample_stride: int = 4,
    depth_mask_max_patch_radius_px: int = 14,
    startup_timeout_s: float = 30.0,
    request_timeout_s: float = 30.0,
    runtime_factory=None,
    pool_factory=None,
):
    """Start external runtimes before constructing Python env workers."""

    managed_runtime = build_managed_runtime_pool(
        Path(worker_spec_file),
        worker_count=int(worker_count),
        output_dir=Path(output_dir),
        task_contract_sha256=str(task_contract_sha256),
        mpl_contract_sha256=str(mpl_contract_sha256),
        max_steps=int(max_steps),
        startup_timeout_s=float(startup_timeout_s),
        runtime_factory=runtime_factory,
    )
    try:
        identity = managed_runtime.start()
        pool = build_runtime_pool(
            Path(worker_spec_file),
            worker_count=int(worker_count),
            reliable_v4=True,
            max_steps=int(max_steps),
            max_sensor_skew_ms=float(max_sensor_skew_ms),
            depth_mask_collision_radius=float(depth_mask_collision_radius),
            depth_mask_slack=float(depth_mask_slack),
            depth_mask_sample_stride=int(depth_mask_sample_stride),
            depth_mask_max_patch_radius_px=int(depth_mask_max_patch_radius_px),
            startup_timeout_s=float(startup_timeout_s),
            request_timeout_s=float(request_timeout_s),
            pool_factory=pool_factory,
        )
        return managed_runtime, pool, identity
    except Exception:
        close = getattr(managed_runtime, "close", None)
        if callable(close):
            close()
        raise


def _resolved_training_config(args, *, source_bc_sha256: str, mpl_sha256: str) -> Dict[str, Any]:
    training_contract = _training_contract(args)
    resolved = {
        "contract_id": AWAC_TRAINING_CONFIG_CONTRACT_ID,
        "algorithm_id": AWAC_ALGORITHM_ID,
        "phase": str(args.phase),
        "training_contract": training_contract,
        "training_contract_sha256": awac_training_contract_sha256(training_contract),
        "bc_checkpoint_path": Path(str(args.bc_checkpoint)).name,
        "source_bc_checkpoint_sha256": str(source_bc_sha256),
        "bc_checkpoint_sha256": str(source_bc_sha256),
        "observation_contract": EXPECTED_OBSERVATION_CONTRACT,
        "observation_source": EXPECTED_OBSERVATION_SOURCE,
        "reliable_execution_enabled": bool(args.reliable_v4),
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": task_contract_sha256(int(args.max_steps)),
        "max_primitive_steps": int(args.max_steps),
        "mpl_contract_sha256": str(mpl_sha256),
        "vec_dim": POLICY_VECTOR_DIM,
        "num_actions": NUM_ACTIONS,
        "depth_history_frames": 1,
        "initial_prev_action": INITIAL_PREV_ACTION,
        "policy_runtime_contract_id": POLICY_RUNTIME_CONTRACT_ID,
        "reward_contract_id": REWARD_CONTRACT_ID,
        "reward_contract_sha256": reward_contract_sha256(),
        "safety_mask": "depth",
        "execution_mode": "continuous",
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "cpu_threads": int(args.cpu_threads),
        "device": str(args.device),
        "env_workers": int(args.env_workers),
        "online_env_steps": (
            None if args.online_env_steps is None else int(args.online_env_steps)
        ),
        "online_env_steps_budget": int(_online_env_steps(args)),
        "online_budget_counter_owner": "phase_local_online_environment_steps",
        "behavior_policy_action_mode": "masked_categorical",
        "behavior_policy_temperature": float(args.warmup_temperature),
        "train_index": str(Path(str(args.train_index)).expanduser().resolve())
        if str(args.train_index).strip()
        else "",
        "fresh_online_calibration": bool(
            str(args.phase) == AWAC_PHASE_CRITIC_CALIBRATION
            and str(args.train_index).strip()
            and not str(args.resume_checkpoint).strip()
        ),
        "parallel_env_owner": "planning.runtime.parallel_env",
        "replay_capacity": int(args.replay_capacity),
        "batch_size": int(args.batch_size),
        "learning_starts": int(args.learning_starts),
        "actor_learning_starts": int(args.actor_learning_starts),
        "critic_burnin_updates": int(args.critic_burnin_updates),
        "actor_update_interval": int(args.actor_update_interval),
        "updates_per_step": float(args.updates_per_step),
        "calibration_split_path": Path(str(args.calibration_split)).name
        if str(args.calibration_split).strip()
        else "",
        "calibration_gate": _calibration_config(args).as_dict(),
        "calibration_safety_cap": _calibration_safety_cap(args).as_dict(),
        "innovation_config": _innovation_config(args),
        "optimization_config": vars(_optimization_config(args)),
    }
    if str(args.phase) == AWAC_PHASE_STANDARD_TRAINING:
        phase1_contract = build_phase1_training_contract(
            actor_depth_lr=float(args.actor_depth_lr),
            critic_depth_lr=float(args.critic_depth_lr),
            online_transition_cap=int(args.phase1_online_transition_cap),
            calibration_safety_cap=_calibration_safety_cap(args),
        )
        resolved["phase1_training_contract"] = phase1_contract
        resolved["phase1_training_contract_sha256"] = phase1_training_contract_sha256(
            phase1_contract
        )
    return resolved


def build_checkpoint_payload(
    *,
    learner: DiscreteAWACLearner,
    args: argparse.Namespace,
    bc_checkpoint: Mapping[str, Any],
    bc_checkpoint_path: Path,
    mpl_sha256: str,
    replay_size: int,
    split_manifest_sha256: str,
) -> Dict[str, Any]:
    resolved = _resolved_training_config(
        args,
        source_bc_sha256=file_sha256(bc_checkpoint_path),
        mpl_sha256=mpl_sha256,
    )
    source_files = (
        "python/planning/awac/checkpoint.py",
        "python/planning/awac/calibration.py",
        "python/planning/awac/contract.py",
        "python/planning/awac/interaction.py",
        "python/planning/awac/learner.py",
        "python/planning/awac/model.py",
        "python/planning/awac/optimization.py",
        "python/planning/awac/phase1.py",
        "python/planning/awac/replay.py",
        "python/planning/awac/trainer.py",
    )
    source_sha = awac_source_sha256(Path(__file__).resolve().parents[3], source_files)
    payload = {
        **build_awac_checkpoint_identity(
            max_primitive_steps=int(args.max_steps)
        ),
        "mpl_contract_sha256": str(mpl_sha256),
        "resolved_training_config": resolved,
        "resolved_training_config_sha256": canonical_sha256(resolved),
        "training_contract": dict(resolved["training_contract"]),
        "training_contract_sha256": str(resolved["training_contract_sha256"]),
        "source_code_sha256": source_sha,
        "split_manifest_sha256": str(split_manifest_sha256),
        "source_bc_checkpoint_sha256": file_sha256(bc_checkpoint_path),
        "bc_reference_fingerprint": actor_state_sha256(bc_checkpoint),
        "global_step": int(getattr(learner, "update_step", 0)),
        "replay_size": int(replay_size),
        "update_step": int(getattr(learner, "update_step", 0)),
        "phase": str(args.phase),
        "actor_update_enabled": str(args.phase) != AWAC_PHASE_CRITIC_CALIBRATION,
        "reward_scale": float(args.reward_scale),
        "reward_scale_owner": "learner_bellman_target",
    }
    payload.update(learner.state_dict())
    if str(args.phase) == AWAC_PHASE_STANDARD_TRAINING:
        phase1_contract = build_phase1_training_contract(
            actor_depth_lr=float(args.actor_depth_lr),
            critic_depth_lr=float(args.critic_depth_lr),
            online_transition_cap=int(args.phase1_online_transition_cap),
            calibration_safety_cap=_calibration_safety_cap(args),
        )
        payload["phase1_training_contract"] = phase1_contract
        payload["phase1_training_contract_sha256"] = phase1_training_contract_sha256(
            phase1_contract
        )
    return payload


def build_phase1_state_machine(*, online_transition_cap: int = PHASE1_DEFAULT_ONLINE_TRANSITION_CAP):
    """Construct the explicit Phase 1 orchestration owner."""

    return Phase1StateMachine(online_transition_cap=int(online_transition_cap))


def build_phase1_calibration_controller(
    *,
    max_transitions: int = 30_000,
    max_episodes: int = 1_000,
) -> Phase1CalibrationController:
    """Construct the bounded calibration gate controller."""

    return Phase1CalibrationController(
        safety_cap=Phase1CalibrationSafetyCap(
            max_transitions=int(max_transitions),
            max_episodes=int(max_episodes),
        )
    )


def _synthetic_batch(torch, *, batch_size: int, depth_channels: int):
    mask = torch.ones((batch_size, NUM_ACTIONS), dtype=torch.bool)
    return {
        "depth": torch.rand((batch_size, depth_channels, 32, 32)),
        "vector": torch.randn((batch_size, POLICY_VECTOR_DIM)),
        "action_mask": mask,
        "action": torch.arange(batch_size, dtype=torch.long) % NUM_ACTIONS,
        "reward": torch.zeros((batch_size,), dtype=torch.float32),
        "next_depth": torch.rand((batch_size, depth_channels, 32, 32)),
        "next_vector": torch.randn((batch_size, POLICY_VECTOR_DIM)),
        "next_action_mask": mask.clone(),
        "done": torch.zeros((batch_size,), dtype=torch.float32),
        "behavior_source": torch.zeros((batch_size,), dtype=torch.long),
    }


def _load_calibration_split(path_value: str, *, fallback: Mapping[str, Any]) -> Dict[str, Any]:
    """Load and validate an episode-level calibration split identity."""

    value = str(path_value).strip()
    if not value:
        split = dict(fallback)
    else:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("calibration split does not exist: {}".format(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("calibration split must contain a JSON object")
        split = dict(payload)
        split["manifest_path"] = str(path)
        split["manifest_sha256"] = file_sha256(path)
    train_ids = split.get("train_episode_ids")
    holdout_ids = split.get("holdout_episode_ids")
    if not isinstance(train_ids, list) or not isinstance(holdout_ids, list):
        raise ValueError("calibration split must contain train/holdout episode IDs")
    if not train_ids or not holdout_ids:
        raise ValueError("calibration split train and holdout must be non-empty")
    train_ids = [str(value) for value in train_ids]
    holdout_ids = [str(value) for value in holdout_ids]
    if len(set(train_ids)) != len(train_ids) or len(set(holdout_ids)) != len(holdout_ids):
        raise ValueError("calibration split contains duplicate episode IDs")
    if set(train_ids).intersection(holdout_ids):
        raise ValueError("calibration train and holdout episode IDs overlap")
    split["train_episode_ids"] = train_ids
    split["holdout_episode_ids"] = holdout_ids
    split.setdefault("contract_id", "awac_calibration_episode_split_v1")
    return split


def _resolve_online_calibration_split(
    *,
    args: argparse.Namespace,
    missions,
    output_dir: Path,
    resume: bool,
) -> Dict[str, Any]:
    """Resolve a persisted or deterministic episode-level runtime split."""

    value = str(args.calibration_split).strip()
    if value:
        split = _load_calibration_split(value, fallback={})
    else:
        generated = Path(output_dir).expanduser().resolve() / "calibration_split.json"
        if bool(resume):
            if not generated.is_file():
                raise FileNotFoundError(
                    "calibration resume split is missing: {}".format(generated)
                )
            split = _load_calibration_split(str(generated), fallback={})
        else:
            split = build_calibration_split(
                missions,
                seed=int(args.seed),
                holdout_fraction=float(args.calibration_holdout_fraction),
            )
            generated.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(generated, split, trailing_newline=True)
            split["manifest_path"] = str(generated)
            split["manifest_sha256"] = file_sha256(generated)
    return split


def _worker_spec_identity(path: Path, worker_count: int) -> Dict[str, Any]:
    """Bind a calibration run to the selected serializable worker topology."""

    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    workers = payload.get("workers") if isinstance(payload, Mapping) else None
    if not isinstance(workers, list) or len(workers) < int(worker_count):
        raise ValueError("worker spec file does not contain the requested worker count")
    ordered = sorted(workers, key=lambda value: int(value.get("worker_id", -1)))[: int(worker_count)]
    if [int(value.get("worker_id", -1)) for value in ordered] != list(range(int(worker_count))):
        raise ValueError("worker spec identities are not contiguous")
    runtime_ids = [str(value.get("runtime_instance_id", "")).strip() for value in ordered]
    if any(not value for value in runtime_ids) or len(set(runtime_ids)) != len(runtime_ids):
        raise ValueError("worker spec runtime identities must be present and unique")
    return {
        "worker_spec_path": str(resolved),
        "worker_spec_sha256": file_sha256(resolved),
        "worker_count": int(worker_count),
        "runtime_instance_ids": runtime_ids,
        "runtime_launch_nonces": [
            str(value.get("runtime_launch_nonce", "")) for value in ordered
        ],
        "training_run_ids": [str(value.get("training_run_id", "")) for value in ordered],
        "unity_sha256": [str(value.get("unity_sha256", "")) for value in ordered],
        "bridge_sha256": [str(value.get("bridge_sha256", "")) for value in ordered],
    }


def _calibration_replay_identity(replay, *, resolved: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the replay metadata identity after an explicit flush."""

    replay.flush()
    metadata_path = Path(replay.directory).expanduser().resolve() / "metadata.json"
    return {
        "run_identity": str(replay.metadata["run_identity"]),
        "mission_source_sha256": str(
            replay.metadata.get("mission_source_sha256", "")
        ),
        "mission_index_sha256": str(
            replay.metadata.get("mission_index_sha256", "")
        ),
        "replay_size": int(replay.size),
        "replay_metadata_sha256": file_sha256(metadata_path),
        "replay_contract_sha256": str(resolved["training_contract"]["replay_contract_sha256"]),
        "behavior_source_phase": AWAC_PHASE_CRITIC_CALIBRATION,
    }


def _standard_replay_identity(replay) -> Dict[str, Any]:
    """Return the current immutable identity of a Standard replay clone."""

    return calibration_replay_identity(replay)


def _standard_online_run_contract(
    *,
    run_identity: str,
    args: argparse.Namespace,
    resolved: Mapping[str, Any],
    mission_source_identity: Mapping[str, Any],
    source_replay_identity: Mapping[str, Any],
    source_calibration_checkpoint_sha256: str,
    worker_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the run-bound identity for an independent Standard replay."""

    return {
        "schema_id": "awac_standard_online_run_v1",
        "phase": AWAC_PHASE_STANDARD_TRAINING,
        "run_identity": str(run_identity),
        "online_env_steps": int(_online_env_steps(args)),
        "online_budget_counter_owner": "phase_local_online_environment_steps",
        "mission_source_identity": dict(mission_source_identity),
        "source_replay_identity": dict(source_replay_identity),
        "source_calibration_checkpoint_sha256": str(
            source_calibration_checkpoint_sha256
        ),
        "worker_identity": dict(worker_identity),
        "training_contract_sha256": str(resolved["training_contract_sha256"]),
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "replay_contract_sha256": str(
            resolved["training_contract"]["replay_contract_sha256"]
        ),
        "reward_contract_sha256": reward_contract_sha256(),
        "actor_update_enabled": True,
    }


def _create_standard_online_replay(
    *,
    source_replay: AWACReplayBuffer,
    output_dir: Path,
    args: argparse.Namespace,
    resolved: Mapping[str, Any],
    mission_source_identity: Mapping[str, Any],
    source_replay_identity: Mapping[str, Any],
    source_calibration_checkpoint_sha256: str,
    worker_identity: Mapping[str, Any],
) -> AWACReplayBuffer:
    """Clone the Calibration replay into the new Standard run boundary."""

    run_identity = Path(output_dir).expanduser().resolve().name
    contract = _standard_online_run_contract(
        run_identity=run_identity,
        args=args,
        resolved=resolved,
        mission_source_identity=mission_source_identity,
        source_replay_identity=source_replay_identity,
        source_calibration_checkpoint_sha256=source_calibration_checkpoint_sha256,
        worker_identity=worker_identity,
    )
    replay_dir = Path(output_dir).expanduser().resolve() / "replay"
    if replay_dir.exists() and any(replay_dir.iterdir()):
        raise FileExistsError(
            "Standard AWAC replay directory is not empty: {}".format(replay_dir)
        )
    return AWACReplayBuffer.clone_from(
        source_replay,
        replay_dir,
        run_identity=run_identity,
        run_contract_sha256=canonical_sha256(contract),
        training_config_sha256=str(resolved["training_contract_sha256"]),
        behavior_source_phase=AWAC_PHASE_STANDARD_TRAINING,
        extra_metadata={
            "standard_online_runtime_schema_id": STANDARD_ONLINE_RUNTIME_SCHEMA_ID,
            "online_budget_counter_owner": "phase_local_online_environment_steps",
            "source_calibration_checkpoint_sha256": str(
                source_calibration_checkpoint_sha256
            ),
            "source_replay_identity": dict(source_replay_identity),
            "mission_source_identity": dict(mission_source_identity),
            "worker_identity": dict(worker_identity),
        },
    )


def _standard_checkpoint_source_sha256() -> str:
    root = Path(__file__).resolve().parents[3]
    source_files = (
        "python/planning/awac/calibration.py",
        "python/planning/awac/calibration_runtime.py",
        "python/planning/awac/checkpoint.py",
        "python/planning/awac/contract.py",
        "python/planning/awac/interaction.py",
        "python/planning/awac/learner.py",
        "python/planning/awac/model.py",
        "python/planning/awac/online_runtime.py",
        "python/planning/awac/optimization.py",
        "python/planning/awac/phase1.py",
        "python/planning/awac/replay.py",
        "python/planning/awac/trainer.py",
    )
    return awac_source_sha256(root, source_files)


def _build_standard_checkpoint_payload(
    *,
    learner: DiscreteAWACLearner,
    args: argparse.Namespace,
    bc_checkpoint: Mapping[str, Any],
    bc_checkpoint_path: Path,
    mpl_sha: str,
    resolved: Mapping[str, Any],
    replay_identity: Mapping[str, Any],
    source_replay_identity: Mapping[str, Any],
    source_calibration_checkpoint_sha256: str,
    mission_source_identity: Mapping[str, Any],
    worker_identity: Mapping[str, Any],
    managed_runtime_identity: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build one fully bound Standard online checkpoint generation."""

    if not isinstance(snapshot, Mapping):
        raise TypeError("Standard AWAC checkpoint snapshot must be a mapping")
    state = learner.state_dict()
    progress = snapshot.get("progress")
    exact_resume_state = snapshot.get("standard_online_resume_state")
    phase1_state = snapshot.get("phase1_state")
    if not isinstance(progress, Mapping) or not isinstance(exact_resume_state, Mapping):
        raise ValueError("Standard AWAC checkpoint progress is incomplete")
    if not isinstance(phase1_state, Mapping):
        raise ValueError("Standard AWAC Phase-1 state is incomplete")
    if "online_transitions_committed" not in snapshot:
        raise ValueError(
            "Standard AWAC checkpoint snapshot is missing "
            "online_transitions_committed"
        )
    if "online_transitions_committed" not in progress:
        raise ValueError(
            "Standard AWAC checkpoint progress is missing "
            "online_transitions_committed"
        )
    if int(snapshot["online_transitions_committed"]) != int(
        progress["online_transitions_committed"]
    ):
        raise ValueError(
            "Standard AWAC checkpoint committed transition counters mismatch"
        )
    resolved_copy = dict(resolved)
    payload = {
        **build_awac_checkpoint_identity(max_primitive_steps=int(args.max_steps)),
        "mpl_contract_sha256": str(mpl_sha),
        "resolved_training_config": resolved_copy,
        "resolved_training_config_sha256": canonical_sha256(resolved_copy),
        "training_contract": dict(resolved_copy["training_contract"]),
        "training_contract_sha256": str(resolved_copy["training_contract_sha256"]),
        "source_code_sha256": _standard_checkpoint_source_sha256(),
        "split_manifest_sha256": "0" * 64,
        "source_bc_checkpoint_sha256": file_sha256(bc_checkpoint_path),
        "bc_reference_fingerprint": actor_state_sha256(bc_checkpoint),
        "global_step": int(getattr(learner, "update_step", 0)),
        "replay_size": int(replay_identity.get("replay_size", 0)),
        "update_step": int(getattr(learner, "update_step", 0)),
        "phase": AWAC_PHASE_STANDARD_TRAINING,
        "actor_update_enabled": True,
        "actor_optimizer_status": "active",
        "actor_optimizer_step_count": int(
            getattr(learner, "actor_optimizer_step_count", 0)
        ),
        "critic_update_count": int(getattr(learner, "critic_update_count", 0)),
        "environment_step_count": int(snapshot.get("environment_step_count", 0)),
        "online_env_steps": int(snapshot.get("online_env_steps", 0)),
        "online_env_steps_budget": int(
            snapshot.get("online_env_steps_budget", _online_env_steps(args))
        ),
        "online_transitions_committed": int(
            snapshot["online_transitions_committed"]
        ),
        # Retain the established checkpoint field as a compatibility alias;
        # the explicit phase counter above is the canonical owner.
        "online_transition_count": int(snapshot["online_transitions_committed"]),
        "starting_replay_size": int(snapshot.get("starting_replay_size", 0)),
        "starting_replay_total_added": int(
            snapshot.get("starting_replay_total_added", 0)
        ),
        "replay_identity": dict(replay_identity),
        "source_replay_identity": dict(source_replay_identity),
        "source_calibration_checkpoint_sha256": str(
            source_calibration_checkpoint_sha256
        ),
        "replay_contract_sha256": str(resolved_copy["training_contract"]["replay_contract_sha256"]),
        "reward_scale": float(args.reward_scale),
        "reward_scale_owner": "learner_bellman_target",
        "standard_online_runtime_schema_id": STANDARD_ONLINE_RUNTIME_SCHEMA_ID,
        "standard_online_progress": dict(progress),
        "standard_online_resume_state": dict(exact_resume_state),
        "phase1_state": dict(phase1_state),
        "phase1_training_contract": dict(
            resolved_copy.get("phase1_training_contract", {})
        ),
        "phase1_training_contract_sha256": str(
            resolved_copy.get("phase1_training_contract_sha256", "")
        ),
        "runtime_identity": {
            "worker_identity": dict(worker_identity),
            "observed_runtime_instance_ids": dict(
                snapshot.get("runtime_identity", {})
            ),
            "managed_runtime_identity": dict(managed_runtime_identity),
        },
        "runtime_metrics": dict(snapshot.get("runtime_metrics", {})),
        "learner_metrics": dict(snapshot.get("learner_metrics", {})),
        "completed_episode_count": int(snapshot.get("completed_episode_count", 0)),
        "actor_state_sha256": actor_state_sha256(
            {"actor_state_dict": learner.actor.state_dict()}
        ),
        "critic1_state_sha256": actor_state_sha256(
            {"actor_state_dict": learner.critic1.state_dict()}
        ),
        "critic2_state_sha256": actor_state_sha256(
            {"actor_state_dict": learner.critic2.state_dict()}
        ),
        "mission_source_identity": dict(mission_source_identity),
        "checkpoint_paths": [],
    }
    payload.update(state)
    return payload


def _behavior_source_counts(replay: AWACReplayBuffer) -> Dict[str, int]:
    """Count the two serialized behavior sources in committed replay rows."""

    values = np.asarray(replay.arrays["behavior_source"][: int(replay.size)], dtype=np.int64)
    return {
        "BC_CALIBRATION": int(np.count_nonzero(values == int(BehaviorSource.BC_CALIBRATION))),
        "AWAC_ONLINE": int(np.count_nonzero(values == int(BehaviorSource.AWAC_ONLINE))),
    }


def _create_online_calibration_replay(
    *,
    replay_dir: Path,
    args: argparse.Namespace,
    resolved: Mapping[str, Any],
    bc_path: Path,
    mpl_sha: str,
    mission_source_identity: Mapping[str, Any],
    split: Mapping[str, Any],
    worker_identity: Mapping[str, Any],
) -> AWACReplayBuffer:
    """Create the existing mmap replay with formal online provenance."""

    run_identity = Path(replay_dir).expanduser().resolve().parent.name
    run_contract = {
        "schema_id": "awac_formal_calibration_run_v1",
        "phase": AWAC_PHASE_CRITIC_CALIBRATION,
        "run_identity": run_identity,
        "mission_source_identity": dict(mission_source_identity),
        "calibration_split": dict(split),
        "worker_identity": dict(worker_identity),
        "bc_checkpoint_sha256": file_sha256(bc_path),
        "mpl_contract_sha256": str(mpl_sha),
        "task_contract_sha256": task_contract_sha256(int(args.max_steps)),
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "reward_scale_owner": "learner_bellman_target",
    }
    return AWACReplayBuffer.create(
        Path(replay_dir).expanduser().resolve(),
        capacity=int(args.replay_capacity),
        depth_shape=(1, 90, 160),
        vector_dim=POLICY_VECTOR_DIM,
        action_dim=NUM_ACTIONS,
        training_config_sha256=str(resolved["training_contract_sha256"]),
        observation_contract=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        observation_source=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        bc_checkpoint_sha256=file_sha256(bc_path),
        task_contract_id=TASK_CONTRACT_ID,
        task_contract_sha256=task_contract_sha256(int(args.max_steps)),
        mpl_contract_sha256=str(mpl_sha),
        run_identity=run_identity,
        run_contract_sha256=canonical_sha256(run_contract),
        mission_source_sha256=str(mission_source_identity["sha256"]),
        mission_index_sha256=str(mission_source_identity["sha256"]),
        reliable_v4_transition_count=0,
        behavior_source_phase=AWAC_PHASE_CRITIC_CALIBRATION,
    )


def _online_calibration_metrics(
    *,
    producer: CalibrationReplayProducer,
    result: Mapping[str, Any],
    status: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Project producer progress into the existing checkpoint metric field."""

    state = str((status or {}).get("state", result.get("status", "PENDING")))
    if state == "BLOCKED_PENDING":
        state = "PENDING"
    metrics = {
        "state": state,
        "calibration_certification": str(
            (status or {}).get("calibration_certification", result.get("status", "PENDING"))
        ),
        "stop_reason": str(result.get("stop_reason", "")),
        "gate_history": [dict(value) for value in result.get("gate_history", [])],
        "progress": dict(result.get("progress", producer.progress)),
        "environment_step_count": int(result.get("environment_step_count", 0)),
        "completed_episode_count": int(result.get("completed_episode_count", 0)),
        "holdout_episode_count": int(result.get("holdout_episode_count", 0)),
        "holdout_transition_count": int(result.get("holdout_transition_count", 0)),
        "actor_unchanged": bool(result.get("actor_unchanged", True)),
        "behavior_policy": dict(
            result.get(
                "behavior_policy",
                getattr(producer, "behavior_policy", {}),
            )
        ),
        "behavior_rng_owner": str(
            getattr(producer, "behavior_policy", {}).get("rng_owner", "")
        ),
    }
    if status:
        metrics["latest_gate_status"] = dict(status)
    return metrics


def _runtime_recovery_diagnostics(
    producer: CalibrationReplayProducer,
    *,
    error: Optional[BaseException] = None,
    error_traceback: str = "",
) -> Dict[str, Any]:
    """Project lifecycle evidence into checkpoint/recovery diagnostics only."""

    metrics = dict(getattr(producer, "metrics", {}))
    diagnostics = {
        "final_flush_close_confirmation": {
            "replay_final_flush": str(
                metrics.get("replay_final_flush", "NOT_ATTEMPTED")
            ),
            "runtime_close": str(metrics.get("runtime_close", "NOT_ATTEMPTED")),
        },
        "runtime_failure_summary": "",
        "interrupted_save_traceback": "",
    }
    if error is not None:
        diagnostics["runtime_failure_summary"] = "{}: {}".format(
            type(error).__name__, str(error)
        )
        diagnostics["interrupted_save_traceback"] = str(error_traceback)
    return diagnostics


def _write_runtime_failure_recovery_summary(
    *,
    output_dir: Path,
    checkpoint_name: str,
    diagnostics: Mapping[str, Any],
) -> Path:
    """Persist runtime-failure evidence even when no snapshot can commit."""

    path = Path(output_dir).expanduser().resolve() / (
        "{}.runtime_failure_recovery.json".format(checkpoint_name)
    )
    payload = {
        "schema_id": "awac_calibration_runtime_failure_recovery_v1",
        "checkpoint_name": str(checkpoint_name),
        **dict(diagnostics),
    }
    write_calibration_transaction_json(path, payload)
    return path


def _standard_online_actor_fingerprints(runner) -> Dict[str, str]:
    """Capture Actor identities without masking a primary runtime failure."""

    before = str(getattr(runner, "_actor_before", "")) if runner is not None else ""
    after = ""
    if runner is not None:
        fingerprint = getattr(runner, "_actor_fingerprint", None)
        if callable(fingerprint):
            try:
                after = str(fingerprint())
            except BaseException:
                after = ""
    return {
        "actor_state_before_sha256": before,
        "actor_state_after_sha256": after,
        "actor_changed": bool(before and after and before != after),
    }


def _standard_online_replay_committed_state(runner) -> Dict[str, Any]:
    """Persist scalar replay state available after a failed finalization."""

    if runner is None:
        return {}
    replay = getattr(runner, "replay", None)
    if replay is None:
        return {}
    metadata = getattr(replay, "metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    size = int(getattr(replay, "size", metadata.get("size", 0)))
    return {
        "directory": str(getattr(replay, "directory", "")),
        "size": size,
        "position": int(metadata.get("position", size)),
        "total_added": int(
            getattr(replay, "total_added", metadata.get("total_added", size))
        ),
        "behavior_source_phase": str(
            metadata.get("behavior_source_phase", "")
        ),
        "replay_contract_sha256": str(
            metadata.get("replay_contract_sha256", "")
        ),
    }


def _standard_online_failure_payload(
    *,
    runner,
    error: BaseException,
    error_traceback: str,
    source_checkpoint_sha256: str,
    mission_source_identity: Mapping[str, Any],
    source_replay_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    checkpoint_save_reason: str,
    checkpoint_final_commit: str,
    checkpoint_paths,
) -> Dict[str, Any]:
    """Build durable fail-stop evidence for Standard finalization errors."""

    try:
        progress = dict(runner.progress) if runner is not None else {}
    except BaseException:
        progress = {}
    metrics = dict(getattr(runner, "metrics", {})) if runner is not None else {}
    actor = _standard_online_actor_fingerprints(runner)
    return {
        "schema_id": "awac_standard_online_runtime_failure_v1",
        "phase": AWAC_PHASE_STANDARD_TRAINING,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": str(error_traceback),
        "source_calibration_checkpoint_sha256": str(source_checkpoint_sha256),
        "mission_source_identity": dict(mission_source_identity),
        "source_replay_identity": dict(source_replay_identity or {}),
        "runtime_identity": dict(runtime_identity),
        "progress": progress,
        "runtime_metrics": metrics,
        "cleanup_owner": "StandardAWACOnlineRunner",
        "checkpoint_save_reason": str(checkpoint_save_reason),
        "checkpoint_final_commit": str(checkpoint_final_commit),
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "checkpoint_transaction": {
            "checkpoint_name": "checkpoint_last",
            "save_reason": str(checkpoint_save_reason),
            "transaction_state": (
                "COMMITTED"
                if str(checkpoint_final_commit) == "PASS"
                else "FAILED_INCOMPLETE"
            ),
            "checkpoint_final_commit": str(checkpoint_final_commit),
        },
        "replay_committed_state": _standard_online_replay_committed_state(runner),
        **actor,
    }


def _write_standard_online_start_manifest(
    *,
    output_dir: Path,
    runner=None,
    resume_checkpoint_path: Path,
    resume_checkpoint_sha256: str,
    source_checkpoint_sha256: str,
    task_contract_sha256_value: str,
    actor_state_before_sha256: Optional[str] = None,
    starting_replay_size: Optional[int] = None,
    starting_replay_total_added: Optional[int] = None,
    online_env_steps_budget: Optional[int] = None,
) -> Path:
    """Persist the Actor identity before starting the managed runtime."""

    if runner is not None:
        actor = _standard_online_actor_fingerprints(runner)
        start_size = int(runner.starting_replay_size)
        start_total_added = int(runner.starting_replay_total_added)
        budget = int(runner.online_env_steps_budget)
    else:
        before = str(actor_state_before_sha256 or "")
        if not before:
            raise ValueError("Standard AWAC start manifest Actor identity is missing")
        if starting_replay_size is None or starting_replay_total_added is None:
            raise ValueError("Standard AWAC start manifest replay identity is missing")
        if online_env_steps_budget is None:
            raise ValueError("Standard AWAC start manifest budget is missing")
        actor = {
            "actor_state_before_sha256": before,
            "actor_state_after_sha256": "",
            "actor_changed": False,
        }
        start_size = int(starting_replay_size)
        start_total_added = int(starting_replay_total_added)
        budget = int(online_env_steps_budget)
    path = Path(output_dir).expanduser().resolve() / "runtime_start.json"
    payload = {
        "schema_id": "awac_standard_online_runtime_start_v1",
        "phase": AWAC_PHASE_STANDARD_TRAINING,
        "actor_state_before_sha256": actor["actor_state_before_sha256"],
        "resume_checkpoint_identity": {
            "path": str(resume_checkpoint_path),
            "sha256": str(resume_checkpoint_sha256),
        },
        "source_calibration_checkpoint_sha256": str(source_checkpoint_sha256),
        "starting_replay_size": start_size,
        "starting_replay_total_added": start_total_added,
        "online_env_steps_budget": budget,
        "online_budget_counter_owner": "phase_local_online_environment_steps",
        "task_contract_sha256": str(task_contract_sha256_value),
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
    }
    write_json_atomic(path, payload, trailing_newline=True)
    return path


def _validate_online_resume_identity(
    checkpoint: Mapping[str, Any],
    *,
    mission_source_identity: Mapping[str, Any],
    worker_identity: Mapping[str, Any],
) -> None:
    """Reject resume when runtime, mission, or durable progress changed."""

    if checkpoint.get("phase") != AWAC_PHASE_CRITIC_CALIBRATION:
        raise ValueError("AWAC calibration resume phase identity mismatch")
    if checkpoint.get("actor_update_enabled") is not False:
        raise ValueError("AWAC calibration resume Actor updates are enabled")
    if str(checkpoint.get("calibration_gate_state", "")) != CALIBRATION_GATE_PENDING:
        raise ValueError(
            "AWAC calibration resume requires a PENDING calibration checkpoint"
        )

    received_source = checkpoint.get("mission_source_identity")
    if received_source != dict(mission_source_identity):
        raise ValueError("AWAC calibration resume mission source identity mismatch")
    received_runtime = checkpoint.get("runtime_identity")
    if not isinstance(received_runtime, Mapping):
        raise ValueError("AWAC calibration resume worker topology identity is missing")
    for field in (
        "worker_spec_path",
        "worker_spec_sha256",
        "worker_count",
        "runtime_instance_ids",
        "runtime_launch_nonces",
        "training_run_ids",
        "unity_sha256",
        "bridge_sha256",
    ):
        if received_runtime.get(field) != worker_identity.get(field):
            raise ValueError(
                "AWAC calibration resume worker topology {} mismatch".format(field)
            )
    progress = checkpoint.get("mission_progress")
    if not isinstance(progress, Mapping):
        raise ValueError("AWAC calibration resume mission progress is missing")
    if progress.get("mission_source_identity") != dict(mission_source_identity):
        raise ValueError("AWAC calibration resume progress source identity mismatch")
    if str(progress.get("schema_id", "")) != "awac_formal_calibration_runtime_v1":
        raise ValueError("AWAC calibration resume progress schema mismatch")
    if str(progress.get("gate_state", "")) not in (
        CALIBRATION_GATE_PENDING,
        "BLOCKED_PENDING",
    ):
        raise ValueError("AWAC calibration resume progress gate state is not PENDING")
    for field in (
        "environment_step_count",
        "completed_episode_count",
        "critic_update_count",
    ):
        try:
            checkpoint_value = int(checkpoint[field])
            progress_value = int(progress[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "AWAC calibration resume progress {} is missing".format(field)
            ) from exc
        if checkpoint_value != progress_value:
            raise ValueError(
                "AWAC calibration resume progress {} mismatch".format(field)
            )


def _managed_runtime_identity_fingerprint(identity: Mapping[str, Any]) -> Dict[str, Any]:
    """Project runtime identity to immutable resume-bound fields."""

    if not isinstance(identity, Mapping):
        raise ValueError("managed runtime identity must be a mapping")
    fields = (
        "schema",
        "mode",
        "workspace",
        "artifacts",
        "contracts",
        "reliable_v4",
        "workers",
        "process_cleanup",
    )
    return {field: identity.get(field) for field in fields}


def _run_online_calibration(
    *,
    args: argparse.Namespace,
    learner: DiscreteAWACLearner,
    checkpoint: Mapping[str, Any],
    bc_path: Path,
    mpl_sha: str,
    resolved: Mapping[str, Any],
    resume_payload: Optional[Mapping[str, Any]],
    torch,
) -> Dict[str, Any]:
    """Run fresh or resumed online critic-only calibration."""

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_logger = None
    if not bool(getattr(args, "no_tensorboard", False)) and str(
        args.tensorboard_log_dir
    ).strip():
        from planning.awac.tensorboard import AWACTensorBoardLogger

        tensorboard_logger = AWACTensorBoardLogger(
            Path(args.tensorboard_log_dir).expanduser().resolve(),
            log_interval_steps=int(args.log_interval_steps),
        )
    missions = load_calibration_missions(
        Path(args.train_index), max_steps=int(args.max_steps)
    )
    source_identity = calibration_mission_source_identity(Path(args.train_index), missions)
    split = _resolve_online_calibration_split(
        args=args,
        missions=missions,
        output_dir=out_dir,
        resume=resume_payload is not None,
    )
    worker_spec_path = str(args.worker_spec_file).strip()
    if not worker_spec_path:
        raise RuntimeError(
            "online critic calibration requires --worker-spec-file for ParallelEnvPool"
        )
    worker_identity = _worker_spec_identity(
        Path(worker_spec_path), int(args.env_workers)
    )
    replay_dir = Path(args.replay_dir).expanduser().resolve() if str(args.replay_dir).strip() else out_dir / "replay"
    replay = None
    managed_runtime = None
    managed_runtime_identity = None
    pool = None
    try:
        if resume_payload is None:
            replay = _create_online_calibration_replay(
                replay_dir=replay_dir,
                args=args,
                resolved=resolved,
                bc_path=bc_path,
                mpl_sha=mpl_sha,
                mission_source_identity=source_identity,
                split=split,
                worker_identity=worker_identity,
            )
        else:
            if not replay_dir.is_dir():
                raise FileNotFoundError(
                    "calibration resume replay is missing: {}".format(replay_dir)
                )
            replay = AWACReplayBuffer.open(replay_dir)
            recover_calibration_replay_from_checkpoint_transaction(
                replay,
                Path(args.resume_checkpoint),
            )
            if int(resume_payload.get("replay_size", -1)) != int(replay.size):
                raise ValueError("AWAC calibration resume replay size mismatch")
            _validate_calibration_replay_resume_identity(
                replay,
                resume_payload,
                resolved_training_config=resolved,
                bc_checkpoint_sha256=file_sha256(bc_path),
                mpl_sha256=mpl_sha,
                max_steps=int(args.max_steps),
                mission_source_sha256=str(source_identity["sha256"]),
            )
            _validate_online_resume_identity(
                resume_payload,
                mission_source_identity=source_identity,
                worker_identity=worker_identity,
            )
            if resume_payload.get("calibration_split") != split:
                raise ValueError("AWAC calibration resume split identity mismatch")

        exact_resume_transition_cap = min(
            int(args.total_env_steps), int(args.calibration_max_transitions)
        )
        if int(replay.capacity) < int(exact_resume_transition_cap):
            raise RuntimeError(
                "AWAC calibration replay capacity cannot preserve an exact "
                "resume without overwriting a committed generation"
            )

        managed_runtime, pool, managed_runtime_identity = start_managed_calibration_runtime(
            Path(worker_spec_path),
            worker_count=int(args.env_workers),
            output_dir=out_dir,
            task_contract_sha256=task_contract_sha256(int(args.max_steps)),
            mpl_contract_sha256=mpl_sha,
            max_steps=int(args.max_steps),
            max_sensor_skew_ms=float(args.max_sensor_skew_ms),
            depth_mask_collision_radius=float(args.depth_mask_collision_radius),
            depth_mask_slack=float(args.depth_mask_slack),
            depth_mask_sample_stride=int(args.depth_mask_sample_stride),
            depth_mask_max_patch_radius_px=int(args.depth_mask_max_patch_radius_px),
            startup_timeout_s=float(args.worker_ready_timeout),
            request_timeout_s=float(args.reset_timeout),
        )

        if resume_payload is not None:
            resume_runtime_identity = resume_payload.get("runtime_identity")
            expected_managed_identity = (
                resume_runtime_identity.get("managed_runtime_identity")
                if isinstance(resume_runtime_identity, Mapping)
                else None
            )
            if not isinstance(expected_managed_identity, Mapping):
                raise ValueError(
                    "AWAC calibration resume managed runtime identity is missing"
                )
            if _managed_runtime_identity_fingerprint(expected_managed_identity) != _managed_runtime_identity_fingerprint(
                managed_runtime_identity
            ):
                raise ValueError(
                    "AWAC calibration resume managed runtime identity mismatch"
                )

        normalizer = VectorNormalizer.from_checkpoint(dict(checkpoint))
        train_ids = {str(value) for value in split["train_episode_ids"]}
        holdout_ids = {str(value) for value in split["holdout_episode_ids"]}
        expected_runtime_ids = {
            index: value
            for index, value in enumerate(worker_identity["runtime_instance_ids"])
        }

        def persist_checkpoint(owner, gate_status):
            """Persist one complete replay/checkpoint generation at a gate."""

            def build_payload(replay_identity):
                snapshot = owner.checkpoint_snapshot(gate_status)
                metrics = _online_calibration_metrics(
                    producer=owner, result=snapshot, status=gate_status
                )
                return build_calibration_checkpoint_payload(
                    learner=learner,
                    bc_checkpoint=checkpoint,
                    bc_checkpoint_sha256=file_sha256(bc_path),
                    mpl_contract_sha256=mpl_sha,
                    training_contract_sha256=str(
                        resolved["training_contract_sha256"]
                    ),
                    training_contract=resolved["training_contract"],
                    calibration_split=split,
                    replay_identity=replay_identity,
                    environment_step_count=int(snapshot["environment_step_count"]),
                    calibration_metrics=metrics,
                    mission_source_identity=source_identity,
                    mission_progress=snapshot["progress"],
                    completed_episode_count=int(
                        snapshot["completed_episode_count"]
                    ),
                    runtime_identity={
                        **worker_identity,
                        "observed_runtime_instance_ids": snapshot[
                            "runtime_identity"
                        ],
                        "managed_runtime_identity": managed_runtime_identity,
                    },
                    exact_resume_state=snapshot["exact_resume_state"],
                )

            return commit_calibration_checkpoint_transaction(
                output_dir=out_dir,
                checkpoint_name="checkpoint_last",
                replay=replay,
                build_payload=build_payload,
                torch=torch,
                save_reason="gate_window",
            )

        calibration_behavior_policy = calibration_behavior_policy_identity(
            checkpoint_sha256=file_sha256(bc_path),
            rng_seed=int(args.seed) + 1,
        )
        policy_context = _calibration_value_policy_context(
            bc_checkpoint_sha256=file_sha256(bc_path),
            behavior_policy=calibration_behavior_policy,
        )
        producer = CalibrationReplayProducer(
            missions=missions,
            pool=pool,
            replay=replay,
            learner=learner,
            train_episode_ids=train_ids,
            holdout_episode_ids=holdout_ids,
            calibration_config=_calibration_config(args),
            torch=torch,
            normalizer=normalizer,
            device=learner.device,
            calibration_window_interval_steps=int(args.calibration_window_interval_steps),
            batch_size=int(args.batch_size),
            learning_starts=int(args.learning_starts),
            updates_per_step=float(args.updates_per_step),
            max_transitions=min(
                int(args.total_env_steps), int(args.calibration_max_transitions)
            ),
            max_episodes=int(args.calibration_max_episodes),
            ready_timeout_s=float(args.worker_ready_timeout),
            reset_timeout_s=float(args.reset_timeout),
            step_timeout_s=float(args.reset_timeout),
            reset_settle=float(args.reset_settle),
            rng=np.random.RandomState(int(args.seed) + 1),
            behavior_policy=calibration_behavior_policy,
            mission_source_identity=source_identity,
            expected_runtime_ids=expected_runtime_ids,
            holdout_evaluator=lambda records: _summarize_calibration_holdout_window(
                learner=learner,
                torch=torch,
                records=records,
                gamma=float(args.gamma),
                reward_scale=float(args.reward_scale),
                policy_context=policy_context,
            ),
            on_checkpoint=persist_checkpoint,
            output_dir=out_dir,
        )
        if resume_payload is not None:
            producer.restore_progress(
                resume_payload["mission_progress"],
                exact_resume_state=resume_payload["exact_resume_state"],
            )

        def checkpoint_payload_builder(
            *,
            snapshot: Mapping[str, Any],
            calibration_metrics: Mapping[str, Any],
            checkpoint_name: str,
        ):
            def build_final_payload(replay_identity):
                checkpoint_kwargs = {
                    "learner": learner,
                    "bc_checkpoint": checkpoint,
                    "bc_checkpoint_sha256": file_sha256(bc_path),
                    "mpl_contract_sha256": mpl_sha,
                    "training_contract_sha256": str(
                        resolved["training_contract_sha256"]
                    ),
                    "training_contract": resolved["training_contract"],
                    "calibration_split": split,
                    "replay_identity": replay_identity,
                    "environment_step_count": int(
                        snapshot["environment_step_count"]
                    ),
                    "calibration_metrics": calibration_metrics,
                    "mission_source_identity": source_identity,
                    "mission_progress": snapshot["progress"],
                    "completed_episode_count": int(
                        snapshot["completed_episode_count"]
                    ),
                    "runtime_identity": {
                        **worker_identity,
                        "observed_runtime_instance_ids": snapshot[
                            "runtime_identity"
                        ],
                        "managed_runtime_identity": managed_runtime_identity,
                    },
                    "exact_resume_state": snapshot["exact_resume_state"],
                }
                if checkpoint_name == "checkpoint_calibration_pass":
                    return build_calibration_pass_checkpoint_payload(
                        **checkpoint_kwargs,
                        actor_depth_lr=float(args.actor_depth_lr),
                        critic_depth_lr=float(args.critic_depth_lr),
                    )
                return build_calibration_checkpoint_payload(**checkpoint_kwargs)

            return build_final_payload

        try:
            result = producer.run()
        except BaseException as runtime_error:
            failure_traceback = traceback.format_exc()
            recovery_diagnostics = _runtime_recovery_diagnostics(
                producer,
                error=runtime_error,
                error_traceback=failure_traceback,
            )
            summary_path = _write_runtime_failure_recovery_summary(
                output_dir=out_dir,
                checkpoint_name="checkpoint_last",
                diagnostics={
                    **recovery_diagnostics,
                    "checkpoint_save_reason": (
                        "keyboard_interrupt"
                        if isinstance(runtime_error, KeyboardInterrupt)
                        else "runtime_exception"
                    ),
                    "checkpoint_final_commit": "NOT_ATTEMPTED",
                },
            )
            # A divergent gate is terminal by contract.  Preserve the failure
            # evidence, but never publish it as a resumable PENDING snapshot.
            if str(producer.controller.certification) != "FAIL_DIVERGED":
                interrupted_snapshot = producer.checkpoint_snapshot(
                    {"state": "PENDING"}
                )
                interrupted_result = {
                    "status": "PENDING",
                    "stop_reason": str(interrupted_snapshot["stop_reason"]),
                    "environment_step_count": int(
                        interrupted_snapshot["environment_step_count"]
                    ),
                    "completed_episode_count": int(
                        interrupted_snapshot["completed_episode_count"]
                    ),
                    "holdout_episode_count": int(
                        interrupted_snapshot["holdout_episode_count"]
                    ),
                    "holdout_transition_count": int(
                        interrupted_snapshot["holdout_transition_count"]
                    ),
                    "gate_history": list(interrupted_snapshot["gate_history"]),
                    "progress": dict(interrupted_snapshot["progress"]),
                    "actor_unchanged": bool(
                        interrupted_snapshot["actor_unchanged"]
                    ),
                }
                interrupted_metrics = _online_calibration_metrics(
                    producer=producer,
                    result=interrupted_result,
                )
                try:
                    committed = commit_calibration_checkpoint_transaction(
                        output_dir=out_dir,
                        checkpoint_name="checkpoint_last",
                        checkpoint_kind="checkpoint_last",
                        replay=replay,
                        build_payload=checkpoint_payload_builder(
                            snapshot=interrupted_snapshot,
                            calibration_metrics=interrupted_metrics,
                            checkpoint_name="checkpoint_last",
                        ),
                        torch=torch,
                        save_reason=(
                            "keyboard_interrupt"
                            if isinstance(runtime_error, KeyboardInterrupt)
                            else "runtime_exception"
                        ),
                        recovery_diagnostics=recovery_diagnostics,
                    )
                    load_committed_calibration_resume_checkpoint(
                        Path(committed["checkpoint_alias"]),
                        torch=torch,
                        map_location="cpu",
                    )
                    _write_runtime_failure_recovery_summary(
                        output_dir=out_dir,
                        checkpoint_name="checkpoint_last",
                        diagnostics={
                            **recovery_diagnostics,
                            "checkpoint_save_reason": (
                                "keyboard_interrupt"
                                if isinstance(runtime_error, KeyboardInterrupt)
                                else "runtime_exception"
                            ),
                            "checkpoint_final_commit": "PASS",
                            "checkpoint_path": str(committed["checkpoint_alias"]),
                            "checkpoint_transaction_diagnostics": str(
                                committed["diagnostics_path"]
                            ),
                        },
                    )
                except BaseException as checkpoint_error:
                    _write_runtime_failure_recovery_summary(
                        output_dir=out_dir,
                        checkpoint_name="checkpoint_last",
                        diagnostics={
                            **recovery_diagnostics,
                            "checkpoint_final_commit": "FAIL",
                            "checkpoint_failure_summary": "{}: {}".format(
                                type(checkpoint_error).__name__,
                                str(checkpoint_error),
                            ),
                            "checkpoint_failure_traceback": traceback.format_exc(),
                            "checkpoint_transaction_diagnostics": str(
                                calibration_checkpoint_transaction_diagnostics_path(
                                    out_dir, "checkpoint_last"
                                )
                            ),
                        },
                    )
            # Preserve the original runtime error and never turn an
            # interruption into a successful calibration invocation.
            raise

        status = str(result["status"])
        final_snapshot = producer.checkpoint_snapshot({"state": status})
        calibration_metrics = _online_calibration_metrics(
            producer=producer, result=result
        )
        checkpoint_name = (
            "checkpoint_calibration_pass"
            if calibration_metrics["state"] == "PASS"
            else "checkpoint_last"
        )
        committed = commit_calibration_checkpoint_transaction(
            output_dir=out_dir,
            checkpoint_name=checkpoint_name,
            checkpoint_kind=checkpoint_name,
            replay=replay,
            build_payload=checkpoint_payload_builder(
                snapshot=final_snapshot,
                calibration_metrics=calibration_metrics,
                checkpoint_name=checkpoint_name,
            ),
            torch=torch,
            save_reason="run_complete",
            recovery_diagnostics=_runtime_recovery_diagnostics(producer),
        )
        load_committed_calibration_resume_checkpoint(
            Path(committed["checkpoint_alias"]),
            torch=torch,
            map_location="cpu",
        )
        result["replay_dir"] = str(replay_dir)
        result["replay_identity"] = committed["replay_identity"]
        result["calibration_split"] = split
        result["mission_source_identity"] = source_identity
        result["worker_identity"] = worker_identity
        result["managed_runtime_identity"] = managed_runtime_identity
        result["checkpoint_path"] = str(committed["checkpoint_alias"])
        return result
    finally:
        if pool is not None:
            close = getattr(pool, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        if managed_runtime is not None:
            close = getattr(managed_runtime, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        if replay is not None:
            replay.close()


def _create_tensorboard_logger(args, *, default_to_output_dir: bool = False):
    """Create the trainer-owned logger with an explicit production default."""

    if bool(getattr(args, "no_tensorboard", False)):
        return None
    log_dir = str(getattr(args, "tensorboard_log_dir", "")).strip()
    if not log_dir and default_to_output_dir:
        log_dir = str(
            Path(args.out_dir).expanduser().resolve() / "tensorboard"
        )
    if not log_dir:
        return None
    from planning.awac.tensorboard import AWACTensorBoardLogger

    return AWACTensorBoardLogger(
        Path(log_dir).expanduser().resolve(),
        log_interval_steps=int(args.log_interval_steps),
    )


def _close_tensorboard_logger(tensorboard_logger, *, output_dir: Path) -> None:
    """Close observability without masking the owning operation's exception."""

    if tensorboard_logger is None:
        return
    try:
        tensorboard_logger.close()
        manifest = tensorboard_logger.manifest()
        manifest["enabled"] = True
        manifest["physical_size_mb"] = tensorboard_logger.physical_size_mb()
        write_json_atomic(
            Path(output_dir) / "tensorboard_manifest.json",
            manifest,
            trailing_newline=True,
        )
    except BaseException:
        # TensorBoard is diagnostic-only.  Its cleanup or manifest failure must
        # never replace the original runtime/checkpoint exception.
        pass


def _run_standard_awac_online(
    *,
    args: argparse.Namespace,
    learner: DiscreteAWACLearner,
    checkpoint: Mapping[str, Any],
    bc_path: Path,
    mpl_sha: str,
    resolved: Mapping[str, Any],
    resume_payload: Mapping[str, Any],
    torch,
) -> Dict[str, Any]:
    """Run Standard AWAC from a Calibration PASS or Standard checkpoint."""

    # This must exist before any exception-prone path can reach runner
    # construction or the cleanup block below.  It is intentionally assigned
    # before the first try so a disabled logger and early failures are both
    # fail-safe.
    tensorboard_logger = None
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    missions = load_calibration_missions(
        Path(args.train_index), max_steps=int(args.max_steps)
    )
    source_identity = calibration_mission_source_identity(
        Path(args.train_index), missions
    )
    worker_spec_path = Path(args.worker_spec_file).expanduser().resolve()
    worker_identity = _worker_spec_identity(worker_spec_path, int(args.env_workers))
    expected_runtime_ids = {
        index: value
        for index, value in enumerate(worker_identity["runtime_instance_ids"])
    }
    normalizer = VectorNormalizer.from_checkpoint(dict(checkpoint))
    managed_runtime = None
    managed_runtime_identity: Mapping[str, Any] = {}
    pool = None
    replay = None
    runner = None
    checkpoint_paths = []
    finalization_started = False
    final_checkpoint_commit = "NOT_ATTEMPTED"
    resume_checkpoint_path = Path(args.resume_checkpoint).expanduser().resolve()
    resume_checkpoint_sha256 = file_sha256(resume_checkpoint_path)
    source_checkpoint_sha256 = str(
        resume_payload.get("source_calibration_checkpoint_sha256", "")
    )
    phase1_state = Phase1StateMachine(
        online_transition_cap=max(
            PHASE1_DEFAULT_ONLINE_TRANSITION_CAP, _online_env_steps(args)
        )
    )
    source_replay_identity = resume_payload.get("replay_identity", {})

    try:
        if resume_payload.get("phase") == AWAC_PHASE_STANDARD_TRAINING:
            if source_identity != resume_payload.get("mission_source_identity"):
                raise Phase1ReadinessError(
                    "Standard AWAC resume mission source identity mismatch"
                )
            source_replay_identity = resume_payload.get("source_replay_identity", {})
            replay = AWACReplayBuffer.open(
                Path(args.replay_dir).expanduser().resolve()
            )
            current_identity = _standard_replay_identity(replay)
            expected_identity = resume_payload.get("replay_identity")
            if not isinstance(expected_identity, Mapping):
                raise Phase1ReadinessError(
                    "Standard AWAC resume replay identity is missing"
                )
            for field, expected in expected_identity.items():
                if current_identity.get(field) != expected:
                    raise Phase1ReadinessError(
                        "Standard AWAC resume replay identity {} mismatch".format(
                            field
                        )
                    )
            try:
                validate_standard_online_replay_counter(resume_payload, replay)
            except (TypeError, ValueError) as error:
                raise Phase1ReadinessError(
                    "Standard AWAC resume committed transition cross-check failed: {}".format(
                        error
                    )
                ) from error
            phase1_state_payload = resume_payload.get("phase1_state", {})
            if not isinstance(phase1_state_payload, Mapping):
                raise Phase1ReadinessError(
                    "Standard AWAC resume Phase-1 state is missing"
                )
            starting_replay_size = int(
                resume_payload.get("starting_replay_size", replay.size)
            )
            starting_replay_total_added = int(
                resume_payload.get(
                    "starting_replay_total_added", replay.total_added
                )
            )
        else:
            source_checkpoint_sha256 = file_sha256(
                Path(args.resume_checkpoint).expanduser().resolve()
            )
            source_replay = AWACReplayBuffer.open(
                Path(args.replay_dir).expanduser().resolve(), read_only=True
            )
            try:
                source_replay_identity = calibration_replay_identity(source_replay)
                validate_standard_awac_handoff_payload(
                    resume_payload,
                    expected_bc_checkpoint_sha256=file_sha256(bc_path),
                    expected_bc_reference_fingerprint=actor_state_sha256(checkpoint),
                    expected_mpl_contract_sha256=str(mpl_sha),
                    expected_task_contract_sha256=task_contract_sha256(
                        int(args.max_steps)
                    ),
                    expected_replay_identity=source_replay_identity,
                )
                if resume_payload.get("mission_source_identity") != source_identity:
                    raise Phase1ReadinessError(
                        "Standard AWAC handoff mission source identity mismatch"
                    )
                replay = _create_standard_online_replay(
                    source_replay=source_replay,
                    output_dir=out_dir,
                    args=args,
                    resolved=resolved,
                    mission_source_identity=source_identity,
                    source_replay_identity=source_replay_identity,
                    source_calibration_checkpoint_sha256=source_checkpoint_sha256,
                    worker_identity=worker_identity,
                )
            finally:
                source_replay.close()
            starting_replay_size = int(replay.size)
            starting_replay_total_added = int(replay.total_added)
            handoff = validate_standard_awac_handoff_payload(
                resume_payload,
                expected_bc_checkpoint_sha256=file_sha256(bc_path),
                expected_bc_reference_fingerprint=actor_state_sha256(checkpoint),
                expected_mpl_contract_sha256=str(mpl_sha),
                expected_task_contract_sha256=task_contract_sha256(
                    int(args.max_steps)
                ),
                expected_replay_identity=source_replay_identity,
            )
            learner.enable_actor_after_calibration_pass(handoff)
            phase1_state.transition_from_calibration(
                gate_state=str(resume_payload["calibration_gate_state"]),
                actor_depth_lr=float(resume_payload["phase1_actor_depth_lr"]),
                critic_depth_lr=float(resume_payload["phase1_critic_depth_lr"]),
                env_step=int(resume_payload.get("environment_step_count", 0)),
                replay_size=int(resume_payload.get("replay_size", replay.size)),
                critic_update_count=int(
                    resume_payload.get("critic_update_count", 0)
                ),
                actor_update_count=int(resume_payload.get("actor_update_count", 0)),
                source_checkpoint=str(
                    resume_payload["checkpoint_transaction"]["checkpoint_filename"]
                ),
            )

        # All source/replay/config identity checks are complete here.  Create
        # the sole trainer-owned logger immediately before runtime startup;
        # StandardAWACOnlineRunner only receives this injected instance.
        tensorboard_logger = _create_tensorboard_logger(
            args, default_to_output_dir=True
        )

        _write_standard_online_start_manifest(
            output_dir=out_dir,
            resume_checkpoint_path=resume_checkpoint_path,
            resume_checkpoint_sha256=resume_checkpoint_sha256,
            source_checkpoint_sha256=source_checkpoint_sha256,
            task_contract_sha256_value=str(
                resolved["training_contract"].get("task_contract_sha256", "")
            ),
            actor_state_before_sha256=learner._state_fingerprint(
                learner.actor.state_dict()
            ),
            starting_replay_size=int(starting_replay_size),
            starting_replay_total_added=int(starting_replay_total_added),
            online_env_steps_budget=_online_env_steps(args),
        )

        managed_runtime, pool, managed_runtime_identity = (
            start_managed_calibration_runtime(
                worker_spec_path,
                worker_count=int(args.env_workers),
                output_dir=out_dir,
                task_contract_sha256=task_contract_sha256(int(args.max_steps)),
                mpl_contract_sha256=str(mpl_sha),
                max_steps=int(args.max_steps),
                max_sensor_skew_ms=float(args.max_sensor_skew_ms),
                depth_mask_collision_radius=float(args.depth_mask_collision_radius),
                depth_mask_slack=float(args.depth_mask_slack),
                depth_mask_sample_stride=int(args.depth_mask_sample_stride),
                depth_mask_max_patch_radius_px=int(
                    args.depth_mask_max_patch_radius_px
                ),
                startup_timeout_s=float(args.worker_ready_timeout),
                request_timeout_s=float(args.reset_timeout),
            )
        )

        runner = StandardAWACOnlineRunner(
            missions=missions,
            pool=pool,
            replay=replay,
            learner=learner,
            normalizer=normalizer,
            torch=torch,
            device=learner.device,
            online_env_steps=_online_env_steps(args),
            batch_size=int(args.batch_size),
            learning_starts=int(args.learning_starts),
            actor_learning_starts=int(args.actor_learning_starts),
            critic_burnin_updates=int(args.critic_burnin_updates),
            actor_update_interval=int(args.actor_update_interval),
            updates_per_step=float(args.updates_per_step),
            behavior_temperature=float(args.warmup_temperature),
            max_episodes=len(missions),
            ready_timeout_s=float(args.worker_ready_timeout),
            reset_timeout_s=float(args.reset_timeout),
            step_timeout_s=float(args.reset_timeout),
            reset_settle=float(args.reset_settle),
            rng=np.random.RandomState(int(args.seed) + 1),
            mission_source_identity=source_identity,
            expected_runtime_ids=expected_runtime_ids,
            replay_identity=_standard_replay_identity(replay),
            output_dir=out_dir,
            phase1_state=phase1_state,
            checkpoint_interval_steps=int(args.checkpoint_interval_steps),
            starting_replay_size=(
                starting_replay_size
                if resume_payload.get("phase") == AWAC_PHASE_STANDARD_TRAINING
                else None
            ),
            starting_replay_total_added=(
                starting_replay_total_added
                if resume_payload.get("phase") == AWAC_PHASE_STANDARD_TRAINING
                else None
            ),
            tensorboard_logger=tensorboard_logger,
            tensorboard_log_interval_steps=int(args.log_interval_steps),
        )

        if resume_payload.get("phase") == AWAC_PHASE_STANDARD_TRAINING:
            runner.restore_progress(
                resume_payload["standard_online_progress"],
                exact_resume_state=resume_payload["standard_online_resume_state"],
            )
            if int(runner.online_transitions_committed) != int(
                resume_payload["online_transitions_committed"]
            ):
                raise Phase1ReadinessError(
                    "Standard AWAC resume committed transition counter mismatch"
                )
            runner._next_checkpoint_step = (
                (runner.online_env_steps // runner.checkpoint_interval_steps) + 1
            ) * runner.checkpoint_interval_steps
        else:
            runner.restore_from_calibration_handoff(
                resume_payload["mission_progress"],
                exact_resume_state=resume_payload["exact_resume_state"],
            )
        if tensorboard_logger is not None:
            runner._maybe_tensorboard(force=True)

        def persist_checkpoint(owner, snapshot):
            del snapshot

            def build_payload(replay_identity):
                current = owner.checkpoint_snapshot({"status": "RUNNING"})
                return _build_standard_checkpoint_payload(
                    learner=learner,
                    args=args,
                    bc_checkpoint=checkpoint,
                    bc_checkpoint_path=bc_path,
                    mpl_sha=mpl_sha,
                    resolved=resolved,
                    replay_identity=replay_identity,
                    source_replay_identity=source_replay_identity,
                    source_calibration_checkpoint_sha256=source_checkpoint_sha256,
                    mission_source_identity=source_identity,
                    worker_identity=worker_identity,
                    managed_runtime_identity=managed_runtime_identity,
                    snapshot=current,
                )

            committed = commit_standard_awac_checkpoint_transaction(
                output_dir=out_dir,
                checkpoint_name="checkpoint_last",
                replay=replay,
                build_payload=build_payload,
                torch=torch,
                save_reason="interval",
            )
            checkpoint_paths.append(str(committed["checkpoint_alias"]))
            return committed

        runner.on_checkpoint = persist_checkpoint
        result = runner.run()
        if tensorboard_logger is not None and runner.tensorboard_logger is not None:
            runner._maybe_tensorboard(force=True)
            result["tensorboard"] = tensorboard_logger.manifest()
        finalization_started = True
        final_snapshot = runner.checkpoint_snapshot({"status": result["status"]})

        def build_final_payload(replay_identity):
            return _build_standard_checkpoint_payload(
                learner=learner,
                args=args,
                bc_checkpoint=checkpoint,
                bc_checkpoint_path=bc_path,
                mpl_sha=mpl_sha,
                resolved=resolved,
                replay_identity=replay_identity,
                source_replay_identity=source_replay_identity,
                source_calibration_checkpoint_sha256=source_checkpoint_sha256,
                mission_source_identity=source_identity,
                worker_identity=worker_identity,
                managed_runtime_identity=managed_runtime_identity,
                snapshot=final_snapshot,
            )

        committed = commit_standard_awac_checkpoint_transaction(
            output_dir=out_dir,
            checkpoint_name="checkpoint_last",
            replay=replay,
            build_payload=build_final_payload,
            torch=torch,
            save_reason="run_complete",
        )
        final_checkpoint_commit = "PASS"
        checkpoint_paths.append(str(committed["checkpoint_alias"]))
        loaded = load_committed_standard_resume_checkpoint(
            Path(committed["checkpoint_alias"]), torch=torch, map_location="cpu"
        )
        counts = _behavior_source_counts(replay)
        summary = build_standard_online_summary(
            result=result,
            phase=AWAC_PHASE_STANDARD_TRAINING,
            start_checkpoint_identity={
                "path": str(resume_checkpoint_path),
                "sha256": resume_checkpoint_sha256,
                "phase": str(resume_payload.get("phase", "")),
                "source_calibration_checkpoint_sha256": source_checkpoint_sha256,
            },
            starting_replay_size=int(runner.starting_replay_size),
            ending_replay_size=int(replay.size),
            behavior_source_counts=counts,
            runtime_identity={
                **dict(worker_identity),
                "observed_runtime_instance_ids": result.get(
                    "runtime_identity", {}
                ),
                "managed_runtime_identity": dict(managed_runtime_identity),
            },
            checkpoint_paths=checkpoint_paths,
        )
        write_json_atomic(out_dir / "summary.json", summary, trailing_newline=True)
        result.update(
            {
                "checkpoint_path": str(committed["checkpoint_alias"]),
                "checkpoint_paths": checkpoint_paths,
                "summary_path": str(out_dir / "summary.json"),
                "summary": summary,
                "replay_dir": str(replay.directory),
                "source_replay_identity": dict(source_replay_identity),
                "loaded_checkpoint_phase": loaded["phase"],
                "tensorboard": (
                    tensorboard_logger.manifest()
                    if tensorboard_logger is not None
                    else {"enabled": False}
                ),
            }
        )
        return result
    except BaseException as error:
        # A learner/runtime failure is fail-stop.  Keep a durable diagnostic
        # record for the owned run, but never turn a partially completed
        # online episode into a checkpoint or silently roll the Actor back.
        failure = _standard_online_failure_payload(
            runner=runner,
            error=error,
            error_traceback=traceback.format_exc(),
            source_checkpoint_sha256=source_checkpoint_sha256,
            mission_source_identity=source_identity,
            source_replay_identity=source_replay_identity,
            runtime_identity=managed_runtime_identity,
            checkpoint_save_reason=(
                "run_complete" if finalization_started else "runtime_exception"
            ),
            checkpoint_final_commit=final_checkpoint_commit,
            checkpoint_paths=checkpoint_paths,
        )
        try:
            write_json_atomic(
                out_dir / "runtime_failure.json",
                failure,
                trailing_newline=True,
            )
        except BaseException:
            pass
        raise
    finally:
        if pool is not None:
            close = getattr(pool, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        if managed_runtime is not None:
            close = getattr(managed_runtime, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        if replay is not None:
            replay.close()
        _close_tensorboard_logger(tensorboard_logger, output_dir=out_dir)


def _calibration_tensor_batch(torch, records):
    """Convert replay-shaped numpy records into a learner batch."""

    if not records:
        raise ValueError("calibration evaluation requires at least one record")

    def stack(name, dtype=np.float32):
        return torch.from_numpy(np.asarray([row[name] for row in records], dtype=dtype))

    return {
        "depth": stack("depth"),
        "vector": stack("vector"),
        "action_mask": stack("action_mask", np.bool_),
        "action": stack("action", np.int64).long(),
        "reward": stack("reward"),
        "next_depth": stack("next_depth"),
        "next_vector": stack("next_vector"),
        "next_action_mask": stack("next_action_mask", np.bool_),
        "done": stack("done"),
        "behavior_source": stack("behavior_source", np.int64).long(),
    }


def _module_parameter_device(module, *, name: str):
    parameters = tuple(module.parameters())
    if not parameters:
        raise RuntimeError("calibration holdout {} has no parameters".format(name))
    device = parameters[0].device
    if any(parameter.device != device for parameter in parameters[1:]):
        raise RuntimeError(
            "calibration holdout {} parameters span multiple devices".format(name)
        )
    return device


def _validate_calibration_holdout_devices(*, learner, torch):
    """Return the concrete Critic compute device after fail-closed checks."""

    critic_device = _module_parameter_device(learner.critic1, name="critic1")
    learner_device = resolve_torch_device(torch, learner.device)
    if learner_device != critic_device:
        raise RuntimeError(
            "calibration holdout critic1 device {} != learner device {}".format(
                critic_device, learner_device
            )
        )
    for name, module in (
        ("critic2", learner.critic2),
        ("target_critic1", learner.target_critic1),
        ("target_critic2", learner.target_critic2),
        ("bc_reference", learner.bc_reference),
    ):
        module_device = _module_parameter_device(module, name=name)
        if module_device != critic_device:
            raise RuntimeError(
                "calibration holdout {} device {} != critic1 device {}".format(
                    name, module_device, critic_device
                )
            )
    return critic_device


def _validate_calibration_replay_resume_identity(
    replay,
    checkpoint: Mapping[str, Any],
    *,
    resolved_training_config: Mapping[str, Any],
    bc_checkpoint_sha256: str,
    mpl_sha256: str,
    max_steps: int,
    mission_source_sha256: str = "",
) -> None:
    """Reject a calibration resume if replay identity drifted since checkpoint."""

    metadata = replay.metadata
    expected = {
        "replay_contract_sha256": checkpoint["replay_contract_sha256"],
        "training_config_sha256": resolved_training_config[
            "training_contract_sha256"
        ],
        "bc_checkpoint_sha256": str(bc_checkpoint_sha256),
        "mpl_contract_sha256": str(mpl_sha256),
        "task_contract_sha256": task_contract_sha256(int(max_steps)),
        "observation_contract": EXPECTED_OBSERVATION_CONTRACT,
        "observation_source": EXPECTED_OBSERVATION_SOURCE,
        "behavior_source_phase": AWAC_PHASE_CRITIC_CALIBRATION,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                "AWAC calibration resume replay {} mismatch: received={} expected={}".format(
                    field, metadata.get(field), value
                )
            )
    if mission_source_sha256:
        for field in ("mission_source_sha256", "mission_index_sha256"):
            if metadata.get(field) != str(mission_source_sha256):
                raise ValueError(
                    "AWAC calibration resume replay mission source identity mismatch"
                )
    replay_identity = checkpoint.get("replay_identity")
    if not isinstance(replay_identity, Mapping):
        raise ValueError("AWAC calibration resume replay identity is missing")
    identity_checks = {
        "run_identity": metadata.get("run_identity"),
        "mission_source_sha256": metadata.get("mission_source_sha256", ""),
        "mission_index_sha256": metadata.get("mission_index_sha256", ""),
        "replay_size": int(replay.size),
        "replay_contract_sha256": metadata.get("replay_contract_sha256"),
        "replay_metadata_sha256": file_sha256(
            Path(replay.directory).resolve() / "metadata.json"
        ),
    }
    for field, value in identity_checks.items():
        if replay_identity.get(field) != value:
            raise ValueError(
                "AWAC calibration resume replay identity {} mismatch: received={} expected={}".format(
                    field, replay_identity.get(field), value
                )
            )


def _calibration_value_policy_context(
    *,
    bc_checkpoint_sha256: str,
    behavior_policy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe one shared frozen-BC distribution for calibration values.

    New calibration trajectories, independent holdout returns, and Bellman
    successor expectations all use masked categorical frozen BC at T=1.  The
    formal Dev evaluator retains its independent argmax/T=0 contract.
    """

    policy = dict(
        behavior_policy
        or calibration_behavior_policy_identity(
            checkpoint_sha256=str(bc_checkpoint_sha256)
        )
    )
    required = calibration_behavior_policy_identity(
        checkpoint_sha256=str(bc_checkpoint_sha256),
        rng_seed=policy.get("rng_seed"),
    )
    if policy != required:
        raise ValueError("calibration behavior policy identity is invalid")
    return {
        "behavior_policy": dict(policy),
        "target_policy": dict(policy),
        # The observed trajectory continuation is the same distribution that
        # generated every recorded action; no cross-policy label is used.
        "return_policy": dict(policy),
    }


def _evaluate_calibration_holdout(
    *, learner, torch, records, gamma, reward_scale, policy_context=None
):
    """Evaluate Q-vs-return diagnostics without changing critic parameters."""

    return_contract = episode_monte_carlo_returns(
        records,
        gamma=float(gamma),
        reward_scale=float(reward_scale),
    )
    context = dict(policy_context or {})
    alignment = assess_value_policy_alignment(
        behavior_policy=context.get("behavior_policy"),
        target_policy=context.get("target_policy"),
        return_policy=context.get("return_policy"),
    )
    critic_device = _validate_calibration_holdout_devices(learner=learner, torch=torch)
    batch = move_awac_batch_to_device(
        _calibration_tensor_batch(torch, records),
        device=critic_device,
    )
    with torch.no_grad():
        q1_all = learner.critic1(batch["depth"], batch["vector"])
        q2_all = learner.critic2(batch["depth"], batch["vector"])
        q1 = q1_all.gather(1, batch["action"][:, None]).squeeze(1)
        q2 = q2_all.gather(1, batch["action"][:, None]).squeeze(1)
        target_q1 = learner.target_critic1(
            batch["next_depth"], batch["next_vector"]
        )
        target_q2 = learner.target_critic2(
            batch["next_depth"], batch["next_vector"]
        )
        bc_logits = learner.bc_reference(
            batch["next_depth"], batch["next_vector"]
        )
        target_info = masked_bellman_target(
            bc_logits=bc_logits,
            target_q1=target_q1,
            target_q2=target_q2,
            next_action_mask=batch["next_action_mask"],
            reward=batch["reward"],
            done=batch["done"],
            gamma=float(gamma),
            reward_scale=float(reward_scale),
            torch=torch,
        )
        target = target_info["target"]
        critic1_loss = torch.nn.functional.mse_loss(q1, target)
        critic2_loss = torch.nn.functional.mse_loss(q2, target)
    if not bool(torch.isfinite(q1).all() and torch.isfinite(q2).all()):
        raise FloatingPointError("calibration holdout Q is non-finite")
    return {
        "critic1_td_loss": float(critic1_loss.cpu().item()),
        "critic2_td_loss": float(critic2_loss.cpu().item()),
        "holdout_td_loss": float(
            0.5 * (critic1_loss.cpu().item() + critic2_loss.cpu().item())
        ),
        "q1": q1.cpu().numpy().tolist(),
        "q2": q2.cpu().numpy().tolist(),
        "rewards": [float(row["reward"]) for row in records],
        "returns": list(return_contract["returns"]),
        "terminal_reasons": [str(row["terminal_reason"]) for row in records],
        "return_contract": return_contract,
        "value_policy_alignment": alignment,
    }


def _summarize_calibration_holdout_window(
    *, learner, torch, records, gamma, reward_scale, policy_context=None
):
    """Convert raw holdout tensors into the canonical calibration gate window."""

    context = dict(policy_context or {})
    alignment = assess_value_policy_alignment(
        behavior_policy=context.get("behavior_policy"),
        target_policy=context.get("target_policy"),
        return_policy=context.get("return_policy"),
    )
    try:
        holdout = _evaluate_calibration_holdout(
            learner=learner,
            torch=torch,
            records=records,
            gamma=float(gamma),
            reward_scale=float(reward_scale),
            policy_context=context,
        )
    except HoldoutEpisodeValidationError as error:
        # An incomplete/aborted holdout is evidence-contract failure, not a
        # fabricated terminal return and not a false Q non-finite finding.
        return {
            "finite": True,
            "reason": "holdout_episode_contract_error",
            "holdout_measurement_status": "CONTRACT_ERROR",
            "holdout_return_semantics": HOLDOUT_RETURN_SEMANTICS,
            "value_policy_alignment": alignment["status"],
            "value_policy_alignment_evidence": alignment,
            "holdout_return_contract": dict(error.report),
        }
    return summarize_calibration_window(
        critic1_td_loss=holdout["critic1_td_loss"],
        critic2_td_loss=holdout["critic2_td_loss"],
        holdout_td_loss=holdout["holdout_td_loss"],
        q1=holdout["q1"],
        q2=holdout["q2"],
        rewards=holdout["rewards"],
        returns=holdout["returns"],
        terminal_reasons=holdout["terminal_reasons"],
        gamma=float(gamma),
        reward_scale=float(reward_scale),
        holdout_return_semantics=str(
            holdout["return_contract"]["return_semantics"]
        ),
        holdout_measurement_status="PASS",
        value_policy_alignment=str(holdout["value_policy_alignment"]["status"]),
        value_policy_alignment_evidence=holdout["value_policy_alignment"],
        complete_episode_count=int(
            holdout["return_contract"]["complete_episode_count"]
        ),
        usable_row_count=int(holdout["return_contract"]["usable_row_count"]),
    )


def run_calibration_smoke(
    *,
    learner,
    torch,
    args,
    bc_checkpoint: Mapping[str, Any],
    bc_checkpoint_path: Path,
    mpl_sha256: str,
    training_contract: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run a bounded, offline Phase-0 mechanism smoke.

    This intentionally uses generated observations rather than Unity.  It
    validates the persistence, behavior-source, learner, checkpoint, and
    audit seams without creating formal data or claiming runtime evidence.
    """

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_episodes = min(int(args.calibration_smoke_episodes), 100)
    requested_transitions = min(int(args.calibration_smoke_transitions), 5000)
    episode_count = min(requested_episodes, requested_transitions)
    steps_per_episode = max(
        1, min(8, requested_transitions // max(1, episode_count))
    )
    total_transitions = episode_count * steps_per_episode
    replay_dir = output_dir / "replay"
    if replay_dir.exists() and any(replay_dir.iterdir()):
        raise FileExistsError("calibration smoke replay directory is not empty: {}".format(replay_dir))

    run_identity = "awac_phase0_critic_calibration_smoke_seed{}".format(args.seed)
    run_contract = {
        "contract_id": "awac_phase0_calibration_smoke_v1",
        "training_contract_sha256": awac_training_contract_sha256(training_contract),
        "run_identity": run_identity,
        "seed": int(args.seed),
        "episode_limit": episode_count,
        "transition_limit": requested_transitions,
        "runtime_mode": "synthetic_offline_smoke",
    }
    replay = AWACReplayBuffer.create(
        replay_dir,
        capacity=max(int(args.batch_size), total_transitions),
        depth_shape=(1, 32, 32),
        vector_dim=POLICY_VECTOR_DIM,
        action_dim=NUM_ACTIONS,
        training_config_sha256=awac_training_contract_sha256(training_contract),
        observation_contract=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        observation_source=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        bc_checkpoint_sha256=file_sha256(bc_checkpoint_path),
        task_contract_id=TASK_CONTRACT_ID,
        task_contract_sha256=task_contract_sha256(int(args.max_steps)),
        mpl_contract_sha256=str(mpl_sha256),
        run_identity=run_identity,
        run_contract_sha256=canonical_sha256(run_contract),
        reliable_v4_transition_count=0,
        behavior_source_phase=AWAC_PHASE_CRITIC_CALIBRATION,
    )
    rng = np.random.RandomState(int(args.seed))
    holdout_records = []
    train_episode_ids = []
    holdout_episode_ids = []
    behavior_total = 0
    behavior_mismatch = 0
    behavior_rng = np.random.RandomState(int(args.seed) + 1)
    calibration_behavior_policy = calibration_behavior_policy_identity(
        checkpoint_sha256=file_sha256(bc_checkpoint_path),
        rng_seed=int(args.seed) + 1,
    )
    policy_context = _calibration_value_policy_context(
        bc_checkpoint_sha256=file_sha256(bc_checkpoint_path),
        behavior_policy=calibration_behavior_policy,
    )
    actor_state_before = learner._state_fingerprint(learner.actor.state_dict())
    learner.freeze_actor_for_calibration()

    for episode in range(episode_count):
        episode_id = "calibration-smoke-{:04d}".format(episode)
        is_holdout = ((episode + 1) % 10) == 0
        (holdout_episode_ids if is_holdout else train_episode_ids).append(episode_id)
        terminal_reason = "success" if episode % 2 == 0 else "collision"
        for step in range(steps_per_episode):
            depth = rng.rand(1, 32, 32).astype(np.float32)
            vector = rng.randn(POLICY_VECTOR_DIM).astype(np.float32)
            next_depth = rng.rand(1, 32, 32).astype(np.float32)
            next_vector = rng.randn(POLICY_VECTOR_DIM).astype(np.float32)
            action_mask = np.ones(NUM_ACTIONS, dtype=np.bool_)
            next_action_mask = np.ones(NUM_ACTIONS, dtype=np.bool_)
            depth_tensor = torch.from_numpy(depth[None, ...])
            vector_tensor = torch.from_numpy(vector[None, ...])
            mask_tensor = torch.from_numpy(action_mask[None, ...])
            with torch.no_grad():
                actor_logits = learner.actor(depth_tensor, vector_tensor)
                bc_logits = learner.bc_reference(depth_tensor, vector_tensor)
                actor_action = torch.argmax(
                    actor_logits.masked_fill(~mask_tensor, -1.0e9), dim=1
                )
                bc_probabilities = torch.softmax(
                    bc_logits.masked_fill(~mask_tensor, -1.0e9), dim=1
                )
            bc_distribution = bc_probabilities[0].cpu().numpy().astype(
                np.float64, copy=False
            )
            bc_action = int(
                behavior_rng.choice(
                    np.arange(NUM_ACTIONS, dtype=np.int64),
                    p=bc_distribution / float(bc_distribution.sum()),
                )
            )
            behavior_total += 1
            if int(actor_action.item()) != int(bc_action):
                behavior_mismatch += 1
            done = step == steps_per_episode - 1
            reward = (
                30.0 if terminal_reason == "success" else -100.0
            ) if done else -0.02
            row = {
                "depth": depth,
                "vector": vector,
                "action_mask": action_mask,
                "action": int(bc_action),
                "reward": reward,
                "next_depth": next_depth,
                "next_vector": next_vector,
                "next_action_mask": next_action_mask,
                "done": done,
                "behavior_source": int(BehaviorSource.BC_CALIBRATION),
            }
            replay.add(**row)
            if is_holdout:
                holdout_row = dict(row)
                holdout_row.update(
                    {
                        "mission_id": "calibration-smoke-mission-{:04d}".format(
                            episode
                        ),
                        "episode_id": episode_id,
                        "episode_transition_index": int(step),
                        "terminal_reason": terminal_reason if done else "",
                    }
                )
                holdout_records.append(holdout_row)
    replay.flush()
    replay_report = audit_awac_replay(replay_dir)
    replay_identity = {
        "run_identity": run_identity,
        "replay_size": int(replay.size),
        "replay_metadata_sha256": file_sha256(replay_dir / "metadata.json"),
        "replay_contract_sha256": str(training_contract["replay_contract_sha256"]),
        "behavior_source_phase": "critic_calibration",
    }
    update_count = max(1, min(8, replay.size // max(1, int(args.batch_size))))
    last_update = None
    sample_rng = np.random.RandomState(int(args.seed) + 1)
    for _ in range(update_count):
        batch = replay.sample(
            min(int(args.batch_size), int(replay.size)),
            rng=sample_rng,
            torch=torch,
            device=learner.device,
        )
        last_update = learner.calibration_update(batch)
    if last_update is None:
        raise RuntimeError("calibration smoke did not perform a critic update")

    window = _summarize_calibration_holdout_window(
        learner=learner,
        torch=torch,
        records=holdout_records,
        gamma=float(args.gamma),
        reward_scale=float(args.reward_scale),
        policy_context=policy_context,
    )
    gate = evaluate_calibration_gate(
        windows=[window],
        replay_transitions=int(replay.size),
        completed_episodes=episode_count,
        critic_updates=int(learner.critic_update_count),
        holdout_episodes=len(holdout_episode_ids),
        config=_calibration_config(args),
    )
    calibration_controller = Phase1CalibrationController(
        safety_cap=_calibration_safety_cap(args)
    )
    calibration_status = calibration_controller.observe_gate(
        gate,
        transitions=int(replay.size),
        episodes=episode_count,
    )
    calibration_metrics = {
        "state": gate["state"],
        "window": window,
        "gate": gate,
        "behavior_source": "bc_calibration",
        "completed_episode_count": episode_count,
        "holdout_episode_count": len(holdout_episode_ids),
        "replay_transition_count": int(replay.size),
        "bc_calibration_action_total": behavior_total,
        "bc_calibration_action_mismatch_count": behavior_mismatch,
        "behavior_policy": calibration_behavior_policy,
        "behavior_rng_owner": calibration_behavior_policy["rng_owner"],
        "calibration_certification": calibration_status[
            "calibration_certification"
        ],
        "calibration_safety_cap": calibration_status[
            "calibration_safety_cap"
        ],
    }
    split = _load_calibration_split(
        str(args.calibration_split),
        fallback={
            "contract_id": "awac_calibration_episode_split_v1",
            "seed": int(args.seed),
            "train_episode_ids": train_episode_ids,
            "holdout_episode_ids": holdout_episode_ids,
            "source": "calibration_smoke_generated_episodes",
        },
    )
    checkpoint_payload = build_calibration_checkpoint_payload(
        learner=learner,
        bc_checkpoint=bc_checkpoint,
        bc_checkpoint_sha256=file_sha256(bc_checkpoint_path),
        mpl_contract_sha256=str(mpl_sha256),
        training_contract_sha256=awac_training_contract_sha256(training_contract),
        training_contract=training_contract,
        calibration_split=split,
        replay_identity=replay_identity,
        environment_step_count=int(replay.size),
        calibration_metrics=calibration_metrics,
    )
    checkpoint_path = output_dir / "checkpoint_calibration.pt"
    save_calibration_checkpoint(checkpoint_path, checkpoint_payload, torch=torch)
    loaded_checkpoint = load_calibration_checkpoint(
        checkpoint_path, torch=torch, map_location="cpu"
    )
    actor_state_after = learner._state_fingerprint(learner.actor.state_dict())
    replay.close()
    return {
        "runtime_mode": "synthetic_offline_smoke",
        "smoke_episode_count": episode_count,
        "smoke_transition_count": int(replay_report["row_count"]),
        "replay_dir": str(replay_dir),
        "replay_report": replay_report,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_phase": loaded_checkpoint["phase"],
        "calibration_gate": gate,
        "calibration_metrics": calibration_metrics,
        "behavior_action_total": behavior_total,
        "behavior_action_mismatch_count": behavior_mismatch,
        "actor_state_before": actor_state_before,
        "actor_state_after": actor_state_after,
        "actor_unchanged": actor_state_before == actor_state_after,
        "critic_update_count": int(learner.critic_update_count),
        "environment_step_count": int(replay_report["row_count"]),
        "holdout_episode_count": len(holdout_episode_ids),
    }


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    torch, nn, _, _, _ = require_torch()
    torch.set_num_threads(max(1, int(args.cpu_threads)))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    bc_path = Path(args.bc_checkpoint).expanduser().resolve()
    checkpoint = _load_bc_checkpoint(bc_path, torch=torch)
    mpl_sha = str(args.mpl_contract_sha256 or checkpoint.get("mpl_contract_sha256", ""))
    validate_bc_checkpoint_for_awac(checkpoint, mpl_contract_sha256=mpl_sha)
    config = _optimization_config(args)
    learner = build_learner(
        torch=torch, nn=nn, device=device, bc_checkpoint=checkpoint, config=config
    )
    learner.assert_optimizer_lrs_match_config(context="initial learner construction")
    resolved = _resolved_training_config(
        args, source_bc_sha256=file_sha256(bc_path), mpl_sha256=mpl_sha
    )
    resume_payload = None
    handoff_replay_identity = None
    if (
        str(args.phase) == AWAC_PHASE_STANDARD_TRAINING
        and str(args.resume_checkpoint).strip()
    ):
        handoff_replay = AWACReplayBuffer.open(
            Path(args.replay_dir), read_only=True
        )
        try:
            handoff_replay_identity = calibration_replay_identity(handoff_replay)
        finally:
            handoff_replay.close()
    if str(args.resume_checkpoint).strip():
        resume_payload = _load_resume_checkpoint(
            Path(args.resume_checkpoint).expanduser().resolve(),
            torch=torch,
            bc_checkpoint=checkpoint,
            bc_checkpoint_path=bc_path,
            args=args,
            mpl_sha256=mpl_sha,
            resolved_training_config=resolved,
            expected_replay_identity=handoff_replay_identity,
        )
        learner.load_state_dict(resume_payload)
        if str(args.phase) == AWAC_PHASE_STANDARD_TRAINING:
            parent_sha256 = file_sha256(
                Path(args.resume_checkpoint).expanduser().resolve()
            )
            if resume_payload.get("phase") == AWAC_PHASE_STANDARD_TRAINING:
                lr_mode = "exact_resume_checkpoint_optimizer_state"
                lr_overrides = {}
                learner.assert_optimizer_lrs_match_config(
                    context="Standard AWAC exact resume"
                )
            else:
                # A Calibration PASS is a controlled handoff, not an exact
                # resume.  Restore Adam state first, then apply only the
                # explicitly resolved Critic LR fork by stable group name.
                lr_overrides = {
                    "critic_critic1_head": float(config.critic_head_lr),
                    "critic_critic1_vector_encoder": float(config.critic_vector_lr),
                    "critic_critic1_depth_encoder": float(config.critic_depth_lr),
                    "critic_critic2_head": float(config.critic_head_lr),
                    "critic_critic2_vector_encoder": float(config.critic_vector_lr),
                    "critic_critic2_depth_encoder": float(config.critic_depth_lr),
                }
                learner.apply_optimizer_lr_overrides(
                    lr_overrides,
                    parent_checkpoint_sha256=parent_sha256,
                )
                lr_mode = "explicit_named_calibration_handoff_override"
                learner.assert_optimizer_lrs_match_config(
                    context="Standard AWAC calibration handoff"
                )
            resolved["optimizer_lr_resolution"] = {
                "mode": lr_mode,
                "parent_checkpoint_sha256": parent_sha256,
                "overrides": dict(sorted(lr_overrides.items())),
                "actual_group_lrs": learner.optimizer_lr_inventory(),
                "config_match_verified_before_first_update": True,
            }
        else:
            learner.assert_optimizer_lrs_match_config(
                context="AWAC checkpoint resume"
            )
        if str(args.phase) == AWAC_PHASE_CRITIC_CALIBRATION and str(
            args.calibration_split
        ).strip():
            split_for_resume = _load_calibration_split(
                str(args.calibration_split),
                fallback={
                    "contract_id": "awac_calibration_episode_split_v1",
                    "seed": int(args.seed),
                    "train_episode_ids": ["external-train-source"],
                    "holdout_episode_ids": ["external-holdout-source"],
                },
            )
            expected_split_hash = canonical_sha256(split_for_resume)
            if resume_payload.get("calibration_split_sha256") != expected_split_hash:
                raise ValueError(
                    "AWAC resume calibration split identity mismatch: received={} expected={}".format(
                        resume_payload.get("calibration_split_sha256"),
                        expected_split_hash,
                    )
                )
    if (
        bool(args.dry_run)
        and str(args.phase) == AWAC_PHASE_CRITIC_CALIBRATION
        and str(args.train_index).strip()
    ):
        # A dry-run still validates the actual mission source and its
        # final-test exclusion.  It deliberately constructs no pool.
        load_calibration_missions(
            Path(args.train_index), max_steps=int(args.max_steps)
        )
    if args.dry_run:
        print(json.dumps({
            "algorithm_id": AWAC_ALGORITHM_ID,
            "checkpoint_contract_id": AWAC_CHECKPOINT_CONTRACT_ID,
            "phase": str(args.phase),
            "training_contract_sha256": str(resolved["training_contract_sha256"]),
            "task_contract_sha256": task_contract_sha256(int(args.max_steps)),
            "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
            "vec_dim": POLICY_VECTOR_DIM,
            "num_actions": NUM_ACTIONS,
            "resolved_training_config_sha256": canonical_sha256(resolved),
            "device": str(device),
        }, sort_keys=True))
        return 0
    if args.phase1_readiness_dry_run:
        dev_manifest_path = Path(args.phase1_dev_manifest).expanduser().resolve()
        if not dev_manifest_path.is_file():
            raise FileNotFoundError(
                "Phase 1 Dev manifest does not exist: {}".format(dev_manifest_path)
            )
        dev_manifest = json.loads(dev_manifest_path.read_text(encoding="utf-8"))
        readiness = build_phase1_readiness_dry_run(
            learner=learner,
            dev_manifest=dev_manifest,
            actor_update_count=int(getattr(learner, "actor_update_count", 0)),
            calibration_safety_cap=_calibration_safety_cap(args),
        )
        readiness["dev_manifest_path"] = str(dev_manifest_path)
        print(json.dumps(readiness, sort_keys=True, default=str))
        return 0
    if args.calibration_smoke or (
        args.synthetic_smoke and str(args.phase) == AWAC_PHASE_CRITIC_CALIBRATION
    ):
        smoke = run_calibration_smoke(
            learner=learner,
            torch=torch,
            args=args,
            bc_checkpoint=checkpoint,
            bc_checkpoint_path=bc_path,
            mpl_sha256=mpl_sha,
            training_contract=resolved["training_contract"],
            output_dir=Path(args.out_dir),
        )
        print(json.dumps({
            "AWAC_CRITIC_CALIBRATION_SMOKE": "PASS",
            "actor_unchanged": smoke["actor_unchanged"],
            "behavior_action_mismatch_count": smoke["behavior_action_mismatch_count"],
            "checkpoint_path": smoke["checkpoint_path"],
            "critic_update_count": smoke["critic_update_count"],
            "gate_state": smoke["calibration_gate"]["state"],
            "replay_audit": smoke["replay_report"]["status"],
            "runtime_mode": smoke["runtime_mode"],
            "transitions": smoke["smoke_transition_count"],
        }, sort_keys=True))
        return 0
    if (
        str(args.phase) == AWAC_PHASE_CRITIC_CALIBRATION
        and str(args.train_index).strip()
    ):
        online = _run_online_calibration(
            args=args,
            learner=learner,
            checkpoint=checkpoint,
            bc_path=bc_path,
            mpl_sha=mpl_sha,
            resolved=resolved,
            resume_payload=resume_payload,
            torch=torch,
        )
        print(
            json.dumps(
                {
                    "AWAC_FORMAL_CRITIC_CALIBRATION": "PASS"
                    if online["status"] == "PASS"
                    else "STOPPED",
                    "calibration_status": online["status"],
                    "calibration_stop_reason": online["stop_reason"],
                    "train_index_actually_consumed": True,
                    "env_workers_actually_consumed": int(args.env_workers),
                    "parallel_env_owner": "planning.runtime.parallel_env",
                    "replay_dir": str(
                        Path(args.replay_dir).expanduser().resolve()
                        if str(args.replay_dir).strip()
                        else Path(args.out_dir).expanduser().resolve() / "replay"
                    ),
                    "replay_size": int(online["replay_size"]),
                    "environment_step_count": int(online["environment_step_count"]),
                    "completed_episode_count": int(online["completed_episode_count"]),
                    "critic_update_count": int(online["critic_update_count"]),
                    "actor_unchanged": bool(online["actor_unchanged"]),
                    "actor_update_count": int(online["actor_update_count"]),
                    "actor_optimizer_step_count": int(online["actor_optimizer_step_count"]),
                    "checkpoint_path": str(online["checkpoint_path"]),
                    "mission_source_identity": online["mission_source_identity"],
                    "replay_identity": online["replay_identity"],
                },
                sort_keys=True,
            )
        )
        return 0 if online["status"] == "PASS" else 2
    if (
        str(args.phase) == AWAC_PHASE_STANDARD_TRAINING
        and resume_payload is not None
        and not bool(args.synthetic_smoke)
    ):
        online = _run_standard_awac_online(
            args=args,
            learner=learner,
            checkpoint=checkpoint,
            bc_path=bc_path,
            mpl_sha=mpl_sha,
            resolved=resolved,
            resume_payload=resume_payload,
            torch=torch,
        )
        print(
            json.dumps(
                {
                    "AWAC_STANDARD_ONLINE": "PASS",
                    "online_env_steps": int(online["online_env_steps"]),
                    "online_transitions": int(
                        online["online_transitions_committed"]
                    ),
                    "critic_update_count": int(online["critic_update_count"]),
                    "actor_update_count": int(online["actor_update_count"]),
                    "replay_dir": online["replay_dir"],
                    "checkpoint_path": online["checkpoint_path"],
                    "summary_path": online["summary_path"],
                    "train_index_actually_consumed": True,
                    "env_workers_actually_consumed": int(args.env_workers),
                    "online_budget_actually_consumed": True,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.synthetic_smoke:
        batch = _synthetic_batch(torch, batch_size=min(2, int(args.batch_size)), depth_channels=1)
        learner.update(batch, update_actor=False)
        out_dir = Path(args.out_dir).expanduser().resolve()
        split_hash = _split_manifest_sha256(args.split_manifest)
        payload = build_checkpoint_payload(
            learner=learner,
            args=args,
            bc_checkpoint=checkpoint,
            bc_checkpoint_path=bc_path,
            mpl_sha256=mpl_sha,
            replay_size=0,
            split_manifest_sha256=split_hash,
        )
        save_awac_checkpoint(out_dir / "checkpoint_smoke.pt", payload, torch=torch)
        print("AWAC_SYNTHETIC_SMOKE=PASS")
        return 0
    if not str(args.replay_dir).strip():
        raise RuntimeError(
            "online runtime integration must be selected explicitly; use --replay-dir or --synthetic-smoke"
        )
    replay = AWACReplayBuffer.open(Path(args.replay_dir))
    if resume_payload is not None and int(resume_payload["replay_size"]) != int(replay.size):
        raise ValueError(
            "AWAC resume replay size mismatch: checkpoint={} replay={}".format(
                resume_payload["replay_size"], replay.size
            )
        )
    if (
        resume_payload is not None
        and str(args.phase) == AWAC_PHASE_STANDARD_TRAINING
    ):
        handoff = validate_standard_awac_handoff_payload(
            resume_payload,
            expected_bc_checkpoint_sha256=file_sha256(bc_path),
            expected_bc_reference_fingerprint=actor_state_sha256(checkpoint),
            expected_mpl_contract_sha256=str(mpl_sha),
            expected_task_contract_sha256=task_contract_sha256(int(args.max_steps)),
            expected_replay_identity=calibration_replay_identity(replay),
        )
        learner.enable_actor_after_calibration_pass(handoff)
    if (
        resume_payload is not None
        and str(args.phase) == AWAC_PHASE_CRITIC_CALIBRATION
    ):
        _validate_calibration_replay_resume_identity(
            replay,
            resume_payload,
            resolved_training_config=resolved,
            bc_checkpoint_sha256=file_sha256(bc_path),
            mpl_sha256=mpl_sha,
            max_steps=int(args.max_steps),
        )
    if replay.size < int(args.batch_size):
        raise RuntimeError("AWAC replay is smaller than batch-size")
    batch = replay.sample(int(args.batch_size), rng=np.random.RandomState(int(args.seed)), torch=torch, device=device)
    out_dir = Path(args.out_dir).expanduser().resolve()
    if str(args.phase) == AWAC_PHASE_CRITIC_CALIBRATION:
        if replay.metadata.get("behavior_source_phase") != AWAC_PHASE_CRITIC_CALIBRATION:
            raise ValueError(
                "critic calibration requires replay behavior_source_phase=critic_calibration"
            )
        if not str(args.calibration_split).strip():
            raise RuntimeError(
                "critic calibration requires --calibration-split for a persistent replay"
            )
        split = _load_calibration_split(
            str(args.calibration_split),
            fallback={
                "contract_id": "awac_calibration_episode_split_v1",
                "seed": int(args.seed),
                "train_episode_ids": ["external-train-source"],
                "holdout_episode_ids": ["external-holdout-source"],
            },
        )
        metrics = learner.calibration_update(batch)
        calibration_metrics = {"state": "PENDING", "last_update": metrics}
        calibration_controller = Phase1CalibrationController(
            safety_cap=_calibration_safety_cap(args)
        )
        calibration_status = calibration_controller.observe_gate(
            calibration_metrics,
            transitions=int(replay.size),
            episodes=0,
        )
        calibration_metrics.update(
            {
                "calibration_certification": calibration_status[
                    "calibration_certification"
                ],
                "calibration_safety_cap": calibration_status[
                    "calibration_safety_cap"
                ],
            }
        )
        replay_identity = {
            "run_identity": str(replay.metadata["run_identity"]),
            "replay_size": int(replay.size),
            "replay_metadata_sha256": file_sha256(
                Path(args.replay_dir).expanduser().resolve() / "metadata.json"
            ),
            "replay_contract_sha256": str(
                resolved["training_contract"]["replay_contract_sha256"]
            ),
            "behavior_source_phase": AWAC_PHASE_CRITIC_CALIBRATION,
        }
        payload = build_calibration_checkpoint_payload(
            learner=learner,
            bc_checkpoint=checkpoint,
            bc_checkpoint_sha256=file_sha256(bc_path),
            mpl_contract_sha256=mpl_sha,
            training_contract_sha256=str(resolved["training_contract_sha256"]),
            training_contract=resolved["training_contract"],
            calibration_split=split,
            replay_identity=replay_identity,
            environment_step_count=int(args.total_env_steps),
            calibration_metrics=calibration_metrics,
        )
        if str(calibration_metrics.get("state")) == "PASS":
            pass_payload = build_calibration_pass_checkpoint_payload(
                learner=learner,
                bc_checkpoint=checkpoint,
                bc_checkpoint_sha256=file_sha256(bc_path),
                mpl_contract_sha256=mpl_sha,
                training_contract_sha256=str(resolved["training_contract_sha256"]),
                training_contract=resolved["training_contract"],
                calibration_split=split,
                replay_identity=replay_identity,
                environment_step_count=int(args.total_env_steps),
                calibration_metrics=calibration_metrics,
                actor_depth_lr=float(args.actor_depth_lr),
                critic_depth_lr=float(args.critic_depth_lr),
            )
            save_calibration_pass_checkpoint(
                out_dir / "checkpoint_calibration_pass.pt", pass_payload, torch=torch
            )
            load_calibration_pass_checkpoint(
                out_dir / "checkpoint_calibration_pass.pt", torch=torch, map_location="cpu"
            )
            print("AWAC_CRITIC_CALIBRATION_PASS_CHECKPOINT=PASS")
        else:
            save_calibration_checkpoint(
                out_dir / "checkpoint_calibration.pt", payload, torch=torch
            )
            load_calibration_checkpoint(
                out_dir / "checkpoint_calibration.pt", torch=torch, map_location="cpu"
            )
            print("AWAC_CRITIC_CALIBRATION_UPDATE=PASS")
    else:
        learner.update(batch, update_actor=True)
        payload = build_checkpoint_payload(
            learner=learner,
            args=args,
            bc_checkpoint=checkpoint,
            bc_checkpoint_path=bc_path,
            mpl_sha256=mpl_sha,
            replay_size=replay.size,
            split_manifest_sha256=_split_manifest_sha256(args.split_manifest),
        )
        save_awac_checkpoint(out_dir / "checkpoint_latest.pt", payload, torch=torch)
        print("AWAC_UPDATE=PASS")
    replay.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "build_checkpoint_payload",
    "build_learner",
    "build_parser",
    "build_phase1_calibration_controller",
    "build_phase1_state_machine",
    "build_managed_runtime_pool",
    "build_runtime_pool",
    "main",
    "run_calibration_smoke",
    "_run_standard_awac_online",
    "start_managed_calibration_runtime",
    "validate_bc_checkpoint_for_awac",
]
