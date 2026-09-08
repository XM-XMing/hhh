"""Independent trainer boundary tests; no Unity or long training."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from planning.bc.model import build_model, require_torch
from planning.contracts.feature import (
    CONTINUOUS_DIM,
    FEATURE_CONTRACT_ID,
    INITIAL_PREV_ACTION,
    NUM_ACTIONS,
    POLICY_VECTOR_DIM,
    policy_input_contract_sha256,
)
from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
from planning.contracts.policy_runtime import POLICY_RUNTIME_CONTRACT_ID
from planning.contracts.task import TASK_CONTRACT_ID, task_contract_sha256


def _bc(nn):
    model = build_model(nn, depth_channels=1)
    return {
        "model_state_dict": model.state_dict(),
        "observation_contract": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "observation_source": EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        "feature_contract_id": FEATURE_CONTRACT_ID,
        "policy_input_contract_sha256": policy_input_contract_sha256(),
        "task_contract_id": TASK_CONTRACT_ID,
        "task_contract_sha256": task_contract_sha256(),
        "mpl_contract_sha256": "m" * 64,
        "vec_dim": POLICY_VECTOR_DIM,
        "state_feature_dim": CONTINUOUS_DIM,
        "previous_action_dim": NUM_ACTIONS,
        "action_ordering": "motion_primitive_index_ascending",
        "num_actions": NUM_ACTIONS,
        "depth_history_frames": 1,
        "initial_prev_action": INITIAL_PREV_ACTION,
        "feature_mean": np.zeros(CONTINUOUS_DIM, dtype=np.float32),
        "feature_std": np.ones(CONTINUOUS_DIM, dtype=np.float32),
        "safety_mask": "depth",
        "execution_mode": "continuous",
        "policy_runtime_contract_id": POLICY_RUNTIME_CONTRACT_ID,
        "reliable_execution": True,
        "legacy_rows": 0,
    }


@pytest.mark.unit
def test_awac_parser_has_no_algorithm_switch_or_legacy_import():
    from planning.awac import trainer

    parser = trainer.build_parser()
    args = parser.parse_args(["--bc-checkpoint", "bc.pt", "--out-dir", "out"])
    assert not hasattr(args, "algorithm")
    assert ".".join(("planning", "rl")) not in inspect.getsource(trainer)


@pytest.mark.unit
def test_awac_trainer_strict_bc_initialization_and_depth_trainability():
    from planning.awac.trainer import build_learner, validate_bc_checkpoint_for_awac

    torch, nn, _, _, _ = require_torch()
    bc = _bc(nn)
    validate_bc_checkpoint_for_awac(bc, mpl_contract_sha256="m" * 64)
    config_module = __import__("planning.awac.learner", fromlist=["AWACOptimizationConfig"])
    config = config_module.AWACOptimizationConfig(
        actor_depth_lr=1.0e-5,
        critic_depth_lr=1.0e-5,
    )
    learner = build_learner(
        torch=torch, nn=nn, device=torch.device("cpu"), bc_checkpoint=bc, config=config
    )
    assert all(torch.equal(value, learner.actor.state_dict()[name]) for name, value in bc["model_state_dict"].items())
    assert any(parameter.requires_grad for parameter in learner.actor.depth_encoder.parameters())
    assert any(parameter.requires_grad for parameter in learner.critic1.depth_encoder.parameters())
    assert any(parameter.requires_grad for parameter in learner.critic2.depth_encoder.parameters())


@pytest.mark.unit
def test_standard_checkpoint_payload_inherits_shared_awac_identity(tmp_path):
    from planning.awac.checkpoint import build_awac_checkpoint_identity
    from planning.awac.trainer import (
        _optimization_config,
        build_checkpoint_payload,
        build_learner,
        build_parser,
    )

    torch, nn, _, _, _ = require_torch()
    bc = _bc(nn)
    bc_path = tmp_path / "bc.pt"
    torch.save(bc, bc_path)
    args = build_parser().parse_args(
        [
            "--bc-checkpoint", str(bc_path),
            "--out-dir", str(tmp_path / "out"),
            "--phase", "awac_training",
            "--device", "cpu",
            "--mpl-contract-sha256", "m" * 64,
        ]
    )
    learner = build_learner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_checkpoint=bc,
        config=_optimization_config(args),
    )
    payload = build_checkpoint_payload(
        learner=learner,
        args=args,
        bc_checkpoint=bc,
        bc_checkpoint_path=bc_path,
        mpl_sha256="m" * 64,
        replay_size=0,
        split_manifest_sha256="0" * 64,
    )
    identity = build_awac_checkpoint_identity(max_primitive_steps=45)

    assert {name: payload[name] for name in identity} == identity


@pytest.mark.unit
def test_awac_trainer_resume_surface_is_explicit():
    from planning.awac.trainer import build_parser

    args = build_parser().parse_args(
        [
            "--bc-checkpoint", "bc.pt",
            "--out-dir", "out",
            "--resume-checkpoint", "checkpoint.pt",
        ]
    )
    assert args.resume_checkpoint == "checkpoint.pt"


@pytest.mark.unit
def test_standard_cli_consumes_online_budget_and_runtime_inputs():
    from planning.awac.trainer import (
        _online_env_steps,
        _resolved_training_config,
        build_parser,
    )

    args = build_parser().parse_args(
        [
            "--bc-checkpoint", "bc.pt",
            "--phase", "awac_training",
            "--train-index", "missions.csv",
            "--out-dir", "out",
            "--resume-checkpoint", "checkpoint_calibration_pass.pt",
            "--replay-dir", "replay",
            "--mpl-contract-sha256", "m" * 64,
            "--online-env-steps", "3965",
            "--env-workers", "2",
            "--worker-spec-file", "worker_runtime_specs.json",
            "--reliable-v4",
        ]
    )

    assert _online_env_steps(args) == 3965
    resolved = _resolved_training_config(
        args,
        source_bc_sha256="b" * 64,
        mpl_sha256="m" * 64,
    )
    assert resolved["online_env_steps"] == 3965
    assert resolved["online_env_steps_budget"] == 3965
    assert resolved["env_workers"] == 2
    assert resolved["train_index"].endswith("/missions.csv")
    assert resolved["online_budget_counter_owner"] == (
        "phase_local_online_environment_steps"
    )


def test_standard_handoff_allows_only_explicit_critic_lr_overrides():
    from planning.awac.config_identity import compare_awac_config_identity
    from planning.awac.trainer import _standard_handoff_config_identity_is_valid

    calibration = {
        "training_contract": {
            "phase": "critic_calibration",
            "actor_update_enabled": False,
            "gamma": 0.99,
        },
        "optimization_config": {
            "critic_depth_lr": 1.0e-5,
            "critic_vector_lr": 1.0e-5,
            "critic_head_lr": 1.0e-4,
            "actor_head_lr": 1.0e-5,
        },
    }
    candidate = {
        "training_contract": {
            "phase": "awac_training",
            "actor_update_enabled": True,
            "gamma": 0.99,
        },
        "optimization_config": {
            "critic_depth_lr": 3.0e-6,
            "critic_vector_lr": 3.0e-6,
            "critic_head_lr": 3.0e-5,
            "actor_head_lr": 1.0e-5,
        },
        "phase1_training_contract": {"actor_depth_lr": 1.0e-6},
    }
    report = compare_awac_config_identity(calibration, candidate)
    assert _standard_handoff_config_identity_is_valid(report)

    changed_actor = dict(candidate)
    changed_actor["optimization_config"] = {
        **candidate["optimization_config"],
        "actor_head_lr": 2.0e-5,
    }
    actor_report = compare_awac_config_identity(calibration, changed_actor)
    assert not _standard_handoff_config_identity_is_valid(actor_report)

    changed_contract = dict(candidate)
    changed_contract["training_contract"] = {
        **candidate["training_contract"],
        "gamma": 0.98,
    }
    contract_report = compare_awac_config_identity(calibration, changed_contract)
    assert not _standard_handoff_config_identity_is_valid(contract_report)
