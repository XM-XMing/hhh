"""Fail-closed Calibration PASS to Standard AWAC handoff tests."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


class _Module:
    def __init__(self, torch):
        self._state = {"weight": torch.tensor([1.0])}

    def state_dict(self):
        return dict(self._state)


class _CalibrationLearner:
    def __init__(self, torch):
        self.torch = torch
        self.config = SimpleNamespace(gamma=0.99, tau=0.005)
        self.actor = _Module(torch)
        self.critic1 = _Module(torch)
        self.critic2 = _Module(torch)
        self.update_step = 7
        self.actor_update_count = 0
        self.actor_optimizer_step_count = 0
        self.critic_update_count = 1
        self.actor_frozen_for_calibration = True

    def state_dict(self):
        optimizer = {"state": {}, "param_groups": [{"lr": 1.0e-5}]}
        return {
            "awac_learner_state_schema_id": "awac_learner_state_v1",
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "target_critic1_state_dict": self.critic1.state_dict(),
            "target_critic2_state_dict": self.critic2.state_dict(),
            "actor_optimizer_state_dict": optimizer,
            "critic_optimizer_state_dict": optimizer,
            "update_step": self.update_step,
            "actor_update_count": 0,
            "actor_awac_update_count": 0,
            "actor_recovery_update_count": 0,
            "actor_trust_region_rejection_count": 0,
            "actor_optimizer_step_count": 0,
            "critic_update_count": self.critic_update_count,
            "actor_frozen_for_calibration": True,
        }


def _pass_payload():
    from planning.awac.calibration import build_calibration_pass_checkpoint_payload
    from planning.awac.checkpoint import build_calibration_exact_resume_state
    from planning.awac.contract import (
        awac_training_contract_sha256,
        build_awac_training_contract,
    )
    from planning.awac.phase1 import build_phase1_training_contract

    from planning.bc.model import require_torch

    torch, _, _, _, _ = require_torch()
    learner = _CalibrationLearner(torch)
    bc = {"model_state_dict": learner.actor.state_dict()}
    training_contract = build_awac_training_contract(
        phase="critic_calibration", max_primitive_steps=45
    )
    payload = build_calibration_pass_checkpoint_payload(
        learner=learner,
        bc_checkpoint=bc,
        bc_checkpoint_sha256="b" * 64,
        mpl_contract_sha256="d" * 64,
        training_contract_sha256=awac_training_contract_sha256(training_contract),
        training_contract=training_contract,
        calibration_split={"train_episode_ids": ["train"], "holdout_episode_ids": ["holdout"]},
        replay_identity={
            "run_identity": "handoff-test",
            "mission_source_sha256": "f" * 64,
            "mission_index_sha256": "f" * 64,
            "replay_size": 7,
            "replay_position": 7,
            "replay_total_added": 7,
            "replay_metadata_sha256": "e" * 64,
            "replay_contract_sha256": "r" * 64,
            "behavior_source_phase": "critic_calibration",
        },
        environment_step_count=7,
        calibration_metrics={"state": "PASS"},
        actor_depth_lr=1.0e-6,
        critic_depth_lr=1.0e-5,
        phase1_training_contract=build_phase1_training_contract(
            actor_depth_lr=1.0e-6,
            critic_depth_lr=1.0e-5,
        ),
        mission_source_identity={"sha256": "f" * 64},
        mission_progress={
            "schema_id": "awac_formal_calibration_runtime_v1",
            "mission_source_identity": {"sha256": "f" * 64},
            "next_mission_index": 0,
            "completed_mission_ids": [],
            "environment_step_count": 7,
            "completed_episode_count": 0,
            "critic_update_count": 1,
            "gate_state": "PASS",
            "gate_history": [],
            "holdout_completed_ids": [],
        },
        exact_resume_state=build_calibration_exact_resume_state(
            raw_holdout_records=[],
            producer_rng=__import__("numpy").random.RandomState(4027),
            torch=torch,
        ),
    )
    payload["checkpoint_transaction"] = {
        "schema_id": "awac_calibration_checkpoint_transaction_v1",
        "generation": 1,
        "checkpoint_name": "checkpoint_calibration_pass",
        "checkpoint_kind": "checkpoint_calibration_pass",
        "checkpoint_filename": "checkpoint_calibration_pass.generation-00000001.pt",
        "transaction_state": "COMMITTED",
        "checkpoint_final_commit": "PASS",
    }
    return payload, bc


def test_standard_handoff_accepts_only_a_committed_calibration_pass():
    from planning.awac.checkpoint import validate_standard_awac_handoff_payload

    payload, _ = _pass_payload()
    expected = {
        "expected_bc_checkpoint_sha256": "b" * 64,
        "expected_bc_reference_fingerprint": payload["bc_reference_fingerprint"],
        "expected_mpl_contract_sha256": "d" * 64,
        "expected_task_contract_sha256": payload["task_contract_sha256"],
        "expected_replay_identity": payload["replay_identity"],
    }
    handoff = validate_standard_awac_handoff_payload(payload, **expected)
    assert handoff["handoff_allowed"] is True
    assert handoff["to_phase"] == "ACTOR_ENABLED_STANDARD_AWAC"
    assert handoff["actor_update_enabled"] is True

    pending = copy.deepcopy(payload)
    pending["calibration_gate_state"] = "PENDING"
    with pytest.raises(ValueError, match="PASS"):
        validate_standard_awac_handoff_payload(pending, **expected)

    uncommitted = copy.deepcopy(payload)
    uncommitted["checkpoint_transaction"]["transaction_state"] = "PENDING"
    with pytest.raises(ValueError, match="committed"):
        validate_standard_awac_handoff_payload(uncommitted, **expected)

    replay_mismatch = dict(expected["expected_replay_identity"])
    replay_mismatch["replay_size"] += 1
    with pytest.raises(ValueError, match="replay identity"):
        validate_standard_awac_handoff_payload(
            payload,
            **{
                **expected,
                "expected_replay_identity": replay_mismatch,
            },
        )


def test_standard_training_requires_a_calibration_pass_resume_checkpoint():
    from planning.awac.phase1 import Phase1ReadinessError
    from planning.awac.trainer import _validate_args, build_parser

    args = build_parser().parse_args(
        [
            "--bc-checkpoint",
            "bc.pt",
            "--out-dir",
            "out",
            "--phase",
            "awac_training",
            "--replay-dir",
            "replay",
        ]
    )
    with pytest.raises(Phase1ReadinessError, match="calibration PASS"):
        _validate_args(args)


def test_learner_actor_enablement_requires_handoff_and_only_changes_grad_state():
    torch, nn, _, _, _ = __import__(
        "planning.bc.model", fromlist=["require_torch"]
    ).require_torch()
    from planning.awac.learner import AWACOptimizationConfig, DiscreteAWACLearner
    from planning.awac.checkpoint import validate_standard_awac_handoff_payload
    from planning.contracts.policy_checkpoint_fingerprint import actor_state_sha256

    payload, _ = _pass_payload()
    handoff = validate_standard_awac_handoff_payload(
        payload,
        expected_bc_checkpoint_sha256="b" * 64,
        expected_bc_reference_fingerprint=payload["bc_reference_fingerprint"],
        expected_mpl_contract_sha256="d" * 64,
        expected_task_contract_sha256=payload["task_contract_sha256"],
        expected_replay_identity=payload["replay_identity"],
    )
    model = __import__("planning.bc.model", fromlist=["build_model"]).build_model(
        nn, depth_channels=1
    )
    learner = DiscreteAWACLearner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_state_dict=model.state_dict(),
        depth_channels=1,
        config=AWACOptimizationConfig(
            actor_depth_lr=1.0e-6,
            critic_depth_lr=1.0e-5,
        ),
    )
    learner.freeze_actor_for_calibration()
    with pytest.raises(RuntimeError, match="validated Calibration PASS"):
        learner.enable_actor_after_calibration_pass({})
    before = actor_state_sha256({"actor_state_dict": learner.actor.state_dict()})
    learner.enable_actor_after_calibration_pass(handoff)
    after = actor_state_sha256({"actor_state_dict": learner.actor.state_dict()})
    assert before == after
    assert learner.actor_frozen_for_calibration is False
    assert all(parameter.requires_grad for parameter in learner.actor.parameters())
    assert learner.actor_optimizer_step_count == 0
