"""Public Phase 1 readiness contracts, written before implementation."""

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def test_phase1_depth_learning_rates_fail_closed_at_zero():
    from planning.awac.phase1 import (
        Phase1ReadinessError,
        validate_phase1_depth_learning_rates,
    )

    with pytest.raises(Phase1ReadinessError, match="actor_depth_lr"):
        validate_phase1_depth_learning_rates(
            actor_depth_lr=0.0,
            critic_depth_lr=1.0e-5,
        )
    with pytest.raises(Phase1ReadinessError, match="critic_depth_lr"):
        validate_phase1_depth_learning_rates(
            actor_depth_lr=1.0e-6,
            critic_depth_lr=0.0,
        )


def test_phase1_positive_depth_learning_rates_require_real_optimizer_groups():
    torch = pytest.importorskip("torch")
    from planning.awac.model import build_actor, optimizer_parameter_groups
    from planning.awac.phase1 import validate_phase1_optimizer_groups

    actor = build_actor(torch.nn, depth_channels=1)
    critic = build_actor(torch.nn, depth_channels=1)
    actor_optimizer = torch.optim.Adam(
        optimizer_parameter_groups(
            actor,
            head_lr=1.0e-5,
            vector_lr=3.0e-6,
            depth_lr=1.0e-6,
        )
    )
    critic_optimizer = torch.optim.Adam(
        [
            dict(group, name="critic1_" + group["name"])
            for group in optimizer_parameter_groups(
                critic,
                head_lr=1.0e-4,
                vector_lr=1.0e-5,
                depth_lr=1.0e-5,
            )
        ]
    )

    report = validate_phase1_optimizer_groups(
        actor_optimizer,
        critic_optimizer,
        actor_depth_lr=1.0e-6,
        critic_depth_lr=1.0e-5,
    )

    assert report["actor_depth_group"] == "depth_encoder"
    assert report["critic_depth_group"] == "critic1_depth_encoder"
    assert report["actor_depth_lr"] == pytest.approx(1.0e-6)
    assert report["critic_depth_lr"] == pytest.approx(1.0e-5)


def test_calibration_checkpoint_creation_is_pass_only():
    from planning.awac.phase1 import (
        Phase1ReadinessError,
        require_calibration_pass,
    )

    with pytest.raises(Phase1ReadinessError, match="PASS"):
        require_calibration_pass("PENDING")
    with pytest.raises(Phase1ReadinessError, match="PASS"):
        require_calibration_pass("FAIL_DIVERGED")
    assert require_calibration_pass("PASS") is True


def test_phase1_state_machine_allows_only_pass_calibration_to_standard_awac():
    from planning.awac.phase1 import (
        PHASE_ACTOR_ENABLED_STANDARD_AWAC,
        PHASE_CRITIC_CALIBRATION,
        PHASE_DIVERGED,
        PHASE_STOPPED,
        Phase1StateMachine,
        Phase1ReadinessError,
    )

    machine = Phase1StateMachine()
    assert machine.phase == PHASE_CRITIC_CALIBRATION
    assert machine.actor_update_enabled is False
    assert machine.transition_from_calibration(
        gate_state="PENDING",
        actor_depth_lr=1.0e-6,
        critic_depth_lr=1.0e-5,
        env_step=10,
        replay_size=10,
        critic_update_count=1,
        actor_update_count=0,
        source_checkpoint="pending.pt",
        timestamp_utc="2026-09-04T00:00:00Z",
    ) is None
    assert machine.phase == PHASE_CRITIC_CALIBRATION

    with pytest.raises(Phase1ReadinessError, match="diverged"):
        machine.transition_from_calibration(
            gate_state="FAIL_DIVERGED",
            actor_depth_lr=1.0e-6,
            critic_depth_lr=1.0e-5,
            env_step=11,
            replay_size=11,
            critic_update_count=2,
            actor_update_count=0,
            source_checkpoint="diverged.pt",
            timestamp_utc="2026-09-04T00:00:01Z",
        )
    assert machine.phase == PHASE_DIVERGED
    assert machine.actor_update_enabled is False

    machine = Phase1StateMachine()
    record = machine.transition_from_calibration(
        gate_state="PASS",
        actor_depth_lr=1.0e-6,
        critic_depth_lr=1.0e-5,
        env_step=100,
        replay_size=100,
        critic_update_count=32,
        actor_update_count=0,
        source_checkpoint="checkpoint_calibration_pass.pt",
        timestamp_utc="2026-09-04T00:00:00Z",
    )
    assert record["from_phase"] == PHASE_CRITIC_CALIBRATION
    assert record["to_phase"] == PHASE_ACTOR_ENABLED_STANDARD_AWAC
    assert machine.phase == PHASE_ACTOR_ENABLED_STANDARD_AWAC
    assert machine.actor_update_enabled is True

    with pytest.raises(Phase1ReadinessError, match="transition"):
        machine.transition_from_calibration(
            gate_state="PASS",
            actor_depth_lr=1.0e-6,
            critic_depth_lr=1.0e-5,
            env_step=101,
            replay_size=101,
            critic_update_count=33,
            actor_update_count=0,
            source_checkpoint="checkpoint_calibration_pass.pt",
            timestamp_utc="2026-09-04T00:00:01Z",
        )


def test_phase1_milestones_save_and_request_dev_then_stop_on_collapse():
    from planning.awac.phase1 import (
        PHASE_STOPPED,
        Phase1StateMachine,
        build_phase1_milestones,
    )

    milestones = build_phase1_milestones()
    assert [item.transition_count for item in milestones] == [10000, 25000, 50000]
    assert [item.checkpoint_name for item in milestones] == [
        "checkpoint_awac_10k.pt",
        "checkpoint_awac_25k.pt",
        "checkpoint_awac_50k.pt",
    ]

    machine = Phase1StateMachine()
    machine.transition_from_calibration(
        gate_state="PASS",
        actor_depth_lr=1.0e-6,
        critic_depth_lr=1.0e-5,
        env_step=100,
        replay_size=100,
        critic_update_count=32,
        actor_update_count=0,
        source_checkpoint="checkpoint_calibration_pass.pt",
        timestamp_utc="2026-09-04T00:00:00Z",
    )
    saved = []
    evaluated = []

    result = machine.observe_online_progress(
        online_transitions=10000,
        save_checkpoint=lambda milestone: saved.append(milestone.checkpoint_name),
        evaluate_dev=lambda milestone: evaluated.append(milestone.transition_count)
        or {"success_rate": 0.79},
        bc_dev_success_rate=0.90,
    )

    assert result["milestones_processed"] == [10000]
    assert saved == ["checkpoint_awac_10k.pt"]
    assert evaluated == [10000]
    assert machine.phase == PHASE_STOPPED
    assert machine.stop_reason == "COLLAPSED"


def test_phase1_dev_selector_accepts_dev_and_rejects_final_only():
    from planning.awac.phase1 import validate_phase1_dev_manifest

    assert validate_phase1_dev_manifest({"role": "DEV", "mission_count": 100})[
        "role"
    ] == "DEV"
    for manifest in (
        {"role": "FINAL_TEST"},
        {"role": "FINAL_TEST_ONLY"},
        {"role": "DEV", "final_test_only": True},
    ):
        with pytest.raises(ValueError, match="FINAL_TEST"):
            validate_phase1_dev_manifest(manifest)


def test_calibration_safety_cap_is_typed_and_stops_pending_at_cap():
    from planning.awac.phase1 import (
        CALIBRATION_GATE_PENDING,
        PHASE_STOPPED,
        Phase1CalibrationController,
        Phase1CalibrationSafetyCap,
    )

    cap = Phase1CalibrationSafetyCap(max_transitions=30_000, max_episodes=1_000)
    assert cap.as_dict() == {"max_transitions": 30_000, "max_episodes": 1_000}
    assert cap.reached(transitions=29_999, episodes=999) is False
    assert cap.reached(transitions=30_000, episodes=999) is True
    assert cap.reached(transitions=1, episodes=1_000) is True

    controller = Phase1CalibrationController(safety_cap=cap)
    pending = controller.observe_gate(
        {"state": CALIBRATION_GATE_PENDING, "window_count": 1},
        transitions=29_999,
        episodes=999,
    )
    assert pending["calibration_certification"] == CALIBRATION_GATE_PENDING
    assert pending["phase"] != PHASE_STOPPED

    stopped = controller.observe_gate(
        {"state": CALIBRATION_GATE_PENDING, "window_count": 2},
        transitions=30_000,
        episodes=999,
    )
    assert stopped["calibration_certification"] == "BLOCKED_PENDING"
    assert stopped["phase"] == PHASE_STOPPED
    assert stopped["actor_update_enabled"] is False
    assert stopped["stop_reason"] == "CALIBRATION_SAFETY_CAP_REACHED"


def test_calibration_controller_preserves_pass_and_fail_without_enabling_actor():
    from planning.awac.phase1 import (
        Phase1CalibrationController,
        Phase1CalibrationSafetyCap,
        PHASE_CRITIC_CALIBRATION,
        PHASE_DIVERGED,
    )

    controller = Phase1CalibrationController(
        safety_cap=Phase1CalibrationSafetyCap()
    )
    passed = controller.observe_gate(
        {"state": "PASS", "q_mean": 1.25},
        transitions=256,
        episodes=16,
    )
    assert passed["calibration_gate_state"] == "PASS"
    assert passed["calibration_certification"] == "PASS"
    assert passed["phase"] == PHASE_CRITIC_CALIBRATION
    assert passed["actor_update_enabled"] is False
    assert passed["q_mean"] == 1.25

    controller = Phase1CalibrationController()
    failed = controller.observe_gate(
        {"state": "FAIL_DIVERGED", "reason": "nonfinite"},
        transitions=256,
        episodes=16,
    )
    assert failed["calibration_certification"] == "FAIL_DIVERGED"
    assert failed["phase"] == PHASE_DIVERGED
    assert failed["actor_update_enabled"] is False
    assert failed["stop_reason"] == "CALIBRATION_DIVERGED"


def test_calibration_pass_checkpoint_roundtrip_rejects_pending(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from planning.awac.calibration import (
        build_calibration_pass_checkpoint_payload,
    )
    from planning.awac.checkpoint import (
        load_calibration_pass_checkpoint,
        save_calibration_pass_checkpoint,
        validate_calibration_pass_checkpoint_payload,
    )
    from planning.awac.contract import (
        awac_training_contract_sha256,
        build_awac_training_contract,
    )
    from planning.awac.checkpoint import build_awac_checkpoint_identity
    from planning.awac.interaction import BehaviorSource
    from planning.awac.learner import AWACOptimizationConfig, DiscreteAWACLearner
    from planning.awac.phase1 import build_phase1_training_contract
    from planning.bc.model import build_model
    from planning.contracts.feature import NUM_ACTIONS, POLICY_VECTOR_DIM
    from planning.contracts.observation import EXACT_ENDPOINT_OBSERVATION_CONTRACT
    from planning.contracts.task import TASK_CONTRACT_ID, task_contract_sha256

    with pytest.raises(ValueError, match="gate must be PASS"):
        build_calibration_pass_checkpoint_payload(
            learner=None,
            bc_checkpoint={},
            bc_checkpoint_sha256="b" * 64,
            mpl_contract_sha256="m" * 64,
            training_contract_sha256="t" * 64,
            calibration_split={"train_episode_ids": ["e0"], "holdout_episode_ids": ["e1"]},
            replay_identity={"run_identity": "phase1"},
            environment_step_count=0,
            calibration_metrics={"state": "PENDING"},
            actor_depth_lr=1.0e-6,
            critic_depth_lr=1.0e-5,
        )

    torch.manual_seed(4026)
    bc_model = build_model(torch.nn, depth_channels=1)
    bc = {"model_state_dict": bc_model.state_dict()}
    learner = DiscreteAWACLearner(
        torch=torch,
        nn=torch.nn,
        device=torch.device("cpu"),
        bc_state_dict=bc["model_state_dict"],
        depth_channels=1,
        config=AWACOptimizationConfig(
            actor_depth_lr=1.0e-6,
            critic_depth_lr=1.0e-5,
        ),
    )
    learner.freeze_actor_for_calibration()
    mask = torch.ones((2, NUM_ACTIONS), dtype=torch.bool)
    learner.calibration_update(
        {
            "depth": torch.rand((2, 1, 32, 32)),
            "vector": torch.randn((2, POLICY_VECTOR_DIM)),
            "action_mask": mask,
            "action": torch.tensor([0, 1]),
            "reward": torch.tensor([1.0, -1.0]),
            "next_depth": torch.rand((2, 1, 32, 32)),
            "next_vector": torch.randn((2, POLICY_VECTOR_DIM)),
            "next_action_mask": mask.clone(),
            "done": torch.tensor([1.0, 1.0]),
            "behavior_source": torch.full(
                (2,), int(BehaviorSource.BC_CALIBRATION), dtype=torch.long
            ),
        }
    )
    training_contract = build_awac_training_contract(
        phase="critic_calibration", max_primitive_steps=45
    )
    phase1_contract = build_phase1_training_contract(
        actor_depth_lr=1.0e-6,
        critic_depth_lr=1.0e-5,
    )
    payload = build_calibration_pass_checkpoint_payload(
        learner=learner,
        bc_checkpoint=bc,
        bc_checkpoint_sha256="b" * 64,
        mpl_contract_sha256="m" * 64,
        training_contract_sha256=awac_training_contract_sha256(training_contract),
        training_contract=training_contract,
        calibration_split={
            "contract_id": "awac_calibration_episode_split_v1",
            "train_episode_ids": ["e0"],
            "holdout_episode_ids": ["e1"],
        },
        replay_identity={"run_identity": "phase1", "replay_size": 2},
        environment_step_count=2,
        calibration_metrics={"state": "PASS", "window_count": 3},
        actor_depth_lr=1.0e-6,
        critic_depth_lr=1.0e-5,
        phase1_training_contract=phase1_contract,
    )
    assert validate_calibration_pass_checkpoint_payload(payload)[
        "calibration_pass_checkpoint"
    ] is True
    identity = build_awac_checkpoint_identity(max_primitive_steps=45)
    assert {name: payload[name] for name in identity} == identity
    path = tmp_path / "checkpoint_calibration_pass.pt"
    save_calibration_pass_checkpoint(path, payload, torch=torch)
    loaded = load_calibration_pass_checkpoint(path, torch=torch)
    assert loaded["calibration_gate_state"] == "PASS"
    assert loaded["algorithm_id"] == "discrete_masked_awac"
    assert loaded["actor_update_count"] == 0
    assert loaded["actor_optimizer_step_count"] == 0
