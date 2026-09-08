"""Critic Calibration learner and checkpoint seams."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import planning.awac.trainer as trainer
from planning.awac.calibration import (
    CALIBRATION_GATE_CONTRACT_ID,
    CriticCalibrationConfig,
    calibration_state_sha256,
    build_calibration_checkpoint_payload,
)
from planning.awac.interaction import BehaviorSource
from planning.awac.learner import (
    AWACOptimizationConfig,
    DiscreteAWACLearner,
    resolve_torch_device,
)
from planning.awac.replay import AWACReplayBuffer
from planning.awac.contract import (
    awac_training_contract_sha256,
    build_awac_training_contract,
)
from planning.awac.checkpoint import (
    build_awac_checkpoint_identity,
    build_calibration_exact_resume_state,
    validate_exact_calibration_resume_payload,
)
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
from planning.contracts.reward import reward_contract_sha256
from planning.contracts.task import TASK_CONTRACT_ID, task_contract_sha256
from planning.common.hashing import file_sha256


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


def _batch(torch, batch_size=6):
    mask = torch.zeros((batch_size, NUM_ACTIONS), dtype=torch.bool)
    mask[:, :4] = True
    return {
        "depth": torch.rand((batch_size, 1, 32, 32)),
        "vector": torch.randn((batch_size, POLICY_VECTOR_DIM)),
        "action_mask": mask,
        "action": torch.arange(batch_size, dtype=torch.long) % 4,
        "reward": torch.linspace(-1.0, 1.0, batch_size),
        "next_depth": torch.rand((batch_size, 1, 32, 32)),
        "next_vector": torch.randn((batch_size, POLICY_VECTOR_DIM)),
        "next_action_mask": mask.clone(),
        "done": torch.zeros((batch_size,), dtype=torch.float32),
        "behavior_source": torch.full(
            (batch_size,),
            int(BehaviorSource.BC_CALIBRATION),
            dtype=torch.long,
        ),
    }


def _holdout_records(torch):
    batch = _batch(torch, batch_size=2)
    batch["done"][1] = 1.0
    batch["next_action_mask"][1] = False
    records = []
    for index in range(2):
        records.append(
            {
                "depth": batch["depth"][index].numpy(),
                "vector": batch["vector"][index].numpy(),
                "action_mask": batch["action_mask"][index].numpy(),
                "action": int(batch["action"][index].item()),
                "reward": float(batch["reward"][index].item()),
                "next_depth": batch["next_depth"][index].numpy(),
                "next_vector": batch["next_vector"][index].numpy(),
                "next_action_mask": batch["next_action_mask"][index].numpy(),
                "done": float(batch["done"][index].item()),
                "behavior_source": int(batch["behavior_source"][index].item()),
                "mission_id": "holdout-mission",
                "episode_id": "holdout-episode",
                "episode_transition_index": index,
                "terminal_reason": "dead_end" if index == 1 else "",
            }
        )
    return records


def _calibration_learner(torch, nn, *, device):
    bc = _bc(nn)
    learner = DiscreteAWACLearner(
        torch=torch,
        nn=nn,
        device=device,
        bc_state_dict=bc["model_state_dict"],
        depth_channels=1,
        config=AWACOptimizationConfig(critic_depth_lr=1.0e-5),
    )
    learner.freeze_actor_for_calibration()
    return learner


class _DeviceOnlyParameter:
    def __init__(self, device):
        self.device = device


class _DeviceOnlyModule:
    def __init__(self, torch, device):
        self._parameter = _DeviceOnlyParameter(torch.device(device))

    def parameters(self):
        return iter((self._parameter,))


def _device_only_holdout_learner(torch, *, learner_device, critic_device):
    class _Learner:
        pass

    learner = _Learner()
    learner.device = torch.device(learner_device)
    learner.critic1 = _DeviceOnlyModule(torch, critic_device)
    learner.critic2 = _DeviceOnlyModule(torch, critic_device)
    learner.target_critic1 = _DeviceOnlyModule(torch, critic_device)
    learner.target_critic2 = _DeviceOnlyModule(torch, critic_device)
    learner.bc_reference = _DeviceOnlyModule(torch, critic_device)
    return learner


def _state_equal(torch, before, after):
    return all(torch.equal(before[name], after[name]) for name in before)


@pytest.mark.unit
def test_calibration_step_freezes_actor_and_updates_both_independent_critics():
    torch, nn, _, _, _ = require_torch()
    torch.manual_seed(4026)
    bc = _bc(nn)
    learner = DiscreteAWACLearner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_state_dict=bc["model_state_dict"],
        depth_channels=1,
        config=AWACOptimizationConfig(
            actor_vector_lr=0.0,
            actor_depth_lr=0.0,
            critic_vector_lr=1.0e-5,
            critic_depth_lr=1.0e-5,
            reward_scale=0.10,
        ),
    )
    learner.freeze_actor_for_calibration()
    actor_before = copy.deepcopy(learner.actor.state_dict())
    critic1_before = copy.deepcopy(learner.critic1.state_dict())
    critic2_before = copy.deepcopy(learner.critic2.state_dict())

    metrics = learner.calibration_update(_batch(torch))

    assert learner.actor.training is False
    assert all(not parameter.requires_grad for parameter in learner.actor.parameters())
    assert _state_equal(torch, actor_before, learner.actor.state_dict())
    assert not _state_equal(torch, critic1_before, learner.critic1.state_dict())
    assert not _state_equal(torch, critic2_before, learner.critic2.state_dict())
    assert learner.actor_optimizer_step_count == 0
    assert learner.critic_update_count == 1
    assert metrics["calibration_actor_update_enabled"] == 0.0
    assert metrics["critic_gradients_finite"] is True

    q1_head = list(learner.critic1.head.parameters())
    q2_head = list(learner.critic2.head.parameters())
    assert q1_head[0].data_ptr() != q2_head[0].data_ptr()
    assert learner.target_critic1 is not learner.target_critic2
    assert list(learner.target_critic1.head.parameters())[0].data_ptr() != q1_head[0].data_ptr()


@pytest.mark.unit
def test_calibration_checkpoint_roundtrip_records_phase_and_resume_identity(tmp_path: Path):
    torch, nn, _, _, _ = require_torch()
    torch.manual_seed(4027)
    bc = _bc(nn)
    learner = DiscreteAWACLearner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_state_dict=bc["model_state_dict"],
        depth_channels=1,
        config=AWACOptimizationConfig(critic_depth_lr=1.0e-5),
    )
    learner.freeze_actor_for_calibration()
    learner.calibration_update(_batch(torch, batch_size=4))
    split = {
        "contract_id": "awac_calibration_episode_split_v1",
        "seed": 4026,
        "train_episode_ids": ["e0", "e1"],
        "holdout_episode_ids": ["e2"],
    }
    training_contract = build_awac_training_contract(
        phase="critic_calibration",
        calibration=CriticCalibrationConfig().as_dict(),
    )
    payload = build_calibration_checkpoint_payload(
        learner=learner,
        bc_checkpoint=bc,
        bc_checkpoint_sha256="b" * 64,
        mpl_contract_sha256="m" * 64,
        training_contract_sha256=awac_training_contract_sha256(training_contract),
        training_contract=training_contract,
        calibration_split=split,
        replay_identity={"run_identity": "phase0-smoke", "replay_sha256": "r" * 64},
        environment_step_count=4,
        calibration_metrics={"state": "PENDING"},
    )
    assert payload["phase"] == "critic_calibration"
    assert payload["algorithm_id"] == "discrete_masked_awac"
    identity = build_awac_checkpoint_identity(max_primitive_steps=45)
    assert {name: payload[name] for name in identity} == identity
    assert payload["actor_update_enabled"] is False
    assert payload["actor_optimizer_status"] == "unused_frozen"
    assert payload["critic_update_count"] == 1
    assert payload["calibration_split"]["holdout_episode_ids"] == ["e2"]
    assert payload["reward_contract_sha256"] == reward_contract_sha256()
    assert payload["actor_state_sha256"] == calibration_state_sha256(
        learner.actor.state_dict(), torch
    )
    assert payload["calibration_gate_contract"]["contract_id"] == (
        CALIBRATION_GATE_CONTRACT_ID
    )
    assert len(payload["calibration_gate_contract_sha256"]) == 64

    from planning.awac.checkpoint import (
        load_calibration_checkpoint,
        save_calibration_checkpoint,
        validate_calibration_checkpoint_payload,
    )

    validated = validate_calibration_checkpoint_payload(payload)
    assert validated["phase"] == "critic_calibration"
    assert validated["calibration_gate_state"] == "PENDING"
    for filename in ("checkpoint_periodic.pt", "checkpoint_last.pt"):
        path = tmp_path / filename
        save_calibration_checkpoint(path, payload, torch=torch)
        loaded = load_calibration_checkpoint(path, torch=torch)
        assert loaded["algorithm_id"] == "discrete_masked_awac"
        assert loaded["phase"] == "critic_calibration"

    resumed = DiscreteAWACLearner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_state_dict=bc["model_state_dict"],
        depth_channels=1,
        config=AWACOptimizationConfig(critic_depth_lr=1.0e-5),
    )
    resumed.load_state_dict(validated)
    assert resumed.actor_frozen_for_calibration is True
    assert resumed.actor_optimizer_step_count == 0
    assert resumed.critic_update_count == 1
    assert calibration_state_sha256(resumed.actor.state_dict(), torch) == validated[
        "actor_state_sha256"
    ]

    bad = copy.deepcopy(payload)
    bad["training_contract_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="training contract"):
        validate_calibration_checkpoint_payload(bad)

    missing_algorithm = copy.deepcopy(payload)
    del missing_algorithm["algorithm_id"]
    with pytest.raises(ValueError, match="algorithm_id"):
        validate_calibration_checkpoint_payload(missing_algorithm)

    wrong_algorithm = copy.deepcopy(payload)
    wrong_algorithm["algorithm_id"] = "sac"
    with pytest.raises(ValueError, match="algorithm_id"):
        validate_calibration_checkpoint_payload(wrong_algorithm)

    wrong_reward_contract = copy.deepcopy(payload)
    wrong_reward_contract["reward_contract_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="reward contract"):
        validate_calibration_checkpoint_payload(wrong_reward_contract)


@pytest.mark.unit
def test_exact_calibration_resume_payload_requires_raw_holdout_and_every_rng(
    tmp_path: Path,
):
    """Exact resume cannot silently degrade into seed-only continuation."""

    torch, nn, _, _, _ = require_torch()
    learner = _calibration_learner(torch, nn, device=torch.device("cpu"))
    split = {
        "contract_id": "awac_calibration_episode_split_v1",
        "seed": 4026,
        "train_episode_ids": ["e0"],
        "holdout_episode_ids": ["e1"],
    }
    exact_resume_state = build_calibration_exact_resume_state(
        raw_holdout_records=[],
        producer_rng=np.random.RandomState(4027),
        torch=torch,
    )
    payload = build_calibration_checkpoint_payload(
        learner=learner,
        bc_checkpoint=_bc(nn),
        bc_checkpoint_sha256="b" * 64,
        mpl_contract_sha256="m" * 64,
        training_contract_sha256=awac_training_contract_sha256(
            build_awac_training_contract(phase="critic_calibration")
        ),
        calibration_split=split,
        replay_identity={
            "run_identity": "exact-resume-test",
            "replay_sha256": "r" * 64,
            "replay_size": 0,
        },
        environment_step_count=0,
        calibration_metrics={"state": "PENDING"},
        mission_source_identity={"sha256": "s" * 64},
        mission_progress={
            "schema_id": "awac_formal_calibration_runtime_v1",
            "mission_source_identity": {"sha256": "s" * 64},
            "next_mission_index": 0,
            "completed_mission_ids": [],
            "environment_step_count": 0,
            "completed_episode_count": 0,
            "critic_update_count": 0,
            "gate_state": "PENDING",
            "gate_history": [],
            "holdout_completed_ids": [],
        },
        completed_episode_count=0,
        runtime_identity={"worker_count": 2},
        exact_resume_state=exact_resume_state,
    )

    assert validate_exact_calibration_resume_payload(payload)[
        "resume_restored_state_count"
    ] == 23
    for missing in (
        "raw_holdout_records",
        "producer_rng_state",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state_all",
    ):
        bad = copy.deepcopy(payload)
        del bad["exact_resume_state"][missing]
        with pytest.raises(ValueError, match="exact resume"):
            validate_exact_calibration_resume_payload(bad)


@pytest.mark.unit
def test_calibration_resume_dry_run_restores_a_valid_checkpoint_without_runtime(
    tmp_path: Path,
    capsys,
):
    torch, nn, _, _, _ = require_torch()
    bc = _bc(nn)
    bc_path = tmp_path / "bc.pt"
    torch.save(bc, bc_path)
    split_path = tmp_path / "calibration_split.json"
    split_path.write_text(
        json.dumps(
            {
                "contract_id": "awac_calibration_episode_split_v1",
                "seed": 4026,
                "train_episode_ids": ["e0"],
                "holdout_episode_ids": ["e1"],
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "checkpoint_last.pt"
    out_dir = tmp_path / "out"
    argv = [
        "--bc-checkpoint", str(bc_path),
        "--out-dir", str(out_dir),
        "--phase", "critic_calibration",
        "--device", "cpu",
        "--mpl-contract-sha256", "m" * 64,
        "--calibration-split", str(split_path),
        "--resume-checkpoint", str(checkpoint_path),
        "--dry-run",
    ]
    args = trainer.build_parser().parse_args(argv)
    config = trainer._optimization_config(args)
    learner = trainer.build_learner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_checkpoint=bc,
        config=config,
    )
    learner.freeze_actor_for_calibration()
    resolved = trainer._resolved_training_config(
        args,
        source_bc_sha256=file_sha256(bc_path),
        mpl_sha256="m" * 64,
    )
    split = trainer._load_calibration_split(str(split_path), fallback={})
    replay = AWACReplayBuffer.create(
        tmp_path / "replay",
        capacity=4,
        depth_shape=(1, 2, 2),
        vector_dim=3,
        action_dim=5,
        training_config_sha256=resolved["training_contract_sha256"],
        bc_checkpoint_sha256=file_sha256(bc_path),
        task_contract_id=TASK_CONTRACT_ID,
        task_contract_sha256=task_contract_sha256(),
        mpl_contract_sha256="m" * 64,
        run_identity="synthetic-resume",
        run_contract_sha256="r" * 64,
        mission_source_sha256="s" * 64,
        mission_index_sha256="s" * 64,
        behavior_source_phase="critic_calibration",
    )
    from planning.awac.checkpoint import (
        build_calibration_exact_resume_state,
        commit_calibration_checkpoint_transaction,
    )

    def build_payload(replay_identity):
        return build_calibration_checkpoint_payload(
            learner=learner,
            bc_checkpoint=bc,
            bc_checkpoint_sha256=file_sha256(bc_path),
            mpl_contract_sha256="m" * 64,
            training_contract_sha256=resolved["training_contract_sha256"],
            training_contract=resolved["training_contract"],
            calibration_split=split,
            replay_identity=replay_identity,
            environment_step_count=0,
            calibration_metrics={"state": "PENDING"},
            mission_source_identity={"sha256": "s" * 64},
            mission_progress={
                "schema_id": "awac_formal_calibration_runtime_v1",
                "mission_source_identity": {"sha256": "s" * 64},
                "next_mission_index": 0,
                "completed_mission_ids": [],
                "environment_step_count": 0,
                "completed_episode_count": 0,
                "critic_update_count": 0,
                "gate_state": "PENDING",
                "gate_history": [],
                "holdout_completed_ids": [],
            },
            completed_episode_count=0,
            runtime_identity={},
            exact_resume_state=build_calibration_exact_resume_state(
                raw_holdout_records=[],
                producer_rng=np.random.RandomState(4028),
                torch=torch,
            ),
        )

    committed = commit_calibration_checkpoint_transaction(
        output_dir=tmp_path,
        checkpoint_name="checkpoint_last",
        replay=replay,
        build_payload=build_payload,
        torch=torch,
        save_reason="dry_run_test",
    )
    checkpoint_path = Path(committed["checkpoint_alias"])

    assert trainer.main(argv) == 0
    assert '"algorithm_id": "discrete_masked_awac"' in capsys.readouterr().out


@pytest.mark.unit
def test_calibration_replay_accepts_bc_calibration_and_audit_has_no_privileged_fields(
    tmp_path: Path,
):
    replay = AWACReplayBuffer.create(
        tmp_path / "replay",
        capacity=4,
        depth_shape=(1, 2, 2),
        vector_dim=3,
        action_dim=5,
        training_config_sha256="t" * 64,
        observation_contract=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        observation_source=EXACT_ENDPOINT_OBSERVATION_CONTRACT,
        bc_checkpoint_sha256="b" * 64,
        task_contract_id=TASK_CONTRACT_ID,
        task_contract_sha256="k" * 64,
        mpl_contract_sha256="m" * 64,
        run_identity="phase0",
        run_contract_sha256="r" * 64,
        behavior_source_phase="critic_calibration",
    )
    mask = np.ones((5,), dtype=np.bool_)
    replay.add(
        depth=np.zeros((1, 2, 2), dtype=np.float32),
        vector=np.zeros(3, dtype=np.float32),
        action_mask=mask,
        action=0,
        reward=1.0,
        next_depth=np.zeros((1, 2, 2), dtype=np.float32),
        next_vector=np.ones(3, dtype=np.float32),
        next_action_mask=mask,
        done=True,
        behavior_source=int(BehaviorSource.BC_CALIBRATION),
    )
    replay.flush()
    from planning.awac.replay_audit import audit_awac_replay

    report = audit_awac_replay(tmp_path / "replay")
    assert report["status"] == "PASS"
    assert report["behavior_source_counts"] == {"0": 1}
    assert report["privileged_field_count"] == 0


@pytest.mark.unit
def test_holdout_accepts_default_cuda_identity_at_current_cuda_index(monkeypatch):
    torch, _, _, _, _ = require_torch()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    learner = _device_only_holdout_learner(
        torch,
        learner_device="cuda",
        critic_device="cuda:0",
    )

    assert trainer._validate_calibration_holdout_devices(
        learner=learner,
        torch=torch,
    ) == torch.device("cuda:0")


@pytest.mark.unit
def test_resolve_torch_device_uses_current_cuda_index_without_rewriting_explicit_index(
    monkeypatch,
):
    torch, _, _, _, _ = require_torch()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)

    assert resolve_torch_device(torch, "cpu") == torch.device("cpu")
    assert resolve_torch_device(torch, "cuda") == torch.device("cuda:1")
    assert resolve_torch_device(torch, "cuda:0") == torch.device("cuda:0")
    assert resolve_torch_device(torch, "cuda:1") == torch.device("cuda:1")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("learner_device", "critic_device", "should_pass"),
    (
        ("cuda:0", "cuda:0", True),
        ("cuda", "cuda:1", False),
        ("cuda:1", "cuda:0", False),
        ("cpu", "cpu", True),
        ("cpu", "cuda:0", False),
    ),
    ids=(
        "explicit_cuda_zero_matches",
        "default_cuda_rejects_nondefault_cuda_one",
        "explicit_cuda_one_rejects_cuda_zero",
        "cpu_matches_cpu",
        "cpu_rejects_cuda_zero",
    ),
)
def test_holdout_device_validation_accepts_only_the_same_concrete_compute_device(
    monkeypatch,
    learner_device,
    critic_device,
    should_pass,
):
    torch, _, _, _, _ = require_torch()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    learner = _device_only_holdout_learner(
        torch,
        learner_device=learner_device,
        critic_device=critic_device,
    )

    if should_pass:
        assert trainer._validate_calibration_holdout_devices(
            learner=learner,
            torch=torch,
        ) == torch.device(critic_device)
    else:
        with pytest.raises(RuntimeError, match="critic1.*device"):
            trainer._validate_calibration_holdout_devices(learner=learner, torch=torch)


@pytest.mark.unit
def test_holdout_moves_every_computation_tensor_to_the_learner_device(monkeypatch):
    torch, nn, _, _, _ = require_torch()
    learner = _calibration_learner(torch, nn, device=torch.device("cpu"))
    records = _holdout_records(torch)
    captured = {}
    moved = {}
    original_batch = trainer._calibration_tensor_batch
    original_to = torch.Tensor.to

    def capture_batch(*args, **kwargs):
        batch = original_batch(*args, **kwargs)
        captured.update({id(value): name for name, value in batch.items()})
        return batch

    def capture_to(value, *args, **kwargs):
        name = captured.get(id(value))
        if name is not None:
            moved[name] = kwargs.get("device", args[0] if args else None)
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(trainer, "_calibration_tensor_batch", capture_batch)
    monkeypatch.setattr(torch.Tensor, "to", capture_to)

    trainer._evaluate_calibration_holdout(
        learner=learner,
        torch=torch,
        records=records,
        gamma=0.99,
        reward_scale=0.10,
    )

    assert set(moved) == {
        "depth",
        "vector",
        "action_mask",
        "action",
        "reward",
        "next_depth",
        "next_vector",
        "next_action_mask",
        "done",
        "behavior_source",
    }
    critic_device = next(learner.critic1.parameters()).device
    assert all(torch.device(device) == critic_device for device in moved.values())


@pytest.mark.unit
def test_holdout_rejects_mixed_twin_critic_devices_before_forward(monkeypatch):
    torch, nn, _, _, _ = require_torch()
    learner = _calibration_learner(torch, nn, device=torch.device("cpu"))
    records = _holdout_records(torch)

    class _MetaParameter:
        device = torch.device("meta")

    def unexpected_forward(*_args, **_kwargs):
        raise AssertionError("mixed-device target critic must not run")

    monkeypatch.setattr(
        learner.target_critic2,
        "parameters",
        lambda: iter((_MetaParameter(),)),
    )
    monkeypatch.setattr(learner.target_critic2, "forward", unexpected_forward)

    with pytest.raises(RuntimeError, match="target_critic2.*device"):
        trainer._evaluate_calibration_holdout(
            learner=learner,
            torch=torch,
            records=records,
            gamma=0.99,
            reward_scale=0.10,
        )


@pytest.mark.unit
def test_cpu_holdout_is_no_grad_preserves_modes_and_accepts_terminal_zero_mask():
    torch, nn, _, _, _ = require_torch()
    learner = _calibration_learner(torch, nn, device=torch.device("cpu"))
    records = _holdout_records(torch)
    modules = {
        "critic1": learner.critic1,
        "critic2": learner.critic2,
        "target_critic1": learner.target_critic1,
        "target_critic2": learner.target_critic2,
        "bc_reference": learner.bc_reference,
    }
    learner.critic1.train(True)
    learner.critic2.train(False)
    learner.target_critic1.train(True)
    learner.target_critic2.train(False)
    learner.bc_reference.train(True)
    modes_before = {name: module.training for name, module in modules.items()}
    states_before = {
        name: copy.deepcopy(module.state_dict()) for name, module in modules.items()
    }
    update_count_before = learner.critic_update_count
    assert next(learner.actor.parameters()).device == learner.device

    result = trainer._evaluate_calibration_holdout(
        learner=learner,
        torch=torch,
        records=records,
        gamma=0.99,
        reward_scale=0.10,
    )

    assert result["terminal_reasons"] == ["", "dead_end"]
    assert result["returns"] == pytest.approx([-0.001, 0.1])
    assert all(np.isfinite(result[name]).all() for name in ("q1", "q2"))
    assert all(np.isfinite(result[name]) for name in (
        "critic1_td_loss", "critic2_td_loss", "holdout_td_loss"
    ))
    assert learner.critic_update_count == update_count_before == 0
    assert {name: module.training for name, module in modules.items()} == modes_before
    for name, module in modules.items():
        assert _state_equal(torch, states_before[name], module.state_dict())
        assert all(parameter.grad is None for parameter in module.parameters())


@pytest.mark.unit
def test_formal_holdout_callback_returns_canonical_gate_window():
    torch, nn, _, _, _ = require_torch()
    learner = _calibration_learner(torch, nn, device=torch.device("cpu"))
    records = _holdout_records(torch)

    window = trainer._summarize_calibration_holdout_window(
        learner=learner,
        torch=torch,
        records=records,
        gamma=0.99,
        reward_scale=0.10,
    )

    assert window["finite"] is True
    assert window["holdout_return_semantics"] == "episode_monte_carlo_return_v1"
    assert window["holdout_measurement_status"] == "PASS"
    assert all(
        field in window
        for field in (
            "q_mean",
            "q_std",
            "q_min",
            "q_max",
            "q_p99",
            "normalized_twin_disagreement",
            "q_explosion_bound",
            "value_ordering_status",
        )
    )


@pytest.mark.cuda
def test_cuda_holdout_moves_cpu_records_to_the_cuda_learner_device():
    torch, nn, _, _, _ = require_torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime unavailable")
    learner = _calibration_learner(torch, nn, device=torch.device("cuda"))
    records = _holdout_records(torch)

    result = trainer._evaluate_calibration_holdout(
        learner=learner,
        torch=torch,
        records=records,
        gamma=0.99,
        reward_scale=0.10,
    )

    assert all(
        next(module.parameters()).device == learner.device
        for module in (
            learner.critic1,
            learner.critic2,
            learner.target_critic1,
            learner.target_critic2,
            learner.bc_reference,
        )
    )
    assert all(np.isfinite(result[name]).all() for name in ("q1", "q2"))
