"""Crash-safe calibration checkpoint and exact-resume contract tests."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import planning.awac.trainer as trainer
from planning.awac.calibration import (
    build_calibration_checkpoint_payload,
    build_calibration_pass_checkpoint_payload,
)
from planning.awac.checkpoint import (
    build_calibration_exact_resume_state,
    calibration_checkpoint_transaction_diagnostics_path,
    commit_calibration_checkpoint_transaction,
    load_committed_calibration_resume_checkpoint,
    recover_calibration_replay_from_checkpoint_transaction,
    restore_calibration_exact_resume_state,
    save_calibration_checkpoint,
)
from planning.awac.contract import (
    awac_training_contract_sha256,
    build_awac_training_contract,
)
from planning.awac.interaction import BehaviorSource
from planning.awac.learner import AWACOptimizationConfig, DiscreteAWACLearner
from planning.awac.replay import AWACReplayBuffer
from planning.bc.model import build_model, require_torch
from planning.contracts.feature import NUM_ACTIONS, POLICY_VECTOR_DIM


pytestmark = pytest.mark.unit


class _Module:
    def __init__(self, torch, value):
        self._state = {"weight": torch.tensor([float(value)])}

    def state_dict(self):
        return dict(self._state)


class _Learner:
    def __init__(self, torch):
        self.torch = torch
        self.config = SimpleNamespace(gamma=0.99, tau=0.005)
        self.actor = _Module(torch, 1.0)
        self.critic1 = _Module(torch, 2.0)
        self.critic2 = _Module(torch, 3.0)
        self.target_critic1 = _Module(torch, 4.0)
        self.target_critic2 = _Module(torch, 5.0)
        self.update_step = 0
        self.actor_update_count = 0
        self.actor_optimizer_step_count = 0
        self.critic_update_count = 0
        self.actor_frozen_for_calibration = True

    def state_dict(self):
        optimizer = {"state": {}, "param_groups": [{"lr": 1.0e-5}]}
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "target_critic1_state_dict": self.target_critic1.state_dict(),
            "target_critic2_state_dict": self.target_critic2.state_dict(),
            "actor_optimizer_state_dict": optimizer,
            "critic_optimizer_state_dict": optimizer,
            "update_step": int(self.update_step),
            "actor_update_count": int(self.actor_update_count),
            "actor_awac_update_count": 0,
            "actor_recovery_update_count": 0,
            "actor_trust_region_rejection_count": 0,
            "actor_optimizer_step_count": int(self.actor_optimizer_step_count),
            "critic_update_count": int(self.critic_update_count),
            "actor_frozen_for_calibration": True,
        }


def _replay(path):
    return AWACReplayBuffer.create(
        path,
        capacity=8,
        depth_shape=(1, 2, 2),
        vector_dim=3,
        action_dim=5,
        training_config_sha256="a" * 64,
        bc_checkpoint_sha256="b" * 64,
        task_contract_id="task",
        task_contract_sha256="c" * 64,
        mpl_contract_sha256="d" * 64,
        run_identity="exact-resume-test",
        run_contract_sha256="e" * 64,
        mission_source_sha256="f" * 64,
        mission_index_sha256="f" * 64,
        behavior_source_phase="critic_calibration",
    )


def _transition(index):
    mask = np.ones(5, dtype=np.bool_)
    return {
        "depth": np.full((1, 2, 2), index / 10.0, dtype=np.float32),
        "vector": np.asarray([index, index + 1, index + 2], dtype=np.float32),
        "action_mask": mask,
        "action": 1,
        "reward": float(index),
        "next_depth": np.full((1, 2, 2), (index + 1) / 10.0, dtype=np.float32),
        "next_vector": np.asarray(
            [index + 1, index + 2, index + 3], dtype=np.float32
        ),
        "next_action_mask": mask,
        "done": True,
        "behavior_source": int(BehaviorSource.BC_CALIBRATION),
    }


def _payload_builder(torch, learner):
    bc_checkpoint = {"model_state_dict": learner.actor.state_dict()}
    split = {
        "contract_id": "awac_calibration_episode_split_v1",
        "seed": 4026,
        "train_episode_ids": ["e0"],
        "holdout_episode_ids": ["e1"],
    }
    training_contract = build_awac_training_contract(phase="critic_calibration")

    def build(replay_identity):
        return build_calibration_checkpoint_payload(
            learner=learner,
            bc_checkpoint=bc_checkpoint,
            bc_checkpoint_sha256="b" * 64,
            mpl_contract_sha256="d" * 64,
            training_contract=training_contract,
            training_contract_sha256=awac_training_contract_sha256(
                training_contract
            ),
            calibration_split=split,
            replay_identity=replay_identity,
            environment_step_count=0,
            calibration_metrics={"state": "PENDING"},
            mission_source_identity={"sha256": "f" * 64},
            mission_progress={
                "schema_id": "awac_formal_calibration_runtime_v1",
                "mission_source_identity": {"sha256": "f" * 64},
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
            exact_resume_state=build_calibration_exact_resume_state(
                raw_holdout_records=[],
                producer_rng=np.random.RandomState(4027),
                torch=torch,
            ),
        )

    return build


class _InjectedCrash(RuntimeError):
    pass


def _crash_at(expected):
    def failpoint(phase):
        if phase == expected:
            raise _InjectedCrash(phase)

    return failpoint


def test_incomplete_generation_recovers_the_previous_committed_checkpoint(
    tmp_path,
):
    """A post-rename pre-marker crash cannot displace the last good snapshot."""

    torch, _, _, _, _ = require_torch()
    learner = _Learner(torch)
    replay = _replay(tmp_path / "replay")
    build_payload = _payload_builder(torch, learner)
    replay.add(**_transition(1))
    first = commit_calibration_checkpoint_transaction(
        output_dir=tmp_path,
        checkpoint_name="checkpoint_last",
        replay=replay,
        build_payload=build_payload,
        torch=torch,
        save_reason="gate_window",
    )
    assert first["generation"] == 1

    replay.add(**_transition(2))
    with pytest.raises(_InjectedCrash, match="after_checkpoint_rename"):
        commit_calibration_checkpoint_transaction(
            output_dir=tmp_path,
            checkpoint_name="checkpoint_last",
            replay=replay,
            build_payload=build_payload,
            torch=torch,
            save_reason="gate_window",
            failpoint=_crash_at("after_checkpoint_rename"),
        )

    recovered = recover_calibration_replay_from_checkpoint_transaction(
        replay,
        tmp_path / "checkpoint_last.pt",
    )
    assert recovered["generation"] == 1
    assert replay.size == 1
    payload = load_committed_calibration_resume_checkpoint(
        tmp_path / "checkpoint_last.pt",
        torch=torch,
    )
    assert payload["checkpoint_transaction"]["generation"] == 1


@pytest.mark.parametrize(
    ("phase", "expected_generation", "expected_replay_size"),
    (
        ("before_replay_flush", 1, 1),
        ("after_replay_flush", 1, 1),
        ("checkpoint_temp_write", 1, 1),
        ("checkpoint_temp_complete", 1, 1),
        ("after_checkpoint_rename", 1, 1),
        ("after_transaction_manifest_commit", 2, 2),
    ),
)
def test_crash_injections_only_publish_a_complete_generation(
    tmp_path,
    phase,
    expected_generation,
    expected_replay_size,
):
    """The marker is authoritative across every save-interruption boundary."""

    torch, _, _, _, _ = require_torch()
    learner = _Learner(torch)
    replay = _replay(tmp_path / "replay")
    build_payload = _payload_builder(torch, learner)
    replay.add(**_transition(1))
    commit_calibration_checkpoint_transaction(
        output_dir=tmp_path,
        checkpoint_name="checkpoint_last",
        replay=replay,
        build_payload=build_payload,
        torch=torch,
        save_reason="initial_boundary",
    )

    replay.add(**_transition(2))
    with pytest.raises(_InjectedCrash, match=phase):
        commit_calibration_checkpoint_transaction(
            output_dir=tmp_path,
            checkpoint_name="checkpoint_last",
            replay=replay,
            build_payload=build_payload,
            torch=torch,
            save_reason="crash_injection:{}".format(phase),
            failpoint=_crash_at(phase),
        )

    recovered = recover_calibration_replay_from_checkpoint_transaction(
        replay,
        tmp_path / "checkpoint_last.pt",
    )
    payload = load_committed_calibration_resume_checkpoint(
        tmp_path / "checkpoint_last.pt", torch=torch
    )
    assert recovered["generation"] == expected_generation
    assert payload["checkpoint_transaction"]["generation"] == expected_generation
    assert replay.size == expected_replay_size

    diagnostics_path = calibration_checkpoint_transaction_diagnostics_path(
        tmp_path, "checkpoint_last"
    )
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["checkpoint_save_reason"] == "crash_injection:{}".format(
        phase
    )
    assert "_InjectedCrash" in diagnostics["interrupted_save_traceback"]
    assert diagnostics["runtime_failure_summary"].startswith("_InjectedCrash:")
    if phase == "after_transaction_manifest_commit":
        assert diagnostics["checkpoint_final_commit"] == "PASS"
        assert diagnostics["transaction_state"] == "COMMITTED"
    else:
        assert diagnostics["checkpoint_final_commit"] == "FAIL"
        assert diagnostics["transaction_state"] == "FAILED_INCOMPLETE"


def test_replay_metadata_mismatch_is_not_recovered_as_an_uncommitted_tail(tmp_path):
    """Only the sidecar-described append-only tail may be rolled back."""

    torch, _, _, _, _ = require_torch()
    learner = _Learner(torch)
    replay = _replay(tmp_path / "replay")
    replay.add(**_transition(1))
    commit_calibration_checkpoint_transaction(
        output_dir=tmp_path,
        checkpoint_name="checkpoint_last",
        replay=replay,
        build_payload=_payload_builder(torch, learner),
        torch=torch,
        save_reason="initial_boundary",
    )
    replay.flush()
    metadata_path = tmp_path / "replay" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["size"] = 2
    metadata["position"] = 2
    metadata["total_added"] = 2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    reopened = AWACReplayBuffer.open(tmp_path / "replay")
    with pytest.raises(ValueError, match="does not match committed checkpoint"):
        recover_calibration_replay_from_checkpoint_transaction(
            reopened,
            tmp_path / "checkpoint_last.pt",
        )


def test_legacy_direct_calibration_checkpoint_is_not_resumable(tmp_path):
    """V1-V5 style direct checkpoint files remain fail-closed for resume."""

    torch, _, _, _, _ = require_torch()
    checkpoint = tmp_path / "checkpoint_last.pt"
    save_calibration_checkpoint(
        checkpoint,
        _payload_builder(torch, _Learner(torch))(
            {
                "run_identity": "exact-resume-test",
                "mission_source_sha256": "f" * 64,
                "mission_index_sha256": "f" * 64,
                "replay_size": 0,
                "replay_position": 0,
                "replay_total_added": 0,
                "replay_metadata_sha256": "0" * 64,
                "replay_contract_sha256": "r" * 64,
                "behavior_source_phase": "critic_calibration",
            }
        ),
        torch=torch,
    )
    with pytest.raises(FileNotFoundError, match="transaction manifest is missing"):
        load_committed_calibration_resume_checkpoint(checkpoint, torch=torch)


class _FakeCudaRng:
    def __init__(self):
        self.states = [np.asarray([31, 32], dtype=np.uint8)]

    def is_available(self):
        return True

    def get_rng_state_all(self):
        return [value.copy() for value in self.states]

    def set_rng_state_all(self, values):
        self.states = [np.asarray(value, dtype=np.uint8).copy() for value in values]


class _FakeTorchRng:
    def __init__(self):
        self.cpu = np.asarray([11, 12], dtype=np.uint8)
        self.cuda = _FakeCudaRng()

    def get_rng_state(self):
        return self.cpu.copy()

    def set_rng_state(self, value):
        self.cpu = np.asarray(value, dtype=np.uint8).copy()


def test_cuda_rng_payload_roundtrips_without_a_cuda_host():
    """CUDA RNG payload plumbing is tested with a serialized fake device."""

    fake_torch = _FakeTorchRng()
    producer_rng = np.random.RandomState(4027)
    snapshot = build_calibration_exact_resume_state(
        raw_holdout_records=[],
        producer_rng=producer_rng,
        torch=fake_torch,
    )
    assert snapshot["torch_cuda_rng_available"] is True
    assert len(snapshot["torch_cuda_rng_state_all"]) == 1

    fake_torch.cpu[:] = 0
    fake_torch.cuda.states = [np.asarray([0, 0], dtype=np.uint8)]
    producer_rng.seed(1)
    restore_calibration_exact_resume_state(
        snapshot,
        producer_rng=producer_rng,
        torch=fake_torch,
    )
    np.testing.assert_array_equal(fake_torch.cpu, np.asarray([11, 12], dtype=np.uint8))
    np.testing.assert_array_equal(
        fake_torch.cuda.states[0], np.asarray([31, 32], dtype=np.uint8)
    )


def test_final_flush_close_confirmation_is_persisted_with_commit(tmp_path):
    """Shutdown lifecycle evidence is durable but not a resume marker."""

    torch, _, _, _, _ = require_torch()
    replay = _replay(tmp_path / "replay")
    replay.add(**_transition(1))
    committed = commit_calibration_checkpoint_transaction(
        output_dir=tmp_path,
        checkpoint_name="checkpoint_last",
        replay=replay,
        build_payload=_payload_builder(torch, _Learner(torch)),
        torch=torch,
        save_reason="run_complete",
        recovery_diagnostics={
            "final_flush_close_confirmation": {
                "replay_final_flush": "PASS",
                "runtime_close": "PASS",
            }
        },
    )
    diagnostics = json.loads(
        Path(committed["diagnostics_path"]).read_text(encoding="utf-8")
    )
    payload = load_committed_calibration_resume_checkpoint(
        committed["checkpoint_alias"], torch=torch
    )
    assert diagnostics["checkpoint_save_reason"] == "run_complete"
    assert diagnostics["checkpoint_final_commit"] == "PASS"
    assert diagnostics["checkpoint_retention_status"] == "PASS"
    assert diagnostics["final_flush_close_confirmation"] == {
        "replay_final_flush": "PASS",
        "runtime_close": "PASS",
    }
    assert payload["checkpoint_recovery_diagnostics"][
        "final_flush_close_confirmation"
    ] == diagnostics["final_flush_close_confirmation"]


def test_retention_failure_does_not_invalidate_committed_transaction(
    tmp_path, monkeypatch
):
    """A post-commit GC failure is a storage warning, not a checkpoint failure."""

    import planning.awac.checkpoint as checkpoint_module

    torch, _, _, _, _ = require_torch()
    replay = _replay(tmp_path / "replay")
    replay.add(**_transition(1))

    def fail_gc(*args, **kwargs):
        raise OSError("synthetic retention failure")

    monkeypatch.setattr(
        checkpoint_module,
        "garbage_collect_checkpoint_generations",
        fail_gc,
    )
    committed = commit_calibration_checkpoint_transaction(
        output_dir=tmp_path,
        checkpoint_name="checkpoint_last",
        replay=replay,
        build_payload=_payload_builder(torch, _Learner(torch)),
        torch=torch,
        save_reason="retention_warning",
    )

    diagnostics = json.loads(
        Path(committed["diagnostics_path"]).read_text(encoding="utf-8")
    )
    assert diagnostics["transaction_state"] == "COMMITTED"
    assert diagnostics["checkpoint_final_commit"] == "PASS"
    assert diagnostics["checkpoint_retention_status"] == "WARNING"
    load_committed_calibration_resume_checkpoint(
        committed["checkpoint_alias"], torch=torch
    )


def test_runtime_failure_summary_persists_without_starting_runtime(tmp_path):
    """An exception report survives even if a snapshot cannot be attempted."""

    producer = SimpleNamespace(
        metrics={"replay_final_flush": "PASS", "runtime_close": "PASS"}
    )
    try:
        raise KeyboardInterrupt("synthetic stop")
    except KeyboardInterrupt as error:
        diagnostics = trainer._runtime_recovery_diagnostics(
            producer,
            error=error,
            error_traceback="synthetic-traceback",
        )
    path = trainer._write_runtime_failure_recovery_summary(
        output_dir=tmp_path,
        checkpoint_name="checkpoint_last",
        diagnostics={
            **diagnostics,
            "checkpoint_save_reason": "keyboard_interrupt",
            "checkpoint_final_commit": "NOT_ATTEMPTED",
        },
    )
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["runtime_failure_summary"] == "KeyboardInterrupt: synthetic stop"
    assert persisted["interrupted_save_traceback"] == "synthetic-traceback"
    assert persisted["final_flush_close_confirmation"] == {
        "replay_final_flush": "PASS",
        "runtime_close": "PASS",
    }


def test_calibration_pass_checkpoint_uses_the_same_commit_marker_protocol(tmp_path):
    """A gate PASS artifact is never published by a direct torch.save path."""

    torch, _, _, _, _ = require_torch()
    learner = _Learner(torch)
    learner.critic_update_count = 1
    replay = _replay(tmp_path / "replay")
    replay.add(**_transition(1))
    training_contract = build_awac_training_contract(phase="critic_calibration")
    split = {
        "contract_id": "awac_calibration_episode_split_v1",
        "seed": 4026,
        "train_episode_ids": ["e0"],
        "holdout_episode_ids": ["e1"],
    }

    def build(replay_identity):
        return build_calibration_pass_checkpoint_payload(
            learner=learner,
            bc_checkpoint={"model_state_dict": learner.actor.state_dict()},
            bc_checkpoint_sha256="b" * 64,
            mpl_contract_sha256="d" * 64,
            training_contract=training_contract,
            training_contract_sha256=awac_training_contract_sha256(
                training_contract
            ),
            calibration_split=split,
            replay_identity=replay_identity,
            environment_step_count=0,
            calibration_metrics={"state": "PASS"},
            actor_depth_lr=1.0e-6,
            critic_depth_lr=1.0e-5,
            mission_source_identity={"sha256": "f" * 64},
            mission_progress={
                "schema_id": "awac_formal_calibration_runtime_v1",
                "mission_source_identity": {"sha256": "f" * 64},
                "next_mission_index": 0,
                "completed_mission_ids": [],
                "environment_step_count": 0,
                "completed_episode_count": 0,
                "critic_update_count": 1,
                "gate_state": "PENDING",
                "gate_history": [],
                "holdout_completed_ids": [],
            },
            completed_episode_count=0,
            runtime_identity={"worker_count": 2},
            exact_resume_state=build_calibration_exact_resume_state(
                raw_holdout_records=[],
                producer_rng=np.random.RandomState(4027),
                torch=torch,
            ),
        )

    committed = commit_calibration_checkpoint_transaction(
        output_dir=tmp_path,
        checkpoint_name="checkpoint_calibration_pass",
        checkpoint_kind="checkpoint_calibration_pass",
        replay=replay,
        build_payload=build,
        torch=torch,
        save_reason="gate_pass",
    )
    payload = load_committed_calibration_resume_checkpoint(
        committed["checkpoint_alias"], torch=torch
    )
    assert payload["calibration_pass_checkpoint"] is True
    assert payload["checkpoint_transaction"]["checkpoint_kind"] == (
        "checkpoint_calibration_pass"
    )


def _real_transition(index):
    mask = np.ones(NUM_ACTIONS, dtype=np.bool_)
    return {
        "depth": np.full((1, 32, 32), index / 10.0, dtype=np.float32),
        "vector": np.linspace(
            float(index), float(index) + 1.0, POLICY_VECTOR_DIM, dtype=np.float32
        ),
        "action_mask": mask,
        "action": int(index % NUM_ACTIONS),
        "reward": float(index) / 10.0,
        "next_depth": np.full(
            (1, 32, 32), (index + 1) / 10.0, dtype=np.float32
        ),
        "next_vector": np.linspace(
            float(index + 1),
            float(index + 2),
            POLICY_VECTOR_DIM,
            dtype=np.float32,
        ),
        "next_action_mask": mask,
        "done": bool(index % 2),
        "behavior_source": int(BehaviorSource.BC_CALIBRATION),
    }


def _real_replay(path):
    return AWACReplayBuffer.create(
        path,
        capacity=16,
        depth_shape=(1, 32, 32),
        vector_dim=POLICY_VECTOR_DIM,
        action_dim=NUM_ACTIONS,
        training_config_sha256="a" * 64,
        bc_checkpoint_sha256="b" * 64,
        task_contract_id="task",
        task_contract_sha256="c" * 64,
        mpl_contract_sha256="d" * 64,
        run_identity="cpu-parity-test",
        run_contract_sha256="e" * 64,
        mission_source_sha256="f" * 64,
        mission_index_sha256="f" * 64,
        behavior_source_phase="critic_calibration",
    )


def _real_learner(torch, nn, bc_state):
    learner = DiscreteAWACLearner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_state_dict=bc_state,
        depth_channels=1,
        config=AWACOptimizationConfig(critic_depth_lr=1.0e-5),
    )
    learner.freeze_actor_for_calibration()
    return learner


def _run_critic_updates(learner, replay, rng, *, count):
    sample_indices = []
    for _ in range(int(count)):
        indices = rng.randint(0, replay.size, size=2)
        sample_indices.append(tuple(int(value) for value in indices))
        batch = replay.sample_indices(indices, torch=learner.torch, device=learner.device)
        learner.calibration_update(batch)
    return sample_indices


def _nested_state_equal(torch, left, right):
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _nested_state_equal(torch, left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_state_equal(torch, a, b) for a, b in zip(left, right)
        )
    if hasattr(left, "dtype") and hasattr(left, "shape"):
        return bool(torch.equal(left, right))
    return left == right


def test_cpu_uninterrupted_and_committed_resume_are_bit_exact(tmp_path):
    """Four CPU critic updates equal two + committed checkpoint + two."""

    torch, nn, _, _, _ = require_torch()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        random.seed(8101)
        np.random.seed(8102)
        torch.manual_seed(8103)
        bc_state = copy.deepcopy(build_model(nn, depth_channels=1).state_dict())
        learner_init_rng = torch.get_rng_state().clone()

        replay_a = _real_replay(tmp_path / "replay-a")
        replay_b = _real_replay(tmp_path / "replay-b")
        for index in range(8):
            replay_a.add(**_real_transition(index))
            replay_b.add(**_real_transition(index))

        random.seed(8101)
        np.random.seed(8102)
        torch.set_rng_state(learner_init_rng.clone())
        uninterrupted = _real_learner(torch, nn, bc_state)
        uninterrupted_rng = np.random.RandomState(8104)
        uninterrupted_indices = _run_critic_updates(
            uninterrupted, replay_a, uninterrupted_rng, count=4
        )
        uninterrupted_state = copy.deepcopy(uninterrupted.state_dict())

        random.seed(8101)
        np.random.seed(8102)
        torch.set_rng_state(learner_init_rng.clone())
        interrupted = _real_learner(torch, nn, bc_state)
        interrupted_rng = np.random.RandomState(8104)
        first_indices = _run_critic_updates(
            interrupted, replay_b, interrupted_rng, count=2
        )

        training_contract = build_awac_training_contract(phase="critic_calibration")

        def build_payload(replay_identity):
            return build_calibration_checkpoint_payload(
                learner=interrupted,
                bc_checkpoint={"model_state_dict": bc_state},
                bc_checkpoint_sha256="b" * 64,
                mpl_contract_sha256="d" * 64,
                training_contract=training_contract,
                training_contract_sha256=awac_training_contract_sha256(
                    training_contract
                ),
                calibration_split={
                    "contract_id": "awac_calibration_episode_split_v1",
                    "seed": 4026,
                    "train_episode_ids": ["e0"],
                    "holdout_episode_ids": ["e1"],
                },
                replay_identity=replay_identity,
                environment_step_count=0,
                calibration_metrics={"state": "PENDING"},
                mission_source_identity={"sha256": "f" * 64},
                mission_progress={
                    "schema_id": "awac_formal_calibration_runtime_v1",
                    "mission_source_identity": {"sha256": "f" * 64},
                    "next_mission_index": 0,
                    "completed_mission_ids": [],
                    "environment_step_count": 0,
                    "completed_episode_count": 0,
                    "critic_update_count": int(interrupted.critic_update_count),
                    "gate_state": "PENDING",
                    "gate_history": [],
                    "holdout_completed_ids": [],
                },
                completed_episode_count=0,
                runtime_identity={"worker_count": 2},
                exact_resume_state=build_calibration_exact_resume_state(
                    raw_holdout_records=[],
                    producer_rng=interrupted_rng,
                    torch=torch,
                ),
            )

        committed = commit_calibration_checkpoint_transaction(
            output_dir=tmp_path / "checkpoint",
            checkpoint_name="checkpoint_last",
            replay=replay_b,
            build_payload=build_payload,
            torch=torch,
            save_reason="cpu_parity_interrupt",
        )
        resumed_payload = load_committed_calibration_resume_checkpoint(
            committed["checkpoint_alias"], torch=torch
        )
        recover_calibration_replay_from_checkpoint_transaction(
            replay_b, committed["checkpoint_alias"]
        )

        # Construction consumes RNG; exact restore must happen after it.
        resumed = _real_learner(torch, nn, bc_state)
        resumed.load_state_dict(resumed_payload)
        resumed_rng = np.random.RandomState(1)
        restore_calibration_exact_resume_state(
            resumed_payload["exact_resume_state"],
            producer_rng=resumed_rng,
            torch=torch,
        )
        resumed_indices = _run_critic_updates(
            resumed, replay_b, resumed_rng, count=2
        )

        assert first_indices == uninterrupted_indices[:2]
        assert resumed_indices == uninterrupted_indices[2:]
        assert _nested_state_equal(
            torch, uninterrupted_state, resumed.state_dict()
        )
        assert uninterrupted.critic_update_count == resumed.critic_update_count == 4
        assert uninterrupted.actor_update_count == resumed.actor_update_count == 0
    finally:
        torch.set_num_threads(original_threads)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
