"""P3-D frozen-checkpoint evidence contract at the artifact seam."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.unit


def test_formal_checkpoint_uses_the_requested_global_step_filename():
    from planning.common.checkpoint_evidence import frozen_checkpoint_filename

    assert frozen_checkpoint_filename(11000) == "checkpoint_11000.pt"
    assert frozen_checkpoint_filename(15000) == "checkpoint_15000.pt"


def _checkpoint_payload(torch, *, step: int):
    return {
        "global_step": step,
        "replay_size": step,
        "update_step": step - 8000,
        "actor_update_count": (step - 9000) // 2,
        "actor_state_dict": {"weight": torch.tensor([1.0])},
        "critic1_state_dict": {"weight": torch.tensor([2.0])},
        "critic2_state_dict": {"weight": torch.tensor([3.0])},
        "target_critic1_state_dict": {"weight": torch.tensor([4.0])},
        "target_critic2_state_dict": {"weight": torch.tensor([5.0])},
        "actor_optimizer_state_dict": {"state": {}, "param_groups": []},
        "critic_optimizer_state_dict": {"state": {}, "param_groups": []},
        "bc_reference_fingerprint": "b" * 64,
    }


def test_frozen_checkpoint_missing_required_metadata_is_rejected():
    """A loadable tensor payload is not valid P3-D evidence by itself."""

    torch = pytest.importorskip("torch")
    from planning.common.checkpoint_evidence import (
        CheckpointEvidenceError,
        validate_frozen_checkpoint_payload,
    )

    payload = {
        "global_step": 11000,
        "replay_size": 11000,
        "update_step": 3000,
        "actor_update_count": 500,
        "actor_state_dict": {"weight": torch.ones(1)},
        "critic1_state_dict": {"weight": torch.ones(1)},
        "critic2_state_dict": {"weight": torch.ones(1)},
        "target_critic1_state_dict": {"weight": torch.ones(1)},
        "target_critic2_state_dict": {"weight": torch.ones(1)},
        "actor_optimizer_state_dict": {"state": {}, "param_groups": []},
        "critic_optimizer_state_dict": {"state": {}, "param_groups": []},
    }

    with pytest.raises(CheckpointEvidenceError, match="bc_reference_fingerprint"):
        validate_frozen_checkpoint_payload(payload, expected_global_step=11000)


def test_frozen_checkpoint_load_rejects_a_saved_actor_fingerprint_mismatch(tmp_path):
    """A post-save tensor mutation cannot silently pass frozen evaluation."""

    torch = pytest.importorskip("torch")
    from planning.common.checkpoint_evidence import (
        CheckpointEvidenceError,
        load_verified_frozen_checkpoint,
        save_frozen_checkpoint,
    )

    path = tmp_path / "checkpoint_11000.pt"
    save_frozen_checkpoint(
        path,
        torch=torch,
        checkpoint=_checkpoint_payload(torch, step=11000),
    )
    mutated = torch.load(path, map_location="cpu", weights_only=False)
    mutated["actor_state_dict"]["weight"][0] = 9.0
    torch.save(mutated, path)

    with pytest.raises(CheckpointEvidenceError, match="actor_fingerprint mismatch"):
        load_verified_frozen_checkpoint(
            path, torch=torch, expected_global_step=11000
        )


@pytest.mark.parametrize("step", (11000, 12000, 13000, 14000, 15000))
def test_each_formal_checkpoint_round_trips_state_optimizer_and_runtime_provenance(
    tmp_path, step
):
    """Every scheduled artifact is independently resumable and attributable."""

    torch = pytest.importorskip("torch")
    from planning.common.checkpoint_evidence import (
        attach_runtime_provenance,
        load_verified_frozen_checkpoint,
        save_frozen_checkpoint,
    )

    original = _checkpoint_payload(torch, step=step)
    path = tmp_path / "checkpoint_{}.pt".format(step)
    save_frozen_checkpoint(path, torch=torch, checkpoint=original)
    attach_runtime_provenance(
        path,
        torch=torch,
        runtime_instance_lifecycle={
            "schema": "reliable_v4_runtime_instance_lifecycle_v1",
            "training_run_id": "p3-d-formal",
            "segments": [{
                "segment_id": "step_{:09d}".format(step),
                "workers": [{
                    "worker_id": 0,
                    "runtime_instance_id": "p3-d-worker-00-runtime-{}".format(step),
                }],
            }],
        },
        runtime_instance_ledger_validation={
            "transition_count": step,
            "unique_runtime_execution_keys": step,
            "duplicate_different_hash_count": 0,
            "runtime_instance_count": 1,
            "runtime_segments_per_worker": {"0": 1},
        },
    )

    restored = load_verified_frozen_checkpoint(
        path,
        torch=torch,
        expected_global_step=step,
        require_runtime_provenance=True,
    )

    assert torch.equal(
        restored["actor_state_dict"]["weight"], original["actor_state_dict"]["weight"]
    )
    assert torch.equal(
        restored["critic1_state_dict"]["weight"], original["critic1_state_dict"]["weight"]
    )
    assert restored["actor_optimizer_state_dict"] == original["actor_optimizer_state_dict"]
    assert restored["critic_optimizer_state_dict"] == original["critic_optimizer_state_dict"]
    assert restored["training_run_id"] == "p3-d-formal"
    assert restored["worker_runtime_provenance_summary"] == {
        "runtime_instance_count": 1,
        "runtime_segments_per_worker": {"0": 1},
        "runtime_instance_ids": ["p3-d-worker-00-runtime-{}".format(step)],
    }


def test_generic_checkpoint_writer_persists_base_frozen_evidence(tmp_path):
    """The shared writer cannot omit immutable model-state fingerprints."""

    torch = pytest.importorskip("torch")
    from planning.common.checkpoint_evidence import save_frozen_checkpoint

    path = tmp_path / "checkpoint_11000.pt"
    save_frozen_checkpoint(
        path,
        torch=torch,
        checkpoint=_checkpoint_payload(torch, step=11000),
    )

    stored = torch.load(path, map_location="cpu", weights_only=False)
    assert stored["p3_frozen_checkpoint_evidence"]["global_step"] == 11000
    assert stored["p3_frozen_checkpoint_evidence"]["actor_fingerprint"]
    assert stored["p3_frozen_checkpoint_evidence"]["critic_fingerprint"]


def test_intermediate_frozen_checkpoint_resumes_awac_actor_critic_and_optimizers(
    tmp_path,
):
    """Loading a 12k-style artifact restores the learner, not only tensors on disk."""

    torch = pytest.importorskip("torch")
    from planning.awac.learner import AWACOptimizationConfig, DiscreteAWACLearner
    from planning.bc.model import build_model, require_torch
    from planning.contracts.feature import NUM_ACTIONS, POLICY_VECTOR_DIM
    from planning.common.checkpoint_evidence import (
        load_verified_frozen_checkpoint,
        save_frozen_checkpoint,
    )
    from planning.contracts.policy_checkpoint_fingerprint import actor_state_sha256

    torch, nn, _, _, _ = require_torch()
    torch.manual_seed(7)
    bc = build_model(nn, depth_channels=1)
    learner = DiscreteAWACLearner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_state_dict=bc.state_dict(),
        depth_channels=1,
        config=AWACOptimizationConfig(
            actor_vector_lr=0.0,
            actor_depth_lr=0.0,
            critic_vector_lr=0.0,
            critic_depth_lr=0.0,
        ),
    )
    mask = torch.ones((2, NUM_ACTIONS), dtype=torch.bool)
    batch = {
        "depth": torch.rand((2, 1, 32, 32)),
        "vector": torch.rand((2, POLICY_VECTOR_DIM)),
        "action_mask": mask,
        "action": torch.tensor([0, 1], dtype=torch.long),
        "reward": torch.tensor([1.0, -1.0]),
        "next_depth": torch.rand((2, 1, 32, 32)),
        "next_vector": torch.rand((2, POLICY_VECTOR_DIM)),
        "next_action_mask": mask.clone(),
        "done": torch.zeros(2),
        "behavior_source": torch.zeros(2, dtype=torch.int64),
    }
    learner.update(batch, update_actor=True)
    payload = {
        **learner.state_dict(),
        "global_step": 12000,
        "replay_size": 12000,
        "bc_reference_fingerprint": actor_state_sha256(
            {"model_state_dict": bc.state_dict()}
        ),
    }
    path = tmp_path / "checkpoint_12000.pt"
    save_frozen_checkpoint(path, torch=torch, checkpoint=payload)
    restored_payload = load_verified_frozen_checkpoint(
        path, torch=torch, expected_global_step=12000
    )
    restored = DiscreteAWACLearner(
        torch=torch,
        nn=nn,
        device=torch.device("cpu"),
        bc_state_dict=bc.state_dict(),
        depth_channels=1,
        config=learner.config,
    )
    restored.load_state_dict(restored_payload)

    assert restored.update_step == learner.update_step
    assert restored.actor_update_count == learner.actor_update_count
    for name, tensor in learner.actor.state_dict().items():
        assert torch.equal(tensor, restored.actor.state_dict()[name])
    for name, tensor in learner.critic1.state_dict().items():
        assert torch.equal(tensor, restored.critic1.state_dict()[name])
    assert restored.actor_optimizer.state_dict()["state"]
    assert restored.critic_optimizer.state_dict()["state"]
