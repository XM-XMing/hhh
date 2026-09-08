"""Integrity audit tests for the formal AWAC replay owner."""

from __future__ import annotations

import json

import numpy as np
import pytest

from planning.awac.interaction import BehaviorSource
from planning.awac.replay import AWACReplayBuffer
from planning.awac.replay_audit import audit_awac_replay


pytestmark = pytest.mark.unit


def _replay(tmp_path):
    replay = AWACReplayBuffer.create(
        tmp_path / "replay",
        capacity=4,
        depth_shape=(1, 2, 2),
        vector_dim=3,
        action_dim=5,
        bc_checkpoint_sha256="b" * 64,
        task_contract_id="task",
        task_contract_sha256="t" * 64,
        mpl_contract_sha256="m" * 64,
        run_identity="run",
        run_contract_sha256="r" * 64,
    )
    mask = np.ones((5,), dtype=np.bool_)
    replay.add(
        depth=np.zeros((1, 2, 2), dtype=np.float32),
        vector=np.zeros(3, dtype=np.float32),
        action_mask=mask,
        action=1,
        reward=1.0,
        next_depth=np.ones((1, 2, 2), dtype=np.float32),
        next_vector=np.ones(3, dtype=np.float32),
        next_action_mask=mask,
        done=True,
        behavior_source=int(BehaviorSource.BC_WARMUP),
    )
    replay.flush()
    return replay


def test_awac_replay_audit_reports_core_integrity(tmp_path):
    _replay(tmp_path)
    report = audit_awac_replay(tmp_path / "replay")

    assert report["status"] == "PASS"
    assert report["row_count"] == 1
    assert report["terminal_count"] == 1
    assert report["legacy_replay_transition_count"] == 0


def test_awac_replay_audit_rejects_extra_array_field(tmp_path):
    _replay(tmp_path)
    np.save(tmp_path / "replay" / "privileged.npy", np.zeros(1, dtype=np.float32))

    with pytest.raises(ValueError, match="unsupported array files"):
        audit_awac_replay(tmp_path / "replay")


def test_awac_replay_audit_cli_writes_json(tmp_path):
    _replay(tmp_path)
    output = tmp_path / "audit.json"
    from planning.awac.replay_audit import main

    assert main(
        ["--replay-dir", str(tmp_path / "replay"), "--output", str(output)]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"
