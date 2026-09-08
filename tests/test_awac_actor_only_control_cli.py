"""Contract tests for the fixed-Critic Actor-only A/B/C command owner."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_awac_actor_only_ab_control.py"
SPEC = importlib.util.spec_from_file_location("awac_actor_only_control_cli", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_fixed_batch_sequences_are_reproducible_and_shared_by_weight_modes():
    """B and C must consume identical train/trust rows for all 100 proposals."""

    first = MODULE.build_fixed_batch_sequences(
        replay_size=301,
        proposal_count=100,
        batch_size=8,
        seed=20260906,
    )
    second = MODULE.build_fixed_batch_sequences(
        replay_size=301,
        proposal_count=100,
        batch_size=8,
        seed=20260906,
    )
    assert first["train_indices"].shape == (100, 8)
    assert first["trust_indices"].shape == (100, 8)
    assert first["validation_indices"].shape == (8,)
    assert (first["train_indices"] == second["train_indices"]).all()
    assert (first["trust_indices"] == second["trust_indices"]).all()
    assert (first["validation_indices"] == second["validation_indices"]).all()


def test_terminal_transition_table_requires_paired_identity_and_one_outcome():
    left = [
        {"episode_id": "0", "mission_id": "m0", "success": "1"},
        {"episode_id": "1", "mission_id": "m1", "dead_end": "1"},
    ]
    right = [
        {"episode_id": "0", "mission_id": "m0", "dead_end": "1"},
        {"episode_id": "1", "mission_id": "m1", "success": "1"},
    ]
    table = MODULE.terminal_transition_table(left, right)
    assert table["success"]["dead_end"] == 1
    assert table["dead_end"]["success"] == 1
