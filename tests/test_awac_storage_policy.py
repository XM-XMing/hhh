"""Tests for AWAC checkpoint retention and disk-budget policy."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def _write_committed_generation(root: Path, generation: int) -> Path:
    artifact = root / "checkpoint_last.generation-{:08d}.pt".format(generation)
    artifact.write_bytes(("generation-{}\n".format(generation)).encode("ascii"))
    manifest = {
        "generation": int(generation),
        "checkpoint_name": "checkpoint_last",
        "checkpoint_kind": "checkpoint_last",
        "checkpoint_filename": artifact.name,
        "checkpoint_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "transaction_state": "COMMITTED",
        "checkpoint_final_commit": "PASS",
    }
    (root / "checkpoint_last.transaction.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return artifact


def test_checkpoint_retention_keeps_latest_two_and_preserves_uncommitted_tail(
    tmp_path: Path,
):
    from planning.awac.storage import (
        checkpoint_retention_plan,
        garbage_collect_checkpoint_generations,
    )

    for generation in (1, 2, 3):
        _write_committed_generation(tmp_path, generation)

    # A generation written after the last committed marker is not safe to GC.
    uncommitted = tmp_path / "checkpoint_last.generation-00000004.pt"
    uncommitted.write_bytes(b"partial-generation")

    plan = checkpoint_retention_plan(tmp_path)
    assert plan["keep_generations"] == [2, 3]
    assert plan["delete_generations"] == [1]
    assert plan["preserved_uncommitted_generations"] == [4]

    result = garbage_collect_checkpoint_generations(tmp_path)
    assert result["status"] == "PASS"
    assert result["deleted_generations"] == [1]
    assert not (tmp_path / "checkpoint_last.generation-00000001.pt").exists()
    assert (tmp_path / "checkpoint_last.generation-00000002.pt").exists()
    assert (tmp_path / "checkpoint_last.generation-00000003.pt").exists()
    assert uncommitted.exists()


def test_rolling_gc_does_not_touch_pinned_checkpoint_names(tmp_path: Path):
    from planning.awac.storage import garbage_collect_checkpoint_generations

    for generation in (1, 2, 3):
        _write_committed_generation(tmp_path, generation)
    pinned = (
        tmp_path / "checkpoint_calibration_pass.generation-00000001.pt",
        tmp_path / "checkpoint_calibration_pass.pt",
        tmp_path / "checkpoint_awac_10k.pt",
    )
    for path in pinned:
        path.write_bytes(b"pinned")

    garbage_collect_checkpoint_generations(tmp_path)

    assert all(path.exists() for path in pinned)


def test_gc_delete_failure_is_a_warning_after_commit(tmp_path: Path, monkeypatch):
    from planning.awac.storage import garbage_collect_checkpoint_generations

    for generation in (1, 2, 3):
        _write_committed_generation(tmp_path, generation)

    def deny_unlink(path):
        raise OSError("synthetic disk permission failure")

    monkeypatch.setattr(Path, "unlink", deny_unlink)
    result = garbage_collect_checkpoint_generations(tmp_path)

    assert result["status"] == "WARNING"
    assert result["gc_warning_count"] == 1
    assert result["deleted_generations"] == []
    assert (tmp_path / "checkpoint_last.generation-00000001.pt").exists()


def test_disk_guard_reserves_formal_headroom():
    from planning.awac.storage import GIB, evaluate_disk_guard

    passing = evaluate_disk_guard(
        free_bytes=138 * GIB,
        estimated_run_bytes=5 * GIB,
    )
    assert passing["status"] == "PASS"
    assert passing["headroom_bytes"] == 133 * GIB

    failing = evaluate_disk_guard(
        free_bytes=22 * GIB,
        estimated_run_bytes=5 * GIB,
    )
    assert failing["status"] == "FAIL"
    assert failing["headroom_bytes"] == 17 * GIB


def test_checkpoint_gc_cli_is_a_dry_run(tmp_path: Path, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "plan_awac_checkpoint_gc.py"
    spec = importlib.util.spec_from_file_location("plan_awac_checkpoint_gc", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    _write_committed_generation(tmp_path, 1)
    _write_committed_generation(tmp_path, 2)
    _write_committed_generation(tmp_path, 3)

    assert module.main(["--checkpoint-dir", str(tmp_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["committed_generation"] == 3
    assert [item["generation"] for item in output["safe_delete_candidates"]] == [1]
    assert (tmp_path / "checkpoint_last.generation-00000001.pt").exists()
