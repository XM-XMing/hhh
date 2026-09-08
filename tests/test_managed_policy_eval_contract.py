"""Identity checks for reusing managed fixed-holdout evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_policy_unity_managed.sh"
_EXACT_OBSERVATION_PROVENANCE = {
    "observation_contract": "reliable_exact_endpoint_snapshot",
    "observation_source": "reliable_exact_endpoint_snapshot",
    "checkpoint_observation_contract": "reliable_exact_endpoint_snapshot",
    "checkpoint_observation_source": "reliable_exact_endpoint_snapshot",
    "expected_observation_contract": "reliable_exact_endpoint_snapshot",
    "runtime_observation_contract": "reliable_exact_endpoint_snapshot",
    "runtime_observation_source": "reliable_exact_endpoint_snapshot",
    "reliable_execution": True,
    "telemetry_observation": False,
    "telemetry_fallback_enabled": False,
    "snapshot_missing_count": 0,
    "telemetry_lookup_count": 0,
    "runtime_contract_override": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_existing(tmp_path: Path, *, checkpoint_contents: bytes):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(checkpoint_contents)
    index = tmp_path / "holdout.csv"
    index.write_text("episode_id\n0\n", encoding="utf-8")
    output = tmp_path / "evaluation"
    output.mkdir()
    summary = {
        "evaluation_contract_id": "policy_unity_fixed_holdout",
        "checkpoint_sha256": _sha256(checkpoint),
        "mission_index_sha256": _sha256(index),
        "episodes": 100,
        "quality_gate_applicable": True,
        "policy_action_mode": "deterministic_argmax",
        "policy_temperature": 0.0,
    }
    summary.update(_EXACT_OBSERVATION_PROVENANCE)
    (output / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    result = subprocess.run(
        [str(SCRIPT), str(checkpoint), str(index), str(output), "100"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    return result, checkpoint


@pytest.mark.unit
def test_managed_evaluation_reuses_only_identical_evidence(tmp_path: Path):
    result, _ = _run_existing(tmp_path, checkpoint_contents=b"candidate-a")
    assert result.returncode == 0, result.stderr
    assert "MANAGED_POLICY_UNITY_EVAL_ALREADY_COMPLETE" in result.stdout


@pytest.mark.unit
def test_managed_evaluation_rejects_stale_checkpoint_summary(tmp_path: Path):
    result, checkpoint = _run_existing(tmp_path, checkpoint_contents=b"candidate-a")
    assert result.returncode == 0
    checkpoint.write_bytes(b"candidate-b")
    stale = subprocess.run(
        [
            str(SCRIPT),
            str(checkpoint),
            str(tmp_path / "holdout.csv"),
            str(tmp_path / "evaluation"),
            "100",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert stale.returncode == 2
    assert "does not match" in stale.stderr


@pytest.mark.unit
def test_managed_evaluation_rejects_legacy_summary_without_action_contract(
    tmp_path: Path,
):
    result, _ = _run_existing(tmp_path, checkpoint_contents=b"candidate-a")
    assert result.returncode == 0
    summary_path = tmp_path / "evaluation" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("policy_action_mode")
    summary.pop("policy_temperature")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    rejected = subprocess.run(
        [
            str(SCRIPT),
            str(tmp_path / "checkpoint.pt"),
            str(tmp_path / "holdout.csv"),
            str(tmp_path / "evaluation"),
            "100",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "does not match" in rejected.stderr


@pytest.mark.unit
def test_managed_formal_evaluation_rejects_small_sample(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"candidate")
    index = tmp_path / "holdout.csv"
    index.write_text("episode_id\n0\n", encoding="utf-8")
    result = subprocess.run(
        [
            str(SCRIPT), str(checkpoint), str(index),
            str(tmp_path / "evaluation"), "99",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "EPISODES >= 100" in result.stderr


@pytest.mark.unit
def test_managed_full_audit_reuse_requires_audit_only_summary(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"candidate")
    index = tmp_path / "holdout.csv"
    index.write_text("episode_id\n0\n", encoding="utf-8")
    output = tmp_path / "audit"
    output.mkdir()
    summary = {
        "evaluation_contract_id": "policy_unity_fixed_holdout",
        "checkpoint_sha256": _sha256(checkpoint),
        "mission_index_sha256": _sha256(index),
        "episodes": 100,
        "quality_gate_applicable": False,
        "policy_action_mode": "deterministic_argmax",
        "policy_temperature": 0.0,
    }
    summary.update(_EXACT_OBSERVATION_PROVENANCE)
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    environment = dict(os.environ)
    environment["EVAL_FULL_AUDIT_ONLY"] = "1"
    result = subprocess.run(
        [str(SCRIPT), str(checkpoint), str(index), str(output), "100"],
        cwd=str(ROOT), text=True, capture_output=True, check=False, env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert "ALREADY_COMPLETE" in result.stdout


@pytest.mark.unit
def test_managed_evaluation_forwards_explicit_normalizer_source(tmp_path: Path):
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'EVAL_NORMALIZER_CHECKPOINT="${EVAL_NORMALIZER_CHECKPOINT:-}"' in source
    assert 'eval_args+=(--normalizer-checkpoint "$EVAL_NORMALIZER_CHECKPOINT")' in source
