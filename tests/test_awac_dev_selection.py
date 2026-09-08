from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from planning.contracts.policy_runtime import (
    POLICY_RUNTIME_CONTRACT_ID,
    POLICY_UNITY_EVALUATION_CONTRACT_ID,
    POLICY_UNITY_OUTCOME_CONTRACT_ID,
)
from planning.awac.dev_selection import (
    AWAC_DEV_SELECTION_CONTRACT_ID,
    select_awac_dev_checkpoint,
)
from planning.evaluation.mission_split import build_awac_mission_split
from planning.common import file_sha256


def _write_index(path: Path, mission_ids) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_id", "mission_id"])
        writer.writeheader()
        for episode_id, mission_id in enumerate(mission_ids):
            writer.writerow({"episode_id": episode_id, "mission_id": mission_id})


def _selection_files(tmp_path: Path):
    train_ids = ["train-{:03d}".format(index) for index in range(10)]
    dev_ids = ["dev-{:03d}".format(index) for index in range(100)]
    final_ids = ["final-{:03d}".format(index) for index in range(100)]
    source = tmp_path / "source.csv"
    dev = tmp_path / "dev.csv"
    final = tmp_path / "final.csv"
    train = tmp_path / "train.csv"
    manifest_path = tmp_path / "split.json"
    _write_index(source, train_ids + dev_ids + final_ids)
    _write_index(dev, dev_ids)
    _write_index(final, final_ids)
    manifest = build_awac_mission_split(
        source,
        dev,
        final,
        train,
        manifest_path,
    )
    candidate = tmp_path / "candidate.pt"
    incumbent = tmp_path / "incumbent.pt"
    candidate.write_bytes(b"candidate-checkpoint")
    incumbent.write_bytes(b"incumbent-checkpoint")
    return manifest_path, manifest, candidate, incumbent


def _summary(
    *,
    checkpoint: Path,
    mission_sha256: str,
    success: int,
    collision: int,
    dead_end: int,
    timeout: int,
):
    episodes = 100
    counts = {
        "success": success,
        "collision": collision,
        "dead_end": dead_end,
        "timeout": timeout,
        "far": 0,
        "altitude_violation": 0,
    }
    assert sum(counts.values()) == episodes
    payload = {
        "evaluation_contract_id": POLICY_UNITY_EVALUATION_CONTRACT_ID,
        "policy_unity_outcome_contract_id": POLICY_UNITY_OUTCOME_CONTRACT_ID,
        "policy_runtime_contract_id": POLICY_RUNTIME_CONTRACT_ID,
        "episodes": episodes,
        "terminal_outcome_total": episodes,
        "unclassified_count": 0,
        "ambiguous_outcome_count": 0,
        "outcome_partition_valid": True,
        "quality_gate_applicable": True,
        "quality_pass": False,
        "checkpoint_sha256": file_sha256(checkpoint),
        "mission_index_sha256": mission_sha256,
        "safety_mask": "depth",
        "execution_mode": "continuous",
        "policy_action_mode": "deterministic_argmax",
        "policy_temperature": 0.0,
        "runtime_contract_override": False,
        "depth_mask_runtime_override": False,
        "local_depth_collision_mask": True,
        "global_collision_mask": False,
        "depth_mask_config": {
            "collision_radius_m": 0.40,
            "slack_m": 0.08,
            "sample_stride": 4,
            "max_patch_radius_px": 14,
        },
        "episode_filter_ids": [],
    }
    for name, count in counts.items():
        payload["{}_count".format(name)] = count
        payload["{}_rate".format(name)] = float(count) / episodes
    return payload


def _write_summary(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.unit
def test_dev_selector_atomically_accepts_strict_safe_improvement(tmp_path: Path):
    manifest_path, manifest, candidate, incumbent = _selection_files(tmp_path)
    dev_sha256 = manifest["dev"]["file_sha256"]
    candidate_summary = _write_summary(
        tmp_path / "candidate.json",
        _summary(
            checkpoint=candidate,
            mission_sha256=dev_sha256,
            success=62,
            collision=4,
            dead_end=24,
            timeout=10,
        ),
    )
    incumbent_summary = _write_summary(
        tmp_path / "incumbent.json",
        _summary(
            checkpoint=incumbent,
            mission_sha256=dev_sha256,
            success=60,
            collision=5,
            dead_end=25,
            timeout=10,
        ),
    )
    selected = tmp_path / "selected.pt"
    decision_path = tmp_path / "decision.json"

    decision = select_awac_dev_checkpoint(
        split_manifest_path=manifest_path,
        candidate_checkpoint_path=candidate,
        candidate_summary_path=candidate_summary,
        incumbent_checkpoint_path=incumbent,
        incumbent_summary_path=incumbent_summary,
        out_checkpoint_path=selected,
        out_decision_path=decision_path,
    )

    assert decision["contract_id"] == AWAC_DEV_SELECTION_CONTRACT_ID
    assert decision["candidate_selected"] is True
    assert selected.read_bytes() == candidate.read_bytes()
    assert json.loads(decision_path.read_text(encoding="utf-8")) == decision


@pytest.mark.unit
def test_dev_selector_keeps_incumbent_when_collision_regresses(tmp_path: Path):
    manifest_path, manifest, candidate, incumbent = _selection_files(tmp_path)
    dev_sha256 = manifest["dev"]["file_sha256"]
    candidate_summary = _write_summary(
        tmp_path / "candidate.json",
        _summary(
            checkpoint=candidate,
            mission_sha256=dev_sha256,
            success=70,
            collision=6,
            dead_end=14,
            timeout=10,
        ),
    )
    incumbent_summary = _write_summary(
        tmp_path / "incumbent.json",
        _summary(
            checkpoint=incumbent,
            mission_sha256=dev_sha256,
            success=60,
            collision=5,
            dead_end=25,
            timeout=10,
        ),
    )
    selected = tmp_path / "selected.pt"

    decision = select_awac_dev_checkpoint(
        split_manifest_path=manifest_path,
        candidate_checkpoint_path=candidate,
        candidate_summary_path=candidate_summary,
        incumbent_checkpoint_path=incumbent,
        incumbent_summary_path=incumbent_summary,
        out_checkpoint_path=selected,
        out_decision_path=tmp_path / "decision.json",
    )

    assert decision["gates"]["score_strict_improvement"] is True
    assert decision["gates"]["collision_non_regression"] is False
    assert decision["candidate_selected"] is False
    assert selected.read_bytes() == incumbent.read_bytes()


@pytest.mark.unit
def test_dev_selector_rejects_final_holdout_or_stale_checkpoint_evidence(tmp_path: Path):
    manifest_path, manifest, candidate, incumbent = _selection_files(tmp_path)
    final_sha256 = manifest["final_holdout"]["file_sha256"]
    dev_sha256 = manifest["dev"]["file_sha256"]
    candidate_payload = _summary(
        checkpoint=candidate,
        mission_sha256=final_sha256,
        success=62,
        collision=4,
        dead_end=24,
        timeout=10,
    )
    incumbent_payload = _summary(
        checkpoint=incumbent,
        mission_sha256=dev_sha256,
        success=60,
        collision=5,
        dead_end=25,
        timeout=10,
    )

    with pytest.raises(ValueError, match="final holdout"):
        select_awac_dev_checkpoint(
            split_manifest_path=manifest_path,
            candidate_checkpoint_path=candidate,
            candidate_summary_path=_write_summary(tmp_path / "candidate.json", candidate_payload),
            incumbent_checkpoint_path=incumbent,
            incumbent_summary_path=_write_summary(tmp_path / "incumbent.json", incumbent_payload),
            out_checkpoint_path=tmp_path / "selected.pt",
            out_decision_path=tmp_path / "decision.json",
        )

    candidate_payload["mission_index_sha256"] = dev_sha256
    candidate_payload["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        select_awac_dev_checkpoint(
            split_manifest_path=manifest_path,
            candidate_checkpoint_path=candidate,
            candidate_summary_path=_write_summary(tmp_path / "candidate.json", candidate_payload),
            incumbent_checkpoint_path=incumbent,
            incumbent_summary_path=tmp_path / "incumbent.json",
            out_checkpoint_path=tmp_path / "selected.pt",
            out_decision_path=tmp_path / "decision.json",
        )


@pytest.mark.unit
def test_dev_selector_rejects_mismatched_runtime_or_incomplete_outcomes(tmp_path: Path):
    manifest_path, manifest, candidate, incumbent = _selection_files(tmp_path)
    dev_sha256 = manifest["dev"]["file_sha256"]
    candidate_payload = _summary(
        checkpoint=candidate,
        mission_sha256=dev_sha256,
        success=62,
        collision=4,
        dead_end=24,
        timeout=10,
    )
    incumbent_payload = _summary(
        checkpoint=incumbent,
        mission_sha256=dev_sha256,
        success=60,
        collision=5,
        dead_end=25,
        timeout=10,
    )
    candidate_payload["depth_mask_config"]["collision_radius_m"] = 0.41

    with pytest.raises(ValueError, match="runtime/depth contracts differ"):
        select_awac_dev_checkpoint(
            split_manifest_path=manifest_path,
            candidate_checkpoint_path=candidate,
            candidate_summary_path=_write_summary(tmp_path / "candidate.json", candidate_payload),
            incumbent_checkpoint_path=incumbent,
            incumbent_summary_path=_write_summary(tmp_path / "incumbent.json", incumbent_payload),
            out_checkpoint_path=tmp_path / "selected.pt",
            out_decision_path=tmp_path / "decision.json",
        )

    candidate_payload["depth_mask_config"]["collision_radius_m"] = 0.40
    candidate_payload["outcome_partition_valid"] = False
    with pytest.raises(ValueError, match="outcome partition"):
        select_awac_dev_checkpoint(
            split_manifest_path=manifest_path,
            candidate_checkpoint_path=candidate,
            candidate_summary_path=_write_summary(tmp_path / "candidate.json", candidate_payload),
            incumbent_checkpoint_path=incumbent,
            incumbent_summary_path=tmp_path / "incumbent.json",
            out_checkpoint_path=tmp_path / "selected.pt",
            out_decision_path=tmp_path / "decision.json",
        )


@pytest.mark.unit
def test_dev_selector_rejects_stochastic_or_missing_action_mode_evidence(
    tmp_path: Path,
):
    manifest_path, manifest, candidate, incumbent = _selection_files(tmp_path)
    dev_sha256 = manifest["dev"]["file_sha256"]
    candidate_payload = _summary(
        checkpoint=candidate,
        mission_sha256=dev_sha256,
        success=62,
        collision=4,
        dead_end=24,
        timeout=10,
    )
    incumbent_payload = _summary(
        checkpoint=incumbent,
        mission_sha256=dev_sha256,
        success=60,
        collision=5,
        dead_end=25,
        timeout=10,
    )
    candidate_payload["policy_action_mode"] = "masked_categorical"
    candidate_payload["policy_temperature"] = 0.5
    with pytest.raises(ValueError, match="deterministic_argmax"):
        select_awac_dev_checkpoint(
            split_manifest_path=manifest_path,
            candidate_checkpoint_path=candidate,
            candidate_summary_path=_write_summary(
                tmp_path / "candidate_stochastic.json", candidate_payload
            ),
            incumbent_checkpoint_path=incumbent,
            incumbent_summary_path=_write_summary(
                tmp_path / "incumbent.json", incumbent_payload
            ),
            out_checkpoint_path=tmp_path / "selected.pt",
            out_decision_path=tmp_path / "decision.json",
        )

    candidate_payload["policy_action_mode"] = "deterministic_argmax"
    candidate_payload.pop("policy_temperature")
    with pytest.raises(ValueError, match="policy temperature"):
        select_awac_dev_checkpoint(
            split_manifest_path=manifest_path,
            candidate_checkpoint_path=candidate,
            candidate_summary_path=_write_summary(
                tmp_path / "candidate_missing_temperature.json", candidate_payload
            ),
            incumbent_checkpoint_path=incumbent,
            incumbent_summary_path=tmp_path / "incumbent.json",
            out_checkpoint_path=tmp_path / "selected.pt",
            out_decision_path=tmp_path / "decision.json",
        )
