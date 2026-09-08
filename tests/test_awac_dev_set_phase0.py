"""Mission isolation and final-test protection for AWAC Phase 0."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from planning.awac.dev_selection import validate_training_mission_source
from planning.evaluation.mission_split import (
    build_awac_calibration_split,
    build_awac_dev_set,
    validate_awac_calibration_split,
    validate_awac_dev_set,
)


def _write_index(path: Path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_id", "mission_id"])
        writer.writeheader()
        for episode_id, mission_id in enumerate(ids):
            writer.writerow({"episode_id": episode_id, "mission_id": mission_id})


@pytest.mark.unit
def test_awac_dev_set_is_deterministic_and_excludes_bc_and_final(tmp_path: Path):
    source = tmp_path / "missions.csv"
    bc = tmp_path / "bc.csv"
    final = tmp_path / "final.csv"
    dev = tmp_path / "awac_dev" / "missions.csv"
    manifest_path = tmp_path / "awac_dev" / "manifest.json"
    source_ids = ["m{:03d}".format(index) for index in range(20)]
    _write_index(source, source_ids)
    _write_index(bc, source_ids[:5])
    _write_index(final, source_ids[5:10])

    manifest = build_awac_dev_set(
        source,
        bc_train_index_paths=[bc],
        final_test_index_paths=[final],
        dev_index_path=dev,
        manifest_path=manifest_path,
        count=4,
        seed=4026,
    )
    validated = validate_awac_dev_set(manifest_path, verify_referenced_indexes=True)
    assert manifest["role"] == "DEV"
    assert validated["dev"]["row_count"] == 4
    assert validated["intersections"] == {
        "dev_bc_train": 0,
        "dev_final_test": 0,
    }
    assert manifest["selection_method"] == "sha256_seed_mission_id_then_source_order"

    final_manifest = tmp_path / "final_manifest.json"
    final_manifest.write_text(
        json.dumps({"role": "FINAL_TEST", "mission_count": 5}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="FINAL_TEST_ONLY"):
        validate_training_mission_source(final, final_test_manifest=final_manifest)


@pytest.mark.unit
def test_final_test_manifest_is_not_accepted_as_calibration_source(tmp_path: Path):
    final_manifest = tmp_path / "final_manifest.json"
    final_manifest.write_text(
        json.dumps(
            {
                "role": "FINAL_TEST",
                "contract_id": "bc_closed_loop_eval_300_manifest",
                "mission_count": 300,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="FINAL_TEST"):
        validate_training_mission_source(
            tmp_path / "final.csv",
            final_test_manifest=final_manifest,
        )


@pytest.mark.unit
def test_calibration_split_is_episode_disjoint_and_source_stable(tmp_path: Path):
    source = tmp_path / "source.csv"
    excluded = tmp_path / "excluded.csv"
    output = tmp_path / "calibration_split.json"
    source_ids = ["m{:03d}".format(index) for index in range(30)]
    _write_index(source, source_ids)
    _write_index(excluded, source_ids[:5])

    split = build_awac_calibration_split(
        source,
        [excluded],
        output,
        seed=4026,
        holdout_fraction=0.20,
    )
    assert split["role"] == "CALIBRATION_SPLIT"
    validated = validate_awac_calibration_split(output)
    assert validated["intersections"] == {
        "train_holdout": 0,
        "train_excluded": 0,
        "holdout_excluded": 0,
    }
    assert len(validated["train_episode_ids"]) + len(
        validated["holdout_episode_ids"]
    ) == 25
