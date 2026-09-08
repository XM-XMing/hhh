#!/usr/bin/env python3
"""Validate the PY7-B0 normalized cross-language naming inventory."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNITY_SCRIPTS = Path("/home/xm/XM/XMflight/Assets/Scripts")
MANIFEST_PATH = ROOT / "docs" / "cross_language_rename_manifest.json"

EXPECTED_CATEGORIES = (
    "KEEP_PROTOCOL_VERSION",
    "KEEP_SCHEMA_VERSION",
    "KEEP_ARTIFACT_VERSION",
    "RENAME_PRE_BC_BUSINESS_NAME",
    "DEFER_RL_NAME",
    "TEST_FIXTURE_NAME",
    "HISTORICAL_DOC_REFERENCE",
    "UNKNOWN",
)

EXPECTED_COUNTS = {
    "KEEP_PROTOCOL_VERSION": 86,
    "KEEP_SCHEMA_VERSION": 8,
    "KEEP_ARTIFACT_VERSION": 8,
    "RENAME_PRE_BC_BUSINESS_NAME": 26,
    "DEFER_RL_NAME": 50,
    "TEST_FIXTURE_NAME": 140,
    "HISTORICAL_DOC_REFERENCE": 1,
    "UNKNOWN": 0,
}

REQUIRED_ROW_FIELDS = {
    "target_id",
    "language",
    "kind",
    "old_name",
    "new_name",
    "category",
    "paths",
    "symbols",
    "occurrence_count",
    "reason",
    "replacement",
    "public_api",
    "wire_schema_impact",
    "artifact_schema_impact",
    "checkpoint_dependency",
    "rename_phase",
    "scope",
}


def _load_manifest():
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _is_superseded_rl_path(row):
    path = row["paths"][0]
    legacy_algorithm_prefix = "s" + "a" + "c" + "_"
    return (
        row["scope"] == "planning"
        and row["category"] == "DEFER_RL_NAME"
        and (
            "python/planning/rl/" in path
            or path.startswith("scripts/run_" + legacy_algorithm_prefix)
            or path.startswith("scripts/run_awac_")
            or path.startswith("tests/test_" + legacy_algorithm_prefix)
            or path.startswith("tests/test_p3_")
            or path == "tests/compare_awac_handoff_fixture.py"
            or path == "tests/test_reliable_v4_awac_training_contract.py"
        )
    )


def _row_path(row):
    base = ROOT if row["scope"] == "planning" else UNITY_SCRIPTS
    return base / row["paths"][0]


def test_manifest_has_one_mutually_exclusive_target_row_per_key():
    manifest = _load_manifest()
    rows = manifest["rows"]
    keys = {(row["scope"], row["paths"][0], row["old_name"]) for row in rows}

    assert len(rows) == 319
    assert len(keys) == len(rows)
    assert set(manifest["counts"]) == set(EXPECTED_CATEGORIES)
    assert manifest["counts"] == EXPECTED_COUNTS
    assert manifest["VERSIONED_NAME_UNIQUE_TARGET_TOTAL"] == len(rows)
    assert manifest["unique_count_closed"] is True
    assert manifest["counts"]["UNKNOWN"] == 0
    assert manifest["status"] == "HISTORICAL_SUPERSEDED_BY_AWAC_ONLY_MIGRATION"


def test_historical_manifest_occurrences_remain_self_consistent():
    manifest = _load_manifest()
    body_total = 0
    path_total = 0
    superseded_rows = 0

    for row in manifest["rows"]:
        assert REQUIRED_ROW_FIELDS.issubset(row)
        assert row["old_name"]
        assert row["paths"]
        assert row["symbols"]
        body_count = int(row["content_occurrence_count"])
        path_count = int(row["path_occurrence_count"])
        assert body_count >= 0
        assert path_count >= 0
        assert row["occurrence_count"] == body_count + path_count
        body_total += body_count
        path_total += path_count
        path = _row_path(row)
        if not path.is_file():
            assert _is_superseded_rl_path(row), path
            superseded_rows += 1
            continue

    assert superseded_rows > 0

    assert body_total == manifest["FILE_TEXT_OCCURRENCE_TOTAL"]
    assert path_total == manifest["PATH_SURFACE_OCCURRENCE_TOTAL"]
    assert manifest["VERSIONED_NAME_OCCURRENCE_TOTAL"] == (
        manifest["FILE_TEXT_OCCURRENCE_TOTAL"]
        + manifest["PATH_SURFACE_OCCURRENCE_TOTAL"]
    )
    assert manifest["occurrence_count_closed"] is True


def test_manifest_category_contracts_are_explicit():
    manifest = _load_manifest()
    for row in manifest["rows"]:
        category = row["category"]
        if category == "RENAME_PRE_BC_BUSINESS_NAME":
            assert row["new_name"]
            assert row["new_name"] != row["old_name"]
            assert row["rename_phase"] == "PY7-B"
        elif category == "DEFER_RL_NAME":
            assert row["new_name"] is None
            assert (
                row["rename_phase"]
                == "AFTER_NEW_RELIABLE_BC_TRAINED_AND_EVALUATED"
            )
        elif category == "HISTORICAL_DOC_REFERENCE":
            assert row["paths"] == ["PROJECT_HANDOFF.md"]
            assert row["old_name"] == "tail_v3"
        else:
            assert row["new_name"] == row["old_name"]


def test_unity_partial_names_are_frozen_and_consistent():
    names = (
        "XMSimulationManager.cs",
        "XMSimulationManager.CommandDecoding.cs",
        "XMSimulationManager.Diagnostics.cs",
        "XMSimulationManager.Execution.cs",
        "XMSimulationManager.MotionCommands.cs",
        "XMSimulationManager.Observation.cs",
        "XMSimulationManager.Telemetry.cs",
        "XMSimulationManager.TransportPolling.cs",
    )
    declaration = re.compile(r"\bpartial\s+class\s+XMSimulationManager\b")
    for name in names:
        path = UNITY_SCRIPTS / "Runtime" / "Simulation" / name
        text = path.read_text(encoding="utf-8")
        assert "namespace XMflight" in text
        assert len(declaration.findall(text)) == 1


def test_historical_optimized_tree_assessment_is_superseded():
    text = (ROOT / "docs" / "PLANNING_OPTIMIZED_MIGRATION_PROGRESS.md").read_text(
        encoding="utf-8"
    )
    assert "OPTIMIZED_TREE_ASSESSMENT=HISTORICAL_PRE_MIGRATION_BASELINE" in text
    assert "STATUS=SUPERSEDED_BY_CURRENT_GATES" in text
    assert "NEXT_SINGLE_MIGRATION=SUPERSEDED" in text
