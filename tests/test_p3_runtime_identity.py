"""TDD contract tests for P3 runtime identity and Unity v4 capability gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from planning.runtime.identity import (
    RuntimeIdentityExpectation,
    RuntimeIdentityMismatchError,
    build_and_validate_runtime_manifest,
)


pytestmark = pytest.mark.unit


CAPABILITY_MARKERS = (
    b"reliable_command_v4",
    b"primitive_result_v4",
    b"snapshot_v4",
    b"reset_v4",
)


def _file(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def _inputs(tmp_path: Path, *, assembly: bytes = b"assembly"):
    files = {
        "player": _file(tmp_path / "XMflight.x86_64", b"player"),
        "assembly": _file(tmp_path / "Assembly-CSharp.dll", assembly),
        "bridge": _file(tmp_path / "unity_bridge_node", b"bridge"),
        "schema": _file(tmp_path / "primitive_execution_schema_v4.md", b"schema"),
        "bc": _file(tmp_path / "checkpoint_best.pt", b"checkpoint"),
    }
    return files


def _expectations(files, *, assembly: bytes | None = None):
    def digest(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    assembly_payload = b"assembly" if assembly is None else assembly
    return RuntimeIdentityExpectation(
        player_sha256=digest(b"player"),
        assembly_csharp_sha256=digest(assembly_payload),
        bridge_sha256=digest(b"bridge"),
        schema_spec_sha256=digest(b"schema"),
        bc_checkpoint_sha256=digest(b"checkpoint"),
    )


def test_same_player_sha_with_wrong_assembly_sha_fails_fast(tmp_path: Path):
    files = _inputs(tmp_path, assembly=b"wrong-assembly")
    expected = _expectations(files, assembly=b"known-good-assembly")

    with pytest.raises(RuntimeIdentityMismatchError, match="STARTUP_RUNTIME_IDENTITY_MISMATCH"):
        build_and_validate_runtime_manifest(
            **files,
            expected=expected,
            manifest_path=tmp_path / "runtime_manifest.json",
        )


def test_correct_player_and_assembly_with_all_capabilities_is_accepted(tmp_path: Path):
    assembly = b"known-good-assembly\0" + b"\0".join(CAPABILITY_MARKERS)
    files = _inputs(tmp_path, assembly=assembly)
    expected = _expectations(files, assembly=assembly)

    manifest = build_and_validate_runtime_manifest(
        **files,
        expected=expected,
        manifest_path=tmp_path / "runtime_manifest.json",
    )

    assert manifest["player_sha256"] == expected.player_sha256
    assert manifest["assembly_csharp_sha256"] == expected.assembly_csharp_sha256
    assert set(manifest["capabilities"]) == set(
        ("reliable_command_v4", "primitive_result_v4", "snapshot_v4", "reset_v4")
    )


def test_missing_reset_v4_capability_fails_before_runtime_start(tmp_path: Path):
    assembly = b"known-good-assembly\0" + b"\0".join(CAPABILITY_MARKERS[:-1])
    files = _inputs(tmp_path, assembly=assembly)
    expected = _expectations(files, assembly=assembly)

    with pytest.raises(RuntimeIdentityMismatchError, match="reset_v4"):
        build_and_validate_runtime_manifest(
            **files,
            expected=expected,
            manifest_path=tmp_path / "runtime_manifest.json",
        )


def test_manifest_persists_player_and_assembly_hashes(tmp_path: Path):
    assembly = b"known-good-assembly\0" + b"\0".join(CAPABILITY_MARKERS)
    files = _inputs(tmp_path, assembly=assembly)
    expected = _expectations(files, assembly=assembly)
    manifest_path = tmp_path / "runtime_manifest.json"

    manifest = build_and_validate_runtime_manifest(
        **files,
        expected=expected,
        manifest_path=manifest_path,
    )

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["player_sha256"] == manifest["player_sha256"]
    assert persisted["assembly_csharp_sha256"] == manifest["assembly_csharp_sha256"]
