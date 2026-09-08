"""PY7-A characterization tests for common and configuration ownership."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def test_common_canonical_json_preserves_the_two_existing_byte_contracts():
    from planning.common.hashing import canonical_json_sha256

    payload = {"z": "雪", "a": [1, 2, 3]}
    for ensure_ascii in (True, False):
        encoded = json.dumps(
            payload,
            ensure_ascii=ensure_ascii,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert canonical_json_sha256(payload, ensure_ascii=ensure_ascii) == hashlib.sha256(
            encoded
        ).hexdigest()


def test_common_json_writer_preserves_newline_selectively(tmp_path: Path):
    from planning.common.io import write_json_atomic

    without_newline = tmp_path / "without.json"
    with_newline = tmp_path / "with.json"
    write_json_atomic(without_newline, {"b": 2, "a": 1})
    write_json_atomic(with_newline, {"b": 2, "a": 1}, trailing_newline=True)
    assert without_newline.read_bytes() == b'{\n  "a": 1,\n  "b": 2\n}'
    assert with_newline.read_bytes() == b'{\n  "a": 1,\n  "b": 2\n}\n'
    assert not list(tmp_path.glob("*.tmp"))


def test_protocol_constants_are_the_single_python_execution_owner():
    from planning.protocol.constants import (
        PRIMITIVE_FRAME_COUNT,
        PROTOCOL_VERSION,
        SCHEMA_VERSION as CONSTANT_SCHEMA_VERSION,
    )
    from planning.protocol.primitive_execution_command_v4 import (
        FRAME_COUNT,
        SCHEMA_VERSION as COMMAND_SCHEMA_VERSION,
    )
    from planning.protocol.primitive_execution_schema_v4 import (
        DEFAULT_REQUESTED_FRAME_COUNT,
        SCHEMA_VERSION as RESULT_SCHEMA_VERSION,
    )

    assert PROTOCOL_VERSION == 4
    assert CONSTANT_SCHEMA_VERSION == 4
    assert PRIMITIVE_FRAME_COUNT == 25
    assert COMMAND_SCHEMA_VERSION == CONSTANT_SCHEMA_VERSION
    assert RESULT_SCHEMA_VERSION == CONSTANT_SCHEMA_VERSION
    assert FRAME_COUNT == PRIMITIVE_FRAME_COUNT
    assert DEFAULT_REQUESTED_FRAME_COUNT == PRIMITIVE_FRAME_COUNT


def test_algorithm_defaults_are_projected_from_domain_owners():
    from planning.bc.trainer import BCTrainingConfig, build_argument_parser
    from planning.safety.depth_action_masks import build_argument_parser as build_depth_parser
    from planning.safety.depth_safety import DepthSafetyConfig
    from planning.teacher.labeling import build_argument_parser as build_label_parser
    from planning.teacher.policy import TeacherConfig, add_teacher_arguments

    teacher = TeacherConfig()
    parser = argparse.ArgumentParser()
    add_teacher_arguments(parser, candidate_top_k=teacher.candidate_top_k)
    teacher_args = parser.parse_args([])
    assert teacher_args.beam_depth == teacher.beam_depth
    assert teacher_args.beam_width == teacher.beam_width
    assert teacher_args.beam_branching == teacher.beam_branching
    assert teacher_args.beam_discount == teacher.beam_discount

    depth = DepthSafetyConfig()
    depth_args = build_depth_parser().parse_args([])
    label_args = build_label_parser().parse_args([])
    assert label_args.num_workers == 20
    assert depth_args.num_workers == 20
    assert depth_args.collision_radius_m == depth.collision_radius_m
    assert depth_args.depth_slack_m == depth.depth_slack_m
    assert depth_args.path_sample_stride == depth.path_sample_stride
    assert depth_args.patch_radius_px == depth.patch_radius_px
    assert depth_args.max_patch_radius_px == depth.max_patch_radius_px

    bc = BCTrainingConfig()
    bc_args = build_argument_parser().parse_args(
        [
            "--index",
            "index.csv",
            "--labels",
            "labels.npz",
            "--out-dir",
            "out",
            "--deployment-safety-mask",
            "height",
            "--deployment-execution-mode",
            "continuous",
        ]
    )
    assert bc_args.epochs == bc.epochs
    assert bc_args.batch_size == bc.batch_size
    assert bc_args.lr == bc.lr
    assert bc_args.weight_decay == bc.weight_decay


def test_experiment_values_have_one_yaml_owner():
    from planning.common.config import load_pre_bc_config

    config = load_pre_bc_config()
    assert config["mission"]["candidate_count"] == 2_000_000
    assert config["mission"]["seed"] == 2026
    assert config["mission"]["required_passing"] == 100_000
    assert config["label"]["workers"] == 20
    assert config["depth_mask"]["workers"] == 20
    assert config["collection"]["workers"] == 20
    assert config["collection"]["target_accepted"] == 60_000


def test_max_steps_remains_the_task_owner():
    from planning.mission.spec import DEFAULT_MAX_PRIMITIVE_STEPS

    assert DEFAULT_MAX_PRIMITIVE_STEPS == 45


def test_generic_helper_migrations_leave_no_target_private_duplicate():
    root = Path(__file__).resolve().parents[1]
    source_checks = {
        "python/planning/bc/trainer.py": ("def _manifest_sha256",),
        "python/planning/data/bc_mmap.py": ("def _sha256(",),
        "python/planning/runtime/identity.py": ("def _sha256(",),
        "python/planning/diagnostics/reliable_single_worker.py": (
            "def _sha256_file",
        ),
    }
    for relative, forbidden in source_checks.items():
        text = (root / relative).read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in text
