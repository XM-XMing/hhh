"""Shared pytest configuration for script-compatibility tests."""

from __future__ import annotations
import copy
import os
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Contract programs retain their CLI-style ``main`` functions and are executed
# by test_script_contracts with dependency markers. Do not import them directly
# during collection, which would eagerly require ROS/Unity.
collect_ignore_glob = ["contracts/test_*.py"]


@pytest.fixture(scope="session", autouse=True)
def generated_motion_primitive_contract(tmp_path_factory):
    """Make a clean checkout self-sufficient without writing project data."""

    from planning.primitives.generator import (
        generate_library,
        save_library,
    )
    from planning.primitives.library import load_motion_primitive_config

    output = tmp_path_factory.mktemp("motion_primitives")
    config = copy.deepcopy(load_motion_primitive_config())
    config["paths"]["motion_primitives_npz"] = str(
        output / "motion_primitives_105.npz"
    )
    config["paths"]["metadata_json"] = str(
        output / "motion_primitives_105.json"
    )
    library = generate_library(config)
    npz_path, metadata_path = save_library(library, config)
    previous_npz = os.environ.get("PLANNING_MOTION_PRIMITIVES_NPZ")
    previous_json = os.environ.get("PLANNING_MOTION_PRIMITIVES_JSON")
    os.environ["PLANNING_MOTION_PRIMITIVES_NPZ"] = str(npz_path)
    os.environ["PLANNING_MOTION_PRIMITIVES_JSON"] = str(metadata_path)
    try:
        yield
    finally:
        if previous_npz is None:
            os.environ.pop("PLANNING_MOTION_PRIMITIVES_NPZ", None)
        else:
            os.environ["PLANNING_MOTION_PRIMITIVES_NPZ"] = previous_npz
        if previous_json is None:
            os.environ.pop("PLANNING_MOTION_PRIMITIVES_JSON", None)
        else:
            os.environ["PLANNING_MOTION_PRIMITIVES_JSON"] = previous_json


def pytest_collection_modifyitems(config, items):
    enabled = {
        "ros": os.environ.get("PLANNING_TEST_ROS") == "1",
        "unity": os.environ.get("PLANNING_TEST_UNITY") == "1",
        "cuda": os.environ.get("PLANNING_TEST_CUDA") == "1",
        "performance": os.environ.get("PLANNING_TEST_PERFORMANCE") == "1",
    }
    for item in items:
        for marker, is_enabled in enabled.items():
            if marker in item.keywords and not is_enabled:
                item.add_marker(pytest.mark.skip(
                    reason="set PLANNING_TEST_{}=1 to enable".format(marker.upper())
                ))
