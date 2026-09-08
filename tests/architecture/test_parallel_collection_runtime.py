from pathlib import Path

import pytest

from planning.runtime.ports import build_collection_worker_specs
from planning.teacher.collection_config import ParallelCollectionConfig


def test_collection_port_derivation_matches_legacy_launcher(tmp_path):
    specs = build_collection_worker_specs(
        workers=2,
        out_dir=tmp_path,
        master_base=11321,
        command_base=10253,
        depth_base=12254,
        port_stride=20,
    )
    assert specs[0].ports == (11321, 10253, 10254, 12254)
    assert specs[1].ports == (11322, 10273, 10274, 12274)


def test_parallel_collection_extra_args_cannot_override_owned_options(tmp_path):
    mission_index = tmp_path / "missions.csv"
    mission_index.write_text("episode_id\n0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="managed option"):
        ParallelCollectionConfig.from_environment(
            mission_index=mission_index,
            out_dir=tmp_path / "rollouts",
            workers=12,
            environment={"COLLECTOR_EXTRA_ARGS": "--max-steps 99"},
            default_run_id="test-run",
        )


def test_parallel_collection_has_no_production_shell_wrapper():
    root = Path(__file__).resolve().parents[2]
    assert not (root / "scripts" / "collect_rollouts_parallel.sh").exists()
    entrypoint = root / "scripts" / "collect_rollouts_parallel.py"
    assert entrypoint.is_file()
    assert "planning.teacher.parallel_collection import main" in entrypoint.read_text(
        encoding="utf-8"
    )


def test_parallel_collection_script_calls_domain_module_directly():
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts" / "collect_rollouts_parallel.py").read_text(encoding="utf-8")
    assert "planning.teacher.parallel_collection import main" in text
    assert "planning.cli" not in text
