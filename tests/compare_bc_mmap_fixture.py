"""Run the same small BC mmap fixture through O and P and compare artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from test_bc_mmap_observation_provenance import _make_fixture


ARRAY_NAMES = (
    "depths",
    "continuous",
    "prev_actions",
    "height_masks",
    "local_depth_masks",
    "behavior_actions",
    "teacher_actions",
    "soft_targets",
    "history_starts",
    "episode_ids",
)


BUILDER = r'''
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
fixture = Path(sys.argv[2]).resolve()
out_dir = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(root / "python"))
if sys.argv[4] == "O":
    from planning.bc_mmap_dataset import build_bc_mmap_dataset
    from planning.dataset import TeacherLabelStore
    from planning.depth_action_masks import DepthActionMaskStore
else:
    from planning.data.bc_mmap import build_bc_mmap_dataset
    from planning.data.rollout import TeacherLabelStore
    from planning.safety.depth_mask import DepthActionMaskStore

index_path = fixture / "index.csv"
labels = TeacherLabelStore(fixture / "labels.npz")
masks = DepthActionMaskStore(fixture / "depth_masks.npz")
rows = [{
    "episode_id": "0",
    "execute_ok": "true",
    "dataset_npz": "episodes/episode_0.npz",
}]
manifest = build_bc_mmap_dataset(
    rows,
    index_path,
    labels,
    out_dir,
    masks,
    expected_observation_contract="reliable_exact_endpoint_snapshot",
)
print(json.dumps(manifest, sort_keys=True))
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(root: Path, fixture: Path, out_dir: Path, label: str) -> dict:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "python")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            BUILDER,
            str(root),
            str(fixture),
            str(out_dir),
            label,
        ],
        cwd=str(root),
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "{} builder failed ({}):\n{}\n{}".format(
                label, result.returncode, result.stdout, result.stderr
            )
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    original = Path("/home/xm/XM/xm_ws/src/planning").resolve()
    optimized = Path("/home/xm/XM/src").resolve()
    with tempfile.TemporaryDirectory(prefix="xm-m01-bc-mmap-") as temporary:
        root = Path(temporary)
        fixture = root / "fixture"
        _make_fixture(fixture)
        original_manifest = _run(original, fixture, root / "out_o", "O")
        optimized_manifest = _run(optimized, fixture, root / "out_p", "P")

        array_hashes = {}
        array_pass = True
        for name in ARRAY_NAMES:
            original_hash = _sha256(root / "out_o" / (name + ".npy"))
            optimized_hash = _sha256(root / "out_p" / (name + ".npy"))
            array_hashes[name] = {"O": original_hash, "P": optimized_hash}
            array_pass = array_pass and original_hash == optimized_hash

        manifest_pass = original_manifest == optimized_manifest
        order_pass = original_manifest["episodes"] == optimized_manifest["episodes"]
        provenance_pass = (
            optimized_manifest.get("observation_contract")
            == "reliable_exact_endpoint_snapshot"
            and optimized_manifest.get("observation_source")
            == "reliable_exact_endpoint_snapshot"
            and optimized_manifest.get("reliable_rows")
            == optimized_manifest.get("transition_count")
            and optimized_manifest.get("legacy_rows") == 0
        )
        print("BC_MMAP_ARRAY_PARITY={}".format("PASS" if array_pass else "FAIL"))
        print("BC_MMAP_MANIFEST_PARITY={}".format("PASS" if manifest_pass else "FAIL"))
        print(
            "BC_MMAP_MANIFEST_SHA256_O={}".format(
                _sha256(root / "out_o" / "manifest.json")
            )
        )
        print(
            "BC_MMAP_MANIFEST_SHA256_P={}".format(
                _sha256(root / "out_p" / "manifest.json")
            )
        )
        print("BC_MMAP_ROW_ORDER_PARITY={}".format("PASS" if order_pass else "FAIL"))
        print(
            "OBSERVATION_PROVENANCE_RESTORED={}".format(
                "YES" if provenance_pass else "NO"
            )
        )
        print("TRAIN_VAL_SPLIT_PARITY=NOT_APPLICABLE_AT_BC_MMAP_SEAM")
        print(json.dumps({"array_sha256": array_hashes}, sort_keys=True))
        return 0 if array_pass and manifest_pass and order_pass and provenance_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
