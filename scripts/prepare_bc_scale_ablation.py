#!/usr/bin/env python3
"""Prepare deterministic nested episode selections for the BC scale study.

The source rollout/label/mask artifacts remain immutable.  This command only
writes small episode-id selections and a provenance manifest; the BC mmap
builder can use those selections to materialize read-only-derived mmap views.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


SCALE_SPECS = {
    "20k": (16_000, 4_000),
    "40k": (32_000, 8_000),
    "60k": (48_000, 12_000),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_ids(path: Path) -> list[int]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = [int(float(row["episode_id"])) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError("source rollout index has duplicate episode ids")
    if values != sorted(values):
        raise ValueError("source rollout index is not in canonical episode order")
    return values


def _checkpoint_split(path: Path) -> tuple[list[int], list[int], dict]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise RuntimeError("PyTorch is required to read the BC split checkpoint") from error
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(str(path), map_location="cpu")
    train = sorted({int(value) for value in checkpoint.get("train_episodes", [])})
    val = sorted({int(value) for value in checkpoint.get("val_episodes", [])})
    if not train or not val or set(train).intersection(val):
        raise ValueError("checkpoint does not contain a disjoint train/val episode split")
    return train, val, checkpoint


def _write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--depth-action-masks", required=True)
    parser.add_argument("--bc-mmap", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    index = Path(args.index).expanduser().resolve()
    labels = Path(args.labels).expanduser().resolve()
    masks = Path(args.depth_action_masks).expanduser().resolve()
    bc_mmap = Path(args.bc_mmap).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    for path in (index, labels, masks, bc_mmap / "manifest.json", checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest_path = out_dir / "split_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError("selection manifest already exists: {}".format(manifest_path))

    source_ids = _episode_ids(index)
    train_ids, val_ids, checkpoint = _checkpoint_split(checkpoint_path)
    if set(train_ids).union(val_ids) != set(source_ids):
        raise ValueError("checkpoint split does not cover the source rollout index")
    if int(args.seed) != 2026:
        raise ValueError("this scale study is frozen to seed=2026")
    checkpoint_args = checkpoint.get("args", {})
    if int(checkpoint_args.get("seed", args.seed)) != int(args.seed):
        raise ValueError("checkpoint split seed does not match requested seed")

    source_sha = {
        "rollout_index": file_sha256(index),
        "teacher_labels": file_sha256(labels),
        "depth_action_masks": file_sha256(masks),
        "bc_mmap_manifest": file_sha256(bc_mmap / "manifest.json"),
        "split_checkpoint": file_sha256(checkpoint_path),
    }
    scales = {}
    for name, (train_count, val_count) in SCALE_SPECS.items():
        selected_train = train_ids[:train_count]
        selected_val = val_ids[:val_count]
        selected = sorted(selected_train + selected_val)
        if len(selected) != train_count + val_count:
            raise ValueError("{} selection has unexpected size".format(name))
        selection_dir = out_dir / "selections"
        ids_path = selection_dir / ("episode_ids_{}.txt".format(name))
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.write_text(
            "".join("{}\n".format(value) for value in selected), encoding="utf-8"
        )
        scales[name] = {
            "total_episodes": len(selected),
            "train_episodes": len(selected_train),
            "val_episodes": len(selected_val),
            "train_episode_ids": selected_train,
            "val_episode_ids": selected_val,
            "episode_ids": selected,
            "episode_ids_file": str(ids_path),
            "episode_ids_file_sha256": file_sha256(ids_path),
            "nested_in": {"20k": [], "40k": [], "60k": []},
        }
    scales["20k"]["nested_in"] = ["40k", "60k"]
    scales["40k"]["nested_in"] = ["60k"]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "study": "BC_DATA_SCALE_20K_40K_60K_CLOSED_LOOP_EVAL_V1",
            "selection_seed": int(args.seed),
            "selection_basis": "sorted checkpoint 60K train/val episode IDs",
            "source_index": str(index),
            "source_labels": str(labels),
            "source_depth_action_masks": str(masks),
            "source_bc_mmap": str(bc_mmap),
            "source_checkpoint": str(checkpoint_path),
            "source_sha256": source_sha,
            "source_episode_count": len(source_ids),
            "source_train_episode_count": len(train_ids),
            "source_val_episode_count": len(val_ids),
            "scales": scales,
            "checkpoint_rule": "best_soft",
            "no_source_artifact_copies": True,
        },
    )
    print("BC_SCALE_SELECTION_MANIFEST={}".format(manifest_path))
    for name, spec in SCALE_SPECS.items():
        train_count, val_count = spec
        print(
            "BC_SCALE_{}=PASS train={} val={} total={}".format(
                name.upper(), train_count, val_count, train_count + val_count
            )
        )
    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
