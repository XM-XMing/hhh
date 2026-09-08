"""Fail-closed integrity audit for a formal AWAC replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from planning.awac.interaction import behavior_source_contract
from planning.awac.replay import AWACReplayBuffer, _FIELD_SPECS


def audit_awac_replay(replay_dir: Path) -> Dict:
    """Validate replay metadata and active rows without changing the artifact."""
    root = Path(replay_dir).expanduser().resolve()
    replay = AWACReplayBuffer.open(root, read_only=True)
    try:
        expected_npy = {"{}.npy".format(name) for name in _FIELD_SPECS}
        actual_npy = {path.name for path in root.glob("*.npy")}
        unexpected = sorted(actual_npy.difference(expected_npy))
        if unexpected:
            raise ValueError("AWAC replay has unsupported array files: {}".format(unexpected))

        size = int(replay.size)
        arrays = {name: array[:size] for name, array in replay.arrays.items()}
        if not np.isfinite(np.asarray(arrays["vector"], dtype=np.float32)).all():
            raise ValueError("AWAC replay vector contains non-finite values")
        if not np.isfinite(np.asarray(arrays["reward"], dtype=np.float32)).all():
            raise ValueError("AWAC replay reward contains non-finite values")
        action = np.asarray(arrays["action"], dtype=np.int64)
        masks = np.asarray(arrays["action_mask"], dtype=np.bool_)
        next_masks = np.asarray(arrays["next_action_mask"], dtype=np.bool_)
        done = np.asarray(arrays["done"], dtype=np.int64)
        if not np.all((action >= 0) & (action < int(replay.action_dim))):
            raise ValueError("AWAC replay action is outside action space")
        if not np.all(masks[np.arange(size), action]):
            raise ValueError("AWAC replay action is outside stored mask")
        if not np.all(masks.any(axis=1)):
            raise ValueError("AWAC replay has an empty current action mask")
        if np.any((done == 0) & (~next_masks.any(axis=1))):
            raise ValueError("AWAC replay has an empty non-terminal next mask")
        if not np.all((done == 0) | (done == 1)):
            raise ValueError("AWAC replay done field is not binary")

        behavior = np.asarray(arrays["behavior_source"], dtype=np.int64)
        allowed = set(int(value) for value in behavior_source_contract().values())
        if not set(int(value) for value in np.unique(behavior)).issubset(allowed):
            raise ValueError("AWAC replay contains an unknown behavior source")
        unique, counts = np.unique(behavior, return_counts=True)
        return {
            "status": "PASS",
            "replay_dir": str(root),
            "contract_id": replay.metadata["contract_id"],
            "replay_contract_sha256": replay.metadata["replay_contract_sha256"],
            "row_count": size,
            "capacity": int(replay.capacity),
            "depth_shape": list(replay.depth_shape),
            "vector_dim": int(replay.vector_dim),
            "action_dim": int(replay.action_dim),
            "dtype": {name: str(array.dtype) for name, array in replay.arrays.items()},
            "behavior_source_contract": behavior_source_contract(),
            "behavior_source_counts": {
                str(int(source)): int(count)
                for source, count in zip(unique.tolist(), counts.tolist())
            },
            "terminal_count": int(done.sum()),
            "nonterminal_count": int(size - done.sum()),
            "legacy_replay_transition_count": int(
                replay.metadata["legacy_replay_transition_count"]
            ),
            "reliable_v4_transition_count": int(
                replay.metadata["reliable_v4_transition_count"]
            ),
            "privileged_field_count": int(
                replay.metadata.get("privileged_field_count", 0)
            ),
            "behavior_source_phase": str(
                replay.metadata.get("behavior_source_phase", "unspecified")
            ),
        }
    finally:
        replay.close()


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = audit_awac_replay(args.replay_dir)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


__all__ = ["audit_awac_replay", "main"]
