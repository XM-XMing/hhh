"""Small atomic torch checkpoint IO and deterministic mapping helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


def load_torch(path: Path, *, torch, map_location="cpu"):
    resolved = Path(path).expanduser().resolve()
    try:
        return torch.load(str(resolved), map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only
        return torch.load(str(resolved), map_location=map_location)


def save_torch_atomic(path: Path, payload: Mapping, *, torch) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(dict(payload), str(temporary))
    os.replace(str(temporary), str(destination))


def mapping_sha256(mapping: Mapping) -> str:
    """Hash JSON-compatible metadata without depending on dict insertion order."""
    if not isinstance(mapping, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    encoded = json.dumps(mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = ["load_torch", "mapping_sha256", "save_torch_atomic"]

