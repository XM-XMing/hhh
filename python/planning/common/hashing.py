"""Shared hashing primitives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(bytes(payload)).hexdigest()


def canonical_json_bytes(payload: Any, *, ensure_ascii: bool = True) -> bytes:
    """Return the compact, sorted JSON bytes used by generic contracts.

    ``ensure_ascii`` is explicit because the collection contract historically
    hashes UTF-8 text while several older contract identities hash the JSON
    encoder's ASCII-escaped representation.  The helper centralizes the
    serializer without conflating those byte contracts.
    """

    return json.dumps(
        payload,
        ensure_ascii=bool(ensure_ascii),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(payload: Any, *, ensure_ascii: bool = True) -> str:
    """Hash generic canonical JSON while retaining its explicit byte mode."""

    return bytes_sha256(canonical_json_bytes(payload, ensure_ascii=ensure_ascii))


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(block)
    return digest.hexdigest()
