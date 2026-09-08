"""Shared filesystem persistence helpers."""

from __future__ import annotations

import json
import os
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict]:
    """Read a UTF-8 CSV artifact through the shared I/O owner."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return []
    with resolved.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(
    path: Path,
    rows: Iterable[Dict],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    """Publish a CSV snapshot atomically; callers own the compaction boundary."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(target))


def _replace_bytes(path: Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(str(temporary), str(target))


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    _replace_bytes(path, bytes(payload))


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    _replace_bytes(path, str(text).encode(encoding))


def write_json_atomic(
    path: Path, payload: Any, *, trailing_newline: bool = False
) -> None:
    """Write sorted pretty JSON atomically with an explicit newline contract."""

    text = json.dumps(payload, indent=2, sort_keys=True)
    if trailing_newline:
        text += "\n"
    write_text_atomic(
        path,
        text,
        encoding="utf-8",
    )


def write_npz_atomic(
    path: Path, arrays: Dict[str, Any], *, compress: bool = False
) -> None:
    """Write a NumPy archive through the shared atomic replace seam."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        import numpy as np

        payload = {key: np.asarray(value) for key, value in arrays.items()}
        if compress:
            np.savez_compressed(handle, **payload)
        else:
            np.savez(handle, **payload)
    os.replace(str(temporary), str(target))
