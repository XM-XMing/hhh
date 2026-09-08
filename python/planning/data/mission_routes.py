"""Canonical streaming immutable storage for one global route per mission.

The route store is deliberately a small deep module: generation knows only
``append``/``commit`` and later stages know only ``open``/``route``.  Points
and offsets are raw, fixed-width binary arrays so a completed store can be
opened as read-only ``numpy.memmap`` views without retaining all routes in
Python memory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from planning.common import file_sha256, write_json_atomic


MISSION_ROUTE_STORE_CONTRACT_ID = "mission_route_store_v1"
MISSION_ROUTE_STORE_SCHEMA_VERSION = 1
_POINTS_SUFFIX = ".points.bin"
_OFFSETS_SUFFIX = ".offsets.bin"
_META_SUFFIX = ".meta.json"


def _normalise_prefix(value: Path) -> Path:
    path = Path(value).expanduser().resolve()
    text = str(path)
    for suffix in (_POINTS_SUFFIX, _OFFSETS_SUFFIX, _META_SUFFIX):
        if text.endswith(suffix):
            return Path(text[: -len(suffix)])
    return path


def _paths(prefix: Path) -> tuple[Path, Path, Path]:
    prefix = _normalise_prefix(prefix)
    return (
        Path(str(prefix) + _POINTS_SUFFIX),
        Path(str(prefix) + _OFFSETS_SUFFIX),
        Path(str(prefix) + _META_SUFFIX),
    )


def resolve_route_store_prefix(
    rows: Sequence[Mapping[str, Any]],
    index_path: Path,
    explicit: Optional[Path] = None,
) -> Path:
    """Resolve one route-store prefix from an artifact or its mission rows.

    Formal downstream artifacts carry ``mission_route_store`` relative to the
    index containing the row.  The default remains the generator's canonical
    sibling prefix so small callers do not need to repeat the path.
    """

    if explicit is not None and str(explicit).strip():
        return _normalise_prefix(Path(explicit))
    candidates = {
        str(row.get("mission_route_store", "")).strip()
        for row in rows
        if str(row.get("mission_route_store", "")).strip()
    }
    if len(candidates) > 1:
        raise ValueError("mission rows reference multiple route stores")
    if candidates:
        raw = Path(next(iter(candidates))).expanduser()
        if not raw.is_absolute():
            raw = Path(index_path).expanduser().resolve().parent / raw
        return _normalise_prefix(raw)
    return _normalise_prefix(Path(index_path).expanduser().resolve().parent / "mission_routes")


def _require_sha(value: Any, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or text.lower() != text:
        raise ValueError("{} must be a lowercase SHA-256 digest".format(name))
    try:
        bytes.fromhex(text)
    except ValueError as error:
        raise ValueError("{} must be a lowercase SHA-256 digest".format(name)) from error
    return text


def _validate_route(route: Any) -> np.ndarray:
    value = np.asarray(route)
    if value.ndim != 2 or value.shape[1:] != (3,) or value.shape[0] < 2:
        raise ValueError("mission route must be a finite Nx3 array with N >= 2")
    result = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("mission route must contain only finite values")
    return result


class MissionRouteStoreWriter:
    """Append route points and publish one immutable route-store artifact."""

    def __init__(
        self,
        prefix: Path,
        *,
        candidate_index_sha256: str = "",
        global_route_contract_id: str,
        route_resolution_m: float,
        tracking_margin_m: float,
        collision_map_identity: Dict[str, Any],
        source_config_identity: Dict[str, Any],
        expected_mission_count: Optional[int] = None,
        overwrite: bool = False,
    ) -> None:
        self.prefix = _normalise_prefix(Path(prefix))
        self.points_path, self.offsets_path, self.meta_path = _paths(self.prefix)
        self.points_tmp = Path(str(self.points_path) + ".tmp")
        self.offsets_tmp = Path(str(self.offsets_path) + ".tmp")
        self.meta_tmp = Path(str(self.meta_path) + ".tmp")
        final_paths = (self.points_path, self.offsets_path, self.meta_path)
        if not overwrite and any(path.exists() for path in final_paths):
            raise FileExistsError(
                "route-store artifact exists; pass overwrite=True: {}".format(
                    self.prefix
                )
            )
        if any(path.exists() for path in (self.points_tmp, self.offsets_tmp, self.meta_tmp)):
            raise FileExistsError(
                "route-store temporary artifact exists: {}".format(self.prefix)
            )
        if int(expected_mission_count or 0) < 0:
            raise ValueError("expected_mission_count must be non-negative")
        if not str(global_route_contract_id):
            raise ValueError("global_route_contract_id is required")
        if not np.isfinite(float(route_resolution_m)) or float(route_resolution_m) <= 0.0:
            raise ValueError("route_resolution_m must be finite and positive")
        if not np.isfinite(float(tracking_margin_m)) or float(tracking_margin_m) < 0.0:
            raise ValueError("tracking_margin_m must be finite and non-negative")
        if not isinstance(collision_map_identity, dict):
            raise TypeError("collision_map_identity must be a mapping")
        if not isinstance(source_config_identity, dict):
            raise TypeError("source_config_identity must be a mapping")

        self._candidate_index_sha256 = str(candidate_index_sha256 or "")
        if self._candidate_index_sha256:
            _require_sha(self._candidate_index_sha256, "candidate_index_sha256")
        self._global_route_contract_id = str(global_route_contract_id)
        self._route_resolution_m = float(route_resolution_m)
        self._tracking_margin_m = float(tracking_margin_m)
        self._collision_map_identity = dict(collision_map_identity)
        self._source_config_identity = dict(source_config_identity)
        self._expected_mission_count = (
            None if expected_mission_count is None else int(expected_mission_count)
        )
        self._mission_count = 0
        self._total_route_points = 0
        self._closed = False
        self._committed = False
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        self._points_handle = self.points_tmp.open("wb")
        self._offsets_handle = self.offsets_tmp.open("wb")
        np.asarray([0], dtype="<i8").tofile(self._offsets_handle)

    @classmethod
    def resume(
        cls,
        prefix: Path,
        *,
        global_route_contract_id: str,
        route_resolution_m: float,
        tracking_margin_m: float,
        collision_map_identity: Dict[str, Any],
        source_config_identity: Dict[str, Any],
    ) -> "MissionRouteStoreWriter":
        """Reopen an interrupted append-only route store without replanning.

        A resume is accepted only for the uncommitted ``.tmp`` pair.  A
        published store is immutable and therefore cannot be appended to by a
        preparation retry.  The offsets tail is the durable route count; the
        preparation engine separately checks it against its routed journal.
        """

        normalised = _normalise_prefix(Path(prefix))
        points_path, offsets_path, meta_path = _paths(normalised)
        points_tmp = Path(str(points_path) + ".tmp")
        offsets_tmp = Path(str(offsets_path) + ".tmp")
        meta_tmp = Path(str(meta_path) + ".tmp")
        if any(path.exists() for path in (points_path, offsets_path, meta_path)):
            raise FileExistsError(
                "published route store is immutable; cannot resume: {}".format(normalised)
            )
        if not points_tmp.is_file() or not offsets_tmp.is_file():
            raise FileNotFoundError(
                "route-store resume requires both temporary files: {}".format(normalised)
            )
        if offsets_tmp.stat().st_size < 8 or offsets_tmp.stat().st_size % 8:
            raise ValueError("route-store temporary offsets are truncated")
        mission_count = offsets_tmp.stat().st_size // 8 - 1
        offsets = np.memmap(
            str(offsets_tmp), dtype="<i8", mode="r", shape=(mission_count + 1,)
        )
        if int(offsets[0]) != 0 or np.any(np.diff(offsets) < 2):
            del offsets
            raise ValueError("route-store temporary offsets are invalid")
        total_route_points = int(offsets[-1])
        del offsets
        expected_points_size = total_route_points * 3 * np.dtype("<f4").itemsize
        if points_tmp.stat().st_size != expected_points_size:
            raise ValueError("route-store temporary points size mismatch")
        if not str(global_route_contract_id):
            raise ValueError("global_route_contract_id is required")
        if not isinstance(collision_map_identity, dict) or not isinstance(
            source_config_identity, dict
        ):
            raise TypeError("route-store resume identities must be mappings")
        result = cls.__new__(cls)
        result.prefix = normalised
        result.points_path, result.offsets_path, result.meta_path = (
            points_path,
            offsets_path,
            meta_path,
        )
        result.points_tmp = points_tmp
        result.offsets_tmp = offsets_tmp
        result.meta_tmp = meta_tmp
        result._candidate_index_sha256 = ""
        result._global_route_contract_id = str(global_route_contract_id)
        result._route_resolution_m = float(route_resolution_m)
        result._tracking_margin_m = float(tracking_margin_m)
        result._collision_map_identity = dict(collision_map_identity)
        result._source_config_identity = dict(source_config_identity)
        result._expected_mission_count = None
        result._mission_count = int(mission_count)
        result._total_route_points = int(total_route_points)
        result._closed = False
        result._committed = False
        result.prefix.parent.mkdir(parents=True, exist_ok=True)
        result._points_handle = result.points_tmp.open("ab")
        result._offsets_handle = result.offsets_tmp.open("ab")
        return result

    @property
    def mission_count(self) -> int:
        return int(self._mission_count)

    @property
    def total_route_points(self) -> int:
        return int(self._total_route_points)

    def set_candidate_index_sha256(self, value: str) -> None:
        self._ensure_open()
        self._candidate_index_sha256 = _require_sha(value, "candidate_index_sha256")

    def append(self, route: Any, *, goal: Optional[Sequence[float]] = None) -> int:
        self._ensure_open()
        route_array = _validate_route(route)
        if goal is not None:
            goal_array = np.asarray(goal, dtype=np.float32).reshape(3)
            if not np.allclose(route_array[-1], goal_array, atol=1.0e-3):
                raise ValueError("mission route endpoint does not match final goal")
        route_index = int(self._mission_count)
        route_array.tofile(self._points_handle)
        self._total_route_points += int(route_array.shape[0])
        np.asarray([self._total_route_points], dtype="<i8").tofile(self._offsets_handle)
        self._mission_count += 1
        return route_index

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("route-store writer is closed")

    def _metadata(self, points_sha256: str, offsets_sha256: str) -> Dict[str, Any]:
        return {
            "contract_id": MISSION_ROUTE_STORE_CONTRACT_ID,
            "schema_version": MISSION_ROUTE_STORE_SCHEMA_VERSION,
            "mission_count": int(self._mission_count),
            "total_route_points": int(self._total_route_points),
            "points_dtype": "float32",
            "points_shape": [int(self._total_route_points), 3],
            "offsets_dtype": "int64",
            "offsets_shape": [int(self._mission_count) + 1],
            "points_file": self.points_path.name,
            "offsets_file": self.offsets_path.name,
            "candidate_index_sha256": self._candidate_index_sha256,
            "global_route_contract_id": self._global_route_contract_id,
            "route_resolution_m": float(self._route_resolution_m),
            "tracking_margin_m": float(self._tracking_margin_m),
            "collision_map_identity": dict(self._collision_map_identity),
            "points_sha256": points_sha256,
            "offsets_sha256": offsets_sha256,
            "source_config_identity": dict(self._source_config_identity),
        }

    def commit(self) -> "MissionRouteStore":
        self._ensure_open()
        if self._expected_mission_count is not None and self._mission_count != self._expected_mission_count:
            raise ValueError(
                "route-store mission count {} != expected {}".format(
                    self._mission_count, self._expected_mission_count
                )
            )
        if not self._candidate_index_sha256:
            raise ValueError("candidate_index_sha256 is required before commit")
        _require_sha(self._candidate_index_sha256, "candidate_index_sha256")
        self._points_handle.flush()
        os.fsync(self._points_handle.fileno())
        self._offsets_handle.flush()
        os.fsync(self._offsets_handle.fileno())
        self._points_handle.close()
        self._offsets_handle.close()
        os.replace(str(self.points_tmp), str(self.points_path))
        os.replace(str(self.offsets_tmp), str(self.offsets_path))
        metadata = self._metadata(
            file_sha256(self.points_path), file_sha256(self.offsets_path)
        )
        write_json_atomic(self.meta_path, metadata, trailing_newline=True)
        self._closed = True
        self._committed = True
        return MissionRouteStore.open(self.prefix, validate=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._points_handle.close()
        finally:
            self._offsets_handle.close()
        for path in (self.points_tmp, self.offsets_tmp, self.meta_tmp):
            path.unlink(missing_ok=True)

    def __enter__(self) -> "MissionRouteStoreWriter":
        self._ensure_open()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


class MissionRouteStore:
    """Read-only, mmap-backed view of a committed mission route artifact."""

    def __init__(self, prefix: Path, metadata: Dict[str, Any]) -> None:
        self.prefix = _normalise_prefix(Path(prefix))
        self.points_path, self.offsets_path, self.meta_path = _paths(self.prefix)
        self.metadata = dict(metadata)
        self._closed = False
        self._points = None
        self._offsets = None
        total_points = int(self.metadata["total_route_points"])
        mission_count = int(self.metadata["mission_count"])
        if total_points > 0:
            points = np.memmap(
                str(self.points_path), dtype="<f4", mode="r", shape=(total_points, 3)
            )
        else:
            points = np.empty((0, 3), dtype=np.float32)
            points.setflags(write=False)
        offsets = np.memmap(
            str(self.offsets_path), dtype="<i8", mode="r", shape=(mission_count + 1,)
        )
        points.setflags(write=False)
        offsets.setflags(write=False)
        self._points = points
        self._offsets = offsets

    @classmethod
    def open(cls, prefix: Path, *, validate: bool = False) -> "MissionRouteStore":
        normalised = _normalise_prefix(Path(prefix))
        points_path, offsets_path, meta_path = _paths(normalised)
        if not meta_path.is_file() or not points_path.is_file() or not offsets_path.is_file():
            raise FileNotFoundError("incomplete mission route store: {}".format(normalised))
        import json

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        required = (
            "contract_id",
            "schema_version",
            "mission_count",
            "total_route_points",
            "points_dtype",
            "offsets_dtype",
            "candidate_index_sha256",
            "global_route_contract_id",
            "route_resolution_m",
            "tracking_margin_m",
            "collision_map_identity",
            "points_sha256",
            "offsets_sha256",
            "source_config_identity",
        )
        missing = [name for name in required if name not in metadata]
        if missing:
            raise ValueError("route store metadata missing: {}".format(",".join(missing)))
        if metadata["contract_id"] != MISSION_ROUTE_STORE_CONTRACT_ID:
            raise ValueError("route store contract mismatch")
        if int(metadata["schema_version"]) != MISSION_ROUTE_STORE_SCHEMA_VERSION:
            raise ValueError("route store schema version mismatch")
        if metadata["points_dtype"] != "float32" or metadata["offsets_dtype"] != "int64":
            raise ValueError("route store dtype contract mismatch")
        if int(metadata["mission_count"]) < 0 or int(metadata["total_route_points"]) < 0:
            raise ValueError("route store counts must be non-negative")
        _require_sha(metadata["candidate_index_sha256"], "candidate_index_sha256")
        _require_sha(metadata["points_sha256"], "points_sha256")
        _require_sha(metadata["offsets_sha256"], "offsets_sha256")
        expected_points_size = int(metadata["total_route_points"]) * 3 * np.dtype("<f4").itemsize
        expected_offsets_size = (int(metadata["mission_count"]) + 1) * np.dtype("<i8").itemsize
        if points_path.stat().st_size != expected_points_size:
            raise ValueError("route store points byte size mismatch")
        if offsets_path.stat().st_size != expected_offsets_size:
            raise ValueError("route store offsets byte size mismatch")
        store = cls(normalised, metadata)
        if validate:
            store.validate()
        return store

    @property
    def points(self) -> np.ndarray:
        self._ensure_open()
        return self._points

    @property
    def offsets(self) -> np.ndarray:
        self._ensure_open()
        return self._offsets

    @property
    def mission_count(self) -> int:
        return int(self.metadata["mission_count"])

    @property
    def total_route_points(self) -> int:
        return int(self.metadata["total_route_points"])

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("mission route store is closed")

    def route(self, index: int) -> np.ndarray:
        self._ensure_open()
        route_index = int(index)
        if route_index < 0 or route_index >= self.mission_count:
            raise IndexError("route index out of range: {}".format(route_index))
        start = int(self._offsets[route_index])
        end = int(self._offsets[route_index + 1])
        result = self._points[start:end]
        if result.shape[0] < 2:
            raise ValueError("route {} has fewer than two points".format(route_index))
        return result

    def validate(
        self,
        index: Optional[int] = None,
        goal: Optional[Sequence[float]] = None,
    ) -> Optional[np.ndarray]:
        self._ensure_open()
        if index is not None:
            route = self.route(index)
            if goal is not None:
                goal_array = np.asarray(goal, dtype=np.float32).reshape(3)
                if not np.allclose(route[-1], goal_array, atol=1.0e-3):
                    raise ValueError("mission route endpoint does not match final goal")
            return route
        if self._offsets.shape != (self.mission_count + 1,):
            raise ValueError("route store offsets shape mismatch")
        if int(self._offsets[0]) != 0 or int(self._offsets[-1]) != self.total_route_points:
            raise ValueError("route store offsets boundary mismatch")
        if np.any(np.diff(self._offsets) < 2):
            raise ValueError("route store contains a route with fewer than two points")
        if not np.isfinite(self._points).all():
            raise ValueError("route store contains non-finite points")
        if file_sha256(self.points_path) != str(self.metadata["points_sha256"]):
            raise ValueError("route store points SHA256 mismatch")
        if file_sha256(self.offsets_path) != str(self.metadata["offsets_sha256"]):
            raise ValueError("route store offsets SHA256 mismatch")
        return None

    def close(self) -> None:
        if self._closed:
            return
        points = self._points
        offsets = self._offsets
        self._points = None
        self._offsets = None
        self._closed = True
        if isinstance(points, np.memmap):
            del points
        if isinstance(offsets, np.memmap):
            del offsets

    def __enter__(self) -> "MissionRouteStore":
        self._ensure_open()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def validate_route_references(
    rows: Sequence[Dict[str, Any]],
    store: MissionRouteStore,
    *,
    candidate_index_sha256: Optional[str] = None,
    require_provenance: bool = False,
) -> None:
    """Fail closed when mission rows do not reference this exact route store."""

    expected_candidate_sha = _require_sha(
        candidate_index_sha256 or store.metadata["candidate_index_sha256"],
        "candidate_index_sha256",
    )
    if str(store.metadata["candidate_index_sha256"]) != expected_candidate_sha:
        raise ValueError("route store candidate index SHA256 mismatch")
    expected_contract = str(store.metadata["global_route_contract_id"])
    expected_store_identity = {
        "mission_route_store_contract_id": str(store.metadata["contract_id"]),
        "mission_route_store_schema_version": int(store.metadata["schema_version"]),
        "mission_route_store_candidate_index_sha256": expected_candidate_sha,
        "mission_route_store_points_sha256": str(store.metadata["points_sha256"]),
        "mission_route_store_offsets_sha256": str(store.metadata["offsets_sha256"]),
    }
    seen = set()
    for row_number, row in enumerate(rows):
        raw_index = row.get("global_route_index", "")
        try:
            route_index = int(float(raw_index))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "mission row {} has invalid global_route_index".format(row_number)
            ) from error
        if route_index in seen:
            raise ValueError("duplicate global_route_index: {}".format(route_index))
        seen.add(route_index)
        if str(row.get("global_route_contract_id", "")) != expected_contract:
            raise ValueError("mission row global route contract mismatch")
        if require_provenance:
            for field, expected in expected_store_identity.items():
                raw = row.get(field, "")
                try:
                    actual = int(float(raw)) if field.endswith("schema_version") else str(raw)
                except (TypeError, ValueError) as error:
                    raise ValueError("mission row has invalid {}".format(field)) from error
                if actual != expected:
                    raise ValueError("mission row {} mismatch".format(field))
        goal = [
            float(row["goal_x"]),
            float(row["goal_y"]),
            float(row["goal_z"]),
        ]
        store.validate(route_index, goal=goal)


def route_store_provenance(store: MissionRouteStore, *, artifact_path: Path) -> Dict[str, Any]:
    """Return the stable route-store identity propagated into downstream rows."""

    return {
        "global_route_contract_id": str(store.metadata["global_route_contract_id"]),
        "mission_route_store_contract_id": str(store.metadata["contract_id"]),
        "mission_route_store_schema_version": int(store.metadata["schema_version"]),
        "mission_route_store": os.path.relpath(
            str(store.meta_path), str(Path(artifact_path).expanduser().resolve().parent)
        ),
        "mission_route_store_candidate_index_sha256": str(
            store.metadata["candidate_index_sha256"]
        ),
        "mission_route_store_points_sha256": str(store.metadata["points_sha256"]),
        "mission_route_store_offsets_sha256": str(store.metadata["offsets_sha256"]),
        "mission_route_store_total_route_points": int(
            store.metadata["total_route_points"]
        ),
    }


__all__ = [
    "MISSION_ROUTE_STORE_CONTRACT_ID",
    "MISSION_ROUTE_STORE_SCHEMA_VERSION",
    "MissionRouteStore",
    "MissionRouteStoreWriter",
    "resolve_route_store_prefix",
    "route_store_provenance",
    "validate_route_references",
]
