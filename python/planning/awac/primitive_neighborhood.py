"""Geometry-only motion-primitive neighborhood artifact and query owner."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


PRIMITIVE_NEIGHBORHOOD_SCHEMA_ID = "awac_primitive_neighborhood_v1"
PRIMITIVE_NEIGHBORHOOD_GENERATOR_VERSION = "geometry_descriptor_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PrimitiveNeighborhood:
    """Immutable distance matrix plus normalized geometry descriptor."""

    distance_matrix: np.ndarray
    descriptor: np.ndarray
    descriptor_center: np.ndarray
    descriptor_scale: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.distance_matrix, dtype=np.float64)
        descriptor = np.asarray(self.descriptor, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("distance matrix must be square")
        if descriptor.ndim != 2 or descriptor.shape[0] != matrix.shape[0]:
            raise ValueError("descriptor/action count mismatch")
        if not np.isfinite(matrix).all() or not np.isfinite(descriptor).all():
            raise ValueError("primitive neighborhood contains non-finite values")
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-7):
            raise ValueError("primitive distance matrix must be symmetric")
        if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1.0e-7):
            raise ValueError("primitive distance matrix diagonal must be zero")
        object.__setattr__(self, "distance_matrix", matrix)
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "descriptor_center", np.asarray(self.descriptor_center, dtype=np.float64))
        object.__setattr__(self, "descriptor_scale", np.asarray(self.descriptor_scale, dtype=np.float64))

    @property
    def action_count(self) -> int:
        return int(self.distance_matrix.shape[0])

    def top_k(self, anchor_primitive: int, k: int, valid_mask: Optional[Any] = None) -> np.ndarray:
        anchor = int(anchor_primitive)
        if anchor < 0 or anchor >= self.action_count:
            raise ValueError("anchor primitive is outside the neighborhood")
        if int(k) <= 0:
            return np.empty((0,), dtype=np.int64)
        candidates = np.arange(self.action_count, dtype=np.int64)
        candidates = candidates[candidates != anchor]
        if valid_mask is not None:
            valid = np.asarray(valid_mask, dtype=bool)
            if valid.shape != (self.action_count,):
                raise ValueError("valid_mask has incorrect action count")
            candidates = candidates[valid[candidates]]
        order = np.argsort(self.distance_matrix[anchor, candidates], kind="stable")
        return candidates[order[: int(k)]]

    def radius(self, anchor_primitive: int, radius: float, valid_mask: Optional[Any] = None) -> np.ndarray:
        if not np.isfinite(float(radius)) or float(radius) < 0.0:
            raise ValueError("radius must be finite and non-negative")
        anchor = int(anchor_primitive)
        if anchor < 0 or anchor >= self.action_count:
            raise ValueError("anchor primitive is outside the neighborhood")
        result = np.flatnonzero(self.distance_matrix[anchor] <= float(radius)).astype(np.int64)
        if valid_mask is not None:
            valid = np.asarray(valid_mask, dtype=bool)
            if valid.shape != (self.action_count,):
                raise ValueError("valid_mask has incorrect action count")
            result = result[valid[result]]
        return result


def _descriptor_from_trajectory(trajectories: np.ndarray) -> np.ndarray:
    if trajectories.ndim != 3 or trajectories.shape[2] != 3 or trajectories.shape[1] < 2:
        raise ValueError("trajectories must have shape [N,T,3]")
    endpoint = trajectories[:, -1, :]
    endpoint_delta = trajectories[:, -1, :] - trajectories[:, -2, :]
    direction = endpoint_delta.copy()
    horizontal = np.linalg.norm(direction[:, :2], axis=1, keepdims=True)
    direction[:, :2] = direction[:, :2] / np.maximum(horizontal, 1.0e-12)
    path_length = np.linalg.norm(np.diff(trajectories, axis=1), axis=2).sum(axis=1, keepdims=True)
    return np.concatenate(
        (endpoint, endpoint_delta, direction[:, :2], path_length), axis=1
    ).astype(np.float64, copy=False)


def build_primitive_neighborhood(
    trajectories: Any, *, metadata: Optional[Mapping[str, Any]] = None
) -> PrimitiveNeighborhood:
    """Build a deterministic descriptor from sampled primitive geometry only."""

    raw = np.asarray(trajectories, dtype=np.float64)
    descriptor_raw = _descriptor_from_trajectory(raw)
    center = np.median(descriptor_raw, axis=0)
    q75, q25 = np.percentile(descriptor_raw, [75.0, 25.0], axis=0)
    scale = q75 - q25
    scale = np.where(
        scale > 0.0,
        scale,
        np.max(np.abs(descriptor_raw - center), axis=0),
    )
    scale = np.where(scale > 0.0, scale, 1.0)
    normalized = (descriptor_raw - center) / scale
    differences = normalized[:, None, :] - normalized[None, :, :]
    matrix = np.linalg.norm(differences, axis=2)
    matrix = (matrix + matrix.T) * 0.5
    np.fill_diagonal(matrix, 0.0)
    base_metadata = {
        "schema_id": PRIMITIVE_NEIGHBORHOOD_SCHEMA_ID,
        "generator_version": PRIMITIVE_NEIGHBORHOOD_GENERATOR_VERSION,
        "descriptor_schema": [
            "endpoint_displacement_xyz",
            "endpoint_delta_xyz",
            "endpoint_direction_xy",
            "path_length",
        ],
        "normalization": "per-feature median center and IQR scale; zero-IQR fallback max absolute deviation then 1",
        "distance_metric": "weighted_euclidean_normalized_descriptor",
        "feature_weights": [1.0] * int(normalized.shape[1]),
        "action_count": int(normalized.shape[0]),
    }
    if metadata:
        base_metadata.update(dict(metadata))
    return PrimitiveNeighborhood(matrix, normalized, center, scale, base_metadata)


def build_primitive_neighborhood_from_mpl(
    path: Path, *, metadata: Optional[Mapping[str, Any]] = None
) -> PrimitiveNeighborhood:
    """Build from the canonical MPL ``pos_ref`` trajectory field."""

    with np.load(Path(path), allow_pickle=False) as data:
        if "pos_ref" not in data.files:
            raise ValueError("canonical MPL is missing pos_ref")
        trajectories = np.asarray(data["pos_ref"], dtype=np.float64)
    return build_primitive_neighborhood(trajectories, metadata=metadata)


def write_primitive_neighborhood_artifact(
    neighborhood: PrimitiveNeighborhood,
    *,
    npz_path: Path,
    metadata_path: Path,
    source_path: str,
    source_sha256: str,
    mpl_contract_sha256: str,
) -> str:
    """Write the immutable matrix and metadata, returning the matrix-file SHA."""

    target = Path(npz_path).expanduser().resolve()
    manifest = Path(metadata_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        distance_matrix=np.asarray(neighborhood.distance_matrix, dtype=np.float32),
        descriptor=np.asarray(neighborhood.descriptor, dtype=np.float32),
        descriptor_center=np.asarray(neighborhood.descriptor_center, dtype=np.float32),
        descriptor_scale=np.asarray(neighborhood.descriptor_scale, dtype=np.float32),
    )
    artifact_sha = _sha256(target)
    payload = {
        **dict(neighborhood.metadata),
        "schema_id": PRIMITIVE_NEIGHBORHOOD_SCHEMA_ID,
        "source_path": str(source_path),
        "source_sha256": str(source_sha256),
        "mpl_contract_sha256": str(mpl_contract_sha256),
        "artifact_path": str(target),
        "artifact_sha256": artifact_sha,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return artifact_sha


def load_primitive_neighborhood(
    npz_path: Path, *, metadata_path: Optional[Path] = None
) -> PrimitiveNeighborhood:
    target = Path(npz_path).expanduser().resolve()
    with np.load(target, allow_pickle=False) as data:
        required = {
            "distance_matrix",
            "descriptor",
            "descriptor_center",
            "descriptor_scale",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                "primitive neighborhood missing fields: {}".format(", ".join(missing))
            )
        arrays = {name: np.asarray(data[name]) for name in required}
    metadata = {}
    if metadata_path is not None:
        metadata = json.loads(
            Path(metadata_path)
            .expanduser()
            .resolve()
            .read_text(encoding="utf-8")
        )
        expected = metadata.get("artifact_sha256")
        if expected and expected != _sha256(target):
            raise ValueError("primitive neighborhood artifact SHA mismatch")
    return PrimitiveNeighborhood(
        arrays["distance_matrix"],
        arrays["descriptor"],
        arrays["descriptor_center"],
        arrays["descriptor_scale"],
        metadata,
    )


__all__ = [
    "PRIMITIVE_NEIGHBORHOOD_GENERATOR_VERSION",
    "PRIMITIVE_NEIGHBORHOOD_SCHEMA_ID",
    "PrimitiveNeighborhood",
    "build_primitive_neighborhood",
    "build_primitive_neighborhood_from_mpl",
    "load_primitive_neighborhood",
    "write_primitive_neighborhood_artifact",
]
