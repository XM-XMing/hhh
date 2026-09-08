"""Primitive-neighbor ownership seam without a neighbor algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple


PRIMITIVE_NEIGHBOR_CONTRACT_ID = "motion_primitive_neighbor_index_v1"


@dataclass(frozen=True)
class PrimitiveNeighborIndex:
    """MPL-bound neighbor artifact; empty neighbors are valid until Phase 0."""

    mpl_contract_sha256: str
    action_count: int
    neighbors: Mapping[int, Tuple[int, ...]]


def empty_neighbor_index(*, mpl_contract_sha256: str, action_count: int = 105) -> PrimitiveNeighborIndex:
    if not str(mpl_contract_sha256):
        raise ValueError("MPL contract SHA is required for primitive neighbors")
    if int(action_count) <= 0:
        raise ValueError("primitive neighbor action count must be positive")
    return PrimitiveNeighborIndex(
        mpl_contract_sha256=str(mpl_contract_sha256),
        action_count=int(action_count),
        neighbors={index: tuple() for index in range(int(action_count))},
    )


__all__ = [
    "PRIMITIVE_NEIGHBOR_CONTRACT_ID",
    "PrimitiveNeighborIndex",
    "empty_neighbor_index",
]

