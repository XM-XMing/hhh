"""Small AWAC interaction boundary for policy and behavior provenance.

Collection and training exchange only the selected action, its safety mask,
and the source of the behavior policy.  Action proposals, trust gates, and
rollback state are intentionally outside this interface.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict


class BehaviorSource(IntEnum):
    """Source of the action stored in one AWAC replay transition."""

    BC_CALIBRATION = 0
    AWAC_ONLINE = 1

    # Numeric aliases retain readability for already-created pre-Phase-0
    # replay fixtures without creating a second serialized enum.
    BC_WARMUP = BC_CALIBRATION
    ACCEPTED_COLLECTION_POLICY = AWAC_ONLINE


def behavior_source_contract() -> Dict[str, int]:
    return {value.name.lower(): int(value) for value in BehaviorSource}


def validate_behavior_source(value: int) -> int:
    source = int(value)
    if source not in {int(item) for item in BehaviorSource}:
        raise ValueError("unknown AWAC behavior source: {}".format(source))
    return source
