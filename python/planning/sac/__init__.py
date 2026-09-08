"""BC-initialized online discrete SAC.

This package is deliberately separate from :mod:`planning.awac`.  It owns the
new online SAC policy, fresh twin critics, identity-bearing replay, and
checkpoint contract; the existing reliable-v4 runtime remains the only
environment lifecycle owner.
"""

from planning.sac.contract import (
    SAC_ALGORITHM_ID,
    SAC_CHECKPOINT_CONTRACT_ID,
    SAC_REPLAY_CONTRACT_ID,
)

__all__ = [
    "SAC_ALGORITHM_ID",
    "SAC_CHECKPOINT_CONTRACT_ID",
    "SAC_REPLAY_CONTRACT_ID",
]
