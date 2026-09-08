"""Reproducible Python/NumPy/PyTorch random-state setup."""

from __future__ import annotations

import random
import hashlib
import numpy as np

DEFAULT_SEED = 2026


def derive_seed(base_seed: int, namespace: str, index: int = 0) -> int:
    """Derive a stable, process-independent child seed.

    Python's built-in ``hash`` is intentionally randomized between processes;
    formal workers therefore derive IDs from explicit UTF-8 bytes instead.
    """

    if int(index) < 0:
        raise ValueError("seed derivation index must be non-negative")
    payload = "{}\0{}\0{}".format(int(base_seed), str(namespace), int(index)).encode(
        "utf-8"
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def seed_everything(seed: int, torch=None) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    if torch is not None:
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
