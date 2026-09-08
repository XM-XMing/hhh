"""Small dependency-free latency and throughput metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

@dataclass
class LatencyTracker:
    values_s: List[float] = field(default_factory=list)

    def add(self, elapsed_s: float) -> None:
        value = float(elapsed_s)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("latency must be finite and non-negative")
        self.values_s.append(value)

    def summary(self) -> Dict[str, float]:
        if not self.values_s:
            return {"count": 0, "mean_s": 0.0, "p50_s": 0.0, "p95_s": 0.0, "p99_s": 0.0}
        values = np.asarray(self.values_s, dtype=np.float64)
        return {
            "count": int(values.size),
            "mean_s": float(values.mean()),
            "p50_s": float(np.percentile(values, 50)),
            "p95_s": float(np.percentile(values, 95)),
            "p99_s": float(np.percentile(values, 99)),
        }

