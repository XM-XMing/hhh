#!/usr/bin/env python3
"""Reproducible CPU hot-path benchmarks; not a correctness test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from planning.safety.depth_safety import (
    DepthSafetyConfig,
    local_depth_action_mask,
    primitive_depth_projection_table,
)
from planning.primitives.library import MotionPrimitiveLibrary


def _median_runtime(callable_, repeats: int) -> float:
    values = []
    for _ in range(int(repeats)):
        started = time.perf_counter()
        callable_()
        values.append(time.perf_counter() - started)
    return float(np.median(values))


def depth_mask_benchmark(repeats: int) -> dict:
    mpl = MotionPrimitiveLibrary()
    config = DepthSafetyConfig()
    depth = np.full((90, 160), 6.0, dtype=np.float32)
    projection = primitive_depth_projection_table(mpl, 90, 160, config)
    local_depth_action_mask(mpl, depth, config, projection=projection)

    cached_s = _median_runtime(
        lambda: local_depth_action_mask(mpl, depth, config, projection=projection), repeats
    )
    rebuild_s = _median_runtime(
        lambda: local_depth_action_mask(
            mpl,
            depth,
            config,
            projection=primitive_depth_projection_table(mpl, 90, 160, config),
        ),
        repeats,
    )
    return {
        "calls": repeats,
        "cached_median_ms": cached_s * 1000.0,
        "projection_rebuild_median_ms": rebuild_s * 1000.0,
        "projection_cache_speedup": rebuild_s / max(cached_s, 1.0e-12),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()
    result = {
        "contract_id": "planning_cpu_hotpaths",
        "depth_mask": depth_mask_benchmark(max(1, args.repeats)),
    }
    if args.out_json:
        output = Path(args.out_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
