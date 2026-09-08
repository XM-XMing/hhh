#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth correctness checker for XMflight Unity D435i-like depth.

Checks:
1) /xm/depth/image_raw metadata: width/height/encoding/payload
2) raw uint16 statistics
3) valid pixel ratio
4) center patch median depth
5) optional expected center depth error
6) optional horizontal/vertical flip indicators by comparing patch medians
7) optional save npz/csv/png debug files

This script does not require a calibration wall. It can do a statistical check in the forest scene.
For absolute metric validation, put a flat wall in front of the drone and pass --expected-center-m.

Examples:
  rosrun planning check_depth_correctness.py
  rosrun planning check_depth_correctness.py --expected-center-m 2.0 --tolerance-m 0.10
  rosrun planning check_depth_correctness.py --samples 30 --save-dir /tmp/depth_check
  rosrun planning check_depth_correctness.py --patch 20 --depth-scale 0.001
"""


import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo

try:
    from planning.msg import XMState
except Exception:
    XMState = None

@dataclass
class DepthStats:
    index: int
    stamp_sec: float
    width: int
    height: int
    encoding: str
    payload_bytes: int
    expected_bytes: int
    valid_ratio: float
    raw_min: int
    raw_max: int
    raw_nonzero_min: int
    raw_nonzero_max: int
    depth_min_m: float
    depth_max_m: float
    depth_mean_m: float
    depth_median_m: float
    center_median_m: float
    center_mean_m: float
    center_std_m: float
    center_valid_ratio: float
    top_median_m: float
    bottom_median_m: float
    left_median_m: float
    right_median_m: float
    min_clearance_m: float
    front_left_m: float
    front_mid_m: float
    front_right_m: float


class Latest:
    def __init__(self):
        self.depth: Optional[Image] = None
        self.camera_info: Optional[CameraInfo] = None
        self.state: Optional[Any] = None


def decode_depth_uint16(msg: Image) -> np.ndarray:
    if msg.encoding not in ("16UC1", "mono16"):
        raise ValueError("unsupported depth encoding: {}".format(msg.encoding))
    arr = np.frombuffer(msg.data, dtype=np.uint16)
    expected = int(msg.width) * int(msg.height)
    if arr.size != expected:
        raise ValueError("depth payload size mismatch: got {}, expected {}".format(arr.size, expected))
    return arr.reshape((int(msg.height), int(msg.width)))


def valid_depth_m(raw: np.ndarray, scale: float, max_raw_valid: int = 65534) -> np.ndarray:
    valid = (raw > 0) & (raw <= max_raw_valid)
    out = raw.astype(np.float32) * float(scale)
    out[~valid] = np.nan
    return out


def finite_median(arr: np.ndarray) -> float:
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def finite_mean(arr: np.ndarray) -> float:
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def finite_std(arr: np.ndarray) -> float:
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return float("nan")
    return float(np.std(values))


def patch(depth_m: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray:
    h, w = depth_m.shape
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    return depth_m[y0:y1, x0:x1]


def compute_stats(index: int, msg: Image, state: Optional[Any], args) -> Tuple[DepthStats, np.ndarray, np.ndarray]:
    raw = decode_depth_uint16(msg)
    depth_m = valid_depth_m(raw, args.depth_scale, args.max_raw_valid)
    h, w = raw.shape
    cx = w // 2
    cy = h // 2
    half = int(args.patch // 2)

    center = patch(depth_m, cx, cy, half)
    top = patch(depth_m, cx, int(h * 0.25), half)
    bottom = patch(depth_m, cx, int(h * 0.75), half)
    left = patch(depth_m, int(w * 0.25), cy, half)
    right = patch(depth_m, int(w * 0.75), cy, half)

    valid = np.isfinite(depth_m)
    nonzero = raw[raw > 0]
    if nonzero.size > 0:
        raw_nonzero_min = int(nonzero.min())
        raw_nonzero_max = int(nonzero.max())
    else:
        raw_nonzero_min = 0
        raw_nonzero_max = 0

    if state is not None:
        min_clearance = float(state.min_clearance)
        fronts = list(state.front_clearances)
        front_left = float(fronts[0]) if len(fronts) > 0 else float("nan")
        front_mid = float(fronts[1]) if len(fronts) > 1 else float("nan")
        front_right = float(fronts[2]) if len(fronts) > 2 else float("nan")
    else:
        min_clearance = float("nan")
        front_left = float("nan")
        front_mid = float("nan")
        front_right = float("nan")

    stamp_sec = float(msg.header.stamp.to_sec()) if msg.header.stamp else time.time()
    stats = DepthStats(
        index=index,
        stamp_sec=stamp_sec,
        width=int(msg.width),
        height=int(msg.height),
        encoding=str(msg.encoding),
        payload_bytes=len(msg.data),
        expected_bytes=int(msg.width) * int(msg.height) * 2,
        valid_ratio=float(np.count_nonzero(valid) / valid.size),
        raw_min=int(raw.min()) if raw.size else 0,
        raw_max=int(raw.max()) if raw.size else 0,
        raw_nonzero_min=raw_nonzero_min,
        raw_nonzero_max=raw_nonzero_max,
        depth_min_m=float(np.nanmin(depth_m)) if np.any(valid) else float("nan"),
        depth_max_m=float(np.nanmax(depth_m)) if np.any(valid) else float("nan"),
        depth_mean_m=finite_mean(depth_m),
        depth_median_m=finite_median(depth_m),
        center_median_m=finite_median(center),
        center_mean_m=finite_mean(center),
        center_std_m=finite_std(center),
        center_valid_ratio=float(np.count_nonzero(np.isfinite(center)) / center.size) if center.size else 0.0,
        top_median_m=finite_median(top),
        bottom_median_m=finite_median(bottom),
        left_median_m=finite_median(left),
        right_median_m=finite_median(right),
        min_clearance_m=min_clearance,
        front_left_m=front_left,
        front_mid_m=front_mid,
        front_right_m=front_right,
    )
    return stats, raw, depth_m


def summarize_numeric(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def print_table(stats_list: List[DepthStats], args) -> int:
    last = stats_list[-1]
    center_summary = summarize_numeric([s.center_median_m for s in stats_list])
    valid_summary = summarize_numeric([s.valid_ratio for s in stats_list])
    min_summary = summarize_numeric([s.depth_min_m for s in stats_list])
    max_summary = summarize_numeric([s.depth_max_m for s in stats_list])
    mid_clear_summary = summarize_numeric([s.front_mid_m for s in stats_list])

    results = []

    def add(name: str, ok: bool, value: str, target: str, note: str = ""):
        results.append((name, ok, value, target, note))

    add("metadata", last.width > 0 and last.height > 0 and last.encoding in ("16UC1", "mono16"),
        "{}x{} {}".format(last.width, last.height, last.encoding), "width>0 height>0 16UC1")

    add("payload", last.payload_bytes == last.expected_bytes,
        "{} bytes".format(last.payload_bytes), "{} bytes".format(last.expected_bytes))

    add("valid_ratio", valid_summary["mean"] >= args.valid_ratio_min,
        "{:.3f}".format(valid_summary["mean"]), ">= {:.3f}".format(args.valid_ratio_min))

    add("raw_nonzero_range", last.raw_nonzero_max > last.raw_nonzero_min > 0,
        "{}..{}".format(last.raw_nonzero_min, last.raw_nonzero_max), "nonzero and spread")

    add("depth_range_m", np.isfinite(min_summary["median"]) and np.isfinite(max_summary["median"]),
        "min_med={:.3f} max_med={:.3f}".format(min_summary["median"], max_summary["median"]), "finite")

    center_required = bool(args.require_center or args.expected_center_m > 0.0)
    center_available = bool(np.isfinite(center_summary["median"]))
    if center_required:
        add("center_valid_ratio", last.center_valid_ratio >= args.center_valid_ratio_min,
            "{:.3f}".format(last.center_valid_ratio), ">= {:.3f}".format(args.center_valid_ratio_min))
        add("center_stability", center_available and center_summary["std"] <= args.center_std_max,
            "std={:.4f}m".format(center_summary["std"]), "<= {:.3f}m".format(args.center_std_max))
    elif center_available:
        add("center_valid_ratio", True, "{:.3f}".format(last.center_valid_ratio), "informational in forest")
        add("center_stability", center_summary["std"] <= args.center_std_max,
            "std={:.4f}m".format(center_summary["std"]), "<= {:.3f}m".format(args.center_std_max))
    else:
        add("center_valid_ratio", True, "no return", "skipped; use --require-center with a target")
        add("center_stability", True, "no return", "skipped; use --require-center with a target")

    if args.expected_center_m > 0.0:
        err = abs(center_summary["median"] - args.expected_center_m)
        add("expected_center_depth", err <= args.tolerance_m,
            "median={:.3f}m err={:.3f}m".format(center_summary["median"], err),
            "{:.3f}±{:.3f}m".format(args.expected_center_m, args.tolerance_m))
    else:
        add("expected_center_depth", True,
            "skipped", "pass --expected-center-m")

    if center_available and np.isfinite(mid_clear_summary["median"]):
        # This is a loose consistency check only. Center depth and raycast front_mid are not identical definitions.
        diff = abs(center_summary["median"] - mid_clear_summary["median"])
        add("front_mid_consistency", diff <= args.front_mid_tolerance_m,
            "center={:.3f}m front_mid={:.3f}m diff={:.3f}m".format(
                center_summary["median"], mid_clear_summary["median"], diff
            ),
            "<= {:.3f}m loose".format(args.front_mid_tolerance_m),
            "loose check; definitions differ")
    else:
        add("front_mid_consistency", True, "skipped", "center return or planning/XMState unavailable")

    name_w = max(22, max(len(r[0]) for r in results))
    print("")
    print("XMflight depth correctness check")
    print("=" * 104)
    print(f"{'CHECK'.ljust(name_w)}  {'PASS'.ljust(6)}  {'VALUE'.ljust(36)}  {'TARGET'.ljust(24)}  NOTE")
    print("-" * 104)
    for name, ok, value, target, note in results:
        print(f"{name.ljust(name_w)}  {('PASS' if ok else 'FAIL').ljust(6)}  {value.ljust(36)}  {target.ljust(24)}  {note}")
    print("=" * 104)

    print("summary_stats")
    print("  samples:", len(stats_list))
    print("  center_median_m: mean={mean:.4f} std={std:.4f} median={median:.4f} min={min:.4f} max={max:.4f}".format(**center_summary))
    print("  valid_ratio:     mean={mean:.4f} std={std:.4f} median={median:.4f} min={min:.4f} max={max:.4f}".format(**valid_summary))
    print("  depth_min_m:     median={:.4f}".format(min_summary["median"]))
    print("  depth_max_m:     median={:.4f}".format(max_summary["median"]))
    print("  patches_last:")
    print("    top={:.3f}m bottom={:.3f}m left={:.3f}m right={:.3f}m center={:.3f}m".format(
        last.top_median_m, last.bottom_median_m, last.left_median_m, last.right_median_m, last.center_median_m
    ))

    passed = sum(1 for _, ok, _, _, _ in results if ok)
    print("  result: {}/{} checks passed".format(passed, len(results)))
    print("")

    return 0 if all(ok for _, ok, _, _, _ in results) else 1


def save_outputs(stats_list: List[DepthStats], last_raw: np.ndarray, last_depth_m: np.ndarray, args) -> None:
    if not args.save_dir:
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # CSV summary.
    csv_path = save_dir / "depth_stats.csv"
    fields = list(DepthStats.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in stats_list:
            writer.writerow({k: getattr(s, k) for k in fields})

    # NPZ last frame.
    npz_path = save_dir / "last_depth_frame.npz"
    np.savez_compressed(
        npz_path,
        raw_uint16=last_raw,
        depth_m=last_depth_m.astype(np.float32),
    )

    # PGM image for raw quick inspection. No external dependency.
    # Scale valid depth to 16-bit grayscale, near=bright, far=dark.
    d = last_depth_m.copy()
    invalid = ~np.isfinite(d)
    d[invalid] = args.depth_max_m
    d = np.clip(d, args.depth_min_m, args.depth_max_m)
    norm = (args.depth_max_m - d) / max(1e-6, args.depth_max_m - args.depth_min_m)
    img16 = np.clip(norm * 65535.0, 0, 65535).astype(np.uint16)
    pgm_path = save_dir / "last_depth_preview.pgm"
    with pgm_path.open("wb") as f:
        f.write(("P5\n{} {}\n65535\n".format(img16.shape[1], img16.shape[0])).encode("ascii"))
        # PGM expects big-endian 16-bit.
        f.write(img16.byteswap().tobytes())

    print("saved:")
    print("  csv:", csv_path)
    print("  npz:", npz_path)
    print("  pgm:", pgm_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-topic", default="/xm/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/xm/depth/camera_info")
    parser.add_argument("--state-topic", default="/xm/state")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--max-raw-valid", type=int, default=65534)
    parser.add_argument("--patch", type=int, default=20, help="center patch size in pixels")
    parser.add_argument("--valid-ratio-min", type=float, default=0.10)
    parser.add_argument("--center-valid-ratio-min", type=float, default=0.50)
    parser.add_argument("--center-std-max", type=float, default=0.20)
    parser.add_argument(
        "--require-center",
        action="store_true",
        help="require the center patch to hit a stable target; use for calibration-wall tests",
    )
    parser.add_argument("--expected-center-m", type=float, default=0.0)
    parser.add_argument("--tolerance-m", type=float, default=0.10)
    parser.add_argument("--front-mid-tolerance-m", type=float, default=0.50)
    parser.add_argument("--depth-min-m", type=float, default=0.30)
    parser.add_argument("--depth-max-m", type=float, default=6.00, help="preview scaling max depth; Unity raw depth is expected to allow up to 6m")
    parser.add_argument("--save-dir", default="")
    args = parser.parse_args()

    if args.patch < 1:
        raise SystemExit("--patch must be >= 1")

    rospy.init_node("check_depth_correctness", anonymous=True)

    latest = Latest()

    def on_depth(msg: Image):
        latest.depth = msg

    def on_info(msg: CameraInfo):
        latest.camera_info = msg

    def on_state(msg):
        latest.state = msg

    rospy.Subscriber(args.depth_topic, Image, on_depth, queue_size=2)
    rospy.Subscriber(args.camera_info_topic, CameraInfo, on_info, queue_size=1)
    if XMState is not None:
        rospy.Subscriber(args.state_topic, XMState, on_state, queue_size=5)

    stats_list: List[DepthStats] = []
    last_raw = None
    last_depth_m = None
    seen_stamps = set()

    deadline = time.time() + args.timeout
    rate = rospy.Rate(100)
    print("waiting for {} samples from {}".format(args.samples, args.depth_topic))

    while not rospy.is_shutdown() and len(stats_list) < args.samples and time.time() < deadline:
        msg = latest.depth
        if msg is None:
            rate.sleep()
            continue

        stamp_key = (int(msg.header.seq), int(msg.header.stamp.to_nsec()) if msg.header.stamp else len(stats_list))
        if stamp_key in seen_stamps:
            rate.sleep()
            continue
        seen_stamps.add(stamp_key)

        try:
            stats, raw, depth_m = compute_stats(len(stats_list), msg, latest.state, args)
        except Exception as exc:
            print("FAIL: cannot decode depth:", exc)
            return 2

        stats_list.append(stats)
        last_raw = raw
        last_depth_m = depth_m
        rate.sleep()

    if not stats_list:
        print("FAIL: no depth frames received from {}".format(args.depth_topic))
        return 2

    if latest.camera_info is not None:
        k = latest.camera_info.K
        print("camera_info:")
        print("  frame_id:", latest.camera_info.header.frame_id)
        print("  width,height:", latest.camera_info.width, latest.camera_info.height)
        print("  fx,fy,cx,cy:", k[0], k[4], k[2], k[5])
    else:
        print("camera_info: not received within timeout")

    rc = print_table(stats_list, args)
    if last_raw is not None and last_depth_m is not None:
        save_outputs(stats_list, last_raw, last_depth_m, args)
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        sys.exit(130)
