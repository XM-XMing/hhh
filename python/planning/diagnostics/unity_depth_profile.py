#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check Unity raw depth profile after increasing max depth to 6m.

Purpose:
  Verify that /xm/depth/image_raw contains nonzero depth beyond 3m while the
  Python/RL observation can still be clipped to 0~3m by UnityForestEnv.

Typical use after changing Unity maxDepthRange to 6.0:

  rosrun planning check_unity_depth_profile.py \
    --raw-depth-max-m 6.0 \
    --rl-depth-max-m 3.0 \
    --require-far-depth \
    --save-dir /tmp/depth_profile_check

If the current scene has no object/background beyond 3m in the camera view,
--require-far-depth may fail. In that case, place a wall/cube/tree at 4~5m in
front of the drone, or run without --require-far-depth for statistics only.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image

@dataclass
class FrameStats:
    seq: int
    stamp_ns: int
    width: int
    height: int
    encoding: str
    valid_ratio: float
    raw_min_m: float
    raw_max_m: float
    raw_mean_m: float
    raw_p95_m: float
    far_gt_rl_ratio: float
    far_3_6_ratio: float
    clipped_norm_min: float
    clipped_norm_max: float
    clipped_saturated_ratio: float


class Latest:
    def __init__(self):
        self.depth: Optional[Image] = None
        self.info: Optional[CameraInfo] = None


def summarize(xs):
    arr = np.asarray([x for x in xs if np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def decode_depth(msg: Image, scale: float):
    if msg.encoding not in ("16UC1", "mono16"):
        raise ValueError("unsupported encoding {}".format(msg.encoding))
    raw = np.frombuffer(msg.data, dtype=np.uint16)
    expected = int(msg.width) * int(msg.height)
    if raw.size != expected:
        raise ValueError("payload size mismatch: got {}, expected {}".format(raw.size, expected))
    raw = raw.reshape((int(msg.height), int(msg.width)))
    depth_m = raw.astype(np.float32) * float(scale)
    valid = raw > 0
    return raw, depth_m, valid


def compute_stats(idx: int, msg: Image, args) -> FrameStats:
    raw, depth_m, valid = decode_depth(msg, args.depth_scale)
    valid_depth = depth_m[valid]
    if valid_depth.size:
        raw_min_m = float(np.min(valid_depth))
        raw_max_m = float(np.max(valid_depth))
        raw_mean_m = float(np.mean(valid_depth))
        raw_p95_m = float(np.percentile(valid_depth, 95.0))
    else:
        raw_min_m = raw_max_m = raw_mean_m = raw_p95_m = float("nan")

    far_gt_rl = valid & (depth_m > float(args.rl_depth_max_m))
    far_3_6 = valid & (depth_m > float(args.rl_depth_max_m)) & (depth_m <= float(args.raw_depth_max_m))

    clipped = depth_m.copy()
    clipped[~valid] = float(args.rl_depth_max_m)
    clipped = np.clip(clipped, float(args.rl_depth_min_m), float(args.rl_depth_max_m))
    clipped_norm = (clipped - float(args.rl_depth_min_m)) / max(1e-6, float(args.rl_depth_max_m - args.rl_depth_min_m))
    clipped_norm = np.clip(clipped_norm, 0.0, 1.0)

    return FrameStats(
        seq=int(getattr(msg.header, "seq", idx)),
        stamp_ns=int(msg.header.stamp.to_nsec()) if msg.header.stamp else idx,
        width=int(msg.width),
        height=int(msg.height),
        encoding=str(msg.encoding),
        valid_ratio=float(np.mean(valid)),
        raw_min_m=raw_min_m,
        raw_max_m=raw_max_m,
        raw_mean_m=raw_mean_m,
        raw_p95_m=raw_p95_m,
        far_gt_rl_ratio=float(np.mean(far_gt_rl)),
        far_3_6_ratio=float(np.mean(far_3_6)),
        clipped_norm_min=float(np.min(clipped_norm)),
        clipped_norm_max=float(np.max(clipped_norm)),
        clipped_saturated_ratio=float(np.mean(clipped_norm >= 0.999)),
    )


def save_outputs(stats: List[FrameStats], last_raw: np.ndarray, last_depth_m: np.ndarray, args):
    if not args.save_dir:
        return
    out = Path(args.save_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "depth_profile_stats.csv"
    fields = list(FrameStats.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in stats:
            w.writerow({k: getattr(s, k) for k in fields})

    npz_path = out / "last_raw_depth_6m.npz"
    np.savez_compressed(npz_path, raw_uint16=last_raw, depth_m=last_depth_m.astype(np.float32))

    # Preview scaled to raw max depth: near bright, far dark, invalid black.
    d = last_depth_m.copy()
    valid = d > 0.0
    d = np.clip(d, float(args.raw_depth_min_m), float(args.raw_depth_max_m))
    norm = (float(args.raw_depth_max_m) - d) / max(1e-6, float(args.raw_depth_max_m - args.raw_depth_min_m))
    img16 = np.clip(norm * 65535.0, 0, 65535).astype(np.uint16)
    img16[~valid] = 0
    pgm_path = out / "last_raw_depth_6m_preview.pgm"
    with pgm_path.open("wb") as f:
        f.write(("P5\n{} {}\n65535\n".format(img16.shape[1], img16.shape[0])).encode("ascii"))
        f.write(img16.byteswap().tobytes())

    print("saved:")
    print("  csv:", csv_path)
    print("  npz:", npz_path)
    print("  pgm:", pgm_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-topic", default="/xm/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/xm/depth/camera_info")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--raw-depth-min-m", type=float, default=0.30)
    parser.add_argument("--raw-depth-max-m", type=float, default=6.00)
    parser.add_argument("--rl-depth-min-m", type=float, default=0.30)
    parser.add_argument("--rl-depth-max-m", type=float, default=3.00)
    parser.add_argument("--valid-ratio-min", type=float, default=0.05)
    parser.add_argument("--require-far-depth", action="store_true")
    parser.add_argument("--far-depth-min-m", type=float, default=3.20)
    parser.add_argument("--far-ratio-min", type=float, default=0.001)
    parser.add_argument("--save-dir", default="")
    args = parser.parse_args()

    rospy.init_node("check_unity_depth_profile", anonymous=True)
    latest = Latest()

    def on_depth(msg: Image):
        latest.depth = msg

    def on_info(msg: CameraInfo):
        latest.info = msg

    rospy.Subscriber(args.depth_topic, Image, on_depth, queue_size=2)
    rospy.Subscriber(args.camera_info_topic, CameraInfo, on_info, queue_size=1)

    stats: List[FrameStats] = []
    seen = set()
    last_raw = None
    last_depth_m = None
    deadline = time.time() + float(args.timeout)
    rate = rospy.Rate(100)

    print("waiting for {} samples from {}".format(args.samples, args.depth_topic))
    while not rospy.is_shutdown() and len(stats) < args.samples and time.time() < deadline:
        msg = latest.depth
        if msg is None:
            rate.sleep()
            continue
        key = (int(getattr(msg.header, "seq", len(stats))), int(msg.header.stamp.to_nsec()) if msg.header.stamp else len(stats))
        if key in seen:
            rate.sleep()
            continue
        seen.add(key)
        try:
            st = compute_stats(len(stats), msg, args)
            raw, depth_m, _ = decode_depth(msg, args.depth_scale)
        except Exception as e:
            print("FAIL: cannot decode depth:", repr(e))
            return 2
        stats.append(st)
        last_raw = raw.copy()
        last_depth_m = depth_m.copy()
        rate.sleep()

    if not stats:
        print("FAIL: no depth frames received")
        return 2

    if latest.info is not None:
        k = latest.info.K
        print("camera_info:")
        print("  frame_id:", latest.info.header.frame_id)
        print("  width,height:", latest.info.width, latest.info.height)
        print("  fx,fy,cx,cy:", k[0], k[4], k[2], k[5])
    else:
        print("camera_info: not received")

    valid = summarize([s.valid_ratio for s in stats])
    raw_max = summarize([s.raw_max_m for s in stats])
    raw_p95 = summarize([s.raw_p95_m for s in stats])
    far = summarize([s.far_gt_rl_ratio for s in stats])
    sat = summarize([s.clipped_saturated_ratio for s in stats])

    results = []
    def add(name, ok, value, target):
        results.append((name, bool(ok), value, target))

    last = stats[-1]
    add("metadata", last.width > 0 and last.height > 0 and last.encoding in ("16UC1", "mono16"),
        "{}x{} {}".format(last.width, last.height, last.encoding), "16UC1/mono16")
    add("valid_ratio", valid["mean"] >= args.valid_ratio_min,
        "mean={:.4f}".format(valid["mean"]), ">= {:.4f}".format(args.valid_ratio_min))
    add("raw_depth_max_seen", raw_max["max"] <= args.raw_depth_max_m + 0.10,
        "max={:.3f}m p95_med={:.3f}m".format(raw_max["max"], raw_p95["median"]), "<= raw_depth_max_m + 0.10")
    add("rl_clip_simulation", all(0.0 <= s.clipped_norm_min <= s.clipped_norm_max <= 1.0001 for s in stats),
        "norm_min/max last={:.3f}/{:.3f}".format(last.clipped_norm_min, last.clipped_norm_max), "normalized within [0,1]")

    if args.require_far_depth:
        add("far_depth_beyond_rl_clip", raw_max["max"] >= args.far_depth_min_m and far["mean"] >= args.far_ratio_min,
            "raw_max={:.3f}m far_ratio_mean={:.5f}".format(raw_max["max"], far["mean"]),
            ">= {:.2f}m and ratio >= {:.5f}".format(args.far_depth_min_m, args.far_ratio_min))
    else:
        add("far_depth_beyond_rl_clip", True,
            "raw_max={:.3f}m far_ratio_mean={:.5f}".format(raw_max["max"], far["mean"]),
            "statistics only; pass --require-far-depth")

    print("")
    print("UNITY_DEPTH_PROFILE_CHECK")
    print("=" * 100)
    print("raw depth expected: {:.2f}..{:.2f}m".format(args.raw_depth_min_m, args.raw_depth_max_m))
    print("rl clip expected:   {:.2f}..{:.2f}m".format(args.rl_depth_min_m, args.rl_depth_max_m))
    print("samples:", len(stats))
    print("raw_max_m: median={:.3f} max={:.3f}".format(raw_max["median"], raw_max["max"]))
    print("raw_p95_m: median={:.3f}".format(raw_p95["median"]))
    print("far_gt_{:.1f}m_ratio: mean={:.5f} max={:.5f}".format(args.rl_depth_max_m, far["mean"], far["max"]))
    print("rl_clipped_saturated_ratio: mean={:.5f}".format(sat["mean"]))
    print("-" * 100)
    for name, ok, value, target in results:
        print("{:<26} {:<6} {:<34} {}".format(name, "PASS" if ok else "FAIL", value, target))
    print("=" * 100)

    if last_raw is not None and last_depth_m is not None:
        save_outputs(stats, last_raw, last_depth_m, args)

    if all(ok for _, ok, _, _ in results):
        print("RESULT=PASS")
        return 0
    print("RESULT=FAIL")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except rospy.ROSInterruptException:
        raise SystemExit(130)
