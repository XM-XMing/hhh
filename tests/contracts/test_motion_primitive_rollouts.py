#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-rollout all MPL primitives in Unity and measure endpoint error.

Default:
  - Reset to [0,0,2] before every action.
  - Wait 0.30s after reset.
  - Execute one 0.50 s primitive.
  - Compare actual displacement in start body frame with primitive reference endpoint.
  - Save CSV.

Usage:
  rosrun planning test_motion_primitive_rollouts.py
  rosrun planning test_motion_primitive_rollouts.py --actions 52
  rosrun planning test_motion_primitive_rollouts.py --actions 0,52,104
  rosrun planning test_motion_primitive_rollouts.py --actions 0-104 --csv /tmp/mpl_rollout_all.csv
"""

from __future__ import annotations

from pathlib import Path



import argparse
import csv
import time
from typing import List

import numpy as np
import rospy

from planning.runtime.unity_env import EnvConfig, UnityForestEnv
from planning.primitives.library import MotionPrimitiveLibrary, rotation_z



def parse_actions(text: str, n_actions: int) -> List[int]:
    if text is None or text.strip() == "" or text.strip().lower() == "all":
        return list(range(n_actions))

    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a)
            b = int(b)
            if b < a:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))

    out = sorted(set(i for i in out if 0 <= i < n_actions))
    if not out:
        raise ValueError("no valid action id selected")
    return out


def execute_cmd_seq(env: UnityForestEnv, cmd_seq: np.ndarray, dt: float, stop_at_end: bool = True) -> None:
    rate = rospy.Rate(1.0 / float(dt))
    for cmd in cmd_seq:
        env._cmd_pub.publish(
            env._make_twist(float(cmd[0]), float(cmd[1]), float(cmd[2]), float(cmd[3]))
        )
        rate.sleep()

    if stop_at_end:
        env._cmd_pub.publish(env._make_twist(0.0, 0.0, 0.0, 0.0))


def rollout_one(env: UnityForestEnv, mpl: MotionPrimitiveLibrary, action_id: int, args) -> dict:
    env.reset(start=args.start, goal=args.goal)
    env.stop()
    if args.settle > 0:
        rospy.sleep(float(args.settle))

    obs0 = env._get_observation_blocking()
    p0 = obs0["state"]["position"].astype(np.float32).copy()
    yaw0 = float(obs0["state"]["yaw"])

    cmd_seq = mpl.command_sequence(action_id)
    ref_body = mpl.endpoint(action_id).astype(np.float32)
    ref_world = rotation_z(yaw0).dot(ref_body)

    t0 = time.time()
    execute_cmd_seq(env, cmd_seq, mpl.control_dt_s, stop_at_end=(not args.no_stop))
    wall_elapsed = time.time() - t0

    if args.post_wait > 0:
        rospy.sleep(float(args.post_wait))

    obs1 = env._get_observation_blocking()
    p1 = obs1["state"]["position"].astype(np.float32).copy()
    yaw1 = float(obs1["state"]["yaw"])

    delta_world = p1 - p0
    delta_body = rotation_z(-yaw0).dot(delta_world)

    err_body = float(np.linalg.norm(delta_body - ref_body))
    err_world = float(np.linalg.norm(delta_world - ref_world))
    err_xy_body = float(np.linalg.norm(delta_body[:2] - ref_body[:2]))
    err_z = float(abs(delta_body[2] - ref_body[2]))

    meta = mpl.action_metadata(action_id)
    reward, done, info = env._compute_reward_done(
        obs1,
        np.asarray([float(action_id), 0.0, 0.0, 0.0], dtype=np.float32),
    )

    row = {
        "action_id": int(action_id),
        "horizontal_index": int(meta["horizontal_index"]),
        "vertical_index": int(meta["vertical_index"]),
        "y_end_ref": float(ref_body[1]),
        "z_end_ref": float(ref_body[2]),
        "heading_deg_ref": float(meta["terminal_heading_deg"]),
        "start_x": float(p0[0]),
        "start_y": float(p0[1]),
        "start_z": float(p0[2]),
        "end_x": float(p1[0]),
        "end_y": float(p1[1]),
        "end_z": float(p1[2]),
        "start_yaw_deg": float(np.rad2deg(yaw0)),
        "end_yaw_deg": float(np.rad2deg(yaw1)),
        "yaw_delta_deg": float(np.rad2deg(yaw1 - yaw0)),
        "delta_body_x": float(delta_body[0]),
        "delta_body_y": float(delta_body[1]),
        "delta_body_z": float(delta_body[2]),
        "ref_body_x": float(ref_body[0]),
        "ref_body_y": float(ref_body[1]),
        "ref_body_z": float(ref_body[2]),
        "endpoint_error_body_m": err_body,
        "endpoint_error_world_m": err_world,
        "endpoint_error_xy_body_m": err_xy_body,
        "endpoint_error_z_m": err_z,
        "ratio_x": float(delta_body[0] / ref_body[0]) if abs(float(ref_body[0])) > 1e-6 else float("nan"),
        "ratio_y": float(delta_body[1] / ref_body[1]) if abs(float(ref_body[1])) > 1e-6 else float("nan"),
        "ratio_z": float(delta_body[2] / ref_body[2]) if abs(float(ref_body[2])) > 1e-6 else float("nan"),
        "wall_elapsed_s": float(wall_elapsed),
        "reward": float(reward),
        "done": bool(done),
        "collided": bool(info.get("collided", False)),
        "altitude_violation": bool(info.get("altitude_violation", False)),
        "hard_altitude_violation": bool(info.get("hard_altitude_violation", False)),
        "min_clearance": float(info.get("min_clearance", float("nan"))),
        "pass": bool(
            err_body <= args.tol
            and err_z <= args.z_tol
            and not bool(info.get("collided", False))
            and not bool(info.get("hard_altitude_violation", False))
        ),
    }
    return row


def summarize(rows: List[dict], tol: float) -> dict:
    arr_err = np.asarray([r["endpoint_error_body_m"] for r in rows], dtype=np.float64)
    arr_xy = np.asarray([r["endpoint_error_xy_body_m"] for r in rows], dtype=np.float64)
    arr_z = np.asarray([r["endpoint_error_z_m"] for r in rows], dtype=np.float64)
    arr_rx = np.asarray([r["ratio_x"] for r in rows], dtype=np.float64)
    pass_count = sum(1 for r in rows if r["pass"])
    collided = sum(1 for r in rows if r["collided"])
    alt = sum(1 for r in rows if r["hard_altitude_violation"] or r["altitude_violation"])

    return {
        "n": len(rows),
        "pass_count": pass_count,
        "fail_count": len(rows) - pass_count,
        "pass_rate": pass_count / max(1, len(rows)),
        "mean_err": float(np.mean(arr_err)),
        "median_err": float(np.median(arr_err)),
        "max_err": float(np.max(arr_err)),
        "p95_err": float(np.percentile(arr_err, 95)),
        "mean_xy_err": float(np.mean(arr_xy)),
        "max_xy_err": float(np.max(arr_xy)),
        "mean_z_err": float(np.mean(arr_z)),
        "max_z_err": float(np.max(arr_z)),
        "mean_ratio_x": float(np.nanmean(arr_rx)),
        "min_ratio_x": float(np.nanmin(arr_rx)),
        "max_ratio_x": float(np.nanmax(arr_rx)),
        "collided": collided,
        "altitude_related": alt,
    }


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", default="all", help="all, 52, 0,52,104, or 0-104")
    parser.add_argument("--start", type=float, nargs=3, default=[0.0, 0.0, 2.0])
    parser.add_argument("--goal", type=float, nargs=3, default=[40.0, 0.0, 2.0])
    parser.add_argument("--settle", type=float, default=0.30)
    parser.add_argument("--post-wait", type=float, default=0.05)
    parser.add_argument("--tol", type=float, default=0.35)
    parser.add_argument("--z-tol", type=float, default=0.20)
    parser.add_argument("--csv", default="/tmp/mpl_rollout_all.csv")
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--no-stop", action="store_true")
    args = parser.parse_args()

    mpl = MotionPrimitiveLibrary()
    actions = parse_actions(args.actions, mpl.num_actions)

    cfg = EnvConfig(depth_out_width=160, depth_out_height=90)
    env = UnityForestEnv(start=args.start, goal=args.goal, config=cfg)

    print("MOTION_PRIMITIVE_ROLLOUT_ALL_START")
    print("  actions:", len(actions), actions[:10], "..." if len(actions) > 10 else "")
    print("  tol:", args.tol, "z_tol:", args.z_tol)
    print("  csv:", args.csv)

    rows: List[dict] = []
    t_all = time.time()

    for idx, action_id in enumerate(actions):
        row = rollout_one(env, mpl, action_id, args)
        rows.append(row)

        status = "PASS" if row["pass"] else "FAIL"
        print(
            "[{}/{}] id={:03d} {} err={:.3f} xy={:.3f} zerr={:.3f} "
            "delta_body=({:.3f},{:.3f},{:.3f}) ref=({:.3f},{:.3f},{:.3f}) "
            "yaw_delta={:.1f} clear={:.3f}".format(
                idx + 1,
                len(actions),
                action_id,
                status,
                row["endpoint_error_body_m"],
                row["endpoint_error_xy_body_m"],
                row["endpoint_error_z_m"],
                row["delta_body_x"],
                row["delta_body_y"],
                row["delta_body_z"],
                row["ref_body_x"],
                row["ref_body_y"],
                row["ref_body_z"],
                row["yaw_delta_deg"],
                row["min_clearance"],
            )
        )

        if args.stop_on_fail and not row["pass"]:
            break

    env.stop()

    if rows:
        csv_path = Path(args.csv).expanduser().resolve()
        write_csv(csv_path, rows)
        s = summarize(rows, args.tol)

        print("")
        print("MOTION_PRIMITIVE_ROLLOUT_ALL_SUMMARY")
        print("  tested:", s["n"])
        print("  pass_count:", s["pass_count"])
        print("  fail_count:", s["fail_count"])
        print("  pass_rate: {:.1f}%".format(100.0 * s["pass_rate"]))
        print("  endpoint_error_body_m: mean={:.3f} median={:.3f} p95={:.3f} max={:.3f}".format(
            s["mean_err"], s["median_err"], s["p95_err"], s["max_err"]
        ))
        print("  endpoint_error_xy_body_m: mean={:.3f} max={:.3f}".format(s["mean_xy_err"], s["max_xy_err"]))
        print("  endpoint_error_z_m: mean={:.3f} max={:.3f}".format(s["mean_z_err"], s["max_z_err"]))
        print("  ratio_x: mean={:.3f} min={:.3f} max={:.3f}".format(
            s["mean_ratio_x"], s["min_ratio_x"], s["max_ratio_x"]
        ))
        print("  collided:", s["collided"])
        print("  altitude_related:", s["altitude_related"])
        print("  csv:", csv_path)
        print("  wall_time_s:", round(time.time() - t_all, 2))

        if s["pass_rate"] < 0.95 or s["max_err"] > max(args.tol, 0.45):
            print("RESULT=FAIL")
            return 1

    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
