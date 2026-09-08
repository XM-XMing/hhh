#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test UnityForestEnv native MPL step(action_id)."""

from __future__ import annotations




import argparse
import numpy as np

from planning.runtime.unity_env import EnvConfig, UnityForestEnv


def run_one(action_id: int, args) -> dict:
    cfg = EnvConfig(
        depth_out_width=160,
        depth_out_height=90,
        reset_settle_s=args.settle,
        primitive_post_wait_s=args.post_wait,
    )
    env = UnityForestEnv(start=args.start, goal=args.goal, config=cfg)
    obs0 = env.reset(start=args.start, goal=args.goal)
    p0 = obs0["state"]["position"].copy()
    yaw0 = obs0["state"]["yaw"]

    obs, reward, done, info = env.step(action_id)
    p1 = obs["state"]["position"].copy()

    print("UNITY_MOTION_PRIMITIVE_STEP")
    print("  action_id:", action_id)
    print("  start_pos:", p0)
    print("  end_pos:", p1)
    print("  start_yaw_deg:", float(np.rad2deg(yaw0)))
    print("  end_yaw_deg:", float(np.rad2deg(obs['state']['yaw'])))
    print("  delta_body:", info["primitive_delta_body"])
    print("  ref_body:", info["primitive_endpoint_ref_body"])
    print("  endpoint_error_body_m:", info["primitive_endpoint_error_body_m"])
    print("  reward:", reward, "done:", done)
    print("  action_mask_valid_count:", int(np.count_nonzero(obs["action_mask"])))
    print("  prev_action_id:", obs["prev_action_id"])
    print("  safety: collided={} altitude_violation={} min_clearance={:.3f} z={:.3f}".format(
        info["collided"],
        info["altitude_violation"],
        info["min_clearance"],
        info["z"],
    ))
    env.stop()
    return {
        "action_id": action_id,
        "error": float(info["primitive_endpoint_error_body_m"]),
        "done": bool(done),
        "collided": bool(info["collided"]),
        "altitude_bad": bool(info["hard_altitude_violation"]),
        "prev_action_id": int(obs["prev_action_id"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", default="0,52,104")
    parser.add_argument("--start", type=float, nargs=3, default=[0.0, 0.0, 2.0])
    parser.add_argument("--goal", type=float, nargs=3, default=[40.0, 0.0, 2.0])
    parser.add_argument("--settle", type=float, default=0.30)
    parser.add_argument("--post-wait", type=float, default=0.05)
    parser.add_argument("--tol", type=float, default=0.35)
    args = parser.parse_args()

    actions = [int(x.strip()) for x in args.actions.split(",") if x.strip()]
    rows = []
    for a in actions:
        rows.append(run_one(a, args))

    print("")
    print("UNITY_MOTION_PRIMITIVE_STEP_SUMMARY")
    for r in rows:
        ok = r["error"] <= args.tol and not r["collided"] and not r["altitude_bad"] and r["prev_action_id"] == r["action_id"]
        print("  id={:03d} {} err={:.3f}".format(r["action_id"], "PASS" if ok else "FAIL", r["error"]))
        if not ok:
            print("RESULT=FAIL")
            return 1

    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
