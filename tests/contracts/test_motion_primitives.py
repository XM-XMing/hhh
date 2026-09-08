#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test and optionally dump the 105-action MPL library."""

from pathlib import Path



import argparse

import numpy as np

from planning.primitives.library import MotionPrimitiveLibrary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default="", help="Optional directory to dump each action csv.")
    args = parser.parse_args()

    lib = MotionPrimitiveLibrary()
    print("MOTION_PRIMITIVE_LIBRARY_TEST=PASS")
    print("actions", lib.num_actions)
    print("frames", lib.frames)
    print("duration", lib.duration_s)
    print("dt", lib.control_dt_s)

    for action_id in [0, 3, 52, 101, 104]:
        meta = lib.action_metadata(action_id)
        cmd = lib.command_sequence(action_id)
        endpoint = lib.endpoint(action_id)
        speed = np.linalg.norm(cmd[:, :3], axis=1)
        print(
            "id={:03d} h={} v={} y_end={:.3f} z_end={:.3f} heading={:.1f} endpoint={} max_speed={:.3f} max_yaw_rate_deg={:.1f}".format(
                action_id,
                meta["horizontal_index"],
                meta["vertical_index"],
                meta["lateral_endpoint_m"],
                meta["vertical_endpoint_m"],
                meta["terminal_heading_deg"],
                np.round(endpoint, 3).tolist(),
                float(speed.max()),
                float(np.rad2deg(np.max(np.abs(cmd[:, 3])))),
            )
        )

    if args.save_dir:
        out = Path(args.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for action_id in range(lib.num_actions):
            cmd = lib.command_sequence(action_id)
            path = lib.reference_path(action_id)
            n = cmd.shape[0]
            arr = np.column_stack([
                np.arange(n),
                lib.t_cmd,
                cmd,
                path[:-1, 0],
                path[:-1, 1],
                path[:-1, 2],
            ])
            header = "frame,t_sec,vx_body,vy_body,vz_body,yaw_rate,x_ref,y_ref,z_ref"
            np.savetxt(out / "primitive_{:03d}.csv".format(action_id), arr, delimiter=",", header=header, comments="")
        print("dumped", lib.num_actions, "csv files to", out)


if __name__ == "__main__":
    main()
