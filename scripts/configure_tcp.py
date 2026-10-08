#!/usr/bin/env python3
"""Inspect or explicitly apply the configured xArm palm TCP offset without moving joints."""
from __future__ import annotations

import argparse

import numpy as np

from _common import add_config_argument, require_motion_confirmation, show_array
from xarm7_xhand1 import XArm7, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.xarm_tcp_offset_mm_rad is None:
        raise ValueError("xarm_tcp_offset_mm_rad is not configured")
    if args.execute:
        require_motion_confirmation(args)

    expected = np.asarray(config.xarm_tcp_offset_mm_rad, dtype=np.float64)
    with XArm7(config, allow_motion=args.execute) as arm:
        before = arm.read_tcp_offset()
        print("configured_tcp_offset_mm_rad:", show_array(expected))
        print("controller_tcp_offset_before_mm_rad:", show_array(before))
        if args.execute:
            arm.configure_tcp_offset()
            after = arm.wait_for_tcp_offset(expected)
        else:
            after = arm.read_tcp_offset()
        pose = arm.read_tcp_pose()
        reduced = arm._arm.get_reduced_states(is_radian=True)
        print("controller_tcp_offset_after_mm_rad:", show_array(after))
        print("current_tcp_pose_mm_rad:", show_array(pose))
        print("controller_reduced_states:", reduced)
        if args.execute and not np.allclose(after, expected, atol=1e-5, rtol=0):
            raise RuntimeError("Controller TCP offset does not match the configured palm TCP")
        if args.execute:
            # Writing the offset leaves the controller disabled in state 5, which
            # the launcher preflight refuses. Return to enabled-and-stopped.
            arm.ensure_enabled()
            arm.stop()
        if not args.execute and not np.allclose(after, expected, atol=1e-5, rtol=0):
            print("status: controller TCP differs from the configured palm TCP; no change was made")

    print("TCP inspection completed; no joint motion command was issued.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
