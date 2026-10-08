#!/usr/bin/env python3
"""Return only xArm7 to the configured safe joint pose after a teleop stop."""

from __future__ import annotations

import argparse

import numpy as np
import zmq

from _common import add_config_argument, require_motion_confirmation
from run_official_vr_xarm_bridge import return_arm_home
from xarm7_xhand1 import XArm7, load_config
from xarm7_xhand1.arm import _check_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--check-only", action="store_true", help="Validate the current joint-space return path without motion")
    parser.add_argument("--home-speed-rad-s", type=float, default=0.12)
    parser.add_argument("--home-acceleration-rad-s2", type=float, default=0.25)
    args = parser.parse_args()
    if not args.check_only:
        require_motion_confirmation(args)
    config = load_config(args.config)
    if config.xarm_home_position_rad is None:
        raise ValueError("xarm_home_position_rad is not configured")
    target = np.asarray(config.xarm_home_position_rad, dtype=np.float64)
    context = zmq.Context()
    status = context.socket(zmq.PUB)
    status.bind("inproc://arm-home-status")
    try:
        with XArm7(config, allow_motion=not args.check_only) as arm:
            start = arm.read_state().position_rad
            reduced = _check_code("get_reduced_states", arm._arm.get_reduced_states(is_radian=True))[1]
            boundary = reduced[1] if len(reduced) >= 6 and bool(reduced[5]) else None
            sampled_xyz = []
            for fraction in np.linspace(0.0, 1.0, 101):
                joints = start + fraction * (target - start)
                pose = _check_code(
                    "get_forward_kinematics",
                    arm._arm.get_forward_kinematics(
                        joints.tolist(), input_is_radian=True, return_is_radian=True,
                    ),
                )[1]
                xyz = np.asarray(pose[:3], dtype=np.float64)
                sampled_xyz.append(xyz)
                if boundary is not None:
                    for axis in range(3):
                        maximum, minimum = float(boundary[axis * 2]), float(boundary[axis * 2 + 1])
                        if not minimum <= xyz[axis] <= maximum:
                            raise ValueError(f"joint return path leaves controller boundary at fraction {fraction:.2f}")
            sampled_xyz = np.asarray(sampled_xyz)
            print(
                "HOME_PATH_CHECK xyz_min_mm=" + str(sampled_xyz.min(axis=0).tolist())
                + " xyz_max_mm=" + str(sampled_xyz.max(axis=0).tolist()),
                flush=True,
            )
            if args.check_only:
                print("CHECK_ONLY: no xArm motion command issued", flush=True)
                return 0
            return_arm_home(
                arm, target, status, timeout_s=180.0, settle_s=1.0,
                speed_rad_s=args.home_speed_rad_s,
                acceleration_rad_s2=args.home_acceleration_rad_s2,
            )
    finally:
        status.close(linger=0)
        context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
