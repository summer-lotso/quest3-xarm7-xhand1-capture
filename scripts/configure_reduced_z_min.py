#!/usr/bin/env python3
"""Change only xArm7's reduced-mode TCP Z minimum while the arm is stationary."""

from __future__ import annotations

import argparse
import json

import numpy as np

from _common import add_config_argument, require_motion_confirmation
from xarm7_xhand1 import XArm7, load_config
from xarm7_xhand1.arm import _check_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--z-min-mm", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    if args.execute:
        require_motion_confirmation(args)

    with XArm7(load_config(args.config), allow_motion=args.execute) as arm:
        states = _check_code("get_reduced_states", arm._arm.get_reduced_states(is_radian=True))[1]
        if len(states) < 6 or not bool(states[5]):
            raise RuntimeError("Controller safety boundary must be enabled")
        reduced_mode_before = bool(states[0])
        before = [int(value) for value in states[1]]
        after = before.copy()
        after[5] = args.z_min_mm
        if args.z_min_mm < 50 or args.z_min_mm >= before[4] - 20:
            raise ValueError("Proposed Z minimum must be at least 50 mm and 20 mm below Z maximum")
        current = arm.read_tcp_pose()
        velocity = arm.read_state().velocity_rad_s
        if np.max(np.abs(velocity)) > 0.01:
            raise RuntimeError("xArm joints are still moving")
        if not after[5] < current[2] < after[4]:
            raise RuntimeError("Current TCP is outside the proposed Z range")
        print("REDUCED_BOUNDARY_BEFORE_MM=" + json.dumps(before), flush=True)
        print("REDUCED_BOUNDARY_PROPOSED_MM=" + json.dumps(after), flush=True)
        print(f"CURRENT_TCP_Z_MM={current[2]:.3f}", flush=True)
        if not args.execute:
            print("DRY_RUN: controller boundary unchanged", flush=True)
            return 0
        _check_code("set_reduced_tcp_boundary", arm._arm.set_reduced_tcp_boundary(after))
        # The SDK requires a reduced-mode reset for boundary edits to take
        # effect.  Restore its original state immediately; the independent
        # safety-boundary flag stays enabled throughout.
        _check_code("set_reduced_mode", arm._arm.set_reduced_mode(True))
        if not reduced_mode_before:
            _check_code("restore_reduced_mode", arm._arm.set_reduced_mode(False))
        result = _check_code("get_reduced_states", arm._arm.get_reduced_states(is_radian=True))[1]
        actual = [int(value) for value in result[1]]
        print("REDUCED_BOUNDARY_AFTER_MM=" + json.dumps(actual), flush=True)
        if actual != after or bool(result[0]) != reduced_mode_before or not bool(result[5]):
            raise RuntimeError("Controller did not confirm the requested reduced-mode boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
