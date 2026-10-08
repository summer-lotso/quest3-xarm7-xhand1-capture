#!/usr/bin/env python3
"""Move xArm7 and XHand1 to the measured, repeatable VR teleoperation start pose."""

from __future__ import annotations

import argparse
import time

import numpy as np

from _common import add_config_argument, require_motion_confirmation, show_array
from xarm7_xhand1 import XArm7, XHand1, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--arm-step-rad", type=float, default=0.0075)
    parser.add_argument("--arm-tolerance-rad", type=float, default=0.003)
    parser.add_argument("--arm-timeout-s", type=float, default=90.0)
    parser.add_argument("--hand-step-rad", type=float, default=0.004)
    # This XHand's unloaded feedback has repeatable 1-2 degree static error.
    parser.add_argument("--hand-tolerance-rad", type=float, default=0.04)
    parser.add_argument("--hand-timeout-s", type=float, default=60.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    require_motion_confirmation(args)
    config = load_config(args.config)
    if config.xarm_teleop_ready_position_rad is None:
        raise ValueError("xarm_teleop_ready_position_rad is not configured")
    if config.xhand_teleop_ready_position_deg is None:
        raise ValueError("xhand_teleop_ready_position_deg is not configured")

    arm_target = np.asarray(config.xarm_teleop_ready_position_rad, dtype=np.float64)
    arm_step = min(args.arm_step_rad, config.xarm_max_step_rad * 0.75)
    with XArm7(config, allow_motion=True) as arm:
        before = arm.read_state().position_rad.copy()
        print("arm_before_rad:", show_array(before), flush=True)
        print("arm_target_rad:", show_array(arm_target), flush=True)
        deadline = time.monotonic() + args.arm_timeout_s
        iteration = 0
        try:
            while True:
                current = arm.read_state().position_rad.copy()
                error = arm_target - current
                max_error = float(np.max(np.abs(error)))
                if max_error <= args.arm_tolerance_rad:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"xArm initialization timed out; max_error_rad={max_error:.6f}")
                arm.move_joints(current + np.clip(error, -arm_step, arm_step))
                iteration += 1
                if iteration == 1 or iteration % 10 == 0:
                    print(f"arm_progress iteration={iteration} max_error_rad={max_error:.6f}", flush=True)
            print("arm_tcp_ready:", show_array(arm.read_tcp_pose()), flush=True)
        finally:
            arm.stop()

    hand_target = np.radians(np.asarray(config.xhand_teleop_ready_position_deg, dtype=np.float64))
    hand_target = np.clip(hand_target, XHand1.JOINT_LIMITS_RAD[:, 0], XHand1.JOINT_LIMITS_RAD[:, 1])
    hand_step = min(args.hand_step_rad, config.xhand_max_step_rad * 0.75)
    with XHand1(config, allow_motion=True) as hand:
        before = hand.read_state().position_rad.copy()
        print("hand_before_deg:", show_array(np.degrees(before)), flush=True)
        deadline = time.monotonic() + args.hand_timeout_s
        iteration = 0
        commanded = np.clip(before, XHand1.JOINT_LIMITS_RAD[:, 0], XHand1.JOINT_LIMITS_RAD[:, 1])
        while True:
            current = hand.read_state().position_rad.copy()
            error = hand_target - current
            max_error = float(np.max(np.abs(error)))
            if max_error <= args.hand_tolerance_rad:
                break
            if time.monotonic() >= deadline:
                hand.hold()
                raise TimeoutError(f"XHand initialization timed out; max_error_rad={max_error:.6f}")
            commanded += np.clip(hand_target - commanded, -hand_step, hand_step)
            commanded = np.clip(commanded, XHand1.JOINT_LIMITS_RAD[:, 0], XHand1.JOINT_LIMITS_RAD[:, 1])
            hand.move_joints(commanded)
            iteration += 1
            if iteration == 1 or iteration % 20 == 0:
                print(f"hand_progress iteration={iteration} max_error_rad={max_error:.6f}", flush=True)
            time.sleep(0.05)
        hand.hold()
        print("hand_ready_deg:", show_array(np.degrees(hand.read_state().position_rad)), flush=True)

    print("VR_TELEOP_INITIALIZED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
