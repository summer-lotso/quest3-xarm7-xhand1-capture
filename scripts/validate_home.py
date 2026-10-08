#!/usr/bin/env python3
"""Explicitly validate the configured xArm and XHand home poses."""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from _common import add_config_argument, require_motion_confirmation, show_array
from xarm7_xhand1 import XArm7, XHand1, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument("--hand-timeout-s", type=float, default=60.0)
    parser.add_argument("--hand-tolerance-rad", type=float, default=0.015)
    parser.add_argument("--hand-command-step-rad", type=float, default=0.004)
    parser.add_argument("--arm-timeout-s", type=float, default=90.0)
    parser.add_argument("--arm-tolerance-rad", type=float, default=0.003)
    parser.add_argument("--arm-command-step-rad", type=float, default=0.0075)
    args = parser.parse_args()
    require_motion_confirmation(args)
    config = load_config(args.config)
    if config.xarm_home_position_rad is None or config.xhand_home_position_deg is None:
        raise ValueError("Both xArm and XHand home positions must be configured")

    arm_target = np.asarray(config.xarm_home_position_rad, dtype=np.float64)
    arm_command_step = min(args.arm_command_step_rad, config.xarm_max_step_rad * 0.75)
    if arm_command_step <= 0 or args.arm_tolerance_rad <= 0 or args.arm_timeout_s <= 0:
        raise ValueError("Arm step, tolerance, and timeout must be positive")
    with XArm7(config, allow_motion=True) as arm:
        arm_before = arm.read_state().position_rad.copy()
        print("arm_before_rad:", show_array(arm_before))
        print("arm_target_rad:", show_array(arm_target))
        print("arm_max_requested_delta_rad:", float(np.max(np.abs(arm_target - arm_before))))
        arm_deadline = time.monotonic() + args.arm_timeout_s
        arm_iteration = 0
        while True:
            current = arm.read_state().position_rad.copy()
            error = arm_target - current
            max_error = float(np.max(np.abs(error)))
            if max_error <= args.arm_tolerance_rad:
                break
            if time.monotonic() >= arm_deadline:
                arm.stop()
                raise TimeoutError(f"xArm home timed out with max error {max_error:.6f} rad")
            command = current + np.clip(error, -arm_command_step, arm_command_step)
            arm.move_joints(command)
            arm_iteration += 1
            if arm_iteration == 1 or arm_iteration % 10 == 0:
                print(f"arm_progress iteration={arm_iteration} max_error_rad={max_error:.6f}")
        arm_after = arm.read_state().position_rad.copy()
        print("arm_after_rad:", show_array(arm_after))
        print("arm_iterations:", arm_iteration)
        print("arm_max_error_rad:", float(np.max(np.abs(arm_target - arm_after))))

    hand_target = np.radians(np.asarray(config.xhand_home_position_deg, dtype=np.float64))
    command_step = min(args.hand_command_step_rad, config.xhand_max_step_rad * 0.75)
    if command_step <= 0 or args.hand_tolerance_rad <= 0 or args.hand_timeout_s <= 0:
        raise ValueError("Hand step, tolerance, and timeout must be positive")
    with XHand1(config, allow_motion=True) as hand:
        hand_before = hand.read_state().position_rad.copy()
        print("hand_before_rad:", show_array(hand_before))
        print("hand_target_rad:", show_array(hand_target))
        deadline = time.monotonic() + args.hand_timeout_s
        iteration = 0
        best_error = float("inf")
        last_progress_iteration = 0
        while True:
            current = hand.read_state().position_rad.copy()
            error = hand_target - current
            max_error = float(np.max(np.abs(error)))
            if max_error <= args.hand_tolerance_rad:
                break
            if max_error < best_error - 0.002:
                best_error = max_error
                last_progress_iteration = iteration
            if iteration - last_progress_iteration >= 40:
                hand.hold()
                raise RuntimeError(
                    "XHand home stalled for 40 commands without 0.002 rad progress; "
                    f"max error remains {max_error:.6f} rad. Do not increase tor_max without review."
                )
            if time.monotonic() >= deadline:
                hand.hold()
                raise TimeoutError(f"XHand home timed out with max error {max_error:.6f} rad")
            command = current + np.clip(error, -command_step, command_step)
            command = np.clip(command, XHand1.JOINT_LIMITS_RAD[:, 0], XHand1.JOINT_LIMITS_RAD[:, 1])
            hand.move_joints(command)
            iteration += 1
            if iteration == 1 or iteration % 10 == 0:
                print(f"hand_progress iteration={iteration} max_error_rad={max_error:.6f}")
            time.sleep(0.05)
        hand_after = hand.read_state().position_rad.copy()
        hand.hold()
        print("hand_after_rad:", show_array(hand_after))
        print("hand_iterations:", iteration)
        print("hand_max_error_rad:", float(np.max(np.abs(hand_target - hand_after))))

    print("Configured home validation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
