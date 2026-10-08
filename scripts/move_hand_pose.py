#!/usr/bin/env python3
"""Move only XHand through a bounded low-speed pose sequence."""

from __future__ import annotations

import argparse
import time

import numpy as np

from _common import add_config_argument, require_motion_confirmation, show_array
from xarm7_xhand1 import XHand1, load_config


def move_segment(
    hand: XHand1,
    target: np.ndarray,
    *,
    step_rad: float,
    tolerance_rad: float,
    timeout_s: float,
    label: str,
) -> np.ndarray:
    deadline = time.monotonic() + timeout_s
    best_error = float("inf")
    last_progress_iteration = 0
    iteration = 0
    commanded = np.clip(
        hand.read_state().position_rad.copy(),
        XHand1.JOINT_LIMITS_RAD[:, 0],
        XHand1.JOINT_LIMITS_RAD[:, 1],
    )
    while True:
        current = hand.read_state().position_rad.copy()
        error = target - current
        max_error = float(np.max(np.abs(error)))
        if max_error <= tolerance_rad:
            hand.hold()
            print(
                f"{label}_complete iterations={iteration} max_error_rad={max_error:.6f}",
                flush=True,
            )
            return current
        if max_error < best_error - 0.002:
            best_error = max_error
            last_progress_iteration = iteration
        if iteration - last_progress_iteration >= 40:
            hand.hold()
            raise RuntimeError(
                f"{label} stalled for 40 commands; max_error_rad={max_error:.6f}"
            )
        if time.monotonic() >= deadline:
            hand.hold()
            raise TimeoutError(f"{label} timed out; max_error_rad={max_error:.6f}")

        commanded = commanded + np.clip(target - commanded, -step_rad, step_rad)
        commanded = np.clip(
            commanded, XHand1.JOINT_LIMITS_RAD[:, 0], XHand1.JOINT_LIMITS_RAD[:, 1]
        )
        hand.move_joints(commanded)
        iteration += 1
        if iteration == 1 or iteration % 20 == 0:
            print(
                f"{label}_progress iteration={iteration} max_error_rad={max_error:.6f}",
                flush=True,
            )
        time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--target-deg", type=float, nargs=12, required=True)
    parser.add_argument("--fraction", type=float, default=0.3)
    parser.add_argument("--step-rad", type=float, default=0.004)
    parser.add_argument("--tolerance-rad", type=float, default=0.015)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--return-home", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()

    config = load_config(args.config)
    if config.xhand_home_position_deg is None:
        raise ValueError("xhand_home_position_deg is not configured")
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")
    step_rad = min(args.step_rad, config.xhand_max_step_rad * 0.75)
    if step_rad <= 0 or args.tolerance_rad <= 0 or args.timeout_s <= 0:
        raise ValueError("Step, tolerance, and timeout must be positive")

    measured_home = np.radians(
        np.asarray(config.xhand_home_position_deg, dtype=np.float64)
    )
    # Feedback can sit a fraction of a degree outside the vendor command
    # range. Use the nearest commandable home without widening hard limits.
    home = np.clip(
        measured_home,
        XHand1.JOINT_LIMITS_RAD[:, 0],
        XHand1.JOINT_LIMITS_RAD[:, 1],
    )
    full_target = np.radians(np.asarray(args.target_deg, dtype=np.float64))
    target = home + args.fraction * (full_target - home)
    for name, pose in (("home", home), ("full_target", full_target), ("test_target", target)):
        if np.any(pose < XHand1.JOINT_LIMITS_RAD[:, 0]) or np.any(
            pose > XHand1.JOINT_LIMITS_RAD[:, 1]
        ):
            raise ValueError(f"{name} exceeds XHand absolute limits")
        print(f"{name}_deg:", show_array(np.degrees(pose)))

    require_motion_confirmation(args)
    with XHand1(config, allow_motion=True) as hand:
        before = hand.read_state().position_rad.copy()
        print("before_deg:", show_array(np.degrees(before)))
        move_segment(
            hand,
            home,
            step_rad=step_rad,
            tolerance_rad=args.tolerance_rad,
            timeout_s=args.timeout_s,
            label="home_before_test",
        )
        move_segment(
            hand,
            target,
            step_rad=step_rad,
            tolerance_rad=args.tolerance_rad,
            timeout_s=args.timeout_s,
            label="test_pose",
        )
        if args.return_home:
            after = move_segment(
                hand,
                home,
                step_rad=step_rad,
                tolerance_rad=args.tolerance_rad,
                timeout_s=args.timeout_s,
                label="home_after_test",
            )
        else:
            after = hand.read_state().position_rad.copy()
            hand.hold()
        print("after_deg:", show_array(np.degrees(after)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
