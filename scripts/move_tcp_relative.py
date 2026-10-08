#!/usr/bin/env python3
"""Move the xArm TCP by a small Cartesian offset with bounded interpolation."""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from _common import add_config_argument, require_motion_confirmation, show_array
from xarm7_xhand1 import XArm7, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--dx-mm", type=float, default=0.0)
    parser.add_argument("--dy-mm", type=float, default=0.0)
    parser.add_argument("--dz-mm", type=float, default=0.0)
    parser.add_argument("--speed-mm-s", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--max-distance-mm", type=float, default=100.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    delta = np.asarray([args.dx_mm, args.dy_mm, args.dz_mm], dtype=np.float64)
    if not np.isfinite(delta).all() or not np.isfinite(args.speed_mm_s) or not np.isfinite(args.rate_hz):
        raise ValueError("Offsets, speed and rate must be finite")
    distance = float(np.linalg.norm(delta))
    if distance <= 0 or distance > args.max_distance_mm:
        raise ValueError(f"Cartesian offset norm must be in (0, {args.max_distance_mm}] mm")
    if args.speed_mm_s <= 0 or args.rate_hz <= 0:
        raise ValueError("Speed and rate must be positive")
    if not args.execute:
        print(f"DRY_RUN relative_tcp_mm={show_array(delta)} distance_mm={distance:.3f}")
        return 0
    require_motion_confirmation(args)

    config = load_config(args.config)
    arm = XArm7(config, allow_motion=True)
    try:
        arm.connect()
        start = arm.start_cartesian_servo()
        target = start.copy(); target[:3] += delta
        step = min(config.xarm_max_cartesian_step_mm * 0.95, args.speed_mm_s / args.rate_hz)
        steps = max(1, int(math.ceil(float(np.max(np.abs(delta))) / step)))
        period = 1.0 / args.rate_hz
        next_tick = time.monotonic()
        print(f"TCP_MOVE_START from={show_array(start)} to={show_array(target)} steps={steps}", flush=True)
        for index in range(1, steps + 1):
            pose = start + (target - start) * (index / steps)
            arm.servo_cartesian(pose)
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0: time.sleep(delay)
        actual = arm.read_tcp_pose()
        if float(np.linalg.norm(actual[:3] - target[:3])) > 2.0:
            raise RuntimeError(f"TCP did not reach target within 2 mm; actual={show_array(actual)}")
        print(f"TCP_MOVE_COMPLETE actual={show_array(actual)}", flush=True)
        return 0
    finally:
        try: arm.stop()
        finally: arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
