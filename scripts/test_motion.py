#!/usr/bin/env python3
"""Operator-confirmed hold, powerless, or single-joint low-speed test."""
import argparse
from _common import add_config_argument, require_motion_confirmation, show_array
from xarm7_xhand1 import XArm7, XHand1, load_config

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); add_config_argument(parser)
    parser.add_argument("device", choices=("arm", "hand")); parser.add_argument("operation", choices=("hold", "jog", "powerless"))
    parser.add_argument("--joint", type=int); parser.add_argument("--delta-rad", type=float)
    parser.add_argument("--return-to-start", action="store_true", help="After a jog, command the measured starting pose again")
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--confirm")
    args = parser.parse_args(); require_motion_confirmation(args); config = load_config(args.config)
    if args.operation == "powerless" and args.device != "hand": parser.error("powerless applies only to hand")
    if args.operation == "jog" and (args.joint is None or args.delta_rad is None): parser.error("jog requires --joint and --delta-rad")
    if args.device == "arm":
        if args.joint is not None and not 0 <= args.joint < 7: parser.error("arm --joint must be 0..6")
        with XArm7(config, allow_motion=True) as device:
            before = device.read_state().position_rad.copy()
            if args.operation == "hold": device.hold()
            else:
                target = before.copy(); target[args.joint] += args.delta_rad; device.move_joints(target)
            after = device.read_state().position_rad.copy()
            print("before_rad:", show_array(before)); print("after_rad:", show_array(after))
            if args.operation == "jog" and args.return_to_start:
                device.move_joints(before)
                print("returned_rad:", show_array(device.read_state().position_rad))
    else:
        if args.joint is not None and not 0 <= args.joint < 12: parser.error("hand --joint must be 0..11")
        with XHand1(config, allow_motion=True) as device:
            before = device.read_state().position_rad.copy()
            if args.operation == "hold": device.hold()
            elif args.operation == "powerless": device.powerless()
            else:
                target = before.copy(); target[args.joint] += args.delta_rad; device.move_joints(target)
            after = device.read_state().position_rad.copy()
            print("before_rad:", show_array(before)); print("after_rad:", show_array(after))
            if args.operation == "jog" and args.return_to_start:
                device.move_joints(before)
                print("returned_rad:", show_array(device.read_state().position_rad))
    return 0
if __name__ == "__main__": raise SystemExit(main())
