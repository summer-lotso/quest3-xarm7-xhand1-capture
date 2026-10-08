#!/usr/bin/env python3
"""Read arm and/or hand state without issuing motion commands."""
import argparse
from _common import add_config_argument, show_array
from xarm7_xhand1 import XArm7, XHand1, load_config

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--device", choices=("arm", "hand", "both"), default="both")
    args = parser.parse_args(); config = load_config(args.config)
    if args.device in {"arm", "both"}:
        with XArm7(config) as arm:
            state = arm.read_state()
            print("xArm position_rad:", show_array(state.position_rad))
            print("xArm velocity_rad_s:", show_array(state.velocity_rad_s))
            print("xArm effort:", show_array(state.effort))
    if args.device in {"hand", "both"}:
        with XHand1(config) as hand:
            state = hand.read_state()
            print("XHand position_rad:", show_array(state.position_rad))
            print("XHand torque_raw:", show_array(state.torque_raw))
            print("XHand temperature_raw:", show_array(state.temperature_raw))
    print("Read-only check complete; no motion command was issued.")
    return 0
if __name__ == "__main__": raise SystemExit(main())
