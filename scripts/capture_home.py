#!/usr/bin/env python3
"""Read current joints and print a YAML snippet; never moves hardware."""
import argparse, math
from _common import add_config_argument, show_array
from xarm7_xhand1 import XArm7, XHand1, load_config

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); add_config_argument(parser)
    parser.add_argument("--device", choices=("arm", "hand", "both"), default="both")
    args = parser.parse_args(); config = load_config(args.config)
    if args.device in {"arm", "both"}:
        with XArm7(config) as arm: print("xarm_home_position_rad:", show_array(arm.read_state().position_rad))
    if args.device in {"hand", "both"}:
        with XHand1(config) as hand: print("xhand_home_position_deg:", show_array([math.degrees(v) for v in hand.read_state().position_rad]))
    print("Copy the reviewed values into hardware.local.yaml manually.")
    return 0
if __name__ == "__main__": raise SystemExit(main())
