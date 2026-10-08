#!/usr/bin/env python3
"""Print explicit read-only xArm checks before Quest teleoperation."""

from __future__ import annotations

import argparse
import json

from _common import add_config_argument
from xarm7_xhand1 import load_config
from xarm7_xhand1.arm_check import check_arm_readiness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    args = parser.parse_args()
    try:
        report = check_arm_readiness(load_config(args.config))
    except Exception as exc:
        print(f"ARM_CHECK_FAIL: {type(exc).__name__}: {exc}", flush=True)
        return 1
    print("ARM_CHECK " + json.dumps(report, ensure_ascii=False, separators=(",", ":")), flush=True)
    if report["ready_for_new_session"]:
        detail = "software stop, recoverable on session start" if report["state"] == 4 else "controller stationary"
        print(f"ARM_CHECK_PASS: {detail}; no motion command was issued", flush=True)
        return 0
    print("ARM_CHECK_FAIL: " + "; ".join(report["problems"]), flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
