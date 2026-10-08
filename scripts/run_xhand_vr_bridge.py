#!/usr/bin/env python3
"""Receive localhost VR commands and drive XHand with fail-closed limits."""

from __future__ import annotations

import argparse
import json
import socket
import time

import numpy as np

from _common import add_config_argument, require_motion_confirmation, show_array
from xarm7_xhand1 import XHand1, load_config


def decode_target(packet: bytes) -> tuple[int, bool, np.ndarray | None]:
    payload = json.loads(packet.decode("utf-8"))
    if payload.get("version") != 1:
        raise ValueError("Unsupported command packet version")
    seq = int(payload["seq"])
    enabled = bool(payload.get("enabled", False))
    if not enabled:
        return seq, False, None
    target = np.asarray(payload.get("positions_rad"), dtype=np.float64)
    if target.shape != (XHand1.DOF,) or not np.isfinite(target).all():
        raise ValueError("Command target must contain 12 finite positions")
    if np.any(target < XHand1.JOINT_LIMITS_RAD[:, 0]) or np.any(
        target > XHand1.JOINT_LIMITS_RAD[:, 1]
    ):
        raise ValueError("Command target exceeds XHand limits")
    return seq, True, target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout-s", type=float, default=0.35)
    parser.add_argument("--step-rad", type=float, default=0.025)
    parser.add_argument("--lead-rad", type=float, default=0.12)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    if args.bind not in {"127.0.0.1", "localhost"}:
        raise ValueError("The first commissioning bridge must bind to localhost")
    if args.timeout_s <= 0 or args.step_rad <= 0 or args.lead_rad <= 0:
        raise ValueError("Timeout, step, and lead must be positive")

    require_motion_confirmation(args)
    config = load_config(args.config)
    step_rad = min(args.step_rad, config.xhand_max_step_rad * 0.9)
    if args.lead_rad >= config.xhand_max_tracking_error_rad:
        raise ValueError(
            "--lead-rad must be smaller than xhand_max_tracking_error_rad"
        )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.05)
    print(f"BRIDGE_READY udp={args.bind}:{args.port} step_rad={step_rad}", flush=True)

    with XHand1(config, allow_motion=True) as hand:
        commanded = np.clip(
            hand.read_state().position_rad,
            XHand1.JOINT_LIMITS_RAD[:, 0],
            XHand1.JOINT_LIMITS_RAD[:, 1],
        )
        hand.hold()
        last_packet_time = time.monotonic()
        last_seq = -1
        holding = True
        faulted = False
        command_count = 0
        try:
            while True:
                try:
                    packet, _ = sock.recvfrom(4096)
                except socket.timeout:
                    if not holding and time.monotonic() - last_packet_time > args.timeout_s:
                        hand.hold()
                        commanded = np.clip(
                            hand.read_state().position_rad,
                            XHand1.JOINT_LIMITS_RAD[:, 0],
                            XHand1.JOINT_LIMITS_RAD[:, 1],
                        )
                        holding = True
                        print("BRIDGE_HOLD packet timeout", flush=True)
                    continue

                seq, enabled, target = decode_target(packet)
                if seq <= last_seq:
                    continue
                last_seq = seq
                last_packet_time = time.monotonic()
                if not enabled or target is None:
                    if not holding:
                        hand.hold()
                        commanded = np.clip(
                            hand.read_state().position_rad,
                            XHand1.JOINT_LIMITS_RAD[:, 0],
                            XHand1.JOINT_LIMITS_RAD[:, 1],
                        )
                        holding = True
                        print("BRIDGE_HOLD operator pause", flush=True)
                    if faulted:
                        faulted = False
                        print("BRIDGE_FAULT_CLEARED; enable again to resume", flush=True)
                    continue

                if faulted:
                    continue

                desired = commanded + np.clip(
                    target - commanded, -step_rad, step_rad
                )
                measured = hand.read_state().position_rad
                commanded = np.clip(
                    desired,
                    np.maximum(
                        XHand1.JOINT_LIMITS_RAD[:, 0], measured - args.lead_rad
                    ),
                    np.minimum(
                        XHand1.JOINT_LIMITS_RAD[:, 1], measured + args.lead_rad
                    ),
                )
                commanded = np.clip(
                    commanded,
                    XHand1.JOINT_LIMITS_RAD[:, 0],
                    XHand1.JOINT_LIMITS_RAD[:, 1],
                )
                try:
                    hand.move_joints(commanded)
                except RuntimeError as exc:
                    hand.hold()
                    commanded = np.clip(
                        hand.read_state().position_rad,
                        XHand1.JOINT_LIMITS_RAD[:, 0],
                        XHand1.JOINT_LIMITS_RAD[:, 1],
                    )
                    holding = True
                    faulted = True
                    print(f"BRIDGE_FAULT {exc}", flush=True)
                    continue
                holding = False
                command_count += 1
                if command_count == 1 or command_count % 30 == 0:
                    print(
                        f"BRIDGE_COMMAND count={command_count} target_deg="
                        f"{show_array(np.degrees(target))}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            return 0
        finally:
            hand.hold()
            sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
