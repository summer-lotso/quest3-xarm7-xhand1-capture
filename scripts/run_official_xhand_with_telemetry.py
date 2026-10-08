#!/usr/bin/env python3
"""Run RobotEra XHand retargeting and publish command/feedback telemetry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import zmq
import yaml


for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return json_value(vars(value))
    return repr(value)


def joint_positions_rad(feedback: dict) -> np.ndarray:
    positions = feedback["data"]["joint_position_dic"]
    result = np.radians([float(positions[f"joint{i}"]) for i in range(12)])
    if result.shape != (12,) or not np.isfinite(result).all():
        raise ValueError("XHand feedback has invalid joint positions")
    return result


def home_command(current: np.ndarray, home: np.ndarray, step_rad: float) -> np.ndarray:
    """One bounded command step toward home; never jump to the full target."""
    if current.shape != (12,) or home.shape != (12,) or not np.isfinite(current).all():
        raise ValueError("XHand home inputs must contain twelve finite joints")
    return current + np.clip(home - current, -step_rad, step_rad)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_meta_quest_usb.yaml")
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--telemetry-bind", default="tcp://*:49511")
    parser.add_argument("--relay-zmq", default="tcp://127.0.0.1:49510")
    parser.add_argument("--home-status-bind", default="tcp://*:49515")
    parser.add_argument("--hardware-config", default=str(Path(__file__).resolve().parents[1] / "configs/hardware.local.yaml"))
    parser.add_argument("--input-timeout-s", type=float, default=0.25)
    parser.add_argument("--home-step-rad", type=float, default=0.008)
    # j0's mechanical stall point drifts; measured offsets against a fixed home
    # target have reached 4.3 deg (0.075 rad), so a tighter tolerance times out.
    # The settle hold below keeps the held pose at what was actually reached.
    parser.add_argument("--home-tolerance-rad", type=float, default=0.09)
    parser.add_argument("--home-timeout-s", type=float, default=30.0)
    args = parser.parse_args()
    if min(args.rate_hz, args.input_timeout_s, args.home_step_rad, args.home_tolerance_rad, args.home_timeout_s) <= 0:
        raise ValueError("Rate and safety timeouts must be positive")

    hardware = yaml.safe_load(Path(args.hardware_config).read_text())
    home = np.radians(np.asarray(hardware["xhand_home_position_deg"], dtype=np.float64))
    if home.shape != (12,) or not np.isfinite(home).all():
        raise ValueError("xhand_home_position_deg must have twelve finite joints")
    max_step = float(hardware["xhand_max_step_rad"])
    if args.home_step_rad > max_step * 0.75:
        raise ValueError("home step exceeds configured XHand motion safeguard")

    from xhand_tele_ops import XHandTeleOps

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM, 8)
    publisher.bind(args.telemetry_bind)
    status = context.socket(zmq.PUB)
    status.setsockopt(zmq.SNDHWM, 8)
    status.bind(args.home_status_bind)
    relay = context.socket(zmq.SUB)
    relay.setsockopt(zmq.SUBSCRIBE, b"hand_data")
    relay.setsockopt(zmq.SUBSCRIBE, b"teleop_control")
    relay.setsockopt(zmq.SUBSCRIBE, b"relay_status")
    relay.setsockopt(zmq.RCVHWM, 512)
    relay.connect(args.relay_zmq)
    node = None
    sample_count = 0
    last_hand_at = None
    last_relay_status_at = None
    last_reconnect_at = 0.0
    tracking_allowed = False  # A fresh, explicit episode start is required.
    home_requested = True
    home_verified = False
    home_started_at = time.monotonic()
    commanded_home = None
    home_fault = None
    last_status_at = 0.0
    # Level-triggered mirror of the relay state machine.  hand_data floods this
    # subscription at ~90 Hz, so a single teleop_control edge can be dropped;
    # tracking and homing must also be derivable from relay_status alone.
    topic_counts = {b"hand_data": 0, b"teleop_control": 0, b"relay_status": 0}
    relay_armed = None
    relay_returning = False
    relay_episode = None
    homed_episode = None
    blocked_by = "startup"
    retarget_warm = False
    last_debug_at = 0.0
    try:
        node = XHandTeleOps(args.config)
        hand_type = node.get_hand_type("hand_a")
        serial = node.get_serial_number("hand_a")
        print(f"XHAND_READY type={hand_type} serial={serial} telemetry={args.telemetry_bind}", flush=True)
        period = 1.0 / args.rate_hz
        next_tick = time.perf_counter()

        def request_home(reason: str, episode: int | None) -> None:
            """Arm one bounded return-to-home cycle; idempotent per episode."""
            nonlocal tracking_allowed, home_requested, home_verified
            nonlocal home_started_at, commanded_home, home_fault, homed_episode
            # The relay keeps returning_home set until the arm and the recorder
            # are ready too, so deduplicate on the episode number alone.
            if homed_episode == episode:
                return
            homed_episode = episode
            tracking_allowed = False
            home_requested = True
            home_verified = False
            home_started_at = time.monotonic()
            commanded_home = None
            home_fault = None
            print(f"XHAND_RETURN_HOME_REQUESTED {reason}", flush=True)

        while True:
            while relay.poll(0, zmq.POLLIN):
                parts = relay.recv_multipart()
                if len(parts) != 2:
                    continue
                try:
                    packet = json.loads(parts[1])
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                topic = parts[0]
                if topic in topic_counts:
                    topic_counts[topic] += 1
                if topic == b"relay_status":
                    last_relay_status_at = time.monotonic()
                    relay_armed = packet.get("armed")
                    relay_returning = bool(packet.get("returning_home"))
                    relay_episode = packet.get("episode")
                elif topic == b"hand_data" and packet.get("handedness") == "left":
                    last_hand_at = time.monotonic()
                elif topic == b"teleop_control":
                    event = packet.get("event")
                    last_relay_status_at = time.monotonic()
                    if event in ("complete", "fault", "abort"):
                        tracking_allowed = False
                    elif event in ("save", "discard"):
                        request_home(f"event={event}", packet.get("episode"))
                    elif event == "start":
                        tracking_allowed = True
            now = time.monotonic()
            relay_link_ready = last_relay_status_at is not None and now - last_relay_status_at <= 1.0
            if relay_link_ready:
                # Prefer the relay's level state over remembered edges.
                if relay_armed is not None:
                    tracking_allowed = bool(relay_armed)
                if relay_returning and homed_episode != relay_episode:
                    request_home(f"relay returning_home episode={relay_episode}", relay_episode)
            else:
                if tracking_allowed:
                    print("XHAND_RELAY_LOST: hand tracking paused until a new episode start", flush=True)
                tracking_allowed = False
                if now - last_reconnect_at > 2.0:
                    relay.disconnect(args.relay_zmq)
                    relay.connect(args.relay_zmq)
                    last_reconnect_at = now
            state = "ready" if home_verified else "hold"
            if home_requested and home_fault is None:
                state = "returning"
                blocked_by = "returning to the safe hand pose"
                try:
                    if now - home_started_at > args.home_timeout_s:
                        raise TimeoutError("XHand home timed out")
                    feedback = node.get_hand_full_info("hand_a", force_update=True, is_print=False)
                    measured = joint_positions_rad(feedback)
                    max_error = float(np.max(np.abs(home - measured)))
                    if max_error <= args.home_tolerance_rad:
                        home_requested = False
                        home_verified = True
                        commanded_home = None
                        state = "ready"
                        # Hold what was actually reached: keeping the unreachable
                        # home target would apply stall torque while idle.
                        node.send_data_xhand({"left": measured.tolist(), "right": None}, control_paras=None, is_print=False)
                        print(
                            f"XHAND_RETURN_HOME_COMPLETE max_error_deg={np.degrees(max_error):.2f} "
                            "settle_hold_at_measured=True",
                            flush=True,
                        )
                    else:
                        if commanded_home is None:
                            commanded_home = measured.copy()
                        commanded_home = home_command(commanded_home, home, args.home_step_rad)
                        node.send_data_xhand({"left": commanded_home.tolist(), "right": None}, control_paras=None, is_print=False)
                except Exception as exc:
                    home_fault = f"{type(exc).__name__}: {exc}"
                    state = "fault"
                    print(f"XHAND_RETURN_HOME_FAULT: {home_fault}", flush=True)
            elif home_fault is not None:
                state = "fault"
                blocked_by = home_fault
            elif not relay_link_ready:
                blocked_by = "relay_status is stale"
            elif not tracking_allowed and retarget_warm:
                blocked_by = f"idle: relay has not armed this episode (armed={relay_armed})"
            elif last_hand_at is None:
                blocked_by = "no left hand_data has ever arrived"
            elif now - last_hand_at > args.input_timeout_s:
                blocked_by = f"left hand_data is stale by {now - last_hand_at:.2f}s"
            else:
                # The vendor retarget JIT-compiles a CasADi kernel on its first
                # call, which takes far longer than the recorder's startup
                # grace.  Run the whole chain whenever fresh hand data exists,
                # but command the hand only once the relay has armed.
                source = node.get_orign_data_from_meta_quest(is_print=False)
                if source is None:
                    state = "hold"
                    blocked_by = "vendor get_orign_data_from_meta_quest returned None"
                else:
                    retarget_started = time.perf_counter()
                    target = node.retarget_data_meta_quest(source, is_print=False)
                    if not retarget_warm:
                        retarget_warm = True
                        print(
                            f"XHAND_RETARGET_WARMUP elapsed_s={time.perf_counter() - retarget_started:.2f} "
                            f"target_valid={target is not None}",
                            flush=True,
                        )
                    if target is None:
                        state = "hold"
                        blocked_by = "vendor retarget_data_meta_quest returned None"
                    elif not tracking_allowed:
                        state = "warmup"
                        blocked_by = "retarget warm; waiting for the relay to arm this episode"
                    else:
                        state = "tracking"
                        blocked_by = ""
                        node.send_data_xhand(target, control_paras=None, is_print=False)
                        feedback = node.get_hand_full_info("hand_a", force_update=False, is_print=False)
                        payload = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "monotonic_time": time.monotonic(),
                            "target": json_value(target),
                            "feedback": json_value(feedback),
                        }
                        try:
                            publisher.send_multipart(
                                [b"xhand_telemetry", json.dumps(payload, separators=(",", ":")).encode()],
                                flags=zmq.NOBLOCK,
                            )
                        except zmq.Again:
                            pass
                        sample_count += 1
                        if sample_count == 1:
                            print("XHAND_TELEMETRY_SCHEMA " + json.dumps(payload, ensure_ascii=False)[:4000], flush=True)
                        elif sample_count % int(max(1, args.rate_hz * 60)) == 0:
                            print(f"XHAND_TELEMETRY samples={sample_count}", flush=True)
            if now - last_status_at >= 0.25:
                status_payload = {
                    "state": state,
                    "reason": home_fault or state,
                    "monotonic_time": now,
                    "relay_link_ready": relay_link_ready,
                    "retarget_warm": retarget_warm,
                }
                try:
                    status.send_multipart([b"xhand_home_status", json.dumps(status_payload).encode()], flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
                last_status_at = now
            # Waiting for the operator to arm, and healthy tracking, are both
            # expected; only a real blockage deserves one line per second.
            actionable = bool(blocked_by) and not blocked_by.startswith("idle:")
            if now - last_debug_at >= (1.0 if actionable else 10.0):
                last_debug_at = now
                print("XHAND_LINK " + json.dumps({
                    "state": state,
                    "blocked_by": blocked_by,
                    "tracking_allowed": tracking_allowed,
                    "retarget_warm": retarget_warm,
                    "relay_link_ready": relay_link_ready,
                    "relay_armed": relay_armed,
                    "relay_episode": relay_episode,
                    "hand_age_s": None if last_hand_at is None else round(now - last_hand_at, 3),
                    "samples": sample_count,
                    "topics": {topic.decode(): count for topic, count in topic_counts.items()},
                }, separators=(",", ":")), flush=True)
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        return 0
    finally:
        if node is not None:
            node.shuntdown_meta_quest()
        relay.close(linger=0)
        status.close(linger=0)
        publisher.close(linger=0)
        context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
