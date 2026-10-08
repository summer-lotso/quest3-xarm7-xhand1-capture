#!/usr/bin/env python3
"""Drive xArm7 from RobotEra's official Meta Quest ZMQ wrist stream."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
import time

import numpy as np
import zmq

from _common import add_config_argument, require_motion_confirmation, show_array
from xarm7_xhand1 import XArm7, load_config
from xarm7_xhand1.official_vr import (
    DeadmanClutch,
    PinchClutch,
    RelativeWristMapper,
    TrackingLimitError,
    decode_gamepad_trigger,
    decode_hand_packet,
    decode_pinch_distance,
    limit_servo_step,
)


def receive_wrist(socket: zmq.Socket, hand: str, timeout_s: float, on_wait=None):
    """Wait for a wrist without losing control events received during startup."""
    deadline = time.monotonic() + timeout_s
    pending_controls = []
    while time.monotonic() < deadline:
        remaining_s = max(0.001, min(0.1, deadline - time.monotonic()))
        wrist, _, controls = receive_inputs(socket, hand, hand, "session", remaining_s)
        pending_controls.extend(controls)
        if wrist is not None:
            return wrist, pending_controls
        if on_wait is not None:
            on_wait()
    raise TimeoutError(f"No fresh {hand} hand_data packet within {timeout_s:.2f}s")


def receive_inputs(socket: zmq.Socket, hand: str, clutch_hand: str, clutch_mode: str, timeout_s: float):
    """Receive and drain the latest wrist and controller updates."""
    if not socket.poll(max(1, int(timeout_s * 1000)), zmq.POLLIN):
        return None, None, []
    latest_wrist = None
    latest_trigger = None
    control_events = []
    while True:
        parts = socket.recv_multipart()
        if len(parts) != 2:
            raise ValueError("Official ZMQ packet must contain topic and JSON payload")
        if parts[0] == b"teleop_control":
            payload = json.loads(parts[1])
            if isinstance(payload, dict) and isinstance(payload.get("event"), str):
                control_events.append(payload)
            if not socket.poll(0, zmq.POLLIN):
                break
            continue
        wrist = decode_hand_packet(*parts, hand)
        if clutch_mode == "pinch":
            clutch_value = decode_pinch_distance(*parts, clutch_hand)
        elif clutch_mode == "trigger":
            clutch_value = decode_gamepad_trigger(*parts, clutch_hand)
        else:
            # Session motion is gated by explicit teleop_control events.  Hand
            # packets may also be present during neutral XHand calibration.
            clutch_value = None
        if wrist is not None:
            latest_wrist = wrist
        if clutch_value is not None:
            latest_trigger = clutch_value
        if not socket.poll(0, zmq.POLLIN):
            break
    return latest_wrist, latest_trigger, control_events


def publish_home_status(socket: zmq.Socket, state: str, reason: str, **fields) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "reason": reason,
        **fields,
    }
    try:
        socket.send_multipart(
            [b"xarm_home_status", json.dumps(payload, separators=(",", ":")).encode()],
            flags=zmq.NOBLOCK,
        )
    except zmq.Again:
        pass


def clamp_cartesian_boundary(target, boundary, margin_mm: float) -> np.ndarray:
    """Keep a servo target inside the controller's reduced-mode box."""
    result = np.asarray(target, dtype=np.float64).copy()
    if boundary is None:
        return result
    for axis in range(3):
        maximum = float(boundary[axis * 2])
        minimum = float(boundary[axis * 2 + 1])
        low, high = minimum + margin_mm, maximum - margin_mm
        if low >= high:
            raise ValueError(f"Reduced-mode boundary is narrower than {2 * margin_mm:.1f} mm")
        result[axis] = np.clip(result[axis], low, high)
    return result


def check_servo_tracking_error(target, measured, max_translation_mm: float, max_rotation_rad: float) -> None:
    """Stop when xArm feedback falls too far behind the commanded TCP."""
    target = np.asarray(target, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    if target.shape != (6,) or measured.shape != (6,) or not np.isfinite(target).all() or not np.isfinite(measured).all():
        raise ValueError("xArm TCP target and feedback must contain six finite values")
    translation = float(np.linalg.norm(target[:3] - measured[:3]))
    # Near pitch=+/-pi/2, xArm can report a different roll/yaw pair for the
    # same physical orientation. Compare rotations instead of Euler fields.
    target_rotation = RelativeWristMapper._rpy_to_matrix(target[3:])
    measured_rotation = RelativeWristMapper._rpy_to_matrix(measured[3:])
    cosine = np.clip((np.trace(target_rotation @ measured_rotation.T) - 1.0) / 2.0, -1.0, 1.0)
    rotation = float(np.arccos(cosine))
    if translation > max_translation_mm or rotation > max_rotation_rad:
        raise RuntimeError(
            f"xArm servo tracking error: translation={translation:.1f} mm, "
            f"rotation={rotation:.3f} rad; limits={max_translation_mm:.1f} mm/{max_rotation_rad:.3f} rad"
        )


def return_arm_home(
    arm: XArm7,
    target_rad,
    status_socket: zmq.Socket,
    *,
    timeout_s: float,
    settle_s: float,
    speed_rad_s: float,
    acceleration_rad_s2: float,
) -> np.ndarray:
    """Move to the configured safe joint pose, then require a stable dwell."""
    target = np.asarray(target_rad, dtype=np.float64)
    started = time.monotonic()
    # A cleared controller error leaves the servos disabled; set_servo_angle
    # would return code 1 until the arm is re-enabled. Idempotent otherwise.
    arm.ensure_enabled()
    initial = arm.read_state().position_rad
    initial_error = max(float(np.max(np.abs(target - initial))), 1e-9)
    last_report = 0.0
    publish_home_status(status_socket, "returning", "xArm returning to safe position")
    while True:
        now = time.monotonic()
        if now - started > timeout_s:
            raise TimeoutError(f"xArm safe-position return exceeded {timeout_s:.1f}s")
        current = arm.read_state().position_rad
        error = target - current
        remaining = float(np.max(np.abs(error)))
        if remaining <= 0.003:
            break
        # Use most of the validated per-command joint limit while retaining
        # a margin for feedback/command rounding. Fewer stop-and-wait moves
        # shorten homing without raising the configured joint velocity.
        step = np.clip(error, -arm.config.xarm_max_step_rad * 0.9, arm.config.xarm_max_step_rad * 0.9)
        arm.move_joints(current + step, wait=True, speed_rad_s=speed_rad_s, acceleration_rad_s2=acceleration_rad_s2)
        if now - last_report >= 0.8:
            elapsed = now - started
            progress = np.clip(1.0 - remaining / initial_error, 0.0, 0.999)
            estimated = elapsed * (1.0 - progress) / progress if progress >= 0.02 else None
            fields = {"remaining_rad": remaining, "elapsed_s": elapsed}
            if estimated is not None:
                fields["estimated_remaining_s"] = estimated + settle_s
            publish_home_status(status_socket, "returning", "xArm returning to safe position", **fields)
            print(f"RETURN_HOME_PROGRESS remaining_rad={remaining:.4f}", flush=True)
            last_report = now
    arm.stop()
    print(f"RETURN_HOME_SETTLING: {settle_s:.1f}s", flush=True)
    settle_deadline = time.monotonic() + settle_s
    while time.monotonic() < settle_deadline:
        remaining_s = max(0.0, settle_deadline - time.monotonic())
        publish_home_status(
            status_socket,
            "returning",
            "xArm at safe position; waiting for stability",
            estimated_remaining_s=remaining_s,
        )
        time.sleep(min(0.25, remaining_s))
    final = arm.read_state().position_rad
    final_error = float(np.max(np.abs(target - final)))
    if final_error > 0.01:
        raise RuntimeError(f"xArm moved after safe-position return; error={final_error:.4f} rad")
    publish_home_status(status_socket, "ready", "safe position stable; next calibration enabled")
    print("RETURN_HOME_COMPLETE: next calibration enabled", flush=True)
    return arm.read_tcp_pose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--zmq", default="tcp://127.0.0.1:49510")
    parser.add_argument("--telemetry-bind", default="tcp://*:49512")
    parser.add_argument("--home-status-bind", default="tcp://*:49513")
    parser.add_argument("--hand", choices=("left", "right"), default="left")
    parser.add_argument("--clutch-mode", choices=("pinch", "trigger", "session"), default="pinch")
    parser.add_argument("--clutch-hand", choices=("left", "right"), default="right")
    parser.add_argument("--clutch-threshold", type=float, default=0.5)
    parser.add_argument("--clutch-timeout-s", type=float, default=0.6)
    parser.add_argument("--pinch-engage-m", type=float, default=0.035)
    parser.add_argument("--pinch-release-m", type=float, default=0.05)
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--timeout-s", type=float, default=0.35)
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--max-translation-mm", type=float, default=400.0)
    parser.add_argument("--max-rotation-rad", type=float, default=2.8)
    parser.add_argument("--max-step-mm", type=float, default=1.0)
    parser.add_argument("--max-orientation-step-rad", type=float, default=0.01)
    parser.add_argument("--max-tracking-error-mm", type=float, default=15.0)
    parser.add_argument("--max-tracking-error-rad", type=float, default=0.2)
    parser.add_argument("--boundary-margin-mm", type=float, default=2.0)
    parser.add_argument(
        "--reject-hold-timeout-s",
        type=float,
        default=2.0,
        help="Fail only after the controller refuses servo commands for this long; single refusals revert and hold.",
    )
    parser.add_argument("--home-timeout-s", type=float, default=180.0)
    parser.add_argument("--home-settle-s", type=float, default=1.0)
    parser.add_argument("--home-speed-rad-s", type=float, default=0.12)
    parser.add_argument("--home-acceleration-rad-s2", type=float, default=0.25)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    if (
        args.rate_hz <= 0
        or args.timeout_s <= 0
        or args.startup_timeout_s <= 0
        or args.home_timeout_s <= 0
        or args.home_settle_s < 0
        or args.home_speed_rad_s <= 0
        or args.home_acceleration_rad_s2 <= 0
    ):
        raise ValueError("Rate and timeouts must be positive")
    if args.max_step_mm is not None and args.max_step_mm <= 0:
        raise ValueError("--max-step-mm must be positive")
    if args.max_orientation_step_rad is not None and args.max_orientation_step_rad <= 0:
        raise ValueError("--max-orientation-step-rad must be positive")
    if args.max_tracking_error_mm <= 0 or args.max_tracking_error_rad <= 0:
        raise ValueError("Servo tracking error limits must be positive")
    if args.boundary_margin_mm < 0:
        raise ValueError("--boundary-margin-mm must be nonnegative")
    if args.reject_hold_timeout_s <= 0:
        raise ValueError("--reject-hold-timeout-s must be positive")
    if args.execute:
        require_motion_confirmation(args)

    config = load_config(args.config)
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"hand_data")
    subscriber.setsockopt(zmq.SUBSCRIBE, b"arm_preview")
    subscriber.setsockopt(zmq.SUBSCRIBE, b"gamepad_data")
    subscriber.setsockopt(zmq.SUBSCRIBE, b"teleop_control")
    subscriber.setsockopt(zmq.RCVHWM, 10)
    subscriber.connect(args.zmq)
    telemetry = context.socket(zmq.PUB)
    telemetry.setsockopt(zmq.SNDHWM, 8)
    telemetry.bind(args.telemetry_bind)
    home_status = context.socket(zmq.PUB)
    home_status.setsockopt(zmq.SNDHWM, 8)
    home_status.bind(args.home_status_bind)

    arm = None
    try:
        if args.execute:
            arm = XArm7(config, allow_motion=True)
            arm.connect()  # Read-only connection; motion starts on teleop_control/start.
            initial_tcp = arm.read_tcp_pose()
        else:
            initial_tcp = np.zeros(6, dtype=np.float64)
            print("DRY_RUN: xArm is not connected and no motion command will be sent.", flush=True)

        def publish_idle_telemetry() -> None:
            measured_joints = arm.read_state().position_rad.tolist() if arm is not None else None
            measured_tcp = arm.read_tcp_pose().tolist() if arm is not None else None
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "monotonic_time": time.monotonic(),
                "moving": False,
                "target_tcp_mm_rad": initial_tcp.tolist(),
                "joint_position_rad": measured_joints,
                "tcp_pose_mm_rad": measured_tcp,
            }
            try:
                telemetry.send_multipart(
                    [b"xarm_telemetry", json.dumps(payload, separators=(",", ":")).encode()],
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                pass

        print(f"Waiting for official {args.hand} wrist stream at {args.zmq} ...", flush=True)
        try:
            initial_wrist, startup_controls = receive_wrist(
                subscriber, args.hand, args.startup_timeout_s, on_wait=publish_idle_telemetry
            )
        except TimeoutError as exc:
            raise RuntimeError(
                "Official ZMQ server sent no matching hand_data. Start the vendor demo, "
                "open its HTTPS :8010 page in Quest, accept the certificate, enter the "
                "immersive AR session, and keep the selected hand visible."
            ) from exc
        clutch = (
            PinchClutch(args.pinch_engage_m, args.pinch_release_m, args.clutch_timeout_s)
            if args.clutch_mode == "pinch"
            else DeadmanClutch(args.clutch_threshold, args.clutch_timeout_s)
        )
        mapper = None
        commanded = initial_tcp.copy()
        latest_wrist = initial_wrist
        latest_wrist_at = time.monotonic()
        period = 1.0 / args.rate_hz
        next_tick = time.monotonic()
        count = 0
        tracking_hold = False
        boundary_hold = False
        reject_hold = False
        reject_since = 0.0
        last_accepted = commanded.copy()
        clutch_engaged = False
        session_active = False
        servo_active = False
        clutch_instruction = (
            f"pinch {args.clutch_hand} thumb and index finger to move; open them to hold"
            if args.clutch_mode == "pinch"
            else (
                f"hold {args.clutch_hand} controller trigger to move; release it to hold"
                if args.clutch_mode == "trigger"
                else "movement follows the relay keyboard session; stale input holds"
            )
        )
        print(f"CLUTCH_READY: {clutch_instruction}", flush=True)
        step_mm = min(
            config.xarm_max_cartesian_step_mm * 0.95,
            args.max_step_mm if args.max_step_mm is not None else float("inf"),
        )
        orientation_step_rad = min(
            config.xarm_max_orientation_step_rad * 0.95,
            args.max_orientation_step_rad if args.max_orientation_step_rad is not None else float("inf"),
        )
        print(
            f"SERVO_LIMITS: translation={step_mm:.3f} mm/frame "
            f"orientation={orientation_step_rad:.4f} rad/frame rate={args.rate_hz:.1f} Hz",
            flush=True,
        )
        while True:
            wrist, clutch_value, control_events = receive_inputs(
                subscriber,
                args.hand,
                args.clutch_hand,
                args.clutch_mode,
                min(period, 0.02),
            )
            if startup_controls:
                control_events = startup_controls + control_events
                startup_controls = []
            now = time.monotonic()
            if wrist is not None:
                latest_wrist = wrist
                latest_wrist_at = now
            for control in control_events:
                event = control["event"]
                reason = str(control.get("reason", event))
                if event == "start":
                    session_active = True
                    mapper = None
                    clutch_engaged = False
                    tracking_hold = False
                    reject_hold = False
                    if arm is not None and not servo_active:
                        commanded = arm.start_cartesian_servo()
                        servo_active = True
                    print(f"SESSION_START: {reason}", flush=True)
                elif event in {"complete", "fault", "abort"}:
                    session_active = False
                    mapper = None
                    clutch_engaged = False
                    reject_hold = False
                    if arm is not None and servo_active:
                        arm.stop()
                        servo_active = False
                    print(f"SESSION_STOP: event={event} reason={reason}", flush=True)
                elif event in {"save", "discard"}:
                    session_active = False
                    mapper = None
                    clutch_engaged = False
                    reject_hold = False
                    if arm is not None and servo_active:
                        arm.stop()
                        servo_active = False
                    print(f"SESSION_RESET: event={event} reason={reason}", flush=True)
                    if arm is not None:
                        if config.xarm_home_position_rad is None:
                            raise ValueError("xarm_home_position_rad is required for automatic episode reset")
                        try:
                            commanded = return_arm_home(
                                arm,
                                config.xarm_home_position_rad,
                                home_status,
                                timeout_s=args.home_timeout_s,
                                settle_s=args.home_settle_s,
                                speed_rad_s=args.home_speed_rad_s,
                                acceleration_rad_s2=args.home_acceleration_rad_s2,
                            )
                        except Exception as exc:
                            publish_home_status(home_status, "fault", str(exc))
                            raise
                    else:
                        publish_home_status(home_status, "ready", "dry-run safe-position reset complete")
                    next_tick = time.monotonic()
            now = time.monotonic()
            if clutch_value is not None:
                clutch.update(clutch_value, now)
            clutch_active = session_active if args.clutch_mode == "session" else clutch.active(now)
            wrist_fresh = now - latest_wrist_at <= args.timeout_s

            if clutch_active and wrist_fresh and not clutch_engaged:
                mapper = RelativeWristMapper(
                    latest_wrist,
                    commanded,
                    translation_scale=args.translation_scale,
                    max_translation_mm=args.max_translation_mm,
                    max_rotation_rad=args.max_rotation_rad,
                )
                clutch_engaged = True
                tracking_hold = False
                print("CLUTCH_ENGAGED: wrist and TCP target recentered", flush=True)
            elif (not clutch_active or not wrist_fresh) and clutch_engaged:
                reason = "clutch released/stale" if not clutch_active else "wrist tracking stale"
                print(f"CLUTCH_RELEASED: {reason}; holding", flush=True)
                clutch_engaged = False
                mapper = None

            if clutch_engaged and mapper is not None:
                try:
                    desired = mapper.target(latest_wrist)
                    if tracking_hold:
                        print("BRIDGE_RESUME: wrist returned inside the configured envelope", flush=True)
                    tracking_hold = False
                except TrackingLimitError as exc:
                    if not tracking_hold:
                        print(f"BRIDGE_HOLD: {exc}", file=sys.stderr, flush=True)
                    tracking_hold = True
                    desired = commanded
            else:
                desired = commanded
            commanded = limit_servo_step(
                commanded,
                desired,
                step_mm,
                orientation_step_rad,
            )
            if arm is not None and servo_active:
                bounded = clamp_cartesian_boundary(
                    commanded, getattr(arm, "_servo_boundary", None), args.boundary_margin_mm,
                )
                clipped = not np.array_equal(bounded, commanded)
                if clipped and not boundary_hold:
                    print("BRIDGE_BOUNDARY_HOLD: TCP target reached the reduced-mode margin", flush=True)
                boundary_hold = clipped
                commanded = bounded
                # The controller can refuse a single servo command (e.g. a low,
                # tilted grasp pose its servo planner will not accept, which its
                # plain IK query still calls reachable). Revert to the last
                # accepted target and hold instead of tearing the stack down;
                # keep probing every frame and fail only if refusals persist.
                try:
                    arm.servo_cartesian(commanded)
                    if reject_hold:
                        reject_hold = False
                        print("BRIDGE_SERVO_RESUME: controller accepted the command again", flush=True)
                    last_accepted = commanded.copy()
                except RuntimeError as exc:
                    commanded = last_accepted.copy()
                    if not reject_hold:
                        reject_hold = True
                        reject_since = now
                        print(
                            f"BRIDGE_SERVO_REJECTED: {exc}; holding at {show_array(commanded)}",
                            file=sys.stderr, flush=True,
                        )
                    elif now - reject_since > args.reject_hold_timeout_s:
                        raise RuntimeError(
                            f"controller refused servo commands for {now - reject_since:.1f}s; "
                            "bring the hand back inside the reachable workspace"
                        ) from exc
            if arm is not None:
                measured_joints = arm.read_state().position_rad.tolist()
                measured_tcp = arm.read_tcp_pose().tolist()
                if servo_active and not reject_hold:
                    check_servo_tracking_error(
                        commanded, measured_tcp,
                        args.max_tracking_error_mm, args.max_tracking_error_rad,
                    )
            else:
                measured_joints = None
                measured_tcp = None
            telemetry_payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "monotonic_time": now,
                "moving": bool(clutch_engaged and not tracking_hold and not reject_hold),
                "target_tcp_mm_rad": commanded.tolist(),
                "joint_position_rad": measured_joints,
                "tcp_pose_mm_rad": measured_tcp,
            }
            try:
                telemetry.send_multipart(
                    [b"xarm_telemetry", json.dumps(telemetry_payload, separators=(",", ":")).encode()],
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                pass
            count += 1
            if count == 1 or count % 30 == 0:
                state = "MOVE" if clutch_engaged and not tracking_hold else "HOLD"
                print(
                    f"{'COMMAND' if arm else 'DRY_RUN'} count={count} state={state} "
                    f"tcp_mm_rad={show_array(commanded)}",
                    flush=True,
                )
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        return 0
    except (TimeoutError, ValueError, RuntimeError, ConnectionError) as exc:
        if arm is not None:
            publish_home_status(home_status, "fault", str(exc))
        print(f"BRIDGE_FAULT: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if arm is not None:
            try:
                arm.stop()
            finally:
                arm.close()
        subscriber.close(linger=0)
        telemetry.close(linger=0)
        home_status.close(linger=0)
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
