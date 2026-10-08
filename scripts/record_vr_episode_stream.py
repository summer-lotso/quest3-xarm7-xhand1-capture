#!/usr/bin/env python3
"""Record synchronized xArm/XHand/RealSense frames from teleop telemetry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import time
from typing import Any

import numpy as np
import zmq

from _common import add_config_argument
from xarm7_xhand1 import load_config
from xarm7_xhand1.cameras import RealSenseCameras
from xarm7_xhand1.recording import ControlEvent, EpisodeRecorder, EpisodeState, LeRobotSink, NpzEpisodeSink


def numeric_vector(value: Any, size: int) -> np.ndarray | None:
    if isinstance(value, dict):
        values = list(value.values())
        try:
            vector = np.asarray(values, dtype=np.float32)
            if vector.shape == (size,) and np.isfinite(vector).all():
                return vector
        except (TypeError, ValueError):
            pass
        for item in value.values():
            found = numeric_vector(item, size)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        try:
            vector = np.asarray(value, dtype=np.float32)
            if vector.shape == (size,) and np.isfinite(vector).all():
                return vector
        except (TypeError, ValueError):
            pass
        for item in value:
            found = numeric_vector(item, size)
            if found is not None:
                return found
    return None


def preferred_vector(payload: Any, names: tuple[str, ...], size: int) -> np.ndarray | None:
    direct = numeric_vector(payload, size)
    if direct is not None:
        return direct
    if isinstance(payload, dict):
        for name in names:
            if name in payload:
                found = numeric_vector(payload[name], size)
                if found is not None:
                    return found
        for value in payload.values():
            found = preferred_vector(value, names, size)
            if found is not None:
                return found
    return None


def drain(socket: zmq.Socket) -> list[tuple[bytes, dict]]:
    messages = []
    while socket.poll(0, zmq.POLLIN):
        topic, payload = socket.recv_multipart()
        messages.append((topic, json.loads(payload)))
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--root", default="datasets/vr_capture")
    parser.add_argument("--task", required=True)
    parser.add_argument("--format", choices=("lerobot-v2.1", "npz"), default="lerobot-v2.1")
    parser.add_argument("--repo-id", default="local/xarm7_xhand1_vr_official")
    parser.add_argument("--rate-hz", type=float, default=20.0, help="Dataset sampling rate; cameras may capture faster")
    parser.add_argument("--teleop-zmq", default="tcp://127.0.0.1:49510")
    parser.add_argument("--xhand-zmq", default="tcp://127.0.0.1:49511")
    parser.add_argument("--xarm-zmq", default="tcp://127.0.0.1:49512")
    parser.add_argument("--status-bind", default="tcp://*:49514")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--freshness-s", type=float, default=0.25)
    parser.add_argument("--startup-grace-s", type=float, default=2.0)
    args = parser.parse_args()
    config = load_config(args.config)
    rate_hz = float(args.rate_hz)
    if rate_hz <= 0 or args.freshness_s <= 0 or args.startup_grace_s <= 0:
        raise ValueError("Rate and freshness must be positive")
    cameras = None if args.no_cameras else RealSenseCameras(config)
    sink = None
    recorder = None
    context = zmq.Context()
    sockets = []
    for address, topic in (
        (args.teleop_zmq, b"teleop_control"),
        (args.xhand_zmq, b"xhand_telemetry"),
        (args.xarm_zmq, b"xarm_telemetry"),
    ):
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, topic)
        socket.setsockopt(zmq.RCVHWM, 20)
        socket.connect(address)
        sockets.append(socket)
    control_socket, hand_socket, arm_socket = sockets
    control_socket.setsockopt(zmq.SUBSCRIBE, b"relay_status")
    status_socket = context.socket(zmq.PUB)
    status_socket.setsockopt(zmq.SNDHWM, 8)
    status_socket.bind(args.status_bind)
    sockets.append(status_socket)
    latest_hand = latest_arm = None
    last_hand_reconnect_at = last_arm_reconnect_at = 0.0
    last_relay_status_at = None
    last_control_reconnect_at = 0.0
    episode_started = None
    period = 1.0 / rate_hz
    next_tick = time.monotonic()
    try:
        if cameras is not None:
            cameras.connect()
            print(f"CAMERAS_READY names={sorted(set(cameras._pipelines) | set(cameras._captures))}", flush=True)
        if args.format == "lerobot-v2.1":
            if rate_hz != int(rate_hz):
                raise ValueError("LeRobot v2.1 fps must be an integer")
            image_shapes = None
            if cameras is not None:
                # RealSense devices can need over a second to deliver the
                # first synchronized color frame after all pipelines start.
                first_images = cameras.read(timeout_ms=3000)
                image_shapes = {name: tuple(image.shape) for name, image in first_images.items()}
            sink = LeRobotSink.create(
                repo_id=args.repo_id,
                root=args.root,
                fps=int(rate_hz),
                image_shapes=image_shapes,
            )
            print(
                f"LEROBOT_DATASET_READY version=v2.1 repo_id={args.repo_id} root={sink.dataset.root} "
                f"episodes={sink.dataset.meta.total_episodes}",
                flush=True,
            )
        else:
            sink = NpzEpisodeSink(args.root)
        recorder = EpisodeRecorder(sink, task=args.task)
        print(f"RECORDER_READY format={args.format} root={args.root} task={args.task!r}", flush=True)
        while True:
            for topic, payload in drain(control_socket):
                if topic == b"relay_status":
                    last_relay_status_at = time.monotonic()
                    continue
                last_relay_status_at = time.monotonic()
                event = ControlEvent(
                    str(payload.get("event", "")),
                    payload.get("episode"),
                    str(payload.get("reason", "")),
                )
                state = recorder.handle(event)
                if event.event == "start":
                    episode_started = time.monotonic()
                elif state != EpisodeState.RECORDING:
                    episode_started = None
                print(f"RECORDER_EVENT event={event.event} episode={event.episode} state={state.value}", flush=True)
            hand_messages = drain(hand_socket)
            arm_messages = drain(arm_socket)
            if hand_messages:
                latest_hand = hand_messages[-1][1]
            if arm_messages:
                latest_arm = arm_messages[-1][1]
            now = time.monotonic()
            relay_link_ready = last_relay_status_at is not None and now - last_relay_status_at <= 1.0
            if not relay_link_ready and now - last_control_reconnect_at > 2.0:
                control_socket.disconnect(args.teleop_zmq)
                control_socket.connect(args.teleop_zmq)
                last_control_reconnect_at = now
            # ZeroMQ SUB does not always reattach after a long-lived publisher
            # has been restarted. Rebuild stale transports while idle as well
            # as during the startup grace period, and never reuse old samples.
            if latest_hand is None or now - float(latest_hand["monotonic_time"]) > 0.5:
                if now - last_hand_reconnect_at > 2.0:
                    hand_socket.disconnect(args.xhand_zmq)
                    hand_socket.connect(args.xhand_zmq)
                    last_hand_reconnect_at = now
                latest_hand = None
            if latest_arm is None or now - float(latest_arm["monotonic_time"]) > 0.5:
                if now - last_arm_reconnect_at > 2.0:
                    arm_socket.disconnect(args.xarm_zmq)
                    arm_socket.connect(args.xarm_zmq)
                    last_arm_reconnect_at = now
                latest_arm = None
            if recorder.state == EpisodeState.RECORDING:
                if not relay_link_ready and now - float(episode_started) > args.startup_grace_s:
                    recorder.handle(ControlEvent("fault", recorder.episode, "relay control heartbeat lost"))
                    episode_started = None
                    print("RECORDER_FAULT relay control heartbeat lost", flush=True)
                    continue
                hand_age = float("inf") if latest_hand is None else now - float(latest_hand["monotonic_time"])
                arm_age = float("inf") if latest_arm is None else now - float(latest_arm["monotonic_time"])
                hand_target = None if latest_hand is None else preferred_vector(latest_hand.get("target"), ("target", "joint_position", "position"), 12)
                hand_position_deg = None if latest_hand is None else preferred_vector(latest_hand.get("feedback"), ("joint_position_dic", "joint_position", "position"), 12)
                hand_position = None if hand_position_deg is None else np.radians(hand_position_deg).astype(np.float32)
                ready = (
                    hand_age <= args.freshness_s
                    and arm_age <= args.freshness_s
                    and hand_target is not None
                    and hand_position is not None
                    and latest_arm.get("joint_position_rad") is not None
                    and latest_arm.get("tcp_pose_mm_rad") is not None
                )
                if not ready and now - float(episode_started) <= args.startup_grace_s:
                    pass
                elif not ready:
                    recorder.handle(ControlEvent("fault", recorder.episode, "telemetry stale or schema invalid"))
                    episode_started = None
                    print(
                        f"RECORDER_FAULT hand_age={hand_age:.3f} arm_age={arm_age:.3f} "
                        f"target={hand_target is not None} feedback={hand_position is not None}",
                        flush=True,
                    )
                else:
                    frame = {
                        "action": np.concatenate((np.asarray(latest_arm["target_tcp_mm_rad"], np.float32), hand_target)),
                        "observation.arm_joint_position": np.asarray(latest_arm["joint_position_rad"], np.float32),
                        "observation.arm_tcp_pose": np.asarray(latest_arm["tcp_pose_mm_rad"], np.float32),
                        "observation.hand_joint_position": hand_position,
                    }
                    if cameras is not None:
                        for name, image in cameras.read(timeout_ms=max(100, int(period * 2000))).items():
                            frame[f"observation.images.{name}"] = image
                    recorder.add_frame(frame, timestamp=now - float(episode_started))
                    if recorder.frame_count == 1 or recorder.frame_count % int(rate_hz * 5) == 0:
                        print(f"RECORDER_FRAMES episode={recorder.episode} count={recorder.frame_count}", flush=True)
            try:
                status_socket.send_multipart(
                    [b"recorder_status", json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "state": recorder.state.value,
                        "episode": recorder.episode,
                        "frames": recorder.frame_count,
                        "relay_link_ready": relay_link_ready,
                    }, separators=(",", ":")).encode()],
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                pass
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        if recorder is not None and recorder.state in {EpisodeState.RECORDING, EpisodeState.PENDING}:
            recorder.handle(ControlEvent("abort", recorder.episode, "recorder interrupted"))
        return 0
    finally:
        if cameras is not None:
            cameras.close()
        if isinstance(sink, LeRobotSink):
            sink.close()
        if isinstance(sink, NpzEpisodeSink):
            sink.clear_episode_buffer()
        for socket in sockets:
            socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
