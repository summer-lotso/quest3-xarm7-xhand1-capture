import sys
from pathlib import Path
import numpy as np, pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from xarm7_xhand1.config import HardwareConfig
from xarm7_xhand1.arm import XArm7
from xarm7_xhand1.arm_check import evaluate_arm_readiness
from xarm7_xhand1.policy_client import PolicyClient, packb, unpackb
from xarm7_xhand1.hand import XHand1
from xarm7_xhand1.official_vr import DeadmanClutch, PinchClutch, RelativeWristMapper, TrackingLimitError, WristPose, decode_gamepad_trigger, decode_hand_packet, decode_pinch_distance, limit_servo_step
from usb_webxr_relay import RelayPublisher, convert_hand_poses, parse_volume_event, patched_index, volume_button_request
from xarm7_xhand1.recording import ControlEvent, EpisodeRecorder, EpisodeState, LeRobotSink, NpzEpisodeSink, validate_recording_frame
from record_vr_episode_stream import numeric_vector, preferred_vector
from run_official_vr_xarm_bridge import check_servo_tracking_error, clamp_cartesian_boundary, receive_wrist
from run_official_xhand_with_telemetry import home_command, joint_positions_rad

def test_config_rejects_bad_dimensions():
    with pytest.raises(ValueError): HardwareConfig(xarm_home_position_rad=(0.0,))
    with pytest.raises(ValueError): HardwareConfig(xarm_teleop_ready_position_rad=(0.0,))
    with pytest.raises(ValueError): HardwareConfig(xarm_tcp_offset_mm_rad=(0.0,))
    with pytest.raises(ValueError): HardwareConfig(xarm_tcp_offset_mm_rad=(0.0, 0.0, 0.0, 0.0, np.nan, 0.0))
    with pytest.raises(ValueError): HardwareConfig(xhand_teleop_ready_position_deg=(0.0,))
def test_config_requires_explicit_torque_for_hand_motion():
    with pytest.raises(ValueError): HardwareConfig(xhand_protocol="RS485", xhand_serial_port="/dev/null").require_hand_motion()


def test_arm_readiness_distinguishes_stopped_state_from_missing_safety_boundary():
    report = {
        "state": 4, "error": 0, "warning": 0, "servo_codes": [0] * 7,
        "reduced_mode_enabled": False, "safety_boundary_enabled": True,
        "collision_rebound_enabled": True,
        "tcp_offset_mm_rad": [0, -3, 43.6, 1.570796, -1.570796, 0],
        "joint_velocity_rad_s": [0] * 7,
        "boundary_mm": [700, 0, 300, -300, 700, 100],
        "tcp_pose_mm_rad": [300, 10, 215, 0, 0, 0],
    }
    assert evaluate_arm_readiness(report, tuple(report["tcp_offset_mm_rad"])) == []
    report["safety_boundary_enabled"] = False
    assert "controller TCP safety boundary is disabled" in evaluate_arm_readiness(report, tuple(report["tcp_offset_mm_rad"]))
    report["safety_boundary_enabled"] = True
    report["state"] = 1
    assert any("state 1" in reason for reason in evaluate_arm_readiness(report, tuple(report["tcp_offset_mm_rad"])))
def test_policy_action_shape_and_finite():
    assert PolicyClient.validate_action(np.zeros(19)).shape == (19,)
    with pytest.raises(ValueError): PolicyClient.validate_action(np.zeros(18))
    bad = np.zeros(19); bad[0] = np.nan
    with pytest.raises(ValueError): PolicyClient.validate_action(bad)
def test_msgpack_numpy_round_trip():
    original = np.arange(6, dtype=np.float32).reshape(2, 3)
    restored = unpackb(packb({"x": original}))["x"]
    np.testing.assert_array_equal(restored, original)
def test_xhand_limits_have_expected_shape():
    assert XHand1.JOINT_LIMITS_RAD.shape == (12, 2)
    assert np.all(XHand1.JOINT_LIMITS_RAD[:, 0] < XHand1.JOINT_LIMITS_RAD[:, 1])

def test_official_packet_extracts_wrist():
    import json
    poses = np.zeros((25, 7)); poses[:, 6] = 1.0; poses[0, :3] = [0.1, 0.2, 0.3]
    packet = json.dumps({"handedness": "left", "left_right_hand_dict": {"left": poses.ravel().tolist()}}).encode()
    wrist = decode_hand_packet(b"hand_data", packet, "left")
    np.testing.assert_allclose(wrist.position_m, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(wrist.quaternion_xyzw, [0, 0, 0, 1])
    assert decode_hand_packet(b"head", b"{}", "left") is None


def test_bridge_startup_preserves_control_events():
    import json, zmq
    context = zmq.Context()
    sender = context.socket(zmq.PAIR)
    receiver = context.socket(zmq.PAIR)
    sender.bind("inproc://startup-control-test")
    receiver.connect("inproc://startup-control-test")
    try:
        poses = np.zeros((25, 7)); poses[:, 6] = 1.0
        sender.send_multipart([b"teleop_control", json.dumps({"event": "start", "episode": 1}).encode()])
        sender.send_multipart([b"hand_data", json.dumps({
            "handedness": "left", "left_right_hand_dict": {"left": poses.ravel().tolist()},
        }).encode()])
        wrist, controls = receive_wrist(receiver, "left", 1.0)
        assert wrist is not None
        assert [control["event"] for control in controls] == ["start"]
    finally:
        sender.close(linger=0)
        receiver.close(linger=0)
        context.term()


def test_bridge_publishes_readiness_while_waiting_for_wrist():
    import zmq

    context = zmq.Context()
    receiver = context.socket(zmq.PAIR)
    receiver.bind("inproc://arm-idle-heartbeat-test")
    heartbeats = []
    try:
        with pytest.raises(TimeoutError, match="No fresh left"):
            receive_wrist(receiver, "left", 0.25, on_wait=lambda: heartbeats.append(True))
        assert len(heartbeats) >= 1
    finally:
        receiver.close(linger=0)
        context.term()


def test_bridge_keeps_servo_targets_inside_reduced_boundary():
    boundary = [700, 200, 500, -500, 700, 200]
    target = [300, 0, 199.999, 0.1, 0.2, 0.3]
    np.testing.assert_allclose(
        clamp_cartesian_boundary(target, boundary, 10), [300, 0, 210, 0.1, 0.2, 0.3]
    )
    np.testing.assert_allclose(clamp_cartesian_boundary(target, None, 10), target)

def test_official_packet_extracts_controller_trigger():
    import json
    packet = json.dumps({"handedness": "right", "left_right_gamepad_dict": {"right": [0] * 7 + [0.8] + [0] * 7}}).encode()
    assert decode_gamepad_trigger(b"gamepad_data", packet, "right") == pytest.approx(0.8)
    assert decode_gamepad_trigger(b"gamepad_data", packet, "left") is None

def test_deadman_clutch_fails_closed_on_release_and_timeout():
    clutch = DeadmanClutch(threshold=0.5, timeout_s=0.25)
    assert not clutch.active(1.0)
    clutch.update(0.8, 1.0)
    assert clutch.active(1.2)
    assert not clutch.active(1.3)
    clutch.update(0.1, 2.0)
    assert not clutch.active(2.0)

def test_official_packet_extracts_pinch_distance():
    import json
    poses = np.zeros((25, 7)); poses[:, 6] = 1.0
    poses[4, :3] = [0.01, 0.02, 0.03]
    poses[9, :3] = [0.04, 0.06, 0.03]
    packet = json.dumps({"handedness": "right", "left_right_hand_dict": {"right": poses.ravel().tolist()}}).encode()
    assert decode_pinch_distance(b"hand_data", packet, "right") == pytest.approx(0.05)
    assert decode_pinch_distance(b"hand_data", packet, "left") is None

def test_pinch_clutch_has_hysteresis_and_fails_closed_on_timeout():
    clutch = PinchClutch(engage_distance_m=0.03, release_distance_m=0.045, timeout_s=0.25)
    clutch.update(0.025, 1.0)
    assert clutch.active(1.1)
    clutch.update(0.04, 1.2)
    assert clutch.active(1.2)
    clutch.update(0.05, 1.3)
    assert not clutch.active(1.3)
    clutch.update(0.02, 2.0)
    assert not clutch.active(2.3)

def test_usb_relay_converts_column_major_webxr_matrices():
    matrix = np.eye(4)
    matrix[:3, 3] = [0.1, 0.2, 0.3]
    poses = np.tile(matrix.reshape(-1, order="F"), 25)
    converted = np.asarray(convert_hand_poses(poses)).reshape(25, 7)
    np.testing.assert_allclose(converted[:, :3], np.tile([0.1, 0.2, 0.3], (25, 1)))
    np.testing.assert_allclose(converted[:, 3:], np.tile([0, 0, 0, 1], (25, 1)))


def test_usb_relay_injects_passthrough_config(tmp_path):
    page = tmp_path / "index.html"
    page.write_text(
        '<!-- SERVER_CONFIG_PLACEHOLDER --><script src="https://cdn.jsdelivr.net/npm/fflate@0.7.4/umd/index.js"></script>\n'
        "reusableHandsPayload.timestamp = performance.now();startWebRTC();"
        "sendGamepadByteToServer(gamepad_buffer);window.onload = initialize;",
        encoding="utf-8",
    )
    patched = patched_index(page)
    assert "SEE_THROUGH: true" in patched
    assert "SERVER_CONFIG_PLACEHOLDER" not in patched
    assert "startUSBWebSocket();" in patched
    assert "showTeleopCue(state.armed, state.phase, state.countdown, state.reason || '');" in patched
    assert "CONTROL ON" in patched
    assert "sendUSBGamepadState" in patched
    assert "HOLD FLAT" in patched
    assert "renderer.xr.addEventListener('sessionend'" in patched
    assert "window.location.replace" in patched


def _right_gamepad_message(*, trigger=0.0, a=0.0, b=0.0):
    data = [0.0] * 15
    data[7] = trigger
    buttons = [0.0] * 6
    buttons[4] = a
    buttons[5] = b
    return {"type": "gamepadData", "handedness": "right", "data": data, "buttons": buttons}


def test_right_controller_cannot_change_recording_state():
    relay = RelayPublisher("inproc://controller-edge-test")
    try:
        for message in (
            _right_gamepad_message(a=1.0),
            _right_gamepad_message(trigger=0.8),
            _right_gamepad_message(b=1.0),
        ):
            assert relay.publish_message(message) == 1
        assert relay.phase == "idle"
    finally:
        relay.close()


def test_right_hand_tracking_cannot_change_recording_state():
    relay = RelayPublisher("inproc://right-hand-no-control-test")
    try:
        poses = np.tile(np.eye(4).reshape(16, order="F"), 25).tolist()
        assert relay.publish_message({
            "type": "handTracking",
            "hands": [{"handedness": "right", "poses": poses}],
        }) == 0
        assert relay.phase == "idle"
    finally:
        relay.close()


def test_quest_volume_press_mapping_and_repeat_filter():
    assert parse_volume_event("[ 1.0] /dev/input/event2: EV_KEY KEY_VOLUMEUP 00000001") == ("up", True)
    assert parse_volume_event("[ 1.0] /dev/input/event2: EV_KEY KEY_VOLUMEDOWN 00000000") == ("down", False)
    assert parse_volume_event("[ 1.0] /dev/input/event2: EV_KEY KEY_VOLUMEUP 00000002") is None
    assert parse_volume_event("[ 1.0] /dev/input/event0: EV_KEY KEY_VOLUMEUP DOWN") == ("up", True)
    assert parse_volume_event("[ 1.0] /dev/input/event0: EV_KEY KEY_VOLUMEUP UP") == ("up", False)
    assert parse_volume_event("[ 1.0] /dev/input/event2: EV_KEY KEY_VOLUMEDOWN REPEAT") is None
    relay = RelayPublisher("inproc://volume-mapping-test")
    try:
        assert volume_button_request(relay, "up") == "toggle"
        assert volume_button_request(relay, "down") is None
        relay.calibrating = True
        assert volume_button_request(relay, "up") is None
        assert volume_button_request(relay, "down") == "discard"
        relay.calibrating = False
        relay.armed = True
        assert volume_button_request(relay, "up") is None
        assert volume_button_request(relay, "down") == "toggle"
        relay.armed = False
        relay.pending_episode = 1
        assert volume_button_request(relay, "up") == "toggle"
        assert volume_button_request(relay, "down") == "discard"
    finally:
        relay.close()


def test_neutral_calibration_requires_three_stable_seconds_and_resets_on_motion():
    relay = RelayPublisher("inproc://calibration-test", calibration_duration_s=3.0)
    try:
        wrist = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
        relay.hands["left"][:7] = wrist.tolist()
        relay.last_seen_hand_at["left"] = __import__("time").monotonic()
        relay.update_hand_home_status({"state": "ready", "relay_link_ready": True, "retarget_warm": True})
        assert relay.begin_calibration("test")[0]
        moved = wrist.copy(); moved[0] += 0.02
        assert relay.update_calibration(moved) == "calibration_reset"
        relay.calibration_stable_since -= 3.1
        assert relay.update_calibration(moved) == "calibrated"
        assert relay.phase == "idle"
    finally:
        relay.close()


def test_return_home_locks_next_calibration_until_ready():
    relay = RelayPublisher("inproc://home-gate-test")
    try:
        wrist = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
        relay.hands["left"][:7] = wrist.tolist()
        relay.last_seen_hand_at["left"] = __import__("time").monotonic()
        relay.begin_return_home("saved")
        assert relay.phase == "returning"
        assert not relay.begin_calibration("too early")[0]
        relay.update_home_status({"state": "returning", "estimated_remaining_s": 2.2})
        assert relay.home_seconds_remaining == 3
        relay.update_home_status({"state": "ready", "reason": "stable"})
        assert relay.phase == "returning"
        assert not relay.begin_calibration("hand not home")[0]
        relay.update_hand_home_status({"state": "ready", "reason": "stable", "relay_link_ready": True, "retarget_warm": True})
        assert relay.phase == "returning"
        relay.update_recorder_status({"state": "pending"})
        assert relay.phase == "returning"
        relay.update_recorder_status({"state": "idle", "relay_link_ready": True})
        assert relay.phase == "idle"
        assert relay.begin_calibration("next episode")[0]
    finally:
        relay.close()


def test_missing_relay_link_blocks_hand_and_recorder_readiness():
    relay = RelayPublisher("inproc://link-gate-test")
    try:
        wrist = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
        relay.hands["left"][:7] = wrist.tolist()
        relay.last_seen_hand_at["left"] = __import__("time").monotonic()
        relay.update_hand_home_status({"state": "ready", "relay_link_ready": False, "retarget_warm": True})
        assert not relay.begin_calibration("link lost")[0]
        relay.update_hand_home_status({"state": "ready", "relay_link_ready": True, "retarget_warm": True})
        assert relay.begin_calibration("link restored")[0]
        relay.cancel_calibration("test")
        relay.begin_return_home("saved")
        relay.update_home_status({"state": "ready"})
        relay.update_hand_home_status({"state": "ready", "relay_link_ready": True, "retarget_warm": True})
        relay.update_recorder_status({"state": "idle", "relay_link_ready": False})
        assert relay.phase == "returning"
        relay.update_recorder_status({"state": "idle", "relay_link_ready": True})
        assert relay.phase == "idle"
    finally:
        relay.close()


def test_home_fault_is_shown_as_fault_and_blocks_new_episode():
    relay = RelayPublisher("inproc://home-fault-test")
    try:
        relay.begin_return_home("save")
        relay.update_home_status({"state": "fault", "reason": "controller error 31"})
        assert relay.phase == "fault"
        assert not relay.begin_calibration("next episode")[0]
    finally:
        relay.close()


def test_fault_stops_episode_and_releases_recorder_for_next_start():
    relay = RelayPublisher("inproc://fault-recovery-test")
    sink = FakeEpisodeSink()
    recorder = EpisodeRecorder(sink, task="pick")
    events = []
    relay.publish_control = lambda event, reason, episode=None: events.append(ControlEvent(event, episode, reason))
    try:
        relay.armed = True
        relay.active_episode = 7
        recorder.handle(ControlEvent("start", 7))
        recorder.add_frame(recording_frame(), timestamp=0.0)
        assert relay.end_episode("fault", "xArm bridge telemetry lost")
        assert [event.event for event in events] == ["fault", "abort"]
        for event in events:
            recorder.handle(event)
        assert recorder.state == EpisodeState.IDLE
        assert sink.saved == 0
        assert recorder.handle(ControlEvent("start", 8)) == EpisodeState.RECORDING
    finally:
        relay.close()


def test_zero_frame_episode_can_be_discarded_and_followed_by_next_episode():
    relay = RelayPublisher("inproc://empty-episode-recovery-test")
    recorder = EpisodeRecorder(FakeEpisodeSink(), task="pick")
    events = []
    relay.publish_control = lambda event, reason, episode=None: events.append(ControlEvent(event, episode, reason))
    try:
        recorder.handle(ControlEvent("start", 9))
        relay.pending_episode = 9
        assert recorder.handle(ControlEvent("complete", 9)) == EpisodeState.FAULT
        assert relay.discard_pending("zero frames")
        assert events[-1].event == "discard"
        assert recorder.handle(events[-1]) == EpisodeState.IDLE
        assert recorder.handle(ControlEvent("start", 10)) == EpisodeState.RECORDING
    finally:
        relay.close()


def test_neutral_calibration_still_publishes_hand_for_xhand_only():
    relay = RelayPublisher("inproc://calibration-hand-test")
    try:
        poses = []
        for _ in range(25):
            poses.extend(np.eye(4).reshape(16, order="F").tolist())
        relay.hands["left"][:7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        relay.last_seen_hand_at["left"] = __import__("time").monotonic()
        relay.update_hand_home_status({"state": "ready", "relay_link_ready": True, "retarget_warm": True})
        assert relay.begin_calibration("test")[0]
        result = relay.publish_message({
            "type": "handTracking",
            "hands": [{"handedness": "left", "poses": poses}],
        })
        assert result == 1
        assert not relay.armed
    finally:
        relay.close()


def test_idle_hand_tracking_bootstraps_arm_without_commanding_xhand():
    import json

    relay = RelayPublisher("inproc://idle-arm-preview-test")
    packets = []
    relay.send = lambda topic, payload: packets.append((topic, payload))
    try:
        pose = np.eye(4).reshape(16, order="F").tolist()
        message = {
            "type": "handTracking",
            "hands": [{"handedness": "left", "poses": pose * 25}],
        }
        # Cold XHand: the vendor client only subscribes to hand_data, and its
        # first retarget call JIT-compiles, so warmup needs the real topic.
        relay.publish_message(message)
        assert [topic for topic, _ in packets] == ["hand_data"]
        wrist = decode_hand_packet(
            b"hand_data", json.dumps(packets[0][1]).encode(), "left"
        )
        assert wrist is not None
        np.testing.assert_allclose(wrist.position_m, [0.0, 0.0, 0.0])
        assert not relay.armed

        packets.clear()
        relay.update_hand_home_status({"state": "ready", "relay_link_ready": True, "retarget_warm": True})
        relay.publish_message(message)
        assert [topic for topic, _ in packets] == ["arm_preview"]
        wrist = decode_hand_packet(
            b"arm_preview", json.dumps(packets[0][1]).encode(), "left"
        )
        assert wrist is not None
        np.testing.assert_allclose(wrist.position_m, [0.0, 0.0, 0.0])
        assert not relay.armed
    finally:
        relay.close()


def test_calibration_is_locked_until_xhand_retarget_is_warm():
    import time as time_module

    relay = RelayPublisher("inproc://retarget-warm-gate-test")
    try:
        relay.hands["left"][:7] = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
        relay.last_seen_hand_at["left"] = time_module.monotonic()
        relay.update_hand_home_status({"state": "ready", "relay_link_ready": True})
        assert not relay.hand_retarget_warm
        allowed, reason = relay.begin_calibration("cold retarget")
        assert not allowed
        assert "warming up" in reason
        relay.update_hand_home_status({"state": "ready", "relay_link_ready": True, "retarget_warm": True})
        assert relay.hand_retarget_warm
        assert relay.begin_calibration("warm retarget")[0]
    finally:
        relay.close()


def test_xhand_home_step_and_feedback_shape():
    feedback = {"data": {"joint_position_dic": {f"joint{i}": float(i) for i in range(12)}}}
    measured = joint_positions_rad(feedback)
    assert measured.shape == (12,)
    target = np.zeros(12)
    step = home_command(measured, target, 0.004)
    assert np.max(np.abs(step - measured)) <= 0.00400001
    assert step[-1] < measured[-1]


def test_reopened_lerobot_dataset_has_no_buffer_to_clear():
    calls = []

    class Writer:
        def wait_until_done(self):
            calls.append("drained")

    class Dataset:
        episode_buffer = None
        image_writer = Writer()
        cleared = 0

        def clear_episode_buffer(self):
            calls.append("cleared")
            self.cleared += 1

    dataset = Dataset()
    sink = LeRobotSink(dataset)
    sink.clear_episode_buffer()
    assert dataset.cleared == 0
    dataset.episode_buffer = {"size": 0}
    sink.clear_episode_buffer()
    assert dataset.cleared == 1
    assert calls == ["drained", "cleared"]


def test_servo_tracking_error_stops_when_measured_tcp_lags():
    target = np.array([300., 10., 150., 0., 0., -np.pi + 0.01])
    measured = np.array([291., 10., 150., 0., 0., np.pi - 0.01])
    check_servo_tracking_error(target, measured, 15., 0.2)
    measured[0] = 280.
    with pytest.raises(RuntimeError, match="tracking error"):
        check_servo_tracking_error(target, measured, 15., 0.2)


def test_servo_tracking_error_accepts_equivalent_orientation_at_gimbal_lock():
    # Captured xArm feedback canonicalized pitch to pi/2 and moved roll/yaw
    # together, although the commanded and measured TCP rotations match.
    target = np.array([301.2954171, 16.9984657, 214.519424,
                       -1.6126525, 1.5603541, -1.4961259])
    measured = np.array([300.530426, 15.86691, 214.646652,
                         -0.116526, np.pi / 2, 0.0])
    check_servo_tracking_error(target, measured, 15.0, 0.2)
    measured[3:] = [0.5, 0.0, 0.0]
    with pytest.raises(RuntimeError, match="tracking error"):
        check_servo_tracking_error(target, measured, 15.0, 0.2)

def test_relative_wrist_mapping_has_no_initial_jump_and_maps_axes():
    initial = WristPose(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    mapper = RelativeWristMapper(initial, np.array([300.0, 10.0, 250.0, 0.1, 0.2, 0.3]))
    np.testing.assert_allclose(mapper.target(initial), [300.0, 10.0, 250.0, 0.1, 0.2, 0.3], atol=1e-12)
    moved = WristPose(np.array([1.01, 2.02, 3.03]), initial.quaternion_xyzw)
    np.testing.assert_allclose(mapper.target(moved)[:3], [270.0, 0.0, 270.0], atol=1e-12)


def test_webxr_forward_maps_to_xarm_forward():
    initial = WristPose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    mapper = RelativeWristMapper(initial, np.array([300.0, 0.0, 300.0, 0.0, 0.0, 0.0]))
    hand_forward = WristPose(np.array([0.0, 0.0, -0.05]), initial.quaternion_xyzw)
    np.testing.assert_allclose(mapper.target(hand_forward)[:3], [350.0, 0.0, 300.0])

def test_relative_wrist_mapping_uses_recoverable_tracking_limit():
    initial = WristPose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    mapper = RelativeWristMapper(initial, np.zeros(6), max_translation_mm=100.0)
    moved = WristPose(np.array([0.0, 0.0, 0.101]), initial.quaternion_xyzw)
    with pytest.raises(TrackingLimitError):
        mapper.target(moved)

def test_cartesian_limiter_wraps_angles():
    previous = np.array([0.0, 0.0, 0.0, np.pi - 0.01, 0.0, 0.0])
    desired = np.array([10.0, -10.0, 1.0, -np.pi + 0.01, 1.0, 0.0])
    limited = limit_servo_step(previous, desired, 2.0, 0.05)
    np.testing.assert_allclose(limited[:3], [2.0, -2.0, 1.0])
    assert abs(((limited[3] - previous[3] + np.pi) % (2 * np.pi)) - np.pi) == pytest.approx(0.02)

def test_xarm_cartesian_servo_rejects_oversized_step_before_sdk_call():
    class FakeArm:
        connected = True
        def set_servo_cartesian(self, *_args, **_kwargs):
            raise AssertionError("unsafe target reached SDK")
    arm = XArm7(HardwareConfig(xarm_max_cartesian_step_mm=2.0), allow_motion=True)
    arm._arm = FakeArm()
    arm._last_servo_pose = np.zeros(6)
    arm._servo_boundary = None
    with pytest.raises(ValueError, match="translation limit"):
        arm.servo_cartesian([2.1, 0, 0, 0, 0, 0])


class FakeEpisodeSink:
    def __init__(self):
        self.frames, self.saved, self.cleared = [], 0, 0
    def add_frame(self, frame, *, task, timestamp):
        self.frames.append((frame, task, timestamp))
    def save_episode(self): self.saved += 1
    def clear_episode_buffer(self): self.cleared += 1


def recording_frame():
    return {
        "action": np.zeros(18),
        "observation.arm_joint_position": np.zeros(7),
        "observation.arm_tcp_pose": np.zeros(6),
        "observation.hand_joint_position": np.zeros(12),
    }


def test_episode_is_only_saved_after_complete_and_accept():
    sink = FakeEpisodeSink(); recorder = EpisodeRecorder(sink, task="pick")
    recorder.handle(ControlEvent("start", 1))
    assert recorder.add_frame(recording_frame(), timestamp=0.0)
    assert recorder.handle(ControlEvent("complete", 1)) == EpisodeState.PENDING
    assert sink.saved == 0
    assert recorder.handle(ControlEvent("save", 1)) == EpisodeState.IDLE
    assert sink.saved == 1


def test_rerecord_and_fault_never_save_partial_episode():
    sink = FakeEpisodeSink(); recorder = EpisodeRecorder(sink, task="pick")
    recorder.handle(ControlEvent("start", 2))
    recorder.add_frame(recording_frame(), timestamp=1.0)
    recorder.handle(ControlEvent("discard", 2))
    recorder.handle(ControlEvent("start", 3))
    recorder.add_frame(recording_frame(), timestamp=2.0)
    assert recorder.handle(ControlEvent("fault", 3, "tracking lost")) == EpisodeState.FAULT
    assert recorder.handle(ControlEvent("complete", 3)) == EpisodeState.FAULT
    assert recorder.handle(ControlEvent("save", 3)) == EpisodeState.FAULT
    assert sink.saved == 0
    assert sink.cleared >= 4


def test_recorder_ignores_frames_outside_recording_and_rejects_bad_time():
    sink = FakeEpisodeSink(); recorder = EpisodeRecorder(sink, task="pick")
    assert not recorder.add_frame(recording_frame(), timestamp=0.0)
    recorder.handle(ControlEvent("start", 1))
    recorder.add_frame(recording_frame(), timestamp=1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        recorder.add_frame(recording_frame(), timestamp=1.0)


def test_recording_frame_validation_copies_images_and_checks_action():
    frame = recording_frame(); frame["observation.images.wrist_view"] = np.zeros((4, 5, 3), np.uint8)
    validated = validate_recording_frame(frame)
    assert validated["action"].dtype == np.float32
    assert validated["observation.images.wrist_view"].shape == (4, 5, 3)
    frame["action"] = np.zeros(17)
    with pytest.raises(ValueError, match="action"):
        validate_recording_frame(frame)


def test_npz_sink_atomically_saves_episode(tmp_path):
    sink = NpzEpisodeSink(tmp_path); recorder = EpisodeRecorder(sink, task="pick")
    recorder.handle(ControlEvent("start", 1))
    recorder.add_frame(recording_frame(), timestamp=0.0)
    recorder.handle(ControlEvent("complete", 1)); recorder.handle(ControlEvent("save", 1))
    archive = np.load(tmp_path / "episodes/episode_000000.npz")
    assert archive["action"].shape == (1, 18)
    assert (tmp_path / "manifest.json").is_file()


def test_npz_sink_streams_frames_to_disk_and_cleans_discard(tmp_path):
    sink = NpzEpisodeSink(tmp_path)
    recorder = EpisodeRecorder(sink, task="pick")
    recorder.handle(ControlEvent("start", 1))
    frame = recording_frame()
    frame["observation.images.wrist_view"] = np.zeros((4, 5, 3), np.uint8)
    for index in range(130):  # Force the disk-backed buffer to grow.
        frame["observation.images.wrist_view"].fill(index % 256)
        recorder.add_frame(frame, timestamp=index / 20)
    assert sink._count == 130
    assert sink._buffer_dir is not None and sink._buffer_dir.is_dir()
    recorder.handle(ControlEvent("complete", 1))
    recorder.handle(ControlEvent("save", 1))
    with np.load(tmp_path / "episodes/episode_000000.npz") as archive:
        assert archive["observation.images.wrist_view"].shape == (130, 4, 5, 3)
        assert archive["observation.images.wrist_view"][129, 0, 0, 0] == 129
    assert not list((tmp_path / "episodes").glob(".buffer-*"))

    recorder.handle(ControlEvent("start", 2))
    recorder.add_frame(frame, timestamp=0.0)
    recorder.handle(ControlEvent("discard", 2))
    assert not list((tmp_path / "episodes").glob(".buffer-*"))


def test_xhand_nested_vector_extraction():
    payload = {"code": 200, "data": {"joint_position_dic": {str(i): i for i in range(12)}}}
    np.testing.assert_array_equal(preferred_vector(payload, ("joint_position_dic",), 12), np.arange(12))
    np.testing.assert_array_equal(preferred_vector({"left": list(range(12)), "right": None}, ("target",), 12), np.arange(12))
    assert numeric_vector({"bad": [1, 2]}, 12) is None
