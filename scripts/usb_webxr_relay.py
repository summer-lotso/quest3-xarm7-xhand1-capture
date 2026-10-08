#!/usr/bin/env python3
"""Serve RobotEra's WebXR page over USB WebSocket and publish compatible ZMQ."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, deque
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import ssl
import sys
import termios
import time
import tty

import numpy as np
import zmq


_VOLUME_EVENT = re.compile(
    r"\bEV_KEY\s+(KEY_VOLUMEUP|KEY_VOLUMEDOWN|0073|0072)\s+(DOWN|UP|REPEAT|[0-9a-fA-F]{8})\b",
    re.IGNORECASE,
)


def parse_volume_event(line: str) -> tuple[str, bool] | None:
    """Decode a physical Quest volume key transition from adb getevent -lt."""
    match = _VOLUME_EVENT.search(line)
    if match is None:
        return None
    raw_value = match.group(2).upper()
    value = {"UP": 0, "DOWN": 1, "REPEAT": 2}.get(raw_value)
    if value is None:
        value = int(raw_value, 16)
    if value not in (0, 1):  # 2 is a held-key repeat, never another command.
        return None
    key = "up" if match.group(1).upper() in ("KEY_VOLUMEUP", "0073") else "down"
    return key, bool(value)


def volume_button_request(publisher: "RelayPublisher", key: str) -> str | None:
    """Map one key press to the existing episode state machine."""
    if key == "up" and not publisher.armed and not publisher.calibrating:
        return "toggle"  # Calibrate/start, or save a completed episode.
    if key == "down" and (publisher.armed or publisher.pending_episode is not None):
        return "toggle" if publisher.armed else "discard"
    if key == "down" and publisher.calibrating:
        return "discard"  # Cancel calibration without recording.
    return None


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized XYZW quaternion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Rotation matrix must be finite and 3x3")
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2
        quaternion = np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            quaternion = np.array([0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale])
        elif axis == 1:
            scale = np.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            quaternion = np.array([(matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale])
        else:
            scale = np.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            quaternion = np.array([(matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale, (matrix[1, 0] - matrix[0, 1]) / scale])
    return quaternion / np.linalg.norm(quaternion)


def convert_hand_poses(flat_poses) -> list[float]:
    """Convert 25 column-major WebXR joint matrices to position + XYZW poses."""
    poses = np.asarray(flat_poses, dtype=np.float64)
    if poses.shape != (25 * 16,) or not np.isfinite(poses).all():
        raise ValueError("WebXR hand must contain 25 finite 4x4 matrices")
    result = np.empty((25, 7), dtype=np.float64)
    for index, raw in enumerate(poses.reshape(25, 16)):
        matrix = raw.reshape(4, 4, order="F")
        result[index, :3] = matrix[:3, 3]
        result[index, 3:] = matrix_to_quaternion_xyzw(matrix[:3, :3])
    return result.ravel().tolist()


def patched_index(source: Path) -> str:
    html = source.read_text(encoding="utf-8")
    # The vendor page loads fflate from jsDelivr, although its only references
    # are commented out.  Remove it so the USB page has no Internet dependency.
    html = html.replace(
        '  <script src="https://cdn.jsdelivr.net/npm/fflate@0.7.4/umd/index.js"></script>\n',
        "",
        1,
    )
    # RobotEra's normal web server replaces this placeholder at request time.
    # The USB relay serves the static file itself, so leaving it untouched makes
    # APP_CONFIG fall back to SEE_THROUGH=false.  The page then puts a large
    # video plane in front of the user; because USB mode has no WebRTC camera
    # stream, that plane is solid black and blocks the Quest passthrough view.
    placeholder = "<!-- SERVER_CONFIG_PLACEHOLDER -->"
    if placeholder not in html:
        raise RuntimeError("Installed WebXR page has no server config placeholder")
    html = html.replace(
        placeholder,
        """<script>
    window.APP_CONFIG = {
      CAMERA_TYPE: 'dummy',
      IS_STEREO: false,
      VIDEO_WIDTH: 1280,
      VIDEO_HEIGHT: 720,
      SINGLE_EYE_WIDTH: 1280,
      SINGLE_EYE_HEIGHT: 720,
      SEE_THROUGH: true
    };
  </script>""",
        1,
    )
    old = "reusableHandsPayload.timestamp = performance.now();"
    gamepad_send = "sendGamepadByteToServer(gamepad_buffer)"
    if old not in html or "startWebRTC();" not in html or "window.onload = initialize;" not in html or gamepad_send not in html:
        raise RuntimeError("Installed WebXR page no longer matches the supported RobotEra layout")
    html = html.replace(old, "reusableHandsPayload.timestamp = Date.now();", 1)
    html = html.replace("startWebRTC();", "startUSBWebSocket();", 1)
    html = html.replace(
        gamepad_send,
        gamepad_send + ";\n            sendUSBGamepadState(pose, gamepad, handedness, reusableHandsPayload.timestamp)",
        1,
    )
    patch = r'''
    // USB transport injected by usb_webxr_relay.py.
    let usbSocket = null;
    let usbClosing = false;
    let arRecoveryRegistered = false;
    let arReloadPending = false;
    let teleopCue = null;
    let teleopCueTimer = null;
    let teleopAudioContext = null;

    function playTeleopTone(armed) {
      try {
        teleopAudioContext = teleopAudioContext || new (window.AudioContext || window.webkitAudioContext)();
        const sound = () => {
          const oscillator = teleopAudioContext.createOscillator();
          const gain = teleopAudioContext.createGain();
          const now = teleopAudioContext.currentTime;
          oscillator.type = 'sine';
          oscillator.frequency.setValueAtTime(armed ? 880 : 330, now);
          gain.gain.setValueAtTime(0.0001, now);
          gain.gain.exponentialRampToValueAtTime(0.18, now + 0.015);
          gain.gain.exponentialRampToValueAtTime(0.0001, now + (armed ? 0.22 : 0.35));
          oscillator.connect(gain).connect(teleopAudioContext.destination);
          oscillator.start(now);
          oscillator.stop(now + (armed ? 0.24 : 0.37));
        };
        if (teleopAudioContext.state === 'suspended') {
          teleopAudioContext.resume().then(sound).catch(() => {});
        } else {
          sound();
        }
      } catch (error) {
        console.warn('Unable to play teleop cue:', error);
      }
    }

    function showTeleopCue(armed, phase = 'idle', countdown = null, reason = '') {
      if (!camera) return;
      if (teleopCueTimer) clearTimeout(teleopCueTimer);
      if (teleopCue) {
        camera.remove(teleopCue);
        teleopCue.material.map.dispose();
        teleopCue.material.dispose();
      }
      const canvas = document.createElement('canvas');
      canvas.width = 1024;
      canvas.height = 256;
      const context = canvas.getContext('2d');
      context.fillStyle = armed
        ? 'rgba(0, 145, 55, 0.92)'
        : (phase === 'returning'
          ? 'rgba(105, 55, 180, 0.94)'
          : (phase === 'pending'
          ? 'rgba(190, 120, 0, 0.94)'
          : (phase === 'calibrating' ? 'rgba(0, 85, 180, 0.94)'
          : (phase === 'fault' ? 'rgba(160, 25, 25, 0.94)' : 'rgba(25, 125, 75, 0.92)'))));
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = 'white';
      context.font = `bold ${phase === 'pending' ? 58 : 82}px sans-serif`;
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      const errorLabel = reason.includes('warming up') ? 'WARMING UP'
        : (/xarm|controller|servo|collision/i.test(reason) ? 'CHECK ARM'
        : (reason.includes('recorder') ? 'CHECK RECORDER' : 'CHECK TRACKING'));
      const label = armed
        ? 'CONTROL ON'
        : (phase === 'returning'
          ? `RETURN HOME  ${countdown == null ? '' : '~' + countdown + 's'}`
          : (phase === 'pending'
          ? 'VOL + SAVE    VOL - REDO'
          : (phase === 'calibrating' ? `HOLD FLAT  ${countdown ?? 3}`
          : (phase === 'fault' ? errorLabel : 'NOT RECORDING'))));
      context.fillText(label, canvas.width / 2, canvas.height / 2);
      const texture = new THREE.CanvasTexture(canvas);
      teleopCue = new THREE.Sprite(new THREE.SpriteMaterial({map: texture, transparent: true, depthTest: false}));
      teleopCue.position.set(0, 0.16, -0.9);
      teleopCue.scale.set(0.64, 0.16, 1);
      teleopCue.renderOrder = 10000;
      camera.add(teleopCue);
      playTeleopTone(armed);
      teleopCueTimer = setTimeout(() => {
        if (!teleopCue) return;
        camera.remove(teleopCue);
        teleopCue.material.map.dispose();
        teleopCue.material.dispose();
        teleopCue = null;
      }, (phase === 'calibrating' || phase === 'returning') ? 1200 : 1600);
    }

    function startUSBWebSocket() {
      if (!arRecoveryRegistered) {
        // Quest Browser can leave the Three/WebXR page unresponsive after an
        // immersive session ends. Tear down this page and reload a fresh one;
        // the user can then press START AR again without restarting Browser.
        renderer.xr.addEventListener('sessionend', () => {
          if (usbClosing || arReloadPending) return;
          arReloadPending = true;
          sendDataToServer({type: 'sessionEnded', timestamp: Date.now()});
          usbClosing = true;
          if (usbSocket) {
            usbSocket.onclose = null;
            usbSocket.close();
          }
          renderer.setAnimationLoop(null);
          window.setTimeout(() => {
            window.location.replace(`${location.origin}${location.pathname}?ar_reset=${Date.now()}`);
          }, 250);
        });
        arRecoveryRegistered = true;
      }
      updateStatus('Connecting USB WebSocket...', 'connecting');
      usbSocket = new WebSocket(`wss://${location.host}/ws`);
      usbSocket.onopen = () => {
        isConnected = true;
        updateStatus('USB connected. Enter AR; use headset volume buttons to control recording.', 'connected');
      };
      usbSocket.onmessage = (event) => {
        try {
          const state = JSON.parse(event.data);
          if (state.type !== 'teleopState') return;
          updateStatus(
            state.armed ? 'TELEOP ARMED' : `TELEOP STOPPED: ${state.reason}`,
            state.armed ? 'connected' : 'info'
          );
          showTeleopCue(state.armed, state.phase, state.countdown, state.reason || '');
        } catch (error) {
          console.error('Invalid USB relay status:', error);
        }
      };
      usbSocket.onclose = () => {
        isConnected = false;
        updateStatus('USB WebSocket disconnected.', 'error');
        if (!usbClosing) setTimeout(startUSBWebSocket, 1000);
      };
      usbSocket.onerror = () => updateStatus('USB WebSocket error.', 'error');
    }

    sendDataToServer = function(payload) {
      if (usbSocket && usbSocket.readyState === WebSocket.OPEN) {
        usbSocket.send(JSON.stringify(payload));
      }
    };

    function sendUSBGamepadState(pose, gamepad, handedness, timestamp) {
      if (!pose || !gamepad) return;
      const p = pose.transform.position;
      const q = pose.transform.orientation;
      const data = [
        p.x, p.y, p.z, q.x, q.y, q.z, q.w,
        gamepad.buttons[0]?.value ?? 0.0,
        gamepad.buttons[1]?.value ?? 0.0,
        PLACEHOLDER_VALUE,
        gamepad.buttons[3]?.value ?? 0.0,
        PLACEHOLDER_VALUE,
        PLACEHOLDER_VALUE,
        gamepad.axes[2] ?? 0.0,
        gamepad.axes[3] ?? 0.0
      ];
      sendDataToServer({
        type: 'gamepadData',
        timestamp,
        handedness,
        data,
        buttons: Array.from(gamepad.buttons, button => button?.value ?? 0.0)
      });
    }

    window.addEventListener('beforeunload', () => {
      usbClosing = true;
      if (usbSocket) usbSocket.close();
    });
    '''
    return html.replace("window.onload = initialize;", patch + "\n    window.onload = initialize;", 1)


class RelayPublisher:
    def __init__(
        self,
        address: str,
        tracking_timeout_s: float = 0.6,
        acquisition_timeout_s: float = 5.0,
        calibration_duration_s: float = 3.0,
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 8)
        self.socket.bind(address)
        self.hands = {"left": [0.0] * 175, "right": [0.0] * 175}
        self.gamepads = {"left": [0.0] * 15, "right": [0.0] * 15}
        self.tracking_timeout_s = tracking_timeout_s
        self.acquisition_timeout_s = acquisition_timeout_s
        self.calibration_duration_s = calibration_duration_s
        self.armed = False
        self.armed_at = None
        self.last_hand_at = None
        self.last_seen_hand_at = {"left": None, "right": None}
        self.episode_number = 0
        self.active_episode = None
        self.pending_episode = None
        self.calibrating = False
        self.calibration_stable_since = None
        self.calibration_position = None
        self.calibration_quaternion = None
        self.calibration_countdown = None
        self.returning_home = False
        self.arm_home_ready = True
        self.hand_home_ready = False
        self.hand_retarget_warm = False
        self.recorder_ready = False
        self.home_fault = None
        self.home_seconds_remaining = None

    @property
    def phase(self) -> str:
        if self.armed:
            return "armed"
        if self.home_fault is not None:
            return "fault"
        if self.calibrating:
            return "calibrating"
        if self.pending_episode is not None:
            return "pending"
        if self.returning_home:
            return "returning"
        return "idle"

    def begin_calibration(self, reason: str) -> tuple[bool, str]:
        if self.returning_home:
            return False, "xArm/XHand are still returning to the safe position"
        if self.home_fault is not None:
            return False, f"robot home failed: {self.home_fault}"
        if not self.hand_home_ready:
            return False, "XHand has not confirmed the safe starting pose"
        if not self.hand_retarget_warm:
            return False, "XHand retarget is still warming up; wait for XHAND_RETARGET_WARMUP"
        now = time.monotonic()
        left_seen_at = self.last_seen_hand_at["left"]
        if left_seen_at is None or now - left_seen_at > self.tracking_timeout_s:
            return False, "no fresh left-hand tracking; show the open left hand before calibration"
        wrist = np.asarray(self.hands["left"][:7], dtype=np.float64)
        if wrist.shape != (7,) or not np.isfinite(wrist).all():
            return False, "left wrist pose is invalid"
        self.calibrating = True
        self.calibration_stable_since = now
        self.calibration_position = wrist[:3].copy()
        self.calibration_quaternion = wrist[3:].copy()
        self.calibration_countdown = int(np.ceil(self.calibration_duration_s))
        print(f"TELEOP_CALIBRATING: {reason}", flush=True)
        return True, reason

    def cancel_calibration(self, reason: str) -> bool:
        if not self.calibrating:
            return False
        self.calibrating = False
        self.calibration_stable_since = None
        self.calibration_position = None
        self.calibration_quaternion = None
        self.calibration_countdown = None
        print(f"TELEOP_CALIBRATION_CANCELLED: {reason}", flush=True)
        return True

    def update_calibration(self, wrist: np.ndarray) -> str | None:
        if not self.calibrating:
            return None
        now = time.monotonic()
        position = wrist[:3]
        quaternion = wrist[3:] / np.linalg.norm(wrist[3:])
        baseline_quaternion = self.calibration_quaternion / np.linalg.norm(self.calibration_quaternion)
        translation_m = float(np.linalg.norm(position - self.calibration_position))
        rotation_rad = 2.0 * float(np.arccos(np.clip(abs(np.dot(quaternion, baseline_quaternion)), -1.0, 1.0)))
        # Require a genuinely steady neutral pose: 15 mm and 10 degrees.
        if translation_m > 0.015 or rotation_rad > np.radians(10.0):
            self.calibration_stable_since = now
            self.calibration_position = position.copy()
            self.calibration_quaternion = quaternion.copy()
            self.calibration_countdown = int(np.ceil(self.calibration_duration_s))
            return "calibration_reset"
        elapsed = now - self.calibration_stable_since
        if elapsed >= self.calibration_duration_s:
            self.calibrating = False
            self.calibration_stable_since = None
            self.calibration_position = None
            self.calibration_quaternion = None
            self.calibration_countdown = None
            print("TELEOP_CALIBRATION_COMPLETE: neutral pose accepted", flush=True)
            return "calibrated"
        countdown = max(1, int(np.ceil(self.calibration_duration_s - elapsed)))
        if countdown != self.calibration_countdown:
            self.calibration_countdown = countdown
            return "calibration_tick"
        return None

    def request_arm(self, reason: str) -> tuple[bool, str]:
        if self.returning_home or self.home_fault is not None:
            return False, "xArm is not ready after homing"
        now = time.monotonic()
        left_seen_at = self.last_seen_hand_at["left"]
        if left_seen_at is None or now - left_seen_at > self.tracking_timeout_s:
            return False, "no fresh hand tracking; enter AR and keep the hand visible"
        self.set_armed(True, reason)
        self.last_hand_at = left_seen_at
        self.episode_number += 1
        self.active_episode = self.episode_number
        self.publish_control("start", reason)
        return True, reason

    def begin_return_home(self, reason: str) -> None:
        self.returning_home = True
        self.arm_home_ready = False
        self.hand_home_ready = False
        self.recorder_ready = False
        self.home_fault = None
        self.home_seconds_remaining = None
        print(f"RETURN_HOME_REQUESTED: {reason}", flush=True)

    def update_home_status(self, payload: dict) -> str:
        state = str(payload.get("state", "")).lower()
        reason = str(payload.get("reason", state or "unknown home status"))
        remaining = payload.get("estimated_remaining_s")
        self.home_seconds_remaining = (
            max(0, int(np.ceil(float(remaining))))
            if isinstance(remaining, (int, float)) and np.isfinite(remaining)
            else None
        )
        if state == "ready":
            self.arm_home_ready = True
            self._complete_return_if_ready()
        elif state == "fault":
            self.cancel_calibration(reason)
            self.returning_home = False
            self.arm_home_ready = False
            self.home_fault = f"xArm: {reason}"
            self.home_seconds_remaining = None
            print(f"RETURN_HOME_FAULT: {reason}", file=sys.stderr, flush=True)
        elif state == "returning":
            self.returning_home = True
            self.home_fault = None
        return reason

    def update_hand_home_status(self, payload: dict) -> str:
        state = str(payload.get("state", "")).lower()
        reason = str(payload.get("reason", state or "unknown XHand status"))
        self.hand_retarget_warm = payload.get("retarget_warm") is True
        if state == "ready":
            self.hand_home_ready = payload.get("relay_link_ready") is True
            self._complete_return_if_ready()
        elif state == "fault":
            self.cancel_calibration(reason)
            self.hand_home_ready = False
            self.returning_home = False
            self.home_fault = f"XHand: {reason}"
            print(f"RETURN_HOME_FAULT: {self.home_fault}", file=sys.stderr, flush=True)
        elif payload.get("relay_link_ready") is not True:
            self.hand_home_ready = False
        return reason

    def update_recorder_status(self, payload: dict) -> None:
        self.recorder_ready = payload.get("state") == "idle" and payload.get("relay_link_ready") is True
        self._complete_return_if_ready()

    def _complete_return_if_ready(self) -> None:
        if self.returning_home and self.arm_home_ready and self.hand_home_ready and self.recorder_ready:
            self.returning_home = False
            self.home_fault = None
            self.home_seconds_remaining = None
            print("RETURN_HOME_COMPLETE: xArm, XHand, and recorder are ready; next episode enabled", flush=True)

    def end_episode(self, event: str, reason: str) -> bool:
        if not self.armed:
            return False
        episode = self.active_episode
        self.set_armed(False, reason)
        self.publish_control(event, reason, episode)
        if event == "fault":
            # The recorder preserves FAULT until abort/discard. A relay
            # interlock has already stopped this episode, so release its
            # buffer now instead of poisoning every subsequent start.
            self.publish_control("abort", reason, episode)
        self.active_episode = None
        return True

    def complete_episode(self, reason: str) -> bool:
        if not self.armed:
            return False
        episode = self.active_episode
        self.set_armed(False, reason)
        self.publish_control("complete", reason, episode)
        self.pending_episode = episode
        self.active_episode = None
        return True

    def commit_pending(self, reason: str) -> bool:
        if self.pending_episode is None:
            return False
        self.publish_control("save", reason, self.pending_episode)
        self.pending_episode = None
        return True

    def discard_pending(self, reason: str) -> bool:
        if self.pending_episode is None:
            return False
        self.publish_control("discard", reason, self.pending_episode)
        self.pending_episode = None
        return True

    def publish_control(self, event: str, reason: str, episode: int | None = None) -> None:
        self.send(
            "teleop_control",
            {
                "timestamp": self.timestamp(),
                "event": event,
                "episode": self.active_episode if episode is None else episode,
                "reason": reason,
            },
        )

    def set_armed(self, armed: bool, reason: str) -> bool:
        changed = self.armed != armed
        self.armed = armed
        if armed:
            self.calibrating = False
            self.armed_at = time.monotonic()
        else:
            self.armed_at = None
            self.last_hand_at = None
        if changed:
            state = "ARMED" if armed else "STOPPED"
            print(f"TELEOP_{state}: {reason}", flush=True)
        return changed

    def stop_if_stale(self) -> str | None:
        if self.calibrating:
            left_seen_at = self.last_seen_hand_at["left"]
            if left_seen_at is None or time.monotonic() - left_seen_at > self.tracking_timeout_s:
                reason = "left-hand tracking lost during neutral calibration"
                self.cancel_calibration(reason)
                return reason
        if not self.armed:
            return None
        now = time.monotonic()
        if self.last_hand_at is not None:
            if now - self.last_hand_at > self.tracking_timeout_s:
                reason = f"hand tracking stale for more than {self.tracking_timeout_s:.1f}s"
                self.end_episode("fault", reason)
                return reason
        elif self.armed_at is not None and now - self.armed_at > self.acquisition_timeout_s:
            reason = f"no hand acquired within {self.acquisition_timeout_s:.1f}s"
            self.end_episode("fault", reason)
            return reason
        return None

    @staticmethod
    def timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def send(self, topic: str, payload: dict) -> None:
        try:
            self.socket.send_multipart(
                [topic.encode(), json.dumps(payload, separators=(",", ":")).encode()],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass

    def publish_message(self, message: dict) -> int:
        kind = message.get("type")
        if kind == "viewerPose":
            if not self.armed:
                return 0
            pose = np.asarray(message.get("pose"), dtype=np.float64)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError("viewerPose must contain seven finite values")
            self.send("head", {"timestamp": self.timestamp(), "head_pose": pose.tolist()})
            return 1
        if kind == "gamepadData":
            side = str(message.get("handedness", "")).lower()
            values = np.asarray(message.get("data"), dtype=np.float64)
            if side not in self.gamepads:
                raise ValueError("gamepadData handedness must be left or right")
            if values.shape != (15,) or not np.isfinite(values).all():
                raise ValueError("gamepadData must contain fifteen finite values")
            self.gamepads[side] = values.tolist()
            self.send(
                "gamepad_data",
                {
                    "timestamp": self.timestamp(),
                    "handedness": side,
                    "left_right_gamepad_dict": self.gamepads,
                },
            )
            return 1
        if kind != "handTracking":
            return 0
        published = 0
        for hand in message.get("hands", []):
            side = str(hand.get("handedness", "")).lower()
            if side not in self.hands:
                continue
            self.hands[side] = convert_hand_poses(hand.get("poses"))
            self.last_seen_hand_at[side] = time.monotonic()
            if side == "right":
                continue
            # Only the left hand drives XHand retargeting and xArm motion.
            calibration_event = self.update_calibration(np.asarray(self.hands[side][:7], dtype=np.float64))
            # During the neutral hold the hand packet is deliberately still
            # published so XHand can align with the user's flat hand.  The arm
            # bridge gates xArm motion on the explicit teleop_control/start.
            if not self.armed and not self.calibrating and self.hand_retarget_warm:
                # The arm bridge needs a wrist pose before it can connect and
                # publish its readiness heartbeat. Use a separate topic so
                # the XHand receiver never treats idle tracking as a command.
                # While XHand is still cold the real hand_data topic is used
                # instead, because the vendor client only subscribes to
                # hand_data and its first retarget call JIT-compiles for tens
                # of seconds.  XHand warms up without commanding the hand, and
                # the arm bridge still gates motion on teleop_control/start.
                self.send(
                    "arm_preview",
                    {
                        "timestamp": self.timestamp(),
                        "handedness": side,
                        "left_right_hand_dict": self.hands,
                    },
                )
                return calibration_event or 0
            if self.armed:
                self.last_hand_at = self.last_seen_hand_at[side]
            self.send(
                "hand_data",
                {
                    "timestamp": self.timestamp(),
                    "handedness": side,
                    "left_right_hand_dict": self.hands,
                    "left_right_gamepad_dict": self.gamepads,
                },
            )
            published += 1
            if calibration_event is not None:
                return calibration_event
        return published

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--https-port", type=int, default=8010)
    parser.add_argument("--zmq-bind", default="tcp://*:49510")
    parser.add_argument("--home-status-zmq", default="tcp://127.0.0.1:49513")
    parser.add_argument("--xarm-telemetry-zmq", default="tcp://127.0.0.1:49512")
    parser.add_argument("--recorder-status-zmq", default="tcp://127.0.0.1:49514")
    parser.add_argument("--xhand-home-status-zmq", default="tcp://127.0.0.1:49515")
    parser.add_argument("--tracking-timeout-s", type=float, default=0.6)
    parser.add_argument("--acquisition-timeout-s", type=float, default=5.0)
    parser.add_argument("--calibration-duration-s", type=float, default=3.0)
    args = parser.parse_args()
    if args.tracking_timeout_s <= 0 or args.acquisition_timeout_s <= 0 or args.calibration_duration_s <= 0:
        raise ValueError("Safety timeouts must be positive")

    from aiohttp import WSMsgType, web

    spec = importlib.util.find_spec("server")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("RobotEra webxr package is not installed in this Python environment")
    server_root = Path(next(iter(spec.submodule_search_locations)))
    static_root = server_root / "static"
    html = patched_index(static_root / "index.html")
    publisher = RelayPublisher(
        args.zmq_bind,
        tracking_timeout_s=args.tracking_timeout_s,
        acquisition_timeout_s=args.acquisition_timeout_s,
        calibration_duration_s=args.calibration_duration_s,
    )
    home_status = publisher.context.socket(zmq.SUB)
    home_status.setsockopt(zmq.SUBSCRIBE, b"xarm_home_status")
    home_status.setsockopt(zmq.RCVHWM, 8)
    home_status.connect(args.home_status_zmq)
    arm_heartbeat = publisher.context.socket(zmq.SUB)
    arm_heartbeat.setsockopt(zmq.SUBSCRIBE, b"xarm_telemetry")
    arm_heartbeat.setsockopt(zmq.RCVHWM, 8)
    arm_heartbeat.connect(args.xarm_telemetry_zmq)
    recorder_status = publisher.context.socket(zmq.SUB)
    recorder_status.setsockopt(zmq.SUBSCRIBE, b"recorder_status")
    recorder_status.setsockopt(zmq.RCVHWM, 8)
    recorder_status.connect(args.recorder_status_zmq)
    hand_home_status = publisher.context.socket(zmq.SUB)
    hand_home_status.setsockopt(zmq.SUBSCRIBE, b"xhand_home_status")
    hand_home_status.setsockopt(zmq.RCVHWM, 8)
    hand_home_status.connect(args.xhand_home_status_zmq)
    last_arm_heartbeat = None
    last_recorder_heartbeat = None
    last_hand_heartbeat = None
    recorder_state = None
    recorder_link_ready = False
    clients = set()
    counts = Counter()
    ages_ms = deque(maxlen=1000)
    last_report = time.monotonic()

    async def broadcast_state(reason: str) -> None:
        phase = publisher.phase
        if phase == "idle" and (
            publisher.home_fault is not None
            or any(word in reason.lower() for word in ("fault", "failed", "lost", "unavailable", "not ready", "invalid"))
        ):
            phase = "fault"
        for client in list(clients):
            await client.send_json({
                "type": "teleopState",
                "armed": publisher.armed,
                "phase": phase,
                "countdown": (
                    publisher.home_seconds_remaining
                    if publisher.returning_home
                    else publisher.calibration_countdown
                ),
                "reason": reason,
            })

    async def handle_control_request(request: str, source: str) -> None:
        if request == "calibration_tick":
            await broadcast_state("hold the left hand flat and pointing forward")
            return
        if request == "calibration_reset":
            print("TELEOP_CALIBRATION_RESET: left wrist moved; restarting 3-second hold", flush=True)
            await broadcast_state("left wrist moved; calibration restarted")
            return
        if request == "calibrated":
            now = time.monotonic()
            if last_arm_heartbeat is None or now - last_arm_heartbeat > 1.0:
                print("TELEOP_START_REFUSED: xArm bridge telemetry is unavailable", flush=True)
                await broadcast_state("xArm bridge is not ready; calibration cancelled")
                return
            if last_recorder_heartbeat is None or now - last_recorder_heartbeat > 1.0 or recorder_state != "idle" or not recorder_link_ready:
                print(f"TELEOP_START_REFUSED: recorder state={recorder_state}, heartbeat={last_recorder_heartbeat}", flush=True)
                await broadcast_state("recorder is not ready; calibration cancelled")
                return
            if last_hand_heartbeat is None or now - last_hand_heartbeat > 1.0 or not publisher.hand_home_ready:
                print("TELEOP_START_REFUSED: XHand telemetry is unavailable", flush=True)
                await broadcast_state("XHand is not ready; calibration cancelled")
                return
            if not publisher.hand_retarget_warm:
                print("TELEOP_START_REFUSED: XHand retarget is still warming up", flush=True)
                await broadcast_state("XHand retarget is still warming up; calibration cancelled")
                return
            started, reason = publisher.request_arm("3-second neutral calibration complete; episode started")
            if not started:
                print(f"TELEOP_START_REFUSED: {reason}", flush=True)
            await broadcast_state(reason)
            return
        if request == "toggle":
            if publisher.armed:
                reason = f"{source}; episode completed and is pending review"
                publisher.complete_episode(reason)
                print("EPISODE_PENDING: headset volume +=save, volume -=discard", flush=True)
            elif publisher.pending_episode is not None:
                # Volume + can arrive before the recorder has processed the
                # preceding complete event. Honor that one press after a
                # short status wait rather than making the user press again.
                deadline = time.monotonic() + 2.0
                while (
                    recorder_state not in {"pending", "fault"}
                    and last_recorder_heartbeat is not None
                    and time.monotonic() - last_recorder_heartbeat <= 1.0
                    and time.monotonic() < deadline
                ):
                    await asyncio.sleep(0.05)
                if recorder_state == "pending" and last_recorder_heartbeat is not None and time.monotonic() - last_recorder_heartbeat <= 1.0:
                    reason = f"{source}; pending episode saved"
                    publisher.commit_pending(reason)
                    publisher.begin_return_home(reason)
                    print("EPISODE_SAVED: returning to safe position before the next episode", flush=True)
                elif recorder_state == "fault":
                    reason = "recorder fault: episode has no valid frames; discarded, returning home"
                    publisher.discard_pending(reason)
                    publisher.begin_return_home(reason)
                    print("EPISODE_DISCARDED: recorder rejected the episode", file=sys.stderr, flush=True)
                else:
                    reason = f"recorder not ready to save: {recorder_state}; wait and press volume + again"
                    print(f"EPISODE_SAVE_WAIT: {reason}", flush=True)
            elif publisher.calibrating:
                reason = f"{source}; calibration cancelled"
                publisher.cancel_calibration(reason)
            else:
                now = time.monotonic()
                if last_arm_heartbeat is None or now - last_arm_heartbeat > 1.0:
                    reason = "xArm bridge is unavailable; check the controller and bridge process"
                elif last_recorder_heartbeat is None or now - last_recorder_heartbeat > 1.0:
                    reason = "recorder is unavailable; check the recording process"
                elif recorder_state != "idle" or not recorder_link_ready:
                    reason = f"recorder is not ready: {recorder_state}"
                elif last_hand_heartbeat is None or now - last_hand_heartbeat > 1.0 or not publisher.hand_home_ready:
                    reason = "XHand telemetry is unavailable"
                else:
                    reason = f"{source}; hold left hand flat and forward for 3 seconds"
                    started, reason = publisher.begin_calibration(reason)
                if not publisher.calibrating:
                    print(f"TELEOP_CALIBRATION_REFUSED: {reason}", flush=True)
            await broadcast_state(reason)
            return
        if request == "discard":
            reason = f"{source}; episode discarded"
            discarded = (
                publisher.end_episode("discard", reason)
                if publisher.armed
                else publisher.discard_pending(reason)
            )
            if not discarded:
                discarded = publisher.cancel_calibration(reason)
            elif not publisher.calibrating:
                publisher.begin_return_home(reason)
            print("EPISODE_DISCARDED" if discarded else "DISCARD_IGNORED", flush=True)
            await broadcast_state(reason)

    async def index(_request):
        return web.Response(text=html, content_type="text/html")

    async def websocket(request):
        nonlocal last_report
        ws = web.WebSocketResponse(max_msg_size=2 * 1024 * 1024, heartbeat=10)
        await ws.prepare(request)
        clients.add(ws)
        print(f"USB_CLIENT_CONNECTED count={len(clients)}", flush=True)
        try:
            async for item in ws:
                if item.type != WSMsgType.TEXT:
                    continue
                message = json.loads(item.data)
                kind = str(message.get("type", "unknown"))
                counts[kind] += 1
                if kind == "sessionEnded":
                    reason = "Quest AR session ended; teleoperation stopped"
                    publisher.end_episode("fault", reason)
                    print("USB_AR_SESSION_ENDED: refreshing Quest browser page", flush=True)
                    await broadcast_state(reason)
                    continue
                source_ms = message.get("timestamp")
                if isinstance(source_ms, (int, float)):
                    ages_ms.append(max(0.0, time.time() * 1000 - float(source_ms)))
                result = publisher.publish_message(message)
                if isinstance(result, str):
                    await handle_control_request(result, "left-hand calibration")
                now = time.monotonic()
                if now - last_report >= 5:
                    age = float(np.percentile(ages_ms, 95)) if ages_ms else float("nan")
                    print(f"USB_RELAY_STATS messages={dict(counts)} source_age_p95_ms={age:.1f}", flush=True)
                    counts.clear(); ages_ms.clear(); last_report = now
        finally:
            clients.discard(ws)
            if not clients:
                publisher.end_episode("fault", "USB client disconnected")
            print(f"USB_CLIENT_DISCONNECTED count={len(clients)}", flush=True)
        return ws

    async def safety_watchdog(app):
        nonlocal last_arm_heartbeat, last_recorder_heartbeat, recorder_state, recorder_link_ready
        last_arm_reconnect_at = 0.0
        last_recorder_reconnect_at = 0.0
        last_relay_heartbeat_at = 0.0
        while True:
            while arm_heartbeat.poll(0, zmq.POLLIN):
                arm_heartbeat.recv_multipart()
                last_arm_heartbeat = time.monotonic()
            while recorder_status.poll(0, zmq.POLLIN):
                parts = recorder_status.recv_multipart()
                if len(parts) != 2:
                    continue
                try:
                    payload = json.loads(parts[1])
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                last_recorder_heartbeat = time.monotonic()
                recorder_state = payload.get("state")
                recorder_link_ready = payload.get("relay_link_ready") is True
                was_returning = publisher.returning_home
                publisher.update_recorder_status(payload)
                if was_returning and not publisher.returning_home:
                    await broadcast_state("xArm, XHand, and recorder ready; next episode enabled")
            now = time.monotonic()
            if now - last_relay_heartbeat_at >= 0.1:
                # Level-triggered state mirror.  Downstream processes must not
                # depend on catching a single teleop_control edge, because
                # control events share the publisher with ~90 Hz hand_data.
                publisher.send("relay_status", {
                    "timestamp": publisher.timestamp(),
                    "phase": publisher.phase,
                    "armed": publisher.armed,
                    "episode": publisher.episode_number,
                    "returning_home": publisher.returning_home,
                    "pending": publisher.pending_episode,
                })
                last_relay_heartbeat_at = now
            if (
                (last_arm_heartbeat is None or now - last_arm_heartbeat > 1.0)
                and now - last_arm_reconnect_at > 2.0
            ):
                # A bridge restart can leave the old SUB transport detached.
                # Rebuild only the subscription; never treat this as proof of readiness.
                arm_heartbeat.disconnect(args.xarm_telemetry_zmq)
                arm_heartbeat.connect(args.xarm_telemetry_zmq)
                last_arm_reconnect_at = now
            if (
                (last_recorder_heartbeat is None or now - last_recorder_heartbeat > 1.0)
                and now - last_recorder_reconnect_at > 2.0
            ):
                # The recorder may be restarted while this relay stays alive.
                # Reconnect stale transports, but require a new real heartbeat
                # before allowing calibration to enable motion.
                recorder_status.disconnect(args.recorder_status_zmq)
                recorder_status.connect(args.recorder_status_zmq)
                last_recorder_reconnect_at = now
            reason = publisher.stop_if_stale()
            if reason:
                await broadcast_state(reason)
            if publisher.armed:
                now = time.monotonic()
                started_at = publisher.armed_at or now
                fault = None
                if now - started_at > 1.0 and (
                    last_arm_heartbeat is None or now - last_arm_heartbeat > 1.0
                ):
                    fault = "xArm bridge telemetry lost"
                elif now - started_at > 1.0 and (
                    last_recorder_heartbeat is None or now - last_recorder_heartbeat > 1.0
                ):
                    fault = "recorder heartbeat lost"
                elif recorder_state == "fault" or not recorder_link_ready:
                    fault = "recorder reported a recording fault"
                elif now - started_at > 2.0 and recorder_state != "recording":
                    fault = "recorder did not enter recording state"
                if fault is not None:
                    publisher.end_episode("fault", fault)
                    print(f"TELEOP_INTERLOCK: {fault}", file=sys.stderr, flush=True)
                    await broadcast_state(fault)
            await asyncio.sleep(0.1)

    async def home_status_watchdog(app):
        nonlocal last_hand_heartbeat
        while True:
            while home_status.poll(0, zmq.POLLIN):
                parts = home_status.recv_multipart()
                if len(parts) != 2:
                    continue
                try:
                    payload = json.loads(parts[1])
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                reason = publisher.update_home_status(payload)
                await broadcast_state(reason)
            while hand_home_status.poll(0, zmq.POLLIN):
                parts = hand_home_status.recv_multipart()
                if len(parts) != 2:
                    continue
                try:
                    payload = json.loads(parts[1])
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                last_hand_heartbeat = time.monotonic()
                was_returning = publisher.returning_home
                reason = publisher.update_hand_home_status(payload)
                if was_returning and (not publisher.returning_home or payload.get("state") == "fault"):
                    await broadcast_state(reason)
            await asyncio.sleep(0.05)

    async def volume_button_monitor():
        """Read headset buttons through USB ADB; reconnect after USB loss."""
        last_action_at = 0.0
        disconnected_reported = False
        while True:
            try:
                reverse = await asyncio.create_subprocess_exec(
                    "adb", "-d", "reverse", "tcp:8010", "tcp:8010",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                await asyncio.wait_for(reverse.communicate(), timeout=3.0)
                if reverse.returncode == 0:
                    print("USB_REVERSE_READY: Quest localhost:8010 forwarded to PC", flush=True)
            except (FileNotFoundError, asyncio.TimeoutError):
                if 'reverse' in locals() and reverse.returncode is None:
                    reverse.kill()
                    await reverse.wait()
            try:
                process = await asyncio.create_subprocess_exec(
                    "adb", "-d", "shell", "getevent", "-lt",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            except FileNotFoundError:
                print("VOLUME_BUTTON_ERROR: adb is not installed", file=sys.stderr, flush=True)
                return
            pressed: set[str] = set()
            error_line = None
            detected = False
            try:
                while True:
                    raw = await process.stdout.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    event = parse_volume_event(line)
                    if event is None:
                        if any(word in line.lower() for word in ("denied", "no devices", "not found", "unauthorized")):
                            error_line = line
                        continue
                    if not detected:
                        print("VOLUME_BUTTON_READY: Quest physical volume keys detected over USB", flush=True)
                        detected = True
                        disconnected_reported = False
                    key, is_down = event
                    if not is_down:
                        pressed.discard(key)
                        continue
                    if key in pressed:
                        continue
                    pressed.add(key)
                    now = time.monotonic()
                    if now - last_action_at < 0.35:
                        continue
                    request = volume_button_request(publisher, key)
                    print(
                        f"VOLUME_BUTTON_PRESS: {'+' if key == 'up' else '-'} "
                        f"phase={publisher.phase} request={request or 'ignored'}",
                        flush=True,
                    )
                    if request is not None:
                        last_action_at = now
                        await handle_control_request(request, f"headset volume {'+' if key == 'up' else '-'} pressed")
                    elif key == "down" and publisher.phase == "idle":
                        now = time.monotonic()
                        if last_arm_heartbeat is None or now - last_arm_heartbeat > 1.0:
                            reason = "not recording: xArm bridge is unavailable; check controller error"
                        elif last_recorder_heartbeat is None or now - last_recorder_heartbeat > 1.0:
                            reason = "not recording: recorder is unavailable"
                        elif recorder_state != "idle":
                            reason = f"not recording: recorder state is {recorder_state}"
                        else:
                            reason = "not recording; press volume + to calibrate and start"
                        await broadcast_state(reason)
            finally:
                if process.returncode is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            if not disconnected_reported:
                detail = f": {error_line}" if error_line else ""
                print(f"VOLUME_BUTTON_WAITING: reconnect Quest USB/ADB{detail}", flush=True)
                disconnected_reported = True
            await asyncio.sleep(2.0)

    async def keyboard_control():
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        input_fd = sys.stdin.fileno()
        if not os.isatty(input_fd):
            return
        saved_terminal = termios.tcgetattr(input_fd)
        tty.setcbreak(input_fd)

        def stdin_ready():
            data = os.read(input_fd, 32)
            if data:
                for byte in data:
                    queue.put_nowait(chr(byte))
            else:
                loop.remove_reader(input_fd)

        loop.add_reader(input_fd, stdin_ready)
        try:
            while True:
                key = await queue.get()
                if key in ("\r", "\n"):
                    await handle_control_request("toggle", "Enter pressed")
                elif key.lower() == "r":
                    await handle_control_request("discard", "R pressed")
                elif key.lower() == "q" or key == "\x1b":
                    reason = "Q/Esc pressed; teleoperation stopped without saving"
                    if publisher.armed and publisher.end_episode("abort", reason):
                        await broadcast_state(reason)
                    elif publisher.cancel_calibration(reason):
                        await broadcast_state(reason)
                    elif publisher.commit_pending("Q/Esc pressed; pending episode accepted"):
                        reason = "pending episode saved; recording session finished"
                        print("RECORDING_SESSION_FINISHED", flush=True)
                        await broadcast_state(reason)
                    else:
                        print("TELEOP_ALREADY_STOPPED", flush=True)
        finally:
            loop.remove_reader(input_fd)
            termios.tcsetattr(input_fd, termios.TCSADRAIN, saved_terminal)

    async def start_background_tasks(app):
        app["safety_watchdog"] = asyncio.create_task(safety_watchdog(app))
        app["home_status_watchdog"] = asyncio.create_task(home_status_watchdog(app))
        app["volume_button_monitor"] = asyncio.create_task(volume_button_monitor())
        app["keyboard_control"] = asyncio.create_task(keyboard_control())

    async def cleanup(_app):
        for task_name in ("safety_watchdog", "home_status_watchdog", "volume_button_monitor", "keyboard_control"):
            task = _app.get(task_name)
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        for ws in list(clients):
            await ws.close()
        home_status.close(linger=0)
        hand_home_status.close(linger=0)
        arm_heartbeat.close(linger=0)
        recorder_status.close(linger=0)
        publisher.close()

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket)
    app.router.add_static("/static", static_root)
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup)
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(server_root / "cert.pem", server_root / "key.pem")
    print(
        f"USB_RELAY_READY https://localhost:{args.https_port} "
        f"zmq={args.zmq_bind}",
        flush=True,
    )
    print(
        "TELEOP_CONTROL: headset volume +=calibrate/start or save, "
        "volume -=stop or discard; Enter=keyboard toggle, R=discard, Q/Esc=finish/abort; single presses only",
        flush=True,
    )
    web.run_app(app, host=args.bind, port=args.https_port, ssl_context=ssl_context, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
