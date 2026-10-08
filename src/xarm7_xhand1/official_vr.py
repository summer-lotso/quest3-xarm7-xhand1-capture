"""Parse RobotEra WebXR ZMQ packets and map wrist motion to xArm poses."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WristPose:
    position_m: np.ndarray
    quaternion_xyzw: np.ndarray


class TrackingLimitError(ValueError):
    """A valid tracking frame lies outside the configured teleop envelope."""


@dataclass
class DeadmanClutch:
    """Fail-closed analog-button clutch with a freshness timeout."""

    threshold: float = 0.5
    timeout_s: float = 0.25
    value: float = 0.0
    updated_at: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0 or self.timeout_s <= 0:
            raise ValueError("Clutch threshold must be in (0, 1] and timeout must be positive")

    def update(self, value: float, now: float) -> None:
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Clutch value must be finite and in [0, 1]")
        self.value = float(value)
        self.updated_at = float(now)

    def active(self, now: float) -> bool:
        return bool(
            self.updated_at is not None
            and now - self.updated_at <= self.timeout_s
            and self.value >= self.threshold
        )


@dataclass
class PinchClutch:
    """Fail-closed pinch clutch with hysteresis and a freshness timeout."""

    engage_distance_m: float = 0.035
    release_distance_m: float = 0.05
    timeout_s: float = 0.6
    updated_at: float | None = None
    engaged: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.engage_distance_m < self.release_distance_m or self.timeout_s <= 0:
            raise ValueError("Pinch distances must satisfy 0 < engage < release and timeout must be positive")

    def update(self, distance_m: float, now: float) -> None:
        if not np.isfinite(distance_m) or distance_m < 0:
            raise ValueError("Pinch distance must be finite and non-negative")
        if self.engaged:
            self.engaged = distance_m < self.release_distance_m
        else:
            self.engaged = distance_m <= self.engage_distance_m
        self.updated_at = float(now)

    def active(self, now: float) -> bool:
        return bool(
            self.engaged
            and self.updated_at is not None
            and now - self.updated_at <= self.timeout_s
        )


def decode_hand_packet(topic: bytes, payload: bytes, hand: str) -> WristPose | None:
    """Decode the wrist from active hand data or idle arm preview."""
    if topic not in (b"hand_data", b"arm_preview"):
        return None
    message: dict[str, Any] = json.loads(payload.decode("utf-8"))
    handedness = str(message.get("handedness", "")).lower()
    hand = hand.lower()
    if handedness != hand:
        return None
    hands = message.get("left_right_hand_dict")
    if not isinstance(hands, dict) or hand not in hands:
        raise ValueError(f"Official hand packet has no {hand!r} hand data")
    values = np.asarray(hands[hand], dtype=np.float64)
    if values.ndim != 1 or values.size < 7 or values.size % 7 != 0 or not np.isfinite(values).all():
        raise ValueError("Official hand packet must contain flattened 7-value joint poses")
    position = values[:3]
    quaternion = values[3:7]
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8 or abs(norm - 1.0) > 0.1:
        raise ValueError(f"Official wrist quaternion has invalid norm {norm:.6f}")
    return WristPose(position.copy(), quaternion / norm)


def decode_gamepad_trigger(topic: bytes, payload: bytes, hand: str) -> float | None:
    """Decode the primary trigger value from an official ``gamepad_data`` packet."""
    if topic != b"gamepad_data":
        return None
    message: dict[str, Any] = json.loads(payload.decode("utf-8"))
    hand = hand.lower()
    if str(message.get("handedness", "")).lower() != hand:
        return None
    gamepads = message.get("left_right_gamepad_dict")
    if not isinstance(gamepads, dict) or hand not in gamepads:
        raise ValueError(f"Official gamepad packet has no {hand!r} controller data")
    values = np.asarray(gamepads[hand], dtype=np.float64)
    if values.ndim != 1 or values.size < 8 or not np.isfinite(values).all():
        raise ValueError("Official gamepad packet must contain at least eight finite values")
    trigger = float(values[7])
    if not 0.0 <= trigger <= 1.0:
        raise ValueError("Official gamepad trigger must be in [0, 1]")
    return trigger


def decode_pinch_distance(topic: bytes, payload: bytes, hand: str) -> float | None:
    """Return thumb-tip to index-tip distance in metres for one tracked hand."""
    if topic != b"hand_data":
        return None
    message: dict[str, Any] = json.loads(payload.decode("utf-8"))
    hand = hand.lower()
    if str(message.get("handedness", "")).lower() != hand:
        return None
    hands = message.get("left_right_hand_dict")
    if not isinstance(hands, dict) or hand not in hands:
        raise ValueError(f"Official hand packet has no {hand!r} hand data")
    values = np.asarray(hands[hand], dtype=np.float64)
    if values.ndim != 1 or values.size < 25 * 7 or values.size % 7 != 0 or not np.isfinite(values).all():
        raise ValueError("Official hand packet must contain at least 25 flattened joint poses")
    joints = values.reshape(-1, 7)
    return float(np.linalg.norm(joints[4, :3] - joints[9, :3]))


def _quaternion_to_matrix_xyzw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_rpy(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to xArm fixed-axis roll, pitch, yaw."""
    pitch = np.arctan2(-matrix[2, 0], np.hypot(matrix[0, 0], matrix[1, 0]))
    if abs(abs(pitch) - np.pi / 2) < 1e-7:
        roll = np.arctan2(-matrix[0, 1], matrix[1, 1])
        yaw = 0.0
    else:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float64)


class RelativeWristMapper:
    """Map Quest wrist deltas onto an xArm TCP pose without an initial jump."""

    # WebXR uses -Z as forward, +X as right and +Y as up.  The xArm base
    # frame used here is +X forward, +Y left and +Z up.
    _VR_TO_ROBOT = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def __init__(
        self,
        initial_wrist: WristPose,
        initial_tcp_mm_rad: np.ndarray,
        *,
        translation_scale: float = 1.0,
        max_translation_mm: float = 250.0,
        max_rotation_rad: float = 1.2,
    ) -> None:
        tcp = np.asarray(initial_tcp_mm_rad, dtype=np.float64)
        if tcp.shape != (6,) or not np.isfinite(tcp).all():
            raise ValueError("Initial xArm TCP pose must contain six finite values")
        if translation_scale <= 0 or max_translation_mm <= 0 or max_rotation_rad <= 0:
            raise ValueError("Mapper scales and limits must be positive")
        self.initial_position = np.asarray(initial_wrist.position_m, dtype=np.float64)
        self.initial_vr_rotation = _quaternion_to_matrix_xyzw(initial_wrist.quaternion_xyzw)
        self.initial_tcp = tcp.copy()
        self.initial_tcp_rotation = self._rpy_to_matrix(tcp[3:])
        self.translation_scale = float(translation_scale)
        self.max_translation_mm = float(max_translation_mm)
        self.max_rotation_rad = float(max_rotation_rad)

    @staticmethod
    def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = rpy
        cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )

    def target(self, wrist: WristPose) -> np.ndarray:
        delta_mm = self._VR_TO_ROBOT @ (wrist.position_m - self.initial_position) * (1000.0 * self.translation_scale)
        if float(np.linalg.norm(delta_mm)) > self.max_translation_mm:
            raise TrackingLimitError("Quest wrist translation exceeds configured workspace offset")
        current_vr_rotation = _quaternion_to_matrix_xyzw(wrist.quaternion_xyzw)
        vr_delta = current_vr_rotation @ self.initial_vr_rotation.T
        robot_delta = self._VR_TO_ROBOT @ vr_delta @ self._VR_TO_ROBOT.T
        angle = float(np.arccos(np.clip((np.trace(robot_delta) - 1.0) / 2.0, -1.0, 1.0)))
        if angle > self.max_rotation_rad:
            raise TrackingLimitError("Quest wrist rotation exceeds configured orientation offset")
        target = self.initial_tcp.copy()
        target[:3] += delta_mm
        target[3:] = _matrix_to_rpy(robot_delta @ self.initial_tcp_rotation)
        return target


def limit_servo_step(previous: np.ndarray, desired: np.ndarray, translation_mm: float, rotation_rad: float) -> np.ndarray:
    """Rate-limit a Cartesian command, including wrapped Euler deltas."""
    previous = np.asarray(previous, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    if previous.shape != (6,) or desired.shape != (6,):
        raise ValueError("Cartesian poses must contain six values")
    result = previous.copy()
    result[:3] += np.clip(desired[:3] - previous[:3], -translation_mm, translation_mm)
    delta = (desired[3:] - previous[3:] + np.pi) % (2 * np.pi) - np.pi
    result[3:] += np.clip(delta, -rotation_rad, rotation_rad)
    result[3:] = (result[3:] + np.pi) % (2 * np.pi) - np.pi
    return result
