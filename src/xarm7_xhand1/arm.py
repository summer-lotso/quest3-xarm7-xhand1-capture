"""Thin xArm7 adapter with explicit motion opt-in."""
from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Sequence
import numpy as np
from .config import HardwareConfig

@dataclass(frozen=True)
class XArmState:
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray
    effort: np.ndarray
    monotonic_time: float

def _check_code(operation: str, result: int | tuple) -> tuple:
    values = result if isinstance(result, tuple) else (result,)
    if not values or int(values[0]) != 0:
        raise RuntimeError(f"xArm {operation} failed: {result!r}")
    return values

class XArm7:
    DOF = 7
    def __init__(self, config: HardwareConfig, *, allow_motion: bool = False):
        self.config, self.allow_motion, self._arm = config, allow_motion, None
    @property
    def connected(self) -> bool:
        return bool(self._arm is not None and self._arm.connected)
    def connect(self) -> None:
        self.config.require_arm()
        if self.connected:
            raise RuntimeError("xArm7 is already connected")
        from xarm.wrapper import XArmAPI
        try:
            self._arm = XArmAPI(self.config.xarm_ip, is_radian=True)
        except Exception as exc:
            self._arm = None
            raise ConnectionError(f"Unable to connect to xArm7 at {self.config.xarm_ip}: {exc}") from exc
        if not self._arm.connected:
            self._arm = None
            raise ConnectionError(f"Unable to connect to xArm7 at {self.config.xarm_ip}")
        result = _check_code("get_err_warn_code", self._arm.get_err_warn_code())
        if result[1][0] != 0:
            raise RuntimeError(f"xArm controller has error/warning codes: {result[1]}")
    def read_state(self) -> XArmState:
        if not self.connected:
            raise RuntimeError("xArm7 is not connected")
        values = _check_code("get_joint_states", self._arm.get_joint_states(is_radian=True, num=3))
        arrays = [np.asarray(item[: self.DOF], dtype=np.float64) for item in values[1]]
        if any(array.shape != (self.DOF,) or not np.isfinite(array).all() for array in arrays):
            raise RuntimeError("xArm returned invalid 7-DOF joint state")
        return XArmState(*arrays, monotonic_time=time.monotonic())
    def read_tcp_offset(self) -> np.ndarray:
        if not self.connected:
            raise RuntimeError("xArm7 is not connected")
        offset = np.asarray(self._arm.tcp_offset, dtype=np.float64)
        if offset.shape != (6,) or not np.isfinite(offset).all():
            raise RuntimeError("xArm returned an invalid TCP offset")
        return offset
    def wait_for_tcp_offset(self, expected: Sequence[float], *, timeout_s: float = 2.0) -> np.ndarray:
        expected_array = np.asarray(expected, dtype=np.float64)
        deadline = time.monotonic() + timeout_s
        while True:
            actual = self.read_tcp_offset()
            if np.allclose(actual, expected_array, atol=1e-5, rtol=0):
                return actual
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"xArm TCP report did not refresh within {timeout_s}s; "
                    f"expected {expected_array.tolist()}, received {actual.tolist()}"
                )
            time.sleep(0.05)
    def read_tcp_pose(self) -> np.ndarray:
        if not self.connected:
            raise RuntimeError("xArm7 is not connected")
        values = _check_code("get_position", self._arm.get_position(is_radian=True))
        pose = np.asarray(values[1], dtype=np.float64)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise RuntimeError("xArm returned an invalid TCP pose")
        return pose
    def configure_tcp_offset(self) -> None:
        if not self.allow_motion:
            raise PermissionError("xArm configuration changes require allow_motion=True")
        if self.config.xarm_tcp_offset_mm_rad is None:
            raise ValueError("xarm_tcp_offset_mm_rad must be configured")
        _check_code(
            "set_tcp_offset",
            self._arm.set_tcp_offset(list(self.config.xarm_tcp_offset_mm_rad), is_radian=True, wait=True),
        )
    def _check_configured_safety_boundary(self, target: np.ndarray) -> None:
        states = _check_code("get_reduced_states", self._arm.get_reduced_states(is_radian=True))[1]
        if len(states) < 6 or not bool(states[5]):
            return
        boundary = states[1]
        pose = _check_code(
            "get_forward_kinematics",
            self._arm.get_forward_kinematics(target.tolist(), input_is_radian=True, return_is_radian=True),
        )[1]
        labels = ("X", "Y", "Z")
        violations = []
        for axis, label in enumerate(labels):
            maximum, minimum, value = float(boundary[axis * 2]), float(boundary[axis * 2 + 1]), float(pose[axis])
            if value < minimum or value > maximum:
                violations.append(f"{label}={value:.1f} mm outside [{minimum:.1f}, {maximum:.1f}]")
        if violations:
            raise ValueError("xArm target violates the controller safety boundary: " + "; ".join(violations))
    def move_joints(
        self,
        target_rad: Sequence[float],
        *,
        wait: bool = True,
        speed_rad_s: float | None = None,
        acceleration_rad_s2: float | None = None,
    ) -> None:
        if not self.allow_motion:
            raise PermissionError("xArm motion is disabled; use allow_motion=True only after operator confirmation")
        target = np.asarray(target_rad, dtype=np.float64)
        if target.shape != (self.DOF,) or not np.isfinite(target).all():
            raise ValueError("xArm target must be 7 finite joint angles in radians")
        if np.any(np.abs(target - self.read_state().position_rad) > self.config.xarm_max_step_rad + 1e-12):
            raise ValueError(f"xArm target exceeds per-command step limit {self.config.xarm_max_step_rad} rad")
        self._check_configured_safety_boundary(target)
        for operation, result in (("motion_enable", self._arm.motion_enable(True)), ("set_mode", self._arm.set_mode(0)), ("set_state", self._arm.set_state(0))):
            _check_code(operation, result)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            state_result = _check_code("get_state", self._arm.get_state())
            error_result = _check_code("get_err_warn_code", self._arm.get_err_warn_code())
            if int(error_result[1][0]) != 0:
                raise RuntimeError(f"xArm controller error after enabling motion: {error_result[1]}")
            # SDK wait_move treats 0/1 as active and 2 as successfully idle;
            # 3 is paused and states >=4 are stopped/error states.
            if int(state_result[1]) in (0, 1, 2):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("xArm did not enter motion state 0 within 2 seconds")
        speed = self.config.xarm_joint_speed_rad_s if speed_rad_s is None else float(speed_rad_s)
        acceleration = self.config.xarm_joint_acceleration_rad_s2 if acceleration_rad_s2 is None else float(acceleration_rad_s2)
        if not np.isfinite(speed) or speed <= 0 or not np.isfinite(acceleration) or acceleration <= 0:
            raise ValueError("xArm joint speed and acceleration must be positive and finite")
        _check_code("set_servo_angle", self._arm.set_servo_angle(angle=target.tolist(), speed=speed, mvacc=acceleration, is_radian=True, wait=wait))
    def ensure_enabled(self) -> None:
        """Re-enable motion after a controller error clear; idempotent."""
        if not self.allow_motion:
            raise PermissionError("xArm motion is disabled; use allow_motion=True only after operator confirmation")
        if not self.connected:
            raise RuntimeError("xArm7 is not connected")
        for operation, result in (
            ("motion_enable", self._arm.motion_enable(True)),
            ("set_mode(position)", self._arm.set_mode(0)),
            ("set_state(ready)", self._arm.set_state(0)),
        ):
            _check_code(operation, result)
    def start_cartesian_servo(self) -> np.ndarray:
        """Enter xArm Cartesian servo mode and return the measured starting pose."""
        if not self.allow_motion:
            raise PermissionError("xArm motion is disabled; use allow_motion=True only after operator confirmation")
        if not self.connected:
            raise RuntimeError("xArm7 is not connected")
        error = _check_code("get_err_warn_code", self._arm.get_err_warn_code())
        if int(error[1][0]) != 0:
            raise RuntimeError(f"xArm controller has an active error: {error[1]}")
        for operation, result in (
            ("motion_enable", self._arm.motion_enable(True)),
            ("set_mode(servo)", self._arm.set_mode(1)),
            ("set_state(motion)", self._arm.set_state(0)),
        ):
            _check_code(operation, result)
        self._last_servo_pose = self.read_tcp_pose()
        reduced = _check_code("get_reduced_states", self._arm.get_reduced_states(is_radian=True))[1]
        self._servo_boundary = reduced[1] if len(reduced) >= 6 and bool(reduced[5]) else None
        return self._last_servo_pose.copy()
    def servo_cartesian(self, target_mm_rad: Sequence[float]) -> None:
        """Send one absolute Cartesian servo target after strict per-packet checks."""
        if not self.allow_motion:
            raise PermissionError("xArm motion is disabled; use allow_motion=True only after operator confirmation")
        if not self.connected:
            raise RuntimeError("xArm7 is not connected")
        target = np.asarray(target_mm_rad, dtype=np.float64)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("xArm Cartesian target must be [x, y, z mm, roll, pitch, yaw rad]")
        previous = getattr(self, "_last_servo_pose", None)
        if previous is None:
            raise RuntimeError("start_cartesian_servo() must be called before servo_cartesian()")
        if np.any(np.abs(target[:3] - previous[:3]) > self.config.xarm_max_cartesian_step_mm + 1e-12):
            raise ValueError(
                f"xArm Cartesian target exceeds per-command translation limit "
                f"{self.config.xarm_max_cartesian_step_mm} mm"
            )
        angle_delta = (target[3:] - previous[3:] + np.pi) % (2 * np.pi) - np.pi
        if np.any(np.abs(angle_delta) > self.config.xarm_max_orientation_step_rad + 1e-12):
            raise ValueError(
                f"xArm Cartesian target exceeds per-command orientation limit "
                f"{self.config.xarm_max_orientation_step_rad} rad"
            )
        boundary = getattr(self, "_servo_boundary", None)
        if boundary is not None:
            for axis, label in enumerate(("X", "Y", "Z")):
                maximum, minimum = float(boundary[axis * 2]), float(boundary[axis * 2 + 1])
                if not minimum <= target[axis] <= maximum:
                    raise ValueError(
                        f"xArm Cartesian target violates controller safety boundary: "
                        f"{label}={target[axis]:.1f} mm outside [{minimum:.1f}, {maximum:.1f}]"
                    )
        _check_code(
            "set_servo_cartesian",
            self._arm.set_servo_cartesian(target.tolist(), is_radian=True),
        )
        self._last_servo_pose = target.copy()
    def hold(self) -> None:
        self.move_joints(self.read_state().position_rad)
    def stop(self) -> None:
        if self.connected and self.allow_motion:
            _check_code("set_state(stop)", self._arm.set_state(4))
    def close(self) -> None:
        if self._arm is not None:
            self._arm.disconnect()
            self._arm = None
    def __enter__(self):
        self.connect(); return self
    def __exit__(self, *_):
        self.close()
