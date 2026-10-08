"""XHand1 adapter; communication failures never fall back to a stub."""
from __future__ import annotations
from dataclasses import dataclass
import os, time
from typing import Sequence
import numpy as np
from .config import HardwareConfig

@dataclass(frozen=True)
class XHandState:
    position_rad: np.ndarray
    torque_raw: np.ndarray
    temperature_raw: np.ndarray
    monotonic_time: float

def _check_error(operation: str, error: object) -> None:
    code = int(getattr(error, "error_code", -1))
    if code != 0:
        raise RuntimeError(f"XHand {operation} failed ({code}): {getattr(error, 'error_message', '')}")

def has_cap_net_raw() -> bool:
    if os.name != "posix": return False
    if os.geteuid() == 0: return True
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            return any(line.startswith("CapEff:") and bool(int(line.split()[1], 16) & (1 << 13)) for line in status)
    except OSError: return False

class XHand1:
    DOF = 12
    # XHand1 vendor limits in joint order. The source LeRobot driver locally
    # raised joints 5/7/9/11 to 5 degrees, but the vendor Palm preset and this
    # device's valid feedback use values down to 0 degrees.
    JOINT_LIMITS_RAD = np.radians(np.asarray([
        [0, 105], [-60, 90], [-10, 105], [-10, 10], [0, 110], [0, 110],
        [0, 110], [0, 110], [0, 110], [0, 110], [0, 110], [0, 110],
    ], dtype=np.float64))
    def __init__(self, config: HardwareConfig, *, allow_motion: bool = False):
        self.config, self.allow_motion, self._device = config, allow_motion, None
        self._last_target_rad: np.ndarray | None = None
    @property
    def connected(self) -> bool: return self._device is not None
    def connect(self) -> None:
        self.config.require_hand()
        if self.connected: raise RuntimeError("XHand1 is already connected")
        from xhand_controller import xhand_control
        if self.config.xhand_protocol == "EtherCAT" and not has_cap_net_raw():
            raise PermissionError("EtherCAT requires CAP_NET_RAW on the active Python interpreter")
        if self.config.xhand_protocol == "RS485":
            port = self.config.xhand_serial_port
            if not os.path.exists(port):
                raise FileNotFoundError(f"XHand serial port does not exist: {port}")
            if not os.access(port, os.R_OK | os.W_OK):
                raise PermissionError(
                    f"No read/write permission for {port}. Add the current user to the device's "
                    "dialout group, then log out and back in."
                )
        device = xhand_control.XHandControl()
        try:
            error = device.open_ethercat(self.config.ethercat_interface) if self.config.xhand_protocol == "EtherCAT" else device.open_serial(self.config.xhand_serial_port, self.config.xhand_baud_rate)
            _check_error("open", error)
            ids = list(device.list_hands_id())
            if self.config.xhand_hand_id not in ids:
                raise ConnectionError(f"Configured XHand ID {self.config.xhand_hand_id} not found; detected {ids}")
        except Exception:
            device.close_device(); raise
        self._device = device
        self._last_target_rad = np.clip(
            self.read_state().position_rad,
            self.JOINT_LIMITS_RAD[:, 0],
            self.JOINT_LIMITS_RAD[:, 1],
        )
    def read_state(self, *, force_update: bool = True) -> XHandState:
        if not self.connected: raise RuntimeError("XHand1 is not connected")
        for attempt in range(1, self.config.xhand_state_read_retries + 1):
            error, state = self._device.read_state(self.config.xhand_hand_id, force_update)
            code = int(getattr(error, "error_code", -1))
            if code == 0:
                break
            # 1501070 is the vendor CRC error. Retry only this known transient;
            # all other communication failures remain immediately fatal.
            if code != 1501070 or attempt == self.config.xhand_state_read_retries:
                _check_error("read_state", error)
            time.sleep(self.config.xhand_state_retry_delay_s)
        fingers = state.finger_state
        arrays = [np.asarray([getattr(fingers[i], attr) for i in range(self.DOF)], dtype=np.float64) for attr in ("position", "torque", "temperature")]
        if any(a.shape != (self.DOF,) or not np.isfinite(a).all() for a in arrays):
            raise RuntimeError("XHand returned invalid 12-DOF state")
        return XHandState(*arrays, monotonic_time=time.monotonic())
    def _send(self, target_rad: Sequence[float], *, mode: int, enforce_step: bool) -> None:
        if not self.allow_motion: raise PermissionError("XHand commands are disabled; use allow_motion=True only after operator confirmation")
        self.config.require_hand_motion()
        target = np.asarray(target_rad, dtype=np.float64)
        if target.shape != (self.DOF,) or not np.isfinite(target).all(): raise ValueError("XHand target must be 12 finite joint angles in radians")
        if np.any(target < self.JOINT_LIMITS_RAD[:, 0]) or np.any(target > self.JOINT_LIMITS_RAD[:, 1]):
            raise ValueError("XHand target exceeds configured absolute joint limits")
        measured = self.read_state().position_rad
        if enforce_step:
            if self._last_target_rad is None:
                self._last_target_rad = np.clip(
                    measured,
                    self.JOINT_LIMITS_RAD[:, 0],
                    self.JOINT_LIMITS_RAD[:, 1],
                )
            if np.any(np.abs(target - self._last_target_rad) > self.config.xhand_max_step_rad + 1e-12):
                raise ValueError(f"XHand target exceeds per-command step limit {self.config.xhand_max_step_rad} rad")
            tracking_error = np.abs(target - measured)
            if np.any(tracking_error > self.config.xhand_max_tracking_error_rad + 1e-12):
                joint = int(np.argmax(tracking_error))
                raise RuntimeError(
                    "XHand measured position is not following the command; "
                    f"joint {joint} error {tracking_error[joint]:.6f} rad exceeds "
                    f"{self.config.xhand_max_tracking_error_rad} rad"
                )
        from xhand_controller import xhand_control
        command = xhand_control.HandCommand_t()
        for index in range(self.DOF):
            finger = command.finger_command[index]
            finger.id = index
            finger.kp, finger.ki, finger.kd = self.config.xhand_kp, self.config.xhand_ki, self.config.xhand_kd
            finger.position, finger.tor_max, finger.mode = float(target[index]), self.config.xhand_tor_max, mode
        _check_error("send_command", self._device.send_command(self.config.xhand_hand_id, command))
        self._last_target_rad = target.copy()
    def move_joints(self, target_rad: Sequence[float]) -> None: self._send(target_rad, mode=3, enforce_step=True)
    def hold(self) -> None: self._send(np.clip(self.read_state().position_rad, self.JOINT_LIMITS_RAD[:, 0], self.JOINT_LIMITS_RAD[:, 1]), mode=3, enforce_step=False)
    def powerless(self) -> None: self._send(self.read_state().position_rad, mode=0, enforce_step=False)
    def close(self) -> None:
        if self._device is not None: self._device.close_device(); self._device = None
        self._last_target_rad = None
    def __enter__(self): self.connect(); return self
    def __exit__(self, *_): self.close()
