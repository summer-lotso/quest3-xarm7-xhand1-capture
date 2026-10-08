"""Configuration loading and validation."""
from __future__ import annotations
from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import Any
import yaml

@dataclass(frozen=True)
class HardwareConfig:
    xarm_ip: str | None = None
    xarm_home_position_rad: tuple[float, ...] | None = None
    xarm_teleop_ready_position_rad: tuple[float, ...] | None = None
    xarm_tcp_offset_mm_rad: tuple[float, ...] | None = None
    xarm_max_step_rad: float = 0.01
    xarm_joint_speed_rad_s: float = 0.05
    xarm_joint_acceleration_rad_s2: float = 0.1
    xarm_max_cartesian_step_mm: float = 3.0
    xarm_max_orientation_step_rad: float = 0.03
    xhand_protocol: str = "EtherCAT"
    ethercat_interface: str | None = None
    xhand_serial_port: str | None = None
    xhand_baud_rate: int = 3_000_000
    xhand_hand_id: int = 0
    xhand_tor_max: int | None = None
    xhand_kp: int = 80
    xhand_ki: int = 0
    xhand_kd: int = 0
    xhand_home_position_deg: tuple[float, ...] | None = None
    xhand_teleop_ready_position_deg: tuple[float, ...] | None = None
    xhand_max_step_rad: float = 0.01
    xhand_max_tracking_error_rad: float = 0.08
    xhand_state_read_retries: int = 3
    xhand_state_retry_delay_s: float = 0.1
    max_state_age_s: float = 0.2
    camera_fps: int = 30
    image_width: int = 640
    image_height: int = 480
    head_camera_serial: str | None = None
    front_left_camera_serial: str | None = None
    wrist_camera_serial: str | None = None
    head_camera_rotation_deg: int = 0
    front_left_camera_rotation_deg: int = 0
    wrist_camera_rotation_deg: int = 0
    policy_server_host: str | None = None
    policy_server_port: int = 8000
    policy_fps: int = 20

    def __post_init__(self) -> None:
        protocol = self.xhand_protocol.upper()
        if protocol not in {"ETHERCAT", "RS485"}:
            raise ValueError("xhand_protocol must be EtherCAT or RS485")
        object.__setattr__(self, "xhand_protocol", "EtherCAT" if protocol == "ETHERCAT" else "RS485")
        for name, value, length in (
            ("xarm_home_position_rad", self.xarm_home_position_rad, 7),
            ("xarm_teleop_ready_position_rad", self.xarm_teleop_ready_position_rad, 7),
            ("xarm_tcp_offset_mm_rad", self.xarm_tcp_offset_mm_rad, 6),
            ("xhand_home_position_deg", self.xhand_home_position_deg, 12),
            ("xhand_teleop_ready_position_deg", self.xhand_teleop_ready_position_deg, 12),
        ):
            if value is not None and len(value) != length:
                raise ValueError(f"{name} must contain {length} values")
            if value is not None and not all(math.isfinite(float(item)) for item in value):
                raise ValueError(f"{name} must contain only finite values")
        if self.xhand_tor_max is not None and not 0 < self.xhand_tor_max <= 65535:
            raise ValueError("xhand_tor_max must be a vendor raw uint16 value in [1, 65535]")
        if (
            self.xarm_max_step_rad <= 0
            or self.xarm_max_cartesian_step_mm <= 0
            or self.xarm_max_orientation_step_rad <= 0
            or self.xhand_max_step_rad <= 0
            or self.max_state_age_s <= 0
        ):
            raise ValueError("step limits and max_state_age_s must be positive")
        if self.xhand_max_tracking_error_rad < self.xhand_max_step_rad:
            raise ValueError(
                "xhand_max_tracking_error_rad must be at least xhand_max_step_rad"
            )
        if self.xhand_state_read_retries < 1 or self.xhand_state_retry_delay_s < 0:
            raise ValueError("XHand read retries must be >= 1 and retry delay must be non-negative")
        for name in ("head_camera_rotation_deg", "front_left_camera_rotation_deg", "wrist_camera_rotation_deg"):
            if getattr(self, name) not in {0, 90, 180, 270}:
                raise ValueError(f"{name} must be one of 0, 90, 180, 270")

    def require_arm(self) -> None:
        if not self.xarm_ip:
            raise ValueError("xarm_ip is required in the local hardware config")

    def require_hand(self) -> None:
        if self.xhand_protocol == "EtherCAT" and not self.ethercat_interface:
            raise ValueError("ethercat_interface is required for EtherCAT")
        if self.xhand_protocol == "RS485" and not self.xhand_serial_port:
            raise ValueError("xhand_serial_port is required for RS485")

    def require_hand_motion(self) -> None:
        self.require_hand()
        if self.xhand_tor_max is None:
            raise ValueError("xhand_tor_max must be explicitly configured before sending commands")

def load_config(path: str | Path) -> HardwareConfig:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Hardware config not found: {path}")
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed = {field.name for field in fields(HardwareConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown hardware config keys: {', '.join(unknown)}")
    for key in (
        "xarm_home_position_rad",
        "xarm_teleop_ready_position_rad",
        "xarm_tcp_offset_mm_rad",
        "xhand_home_position_deg",
        "xhand_teleop_ready_position_deg",
    ):
        if raw.get(key) is not None:
            raw[key] = tuple(float(v) for v in raw[key])
    return HardwareConfig(**raw)
