"""Read-only xArm readiness checks shared by the CLI and Quest launcher."""

from __future__ import annotations

from typing import Any

import numpy as np

from .arm import XArm7
from .config import HardwareConfig


def evaluate_arm_readiness(report: dict[str, Any], expected_tcp_offset: tuple[float, ...]) -> list[str]:
    """Return reasons that prevent starting a new teleoperation session."""
    problems: list[str] = []
    if report["state"] not in (0, 2, 4):
        problems.append(f"controller state {report['state']} is moving, paused or faulted")
    if report["error"] or report["warning"]:
        problems.append(f"controller error/warning={report['error']}/{report['warning']}")
    if any(code != 0 for code in report["servo_codes"]):
        problems.append(f"servo fault codes={report['servo_codes']}")
    if not report["safety_boundary_enabled"]:
        problems.append("controller TCP safety boundary is disabled")
    if not report["collision_rebound_enabled"]:
        problems.append("controller collision rebound is disabled")
    if not np.allclose(report["tcp_offset_mm_rad"], expected_tcp_offset, atol=1e-3, rtol=0):
        problems.append("controller TCP offset differs from hardware.local.yaml")
    if max(abs(value) for value in report["joint_velocity_rad_s"]) > 0.01:
        problems.append("xArm joints are still moving")
    if report["safety_boundary_enabled"]:
        for axis, label in enumerate(("X", "Y", "Z")):
            maximum = report["boundary_mm"][axis * 2]
            minimum = report["boundary_mm"][axis * 2 + 1]
            value = report["tcp_pose_mm_rad"][axis]
            if not minimum + 2.0 <= value <= maximum - 2.0:
                problems.append(f"TCP {label}={value:.1f} mm is outside the safety boundary margin")
    return problems


def check_arm_readiness(config: HardwareConfig) -> dict[str, Any]:
    """Inspect the real controller without sending a motion or state command."""
    config.require_arm()
    if config.xarm_tcp_offset_mm_rad is None:
        raise ValueError("xarm_tcp_offset_mm_rad must be configured")
    with XArm7(config) as arm:
        controller = arm._arm
        state_code, state = controller.get_state()
        error_code, error_warning = controller.get_err_warn_code()
        servo_code, servos = controller.get_servo_debug_msg()
        reduced_code, reduced = controller.get_reduced_states(is_radian=True)
        if state_code or error_code or servo_code or reduced_code:
            raise RuntimeError(
                f"xArm status read failed: state={state_code}, error={error_code}, "
                f"servo={servo_code}, reduced={reduced_code}"
            )
        if len(servos) < 7 or len(reduced) < 7 or len(reduced[1]) != 6:
            raise RuntimeError("xArm returned an incomplete servo or safety-boundary report")
        joints = arm.read_state()
        report: dict[str, Any] = {
            "state": int(state),
            "mode": int(controller.mode),
            "error": int(error_warning[0]),
            "warning": int(error_warning[1]),
            "servo_codes": [int(item["code"]) for item in servos[:7]],
            "reduced_mode_enabled": bool(reduced[0]),
            "safety_boundary_enabled": bool(reduced[5]),
            "collision_rebound_enabled": bool(reduced[6]),
            "boundary_mm": [float(value) for value in reduced[1]],
            "joint_position_rad": joints.position_rad.tolist(),
            "joint_velocity_rad_s": joints.velocity_rad_s.tolist(),
            "tcp_pose_mm_rad": arm.read_tcp_pose().tolist(),
            "tcp_offset_mm_rad": arm.read_tcp_offset().tolist(),
        }
    problems = evaluate_arm_readiness(report, config.xarm_tcp_offset_mm_rad)
    report["ready_for_new_session"] = not problems
    report["problems"] = problems
    return report
