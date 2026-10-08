#!/usr/bin/env python3
"""Start and supervise the Quest 3 -> XHand1/xArm7 -> recorder stack.

The headset volume buttons control episodes.  Any process failure stops the
whole stack; restarting this command always creates a fresh relay state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import yaml

from _common import require_motion_confirmation
from xarm7_xhand1.runtime_config import load_runtime_config


ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT.parent / "teleop_software_pkg"
VENDOR_PYTHON = Path.home() / "miniconda3/envs/xhand_tele_env_310/bin/python"
LEROBOT_SRC = ROOT.parent / "lerobot-v21/src"
PORTS = (8010, 49510, 49511, 49512, 49513, 49514, 49515)
PROXIMITY_CLOSE_ACTION = "com.oculus.vrpowermanager.prox_close"
PROXIMITY_RESTORE_ACTION = "com.oculus.vrpowermanager.automation_disable"


@dataclass
class Child:
    name: str
    process: subprocess.Popen[str]
    ready_marker: str
    ready: threading.Event = field(default_factory=threading.Event)
    failure: str | None = None
    reader: threading.Thread | None = None


def check_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is occupied; stop the previous Quest capture stack first") from exc


def check_adb(adb: str) -> None:
    result = subprocess.run([adb, "devices", "-l"], capture_output=True, text=True, timeout=5, check=True)
    devices = re.findall(r"^\S+\s+device\b", result.stdout, flags=re.MULTILINE)
    if len(devices) != 1:
        raise RuntimeError(f"Expected one authorized Quest over ADB, found {len(devices)}: {result.stdout.strip()}")


def set_headset_proximity_override(adb: str, *, keep_awake: bool) -> None:
    """Set/clear Meta's ADB proximity override for a fixed, unworn headset."""
    action = PROXIMITY_CLOSE_ACTION if keep_awake else PROXIMITY_RESTORE_ACTION
    result = subprocess.run(
        [adb, "shell", "am", "broadcast", "-a", action],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Could not set Quest proximity state ({action}): {detail}")


def lerobot_environment(lerobot_src: Path = LEROBOT_SRC) -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(lerobot_src), str(ROOT / "src")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def vendor_command(python: Path, *arguments: str) -> list[str]:
    """Activate the vendor Conda environment, then replace bash with Python."""
    prefix = python.parent.parent
    conda_sh = prefix.parent.parent / "etc/profile.d/conda.sh"
    if not conda_sh.is_file():
        raise RuntimeError(f"Conda activation script not found: {conda_sh}")
    return [
        "bash", "-c",
        'set -e; source "$1"; conda activate "$2"; shift 2; exec "$@"',
        "vendor-env", str(conda_sh), str(prefix), str(python), *arguments,
    ]


def preflight(args: argparse.Namespace) -> str:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Run this launcher with the xarm7-xhand1-deploy Python 3.10 environment")
    if not args.vendor_python.is_file():
        raise RuntimeError(f"Vendor Python not found: {args.vendor_python}")
    if args.format == "lerobot-v2.1":
        if not (args.lerobot_src / "lerobot/datasets/lerobot_dataset.py").is_file():
            raise RuntimeError(f"Local LeRobot v2.1 source not found: {args.lerobot_src}")
        result = subprocess.run(
            vendor_command(
                args.vendor_python, "-c",
                "import os; from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION; "
                "print(CODEBASE_VERSION); print(os.environ.get('CONDA_PREFIX', ''))",
            ),
            env=lerobot_environment(args.lerobot_src), capture_output=True, text=True, timeout=15,
        )
        expected = ["v2.1", str(args.vendor_python.parent.parent)]
        if result.returncode != 0 or result.stdout.strip().splitlines()[-2:] != expected:
            raise RuntimeError(f"Vendor Python cannot load local LeRobot v2.1: {result.stderr.strip()}")
    vendor_config = args.vendor_config
    if not vendor_config.is_file():
        raise RuntimeError(f"Vendor USB config not found: {vendor_config}")
    from xarm7_xhand1 import load_config
    from xarm7_xhand1.arm_check import check_arm_readiness

    hardware = load_config(args.hardware_config)
    hardware.require_arm()
    hardware.require_hand_motion()
    if hardware.xarm_home_position_rad is None or hardware.xhand_home_position_deg is None:
        raise RuntimeError("Both safe home poses must be configured")
    vendor = yaml.safe_load(vendor_config.read_text(encoding="utf-8"))
    if vendor["meta_quest"].get("start_web") is not False:
        raise RuntimeError("Vendor USB config must have meta_quest.start_web: false")
    if vendor["hands_count"].get("use_hand_b"):
        raise RuntimeError("This launcher expects exactly one XHand")
    hand = vendor["hands_count"]["hand_a"]
    vendor_serial = Path(hand["rs485_serial_port"])
    local_serial = Path(hardware.xhand_serial_port)
    if not vendor_serial.exists() or not local_serial.exists() or vendor_serial.resolve() != local_serial.resolve():
        raise RuntimeError(f"XHand serial paths disagree or are disconnected: {vendor_serial}, {local_serial}")
    if not os.access(vendor_serial, os.R_OK | os.W_OK):
        raise RuntimeError(f"XHand serial port is not readable/writable by this user: {vendor_serial}")
    for key in ("auth_info_file_path", "key_file_path"):
        credential = Path(hand[key])
        if not credential.is_absolute():
            credential = args.vendor_dir / credential
        if not credential.is_file():
            raise RuntimeError(f"XHand vendor credential missing: {credential}")
    for port in PORTS:
        check_port_available(port)
    adb = shutil.which("adb")
    if not adb:
        raise RuntimeError("adb was not found on PATH")
    check_adb(adb)
    dataset_root = args.root if args.root.is_absolute() else ROOT / args.root
    disk = dataset_root
    while not disk.exists():
        disk = disk.parent
    free_gib = shutil.disk_usage(disk).free / (1024 ** 3)
    minimum_gib = 8 if not args.no_cameras else 1
    if free_gib < minimum_gib:
        raise RuntimeError(f"Dataset disk has only {free_gib:.1f} GiB free; need at least {minimum_gib} GiB")
    print(f"PREFLIGHT_DISK free_gib={free_gib:.1f} root={dataset_root}", flush=True)
    if not args.no_cameras:
        import pyrealsense2 as rs

        available = {device.get_info(rs.camera_info.serial_number) for device in rs.context().query_devices()}
        for name in ("head_camera_serial", "front_left_camera_serial", "wrist_camera_serial"):
            serial = getattr(hardware, name)
            if not serial:
                continue
            if serial.startswith("/dev/"):
                if not Path(serial).exists():
                    raise RuntimeError(f"Configured camera {name} is disconnected: {serial}")
            elif serial not in available:
                raise RuntimeError(f"Configured RealSense {name} is disconnected: {serial}")
    arm_report = check_arm_readiness(hardware)
    print("PREFLIGHT_ARM " + json.dumps(arm_report, separators=(",", ":")), flush=True)
    if not arm_report["ready_for_new_session"]:
        raise RuntimeError("xArm check failed: " + "; ".join(arm_report["problems"]))
    print("PREFLIGHT_OK: Quest, XHand serial, cameras, xArm, config and ports", flush=True)
    return adb


def stop_child(child: Child, hardware_config: Path = ROOT / "configs/hardware.local.yaml") -> bool:
    if child.process.poll() is not None:
        return True
    try:
        os.killpg(child.process.pid, signal.SIGINT)
    except ProcessLookupError:
        return True
    try:
        child.process.wait(timeout=10)
        return True
    except subprocess.TimeoutExpired:
        print(f"STOP_TIMEOUT {child.name}: escalating shutdown (pid={child.process.pid})", file=sys.stderr, flush=True)
        if child.name == "xarm":
            # The bridge normally stops in its finally block.  If it is hung,
            # send a separate controller stop before terminating the process.
            try:
                from xarm.wrapper import XArmAPI
                from xarm7_xhand1 import load_config

                ip = load_config(hardware_config).xarm_ip
                arm = XArmAPI(ip, is_radian=True)
                try:
                    print(f"DIRECT_ARM_STOP result={arm.set_state(4)} state={arm.get_state()}", flush=True)
                finally:
                    arm.disconnect()
            except Exception as exc:
                print(f"DIRECT_ARM_STOP_FAILED: {exc}", file=sys.stderr, flush=True)
        try:
            os.killpg(child.process.pid, signal.SIGTERM)
            child.process.wait(timeout=3)
            return True
        except (ProcessLookupError, subprocess.TimeoutExpired):
            print(f"STOP_INCOMPLETE {child.name}: inspect process pid={child.process.pid} before moving the robot", file=sys.stderr, flush=True)
            return False


def main() -> int:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--runtime-config", type=Path)
    selected, _ = probe.parse_known_args()
    defaults = load_runtime_config(selected.runtime_config) if selected.runtime_config else {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Read-only deployment preflight; do not start any control process")
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "configs/hardware.local.yaml")
    parser.add_argument("--vendor-dir", type=Path, default=VENDOR_DIR)
    parser.add_argument("--vendor-config", type=Path, default=Path("config_meta_quest_usb.yaml"))
    parser.add_argument("--lerobot-src", type=Path, default=LEROBOT_SRC)
    parser.add_argument("--task", required="task" not in defaults)
    parser.add_argument("--root", type=Path, required="root" not in defaults)
    parser.add_argument("--rate-hz", type=int, default=20)
    parser.add_argument("--format", choices=("lerobot-v2.1", "npz"), default="lerobot-v2.1")
    parser.add_argument("--repo-id", default="local/xarm7_xhand1_quest3")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument(
        "--no-wear",
        action="store_true",
        help="Keep a fixed Quest awake by overriding the proximity sensor; restore the sensor on exit.",
    )
    parser.add_argument("--vendor-python", type=Path, default=VENDOR_PYTHON)
    # Middle profile between the first-test conservative limits and the
    # hardware ceiling: 30 mm/s and 0.45 rad/s at 30 Hz, 300 mm / 2.0 rad
    # envelope.  45 mm/s outran the controller (19.3 mm lead) and tripped the
    # 15 mm tracking-error interlock.  The bridge still clamps against
    # hardware.local.yaml's per-command limits (95% of them).
    parser.add_argument("--translation-scale", type=float, default=0.6)
    parser.add_argument("--max-translation-mm", type=float, default=300.0)
    parser.add_argument("--max-rotation-rad", type=float, default=2.0)
    parser.add_argument("--max-step-mm", type=float, default=1.0)
    parser.add_argument("--max-orientation-step-rad", type=float, default=0.015)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    for field in ("hardware_config", "vendor_dir", "vendor_python", "lerobot_src"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    args.vendor_config = args.vendor_config.expanduser()
    if not args.vendor_config.is_absolute():
        args.vendor_config = args.vendor_dir / args.vendor_config
    if not args.task.strip() or args.rate_hz <= 0:
        parser.error("--task must be nonempty and --rate-hz must be positive")
    for field in ("translation_scale", "max_translation_mm", "max_rotation_rad", "max_step_mm", "max_orientation_step_rad"):
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive and finite")
    args.root = args.root.expanduser()
    dataset_root = args.root if args.root.is_absolute() else ROOT / args.root
    children: list[Child] = []
    log_path = ROOT / "logs" / f"quest3_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_lock = threading.Lock()
    exit_code = 0
    adb: str | None = None
    proximity_override_attempted = False
    try:
        if not args.check_only:
            require_motion_confirmation(args)
        adb = preflight(args)
        if args.check_only:
            print("CAPTURE_PREFLIGHT_PASS: no control processes were started", flush=True)
            return 0
        # Clear a stale developer override left by an interrupted previous run,
        # then optionally keep the headset awake while it is mounted off-face.
        proximity_override_attempted = True
        set_headset_proximity_override(adb, keep_awake=args.no_wear)
        if args.no_wear:
            print("HEADSET_NO_WEAR: proximity override enabled; Quest must stay on its fixed mount", flush=True)
        subprocess.run([adb, "reverse", "tcp:8010", "tcp:8010"], check=True, timeout=5)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", buffering=1) as logfile:
            def launch(name: str, command: list[str], cwd: Path, marker: str, timeout_s: float, env: dict[str, str] | None = None) -> Child:
                process = subprocess.Popen(
                    command, cwd=cwd, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, errors="replace", bufsize=1, start_new_session=True, env=env,
                )
                child = Child(name, process, marker)
                children.append(child)

                def read_output() -> None:
                    assert process.stdout is not None
                    for line in process.stdout:
                        message = f"[{name}] {line.rstrip()}"
                        with log_lock:
                            print(message, flush=True)
                            if not logfile.closed:
                                logfile.write(message + "\n")
                        if marker in line:
                            child.ready.set()
                        if any(value in line for value in ("BRIDGE_FAULT:", "RECORDER_FAULT", "XHAND_RETURN_HOME_FAULT", "RETURN_HOME_FAULT:")):
                            child.failure = message

                child.reader = threading.Thread(target=read_output, name=f"log-{name}", daemon=True)
                child.reader.start()
                deadline = time.monotonic() + timeout_s
                while not child.ready.is_set():
                    if child.failure:
                        raise RuntimeError(child.failure)
                    code = process.poll()
                    if code is not None:
                        raise RuntimeError(f"{name} exited during startup with status {code}; see {log_path}")
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{name} did not report {marker!r} within {timeout_s:.0f}s; see {log_path}")
                    time.sleep(0.1)
                print(f"PROCESS_READY {name} pid={process.pid}", flush=True)
                return child

            launch("relay", vendor_command(args.vendor_python, str(ROOT / "scripts/usb_webxr_relay.py")), ROOT, "USB_RELAY_READY", 15)
            launch("xhand", vendor_command(args.vendor_python, str(ROOT / "scripts/run_official_xhand_with_telemetry.py"), "--config", str(args.vendor_config), "--hardware-config", str(args.hardware_config)), args.vendor_dir, "XHAND_RETURN_HOME_COMPLETE", 45)
            recorder_python = args.vendor_python if args.format == "lerobot-v2.1" else Path(sys.executable)
            recording_args = [str(ROOT / "scripts/record_vr_episode_stream.py"), "--format", args.format, "--root", str(dataset_root.resolve()), "--task", args.task, "--rate-hz", str(args.rate_hz), "--config", str(args.hardware_config)]
            if args.format == "lerobot-v2.1":
                recording_args.extend(("--repo-id", args.repo_id))
            if args.no_cameras:
                recording_args.append("--no-cameras")
            recording = (
                vendor_command(recorder_python, *recording_args)
                if args.format == "lerobot-v2.1"
                else [str(recorder_python), *recording_args]
            )
            recorder_env = lerobot_environment(args.lerobot_src) if args.format == "lerobot-v2.1" else None
            launch("recorder", recording, ROOT, "RECORDER_READY", 45, env=recorder_env)
            bridge = [
                str(Path(sys.executable)), str(ROOT / "scripts/run_official_vr_xarm_bridge.py"),
                "--config", str(args.hardware_config),
                "--hand", "left", "--clutch-mode", "session",
                "--translation-scale", str(args.translation_scale),
                "--max-translation-mm", str(args.max_translation_mm),
                "--max-rotation-rad", str(args.max_rotation_rad),
                "--max-step-mm", str(args.max_step_mm),
                "--max-orientation-step-rad", str(args.max_orientation_step_rad),
                "--startup-timeout-s", "3600",
                "--execute", "--confirm", args.confirm,
            ]
            launch("xarm", bridge, ROOT, "Waiting for official left wrist stream", 15)
            print(f"CAPTURE_STACK_READY log={log_path} Quest=https://localhost:8010; use headset volume + / -", flush=True)
            while True:
                for child in children:
                    if child.failure:
                        raise RuntimeError(child.failure)
                    code = child.process.poll()
                    if code is not None:
                        raise RuntimeError(f"{child.name} exited with status {code}; see {log_path}")
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("CAPTURE_STOP_REQUESTED", flush=True)
    except (OSError, RuntimeError, ValueError, TimeoutError, subprocess.CalledProcessError) as exc:
        print(f"CAPTURE_STOPPED: {exc}", file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        all_stopped = True
        for child in reversed(children):
            all_stopped = stop_child(child, args.hardware_config) and all_stopped
        for child in children:
            if child.reader is not None:
                child.reader.join(timeout=1)
        if proximity_override_attempted and adb is not None:
            try:
                set_headset_proximity_override(adb, keep_awake=False)
                print("HEADSET_PROXIMITY_RESTORED", flush=True)
            except (OSError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                print(f"HEADSET_PROXIMITY_RESTORE_FAILED: {exc}", file=sys.stderr, flush=True)
                exit_code = 1
        if not all_stopped:
            exit_code = 1
        print(f"CAPTURE_CLEANUP_{'DONE' if all_stopped else 'INCOMPLETE'} log={log_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
