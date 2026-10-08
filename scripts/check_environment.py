#!/usr/bin/env python3
"""Check imports and expected SDK symbols without connecting to hardware."""
import importlib
import importlib.metadata
import ctypes
import platform
import sys
from pathlib import Path


def main():
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.system()} {platform.machine()}")
    failed = False
    for package, module in (
        ("xarm-python-sdk", "xarm.wrapper"),
        ("xhand_controller", "xhand_controller.xhand_control"),
        ("numpy", "numpy"),
        ("PyYAML", "yaml"),
        ("msgpack", "msgpack"),
        ("websockets", "websockets.sync.client"),
        ("pyrealsense2", "pyrealsense2"),
        ("opencv-python-headless", "cv2"),
        ("pytest", "pytest"),
    ):
        try:
            loaded = importlib.import_module(module)
            if package == "xarm-python-sdk":
                assert hasattr(loaded, "XArmAPI"), "XArmAPI missing"
            if package == "xhand_controller":
                for symbol in ("XHandControl", "HandCommand_t"):
                    assert hasattr(loaded, symbol), f"{symbol} missing"
                for method in ("open_ethercat", "open_serial", "read_state", "send_command", "close_device"):
                    assert hasattr(loaded.XHandControl, method), f"{method} missing"
                native_library = Path(loaded.__file__).with_name("libxhand_control.so")
                ctypes.CDLL(str(native_library))
                print(f"OK native library {native_library}")
            print(f"OK {package}=={importlib.metadata.version(package)}")
        except Exception as exc:
            failed = True
            print(f"FAIL {package}: {exc}")
    print("No device connections or motion commands were issued.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
