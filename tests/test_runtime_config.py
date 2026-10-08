import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from xarm7_xhand1.runtime_config import load_runtime_config
import start_quest3_capture as launcher
from package_capture import source_files


def test_runtime_paths_follow_config_location(tmp_path):
    path = tmp_path / "runtime.yaml"
    path.write_text("task: pick\nroot: data\nhardware_config: hardware.yaml\nrate_hz: 20\nformat: lerobot-v2.1\n")
    result = load_runtime_config(path)
    assert result["root"] == tmp_path / "data"
    assert result["hardware_config"] == tmp_path / "hardware.yaml"
    assert result["rate_hz"] == 20


@pytest.mark.parametrize("text", ["execute: true", "rate_hz: 0", "rate_hz: 20.5", "translation_scale: .nan", "no_cameras: 'false'", "format: v3"])
def test_runtime_rejects_invalid_or_motion_permission_fields(tmp_path, text):
    path = tmp_path / "runtime.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_runtime_config(path)


def test_check_only_honors_runtime_and_cli_overrides_without_starting_devices(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text("task: pick\nroot: data\nvendor_dir: vendor\nvendor_config: vendor/quest.yaml\nlerobot_src: lerobot/src\nhardware_config: hardware.yaml\n")
    captured = []
    monkeypatch.setattr(sys, "argv", ["capture", "--runtime-config", str(runtime), "--check-only", "--translation-scale", "0.3"])
    monkeypatch.setattr(launcher, "preflight", lambda args: captured.append(args) or "adb")
    monkeypatch.setattr(launcher, "set_headset_proximity_override", lambda *args, **kwargs: pytest.fail("check-only changed headset state"))
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("check-only started a process"))
    assert launcher.main() == 0
    args = captured[0]
    assert args.hardware_config == tmp_path / "hardware.yaml"
    assert args.vendor_dir == tmp_path / "vendor"
    assert args.vendor_config == tmp_path / "vendor/quest.yaml"
    assert args.lerobot_src == tmp_path / "lerobot/src"
    assert args.translation_scale == 0.3
    assert not args.execute


def test_delivery_manifest_excludes_machine_data_and_credentials():
    paths = [str(path.relative_to(ROOT)) for path in source_files()]
    assert "configs/runtime.example.yaml" in paths
    assert "scripts/start_quest3_capture.py" in paths
    assert all(".local." not in path for path in paths)
    assert all(not path.startswith(("datasets/", "logs/")) for path in paths)
    assert all(not path.endswith(("key.dat", "auth_info.json", ".whl")) for path in paths)
