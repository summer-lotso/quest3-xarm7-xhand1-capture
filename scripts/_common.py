from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_CONFIG = ROOT / "configs" / "hardware.local.yaml"

def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

def require_motion_confirmation(args: argparse.Namespace) -> None:
    if not args.execute or args.confirm != "I_HAVE_CLEARED_THE_WORKSPACE":
        raise SystemExit("Motion refused. Pass --execute --confirm I_HAVE_CLEARED_THE_WORKSPACE after checking the hardware and E-stop.")

def show_array(values) -> str:
    return json.dumps([round(float(v), 7) for v in values])
