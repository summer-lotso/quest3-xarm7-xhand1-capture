#!/usr/bin/env python3
"""Build a portable source archive with examples, excluding data and credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "quest3-xarm7-xhand1-capture"


def source_files() -> list[Path]:
    files = [ROOT / name for name in ("README.md", "pyproject.toml", "environment.yml", "requirements-lock.txt", ".gitignore", "vendor/README.md")]
    for folder, pattern in (("src", "*.py"), ("scripts", "*.py"), ("tests", "*.py"), ("docs", "*.md"), ("configs", "*.example.yaml")):
        files.extend((ROOT / folder).rglob(pattern))
    return sorted(set(path for path in files if path.is_file()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "releases/quest3-xarm7-xhand1-capture-v0.1.0.zip")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = source_files()
    manifest = {
        "application_version": "0.1.0",
        "dataset_format": "LeRobot Dataset v2.1",
        "python": "3.10",
        "entrypoint": "scripts/start_quest3_capture.py",
        "external_dependencies": ["licensed RobotEra teleoperation wheels and credentials", "LeRobot source with CODEBASE_VERSION=v2.1"],
        "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    fd, temporary = tempfile.mkstemp(prefix=".capture-bundle-", suffix=".zip", dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, f"{PREFIX}/{path.relative_to(ROOT).as_posix()}")
            archive.writestr(f"{PREFIX}/MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{checksum}  {output.name}\n")
    print(f"CAPTURE_BUNDLE {output} files={len(files)} sha256={checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
