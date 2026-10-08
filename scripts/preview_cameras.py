#!/usr/bin/env python3
"""Show every configured camera view live, with per-view liveness and drop counts.

The cameras are exclusive devices: stop the capture stack before running this.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import time
from pathlib import Path

import cv2
import numpy as np

from _common import add_config_argument
from xarm7_xhand1 import load_config
from xarm7_xhand1.cameras import CAMERA_CONFIG_FIELDS, RealSenseCameras


GUI_HINT = (
    "This OpenCV build has no GUI support, so it cannot open a window. Run this "
    "script with an interpreter whose cv2 has QT/GTK support, for example "
    "~/miniconda3/envs/xhand_tele_env_310/bin/python"
)


def require_gui() -> None:
    try:
        cv2.namedWindow("__gui_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__gui_probe__")
    except cv2.error as exc:
        raise SystemExit(GUI_HINT) from exc


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def build_tile(name: str, source: str, image, stat: dict, tile_height: int) -> np.ndarray:
    """One labelled BGR panel; shows the last good frame while a view is failing."""
    if image is None:
        tile = np.zeros((tile_height, int(tile_height * 4 / 3), 3), np.uint8)
    else:
        scale = tile_height / image.shape[0]
        tile = cv2.resize(
            image, (max(1, round(image.shape[1] * scale)), tile_height), interpolation=cv2.INTER_AREA
        )
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
    error = stat.get("error")
    age = stat.get("age_s")
    stale = age is not None and age > 0.5
    if error or stale:
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (0, 0, 200), 3)
    header = np.full((54, tile.shape[1], 3), 28, np.uint8)
    cv2.putText(header, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, source[:46], (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
    detail = f"frames={stat.get('frames', 0)}" if stat else "not connected"
    if age is not None:
        detail += f"  age={age:.2f}s"
    cv2.putText(header, detail, (6, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
    panel = np.vstack((header, tile))
    if error:
        cv2.putText(panel, error[:70], (6, panel.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
    elif stale:
        cv2.putText(panel, f"STALE {age:.1f}s", (6, panel.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1, cv2.LINE_AA)
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--tile-height", type=int, default=360)
    parser.add_argument("--read-timeout-ms", type=int, default=200)
    parser.add_argument("--reconnect-s", type=float, default=3.0, help="Cooldown before retrying a failed connect")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("/tmp/camera_snapshots"))
    parser.add_argument("--duration-s", type=float, default=0.0, help="Exit after this many seconds; 0 waits for q/Esc")
    args = parser.parse_args()
    if args.tile_height <= 0 or args.read_timeout_ms <= 0 or args.reconnect_s <= 0 or args.duration_s < 0:
        raise ValueError("Tile height, timeouts and duration must be positive")
    require_gui()

    config = load_config(args.config)
    sources = {
        name: str(getattr(config, field)) for name, field in CAMERA_CONFIG_FIELDS.items()
        if getattr(config, field)
    }
    if not sources:
        raise SystemExit("No camera serial is configured in hardware.local.yaml")
    cameras = RealSenseCameras(config)
    window = "xarm7-xhand1 cameras"
    last_images: dict[str, np.ndarray] = {}
    drop_events: list[str] = []
    reported_failure = None
    next_reconnect_at = None
    shown = 0
    fps = 0.0
    fps_window_start = time.monotonic()
    started = fps_window_start
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        cameras.connect()
        print(f"CAMERAS_READY views={sorted(sources)} shape={config.image_width}x{config.image_height} "
              f"fps={config.camera_fps}", flush=True)
        while True:
            stats = cameras.stats()
            try:
                images = cameras.read(timeout_ms=args.read_timeout_ms)
                if reported_failure is not None:
                    print(f"{stamp()} CAMERA_RECOVERED after: {reported_failure}", flush=True)
                    reported_failure = None
                next_reconnect_at = None
                last_images.update(images)
            except (RuntimeError, TimeoutError) as exc:
                message = f"{type(exc).__name__}: {exc}"
                if message != reported_failure:
                    reported_failure = message
                    # A read() timeout sets no per-view error, so a stalled view
                    # has to be identified by the age of its last good frame.
                    # Re-sample after the failed read: a view that was merely
                    # slow to start now has frames and a small age.
                    failed_stats = cameras.stats()
                    suspect = sorted(
                        name for name, stat in failed_stats.items()
                        if stat.get("error")
                        or stat.get("frames", 0) == 0
                        or (stat.get("age_s") or 0.0) > 0.5
                    )
                    drop_events.append(f"{stamp()} {message} views={suspect or 'unknown'}")
                    print(f"{stamp()} CAMERA_FAILURE {message} views={suspect or 'unknown'}", flush=True)
                    next_reconnect_at = time.monotonic() + args.reconnect_s
            if next_reconnect_at is not None and time.monotonic() >= next_reconnect_at:
                next_reconnect_at = None
                print(f"{stamp()} CAMERA_RECONNECTING", flush=True)
                try:
                    cameras.close()
                    cameras.connect()
                    print(f"{stamp()} CAMERA_RECONNECTED views={sorted(cameras.stats())}", flush=True)
                except Exception as exc:
                    print(f"{stamp()} CAMERA_RECONNECT_FAILED: {type(exc).__name__}: {exc}", flush=True)
                    next_reconnect_at = time.monotonic() + args.reconnect_s

            panels = [
                build_tile(name, sources[name], last_images.get(name), stats.get(name, {}), args.tile_height)
                for name in sorted(sources)
            ]
            height = max(panel.shape[0] for panel in panels)
            panels = [
                np.vstack((panel, np.zeros((height - panel.shape[0], panel.shape[1], 3), np.uint8)))
                if panel.shape[0] < height else panel
                for panel in panels
            ]
            canvas = np.hstack(panels)
            elapsed = time.monotonic() - started
            shown += 1
            if shown % 15 == 0:
                now = time.monotonic()
                fps = 15.0 / max(1e-6, now - fps_window_start)
                fps_window_start = now
            banner = (
                f"display {fps:.1f} fps   uptime {elapsed:.0f}s   drops {len(drop_events)}   "
                f"[q/Esc] quit  [s] snapshot  [r] reconnect now"
            )
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (20, 20, 20), -1)
            cv2.putText(canvas, banner, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                path = args.snapshot_dir / f"cameras_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(path), canvas)
                print(f"{stamp()} SNAPSHOT_SAVED {path}", flush=True)
            elif key == ord("r"):
                next_reconnect_at = time.monotonic()
            if args.duration_s and elapsed >= args.duration_s:
                print(f"{stamp()} DURATION_REACHED {elapsed:.1f}s", flush=True)
                break
    except KeyboardInterrupt:
        print(f"{stamp()} INTERRUPTED", flush=True)
    finally:
        final_stats = cameras.stats()
        try:
            cameras.close()
        finally:
            cv2.destroyAllWindows()
        elapsed = time.monotonic() - started
        print(f"PREVIEW_SUMMARY uptime_s={elapsed:.1f} displayed={shown} drops={len(drop_events)}", flush=True)
        for name, stat in sorted(final_stats.items()):
            print(f"  {name}: frames={stat['frames']} age_s={stat['age_s']} error={stat['error']}", flush=True)
        for event in drop_events:
            print(f"  DROP {event}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
