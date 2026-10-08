"""Minimal RealSense RGB capture for one to three stable policy camera names."""
from __future__ import annotations
from contextlib import suppress
import threading
import time
import numpy as np
from .config import HardwareConfig

CAMERA_CONFIG_FIELDS = {
    "head_view": "head_camera_serial",
    "front_left_view": "front_left_camera_serial",
    "wrist_view": "wrist_camera_serial",
}

class RealSenseCameras:
    def __init__(self, config: HardwareConfig):
        self.config, self._pipelines, self._captures, self._rotations = config, {}, {}, {}
        self._latest, self._errors, self._threads = {}, {}, []
        self._counts, self._stored_at = {}, {}
        self._generation, self._last_read_generation = {}, {}
        self._lock, self._stop_event = threading.Lock(), threading.Event()
    def connect(self) -> None:
        import pyrealsense2 as rs
        if self._pipelines or self._captures: raise RuntimeError("Cameras are already connected")
        # close() keeps the previous errors so read() can report them; a fresh
        # connection must not inherit a failure it did not cause.
        self._errors.clear()
        configured = {
            name: str(getattr(self.config, field))
            for name, field in CAMERA_CONFIG_FIELDS.items()
            if getattr(self.config, field)
        }
        if not configured:
            raise ValueError("At least one camera serial must be configured")
        try:
            for name, serial in configured.items():
                rotation_field = f"{CAMERA_CONFIG_FIELDS[name].removesuffix('_serial')}_rotation_deg"
                self._rotations[name] = int(getattr(self.config, rotation_field))
            # Open every UVC camera before starting RealSense streaming. Some
            # hosts otherwise renegotiate the second UVC stream at ~18 Hz.
            import cv2
            for name, serial in configured.items():
                if not serial.startswith("/dev/"): continue
                capture = cv2.VideoCapture(serial, cv2.CAP_V4L2)
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.image_width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.image_height)
                capture.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
                if not capture.isOpened(): raise ConnectionError(f"Unable to open V4L2 camera {name} at {serial}")
                self._captures[name] = capture
            for name, serial in configured.items():
                if serial.startswith("/dev/"): continue
                pipeline, cfg = rs.pipeline(), rs.config()
                cfg.enable_device(serial)
                cfg.enable_stream(rs.stream.color, self.config.image_width, self.config.image_height, rs.format.rgb8, self.config.camera_fps)
                pipeline.start(cfg); self._pipelines[name] = pipeline
            self._stop_event.clear()
            for name, pipeline in self._pipelines.items():
                self._start_worker(name, self._realsense_worker, pipeline)
            for name, capture in self._captures.items():
                self._start_worker(name, self._v4l2_worker, capture)
        except Exception:
            self.close(); raise
    def _start_worker(self, name, target, device):
        self._generation[name] = 0
        self._last_read_generation[name] = 0
        thread = threading.Thread(target=target, args=(name, device), name=f"camera-{name}", daemon=True)
        thread.start(); self._threads.append(thread)
    def _store(self, name, image):
        expected = (self.config.image_height, self.config.image_width, 3)
        if image.shape != expected: raise RuntimeError(f"{name} returned {image.shape}, expected {expected}")
        rotation = self._rotations[name]
        if rotation: image = np.rot90(image, k=rotation // 90).copy()
        with self._lock:
            self._latest[name] = image
            self._generation[name] += 1
            self._counts[name] = self._counts.get(name, 0) + 1
            self._stored_at[name] = time.monotonic()
    def _realsense_worker(self, name, pipeline):
        try:
            while not self._stop_event.is_set():
                frame = pipeline.wait_for_frames(1000).get_color_frame()
                if frame: self._store(name, np.asanyarray(frame.get_data()).copy())
        except Exception as exc:
            with self._lock: self._errors[name] = exc
    def _v4l2_worker(self, name, capture):
        import cv2
        try:
            while not self._stop_event.is_set():
                ok, bgr = capture.read()
                if not ok or bgr is None: raise TimeoutError(f"No color frame from {name}")
                self._store(name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        except Exception as exc:
            with self._lock: self._errors[name] = exc
    def read(self, timeout_ms: int = 1000) -> dict[str, np.ndarray]:
        if not self._pipelines and not self._captures: raise RuntimeError("No cameras are connected")
        names = set(self._pipelines) | set(self._captures)
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            with self._lock:
                if self._errors:
                    name, error = next(iter(self._errors.items()))
                    raise RuntimeError(f"Camera worker {name} failed: {error}") from error
                ready = all(self._generation.get(name, 0) > self._last_read_generation.get(name, 0) for name in names)
                if ready:
                    images = {name: self._latest[name].copy() for name in names}
                    self._last_read_generation = {name: self._generation[name] for name in names}
                    return images
            if time.monotonic() >= deadline: raise TimeoutError("Timed out waiting for a fresh frame from every camera")
            time.sleep(0.001)
    def stats(self) -> dict:
        """Per-view liveness; unlike read() this works while one view has failed."""
        now = time.monotonic()
        with self._lock:
            errors, counts, stored_at = dict(self._errors), dict(self._counts), dict(self._stored_at)
        result = {}
        for name in set(self._pipelines) | set(self._captures) | set(errors):
            result[name] = {
                "frames": counts.get(name, 0),
                "age_s": None if name not in stored_at else round(now - stored_at[name], 3),
                "error": None if name not in errors else f"{type(errors[name]).__name__}: {errors[name]}",
            }
        return result
    def close(self) -> None:
        self._stop_event.set()
        for thread in self._threads: thread.join(timeout=2.0)
        self._threads.clear()
        for pipeline in self._pipelines.values():
            # A USB reset may already have stopped a RealSense pipeline.
            with suppress(RuntimeError): pipeline.stop()
        self._pipelines.clear()
        for capture in self._captures.values(): capture.release()
        self._captures.clear()
        self._rotations.clear()
        self._latest.clear(); self._errors.clear(); self._generation.clear(); self._last_read_generation.clear()
