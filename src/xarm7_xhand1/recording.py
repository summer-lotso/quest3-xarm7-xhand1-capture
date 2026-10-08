"""Fail-closed episode buffering for VR teleoperation data collection.

The USB relay owns the operator-facing state machine.  This module mirrors its
control events at the dataset boundary so a completed episode is never written
until the operator explicitly accepts it.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol
import json
import os
import shutil
import tempfile

import numpy as np


class EpisodeState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PENDING = "pending"
    FAULT = "fault"


class EpisodeSink(Protocol):
    def add_frame(self, frame: dict[str, Any], *, task: str, timestamp: float) -> None: ...
    def save_episode(self) -> None: ...
    def clear_episode_buffer(self) -> None: ...


@dataclass(frozen=True)
class ControlEvent:
    event: str
    episode: int | None
    reason: str = ""


class EpisodeRecorder:
    """Validate frames and make save/discard semantics atomic and testable."""

    ACTION_SIZE = 18  # Cartesian xArm target (6) + XHand target (12)
    ARM_JOINT_SIZE = 7
    ARM_TCP_SIZE = 6
    HAND_SIZE = 12

    def __init__(self, sink: EpisodeSink, *, task: str):
        if not task.strip():
            raise ValueError("Dataset task must not be empty")
        self.sink = sink
        self.task = task
        self.state = EpisodeState.IDLE
        self.episode: int | None = None
        self.frame_count = 0
        self.last_timestamp: float | None = None
        self.last_reason = ""

    def handle(self, control: ControlEvent) -> EpisodeState:
        event = control.event.lower()
        if event == "start":
            if self.state == EpisodeState.PENDING:
                raise RuntimeError("Pending episode must be saved or discarded before a new start")
            if self.state == EpisodeState.RECORDING:
                raise RuntimeError("Received start while an episode is already recording")
            self._clear_buffer()
            self.state = EpisodeState.RECORDING
            self.episode = control.episode
            self.frame_count = 0
            self.last_timestamp = None
        elif event == "complete":
            if self.state == EpisodeState.FAULT:
                self._require_episode_if_present(control)
                self.last_reason = control.reason
                return self.state
            self._require_current(control, EpisodeState.RECORDING)
            if self.frame_count == 0:
                self._clear_buffer()
                self.state = EpisodeState.FAULT
                self.last_reason = "completed episode contained no frames"
                return self.state
            self.state = EpisodeState.PENDING
        elif event == "save":
            if self.state == EpisodeState.FAULT:
                self._require_episode_if_present(control)
                self.last_reason = control.reason
                return self.state
            self._require_current(control, EpisodeState.PENDING)
            self.sink.save_episode()
            self._reset(EpisodeState.IDLE)
        elif event in {"discard", "abort"}:
            if self.state not in {EpisodeState.RECORDING, EpisodeState.PENDING, EpisodeState.FAULT}:
                return self.state
            self._require_episode_if_present(control)
            self._clear_buffer()
            self._reset(EpisodeState.IDLE)
        elif event == "fault":
            if self.state == EpisodeState.RECORDING:
                self._require_episode_if_present(control)
                self._clear_buffer()
            self.state = EpisodeState.FAULT
            self.episode = control.episode
            self.frame_count = 0
            self.last_timestamp = None
        else:
            raise ValueError(f"Unsupported teleop control event: {control.event!r}")
        self.last_reason = control.reason
        return self.state

    def add_frame(self, frame: Mapping[str, Any], *, timestamp: float) -> bool:
        if self.state != EpisodeState.RECORDING:
            return False
        if not np.isfinite(timestamp):
            raise ValueError("Frame timestamp must be finite")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Frame timestamps must be strictly increasing")
        validated = validate_recording_frame(frame)
        self.sink.add_frame(validated, task=self.task, timestamp=float(timestamp))
        self.frame_count += 1
        self.last_timestamp = float(timestamp)
        return True

    def _require_current(self, control: ControlEvent, expected: EpisodeState) -> None:
        if self.state != expected:
            raise RuntimeError(f"Event {control.event!r} requires state {expected.value}, got {self.state.value}")
        self._require_episode_if_present(control)

    def _require_episode_if_present(self, control: ControlEvent) -> None:
        if control.episode is not None and self.episode is not None and control.episode != self.episode:
            raise RuntimeError(
                f"Control event episode {control.episode} does not match active episode {self.episode}"
            )

    def _clear_buffer(self) -> None:
        self.sink.clear_episode_buffer()

    def _reset(self, state: EpisodeState) -> None:
        self.state = state
        self.episode = None
        self.frame_count = 0
        self.last_timestamp = None


def _vector(frame: Mapping[str, Any], key: str, size: int) -> np.ndarray:
    value = np.asarray(frame.get(key), dtype=np.float32)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{key} must contain {size} finite values")
    return value


def validate_recording_frame(frame: Mapping[str, Any]) -> dict[str, Any]:
    """Return a defensive, LeRobot-compatible copy of one synchronized frame."""
    result: dict[str, Any] = {
        "action": _vector(frame, "action", EpisodeRecorder.ACTION_SIZE),
        "observation.arm_joint_position": _vector(
            frame, "observation.arm_joint_position", EpisodeRecorder.ARM_JOINT_SIZE
        ),
        "observation.arm_tcp_pose": _vector(
            frame, "observation.arm_tcp_pose", EpisodeRecorder.ARM_TCP_SIZE
        ),
        "observation.hand_joint_position": _vector(
            frame, "observation.hand_joint_position", EpisodeRecorder.HAND_SIZE
        ),
    }
    for key, value in frame.items():
        if not key.startswith("observation.images."):
            continue
        image = np.asarray(value)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{key} must be an HxWx3 uint8 image")
        result[key] = image.copy()
    return result


def dataset_features(*, image_shapes: Mapping[str, tuple[int, int, int]] | None = None) -> dict[str, dict]:
    features: dict[str, dict] = {
        "action": {"dtype": "float32", "shape": (18,), "names": None},
        "observation.arm_joint_position": {"dtype": "float32", "shape": (7,), "names": None},
        "observation.arm_tcp_pose": {"dtype": "float32", "shape": (6,), "names": None},
        "observation.hand_joint_position": {"dtype": "float32", "shape": (12,), "names": None},
    }
    for name, shape in (image_shapes or {}).items():
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(f"Image shape for {name} must be HxWx3")
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": tuple(shape),
            "names": ["height", "width", "channels"],
        }
    return features


class LeRobotSink:
    """Small lazy adapter so robot-only tools do not import torch/LeRobot."""

    def __init__(self, dataset: Any):
        self.dataset = dataset

    @classmethod
    def create(
        cls,
        *,
        repo_id: str,
        root: str | Path,
        fps: int,
        image_shapes: Mapping[str, tuple[int, int, int]] | None = None,
        image_writer_threads: int = 4,
    ) -> "LeRobotSink":
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot Dataset v2.1 could not be imported; configure lerobot_src and the recorder's vendor_python environment"
            ) from exc
        # The vendor teleoperation environment ships a PyAV build with H.264
        # but without LeRobot's preferred libsvtav1 encoder.  H.264 is a
        # supported v2.1 codec, so select it explicitly when AV1 is absent.
        import av
        if "libsvtav1" not in av.codecs_available:
            import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
            from lerobot.datasets.video_utils import get_video_pixel_channels
            if not getattr(lerobot_dataset_module.encode_video_frames, "_xarm_h264_fallback", False):
                original_encoder = lerobot_dataset_module.encode_video_frames

                def encode_h264(*args, **kwargs):
                    kwargs.setdefault("vcodec", "h264")
                    return original_encoder(*args, **kwargs)

                encode_h264._xarm_h264_fallback = True
                lerobot_dataset_module.encode_video_frames = encode_h264
            # PyAV 13 calls this field ``name``; PyAV 14 added
            # ``canonical_name``.  Populate the same LeRobot metadata without
            # requiring a system FFmpeg/PyAV rebuild on the robot computer.
            if not getattr(lerobot_dataset_module.get_video_info, "_xarm_pyav13_compat", False):
                def get_video_info_compat(video_path):
                    with av.open(str(video_path), "r") as video_file:
                        stream = video_file.streams.video[0]
                        info = {
                            "video.height": stream.height,
                            "video.width": stream.width,
                            "video.codec": getattr(stream.codec, "canonical_name", stream.codec.name),
                            "video.pix_fmt": stream.pix_fmt,
                            "video.is_depth_map": False,
                            "video.fps": int(stream.base_rate),
                            "video.channels": get_video_pixel_channels(stream.pix_fmt),
                            "has_audio": False,
                        }
                    return info

                get_video_info_compat._xarm_pyav13_compat = True
                lerobot_dataset_module.get_video_info = get_video_info_compat
        root_path = Path(root).expanduser().resolve()
        features = dataset_features(image_shapes=image_shapes)
        info_path = root_path / "meta" / "info.json"
        if info_path.is_file():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            # LeRobot v2.1 does not reopen a just-created dataset with zero
            # episodes: its tasks/episodes/parquet files do not yet exist, so
            # it incorrectly attempts a Hub download.  Keep the empty
            # metadata recoverable and create a fresh local dataset instead.
            if (
                info.get("total_episodes") == 0
                and info.get("total_frames") == 0
                and info.get("total_videos") == 0
                and {path for path in root_path.rglob("*") if path.is_file()} == {info_path}
            ):
                if info.get("codebase_version") != "v2.1" or info.get("fps") != fps:
                    raise RuntimeError("Empty dataset version or fps differs from the requested recording")
                actual = set(info.get("features", {})) - {"timestamp", "frame_index", "episode_index", "index", "task_index"}
                if actual != set(features):
                    raise RuntimeError("Empty dataset feature keys differ from the requested recording")
                suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                backup = root_path.with_name(f"{root_path.name}.empty-backup-{suffix}-{os.getpid()}")
                root_path.rename(backup)
                print(f"EMPTY_DATASET_BACKUP: {backup}", flush=True)
        if info_path.is_file():
            dataset = LeRobotDataset(repo_id=repo_id, root=root_path)
            if dataset.meta.info.get("codebase_version") != "v2.1":
                raise RuntimeError(
                    f"Existing dataset is {dataset.meta.info.get('codebase_version')!r}, expected 'v2.1'"
                )
            if dataset.fps != fps:
                raise RuntimeError(f"Existing dataset fps={dataset.fps}, requested fps={fps}")
            expected = set(features)
            actual = set(dataset.features) - {"timestamp", "frame_index", "episode_index", "index", "task_index"}
            if actual != expected:
                raise RuntimeError(
                    f"Existing LeRobot feature keys differ: expected={sorted(expected)} actual={sorted(actual)}"
                )
            if image_shapes:
                dataset.start_image_writer(num_threads=image_writer_threads)
        else:
            if root_path.exists() and any(root_path.iterdir()):
                raise RuntimeError(f"Dataset root exists but is not a LeRobot v2.1 dataset: {root_path}")
            if root_path.exists():
                root_path.rmdir()
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=root_path,
                fps=fps,
                features=features,
                robot_type="xarm7_xhand1_vr",
                use_videos=bool(image_shapes),
                image_writer_threads=image_writer_threads if image_shapes else 0,
            )
        return cls(dataset)

    def add_frame(self, frame: dict[str, Any], *, task: str, timestamp: float) -> None:
        # LeRobot v2.1 validates timestamps against the fixed dataset FPS with
        # a tight tolerance.  Let it derive timestamp = frame_index / fps;
        # wall-clock jitter is not part of the dataset timeline.
        self.dataset.add_frame(frame, task=task)

    def save_episode(self) -> None:
        self.dataset.save_episode()

    def clear_episode_buffer(self) -> None:
        # Reopened LeRobot datasets start with episode_buffer=None. The first
        # add_frame() creates it lazily; there is nothing to clear on start.
        if self.dataset.episode_buffer is not None:
            # Discard can arrive while image worker threads are still writing
            # frames. LeRobot's clear_episode_buffer removes their directory,
            # so drain the queue before deleting it or the recorder can crash.
            if self.dataset.image_writer is not None:
                self.dataset.image_writer.wait_until_done()
            self.dataset.clear_episode_buffer()

    def close(self) -> None:
        self.dataset.stop_image_writer()


class NpzEpisodeSink:
    """Atomic local capture with disk-backed frames and bounded RAM use."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.episodes_dir = self.root / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self._buffer_dir: Path | None = None
        self._arrays: dict[str, np.memmap] = {}
        self._specs: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
        self._paths: dict[str, Path] = {}
        self._count = 0
        self._capacity = 0
        self.tasks: list[str] = []
        self.timestamps: list[float] = []

    def _grow(self) -> None:
        new_capacity = max(128, self._capacity * 2)
        for key in tuple(self._arrays):
            array = self._arrays.pop(key)
            array.flush()
            mapping = array._mmap
            del array
            mapping.close()
            shape, dtype = self._specs[key]
            with self._paths[key].open("r+b") as stream:
                stream.truncate(new_capacity * int(np.prod(shape)) * dtype.itemsize)
            self._arrays[key] = np.memmap(
                self._paths[key], dtype=dtype, mode="r+", shape=(new_capacity, *shape)
            )
        self._capacity = new_capacity

    def add_frame(self, frame: dict[str, Any], *, task: str, timestamp: float) -> None:
        if self._buffer_dir is None:
            self._buffer_dir = Path(tempfile.mkdtemp(prefix=".buffer-", dir=self.episodes_dir))
            self._capacity = 128
            for index, (key, value) in enumerate(frame.items()):
                array = np.asarray(value)
                path = self._buffer_dir / f"feature_{index:03d}.raw"
                self._specs[key] = (array.shape, array.dtype)
                self._paths[key] = path
                self._arrays[key] = np.memmap(
                    path, dtype=array.dtype, mode="w+", shape=(self._capacity, *array.shape)
                )
        if set(frame) != set(self._arrays):
            raise ValueError("Frame schema changed inside an episode")
        for key, value in frame.items():
            array = np.asarray(value)
            if (array.shape, array.dtype) != self._specs[key]:
                raise ValueError(f"Frame shape or dtype changed for {key}")
        if self._count == self._capacity:
            self._grow()
        for key, value in frame.items():
            self._arrays[key][self._count] = value
        self._count += 1
        self.tasks.append(task)
        self.timestamps.append(float(timestamp))

    def save_episode(self) -> None:
        if self._count == 0:
            raise RuntimeError("Cannot save an empty episode")
        existing = [int(path.stem.removeprefix("episode_")) for path in self.episodes_dir.glob("episode_*.npz") if path.stem.removeprefix("episode_").isdigit()]
        index = max(existing, default=-1) + 1
        target = self.episodes_dir / f"episode_{index:06d}.npz"
        for array in self._arrays.values():
            array.flush()
        arrays = {key: array[:self._count] for key, array in self._arrays.items()}
        arrays["timestamp"] = np.asarray(self.timestamps, dtype=np.float64)
        arrays["task"] = np.asarray(self.tasks, dtype=np.str_)
        fd, temporary_name = tempfile.mkstemp(prefix=".episode-", suffix=".npz", dir=self.episodes_dir)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        manifest = {
            "format": "xarm7_xhand1_npz_v1",
            "episodes": index + 1,
            "latest": target.name,
            "latest_frames": self._count,
            "features": {key: list(shape) for key, (shape, _) in self._specs.items()},
        }
        manifest_path = self.root / "manifest.json"
        manifest_tmp = self.root / ".manifest.json.tmp"
        manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(manifest_tmp, manifest_path)
        self.clear_episode_buffer()

    def clear_episode_buffer(self) -> None:
        for key in tuple(self._arrays):
            array = self._arrays.pop(key)
            array.flush()
            mapping = array._mmap
            del array
            mapping.close()
        if self._buffer_dir is not None:
            shutil.rmtree(self._buffer_dir)
        self._buffer_dir = None
        self._paths.clear()
        self._specs.clear()
        self._count = 0
        self._capacity = 0
        self.tasks.clear()
        self.timestamps.clear()
