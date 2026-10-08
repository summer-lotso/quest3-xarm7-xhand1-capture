"""WebSocket/MessagePack policy transport compatible with the LeRobot client."""
from __future__ import annotations
import functools
from typing import Any
import msgpack, numpy as np
from websockets.sync.client import connect

def _pack_array(obj: Any):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "VOc": raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic): return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj
def _unpack_array(obj: dict):
    if b"__ndarray__" in obj: return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj: return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj
packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)

class PolicyClient:
    def __init__(self, host: str, port: int = 8000, *, api_key: str | None = None, timeout_s: float = 10):
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        self._socket = connect(uri, compression=None, max_size=None, additional_headers=headers, open_timeout=timeout_s)
        self.metadata = unpackb(self._socket.recv())
    def infer(self, observation: dict) -> dict:
        self._socket.send(packb(observation)); response = self._socket.recv()
        if isinstance(response, str): raise RuntimeError(f"Policy server error: {response}")
        return unpackb(response)
    @staticmethod
    def validate_action(action: Any) -> np.ndarray:
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (19,) or not np.isfinite(value).all(): raise ValueError("Policy action must be exactly 19 finite values (7 arm + 12 hand)")
        return value
    def close(self) -> None: self._socket.close()
