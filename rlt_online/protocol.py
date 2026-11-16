"""Binary WebSocket messages and strict source/observation validation."""

from __future__ import annotations

import io
import msgpack
import numpy as np
from PIL import Image

TASK = "Pick up the red block and place it on top of the green block."
SOURCES = {"VLA", "ACTOR", "HUMAN"}
MAX_MESSAGE_BYTES = 4 * 1024 * 1024


def pack(payload: dict) -> bytes:
    result = msgpack.packb(payload, use_bin_type=True)
    if len(result) > MAX_MESSAGE_BYTES:
        raise ValueError("WebSocket message exceeds size limit")
    return result


def unpack(message: bytes) -> dict:
    if not isinstance(message, bytes) or len(message) > MAX_MESSAGE_BYTES:
        raise ValueError("Expected a bounded binary MessagePack frame")
    payload = msgpack.unpackb(message, raw=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("req_id"), int) or payload["req_id"] < 0:
        raise ValueError("Every protocol message needs a nonnegative req_id")
    return payload


def joints(values, *, rows=False, max_rows=50):
    result = np.asarray(values, dtype=np.float32)
    expected = result.ndim == 2 and result.shape[1] == 6 and 1 <= result.shape[0] <= max_rows if rows else result.shape == (6,)
    if not expected or not np.isfinite(result).all():
        raise ValueError("Invalid SO-101 joint state or action chunk")
    return result


def jpeg(frame) -> bytes:
    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("Camera frame must be uint8 HWC RGB")
    output = io.BytesIO()
    Image.fromarray(frame).save(output, format="JPEG", quality=90)
    return output.getvalue()


def image(data: bytes) -> np.ndarray:
    if not isinstance(data, bytes) or len(data) > 2_000_000:
        raise ValueError("Invalid JPEG payload")
    with Image.open(io.BytesIO(data)) as frame:
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
