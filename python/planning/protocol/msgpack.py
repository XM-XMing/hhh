"""Typed canonical MessagePack writer for the protocol-v4 wire owners.

The protocol schemas intentionally emit fixed integer/float widths and sorted
map keys.  Keeping this small writer in one module prevents the command,
result, and endpoint-snapshot encoders from drifting while preserving their
existing byte-level contracts.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping as MappingABC
from typing import Any, Type

import msgpack


class CanonicalMessagePackWriter:
    """Append fixed-width canonical MessagePack values to one byte buffer."""

    def __init__(self, error_type: Type[Exception] = ValueError) -> None:
        self.data = bytearray()
        self._error_type = error_type

    def _fail(self, message: str):
        raise self._error_type(message)

    def _append_uint(self, value: int, width: int) -> None:
        self.data.extend(value.to_bytes(width, byteorder="big", signed=False))

    def map_header(self, count: int) -> None:
        if count < 0 or count > 0xFFFFFFFF:
            self._fail("invalid MessagePack map count")
        if count <= 15:
            self.data.append(0x80 | count)
        elif count <= 0xFFFF:
            self.data.append(0xDE)
            self._append_uint(count, 2)
        else:
            self.data.append(0xDF)
            self._append_uint(count, 4)

    def array_header(self, count: int) -> None:
        if count < 0 or count > 0xFFFFFFFF:
            self._fail("invalid MessagePack array count")
        if count <= 15:
            self.data.append(0x90 | count)
        elif count <= 0xFFFF:
            self.data.append(0xDC)
            self._append_uint(count, 2)
        else:
            self.data.append(0xDD)
            self._append_uint(count, 4)

    def nil(self) -> None:
        self.data.append(0xC0)

    def string(self, value: str) -> None:
        if not isinstance(value, str):
            self._fail("v4 strings must be text")
        encoded = value.encode("utf-8")
        length = len(encoded)
        if length <= 0xFF:
            self.data.append(0xD9)
            self._append_uint(length, 1)
        elif length <= 0xFFFF:
            self.data.append(0xDA)
            self._append_uint(length, 2)
        elif length <= 0xFFFFFFFF:
            self.data.append(0xDB)
            self._append_uint(length, 4)
        else:
            self._fail("v4 string is too long")
        self.data.extend(encoded)

    def binary32(self, value: bytes) -> None:
        if not isinstance(value, bytes) or len(value) != 32:
            self._fail("v4 binary fields must be bytes32")
        self.data.append(0xC4)
        self._append_uint(32, 1)
        self.data.extend(value)

    def binary(self, value: bytes) -> None:
        if not isinstance(value, bytes):
            self._fail("v4 binary fields must be bytes")
        length = len(value)
        if length <= 0xFF:
            self.data.append(0xC4)
            self._append_uint(length, 1)
        elif length <= 0xFFFF:
            self.data.append(0xC5)
            self._append_uint(length, 2)
        elif length <= 0xFFFFFFFF:
            self.data.append(0xC6)
            self._append_uint(length, 4)
        else:
            self._fail("v4 binary field is too long")
        self.data.extend(value)

    def uint32(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
            self._fail("value is not uint32")
        self.data.append(0xCE)
        self._append_uint(value, 4)

    def uint64(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
            self._fail("value is not uint64")
        self.data.append(0xCF)
        self._append_uint(value, 8)

    def int32(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or not -0x80000000 <= value <= 0x7FFFFFFF:
            self._fail("value is not int32")
        self.data.append(0xD2)
        self.data.extend(struct.pack(">i", value))

    def int64(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or not -0x8000000000000000 <= value <= 0x7FFFFFFFFFFFFFFF:
            self._fail("value is not int64")
        self.data.append(0xD3)
        self.data.extend(struct.pack(">q", value))

    def float32(self, value: Any) -> None:
        if isinstance(value, bool):
            self._fail("v4 action values must be finite float32")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise self._error_type("v4 action values must be finite float32") from error
        if not math.isfinite(numeric):
            self._fail("v4 action values reject NaN and infinity")
        try:
            self.data.append(0xCA)
            self.data.extend(struct.pack(">f", numeric))
        except (OverflowError, struct.error) as error:
            raise self._error_type("v4 action value is outside float32 range") from error

    def bytes(self) -> bytes:
        return bytes(self.data)


def messagepack_bytes(payload: Any) -> bytes:
    """Preserve the ordinary ``msgpack.packb`` contract for legacy messages."""

    return msgpack.packb(payload, use_bin_type=True)


def _canonicalize(payload: Any) -> Any:
    if isinstance(payload, MappingABC):
        entries = []
        for key, value in payload.items():
            if not isinstance(key, str):
                raise TypeError("canonical maps require string keys, got {!r}".format(key))
            entries.append((key, _canonicalize(value)))
        entries.sort(key=lambda pair: pair[0].encode("utf-8"))
        return {key: value for key, value in entries}
    if isinstance(payload, (list, tuple)):
        return [_canonicalize(value) for value in payload]
    if isinstance(payload, bool) or payload is None or isinstance(payload, (str, bytes, int)):
        return payload
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise ValueError("canonical payload does not allow NaN or infinity")
        return payload
    raise TypeError(
        "unsupported canonical payload value type: {}".format(type(payload).__name__)
    )


def canonical_messagepack_bytes(payload: Any) -> bytes:
    """Encode generic deterministic MessagePack with recursively sorted maps."""

    return msgpack.packb(
        _canonicalize(payload),
        use_bin_type=True,
        strict_types=True,
    )


def messagepack_unpack(payload: bytes) -> Any:
    """Decode an ordinary protocol MessagePack payload."""

    return msgpack.unpackb(payload, raw=False)


__all__ = [
    "CanonicalMessagePackWriter",
    "canonical_messagepack_bytes",
    "messagepack_bytes",
    "messagepack_unpack",
]
