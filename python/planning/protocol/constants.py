"""Single Python owner for frozen execution-protocol constants."""

from __future__ import annotations


PROTOCOL_VERSION = 4
SCHEMA_VERSION = PROTOCOL_VERSION
PRIMITIVE_FRAME_COUNT = 25


__all__ = ["PRIMITIVE_FRAME_COUNT", "PROTOCOL_VERSION", "SCHEMA_VERSION"]
