"""Shared fail-closed primitive protocol validation helpers.

Schema modules own message-specific invariants.  This module owns only the
reusable type, presence, and digest checks so Python, C++, and C# golden
vectors continue to exercise the same schema-specific callers.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any, Optional, Type


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _error_type(error_type: Optional[Type[Exception]]) -> Type[Exception]:
    if error_type is not None:
        return error_type
    # Import lazily: the v4 schema imports these helpers.
    from planning.protocol.primitive_execution_schema_v4 import (
        PrimitiveExecutionProtocolError,
    )

    return PrimitiveExecutionProtocolError


def _raise(error_type: Optional[Type[Exception]], message: str) -> None:
    raise _error_type(error_type)(message)


def require_mapping(
    value: Any, *, name: str, error_type: Optional[Type[Exception]] = None
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _raise(error_type, "{} must be a mapping".format(name))
    return value


def require_field(
    mapping: Mapping[str, Any],
    name: str,
    *,
    error_type: Optional[Type[Exception]] = None,
) -> Any:
    if name not in mapping:
        _raise(error_type, "{} is required".format(name))
    return mapping[name]


def require_non_empty_text(
    value: Any, *, name: str, error_type: Optional[Type[Exception]] = None
) -> str:
    if not isinstance(value, str) or not value:
        _raise(error_type, "{} must be a non-empty string".format(name))
    return value


def require_uint32(
    value: Any, *, name: str, error_type: Optional[Type[Exception]] = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
        _raise(error_type, "{} must be uint32".format(name))
    return value


def require_uint64(
    value: Any, *, name: str, error_type: Optional[Type[Exception]] = None
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 0xFFFFFFFFFFFFFFFF
    ):
        _raise(error_type, "{} must be uint64".format(name))
    return value


def require_int(
    value: Any, *, name: str, error_type: Optional[Type[Exception]] = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _raise(error_type, "{} must be an integer".format(name))
    return value


def require_digest(
    value: Any, *, name: str, error_type: Optional[Type[Exception]] = None
) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _raise(error_type, "{} must be a lowercase SHA-256 digest".format(name))
    return value


def require_bytes(
    value: Any, *, name: str, error_type: Optional[Type[Exception]] = None
) -> bytes:
    if not isinstance(value, bytes):
        _raise(error_type, "{} must be bytes".format(name))
    return value


def __getattr__(name: str):
    # Keep the historical facade importable without making schema modules
    # import each other at module initialization time.
    if name in {
        "PrimitiveExecutionProtocolError",
        "SchemaMismatchError",
        "validate_execution_result",
    }:
        from planning.protocol import primitive_execution_schema_v4 as schema

        return getattr(schema, name)
    if name in {
        "PrimitiveExecutionCommandSchemaError",
        "PrimitiveExecutionCommand",
    }:
        from planning.protocol import primitive_execution_command_v4 as command

        return getattr(command, name)
    raise AttributeError(name)


__all__ = [
    "PrimitiveExecutionCommand",
    "PrimitiveExecutionCommandSchemaError",
    "PrimitiveExecutionProtocolError",
    "SchemaMismatchError",
    "require_bytes",
    "require_digest",
    "require_field",
    "require_int",
    "require_mapping",
    "require_non_empty_text",
    "require_uint32",
    "require_uint64",
    "validate_execution_result",
]
