"""Strict JSON snapshots shared by processor execution and config loading."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from typing import TypeAlias


JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)


def require_identifier(value: object, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"Processor {field} must be an explicit non-empty string.")
    return value


def snapshot_config(config: object) -> dict[str, JSONValue]:
    if not isinstance(config, Mapping):
        raise TypeError("Processor config must be a JSON mapping with string keys.")
    return _snapshot_json(config, set(), "config")  # type: ignore[return-value]


def _snapshot_json(value: object, ancestors: set[int], path: str) -> JSONValue:
    if callable(value):
        raise TypeError(f"{path} must not contain callables.")
    if value is None or type(value) in (str, bool, int):
        return value  # type: ignore[return-value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers.")
        return value
    if not isinstance(value, Mapping) and type(value) is not list:
        raise TypeError(f"{path} contains an unsupported JSON type: {type(value).__name__}.")
    if id(value) in ancestors:
        raise ValueError(f"{path} contains a circular JSON reference.")
    ancestors.add(id(value))
    try:
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(f"{path} must use string keys.")
                result[key] = _snapshot_json(item, ancestors, f"{path}.{key}")
            return result
        return [
            _snapshot_json(item, ancestors, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    finally:
        ancestors.remove(id(value))


def canonical_json(value: object) -> str:
    """Canonical format: sorted object keys, compact separators, ASCII escapes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
