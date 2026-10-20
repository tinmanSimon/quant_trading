"""Processor-owned configuration schemas and JSON-to-Python construction.

Typed constructor/dataclass parameters are the source of configuration fields.
A processor can add JSON Schema constraints and UI hints with ``config_schema``
or dataclass ``field(metadata={"schema": ...})``. Unknown settings and invalid
values are rejected identically for the dashboard, CLI, and registry API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import MISSING, fields, is_dataclass
from datetime import date, datetime, time
from enum import Enum
import inspect
import types
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from jsonschema import Draft202012Validator, FormatChecker

from ._config import snapshot_config


_NO_DEFAULT = object()
_FORMAT_CHECKER = FormatChecker()


def _iso_checker(parser):
    def check(value):
        if isinstance(value, str):
            parser(value)
        return True
    return check


# Use Python's ISO parsers consistently with construction. jsonschema's optional
# format extras otherwise accept/reject different datetime/time forms by install.
for _format, _parser in (("date", date.fromisoformat), ("date-time", datetime.fromisoformat),
                         ("time", time.fromisoformat)):
    _FORMAT_CHECKER.checks(_format, raises=(TypeError, ValueError))(_iso_checker(_parser))


def _parameters(cls: type) -> list[tuple[str, Any, Any, dict]]:
    """Return config names, types, defaults and per-field schema annotations."""
    if is_dataclass(cls):
        hints = get_type_hints(cls, include_extras=True)
        result = []
        for field in fields(cls):
            if not field.init:
                continue
            default = field.default
            if default is MISSING:
                default = field.default_factory() if field.default_factory is not MISSING else _NO_DEFAULT
            result.append((field.name, hints.get(field.name, Any), default,
                           dict(field.metadata.get("schema", {}))))
        return result
    if cls.__init__ is object.__init__:
        return []
    hints = get_type_hints(cls.__init__, include_extras=True)
    result = []
    for name, parameter in inspect.signature(cls).parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.kind is parameter.POSITIONAL_ONLY:
            raise TypeError(f"Configuration parameter {name!r} must accept a keyword argument.")
        default = parameter.default if parameter.default is not inspect.Parameter.empty else _NO_DEFAULT
        annotation = hints.get(name, Any)
        if annotation is Any and default is not _NO_DEFAULT and default is not None:
            annotation = type(default)
        result.append((name, annotation, default, {}))
    return result


def json_value(value: Any) -> Any:
    """Serialize typed settings to the stable JSON representation used in identity."""
    if isinstance(value, Enum):
        return json_value(value.value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value) if field.init}
    if isinstance(value, Mapping):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def instance_config(instance: object) -> dict:
    """Default config for processors that do not need a custom config property."""
    values = {name: json_value(getattr(instance, name)) for name, *_ in _parameters(type(instance))}
    return snapshot_config(values)


def _type_schema(annotation: Any, active: frozenset[type]) -> dict:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        schema = _type_schema(args[0], active)
        for metadata in args[1:]:
            if isinstance(metadata, Mapping):
                schema.update(deepcopy(metadata))
        return schema
    if annotation in (Any, inspect.Parameter.empty):
        return {}
    if annotation is type(None):
        return {"type": "null"}
    if origin in (Union, types.UnionType):
        return {"anyOf": [_type_schema(arg, active) for arg in args]}
    if origin is Literal:
        values = [json_value(value) for value in args]
        schema = {"enum": values}
        if values and len({type(value) for value in values}) == 1:
            schema.update(_type_schema(type(values[0]), active))
        return schema
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        values = [json_value(member.value) for member in annotation]
        return {"enum": values, **(_type_schema(type(values[0]), active) if values else {})}
    if annotation in (str, bool, int, float):
        return {"type": {str: "string", bool: "boolean", int: "integer", float: "number"}[annotation]}
    if annotation in (date, datetime, time):
        return {"type": "string", "format": {date: "date", datetime: "date-time", time: "time"}[annotation]}
    if origin in (list, tuple, Sequence) or annotation in (list, tuple):
        schema = {"type": "array"}
        if origin is tuple and args and args[-1] is not Ellipsis:
            schema.update(prefixItems=[_type_schema(arg, active) for arg in args],
                          minItems=len(args), maxItems=len(args))
        else:
            schema["items"] = _type_schema(args[0], active) if args else {}
        return schema
    if origin in (dict, Mapping) or annotation is dict:
        if args and args[0] not in (str, Any):
            raise TypeError("Configuration dictionaries must use string keys.")
        return {"type": "object", "additionalProperties": _type_schema(args[1], active) if args else {}}
    if inspect.isclass(annotation) and is_dataclass(annotation):
        if annotation in active:
            raise TypeError("Recursive dataclass configuration requires an explicit JSON schema and factory.")
        return _object_schema(annotation, active | {annotation})
    # An explicit config_schema/from_config can support application-specific types.
    return {}


def _object_schema(cls: type, active: frozenset[type]) -> dict:
    properties, required = {}, []
    for name, annotation, default, metadata in _parameters(cls):
        schema = _type_schema(annotation, active)
        schema.update(metadata)
        if default is _NO_DEFAULT:
            required.append(name)
        else:
            schema["default"] = json_value(default)
        properties[name] = schema
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def configuration_schema(cls: type) -> dict:
    schema = _object_schema(cls, frozenset({cls}))
    override = getattr(cls, "config_schema", None)
    if override is not None:
        if not isinstance(override, Mapping):
            raise TypeError("config_schema must be a JSON Schema mapping.")
        override = deepcopy(dict(override))
        for name, field_schema in override.pop("properties", {}).items():
            schema["properties"][name] = {**schema["properties"].get(name, {}), **field_schema}
        schema.update(override)
    schema = snapshot_config(schema)
    if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        raise TypeError("Processor configuration schema must describe an object with properties.")
    Draft202012Validator.check_schema(schema)
    return schema


def _apply_defaults(value: Any, schema: dict) -> Any:
    if not isinstance(schema, dict):
        return deepcopy(value)
    if isinstance(value, dict):
        value = deepcopy(value)
        for name, property_schema in schema.get("properties", {}).items():
            if name not in value and isinstance(property_schema, dict) and "default" in property_schema:
                value[name] = deepcopy(property_schema["default"])
            if name in value:
                value[name] = _apply_defaults(value[name], property_schema)
    elif isinstance(value, list):
        value = [_apply_defaults(item, schema.get("prefixItems", [])[index]
                                 if index < len(schema.get("prefixItems", []))
                                 else schema.get("items", {})) for index, item in enumerate(value)]
    return value


def validated_config(config: Mapping, schema: dict) -> dict:
    config = _apply_defaults(snapshot_config(config), schema)
    validator = Draft202012Validator(schema, format_checker=_FORMAT_CHECKER)
    error = next(iter(validator.iter_errors(config)), None)
    if error is not None:
        path = ".".join(str(part) for part in error.absolute_path)
        raise ValueError(f"Invalid configuration{f' at {path}' if path else ''}: {error.message}")
    return config


def _coerce(value: Any, annotation: Any) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _coerce(value, args[0])
    if annotation in (Any, inspect.Parameter.empty):
        return value
    if origin in (Union, types.UnionType):
        for arg in args:
            try:
                return _coerce(value, arg)
            except (TypeError, ValueError):
                pass
        raise TypeError(f"Value {value!r} does not match {annotation}.")
    if origin is Literal:
        if value not in args or not any(type(value) is type(arg) for arg in args if arg == value):
            raise TypeError(f"Expected one of {args!r}.")
        return value
    if annotation is type(None):
        if value is not None:
            raise TypeError("Expected null.")
        return None
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        return annotation(value)
    if annotation is float:
        if type(value) not in (float, int):
            raise TypeError("Expected a number, excluding booleans.")
        return float(value)
    if annotation in (str, bool, int):
        if type(value) is not annotation:
            raise TypeError(f"Expected {annotation.__name__}.")
        return value
    if annotation in (date, datetime, time):
        if not isinstance(value, str):
            raise TypeError("Date and time configuration must use ISO strings.")
        return annotation.fromisoformat(value)
    if origin in (list, tuple, Sequence) or annotation in (list, tuple):
        if type(value) is not list:
            raise TypeError("Expected an array.")
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(value) != len(args):
                raise TypeError("Tuple configuration has the wrong number of items.")
            result = [_coerce(item, arg) for item, arg in zip(value, args, strict=True)]
        else:
            result = [_coerce(item, args[0] if args else Any) for item in value]
        return tuple(result) if origin is tuple or annotation is tuple else result
    if origin in (dict, Mapping) or annotation is dict:
        if not isinstance(value, dict):
            raise TypeError("Expected an object.")
        return {key: _coerce(item, args[1] if args else Any) for key, item in value.items()}
    if inspect.isclass(annotation) and is_dataclass(annotation):
        if not isinstance(value, dict):
            raise TypeError("Expected an object for nested dataclass configuration.")
        return construct_from_config(annotation, value)
    return value


def construct_from_config(cls: type, config: Mapping) -> Any:
    hints = {name: annotation for name, annotation, *_ in _parameters(cls)}
    values = {name: _coerce(value, hints.get(name, Any)) for name, value in config.items()}
    return cls(**values)
