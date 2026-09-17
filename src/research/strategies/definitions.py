"""Detached strategy descriptions for discovery and configuration controls."""

from dataclasses import dataclass
import json
import math


def json_object(value, name):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys.")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON values, without NaN or infinity.") from exc


def validate_parameter(name, value, parameter):
    kind = parameter["type"]
    valid = {
        "integer": lambda: type(value) is int,
        "number": lambda: type(value) in (int, float) and math.isfinite(value),
        "boolean": lambda: type(value) is bool,
        "string": lambda: isinstance(value, str),
        "json": lambda: True,
    }[kind]()
    if not valid:
        raise ValueError(f"{name} must be of type {kind}.")
    if "choices" in parameter and not any(type(value) is type(choice) and value == choice for choice in parameter["choices"]):
        raise ValueError(f"{name} must be one of its registered choices.")
    if kind in ("integer", "number"):
        if "minimum" in parameter and value < parameter["minimum"]:
            raise ValueError(f"{name} must be at least {parameter['minimum']}.")
        if "maximum" in parameter and value > parameter["maximum"]:
            raise ValueError(f"{name} must be at most {parameter['maximum']}.")


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    version: str
    label: str
    description: str
    default_config: dict
    parameters: dict | None

    @property
    def key(self):
        return (self.name, self.version)

    def validate_config(self, config):
        config = json_object(config, "Strategy config")
        for name, parameter in (self.parameters or {}).items():
            if name in config:
                validate_parameter(name, config[name], parameter)
        return config


def definition(name, version, *, label=None, description="", default_config=None, parameters=None):
    label = name if label is None else label
    if not isinstance(label, str) or not label.strip() or not isinstance(description, str):
        raise ValueError("Strategy label must be nonempty and description must be a string.")
    defaults = json_object({} if default_config is None else default_config, "default_config")
    if parameters is not None:
        parameters = json_object(parameters, "parameters")
        for field, spec in parameters.items():
            if not field or not isinstance(spec, dict):
                raise ValueError("Parameter descriptions must be objects with nonempty names.")
            if set(spec) - {"type", "label", "help", "minimum", "maximum", "step", "choices"}:
                raise ValueError(f"Unknown parameter metadata for {field}.")
            if not isinstance(spec.get("type"), str) or spec["type"] not in {"integer", "number", "boolean", "string", "json"}:
                raise ValueError(f"Unsupported parameter type for {field}.")
            for key in ("label", "help"):
                if key in spec and not isinstance(spec[key], str):
                    raise ValueError(f"{field} {key} must be a string.")
            for bound in ("minimum", "maximum", "step"):
                if bound in spec:
                    value = spec[bound]
                    valid = (type(value) is int if spec["type"] == "integer" else
                             spec["type"] == "number" and type(value) in (int, float) and math.isfinite(value))
                    if not valid or (bound == "step" and value <= 0):
                        raise ValueError(f"Invalid {bound} for {field}.")
            if "minimum" in spec and "maximum" in spec and spec["minimum"] > spec["maximum"]:
                raise ValueError(f"Minimum exceeds maximum for {field}.")
            if "choices" in spec:
                choices = spec["choices"]
                if not isinstance(choices, list) or not choices or spec["type"] == "json":
                    raise ValueError(f"{field} choices must be a nonempty list of scalar values.")
                for choice in choices:
                    validate_parameter(field, choice, {key: value for key, value in spec.items() if key != "choices"})
            if field not in defaults:
                raise ValueError(f"Parameter {field} requires a value in default_config.")
    result = StrategyDefinition(name, version, label, description, defaults, parameters)
    result.validate_config(defaults)
    return result
