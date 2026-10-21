"""Automatic discovery with isolated import failures and duplicate detection."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
import importlib
import inspect
import pkgutil
import re
from types import ModuleType

from data_pipeline.processing.configuration import validated_config
from jsonschema import Draft202012Validator

from .base import BaseChartLogic, LogicResult, json_copy


@dataclass(frozen=True)
class LogicDefinition:
    name: str
    version: str
    label: str
    description: str
    schema: dict
    display_schema: dict = field(default_factory=lambda: {"type": "object", "properties": {}, "additionalProperties": False})
    result_version: str = "1"
    implementation_fingerprint: str = ""


class LogicRegistry:
    def __init__(self):
        self._classes = {}
        self._definitions = {}
        self._issues = []

    @property
    def issues(self):
        return tuple(self._issues)

    def definitions(self):
        return tuple(replace(item, schema=deepcopy(item.schema), display_schema=deepcopy(item.display_schema))
                     for _, item in sorted(self._definitions.items()))

    def definition(self, name, version):
        try:
            item = self._definitions[name, version]
        except KeyError as error:
            raise ValueError(f"Chart logic {name!r} version {version!r} is unavailable.") from error
        return replace(item, schema=deepcopy(item.schema), display_schema=deepcopy(item.display_schema))

    def register_class(self, cls):
        if not inspect.isclass(cls) or not issubclass(cls, BaseChartLogic) or inspect.isabstract(cls):
            raise ValueError("Chart logic must be a concrete BaseChartLogic subclass.")
        for value in (cls.name, cls.version):
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
                raise ValueError("Logic names and versions require nonempty alphanumeric identifiers.")
        key = cls.name, cls.version
        if key in self._classes:
            raise ValueError(f"Duplicate chart logic {key}.")
        schema = json_copy(cls.config_schema)
        Draft202012Validator.check_schema(schema)
        if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
            raise ValueError("Chart settings must declare an object schema with properties.")
        display_schema = self._check_display_schema(cls.display_schema)
        if set(schema["properties"]) & set(display_schema["properties"]):
            raise ValueError("Calculation and display settings must have distinct names.")
        if not isinstance(cls.result_version, str) or not cls.result_version:
            raise ValueError("Chart result_version must be a nonempty string.")
        if not isinstance(cls.implementation_fingerprint, str):
            raise ValueError("Implementation fingerprint must be a string.")
        if cls.input_policy not in ("original", "completed") or type(cls.uses_cutoff) is not bool:
            raise ValueError("Invalid chart logic input policy.")
        if not all(isinstance(value, str) for value in (*cls.required_columns, *cls.supported_timeframes)):
            raise ValueError("Required columns and supported timeframes must contain strings.")
        if not isinstance(cls.label, str) or not isinstance(cls.description, str):
            raise ValueError("Logic labels and descriptions must be strings.")
        self._classes[key] = cls
        self._definitions[key] = LogicDefinition(
            cls.name, cls.version, cls.label or cls.name.replace("_", " ").title(), cls.description, schema, display_schema, cls.result_version,
            cls.implementation_fingerprint)

    def create(self, name, version):
        self.definition(name, version)
        return self._classes[name, version]()

    def defaults(self, name, version):
        return {name: deepcopy(prop["default"]) for name, prop in
                self.definition(name, version).schema["properties"].items() if isinstance(prop, dict) and "default" in prop}

    def configure(self, name, version, config):
        schema = self.definition(name, version).schema
        try:
            result = self.create(name, version).validate(validated_config(config, schema))
            return validated_config(result, schema)
        except Exception as error:
            raise ValueError(f"Chart logic {name!r} has invalid settings: {error}") from error

    @staticmethod
    def _check_display_schema(schema):
        schema = json_copy(schema)
        Draft202012Validator.check_schema(schema)
        if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
            raise ValueError("Display settings must declare an object schema with properties.")
        return schema

    def display_schema(self, name, version, result=None):
        try:
            if result is None:
                schema = self.definition(name, version).display_schema
            else:
                logic = self.create(name, version)
                method = logic.display_schema_for
                if getattr(method, "__func__", None) is BaseChartLogic.display_schema_for:
                    # Static controls never inspect results. Avoid copying a
                    # potentially large archive merely to read their schema.
                    schema = logic.display_schema
                else:
                    schema = method(deepcopy(result))
            return self._check_display_schema(schema)
        except Exception as error:
            raise ValueError(f"Chart logic {name!r} has invalid display controls: {error}") from error

    def display_defaults(self, name, version, result=None):
        return {key: deepcopy(prop["default"]) for key, prop in
                self.display_schema(name, version, result)["properties"].items()
                if isinstance(prop, dict) and "default" in prop}

    def configure_display(self, name, version, config, result):
        try:
            schema = self.display_schema(name, version, result)
            values = validated_config(config, schema)
            normalized = self.create(name, version).validate_display(values, deepcopy(result))
            return validated_config(normalized, schema)
        except Exception as error:
            raise ValueError(f"Chart logic {name!r} has invalid display settings: {error}") from error

    def present(self, name, version, result, config):
        values = self.configure_display(name, version, config, result)
        presented = self.create(name, version).present(deepcopy(result), deepcopy(values))
        if not isinstance(presented, LogicResult):
            raise ValueError("Chart logic presentation must return a LogicResult.")
        return presented.validate()


def discover_logics(package: str | ModuleType = "research.analysis.builtins"):
    """Import trusted local modules; new modules need no package export edits."""
    registry = LogicRegistry()
    importlib.invalidate_caches()
    try:
        root = importlib.import_module(package) if isinstance(package, str) else package
    except Exception as error:
        registry._issues.append(f"Cannot import chart logic package {package!r}: {error}")
        return registry
    identities = {}
    seen_classes = set()

    def visit(module):
        for cls in vars(module).values():
            if (not inspect.isclass(cls) or cls in seen_classes or cls.__module__ != module.__name__
                    or cls is BaseChartLogic or not issubclass(cls, BaseChartLogic) or inspect.isabstract(cls)):
                continue
            seen_classes.add(cls)
            origin = f"{module.__name__}.{cls.__name__}"
            try:
                key = cls.name, cls.version
                if key in identities:
                    registry._classes.pop(key, None)
                    registry._definitions.pop(key, None)
                    registry._issues.append(f"Duplicate chart logic {key}: {identities[key]} and {origin}; unavailable.")
                    continue
                identities[key] = origin
                registry.register_class(cls)
            except Exception as error:
                registry._issues.append(f"Cannot register {origin}: {error}")
        paths = getattr(module, "__path__", None)
        if paths is None:
            return
        try:
            children = sorted(pkgutil.iter_modules(paths, module.__name__ + "."), key=lambda item: item.name)
        except Exception as error:
            registry._issues.append(f"Cannot inspect {module.__name__}: {error}")
            return
        for child in children:
            try:
                imported = importlib.import_module(child.name)
            except Exception as error:
                registry._issues.append(f"Cannot import {child.name}: {error}")
                continue
            visit(imported)

    visit(root)
    return registry
