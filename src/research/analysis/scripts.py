"""Adapt ordinary, typed Python functions to the reusable analysis contract."""

from copy import deepcopy
from dataclasses import field, fields, make_dataclass
from hashlib import sha256
import inspect
import re
from typing import Any, get_type_hints

from data_pipeline.processing.configuration import (
    configuration_schema, construct_from_config, validated_config,
)

from .base import BaseChartLogic, LogicContext, LogicResult, json_copy
from .context import AnalysisContext
from .results import AnalysisResult


TOOLKIT_VERSION = "1"
_METADATA = "__research_analysis__"


def analysis(name, *, label=None, description=None, version="1", parameters=None, uses_cutoff=False):
    """Declare an analysis without wrapping or changing its Python function.

    The first argument is an AnalysisContext; remaining arguments are settings.
    Typed defaults become dashboard controls. ``parameters`` adds per-setting
    JSON Schema constraints, e.g. ``{'period': {'minimum': 2}}``.
    Use relative imports for helpers in a private analysis package.
    By default caching follows completed bars. Set ``uses_cutoff=True`` when
    rules also depend on the exact context.cutoff between bar completions.
    """
    for value in (name, version):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError("Analysis names and versions must be nonempty alphanumeric identifiers.")
    if type(uses_cutoff) is not bool:
        raise ValueError("uses_cutoff must be a boolean.")
    if label is not None and (not isinstance(label, str) or not label.strip()):
        raise ValueError("Analysis labels must be nonempty strings.")
    if description is not None and not isinstance(description, str):
        raise ValueError("Analysis descriptions must be strings.")

    def decorate(function):
        if (not inspect.isfunction(function) or inspect.iscoroutinefunction(function)
                or inspect.isgeneratorfunction(function) or inspect.isasyncgenfunction(function)):
            raise TypeError("An analysis must be a synchronous Python function.")
        signature = inspect.signature(function)
        arguments = list(signature.parameters.values())
        if not arguments or arguments[0].kind not in (inspect.Parameter.POSITIONAL_ONLY,
                                                       inspect.Parameter.POSITIONAL_OR_KEYWORD):
            raise TypeError("The first analysis argument must accept an AnalysisContext.")
        hints = get_type_hints(function, include_extras=True)
        settings = []
        for parameter in arguments[1:]:
            if parameter.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                raise TypeError("Analysis settings must be named parameters, without *args or **kwargs.")
            default = parameter.default
            annotation = hints.get(parameter.name, type(default) if default not in (None, inspect.Parameter.empty) else Any)
            if default is inspect.Parameter.empty:
                settings.append((parameter.name, annotation))
            else:
                settings.append((parameter.name, annotation, field(default_factory=lambda value=deepcopy(default): deepcopy(value))))
        configuration = make_dataclass("AnalysisSettings", settings, kw_only=True)
        overrides = json_copy(parameters or {})
        if not isinstance(overrides, dict) or set(overrides) - {item[0] for item in settings}:
            raise ValueError("Parameter metadata must describe declared function settings.")
        configuration.config_schema = {"properties": overrides}
        schema = configuration_schema(configuration)
        setattr(function, _METADATA, {"name": name, "version": version,
                                     "label": label or name.replace("_", " ").title(),
                                     "description": description if description is not None else inspect.getdoc(function) or "",
                                     "schema": schema, "configuration": configuration,
                                     "uses_cutoff": uses_cutoff})
        return function
    return decorate


def is_analysis(function):
    return inspect.isfunction(function) and hasattr(function, _METADATA)


def _configuration(function, values):
    metadata = getattr(function, _METADATA)
    values = validated_config(values, metadata["schema"])
    typed = construct_from_config(metadata["configuration"], values)
    return values, {item.name: getattr(typed, item.name) for item in fields(typed)}


def run_script(function, context, **settings):
    """Execute the same script from research, returning serializable findings.

    Accepts an AnalysisContext or LogicContext. Calling the decorated function
    directly is also supported; this helper additionally validates settings.
    """
    if not is_analysis(function):
        raise TypeError("Use @analysis to declare the script before running it.")
    if isinstance(context, LogicContext):
        context = AnalysisContext.from_logic(context, shared_cache=context.calculation_cache)
    if not isinstance(context, AnalysisContext):
        raise TypeError("Scripts require an AnalysisContext or LogicContext.")
    _, values = _configuration(function, settings)
    result = function(context, **values)
    if not isinstance(result, AnalysisResult):
        raise TypeError("An analysis script must return ctx.result() with text or tables.")
    return result.to_logic_result()


def _layers(result):
    layers = result.data.get("analysis", {}).get("layers", [])
    names = [item["name"] for item in layers]
    labels = [item["name"] if names.count(item["name"]) == 1 else f"{item['name']} ({item['type']})"
              for item in layers]
    return {(label if labels.count(label) == 1 else f"{label} [{item['id']}]"): item["id"]
            for label, item in zip(labels, layers)}


def script_class(function, fingerprint=""):
    """Internal adapter; metadata identities stay stable when code is reloaded."""
    if not is_analysis(function):
        raise TypeError("Expected a decorated analysis function.")
    metadata = getattr(function, _METADATA)
    if not fingerprint:
        try:
            fingerprint = sha256(inspect.getsource(function).encode()).hexdigest()
        except (OSError, TypeError):
            raise ValueError("An explicit fingerprint is required for a dynamically defined script.") from None

    class ScriptAnalysis(BaseChartLogic):
        name = metadata["name"]
        version = metadata["version"]
        label = metadata["label"]
        description = metadata["description"]
        config_schema = deepcopy(metadata["schema"])
        implementation_fingerprint = fingerprint
        input_policy = "completed"
        uses_cutoff = metadata["uses_cutoff"]

        def validate(self, config):
            return _configuration(function, config)[0]

        def calculate(self, context, config, progress=None):
            result = run_script(function, context, **config)
            result.data["implementation_fingerprint"] = self.implementation_fingerprint
            return result

        def display_schema_for(self, result):
            options = list(_layers(result))
            properties = ({"visible_layers": {"type": "array", "title": "Visible chart layers",
                                               "items": {"type": "string", "enum": options},
                                               "uniqueItems": True, "default": options,
                                               "x-ui-options": options}} if options else {})
            return {"type": "object", "properties": properties, "additionalProperties": False}

        def present(self, result, config):
            layers = _layers(result)
            visible = {layers[name] for name in config.get("visible_layers", layers)}
            result.overlays = [item for item in result.overlays if item["id"] in visible]
            return result

    return ScriptAnalysis
