"""Explicitly versioned processor factories for JSON/CLI configurations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json

from ..exceptions import ProcessingError
from ._config import JSONValue, require_identifier, snapshot_config
from .pipeline import Pipeline, Processor
from .processors import ScalePrices
from .processors.resampling import resampler_from_config


ProcessorFactory = Callable[[Mapping[str, JSONValue]], Processor]


def _unique_object(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value!r} is unsupported.")


class ProcessorRegistry:
    """Factories keyed by exact ``(name, version)``; no implicit latest version.

    A fresh registry is empty. Factories receive a defensive JSON config copy
    and return a Python processor. Registering the same identity twice is an
    error. Direct ``Pipeline`` construction requires no registration.
    """

    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], ProcessorFactory] = {}

    def register(self, name: str, version: str, factory: ProcessorFactory) -> None:
        try:
            name = require_identifier(name, "name")
            version = require_identifier(version, "version")
            if not callable(factory):
                raise TypeError("Processor factory must be callable.")
            if (name, version) in self._factories:
                raise ValueError("Processor identity is already registered.")
            self._factories[name, version] = factory
        except Exception as exc:
            raise ProcessingError(
                f"Cannot register processor {name!r} version {version!r}: {exc}"
            ) from exc

    def load(self, specs: str | list[Mapping[str, JSONValue]]) -> Pipeline:
        """Load an ordered list of exact ``{name, version, config}`` objects."""
        try:
            if isinstance(specs, str):
                specs = json.loads(
                    specs, object_pairs_hook=_unique_object, parse_constant=_reject_constant
                )
            if type(specs) is not list:
                raise TypeError("Pipeline configuration must be a JSON list.")
        except Exception as exc:
            raise ProcessingError(f"Invalid processor pipeline JSON: {exc}") from exc

        processors = []
        for index, spec in enumerate(specs):
            label = f"Processor at index {index}"
            try:
                if not isinstance(spec, Mapping):
                    raise TypeError("Processor specification must be a JSON mapping.")
                name = require_identifier(spec.get("name"), "name")
                label = f"Processor {name!r} at index {index}"
                version = require_identifier(spec.get("version"), "version")
                label = f"Processor {name!r} version {version!r} at index {index}"
                if set(spec) != {"name", "version", "config"}:
                    raise ValueError(
                        "Processor specification requires exactly name, version and config."
                    )
                config = snapshot_config(spec["config"])
                try:
                    factory = self._factories[name, version]
                except KeyError as exc:
                    raise ValueError("Unknown processor name/version.") from exc
                processor = factory(config)
                if processor.name != name or processor.version != version:
                    raise ValueError("Factory returned a different processor name/version.")
                processors.append(processor)
            except Exception as exc:
                raise ProcessingError(f"{label} could not be loaded: {exc}") from exc
        return Pipeline(processors)


_BUILTINS = ProcessorRegistry()
_BUILTINS.register("scale_prices", "1", lambda config: ScalePrices(**config))
_BUILTINS.register("resample_ohlcv", "1", resampler_from_config)


def load_pipeline(
    specs: str | list[Mapping[str, JSONValue]],
    *,
    registry: ProcessorRegistry | None = None,
) -> Pipeline:
    """Load CLI JSON using the built-in registry, or an explicitly supplied one.

    Example: ``[{"name": "scale_prices", "version": "1", "config": {"factor": 2}}]``.
    The pipeline snapshots effective processor configs (ScalePrices normalizes
    its factor to float), making direct Python and JSON construction agree.
    """
    return (_BUILTINS if registry is None else registry).load(specs)
