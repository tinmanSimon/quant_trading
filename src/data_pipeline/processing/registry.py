"""Automatically discovered, explicitly versioned processor definitions and factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import importlib
import inspect
import json
import pkgutil
from types import ModuleType

from ..exceptions import ProcessingError
from ._config import JSONValue, require_identifier, snapshot_config
from .pipeline import Pipeline, Processor
from .configuration import configuration_schema, construct_from_config, validated_config
from .processors.base import BaseProcessor


ProcessorFactory = Callable[[Mapping[str, JSONValue]], Processor]


@dataclass(frozen=True, slots=True)
class ProcessorDefinition:
    """Display/configuration contract carried entirely by a processor class.

    ``causal`` must be explicitly true before strict research accepts a processor.
    Adjusted-price research makes a narrow exception for the built-in corporate
    action processor without changing its causality declaration.
    ``price_basis`` is either ``preserve`` or ``transformed``. These describe
    semantics without changing the versioned config or historical fingerprints.
    """

    name: str
    version: str
    label: str
    description: str
    schema: dict
    causal: bool = False
    price_basis: str = "preserve"


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
        self._definitions: dict[tuple[str, str], ProcessorDefinition] = {}
        self._issues: list[str] = []

    @property
    def issues(self) -> tuple[str, ...]:
        """Discovery failures; unaffected processors remain available."""
        return tuple(self._issues)

    def definitions(self) -> tuple[ProcessorDefinition, ...]:
        return tuple(replace(definition, schema=deepcopy(definition.schema))
                     for _, definition in sorted(self._definitions.items()))

    def definition(self, name: str, version: str) -> ProcessorDefinition:
        try:
            definition = self._definitions[name, version]
        except KeyError as error:
            raise ProcessingError(f"Unknown processor definition {name!r} version {version!r}.") from error
        return replace(definition, schema=deepcopy(definition.schema))

    def register_class(self, cls: type[BaseProcessor]) -> None:
        """Infer a complete definition and factory from a concrete processor."""
        name, version = getattr(cls, "name", None), getattr(cls, "version", None)
        try:
            if not inspect.isclass(cls) or not issubclass(cls, BaseProcessor) or inspect.isabstract(cls):
                raise TypeError("Only concrete BaseProcessor subclasses can be discovered.")
            name = require_identifier(name, "name")
            version = require_identifier(version, "version")
            schema = configuration_schema(cls)
            causal = getattr(cls, "causal", False)
            price_basis = getattr(cls, "price_basis", "preserve")
            if type(causal) is not bool:
                raise TypeError("causal metadata must be a boolean.")
            if price_basis not in ("preserve", "transformed"):
                raise ValueError("price_basis metadata must be 'preserve' or 'transformed'.")
            label = getattr(cls, "label", None) or name.replace("_", " ").title()
            description = getattr(cls, "description", None) or inspect.cleandoc(cls.__doc__ or "")
            if not isinstance(label, str) or not isinstance(description, str):
                raise TypeError("Processor label and description must be strings.")
            definition = ProcessorDefinition(name, version, label, description, schema, causal, price_basis)

            def factory(config: Mapping[str, JSONValue]) -> Processor:
                values = validated_config(config, schema)
                custom_factory = getattr(cls, "from_config", None)
                return custom_factory(values) if custom_factory is not None else construct_from_config(cls, values)

            self.register(name, version, factory)
            self._definitions[name, version] = definition
        except Exception as exc:
            raise ProcessingError(f"Cannot describe processor {name!r} version {version!r}: {exc}") from exc

    def create(self, name: str, version: str, config: Mapping[str, JSONValue]) -> Processor:
        """Construct one processor, enforcing the same rules as load()."""
        try:
            name = require_identifier(name, "name")
            version = require_identifier(version, "version")
            try:
                factory = self._factories[name, version]
            except KeyError as exc:
                raise ValueError("Unknown processor name/version.") from exc
            processor = factory(snapshot_config(config))
            if processor.name != name or processor.version != version:
                raise ValueError("Factory returned a different processor name/version.")
            return processor
        except Exception as exc:
            raise ProcessingError(f"Processor {name!r} version {version!r} could not be created: {exc}") from exc

    def configure(self, name: str, version: str, config: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
        """Validate settings and return normalized, effective config including defaults."""
        return snapshot_config(self.create(name, version, config).config)

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


def discover_processors(
    package: str | ModuleType = "data_pipeline.processing.processors",
) -> ProcessorRegistry:
    """Discover concrete classes defined in a package and all of its subpackages.

    New modules need no imports/registration elsewhere. Discovery imports code,
    so this is for trusted local processors. Broken modules are isolated. If two
    classes claim an identity, neither is usable; other identities still load.
    Restart the dashboard after changing already-imported processor code.
    """
    registry = ProcessorRegistry()
    importlib.invalidate_caches()
    try:
        root = importlib.import_module(package) if isinstance(package, str) else package
    except Exception as error:
        registry._issues.append(f"Cannot import processor package {package!r}: {error}")
        return registry
    identities: dict[tuple[str, str], str] = {}
    duplicates: set[tuple[str, str]] = set()
    seen_classes: set[type] = set()

    def inspect_module(module: ModuleType) -> None:
        for cls in vars(module).values():
            if (not inspect.isclass(cls) or cls in seen_classes or cls.__module__ != module.__name__
                    or cls is BaseProcessor or not issubclass(cls, BaseProcessor) or inspect.isabstract(cls)):
                continue
            seen_classes.add(cls)
            origin = f"{module.__name__}.{cls.__name__}"
            try:
                name = require_identifier(getattr(cls, "name", None), "name")
                version = require_identifier(getattr(cls, "version", None), "version")
                key = name, version
                if key in identities:
                    duplicates.add(key)
                    registry._factories.pop(key, None)
                    registry._definitions.pop(key, None)
                    registry._issues.append(
                        f"Duplicate processor identity {name!r} version {version!r}: "
                        f"{identities[key]} and {origin}; this identity is unavailable."
                    )
                    continue
                identities[key] = origin
                if key not in duplicates:
                    registry.register_class(cls)
            except Exception as error:
                registry._issues.append(f"Cannot register {origin}: {error}")

    def visit(module: ModuleType) -> None:
        inspect_module(module)
        paths = getattr(module, "__path__", None)
        if paths is None:
            return
        try:
            children = sorted(pkgutil.iter_modules(paths, module.__name__ + "."), key=lambda item: item.name)
        except Exception as error:
            registry._issues.append(f"Cannot inspect processor package {module.__name__}: {error}")
            return
        for child in children:
            try:
                imported = importlib.import_module(child.name)
            except Exception as error:
                registry._issues.append(f"Cannot import processor module {child.name}: {error}")
                continue
            visit(imported)

    visit(root)
    return registry


def load_pipeline(
    specs: str | list[Mapping[str, JSONValue]],
    *,
    registry: ProcessorRegistry | None = None,
) -> Pipeline:
    """Load CLI JSON using discovered processors, or an explicitly supplied registry.

    Example: ``[{"name": "scale_prices", "version": "1", "config": {"factor": 2}}]``.
    The pipeline snapshots effective processor configs (ScalePrices normalizes
    its factor to float), making direct Python and JSON construction agree.
    """
    return (discover_processors() if registry is None else registry).load(specs)
