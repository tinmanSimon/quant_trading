"""Automatic discovery and shared configuration validation for local processors."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest

from data_pipeline.exceptions import ProcessingError
from data_pipeline.processing import (
    DataContract, Pipeline, Processor, ProcessorRegistry, ScalePrices, discover_processors, load_pipeline,
)


_BASE = '''
from data_pipeline.processing import BaseProcessor
from typing import ClassVar

class Identity(BaseProcessor):
    name = "identity"
    version = "1"

    def _validate_input(self, frame, input_contract):
        pass

    def _transform(self, frame, input_contract, output_contract):
        return frame

    def _validate_output(self, input_frame, output_frame, input_contract, output_contract):
        pass
'''


@pytest.fixture
def processor_package(tmp_path, monkeypatch):
    path = tmp_path / "test_local_processors"
    path.mkdir()
    (path / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield path
    for name in list(sys.modules):
        if name == path.name or name.startswith(path.name + "."):
            sys.modules.pop(name, None)


def add_module(package: Path, name: str, source: str):
    path = package / (name + ".py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    importlib.invalidate_caches()


def test_builtins_are_described_and_keep_existing_fingerprints():
    registry = discover_processors()
    assert not registry.issues
    assert {(item.name, item.version) for item in registry.definitions()} == {
        ("scale_prices", "1"), ("resample_ohlcv", "1"),
    }
    scale = registry.definition("scale_prices", "1")
    assert scale.causal is True
    assert scale.price_basis == "transformed"
    assert scale.schema["properties"]["factor"]["exclusiveMinimum"] == 0
    assert scale.schema["required"] == ["factor"]
    assert registry.configure("scale_prices", "1", {"factor": 2}) == {"factor": 2.0}
    assert registry.load([{"name": "scale_prices", "version": "1", "config": {"factor": 2}}]).fingerprint == Pipeline([ScalePrices(2)]).fingerprint
    resample = registry.definition("resample_ohlcv", "1")
    assert resample.schema["properties"]["sessions"]["format"] == "trading_sessions"
    assert resample.schema["properties"]["incomplete"]["default"] == "raise"
    assert resample.causal is True and resample.price_basis == "preserve"
    scale.schema["properties"]["factor"]["exclusiveMinimum"] = -1
    assert registry.definition("scale_prices", "1").schema["properties"]["factor"]["exclusiveMinimum"] == 0


def test_new_module_is_found_without_registration_and_no_config_is_usable(processor_package, sample_ohlcv_frame):
    assert discover_processors(processor_package.name).definitions() == ()
    add_module(processor_package, "identity", _BASE)
    registry = discover_processors(processor_package.name)
    assert not registry.issues
    definition, = registry.definitions()
    assert definition.name == "identity"
    assert definition.schema["properties"] == {}
    assert definition.schema["required"] == []
    assert definition.causal is False
    assert registry.configure("identity", "1", {}) == {}
    processor = registry.create("identity", "1", {})
    assert isinstance(processor, Processor)
    pipeline = registry.load([{"name": "identity", "version": "1", "config": {}}])
    assert pipeline.run(sample_ohlcv_frame, DataContract("1h")).frame.equals(sample_ohlcv_frame)
    with pytest.raises(ProcessingError, match="Additional properties"):
        registry.create("identity", "1", {"unexpected": 2})


def test_default_package_discovers_new_file_without_editing_exports(tmp_path, monkeypatch):
    import data_pipeline.processing.processors as processors
    add_module(tmp_path, "added_identity", _BASE)
    monkeypatch.setattr(processors, "__path__", [*processors.__path__, str(tmp_path)])
    try:
        registry = discover_processors()
        assert registry.definition("identity", "1").schema["properties"] == {}
        assert load_pipeline([{"name": "identity", "version": "1", "config": {}}]).identities[0]["config"] == {}
    finally:
        sys.modules.pop("data_pipeline.processing.processors.added_identity", None)


def test_recursive_discovery_isolates_broken_modules_and_skips_imported_aliases(processor_package):
    add_module(processor_package, "identity", _BASE)
    add_module(processor_package, "broken", "raise RuntimeError('broken module')")
    add_module(processor_package, "aliases", "from .identity import Identity\nAlias = Identity")
    add_module(processor_package, "group/__init__", "")
    add_module(processor_package, "group/nested", _BASE.replace('"identity"', '"nested"'))
    add_module(processor_package, "bad_group/__init__", "raise RuntimeError('bad package')")
    registry = discover_processors(processor_package.name)
    assert {item.name for item in registry.definitions()} == {"identity", "nested"}
    assert len(registry.issues) == 2
    assert any("broken module" in issue for issue in registry.issues)
    assert any("bad package" in issue for issue in registry.issues)


def test_duplicate_identity_is_unavailable_without_hiding_other_processors(processor_package):
    add_module(processor_package, "first", _BASE)
    add_module(processor_package, "second", _BASE)
    add_module(processor_package, "good", _BASE.replace('"identity"', '"good"'))
    registry = discover_processors(processor_package.name)
    assert [item.name for item in registry.definitions()] == ["good"]
    assert len(registry.issues) == 1
    assert "Duplicate processor identity" in registry.issues[0]
    assert "first.Identity" in registry.issues[0] and "second.Identity" in registry.issues[0]
    with pytest.raises(ProcessingError, match="Unknown"):
        registry.create("identity", "1", {})


def test_typed_nested_configuration_defaults_and_constraints(processor_package):
    add_module(processor_package, "typed", _BASE + '''
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Literal

@dataclass(frozen=True)
class Window:
    length: int = field(default=3, metadata={"schema": {"minimum": 1}})
    selection: Literal["first", "last"] = "last"

class Mode(str, Enum):
    FAST = "fast"
    SLOW = "slow"

@dataclass(frozen=True)
class Typed(Identity):
    name: ClassVar[str] = "typed"
    options: Window = field(default_factory=Window)
    weights: tuple[float, ...] = (1.0, 2.0)
    enabled: bool = True
    start: date = date(2024, 1, 2)
    mode: Mode = Mode.FAST
    note: str | None = None
    config_schema: ClassVar[dict] = {"properties": {"weights": {"minItems": 1}}}
''')
    registry = discover_processors(processor_package.name)
    assert not registry.issues
    result = registry.create("typed", "1", {"options": {"length": 7}, "weights": [2, 3.5], "mode": "slow"})
    assert result.options.length == 7 and result.options.selection == "last"
    assert result.weights == (2.0, 3.5)
    assert result.start.isoformat() == "2024-01-02"
    assert result.mode.value == "slow"
    assert result.config == {"options": {"length": 7, "selection": "last"}, "weights": [2.0, 3.5],
                             "enabled": True, "start": "2024-01-02", "mode": "slow", "note": None}
    loaded = registry.load(json.dumps([{"name": "typed", "version": "1", "config": result.config}]))
    assert loaded.identities[0]["config"] == result.config
    for invalid in ({"options": {"length": 0}}, {"options": {"extra": 1}}, {"weights": []},
                    {"enabled": "yes"}, {"start": "yesterday"}, {"mode": "other"}, {"note": 2}):
        with pytest.raises(ProcessingError):
            registry.create("typed", "1", invalid)


def test_typed_constructor_fields_are_configurable_without_dataclasses(processor_package):
    add_module(processor_package, "constructor", _BASE + '''
class Constructor(Identity):
    name = "constructor"
    def __init__(self, factor: float, count: int = 2, enabled: bool = False):
        self.factor, self.count, self.enabled = factor, count, enabled
''')
    registry = discover_processors(processor_package.name)
    assert not registry.issues
    definition = registry.definition("constructor", "1")
    assert definition.schema["required"] == ["factor"]
    assert registry.configure("constructor", "1", {"factor": 4}) == {"factor": 4.0, "count": 2, "enabled": False}
    for invalid in ({}, {"factor": True}, {"factor": "4"}, {"factor": 4, "count": 2.5}):
        with pytest.raises(ProcessingError):
            registry.create("constructor", "1", invalid)


def test_abstract_and_invalid_processors_do_not_block_valid_ones(processor_package):
    add_module(processor_package, "classes", _BASE + '''
from abc import abstractmethod
class Abstract(Identity):
    @abstractmethod
    def extra(self): ...

class Invalid(Identity):
    name = "invalid"
    config_schema = {"properties": {"bad": {"type": "unsupported"}}}
''')
    registry = discover_processors(processor_package.name)
    assert [item.name for item in registry.definitions()] == ["identity"]
    assert len(registry.issues) == 1 and "Invalid" in registry.issues[0]


def test_missing_package_is_reported_as_discovery_issue():
    registry = discover_processors("missing_processor_package_name")
    assert registry.definitions() == ()
    assert "Cannot import processor package" in registry.issues[0]


def test_explicit_class_registration_uses_the_same_schema_validation():
    registry = ProcessorRegistry()
    registry.register_class(ScalePrices)
    with pytest.raises(ProcessingError, match="minimum"):
        registry.configure("scale_prices", "1", {"factor": 0})
    with pytest.raises(ProcessingError, match="already registered"):
        registry.register_class(ScalePrices)


def test_datetime_and_time_fields_accept_iso_offsets_and_validate_syntax(processor_package):
    add_module(processor_package, "times", _BASE + """
from dataclasses import dataclass
from datetime import datetime, time
@dataclass
class Times(Identity):
    name: ClassVar[str] = "times"
    opening: time
    cutoff: datetime
""")
    registry = discover_processors(processor_package.name)
    settings = {"opening": "09:30:00+08:00", "cutoff": "2024-01-02T15:00:00+08:00"}
    assert registry.configure("times", "1", settings) == settings
    with pytest.raises(ProcessingError):
        registry.configure("times", "1", {**settings, "cutoff": "tomorrow"})
