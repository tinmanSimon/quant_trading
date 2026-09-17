"""Offline checks for ordered processing, reproducibility, and OHLCV safety."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline.exceptions import (
    DuplicateBarError,
    EmptyDataError,
    InvalidOHLCVError,
    ProcessingError,
    SchemaValidationError,
)
from data_pipeline.processing import (
    Pipeline,
    Processor,
    ProcessorRegistry,
    ScalePrices,
    load_pipeline,
    BaseProcessor,
    DataContract,
)
from data_pipeline.processing.validators import SubsetKeys
from data_pipeline.schemas import OHLCV_SCHEMA


class PythonProcessor(BaseProcessor):
    """Test processor explicitly selects subset-key validation; callbacks are code."""

    def __init__(
        self,
        name: str = "custom",
        version: str = "1",
        config: object = None,
        action: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.config = {} if config is None else config
        self.action = action

    def _validate_input(self, frame, input_contract):
        pass

    def _validate_output(self, input_frame, output_frame, input_contract, output_contract):
        SubsetKeys()(input_frame, output_frame)

    def _transform(self, frame: pl.DataFrame, input_contract, output_contract) -> pl.DataFrame:
        return frame if self.action is None else self.action(frame)


def test_custom_processor_satisfies_protocol() -> None:
    assert isinstance(PythonProcessor(), Processor)
    assert isinstance(ScalePrices(2), Processor)


def test_pipeline_order_changes_execution_and_fingerprint(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    calls = []

    def add_prices(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append("add")
        return frame.with_columns(
            *[pl.col(column) + 10 for column in ("open", "high", "low", "close")]
        )

    def multiply_prices(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append("multiply")
        return ScalePrices(2).transform(frame, DataContract("1h")).frame

    add = PythonProcessor("add", config={"amount": 10}, action=add_prices)
    multiply = PythonProcessor("multiply", config={"factor": 2}, action=multiply_prices)
    first = Pipeline([add, multiply])
    second = Pipeline([multiply, add])

    result_first = first.transform(sample_ohlcv_frame, contract=DataContract("1h"))
    result_second = second.transform(sample_ohlcv_frame, contract=DataContract("1h"))

    assert calls == ["add", "multiply", "multiply", "add"]
    assert first.fingerprint != second.fingerprint
    assert result_first["close"].to_list() == [
        (price + 10) * 2 for price in sample_ohlcv_frame["close"]
    ]
    assert result_second["close"].to_list() == [
        price * 2 + 10 for price in sample_ohlcv_frame["close"]
    ]


def test_fingerprint_uses_documented_canonical_json() -> None:
    pipeline = Pipeline([ScalePrices(2), ScalePrices(3)])
    expected = (
        '[{"config":{"factor":2.0},"name":"scale_prices","version":"1"},'
        '{"config":{"factor":3.0},"name":"scale_prices","version":"1"}]'
    )
    assert pipeline.canonical_json == expected
    assert pipeline.fingerprint == hashlib.sha256(expected.encode("utf-8")).hexdigest()


def test_fingerprint_is_deterministic_for_nested_mapping_order() -> None:
    left = {
        "z": [None, True, {"b": 2, "a": "价格"}],
        "a": {"second": 1.5, "first": False},
    }
    right = {
        "a": {"first": False, "second": 1.5},
        "z": [None, True, {"a": "价格", "b": 2}],
    }
    first = Pipeline([PythonProcessor(config=left)])
    second = Pipeline([PythonProcessor(config=right)])

    assert first.canonical_json == second.canonical_json
    assert first.fingerprint == second.fingerprint
    assert "\\u4ef7" in first.canonical_json
    assert Pipeline([PythonProcessor(config=left)]).fingerprint == first.fingerprint


@pytest.mark.parametrize(
    "processor",
    [
        PythonProcessor(name="different", config={"items": [1, 2]}),
        PythonProcessor(version="2", config={"items": [1, 2]}),
        PythonProcessor(config={"items": [2, 1]}),
        PythonProcessor(config={"items": [1, 3]}),
        PythonProcessor(config={"items": [1, 2], "extra": None}),
    ],
)
def test_name_version_and_config_participate_in_fingerprint(
    processor: PythonProcessor,
) -> None:
    original = Pipeline([PythonProcessor(config={"items": [1, 2]})])
    assert Pipeline([processor]).fingerprint != original.fingerprint


def test_repeated_processors_are_preserved() -> None:
    single = Pipeline([ScalePrices(2)])
    repeated = Pipeline([ScalePrices(2), ScalePrices(2)])
    assert repeated.identities == single.identities * 2
    assert repeated.fingerprint != single.fingerprint


def test_snapshot_is_deep_and_metadata_access_is_defensive() -> None:
    config = {"nested": [{"threshold": 1}]}
    processor = PythonProcessor(config=config)
    processors = [processor]
    pipeline = Pipeline(processors)
    original_json, original_fingerprint = pipeline.canonical_json, pipeline.fingerprint

    config["nested"][0]["threshold"] = 99
    processors.append(ScalePrices(2))
    exposed = pipeline.identities
    exposed[0]["config"]["nested"][0]["threshold"] = 77
    exposed.clear()

    assert pipeline.canonical_json == original_json
    assert pipeline.fingerprint == original_fingerprint
    assert pipeline.identities == [
        {"name": "custom", "version": "1", "config": {"nested": [{"threshold": 1}]}}
    ]


def test_changing_source_processor_sequence_does_not_change_execution(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    processors = [ScalePrices(2)]
    pipeline = Pipeline(processors)
    processors[:] = [ScalePrices(3)]
    assert_frame_equal(
        pipeline.transform(sample_ohlcv_frame, contract=DataContract("1h")),
        ScalePrices(2).transform(sample_ohlcv_frame, DataContract("1h")).frame,
    )


@pytest.mark.parametrize(
    "field,value", [("name", "changed"), ("version", "2"), ("config", {"new": 1})]
)
def test_identity_drift_is_rejected_before_execution(
    sample_ohlcv_frame: pl.DataFrame, field: str, value: object
) -> None:
    calls = []
    processor = PythonProcessor(action=lambda frame: calls.append(True) or frame)
    pipeline = Pipeline([processor])
    setattr(processor, field, value)

    with pytest.raises(ProcessingError, match="custom.*changed after.*snapshot") as error:
        pipeline.transform(sample_ohlcv_frame, contract=DataContract("1h"))
    assert isinstance(error.value.__cause__, ValueError)
    assert calls == []


def test_identity_changes_during_transform_are_rejected(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    processor = PythonProcessor()

    def change_config(frame: pl.DataFrame) -> pl.DataFrame:
        processor.config["new"] = 1
        return frame

    processor.action = change_config
    pipeline = Pipeline([processor])
    with pytest.raises(ProcessingError, match="custom.*changed after.*snapshot"):
        pipeline.transform(sample_ohlcv_frame, contract=DataContract("1h"))


@pytest.mark.parametrize("field", ["name", "version", "config", "transform"])
def test_contract_fields_are_required(field: str) -> None:
    class Incomplete:
        pass

    processor = Incomplete()
    attrs = {"name": "custom", "version": "1", "config": {}, "transform": lambda frame: frame}
    for key, value in attrs.items():
        if key != field:
            setattr(processor, key, value)
    with pytest.raises(ProcessingError, match=field) as error:
        Pipeline([processor])
    assert isinstance(error.value.__cause__, AttributeError)


@pytest.mark.parametrize("field", ["name", "version"])
@pytest.mark.parametrize("value", [None, "", "  ", 1, True, lambda: "1"])
def test_identity_fields_must_be_explicit_strings(field: str, value: object) -> None:
    processor = PythonProcessor()
    setattr(processor, field, value)
    with pytest.raises(ProcessingError, match=field) as error:
        Pipeline([processor])
    assert isinstance(error.value.__cause__, ValueError)


def test_non_callable_transform_is_rejected() -> None:
    processor = PythonProcessor()
    processor.transform = None
    with pytest.raises(ProcessingError, match="custom.*transform must be callable"):
        Pipeline([processor])


@pytest.mark.parametrize("value", [None, [], "{}", 2, lambda: {}])
def test_config_must_be_mapping(value: object) -> None:
    processor = PythonProcessor()
    processor.config = value
    with pytest.raises(ProcessingError, match="custom.*JSON mapping"):
        Pipeline([processor])


@pytest.mark.parametrize(
    "value",
    [
        float("nan"), float("inf"), float("-inf"),
        lambda: 1, object(), Path("relative"), datetime(2024, 1, 1),
        Decimal("2"), {1, 2}, (1, 2), b"bytes", {1: "integer-key"},
    ],
)
def test_config_rejects_unstable_or_non_json_nested_values(value: object) -> None:
    with pytest.raises(ProcessingError, match="custom.*config") as error:
        Pipeline([PythonProcessor(config={"nested": [{"value": value}]})])
    assert isinstance(error.value.__cause__, (TypeError, ValueError))


def test_circular_configs_are_rejected() -> None:
    config = {}
    config["self"] = config
    with pytest.raises(ProcessingError, match="custom.*circular") as error:
        Pipeline([PythonProcessor(config=config)])
    assert isinstance(error.value.__cause__, ValueError)


def test_json_encoder_failure_is_chained_and_identifies_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A deterministic stand-in for encoder limits, such as Python's limit on
    # decimal digits in huge integers, independent of interpreter settings.
    failure = ValueError("JSON encoder limit exceeded")

    def fail(value: object) -> str:
        raise failure

    monkeypatch.setattr("data_pipeline.processing.pipeline.canonical_json", fail)
    with pytest.raises(ProcessingError, match="custom.*JSON encoder limit") as error:
        Pipeline([PythonProcessor(config={"value": 1})])
    assert error.value.__cause__ is failure


def test_read_only_mappings_and_repeated_non_circular_values_are_supported() -> None:
    shared = MappingProxyType({"x": 1})
    config = MappingProxyType({"items": [shared, shared]})
    pipeline = Pipeline([PythonProcessor(config=config)])
    assert pipeline.identities[0]["config"] == {"items": [{"x": 1}, {"x": 1}]}


@pytest.mark.parametrize("processors", [set(), frozenset(), {}, "scale_prices", b"", None])
def test_unordered_or_invalid_processor_sequences_are_rejected(processors: object) -> None:
    with pytest.raises(ProcessingError, match="processor sequence"):
        Pipeline(processors)


def test_bare_callable_is_not_a_processor() -> None:
    with pytest.raises(ProcessingError, match="index 0"):
        Pipeline([lambda frame: frame])


def test_ordered_generator_is_snapshotted() -> None:
    pipeline = Pipeline(ScalePrices(factor) for factor in [2, 3])
    assert [identity["config"]["factor"] for identity in pipeline.identities] == [2, 3]


@pytest.mark.parametrize("variant", ["canonical", "empty", "noncanonical"])
def test_empty_pipeline_validates_explicit_contract_and_returns_a_clone(
    sample_ohlcv_frame: pl.DataFrame, variant: str
) -> None:
    frame = sample_ohlcv_frame
    if variant == "empty":
        frame = frame.head(0)
    elif variant == "noncanonical":
        frame = frame.reverse().select("close", "symbol")
    pipeline = Pipeline()
    if variant != "canonical":
        with pytest.raises(ProcessingError, match="Input failed contract validation"):
            pipeline.transform(frame, contract=DataContract("1h"))
        return
    output = pipeline.transform(frame, contract=DataContract("1h"))

    assert output is not frame
    assert_frame_equal(output, frame)
    assert pipeline.identities == []
    assert pipeline.canonical_json == "[]"
    assert pipeline.fingerprint == hashlib.sha256(b"[]").hexdigest()


@pytest.mark.parametrize("processors", [[], [ScalePrices(2)]])
def test_non_dataframe_input_has_a_chained_error(processors: list[Processor]) -> None:
    with pytest.raises(ProcessingError, match="Polars DataFrame") as error:
        Pipeline(processors).transform("not a dataframe", contract=DataContract("1h"))
    assert isinstance(error.value.__cause__, TypeError)


def test_scale_prices_only_changes_ohlc(sample_ohlcv_frame: pl.DataFrame) -> None:
    original = sample_ohlcv_frame.clone()
    processor = ScalePrices(2)
    result = Pipeline([processor]).transform(sample_ohlcv_frame, contract=DataContract("1h"))

    assert result.schema == OHLCV_SCHEMA
    for column in ("open", "high", "low", "close"):
        assert result[column].to_list() == [value * 2 for value in original[column]]
    assert_frame_equal(
        result.select("timestamp", "symbol", "volume"),
        original.select("timestamp", "symbol", "volume"),
    )
    assert_frame_equal(sample_ohlcv_frame, original)
    assert processor.name == "scale_prices"
    assert processor.version == "1"
    assert processor.config == {"factor": 2.0}
    processor.config["factor"] = 99
    assert processor.factor == 2.0


@pytest.mark.parametrize(
    "factor",
    [0, -1, math.nan, math.inf, -math.inf, True, False, "2", None, Decimal("2"), 10**400],
)
def test_scale_prices_rejects_invalid_factor(factor: object) -> None:
    with pytest.raises(ProcessingError, match="scale_prices.*positive finite") as error:
        ScalePrices(factor)
    assert isinstance(error.value.__cause__, (TypeError, ValueError, OverflowError))


@pytest.mark.parametrize("factor", [1e308, 5e-324])
def test_scaled_results_must_still_be_valid(
    sample_ohlcv_frame: pl.DataFrame, factor: float
) -> None:
    frame = sample_ohlcv_frame
    if factor < 1:
        frame = frame.with_columns(
            *[pl.lit(0.1).alias(column) for column in ("open", "high", "low", "close")]
        )
    with pytest.raises(ProcessingError, match="scale_prices") as error:
        Pipeline([ScalePrices(factor)]).transform(frame, contract=DataContract("1h"))
    assert isinstance(error.value.__cause__, InvalidOHLCVError)


@pytest.mark.parametrize(
    "action,cause",
    [
        (lambda frame: None, SchemaValidationError),
        (lambda frame: frame.lazy(), SchemaValidationError),
        (lambda frame: frame.with_columns(pl.lit(1).alias("feature")), SchemaValidationError),
        (lambda frame: frame.drop("volume"), SchemaValidationError),
        (lambda frame: frame.with_columns(pl.lit("bad").alias("close")), SchemaValidationError),
        (
            lambda frame: frame.with_columns(pl.col("timestamp").dt.replace_time_zone(None)),
            SchemaValidationError,
        ),
        (lambda frame: frame.with_columns(pl.lit(-1.0).alias("volume")), InvalidOHLCVError),
        (lambda frame: frame.with_columns(pl.lit(math.nan).alias("open")), InvalidOHLCVError),
        (lambda frame: frame.with_columns(pl.lit(1.0).alias("high")), InvalidOHLCVError),
        (lambda frame: frame.reverse(), InvalidOHLCVError),
        (lambda frame: pl.concat([frame.head(1), frame]), DuplicateBarError),
        (lambda frame: frame.head(0), EmptyDataError),
    ],
)
def test_invalid_transform_result_preserves_validation_cause_and_stops(
    sample_ohlcv_frame: pl.DataFrame,
    action: Callable[[pl.DataFrame], pl.DataFrame],
    cause: type[Exception],
) -> None:
    original = sample_ohlcv_frame.clone()
    later_calls = []
    bad = PythonProcessor("bad_output", version="7", action=action)
    later = PythonProcessor("later", action=lambda frame: later_calls.append(True) or frame)

    with pytest.raises(ProcessingError, match="bad_output.*'7'.*index 1") as error:
        Pipeline([ScalePrices(2), bad, later]).transform(sample_ohlcv_frame, contract=DataContract("1h"))

    assert isinstance(error.value.__cause__, cause)
    assert later_calls == []
    assert_frame_equal(sample_ohlcv_frame, original)


def test_invalid_input_is_rejected_before_processor_runs(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    calls = []
    processor = PythonProcessor(action=lambda frame: calls.append(True) or frame)
    with pytest.raises(ProcessingError, match="custom") as error:
        Pipeline([processor]).transform(sample_ohlcv_frame.reverse(), contract=DataContract("1h"))
    assert isinstance(error.value.__cause__, InvalidOHLCVError)
    assert calls == []


def test_failure_preserves_exact_exception_and_input_after_in_place_mutation(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    original = sample_ohlcv_frame.clone()
    failure = RuntimeError("deliberate failure")
    received = []

    def fail(frame: pl.DataFrame) -> pl.DataFrame:
        received.append(frame)
        frame[0, "volume"] = 999.0
        raise failure

    with pytest.raises(ProcessingError, match="explodes.*deliberate failure") as error:
        Pipeline([PythonProcessor("explodes", action=fail)]).transform(sample_ohlcv_frame, contract=DataContract("1h"))
    assert error.value.__cause__ is failure
    assert received[0] is not sample_ohlcv_frame
    assert_frame_equal(sample_ohlcv_frame, original)


def test_successful_in_place_mutation_is_isolated_and_repeatable(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    original = sample_ohlcv_frame.clone()

    def mutate(frame: pl.DataFrame) -> pl.DataFrame:
        frame[0, "volume"] = 999.0
        return frame

    pipeline = Pipeline([PythonProcessor(action=mutate)])
    first = pipeline.transform(sample_ohlcv_frame, contract=DataContract("1h"))
    second = pipeline.transform(sample_ohlcv_frame, contract=DataContract("1h"))
    assert first[0, "volume"] == 999.0
    assert_frame_equal(first, second)
    assert_frame_equal(sample_ohlcv_frame, original)


def test_row_filtering_preserves_canonical_schema(sample_ohlcv_frame: pl.DataFrame) -> None:
    pipeline = Pipeline(
        [PythonProcessor("filter", action=lambda frame: frame.head(2)), ScalePrices(2)]
    )
    filtered = pipeline.transform(sample_ohlcv_frame, contract=DataContract("1h"))
    assert filtered.schema == OHLCV_SCHEMA
    assert_frame_equal(filtered, ScalePrices(2).transform(sample_ohlcv_frame.head(2), DataContract("1h")).frame)


@pytest.mark.parametrize(
    "action",
    [
        lambda frame: frame.with_columns(pl.lit("NEW").alias("symbol")),
        lambda frame: frame.with_columns(pl.col("timestamp") + pl.duration(days=1)),
        lambda frame: pl.concat(
            [frame, frame.tail(1).with_columns(pl.col("timestamp") + pl.duration(days=1))]
        ),
    ],
)
def test_new_bar_keys_are_rejected(
    sample_ohlcv_frame: pl.DataFrame, action: Callable[[pl.DataFrame], pl.DataFrame]
) -> None:
    with pytest.raises(ProcessingError, match="new_keys.*new .*keys") as error:
        Pipeline([PythonProcessor("new_keys", action=action)]).transform(sample_ohlcv_frame, contract=DataContract("1h"))
    assert isinstance(error.value.__cause__, ValueError)


def test_keys_are_checked_against_each_stage_not_only_original_input(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    drop = PythonProcessor("drop", action=lambda frame: frame.head(1))
    restore = PythonProcessor("restore", action=lambda frame: sample_ohlcv_frame.clone())
    with pytest.raises(ProcessingError, match="restore.*new .*keys"):
        Pipeline([drop, restore]).transform(sample_ohlcv_frame, contract=DataContract("1h"))


def test_existing_symbols_and_timestamps_cannot_form_new_key_pairs(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    frame = sample_ohlcv_frame.with_columns(pl.Series("symbol", ["AAPL", "AAPL", "MSFT"]))

    def recombine(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.Series("symbol", ["AAPL", "MSFT", "MSFT"]))

    with pytest.raises(ProcessingError, match="recombine.*new .*keys"):
        Pipeline([PythonProcessor("recombine", action=recombine)]).transform(frame, contract=DataContract("1h"))


def test_key_snapshot_survives_in_place_key_mutation(sample_ohlcv_frame: pl.DataFrame) -> None:
    original = sample_ohlcv_frame.clone()

    def mutate(frame: pl.DataFrame) -> pl.DataFrame:
        frame.replace_column(
            frame.get_column_index("symbol"), pl.Series("symbol", ["NEW"] * frame.height)
        )
        return frame

    with pytest.raises(ProcessingError, match="new .*keys"):
        Pipeline([PythonProcessor(action=mutate)]).transform(sample_ohlcv_frame, contract=DataContract("1h"))
    assert_frame_equal(sample_ohlcv_frame, original)


def test_validator_normalizes_schema_after_each_stage(sample_ohlcv_frame: pl.DataFrame) -> None:
    observed = []

    def reorder(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.select(*reversed(frame.columns)).with_columns(pl.col("volume").cast(pl.Int64))

    def inspect(frame: pl.DataFrame) -> pl.DataFrame:
        observed.append(frame.schema)
        return frame

    pipeline = Pipeline(
        [PythonProcessor("reorder", action=reorder), PythonProcessor("inspect", action=inspect)]
    )
    result = pipeline.transform(sample_ohlcv_frame, contract=DataContract("1h"))
    assert observed == [OHLCV_SCHEMA]
    assert result.schema == OHLCV_SCHEMA


@pytest.mark.parametrize("as_json", [False, True])
def test_builtin_loader_matches_direct_python_pipeline(
    sample_ohlcv_frame: pl.DataFrame, as_json: bool
) -> None:
    specs = [
        {"name": "scale_prices", "version": "1", "config": {"factor": 2}},
        {"config": {"factor": 3}, "version": "1", "name": "scale_prices"},
    ]
    loaded = load_pipeline(json.dumps(specs) if as_json else specs)
    direct = Pipeline([ScalePrices(2), ScalePrices(3)])
    assert loaded.identities == direct.identities
    assert loaded.canonical_json == direct.canonical_json
    assert loaded.fingerprint == direct.fingerprint
    assert_frame_equal(loaded.transform(sample_ohlcv_frame, contract=DataContract("1h")), direct.transform(sample_ohlcv_frame, contract=DataContract("1h")))
    assert specs[0]["config"] == {"factor": 2}


@pytest.mark.parametrize("specs", [[], "[]"])
def test_loader_supports_empty_pipeline(specs: object) -> None:
    assert load_pipeline(specs).fingerprint == Pipeline().fingerprint


@pytest.mark.parametrize(
    "specs",
    [
        None, {}, (), "{}", "null", "[", [None], ["scale_prices"],
        [{"name": "scale_prices", "config": {"factor": 2}}],
        [{"name": "scale_prices", "version": 1, "config": {"factor": 2}}],
        [{"name": "scale_prices", "version": "", "config": {"factor": 2}}],
        [{"name": "scale_prices", "version": "1"}],
        [{"name": "scale_prices", "version": "1", "config": {}, "extra": True}],
        [{"name": "scale_prices", "version": "1", "config": {}}],
        [{"name": "scale_prices", "version": "1", "config": {"factor": 2, "extra": 3}}],
        [{"name": "scale_prices", "version": "1", "config": {"factor": True}}],
        [{"name": "scale_prices", "version": "1", "config": {"factor": lambda: 2}}],
        [{"name": "scale_prices", "version": "1", "config": None}],
        '[{"name":"scale_prices","version":"1","config":{"factor":NaN}}]',
        '[{"name":"scale_prices","version":"1","config":{"factor":Infinity}}]',
        '[{"name":"scale_prices","version":"1","config":{"factor":1e999}}]',
        '[{"name":"scale_prices","version":"1","version":"2","config":{"factor":2}}]',
        '[{"name":"scale_prices","version":"1","config":{"factor":2,"factor":3}}]',
    ],
)
def test_loader_rejects_malformed_ambiguous_or_invalid_configs(specs: object) -> None:
    with pytest.raises(ProcessingError) as error:
        load_pipeline(specs)
    assert error.value.__cause__ is not None


@pytest.mark.parametrize("name,version", [("missing", "1"), ("scale_prices", "2")])
def test_loader_requires_exact_registered_name_and_version(name: str, version: str) -> None:
    with pytest.raises(ProcessingError, match=f"{name}.*{version}.*Unknown"):
        load_pipeline([{"name": name, "version": version, "config": {"factor": 2}}])


def test_registry_supports_custom_factories_without_mutating_caller_config(
    sample_ohlcv_frame: pl.DataFrame,
) -> None:
    registry = ProcessorRegistry()

    def factory(config: Mapping[str, object]) -> Processor:
        config["nested"]["factory"] = True
        return PythonProcessor("custom", version="v2", config=config)

    registry.register("custom", "v2", factory)
    specs = [{"name": "custom", "version": "v2", "config": {"nested": {"input": True}}}]
    pipeline = load_pipeline(specs, registry=registry)

    assert specs[0]["config"] == {"nested": {"input": True}}
    assert pipeline.identities[0]["config"] == {"nested": {"input": True, "factory": True}}
    assert_frame_equal(pipeline.transform(sample_ohlcv_frame, contract=DataContract("1h")), sample_ohlcv_frame)


def test_registry_rejects_duplicate_registration() -> None:
    registry = ProcessorRegistry()
    registry.register("custom", "1", lambda config: PythonProcessor(config=config))
    with pytest.raises(ProcessingError, match="custom.*already registered"):
        registry.register("custom", "1", lambda config: PythonProcessor(config=config))


@pytest.mark.parametrize(
    "name,version,factory",
    [
        ("", "1", lambda config: PythonProcessor()),
        ("custom", None, lambda config: PythonProcessor()),
        ("custom", "1", None),
    ],
)
def test_registry_rejects_invalid_registration(
    name: object, version: object, factory: object
) -> None:
    with pytest.raises(ProcessingError) as error:
        ProcessorRegistry().register(name, version, factory)
    assert error.value.__cause__ is not None


def test_registry_rejects_factory_identity_mismatch() -> None:
    registry = ProcessorRegistry()
    registry.register("custom", "1", lambda config: PythonProcessor(version="2"))
    with pytest.raises(ProcessingError, match="custom.*different processor name/version"):
        registry.load([{"name": "custom", "version": "1", "config": {}}])


def test_registry_preserves_factory_failure_cause() -> None:
    failure = RuntimeError("factory failed")

    def factory(config: Mapping[str, object]) -> Processor:
        raise failure

    registry = ProcessorRegistry()
    registry.register("custom", "1", factory)
    with pytest.raises(ProcessingError, match="custom.*factory failed") as error:
        registry.load([{"name": "custom", "version": "1", "config": {}}])
    assert error.value.__cause__ is failure
