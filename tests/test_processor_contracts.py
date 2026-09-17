"""Template hooks, explicit contracts, and pipeline-level safety checks."""

from dataclasses import replace
from datetime import timedelta

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from data_pipeline.exceptions import ProcessingError
from data_pipeline.processing import BaseProcessor, DataContract, Pipeline, Processor, ProcessingResult, ScalePrices
from data_pipeline.processing.validators import PreserveKeys


class HookProcessor(BaseProcessor):
    name = "hooks"
    version = "1"
    config = {}

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def _visit(self, hook):
        self.calls.append(hook)
        if self.fail == hook:
            raise ValueError(f"failure in {hook}")

    def _validate_input(self, frame, input_contract):
        self._visit("input")

    def _transform(self, frame, input_contract, output_contract):
        self._visit("transform")
        frame[0, "volume"] = 7.0
        return frame

    def _validate_output(self, original, output, input_contract, output_contract):
        self._visit("output")
        assert original[0, "volume"] != 7.0
        assert output[0, "volume"] == 7.0


@pytest.mark.parametrize("failure,expected", [
    (None, ["input", "transform", "output"]),
    ("input", ["input"]),
    ("transform", ["input", "transform"]),
    ("output", ["input", "transform", "output"]),
])
def test_wrapper_hook_order_and_failure_isolation(sample_ohlcv_frame, failure, expected):
    original = sample_ohlcv_frame.clone()
    processor = HookProcessor(failure)
    assert type(processor).transform is BaseProcessor.transform
    if failure:
        with pytest.raises(ProcessingError, match=f"failure in {failure}"):
            Pipeline([processor]).run(original, DataContract("1h"))
    else:
        result = Pipeline([processor]).run(original, DataContract("1h"))
        assert result.contract == DataContract("1h")
    assert processor.calls == expected
    assert_frame_equal(original, sample_ohlcv_frame)


def test_contract_propagates_to_later_processor(sample_ohlcv_frame):
    calls = []

    class DailyLabel(HookProcessor):
        def output_contract(self, contract):
            assert contract.timeframe == "1h"
            return DataContract("1d")

        def _transform(self, frame, input_contract, output_contract):
            calls.append((input_contract.timeframe, output_contract.timeframe))
            return frame.head(1).with_columns(pl.col("timestamp").dt.truncate("1d"))

        def _validate_output(self, original, output, input_contract, output_contract):
            pass

    class Inspect(ScalePrices):
        def _transform(self, frame, input_contract, output_contract):
            calls.append((input_contract.timeframe, output_contract.timeframe))
            return super()._transform(frame, input_contract, output_contract)

    result = Pipeline([DailyLabel(), Inspect(2)]).run(sample_ohlcv_frame, DataContract("1h"))
    assert result.contract == DataContract("1d")
    assert calls == [("1h", "1d"), ("1d", "1d")]
    assert result.frame["timestamp"][0].hour == 0


def test_pipeline_does_not_impose_key_policy(sample_ohlcv_frame):
    class Shift(HookProcessor):
        def _transform(self, frame, input_contract, output_contract):
            return frame.with_columns(pl.col("timestamp") + pl.duration(minutes=1))

        def _validate_output(self, original, output, input_contract, output_contract):
            assert output["timestamp"].equals(original["timestamp"] + timedelta(minutes=1))

    # New timestamps are accepted when the processor permits them.
    result = Pipeline([Shift()]).run(sample_ohlcv_frame, DataContract("1h"))
    assert result.frame.height == sample_ohlcv_frame.height
    with pytest.raises(ValueError, match="new .*keys"):
        PreserveKeys()(sample_ohlcv_frame, result.frame)


def test_shared_contract_checks_survive_custom_transform_override(sample_ohlcv_frame):
    class Bypass(HookProcessor):
        def output_contract(self, contract):
            raise AssertionError("Pipeline must not resolve processor output contracts.")

        def transform(self, frame, input_contract):
            # Deliberately bypass the wrapper and all hooks.
            return ProcessingResult(frame, DataContract("1d"))

    with pytest.raises(ProcessingError, match="midnight UTC"):
        Pipeline([Bypass()]).run(sample_ohlcv_frame, DataContract("1h"))


@pytest.mark.parametrize("changes", [
    {"schema_version": 2}, {"dataset": "features"}, {"schema_version": True},
    {"timeframe": "bad"}, {"timeframe": None}, {"timestamp_convention": "session_date"},
])
def test_invalid_contract_rejected(changes):
    with pytest.raises(ProcessingError):
        replace(DataContract("1h"), **changes)


def test_contract_alias_and_schema_are_defensive():
    assert DataContract("60m") == DataContract("1h")
    contract = DataContract("1d")
    contract.schema.clear()
    assert len(contract.schema) == 7


def test_invalid_declared_contract_stops_before_computation(sample_ohlcv_frame):
    class Bad(HookProcessor):
        def output_contract(self, contract):
            return "1d"

    bad = Bad()
    with pytest.raises(ProcessingError, match="declare a DataContract"):
        Pipeline([bad]).run(sample_ohlcv_frame, DataContract("1h"))
    assert not bad.calls


@pytest.mark.parametrize("via_pipeline", [False, True])
def test_processor_resolves_contract_once_before_all_hooks(sample_ohlcv_frame, via_pipeline):
    events = []
    declared = DataContract("1d")

    class Daily(BaseProcessor):
        name, version, config = "daily", "1", {}

        def output_contract(self, input_contract):
            events.append("contract")
            assert input_contract == DataContract("1h")
            return declared

        def _validate_input(self, frame, input_contract):
            events.append("input")
            assert input_contract == DataContract("1h")

        def _transform(self, frame, input_contract, output_contract):
            events.append("transform")
            assert output_contract is declared
            return frame.head(1).with_columns(pl.col("timestamp").dt.truncate("1d"))

        def _validate_output(self, original, output, input_contract, output_contract):
            events.append("output")
            assert output_contract is declared
            assert original.height == 3 and output.height == 1

    processor = Daily()
    result = (Pipeline([processor]).run(sample_ohlcv_frame, DataContract("1h")) if via_pipeline
              else processor.transform(sample_ohlcv_frame, DataContract("1h")))
    assert isinstance(result, ProcessingResult)
    assert result.contract is declared
    assert events == ["contract", "input", "transform", "output"]


def test_structural_processor_does_not_need_output_contract_method(sample_ohlcv_frame):
    class Standalone:
        name, version, config = "standalone", "1", {}

        def transform(self, frame, input_contract):
            return ProcessingResult(frame, input_contract)

    processor = Standalone()
    assert isinstance(processor, Processor)
    assert not hasattr(processor, "output_contract")
    result = Pipeline([processor]).run(sample_ohlcv_frame, DataContract("1h"))
    assert_frame_equal(result.frame, sample_ohlcv_frame)
    assert result.contract == DataContract("1h")


@pytest.mark.parametrize("invalid", ["none", "bare-frame", "bad-contract", "missing-contract", "bad-frame", "lazy-frame"])
def test_pipeline_rejects_invalid_processor_results_before_next_step(sample_ohlcv_frame, invalid):
    class Broken:
        name, version, config = "broken_result", "1", {}

        def transform(self, frame, input_contract):
            return {
                "none": None,
                "bare-frame": frame,
                "bad-contract": ProcessingResult(frame, "1h"),
                "missing-contract": ProcessingResult(frame, None),
                "bad-frame": ProcessingResult(None, input_contract),
                "lazy-frame": ProcessingResult(frame.lazy(), input_contract),
            }[invalid]

    later = HookProcessor()
    original = sample_ohlcv_frame.clone()
    with pytest.raises(ProcessingError, match="broken_result.*index 0"):
        Pipeline([Broken(), later]).run(sample_ohlcv_frame, DataContract("1h"))
    assert later.calls == []
    assert_frame_equal(sample_ohlcv_frame, original)


@pytest.mark.parametrize("invalid", [None, "1h", {}, 1])
def test_explicit_contract_is_required_for_processor_and_pipeline(sample_ohlcv_frame, invalid):
    processor = HookProcessor()
    for operation in (
        lambda: processor.transform(sample_ohlcv_frame, invalid),
        lambda: Pipeline([processor]).run(sample_ohlcv_frame, invalid),
        lambda: Pipeline().transform(sample_ohlcv_frame, contract=invalid),
    ):
        with pytest.raises(ProcessingError, match="explicit.*contract"):
            operation()
    assert processor.calls == []


def test_omitting_contract_is_not_silently_inferred(sample_ohlcv_frame):
    with pytest.raises(TypeError, match="input_contract"):
        ScalePrices(2).transform(sample_ohlcv_frame)
    with pytest.raises(TypeError, match="contract"):
        Pipeline([ScalePrices(2)]).transform(sample_ohlcv_frame)


def test_reusing_processor_keeps_contracts_invocation_local(sample_ohlcv_frame):
    processor = ScalePrices(2)
    pipeline = Pipeline([processor])
    fingerprint = pipeline.fingerprint
    original_state = vars(processor).copy()
    daily = sample_ohlcv_frame.head(1).with_columns(pl.col("timestamp").dt.truncate("1d"))
    first = pipeline.run(sample_ohlcv_frame, DataContract("1h"))
    second = pipeline.run(daily, DataContract("1d"))
    third = pipeline.run(sample_ohlcv_frame, DataContract("1h"))
    assert first.contract == third.contract == DataContract("1h")
    assert second.contract == DataContract("1d")
    assert_frame_equal(first.frame, third.frame)
    assert vars(processor) == original_state
    assert pipeline.fingerprint == fingerprint
