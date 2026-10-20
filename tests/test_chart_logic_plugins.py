"""Built-in chart logics operate on original bars with declarative output."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import math

import polars as pl
import pytest

from chart_logics.base import LogicContext, LogicResult
from chart_logics.plugins.open_price_bands import OpenPriceBandsLogic
from chart_logics.plugins.trend import TrendLogic
from chart_logics.registry import discover_logics
from dashboard.trend_selection import select_trends
from trend_analysis import TrendConfig, analyze


def bars(stamps, opens=None):
    opens = opens if opens is not None else [100.0 + index for index in range(len(stamps))]
    return pl.DataFrame({"timestamp": stamps, "open": opens, "close": opens})


def context(frame, timeframe="1m", support=None, cutoff=None):
    return LogicContext(frame=frame, timeframe=timeframe, identity="TEST",
                        cutoff=cutoff or datetime(2025, 1, 1, tzinfo=UTC), support_frame=support)


def configured(logic, **values):
    registry = discover_logics()
    return registry.configure(logic.name, logic.version, values)


def trend_configured(settings, **values):
    logic = TrendLogic()
    calculation = {name: value for name, value in settings.to_dict().items()
                   if name in logic.config_schema["properties"]}
    return configured(logic, **{**calculation, **values})


def presented(logic, result, **values):
    return discover_logics().present(logic.name, logic.version, result, values)


def level(result, suffix):
    return [item for item in result.overlays if item["id"].endswith(":" + suffix)]


def test_builtins_are_discovered_and_all_defaults_validate():
    registry = discover_logics()
    assert not registry.issues
    assert {item.name for item in registry.definitions()} >= {"trend", "open_price_bands"}
    for definition in registry.definitions():
        assert registry.configure(definition.name, definition.version, {}) == registry.defaults(
            definition.name, definition.version)


@pytest.mark.parametrize("reference,expected", [
    ("first_bar", [105.0, 105.0]),
    ("each_bar", [105.0, 210.0, 315.0]),
])
def test_percentage_levels_use_selected_open_reference(reference, expected):
    frame = bars([datetime(2024, 1, 2, 15, i, tzinfo=UTC) for i in range(3)], [100., 200., 300.])
    logic = OpenPriceBandsLogic()
    result = logic.calculate(context(frame), configured(logic, reference=reference))
    assert level(result, "upper")[0]["y"] == pytest.approx(expected)
    assert level(result, "lower")[0]["y"] == pytest.approx([price * 95 / 105 for price in expected])
    assert level(result, "open")[0]["x"] == ([0, 2] if reference == "first_bar" else [0, 1, 2])
    result.validate()
    json.dumps(result.to_dict(), allow_nan=False)
    assert frame["open"].to_list() == [100., 200., 300.]


def test_session_opens_reset_across_dst_without_bridging_sessions():
    # Friday is EST, Monday is EDT. Both are actual NYSE opening times.
    stamps = [datetime(2024, 3, 8, 14, 30, tzinfo=UTC), datetime(2024, 3, 8, 14, 31, tzinfo=UTC),
              datetime(2024, 3, 11, 13, 30, tzinfo=UTC), datetime(2024, 3, 11, 13, 31, tzinfo=UTC)]
    logic = OpenPriceBandsLogic()
    result = logic.calculate(context(bars(stamps, [100., 101., 200., 201.])), configured(logic))
    upper = level(result, "upper")
    assert [item["x"] for item in upper] == [[0, 1, 2, 2, 3]]
    assert [item["y"] for item in upper] == [[105., 105., None, 210., 210.]]
    assert all(item["type"] == "line" for item in upper)
    result.validate()


def test_mid_session_selection_never_substitutes_first_visible_open():
    opening = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    frame = bars([opening + timedelta(minutes=30), opening + timedelta(minutes=31)], [120., 121.])
    logic = OpenPriceBandsLogic()
    missing = logic.calculate(context(frame), configured(logic))
    assert not missing.overlays
    assert any("Session open unavailable" in message for message in missing.messages)
    support = bars([opening], [100.])
    found = logic.calculate(context(frame, support=support), configured(logic))
    assert level(found, "upper")[0]["x"] == [0, 1]
    assert level(found, "upper")[0]["y"] == [105., 105.]
    assert [row["source_index"] for row in found.tables[0]["rows"]] == [0, 1]


def test_regular_sessions_exclude_holidays_and_after_early_close():
    stamps = [datetime(2024, 7, 3, 13, 30, tzinfo=UTC), datetime(2024, 7, 3, 16, 59, tzinfo=UTC),
              datetime(2024, 7, 3, 17, 0, tzinfo=UTC), datetime(2024, 7, 4, 13, 30, tzinfo=UTC)]
    logic = OpenPriceBandsLogic()
    result = logic.calculate(context(bars(stamps)), configured(logic))
    assert level(result, "upper")[0]["x"] == [0, 1]
    assert len(level(result, "upper")) == 1
    assert any("2 selected bars are outside" in message for message in result.messages)


def test_extended_hours_use_four_am_open_and_early_late_close():
    stamps = [datetime(2024, 7, 3, 8, 0, tzinfo=UTC), datetime(2024, 7, 3, 20, 59, tzinfo=UTC),
              datetime(2024, 7, 3, 21, 0, tzinfo=UTC)]
    logic = OpenPriceBandsLogic()
    result = logic.calculate(context(bars(stamps, [100., 120., 130.])),
                             configured(logic, session_hours="extended"))
    assert level(result, "upper")[0]["x"] == [0, 1]
    assert level(result, "upper")[0]["y"] == [105., 105.]
    assert any("1 selected bars are outside" in message for message in result.messages)


def test_daily_session_bars_are_date_labels_and_singletons_are_visible():
    frame = bars([datetime(2024, 7, 3, tzinfo=UTC), datetime(2024, 7, 4, tzinfo=UTC),
                  datetime(2024, 7, 5, tzinfo=UTC)], [100., 110., 200.])
    logic = OpenPriceBandsLogic()
    raw = logic.calculate(context(frame, "1d"), configured(logic))
    result = presented(logic, raw, show_open=False)
    assert [item["x"] for item in level(result, "upper")] == [[0, 2]]
    assert [item["y"] for item in level(result, "upper")] == [[105., 210.]]
    assert all(item["type"] == "markers" for item in result.overlays)
    assert not level(result, "open")
    result.validate()


def test_daily_extended_reference_is_explicit_about_daily_open():
    frame = bars([datetime(2024, 7, 3, tzinfo=UTC)])
    logic = OpenPriceBandsLogic()
    result = logic.calculate(context(frame, "1d"), configured(logic, session_hours="extended"))
    assert any("cannot distinguish regular and extended" in message for message in result.messages)


def test_multiday_session_reference_is_rejected_but_first_bar_is_supported():
    frame = bars([datetime(2024, 1, 2, tzinfo=UTC)])
    logic = OpenPriceBandsLogic()
    with pytest.raises(ValueError, match="intraday or 1d"):
        logic.calculate(context(frame, "1wk"), configured(logic))
    result = logic.calculate(context(frame, "1wk"), configured(logic, reference="first_bar"))
    assert level(result, "upper")[0]["y"] == [105.]


@pytest.mark.parametrize("logic,values", [
    (OpenPriceBandsLogic(), {"upper_percent": -1}),
    (OpenPriceBandsLogic(), {"lower_percent": 101}),
    (OpenPriceBandsLogic(), {"lower_percent": float("inf")}),
    (OpenPriceBandsLogic(), {"upper_percent": True}),
    (OpenPriceBandsLogic(), {"reference": "unknown"}),
    (OpenPriceBandsLogic(), {"unknown": 1}),
    (TrendLogic(), {"weight_ols": 0.8}),
    (TrendLogic(), {"slope_mode": "range", "slope_min": 0.2, "slope_max": 0.1}),
    (TrendLogic(), {"selected_lambdas": [-1]}),
    (TrendLogic(), {"selected_lambdas": [1., 1.]}),
])
def test_schema_and_cross_field_validation(logic, values):
    with pytest.raises(ValueError):
        configured(logic, **values)


@pytest.fixture(scope="module")
def trend_input():
    stamps = [datetime(2024, 1, 2, 14, 30, tzinfo=UTC) + timedelta(minutes=i) for i in range(32)]
    frame = bars(stamps, [100 * math.exp(index * .002) for index in range(len(stamps))])
    config = TrendConfig(max_history=32, penalty_grid_levels=3)
    return frame, config, analyze(frame, config=config)


def test_trend_adapter_preserves_fit_coordinates_metrics_and_table_statuses(trend_input, monkeypatch):
    from chart_logics.plugins import trend as module
    frame, settings, analysis = trend_input
    monkeypatch.setattr(module, "analyze", lambda *args, **kwargs: analysis)
    logic = TrendLogic()
    raw = logic.calculate(context(frame), trend_configured(settings))
    assert len(raw.overlays) == len(analysis.trends)
    output = presented(logic, raw)
    selected = select_trends(analysis.trends)
    assert {item["id"] for item in output.overlays} == {item.trend_id for item in selected.chart_trends}
    by_id = {item.trend_id: item for item in analysis.trends}
    for overlay in output.overlays:
        original = by_id[overlay["id"]]
        assert overlay["type"] == "log_line"
        assert (overlay["start_index"], overlay["end_index"], overlay["slope"], overlay["intercept"]) == (
            original.start_index, original.end_index, original.slope, original.intercept)
    accepted = output.tables[0]["rows"]
    assert len(accepted) == len(analysis.trends)
    assert {row["status"] for row in accepted} == {"On chart", "Redundant"}
    for row in accepted:
        assert row["rate_per_bar"] == by_id[row["trend_id"]].rate_per_bar * 100
        assert isinstance(row["start"], str)
        assert row["redundant_to"] == selected.redundant_to.get(row["trend_id"])
    output.validate()


def test_trend_filters_hide_lines_without_dropping_archive_or_recalculating(trend_input, monkeypatch):
    from chart_logics.plugins import trend as module
    frame, settings, analysis = trend_input
    calculations = []

    def capture(*args, **kwargs):
        calculations.append(1)
        return analysis

    monkeypatch.setattr(module, "analyze", capture)
    logic = TrendLogic()
    raw = logic.calculate(context(frame), trend_configured(settings))
    archive = json.dumps(raw.to_dict(), sort_keys=True)
    # Display still works after serialized workspace/cache restoration.
    raw = LogicResult.from_dict(json.loads(archive))
    baseline = presented(logic, raw, show_redundant=True)
    assert len(baseline.overlays) == len(analysis.trends)
    hidden = presented(logic, raw, slope_mode="negative")
    assert not hidden.overlays
    assert len(hidden.tables[0]["rows"]) == len(analysis.trends)
    assert [row["redundant_to"] for row in hidden.tables[0]["rows"]] == [
        row["redundant_to"] for row in baseline.tables[0]["rows"]]
    assert not any(row["on_chart"] for row in hidden.tables[0]["rows"])
    no_lambda = presented(logic, raw, show_all_lambdas=False)
    assert not no_lambda.overlays
    assert any("Show all λ values" in message for message in no_lambda.messages)
    chosen = analysis.trends[0].lambda_values[0]
    exact = presented(logic, raw, show_redundant=True, show_all_lambdas=False,
                      selected_lambdas=[chosen])
    assert {item["id"] for item in exact.overlays} == {
        item.trend_id for item in analysis.trends if chosen in item.lambda_values}
    assert len(calculations) == 1
    assert json.dumps(raw.to_dict(), sort_keys=True) == archive


def test_trend_suggestions_cover_complete_archive_and_do_not_mutate_schema(trend_input):
    _, settings, analysis = trend_input
    logic = TrendLogic()
    raw = logic.from_analysis(analysis, trend_configured(settings))
    registry = discover_logics()
    schema_before = json.dumps(logic.display_schema, sort_keys=True)
    schema = registry.display_schema(logic.name, logic.version, raw)
    expected = sorted({value for trend in analysis.trends for value in trend.lambda_values})
    assert schema["properties"]["selected_lambdas"]["items"]["enum"] == expected
    assert schema["properties"]["selected_lambdas"]["x-ui-options"] == expected
    defaults = registry.display_defaults(logic.name, logic.version, raw)
    assert defaults["slope_min"] == min(trend.slope for trend in analysis.trends)
    assert defaults["slope_max"] == max(trend.slope for trend in analysis.trends)
    hidden = presented(logic, raw, show_all_lambdas=False, selected_lambdas=[])
    assert not hidden.overlays
    assert registry.display_schema(logic.name, logic.version, hidden) == schema
    assert json.dumps(logic.display_schema, sort_keys=True) == schema_before
    empty = logic.from_analysis(replace(analysis, trends=()), trend_configured(settings))
    assert registry.display_schema(logic.name, logic.version, empty)["properties"][
        "selected_lambdas"]["x-ui-options"] == []


@pytest.mark.parametrize("values", [
    {"slope_mode": "range", "slope_min": 0.2, "slope_max": 0.1},
    {"selected_lambdas": [-1]},
    {"selected_lambdas": [1., 1.]},
    {"redundancy_percentage": 101},
    {"redundancy_boundary_rule": "invalid"},
    {"redundancy_min_bars": -1},
])
def test_trend_display_validation(trend_input, values):
    _, settings, analysis = trend_input
    logic = TrendLogic()
    raw = logic.from_analysis(analysis, trend_configured(settings))
    with pytest.raises(ValueError):
        presented(logic, raw, **values)


def test_redundancy_is_display_only_and_can_restore_all_fits(trend_input, monkeypatch):
    from chart_logics.plugins import trend as module
    _, settings, analysis = trend_input
    logic = TrendLogic()
    raw = logic.from_analysis(analysis, trend_configured(settings))
    monkeypatch.setattr(module, "analyze", lambda *a, **k: pytest.fail("Display reran analysis"))
    exact = presented(logic, raw, redundancy_percentage=0, redundancy_min_bars=0)
    assert len(exact.overlays) == len(analysis.trends)
    reduced = presented(logic, raw, redundancy_percentage=100, redundancy_boundary_rule="or",
                        redundancy_min_bars=0)
    assert len(reduced.overlays) < len(exact.overlays)
    assert len(reduced.tables[0]["rows"]) == len(analysis.trends)
    assert {overlay["id"] for overlay in presented(logic, raw, show_redundant=True).overlays} == {
        trend.trend_id for trend in analysis.trends}


def test_trend_analysis_receives_only_completed_original_prefix(trend_input, monkeypatch):
    from chart_logics.plugins import trend as module
    frame, settings, analysis = trend_input
    captured = []

    def capture(frame, **kwargs):
        captured.append((frame, kwargs))
        return replace(analysis, trends=(), diagnostics=(), status="insufficient_data")

    monkeypatch.setattr(module, "analyze", capture)
    cutoff = frame["timestamp"][-1] + timedelta(seconds=30)
    calls = []
    callback = lambda done, total: calls.append((done, total))
    logic = TrendLogic()
    result = logic.calculate(context(frame, cutoff=cutoff), trend_configured(settings),
                             progress=callback)
    assert captured[0][0].equals(frame.head(frame.height - 1))
    assert captured[0][1]["progress"] is callback
    assert captured[0][1]["as_of"] == cutoff
    assert not result.overlays
    assert any("Insufficient completed bars" in message for message in result.messages)
    assert logic.input_policy == "completed"


def test_tail_derived_trend_remains_selectable_without_lambda(trend_input, monkeypatch):
    from chart_logics.plugins import trend as module
    frame, settings, analysis = trend_input
    tail = replace(analysis.trends[-1], trend_id="tail-example", lambda_values=(), lambda_levels=(),
                   origin="tail", direction=-1, slope=-.002)
    mixed = replace(analysis, trends=(*analysis.trends, tail))
    monkeypatch.setattr(module, "analyze", lambda *args, **kwargs: mixed)
    logic = TrendLogic()
    raw = logic.calculate(context(frame), trend_configured(settings))
    result = presented(logic, raw, show_redundant=True, show_all_lambdas=False,
                       include_without_lambda=True, slope_mode="negative")
    assert [item["id"] for item in result.overlays] == ["tail-example"]
    assert result.overlays[0]["line"]["dash"] == "dash"
    row = next(row for row in result.tables[0]["rows"] if row["trend_id"] == "tail-example")
    assert row["lambda_label"] == "Tail-derived; no λ"
    assert row["on_chart"]
    result.validate()


def test_saved_trend_adapter_reuses_verified_result_and_rejects_wrong_settings(trend_input, monkeypatch):
    from chart_logics.plugins import trend as module
    _, settings, analysis = trend_input
    monkeypatch.setattr(module, "analyze", lambda *args, **kwargs: pytest.fail("Saved result was recomputed"))
    logic = TrendLogic()
    adapted = logic.from_analysis(analysis, trend_configured(settings))
    assert adapted.overlays
    adapted.validate()
    with pytest.raises(ValueError, match="settings do not match"):
        logic.from_analysis(analysis, trend_configured(settings, clarity_threshold=0.9))


def test_many_session_bands_have_bounded_trace_count_and_flat_endpoints():
    stamps = []
    for week in range(20):
        opening = datetime(2024, 1, 2, 14, 30, tzinfo=UTC) + timedelta(weeks=week)
        # Use first_bar to exercise a long constant interval independently of DST.
        stamps.extend(opening + timedelta(minutes=minute) for minute in range(10))
    logic = OpenPriceBandsLogic()
    frame = bars(stamps)
    constant = logic.calculate(context(frame), configured(logic, reference="first_bar"))
    assert len(constant.overlays) == 3
    assert all(item["x"] == [0, frame.height - 1] for item in constant.overlays)
    assert len(constant.tables[0]["rows"]) == frame.height
    # Daily sessions use one marker series per level, not one trace per day.
    daily = bars([datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=day) for day in range(200)])
    result = logic.calculate(context(daily, "1d"), configured(logic))
    assert len(result.overlays) == 3
    assert all(item["type"] == "markers" for item in result.overlays)
    assert len(result.overlays[0]["x"]) > 100
    result.validate()


def test_saved_trend_adapter_ignores_previous_display_redundancy_settings(trend_input):
    _, settings, analysis = trend_input
    saved = replace(analysis, config=replace(settings, redundancy_percentage=90,
                                             redundancy_min_bars=25, redundancy_boundary_rule="or"))
    logic = TrendLogic()
    raw = logic.from_analysis(saved, trend_configured(settings))
    assert len(raw.overlays) == len(analysis.trends)
    assert len(raw.data["trends"]) == len(analysis.trends)


def test_open_band_display_toggles_reuse_all_calculated_levels(monkeypatch):
    frame = bars([datetime(2024, 1, 2, 15, index, tzinfo=UTC) for index in range(3)])
    logic = OpenPriceBandsLogic()
    raw = logic.calculate(context(frame), configured(logic, reference="first_bar"))
    archived = raw.to_dict()
    monkeypatch.setattr(OpenPriceBandsLogic, "calculate", lambda *a, **k: pytest.fail("Display recalculated levels"))
    upper_only = presented(logic, raw, show_open=False, show_lower=False)
    assert len(upper_only.overlays) == 1
    assert level(upper_only, "upper")
    none = presented(logic, raw, show_upper=False, show_lower=False, show_open=False)
    assert not none.overlays
    assert any("levels are hidden" in message for message in none.messages)
    restored = presented(logic, raw)
    assert restored.to_dict() == archived
    assert raw.to_dict() == archived


def test_builtin_calculation_and_display_schemas_are_separate():
    for logic in (TrendLogic(), OpenPriceBandsLogic()):
        assert not set(logic.config_schema["properties"]) & set(logic.display_schema["properties"])
        for name, field in logic.display_schema["properties"].items():
            with pytest.raises(ValueError):
                configured(logic, **{name: field["default"]})
        assert logic.result_version == "2"


def test_hidden_trend_instance_has_consistent_messages_and_table_flags(trend_input):
    from chart_logics import AppliedRun, LogicRunner, LogicSpec

    _, settings, analysis = trend_input
    logic = TrendLogic()
    spec = LogicSpec("hidden-trend", logic.name, logic.version, trend_configured(settings))
    raw = logic.from_analysis(analysis, spec.config)
    run = AppliedRun("source", [spec], {spec.instance_id: raw}, {}, {})
    displayed = LogicRunner(discover_logics()).present(run, visibility={spec.instance_id: False})
    result = displayed.results[spec.instance_id]
    assert not result.overlays
    assert not any(row["on_chart"] for row in result.tables[0]["rows"])
    assert not any(row["status"] == "On chart" for row in result.tables[0]["rows"])
    assert any("Hidden on chart" in message for message in result.messages)
    assert any("trend lines match the display filters" in message for message in result.messages)
    assert not any("intervals on chart" in message for message in result.messages)
    assert all(row["on_chart"] for row in raw.tables[0]["rows"])
