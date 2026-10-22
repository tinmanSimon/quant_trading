"""Batch summaries respect metric meaning, missing observations and table shape."""

import json

import pytest

from research.analysis.batch.aggregation import (
    MetricSpec, aggregate_metrics, extract_metrics, infer_metrics, merge_summaries, rank_metrics,
)


def table(name, rows, columns=None):
    return {"name": name, "columns": columns or list(rows[0]) if rows else columns or [], "rows": rows}


def outcomes(win, loss, inconclusive):
    return [table("Summary", [
        {"outcome": "win", "count": win},
        {"outcome": "loss", "count": loss},
        {"outcome": "inconclusive", "count": inconclusive},
        {"outcome": "total", "count": win + loss + inconclusive},
    ])]


def test_sma_rsi_outcomes_aggregate_total_independently_and_rank_without_losing_zero():
    specs = infer_metrics(outcomes(10, 5, 2))
    assert [spec.name for spec in specs] == ["win", "loss", "inconclusive", "total"]
    rows = {
        "AAPL": extract_metrics(outcomes(10, 5, 2), specs),
        "MSFT": extract_metrics(outcomes(8, 7, 1), specs),
        "EMPTY": extract_metrics(outcomes(0, 0, 0), specs),
        "MISSING": extract_metrics([], specs),
    }
    totals = aggregate_metrics(rows, specs)
    assert {name: item["value"] for name, item in totals.items()} == {
        "win": 18, "loss": 12, "inconclusive": 3, "total": 33,
    }
    assert all(item["contributors"] == 3 for item in totals.values())
    assert rank_metrics(rows, "win") == ["AAPL", "MSFT", "EMPTY", "MISSING"]
    assert rank_metrics(rows, "win", descending=False) == ["EMPTY", "MSFT", "AAPL", "MISSING"]


def test_non_count_single_row_metrics_are_rankable_without_guessing_aggregation():
    data = [table("Summary", [{"signals": 20, "average_duration": 8.4, "success_rate": .65,
                                "source_index": 99, "entry_price": 123., "is_valid": True}])]
    specs = infer_metrics(data)
    assert {spec.name: spec.aggregation for spec in specs} == {
        "signals": "sum", "average_duration": "none", "success_rate": "none",
    }
    values = extract_metrics(data, specs)
    totals = aggregate_metrics({"A": values}, specs)
    assert totals["signals"]["value"] == 20
    assert totals["average_duration"] == {"value": None, "contributors": 1, "aggregation": "none"}
    assert values["success_rate"] == .65


@pytest.mark.parametrize("rows", [
    [{"outcome": "win", "count": 2}, {"outcome": "win", "count": 3}],
    [{"outcome": "win", "count": "3"}],
    [{"outcome": "win", "count": True}],
    [{"outcome": "win", "count": -1}],
    [{"outcome": "win", "count": 1.5}],
    [{"outcome": None, "count": 1}],
])
def test_ambiguous_or_invalid_category_counts_are_not_inferred(rows):
    assert infer_metrics([table("Summary", rows)]) == []


def test_detail_records_and_duplicate_table_names_are_not_automatically_summarized():
    assert infer_metrics([table("Completed observations", [
        {"record_id": 1, "signal_index": 4, "entry_price": 101., "exit_price": 103., "duration": 2},
    ])]) == []
    assert infer_metrics([table("Prices", [{"open": 100, "high": 105, "close": 103}])]) == []
    assert infer_metrics(outcomes(1, 2, 3) + outcomes(4, 5, 6)) == []


def test_explicit_selection_handles_detail_tables_and_nulls_without_rewriting_analysis():
    data = [table("Findings", [
        {"group": "entry", "duration": 2, "label": "a"},
        {"group": "entry", "duration": None, "label": "b"},
        {"group": "entry", "duration": 10, "label": None},
        {"group": "exit", "duration": 100, "label": "c"},
    ])]
    specs = [
        MetricSpec("entry_duration", "Findings", "duration", reducer="mean", aggregation="mean",
                   category_column="group", category_value="entry"),
        MetricSpec("labels", "Findings", "label", reducer="count", aggregation="sum"),
        MetricSpec("longest", "Findings", "duration", reducer="max", aggregation="max"),
    ]
    assert extract_metrics(data, specs) == {"entry_duration": 6, "labels": 3, "longest": 100}
    assert aggregate_metrics({"A": extract_metrics(data, specs)}, specs)["entry_duration"]["value"] == 6


def test_empty_count_differs_from_absent_table_category_or_column():
    spec = MetricSpec("count", "Findings", "value", reducer="count", aggregation="sum")
    assert extract_metrics([table("Findings", [], ["value"])], [spec]) == {"count": 0}
    assert extract_metrics([], [spec]) == {"count": None}
    assert extract_metrics([table("Findings", [], ["different"])], [spec]) == {"count": None}
    category = MetricSpec("missing", "Findings", "value", reducer="count",
                          category_column="category", category_value="absent")
    assert extract_metrics([table("Findings", [{"category": "present", "value": 1}])], [category]) == {
        "missing": None,
    }


def test_first_last_use_table_order_and_keep_null_endpoint_instead_of_stale_indicator():
    data = [table("RSI", [{"value": None}, {"value": 49}, {"value": 51}, {"value": None}])]
    specs = [MetricSpec("first", "RSI", "value", reducer="first"),
             MetricSpec("latest", "RSI", "value", reducer="last")]
    assert extract_metrics(data, specs) == {"first": None, "latest": None}
    data[0]["rows"] = data[0]["rows"][1:3]
    assert extract_metrics(data, specs) == {"first": 49, "latest": 51}


def test_missing_null_and_zero_are_preserved_in_summaries():
    specs = [MetricSpec("count", "Summary", "count", aggregation="sum")]
    assert extract_metrics([table("Summary", [{"count": None}])], specs) == {"count": None}
    assert aggregate_metrics({"A": {"count": None}, "B": {}}, specs)["count"] == {
        "value": None, "contributors": 0, "aggregation": "sum",
    }
    assert aggregate_metrics({"A": {"count": 0}, "B": {}}, specs)["count"] == {
        "value": 0, "contributors": 1, "aggregation": "sum",
    }


def test_ratio_uses_sum_of_matching_counts_instead_of_mean_of_ticker_rates():
    specs = [MetricSpec("wins", "Summary", "wins", aggregation="sum"),
             MetricSpec("total", "Summary", "total", aggregation="sum"),
             MetricSpec("win_rate", numerator="wins", denominator="total", scale=100)]
    rows = {
        "small": extract_metrics([table("Summary", [{"wins": 1, "total": 1}])], specs),
        "large": extract_metrics([table("Summary", [{"wins": 1, "total": 9}])], specs),
        "partial": {"wins": 80, "total": None},
        "zero": extract_metrics([table("Summary", [{"wins": 0, "total": 0}])], specs),
    }
    assert rows["small"]["win_rate"] == 100
    assert rows["zero"]["win_rate"] is None
    total = aggregate_metrics(rows, specs)
    assert total["wins"]["value"] == 82
    assert total["win_rate"] == {"value": 20, "contributors": 3, "aggregation": "ratio"}


def test_ratio_with_no_denominator_or_no_contributors_has_no_value():
    specs = [MetricSpec("a", "Stats", "a"), MetricSpec("b", "Stats", "b"),
             MetricSpec("ratio", numerator="a", denominator="b")]
    assert aggregate_metrics({}, specs)["ratio"]["value"] is None
    assert aggregate_metrics({"A": {"a": 0, "b": 0}}, specs)["ratio"]["value"] is None


def test_rank_is_stable_for_ties_and_missing_values_stay_last_in_both_directions():
    rows = {"missing1": {}, "first": {"metric": 5}, "second": {"metric": 5},
            "zero": {"metric": 0}, "missing2": {"metric": None}}
    assert rank_metrics(rows, "metric") == ["first", "second", "zero", "missing1", "missing2"]
    assert rank_metrics(rows, "metric", descending=False) == ["zero", "first", "second", "missing1", "missing2"]


@pytest.mark.parametrize("value", [True, "10", float("inf"), float("nan")])
def test_non_numeric_or_nonfinite_values_do_not_become_false_totals(value):
    spec = MetricSpec("count", "Summary", "count", aggregation="sum")
    with pytest.raises(ValueError, match="finite numbers"):
        extract_metrics([table("Summary", [{"count": value}])], [spec])
    with pytest.raises(ValueError, match="finite numbers"):
        aggregate_metrics({"A": {"count": value}}, [spec])
    with pytest.raises(ValueError, match="finite numbers"):
        rank_metrics({"A": {"count": value}}, "count")


def test_duplicate_category_requires_explicit_reducer_and_duplicate_table_is_rejected():
    data = [table("Summary", [{"outcome": "win", "count": 2}, {"outcome": "win", "count": 3}])]
    spec = MetricSpec("wins", "Summary", "count", category_column="outcome", category_value="win")
    with pytest.raises(ValueError, match="exactly one"):
        extract_metrics(data, [spec])
    deliberate = MetricSpec.from_dict({**spec.to_dict(), "reducer": "sum"})
    assert extract_metrics(data, [deliberate]) == {"wins": 5}
    with pytest.raises(ValueError, match="ambiguous"):
        extract_metrics(data + data, [deliberate])


def test_metric_configuration_round_trips_as_plain_json_and_disambiguates_tables():
    data = [table("First summary", [{"signals": 2}]), table("Second summary", [{"signals": 4}])]
    specs = infer_metrics(data)
    assert [spec.name for spec in specs] == ["First summary: signals", "Second summary: signals"]
    saved = json.loads(json.dumps([spec.to_dict() for spec in specs]))
    assert [MetricSpec.from_dict(value) for value in saved] == specs
    assert extract_metrics(data, saved) == {"First summary: signals": 2, "Second summary: signals": 4}


def saved_summary(tables):
    specs = infer_metrics(tables)
    return {"metrics": [spec.to_dict() for spec in specs], "values": extract_metrics(tables, specs)}


def test_saved_ticker_summaries_union_sources_even_when_display_names_differ():
    first = table("First summary", [{"signals": 2}])
    second = table("Second summary", [{"signals": 4}])
    specs, values = merge_summaries({
        "A": saved_summary([first, second]),
        "B": saved_summary([first]),
        "C": saved_summary([second]),
    })
    assert [spec.name for spec in specs] == ["First summary: signals", "Second summary: signals"]
    assert values == {
        "A": {"First summary: signals": 2, "Second summary: signals": 4},
        "B": {"First summary: signals": 2, "Second summary: signals": None},
        "C": {"First summary: signals": None, "Second summary: signals": 4},
    }
    totals = aggregate_metrics(values, specs)
    assert totals["First summary: signals"]["value"] == 4
    assert totals["Second summary: signals"]["value"] == 8
    assert totals["First summary: signals"]["contributors"] == 2


def test_same_named_metrics_from_different_tables_never_merge_and_later_categories_are_included():
    specs, values = merge_summaries({
        "A": saved_summary([table("Counts", [{"count": 3}])]),
        "B": saved_summary([table("Other counts", [{"count": 8}])]),
    })
    assert len(specs) == 2
    assert values["A"] == {"Counts: count": 3, "Other counts: count": None}
    first, second = outcomes(1, 2, 3), outcomes(4, 5, 6)
    first[0]["rows"] = first[0]["rows"][:1]
    specs, values = merge_summaries({"A": saved_summary(first), "B": saved_summary(second)})
    assert [spec.name for spec in specs] == ["win", "loss", "inconclusive", "total"]
    assert values["A"] == {"win": 1, "loss": None, "inconclusive": None, "total": None}


def test_incompatible_saved_metric_settings_are_rejected_instead_of_misaggregated():
    summary = saved_summary(outcomes(1, 2, 3))
    changed = saved_summary(outcomes(1, 2, 3))
    changed["metrics"][0]["aggregation"] = "mean"
    with pytest.raises(ValueError, match="conflicting aggregation"):
        merge_summaries({"A": summary, "B": changed})


def test_same_name_and_undefined_or_nested_ratio_references_are_rejected():
    base = MetricSpec("a", "Summary", "count")
    with pytest.raises(ValueError, match="unique"):
        aggregate_metrics({}, [base, base])
    with pytest.raises(ValueError, match="source metrics"):
        aggregate_metrics({}, [base, MetricSpec("ratio", numerator="a", denominator="missing")])
    with pytest.raises(ValueError, match="source metrics"):
        aggregate_metrics({}, [base, MetricSpec("ratio", numerator="a", denominator="a"),
                               MetricSpec("nested", numerator="ratio", denominator="a")])


@pytest.mark.parametrize("fields", [
    {"name": ""}, {"name": "a", "table": "Summary"},
    {"name": "a", "table": "Summary", "column": "count", "aggregation": "guess"},
    {"name": "a", "numerator": "x"},
    {"name": "a", "numerator": "x", "denominator": "y", "table": "Summary"},
    {"name": "a", "table": "Summary", "column": "count", "scale": float("nan")},
])
def test_invalid_metric_configuration_is_rejected(fields):
    with pytest.raises(ValueError):
        MetricSpec.from_dict(fields)
