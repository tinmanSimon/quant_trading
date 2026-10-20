"""Percentage redundancy preserves records and greedily selects representatives."""

from itertools import permutations
import random
from types import SimpleNamespace

import pytest

from dashboard.trend_selection import filter_trends_by_lambda, filter_trends_by_slope, select_trends


def test_lambda_filter_matches_any_provenance_and_preserves_original_intervals():
    first = SimpleNamespace(lambda_values=(.1, .2))
    second = SimpleNamespace(lambda_values=(.3,))
    tail = SimpleNamespace(lambda_values=())
    archive = (first, second, tail)
    assert filter_trends_by_lambda(archive) == archive
    assert filter_trends_by_lambda(archive, {.2}) == (first,)
    assert filter_trends_by_lambda(archive, {.2, .3}) == (first, second)
    assert filter_trends_by_lambda(archive, set()) == ()
    assert filter_trends_by_lambda(archive, set(), include_without_lambda=True) == (tail,)
    assert filter_trends_by_lambda(archive, {.1}, include_without_lambda=True) == (first, tail)
    assert filter_trends_by_lambda(archive, {.2})[0] is first


def trend(start, end, clarity=1.0, direction=1, *, trend_id=None, lambda_levels=(0,)):
    return SimpleNamespace(start_index=start, end_index=end, duration=end-start+1,
                           clarity=clarity, direction=direction,
                           trend_id=trend_id or f"{start}-{end}-{direction}", lambda_levels=lambda_levels)


def test_default_chronological_selection_uses_five_percent_and_both_boundaries():
    clear = trend(10, 109, .95)
    redundant = trend(15, 114, .8)
    different_start = trend(16, 108, .7)
    different_end = trend(9, 115, .7)
    archive = (redundant, different_start, clear, different_end)
    original_values = [vars(item).copy() for item in archive]
    selected = select_trends(archive)
    assert selected.chart_trends == (different_end, clear, different_start)
    assert selected.redundant_to == {redundant.trend_id: clear.trend_id}
    assert all(any(item is original for original in archive) for item in selected.chart_trends)
    assert [vars(item) for item in archive] == original_values


@pytest.mark.parametrize("rule, shorter_bounds", [("and", (20, 109)), ("or", (95, 114))])
def test_tolerance_uses_shorter_interval_even_when_longer_was_picked(rule, shorter_bounds):
    broad = trend(0, 99)
    shorter = trend(*shorter_bounds, .8)
    selected = select_trends((broad, shorter), redundancy_percentage=10,
                             redundancy_boundary_rule=rule, redundancy_min_bars=0)
    assert selected.chart_trends == (broad, shorter)
    assert selected.redundant_to == {}


def test_tolerance_uses_shorter_interval_when_shorter_was_picked():
    shorter = trend(10, 89)
    broad = trend(0, 99, .8)
    selected = select_trends((shorter, broad), redundancy_percentage=10)
    assert selected.chart_trends == (broad, shorter)
    assert selected.redundant_to == {}


@pytest.mark.parametrize("end, percentage, distance, redundant", [
    (19, 10, 2, True),  # Inclusive length 20 produces an exact 2-bar boundary.
    (19, 10, 3, False),
    (20, 9, 2, False),  # Resolution 1.89 must not round up to 2.
    (20, 10, 2, True),  # Resolution 2.1 admits 2 bars, but not 3.
    (20, 10, 3, False),
    (99, 58, 58, True),  # Computing 58 / 100 first would undershoot this boundary.
    (19, 100, 20, True),
])
def test_fractional_resolution_is_unrounded_and_boundary_inclusive(end, percentage, distance, redundant):
    first, second = trend(0, end), trend(distance, end + distance, .8)
    selected = select_trends((first, second), redundancy_percentage=percentage, redundancy_min_bars=0)
    assert (second.trend_id in selected.redundant_to) is redundant


@pytest.mark.parametrize("nearby", [(1, 100), (60, 149)])
def test_either_rule_matches_one_close_boundary_where_both_rule_does_not(nearby):
    first, second = trend(0, 149), trend(*nearby, .8)
    assert select_trends((first, second)).chart_trends == (first, second)
    selected = select_trends((first, second), redundancy_boundary_rule="or")
    winner, redundant = (second, first) if second.end_index < first.end_index else (first, second)
    assert selected.chart_trends == (winner,)
    assert selected.redundant_to == {redundant.trend_id: winner.trend_id}


@pytest.mark.parametrize("rule", ["and", "or"])
def test_opposite_directions_never_suppress_each_other(rule):
    up, down = trend(10, 100), trend(10, 100, .8, -1)
    selected = select_trends((up, down), redundancy_percentage=100, redundancy_boundary_rule=rule)
    assert {item.trend_id for item in selected.chart_trends} == {up.trend_id, down.trend_id}
    assert selected.redundant_to == {}


def test_short_opposite_trend_and_distinct_nested_or_crossing_intervals_remain_visible():
    broad = trend(0, 100)
    nested = trend(20, 60, .8)
    countertrend = trend(21, 61, .75, -1)
    crossing = trend(70, 140, .9)
    nearby_opposite = trend(1, 101, .7, -1)
    selected = select_trends((crossing, countertrend, broad, nested, nearby_opposite))
    assert selected.chart_trends == (broad, nearby_opposite, nested, countertrend, crossing)
    assert selected.redundant_to == {}


@pytest.mark.parametrize("rule", ["and", "or"])
def test_similarity_never_chains_through_a_redundant_interval(rule):
    first, middle, last = trend(0, 99, .99), trend(5, 104, .95), trend(10, 109, .9)
    selected = select_trends((first, middle, last), redundancy_boundary_rule=rule)
    assert selected.chart_trends == (first, last)
    assert selected.redundant_to == {middle.trend_id: first.trend_id}


def test_different_lambda_levels_do_not_prevent_nearby_redundancy():
    clear = trend(0, 20, .95, lambda_levels=(0, 2))
    lower = trend(1, 21, .8, lambda_levels=(6,))
    selected = select_trends((lower, clear))
    assert selected.chart_trends == (clear,)
    assert selected.redundant_to == {lower.trend_id: clear.trend_id}


@pytest.mark.parametrize("rule", ["and", "or"])
def test_zero_percentage_matches_exact_boundaries_using_selected_rule(rule):
    winner = trend(0, 20, .9, trend_id="winner")
    duplicate = trend(0, 20, .8, trend_id="duplicate")
    same_start = trend(0, 21, .7)
    same_end = trend(1, 20, .65)
    opposite = trend(0, 20, .6, -1)
    archive = (winner, duplicate, same_start, same_end, opposite)
    selected = select_trends(archive, redundancy_percentage=0, redundancy_boundary_rule=rule,
                             redundancy_min_bars=0)
    redundant = (duplicate,) if rule == "and" else (duplicate, same_start, same_end)
    assert selected.redundant_to == {item.trend_id: winner.trend_id for item in redundant}
    assert {item.trend_id for item in selected.chart_trends} == {
        item.trend_id for item in archive if item not in redundant
    }


@pytest.mark.parametrize("rule", ["and", "or"])
def test_earliest_endpoint_wins_over_clarity_duration_and_onset(rule):
    first = trend(10, 29, .7)
    later = trend(9, 30, .99)
    for archive in ((first, later), (later, first)):
        selected = select_trends(archive, redundancy_boundary_rule=rule)
        assert selected.chart_trends == (first,)
        assert selected.redundant_to == {later.trend_id: first.trend_id}


def test_same_endpoint_ties_prefer_clarity_then_duration_then_id_regardless_of_input_order():
    less_clear = trend(0, 20, .8, trend_id="less_clear")
    shorter = trend(2, 20, .9, trend_id="short")
    clear_z = trend(1, 20, .9, trend_id="Z")
    clear_a = trend(1, 20, .9, trend_id="A")
    archive = (shorter, less_clear, clear_z, clear_a)
    selected = select_trends(archive)
    for ordering in permutations(archive):
        assert select_trends(ordering) == selected
    assert selected.chart_trends == (clear_a,)
    assert selected.redundant_to == {item.trend_id: "A" for item in archive if item is not clear_a}


@pytest.mark.parametrize("rule, second_end, last_end", [("and", 100, 104), ("or", 107, 110)])
def test_if_close_to_two_representatives_choose_the_earliest_one(rule, second_end, last_end):
    first = trend(0, 99, .7)
    clearer = trend(8, second_end, .9)
    last = trend(4, last_end, .99)
    selected = select_trends((clearer, last, first), redundancy_boundary_rule=rule)
    assert selected.chart_trends == (first, clearer)
    assert selected.redundant_to == {last.trend_id: first.trend_id}


@pytest.mark.parametrize("rule", ["and", "or"])
@pytest.mark.parametrize("percentage, minimum", [(0, 0), (0, 5), (5, 5), (100, 0)])
def test_appending_future_endpoints_preserves_every_earlier_selection(rule, percentage, minimum):
    rng = random.Random(147)
    archive = tuple(trend(rng.randrange(end - 6), end, rng.choice((.7, .8, .99)),
                          rng.choice((-1, 1)), trend_id=f"T{index}")
                    for index, end in enumerate(rng.randrange(10, 150) for _ in range(200)))
    settings = dict(redundancy_boundary_rule=rule, redundancy_percentage=percentage,
                    redundancy_min_bars=minimum)
    full = select_trends(archive, **settings)
    by_id = {item.trend_id: item for item in archive}
    for cutoff in (30, 60, 100, 140):
        prefix = tuple(item for item in archive if item.end_index <= cutoff)
        selected = select_trends(prefix, **settings)
        assert selected.chart_trends == tuple(item for item in full.chart_trends if item.end_index <= cutoff)
        assert selected.redundant_to == {key: value for key, value in full.redundant_to.items()
                                         if by_id[key].end_index <= cutoff}


def test_long_history_both_rule_keeps_distinct_updates_and_either_rule_collapses_shared_start():
    archive = tuple(trend(0, end, .9) for end in range(7, 100))
    selected = select_trends(archive)
    assert len(selected.chart_trends) > 1
    assert len(selected.chart_trends) + len(selected.redundant_to) == len(archive)
    by_id = {item.trend_id: item for item in archive}
    for redundant_id, representative_id in selected.redundant_to.items():
        redundant, representative = by_id[redundant_id], by_id[representative_id]
        assert abs(redundant.end_index - representative.end_index) <= max(
            5, .05 * min(redundant.duration, representative.duration))
    assert select_trends(archive, redundancy_boundary_rule="or").chart_trends == (archive[0],)


def test_custom_percentage_and_empty_archive():
    first, second = trend(10, 109), trend(16, 115, .8)
    assert len(select_trends((first, second)).chart_trends) == 2
    assert select_trends((first, second), redundancy_percentage=6).chart_trends == (first,)
    assert select_trends(()).chart_trends == ()
    assert select_trends(()).redundant_to == {}


@pytest.mark.parametrize("value", [-1, 100.01, True, False, "5", None, float("nan"), float("inf"), -float("inf")])
def test_invalid_percentage_rejected(value):
    with pytest.raises(ValueError, match="finite number between 0 and 100"):
        select_trends((), redundancy_percentage=value)


@pytest.mark.parametrize("value", ["AND", "OR", "xor", "", None, True, 0])
def test_invalid_boundary_rule_rejected(value):
    with pytest.raises(ValueError, match="redundancy_boundary_rule"):
        select_trends((), redundancy_boundary_rule=value)


@pytest.mark.parametrize("percentage", [0, .1, 5, 17.3, 100])
@pytest.mark.parametrize("rule", ["and", "or"])
@pytest.mark.parametrize("minimum", [0, 5, 12])
def test_vectorized_selection_matches_independent_scalar_greedy_oracle(percentage, rule, minimum):
    rng = random.Random(681)
    archive = []
    for index in range(350):
        start = rng.randrange(600)
        archive.append(trend(start, start + rng.randrange(7, 500), rng.choice((.6, .8, .9)),
                             rng.choice((-1, 1)), trend_id=f"T{index}"))
    expected_retained, expected_redundant = [], {}
    ranked = sorted(archive, key=lambda item: (item.end_index, -item.clarity, -item.duration, item.trend_id))
    for item in ranked:
        for representative in expected_retained:
            if item.direction != representative.direction:
                continue
            scaled_resolution = max(100 * minimum, percentage * min(
                item.end_index - item.start_index + 1,
                representative.end_index - representative.start_index + 1,
            ))
            checks = (100 * abs(item.start_index - representative.start_index) <= scaled_resolution,
                      100 * abs(item.end_index - representative.end_index) <= scaled_resolution)
            if (all(checks) if rule == "and" else any(checks)):
                expected_redundant[item.trend_id] = representative.trend_id
                break
        else:
            expected_retained.append(item)
    selected = select_trends(archive, redundancy_percentage=percentage,
                             redundancy_boundary_rule=rule, redundancy_min_bars=minimum)
    assert selected.redundant_to == expected_redundant
    assert selected.chart_trends == tuple(sorted(expected_retained, key=lambda item: (
        item.start_index, item.end_index, item.trend_id,
    )))


@pytest.mark.parametrize("rule", ["and", "or"])
def test_default_minimum_is_five_actual_bars_in_scaled_comparison(rule):
    clear = trend(10, 19, .99)
    at_limit = trend(15, 24, .9)
    outside = trend(16, 25, .8)
    selected = select_trends((clear, at_limit, outside), redundancy_percentage=0,
                             redundancy_boundary_rule=rule)
    assert selected.chart_trends == (clear, outside)
    assert selected.redundant_to == {at_limit.trend_id: clear.trend_id}


@pytest.mark.parametrize("rule", ["and", "or"])
def test_custom_minimum_and_larger_percentage_both_apply(rule):
    first, second = trend(0, 99), trend(7, 106, .8)
    assert len(select_trends((first, second), redundancy_boundary_rule=rule).chart_trends) == 2
    assert select_trends((first, second), redundancy_min_bars=7,
                         redundancy_boundary_rule=rule).chart_trends == (first,)
    assert select_trends((first, second), redundancy_percentage=8, redundancy_min_bars=2,
                         redundancy_boundary_rule=rule).chart_trends == (first,)


@pytest.mark.parametrize("value", [-1, 1.5, True, float("inf"), "5", None])
def test_invalid_minimum_rejected(value):
    with pytest.raises(ValueError, match="redundancy_min_bars"):
        select_trends((), redundancy_min_bars=value)


@pytest.mark.parametrize("mode, bounds, expected", [
    ("all", {}, [-.002, -.001, 0., .001, .002]),
    ("positive", {}, [.001, .002]),
    ("negative", {}, [-.002, -.001]),
    ("range", {"minimum": -.001, "maximum": .001}, [-.001, 0., .001]),
    ("range", {"minimum": .001, "maximum": .002}, [.001, .002]),
    ("range", {"minimum": -.002, "maximum": -.001}, [-.002, -.001]),
    ("range", {"minimum": 0., "maximum": 0.}, [0.]),
    ("range", {"minimum": .003, "maximum": .004}, []),
])
def test_slope_filter_signed_ranges_and_original_records(mode, bounds, expected):
    archive = tuple(SimpleNamespace(slope=value) for value in [-.002, -.001, 0., .001, .002])
    result = filter_trends_by_slope(iter(archive), mode, **bounds)
    assert [item.slope for item in result] == expected
    assert all(any(item is original for original in archive) for item in result)
    assert filter_trends_by_slope((), mode, **bounds) == ()


@pytest.mark.parametrize("minimum, maximum", [
    (1., -1.), (None, 1.), (0., None), (float("nan"), 1.),
    (0., float("inf")), (float("-inf"), 0.), (True, 1.), (0., "1"),
])
def test_slope_filter_rejects_invalid_bounds_even_for_empty_results(minimum, maximum):
    with pytest.raises(ValueError, match="slope|Slope"):
        filter_trends_by_slope((), "range", minimum=minimum, maximum=maximum)


def test_slope_filter_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        filter_trends_by_slope((), "unknown")


def test_slope_filter_combines_with_lambda_and_tail_visibility():
    up = SimpleNamespace(slope=.001, lambda_values=(.1, .2))
    down = SimpleNamespace(slope=-.001, lambda_values=(.2,))
    tail = SimpleNamespace(slope=-.002, lambda_values=())
    archive = (up, down, tail)
    eligible = filter_trends_by_lambda(archive, {.2}, include_without_lambda=True)
    assert filter_trends_by_slope(eligible, "negative") == (down, tail)
    assert filter_trends_by_slope(filter_trends_by_lambda(archive, {.1}), "negative") == ()
