"""Trend controls, bounded session caching, and explanatory result tables.

The caller supplies verified source bars, already restricted to completed bars.
Chart windows and display grouping must never be passed as analysis inputs.
"""

from collections import OrderedDict
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

import polars as pl
import streamlit as st

from trend_analysis import ALGORITHM_VERSION, TrendConfig, analyze, penalty_grid
from dashboard.trend_selection import TrendSelection, select_trends
from dashboard.trend_settings import load_trend_settings, save_trend_settings


_STATE_KEY = "_trend_controls_state"
_WIDGET_PREFIX = "_trend_setting_"
_CACHE_LIMIT = 4

# (field, label, help, minimum, maximum, step). Defaults come only from TrendConfig.
_DISCOVERY = (
    ("max_history", "Maximum lookback per endpoint (bars)", "At each historical endpoint, use at most this many preceding source bars, including that endpoint. Every eligible endpoint in the selected interval is analyzed. This does not limit total coverage; the prototype supports lookbacks up to 2,000 bars.", 8, 2000, 100),
    ("min_segment_length", "Minimum segment length (bars)", "Shorter candidate segments are excluded.", 8, 2000, 1),
    ("penalty_multiplier", "Starting λ", "First numerical penalty in the fixed grid shared by every endpoint. Stored as penalty_multiplier for compatibility. Must be positive.", 0.0, None, 0.1),
    ("penalty_grid_step", "Penalty grid step", "Each next λ is multiplied by 2 raised to this value. A step of 3 multiplies by 8. The same grid is used at every endpoint.", 0.25, 4.0, 0.25),
    ("penalty_grid_levels", "Number of λ levels", "Number of fixed penalties tested at every endpoint. More levels extend the grid toward larger penalties; prices and history length never change this grid.", 1, 256, 1),
)
_ACCEPTANCE = (
    ("clarity_threshold", "Minimum clarity", "Required combined directional evidence; this is not a probability.", 0.0, 1.0, 0.05),
    ("magnitude_threshold", "Minimum normalized slope", "Reject movements that are too small relative to return volatility. Must be positive.", 0.0, None, 0.1),
)
_WEIGHTS = (
    ("weight_ols", "Regression weight", "Contribution from ordinary regression and fit quality. The four weights must sum to 1.", 0.0, 1.0, 0.05),
    ("weight_robust", "Robust slope weight", "Contribution from the Theil–Sen slope.", 0.0, 1.0, 0.05),
    ("weight_ordering", "Chronological ordering weight", "Contribution from whether later prices are generally higher or lower.", 0.0, 1.0, 0.05),
    ("weight_efficiency", "Directional efficiency weight", "Contribution from net progress relative to backtracking.", 0.0, 1.0, 0.05),
)
_BOUNDARIES = (
    ("boundary_fraction", "Boundary perturbation fraction", "Shift each proposed start around its estimate by this fraction of the interval length.", 0.0, 1.0, 0.01),
    ("boundary_score_threshold", "Boundary evidence threshold", "Minimum clarity required at a perturbed boundary.", 0.0, 1.0, 0.05),
    ("boundary_acceptance", "Required boundary robustness", "Required fraction of perturbed starts supporting the same direction.", 0.0, 1.0, 0.05),
)
_TAILS = (
    ("min_tail_length", "Minimum tail length (bars)", "Minimum observations needed to assess a fitted recent segment.", 8, 2000, 1),
    ("max_tail_fraction", "Maximum tail fraction", "Limit the recent portion examined relative to its parent interval. Must be greater than zero.", 0.0, 0.5, 0.05),
    ("tail_fit_threshold", "Tail fit-improvement threshold", "Required proportional error reduction from splitting prefix and tail fits.", 0.0, 1.0, 0.05),
    ("tail_slope_threshold", "Tail slope-change threshold", "Required volatility-normalized change in slope. Must be positive.", 0.0, None, 0.1),
    ("tail_level_threshold", "Tail level-change threshold", "Required volatility-normalized level jump between fitted pieces. Must be positive.", 0.0, None, 0.25),
    ("slowing_ratio", "Slowing ratio", "A sufficiently small tail slope relative to the broader slope indicates slowing.", 0.0, 1.0, 0.05),
    ("retracement_threshold", "Material retracement threshold", "Fraction of the broader movement retraced before it is materially challenged.", 0.0, 1.0, 0.05),
    ("countertrend_fraction", "Material countertrend duration fraction", "Relative local-trend duration used to flag a material challenge.", 0.0, 1.0, 0.05),
    ("shock_threshold", "Young counter-move threshold", "Volatility-normalized threshold for a very recent move with too few bars for a trend. Must be positive.", 0.0, None, 0.25),
)
_DISPLAY = (
    ("redundancy_min_bars", "Minimum redundancy resolution (bars)", "The boundary tolerance is the larger of this minimum and the configured percentage of the shorter interval. Applies to both AND and OR. Set to zero to use only the percentage.", 0, None, 1),
    ("redundancy_percentage", "Redundancy tolerance (%)", "For each pair, calculate this percentage of the shorter interval's length, including both endpoint bars, then use the larger of that value and the minimum resolution in bars. Boundary differences are compared directly without rounding. Intervals are considered by earliest ending bar, with higher clarity, longer duration, then interval ID breaking same-bar ties; an interval matching a selected same-direction representative stays in the table as redundant.", 0.0, 100.0, 0.5),
)
_ALL_FIELDS = _DISCOVERY + _ACCEPTANCE + _WEIGHTS + _BOUNDARIES + _TAILS + _DISPLAY
_BOUNDARY_RULE_LABELS = {"and": "Both boundaries (AND)", "or": "Either boundary (OR)"}
_BOUNDARY_RULE_KEY = f"{_WIDGET_PREFIX}redundancy_boundary_rule"


def _fingerprint(frame: pl.DataFrame) -> str:
    """Include source contents, order, and types, not just timestamp endpoints."""
    digest = sha256(repr(tuple(frame.schema.items())).encode())
    digest.update(str(frame.height).encode())
    digest.update(frame.hash_rows(seed=41, seed_1=73, seed_2=109, seed_3=151).to_numpy().tobytes())
    return digest.hexdigest()


def _result_key(frame: pl.DataFrame, *, timeframe: str, identity: object,
                as_of: datetime | None, config: TrendConfig) -> str:
    values = {"data": _fingerprint(frame), "timeframe": timeframe,
              "identity": repr(identity), "as_of": as_of.isoformat() if as_of else None,
              "config": config.to_dict(), "algorithm_version": ALGORITHM_VERSION}
    return sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _set_form_values(config: TrendConfig) -> None:
    values = config.to_dict()
    for field, _, _, minimum, _, _ in _ALL_FIELDS:
        st.session_state[f"{_WIDGET_PREFIX}{field}"] = type(minimum)(values[field])
    st.session_state[_BOUNDARY_RULE_KEY] = values["redundancy_boundary_rule"]


def _controls(fields, values: dict, memory=None) -> None:
    columns = st.columns(2)
    for index, (field, label, help_text, minimum, maximum, step) in enumerate(fields):
        key = f"{_WIDGET_PREFIX}{field}"
        render = columns[index % 2].number_input
        kwargs = dict(min_value=minimum, max_value=maximum, step=step, key=key, help=help_text)
        values[field] = (memory.widget(render, label, value=type(minimum)(values[field]), **kwargs)
                         if memory else render(label, **kwargs))


def _analyze_history(frame: pl.DataFrame, *, as_of: datetime | None, config: TrendConfig):
    """Show endpoint progress without flooding Streamlit with per-bar updates."""
    indicator = st.progress(0.0, text="Preparing historical trend analysis…")
    last_update = 0.0

    def update(done: int, total: int) -> None:
        nonlocal last_update
        now = monotonic()
        if done == 0 or done == total or now - last_update >= 0.25:
            fraction = min(1.0, max(0.0, done / total)) if total else 0.0
            indicator.progress(fraction, text=f"Analyzed {done:,} of {total:,} historical endpoints")
            last_update = now

    try:
        return analyze(frame, as_of=as_of, config=config, progress=update)
    finally:
        indicator.empty()


def render_trend_controls(frame: pl.DataFrame, *, timeframe: str, identity: object,
                          as_of: datetime | None = None,
                          settings_path: Path | str | None = None, memory=None):
    """Return only a completed result matching these bars and applied settings.

    A new source selection requires an explicit Apply. Ordinary chart reruns
    reuse the current result; computation never follows chart viewport changes.
    """
    path_identity = str(Path(settings_path).resolve()) if settings_path is not None else "default"
    if memory:
        path_identity += ":" + memory.scope
    state = st.session_state.get(_STATE_KEY)
    if (state is None or state["path"] != path_identity
            or state.get("algorithm_version") != ALGORITHM_VERSION):
        loaded = load_trend_settings(settings_path)
        config = loaded.config
        if memory and "applied_config" in memory.values:
            try:
                config = TrendConfig.from_dict(memory.values["applied_config"])
            except (TypeError, ValueError):
                st.warning("Saved workspace trend settings are invalid; using the saved trend defaults.")
        state = {"path": path_identity, "applied": config, "algorithm_version": ALGORITHM_VERSION,
                 "warning": loaded.warning, "cache": OrderedDict(), "result_key": None}
        st.session_state[_STATE_KEY] = state
        _set_form_values(config)

    enabled = (memory.widget(st.checkbox, "Analyze trends", key="analyze-trends") if memory else
               st.checkbox("Analyze trends", key="analyze-trends"))
    if not enabled:
        if memory is None:
            _set_form_values(state["applied"])
        return None

    # After a page switch Streamlit can restore an old, pre-widget value under
    # a user key. The action button disappearing tells us the form was removed;
    # reseed the whole form from the separately retained applied configuration.
    if memory is None and "_trend_restore_defaults" not in st.session_state:
        _set_form_values(state["applied"])

    if state["warning"]:
        st.warning(state["warning"])
    st.caption("Analyze every eligible historical endpoint across the selected interval using original, completed bars. "
               "Each endpoint uses only bars available through that time, with the configured lookback. "
               "Long intervals take longer to analyze. Zooming and panning only change the view. "
               "Chart and table results use the last applied settings; form edits take effect when you apply them.")

    if st.button("Restore trend defaults", key="_trend_restore_defaults"):
        defaults = TrendConfig()
        try:
            save_trend_settings(defaults, settings_path)
        except (OSError, ValueError) as exc:
            st.error(f"Could not save default trend settings: {exc}")
        else:
            state.update(applied=defaults, warning=None, result_key=None)
            _set_form_values(defaults)
            if memory:
                memory.values.update(applied_config=defaults.to_dict(), analysis_key=None)
                memory.values["widgets"].update({f"{_WIDGET_PREFIX}{name}": value
                                                 for name, value in defaults.to_dict().items()})
            st.success("Default trend settings restored and saved. Apply and analyze to update the results.")

    # Streamlit removes widget state when its checkbox/containing page is hidden.
    # Keep applied values separately, then recreate widgets from those values.
    applied_values = state["applied"].to_dict()
    for field, _, _, minimum, _, _ in _ALL_FIELDS:
        key = f"{_WIDGET_PREFIX}{field}"
        if key not in st.session_state:
            st.session_state[key] = type(minimum)(applied_values[field])
    if _BOUNDARY_RULE_KEY not in st.session_state:
        st.session_state[_BOUNDARY_RULE_KEY] = applied_values["redundancy_boundary_rule"]

    # Preserve retired fields in older saved configurations while allowing the
    # current form to update all active settings.
    values = dict(applied_values)
    with (st.container() if memory else st.form("trend_configuration")):
        st.markdown("**Interval discovery**")
        _controls(_DISCOVERY, values, memory)
        st.markdown("**Acceptance**")
        _controls(_ACCEPTANCE, values, memory)
        st.markdown("**Chart display**")
        _controls(_DISPLAY, values, memory)
        rule_options = dict(options=tuple(_BOUNDARY_RULE_LABELS),
                            format_func=_BOUNDARY_RULE_LABELS.get, key=_BOUNDARY_RULE_KEY,
                            help="Both boundaries (AND) requires the start and end differences to fall within tolerance. "
                                 "Either boundary (OR) requires only one. OR can hide nested trends sharing a start or end, "
                                 "even at 0%. Only intervals with the same direction can be redundant.")
        values["redundancy_boundary_rule"] = (
            memory.widget(st.selectbox, "Boundary matching rule",
                          index=tuple(_BOUNDARY_RULE_LABELS).index(applied_values["redundancy_boundary_rule"]),
                          **rule_options) if memory else st.selectbox("Boundary matching rule", **rule_options))
        with (memory.widget(st.expander, "Advanced trend settings", key="trend-advanced-open") if memory else
              st.expander("Advanced trend settings")):
            for title, fields in (("Evidence weights", _WEIGHTS),
                                  ("Boundary robustness", _BOUNDARIES),
                                  ("Tail diagnosis", _TAILS)):
                st.markdown(f"**{title}**")
                _controls(fields, values, memory)
        submitted = (st.button("Apply and analyze", type="primary") if memory else
                     st.form_submit_button("Apply and analyze", type="primary"))

    with (memory.widget(st.expander, "Fixed λ grid preview", key="trend-grid-open") if memory else
          st.expander("Fixed λ grid preview")):
        try:
            preview = penalty_grid(TrendConfig.from_dict(values))
        except (TypeError, ValueError):
            st.caption("Enter valid settings and apply them to preview the fixed λ grid.")
        else:
            st.caption("Starting λ × 2^(level × grid step), with level starting at 0. "
                       "The form values below take effect when applied; every endpoint uses the same grid.")
            st.code(", ".join(f"{value:.12g}" for value in preview), language=None)

    if submitted:
        try:
            config = TrendConfig.from_dict(values)
        except (TypeError, ValueError) as exc:
            st.error(f"Invalid trend settings: {exc}")
        else:
            try:
                save_trend_settings(config, settings_path)
            except (OSError, ValueError) as exc:
                st.error(f"Could not save trend settings; analysis was not started: {exc}")
            else:
                # Saving valid settings succeeds independently of analysis. Never
                # leave a result from the previous settings visible on failure.
                state.update(applied=config, warning=None, result_key=None)
                if memory:
                    memory.values.update(applied_config=config.to_dict(), analysis_key=None)
                try:
                    key = _result_key(frame, timeframe=timeframe, identity=identity,
                                      as_of=as_of, config=config)
                    if key not in state["cache"]:
                        state["cache"][key] = _analyze_history(frame, as_of=as_of, config=config)
                    state["cache"].move_to_end(key)
                    while len(state["cache"]) > _CACHE_LIMIT:
                        state["cache"].popitem(last=False)
                    state["result_key"] = key
                    if memory:
                        try:
                            memory.save_analysis(key, state["cache"][key])
                        except (OSError, TypeError, ValueError) as exc:
                            st.warning(f"Analysis is available in this session but could not be saved for reopening: {exc}")
                except Exception as exc:
                    # Keep market data and chart navigation usable when an
                    # optional analysis fails, including unexpected engine errors.
                    st.error(f"Trend analysis failed: {exc}. Your valid settings were saved.")

    current_key = _result_key(frame, timeframe=timeframe, identity=identity,
                              as_of=as_of, config=state["applied"])
    if memory and current_key != state["result_key"] and memory.values.get("analysis_key") == current_key:
        restored = state["cache"].get(current_key)
        if restored is None:
            restored = memory.load_analysis(current_key, state["applied"])
        if restored is not None:
            state["cache"][current_key] = restored
            state["cache"].move_to_end(current_key)
            while len(state["cache"]) > _CACHE_LIMIT:
                state["cache"].popitem(last=False)
            state["result_key"] = current_key
    if current_key != state["result_key"]:
        state["result_key"] = None
        st.info("Apply and analyze to calculate trends for the current data and settings.")
        return None
    return state["cache"].get(current_key)


def _with_lambda_labels(table: pl.DataFrame) -> pl.DataFrame:
    """Show all discovery penalties without disguising tail-derived intervals."""
    if "lambda_values" not in table.columns:
        return table
    labels = []
    for row in table.select("lambda_values", "lambda_levels", "origin").iter_rows(named=True):
        labels.append("; ".join(
            f"λ={value:.6g} (level {level})"
            for value, level in zip(row["lambda_values"], row["lambda_levels"])
        ) or ("Tail-derived; no λ" if row["origin"] == "tail" else "No λ recorded"))
    return table.with_columns(pl.Series("lambda_label", labels, dtype=pl.String))


def render_trend_results(result, *, selection: TrendSelection | None = None,
                         show_redundant: bool = False, chart_trend_ids: set[str] | None = None, memory=None) -> None:
    """Distinguish chart intervals, retained redundancies, and uncertain candidates."""
    if result is None:
        return
    if selection is None:
        selection = select_trends(result.trends,
                                  redundancy_percentage=result.config.redundancy_percentage,
                                  redundancy_boundary_rule=result.config.redundancy_boundary_rule,
                                  redundancy_min_bars=result.config.redundancy_min_bars)
    eligible_ids = {trend.trend_id for trend in (result.trends if show_redundant else selection.chart_trends)}
    if chart_trend_ids is None:
        chart_trend_ids = eligible_ids
    diagnostics = result.diagnostics_frame()
    uncertain = diagnostics.filter(~pl.col("accepted")) if not diagnostics.is_empty() else diagnostics
    st.subheader("Trend analysis")
    if result.analysis_end is not None:
        start = result.analysis_start.isoformat() if result.analysis_start is not None else "unavailable"
        st.caption(f"Analysis range: {start} to {result.analysis_end.isoformat()}. "
                   "Durations are source bars. Clarity describes directional evidence, not probability or future continuation.")
    if getattr(result, "analysis_mode", "history") == "history":
        st.caption("This table includes every accepted historical interval. Each row was computed using only bars "
                   "through that interval's own endpoint, shown in End / analyzed through. Later bars do not change "
                   "that earlier result. Start is an estimated onset, not a first-recognition date.")
    boundary_match = ("both start and end differences are" if result.config.redundancy_boundary_rule == "and"
                      else "either the start or end difference is")
    display_description = ("Intervals are classified as redundant when" if show_redundant
                           else "Overlapping intervals are shown unless")
    st.caption(f"{len(chart_trend_ids):,} intervals on chart; {len(selection.redundant_to):,} redundant intervals; "
               f"{len(uncertain):,} uncertain candidates. All {len(result.trends):,} accepted intervals remain in the table. "
               f"{display_description} {boundary_match} within the larger of "
               f"{result.config.redundancy_min_bars} bars and "
               f"{result.config.redundancy_percentage:g}% of the shorter interval's inclusive bar count "
               "relative to an already selected same-direction representative. Selection proceeds by earliest ending bar, "
               "then higher clarity, longer duration, and interval ID for same-bar ties. With unchanged "
               "settings and earlier records, later endpoints cannot replace earlier selections. "
               "The underlying historical results stay available here.")
    if show_redundant:
        st.caption("Redundant intervals are eligible for display, subject to the λ and slope filters. "
                   "They keep their Redundant status and representative reference; On chart indicates current visibility.")
    if chart_trend_ids != eligible_ids:
        st.caption("The λ and slope filters hide nonmatching lines only. All interval results remain in the table; "
                   "On chart reflects current visibility.")
    st.caption("Line color identifies the smallest numeric λ that discovered the interval. Hover and table list "
               "every λ; every endpoint uses the same configured grid. Tail-derived intervals use a gray dashed line "
               "and have no λ.")
    if result.status == "insufficient_data":
        st.info("Insufficient completed bars for trend analysis with these settings.")
    elif not result.trends:
        st.info("No clear trend found for this interval.")
    else:
        table = _with_lambda_labels(result.to_frame())
        if "rate_per_bar" in table.columns:
            table = table.with_columns((pl.col("rate_per_bar") * 100).alias("rate_per_bar"))
        ids = table["trend_id"].to_list()
        table = table.with_columns(
            pl.col("trend_id").is_in(sorted(chart_trend_ids)).alias("on_chart"),
            pl.Series("status", ["Redundant" if item in selection.redundant_to else
                                 "On chart" if item in chart_trend_ids else "Filtered out" for item in ids]),
            pl.Series("redundant_to", [selection.redundant_to.get(item) for item in ids], dtype=pl.String),
        )
        leading = ("trend_id", "status", "redundant_to", "start_index", "end_index", "direction",
                   "clarity", "lambda_label", "origin")
        table = table.select([name for name in leading if name in table.columns]
                             + [name for name in table.columns if name not in leading])
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "trend_id": st.column_config.TextColumn("Trend"),
            "status": st.column_config.TextColumn("Display status"),
            "redundant_to": st.column_config.TextColumn("Representative trend"),
            "lambda_label": st.column_config.TextColumn("λ / penalty level"),
            "lambda_values": None,
            "lambda_levels": None,
            "origin": st.column_config.TextColumn("Discovery origin"),
            "on_chart": st.column_config.CheckboxColumn("On chart"),
            "direction": st.column_config.TextColumn("Direction"),
            "start": st.column_config.DatetimeColumn("Start"),
            "end": st.column_config.DatetimeColumn("End / analyzed through"),
            "start_index": st.column_config.NumberColumn("Start bar index", format="%d"),
            "end_index": st.column_config.NumberColumn("Endpoint bar index", format="%d"),
            "bars": st.column_config.NumberColumn("Bars"),
            "clarity": st.column_config.NumberColumn("Clarity", format="%.3f"),
            "rate_per_bar": st.column_config.NumberColumn("Fitted %/bar", format="%.4f"),
            "slope": st.column_config.NumberColumn("Slope (log-price per bar)", format="%.10g"),
            "boundary_robustness": st.column_config.NumberColumn("Boundary robustness", format="%.3f"),
            "magnitude": st.column_config.NumberColumn("Normalized slope", format="%.3f"),
            "tail_status": st.column_config.TextColumn("Tail status"),
            "retracement": st.column_config.NumberColumn("Retracement fraction", format="%.3f"),
            "onset_truncated": st.column_config.CheckboxColumn(
                "Touches lookback boundary",
                help="The interval starts at the earliest bar available to its endpoint analysis; the true start could be earlier.",
            ),
        })
    with (memory.widget(st.expander, "Applied trend settings", key="trend-applied-open") if memory else
          st.expander("Applied trend settings")):
        st.caption(f"Algorithm version: {result.algorithm_version}")
        st.json(result.config.to_dict())
    with (memory.widget(st.expander, "Uncertain intervals", key="trend-uncertain-open") if memory else
          st.expander("Uncertain intervals")):
        st.caption("These candidates did not pass the analysis acceptance gates. They are distinct from "
                   "redundant intervals, which passed but match a displayed representative under the applied "
                   "boundary matching rule and tolerance.")
        if uncertain.is_empty():
            st.caption("No uncertain candidates are available for this interval.")
        else:
            uncertain = _with_lambda_labels(uncertain).with_columns(pl.lit("Uncertain").alias("status"))
            leading = ("status", "start_index", "end_index", "direction", "clarity",
                       "rejection_reason", "lambda_label", "origin")
            uncertain = uncertain.select([name for name in leading if name in uncertain.columns]
                                         + [name for name in uncertain.columns if name not in leading])
            st.dataframe(uncertain, hide_index=True, width="stretch", column_config={
                "status": st.column_config.TextColumn("Display status"),
                "lambda_label": st.column_config.TextColumn("λ / penalty level"),
                "lambda_values": None,
                "lambda_levels": None,
                "origin": st.column_config.TextColumn("Discovery origin"),
                "rejection_reason": st.column_config.TextColumn("Reason uncertain"),
                "clarity": st.column_config.NumberColumn("Clarity", format="%.3f"),
            })
