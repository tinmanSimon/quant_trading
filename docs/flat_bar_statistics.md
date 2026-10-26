# Flat-period bar statistics

On the dashboard's **Analysis** page, add **Flat-period bar statistics** to run
the analysis independently for each selected ticker. Choose the high threshold
and flat-period detection settings, then click **Run analysis**. The usual
**Save analysis** and **Saved analysis run** workflow preserves and reopens the
settings, ticker summaries, and period details.

The analysis uses the shared [flat analysis package](flat_analysis.md) on
completed bars from the selected data. All fixed/automatic width, duration,
percentile, multiplier, and optional longest-to-median bar settings are available.
Automatic width uses each candidate period's own bars, with no separate lookback.

## Bars included

The detector's latest-first, strictly disjoint summary defines the counted
population. A bar is counted at most once. Overlapping historical candidates
are not separate observations, and bars outside the retained flat periods are
excluded. Period endpoints are included. Maximum duration can split a long
flat history into several adjacent periods; the same selection rules apply
as in ordinary Flat analysis.

When run from the dashboard, recorded ingestion omissions separate periods.
A period cannot bridge a known missing bar, and neither can a close-to-close
return. This uses omission records from the selected datasets rather than
treating ordinary overnight or weekend intervals as missing bars. Historical
omission records whose timestamps are now present in the selected data are
treated as repaired. Direct callers can supply ISO timestamps through
`LogicContext.metadata["omitted_bar_timestamps"]`; a bare frame alone cannot
identify previously excluded candles.

## High threshold and red bars

**High threshold (% of open)** is a direct percentage, defaulting to **102**:

```text
above_threshold = high > open × (high_percentage / 100)
red = close < open
```

Thus 102 means `high > open × 1.02`. An exactly equal high does not qualify.
Doji bars (`close == open`) are not red, but they can contribute to the count
of all bars above the threshold. Every bar uses its own opening price.

The **Flat-bar summary** table reports:

| Column | Meaning |
| --- | --- |
| `source_bar_count` | All selected completed source bars. |
| `flat_period_count` | Number of retained disjoint flat periods. |
| `flat_bar_count` | Total bars inside those periods. |
| `red_bar_count` | Flat-period bars with `close < open`. |
| `red_high_above_count` | Red bars whose high is strictly above the threshold. |
| `red_high_not_above_count` | Red bars whose high is at or below the threshold. |
| `red_high_above_ratio` | `red_high_above_count / red_bar_count`; 0.6 means 60%. Unavailable when there are no red bars. |
| `all_high_above_count` | All qualifying flat-period bars, including red, green, and doji bars. |
| `return_count` | Number of close-to-close returns within periods. |
| `return_volatility_pct` | Population standard deviation of those returns, in percentage points. |

The two red-bar counts sum to `red_bar_count`. Count metrics are summed across
tickers in the dashboard's overall summary. Volatility is available for ticker
comparison and ranking; it is not automatically summed or averaged across tickers.

## Last-month summary

Alongside the full-range summary, **Last-month flat-bar summary** reports the
same statistics for the most recent calendar month, ending at the latest
completed source bar in the selected data. It uses the data's end rather than
today, so historical selections also have a meaningful recent summary. The
cutoff is inclusive, with the same time of day one calendar month earlier;
month-end dates clamp to the previous month's last day (March 31 → February
28 or 29).

The full-range flat periods are detected once. Periods crossing the cutoff
contribute only their in-window bars, even if that clipped portion is shorter
than the detection minimum duration. Both closes must be in the window and
in the same original flat period for a return to count. The recent period
count includes every retained period with at least one bar in the window.

Recent metrics use the `last_month_` prefix, including
`last_month_return_volatility_pct`, and are available for ticker comparison
and ranking. `last_month_cutoff` shows the requested lower bound;
`last_month_start` and `last_month_end` show the first and last included source
bars. A selected history shorter than a month uses all its available completed
bars. With no recent flat bars, counts are zero and volatility is unavailable.
Existing saved results keep their original output; rerun the analysis to add
the recent summary.

## Volatility

For each period, calculate simple close-to-close returns only when both closes
belong to that same period:

```text
return[i] = close[i] / close[i − 1] − 1
return_volatility_pct = 100 × std(returns, ddof=0)
```

The first bar in every period contributes no return. Changes between separate
periods are excluded, including between adjacent periods created by the maximum
duration cap. Consequently, `return_count = flat_bar_count − flat_period_count`.

The ticker summary pools all within-period returns before calculating the
standard deviation. It does not average period standard deviations. Values are
not annualized. A single return has population standard deviation zero;
constant returns also produce zero. With no detected periods, counts are zero
and volatility is null rather than zero. Insufficient data is identified in
the result message and metadata.

The **Flat-period details** table provides the same counts and volatility for
each period, along with its indices, timestamps, bounds, actual width, and
allowed width. It is ordered newest first.

This analysis is registered as `flat_bar_statistics`; its version includes the
flat engine version so changes to detection do not reuse earlier calculation
caches. Saved results retain their original settings and numerical outputs.
