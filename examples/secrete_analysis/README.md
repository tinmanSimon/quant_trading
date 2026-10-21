# Personal analysis example

Copy this directory to `secrete_analysis/` at the repository root. That root
directory is ignored by Git; this generic template remains tracked. Existing
private files should not be overwritten when updating the template.

In **Market data**, click **Reload analyses**, choose **Example: MA crossover
and RSI**, add it, configure its settings, and click **Apply**. Display controls
can hide individual layers without recalculating. Use analysis navigation above
the chart or select a signal table row to jump to an observation.

Edit `analyses/ma_rsi_example.py` or add other decorated functions. Shared code
belongs in `helpers/`; import it relatively, as the example does. Only trusted
local Python files should be placed here: scripts run with the application's
permissions. Script and helper changes invalidate cached results.

The example finds independent MA/RSI observations, optionally confirmed by a
trend observed at the same bar. It does not simulate positions or returns.
Periods count bars of the selected timeframe. See `docs/analysis_scripts.md`
in the repository for the reusable API and research usage.
