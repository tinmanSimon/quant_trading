"""Reusable research analyses, aligned indicators, and ordinary Python scripts.

Imports are lazy so numerical research does not initialize dashboard or
private-script discovery.
"""

from importlib import import_module

_EXPORTS = {
    "BaseChartLogic": "base", "LogicContext": "base", "LogicResult": "base", "LogicSpec": "base",
    "LogicDefinition": "registry", "LogicRegistry": "registry", "discover_logics": "registry",
    "AppliedRun": "runner", "LogicRunner": "runner",
    "AnalysisContext": "context", "Feature": "context", "AnalysisResult": "results",
    "analysis": "scripts", "run_script": "scripts", "load_registry": "loader",
    "crossed_above": "signals", "crossed_below": "signals",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
