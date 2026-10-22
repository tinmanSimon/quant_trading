"""Reusable research analyses, aligned indicators, and ordinary Python scripts.

Imports are lazy so numerical research does not initialize dashboard or
private-script discovery.
"""

from importlib import import_module

_EXPORTS = {
    "BaseChartLogic": "core.base", "LogicContext": "core.base", "LogicResult": "core.base", "LogicSpec": "core.base",
    "LogicDefinition": "core.registry", "LogicRegistry": "core.registry", "discover_logics": "core.registry",
    "AppliedRun": "core.runner", "LogicRunner": "core.runner",
    "AnalysisContext": "scripting.context", "Feature": "scripting.context", "AnalysisResult": "scripting.results",
    "analysis": "scripting.decorator", "run_script": "scripting.decorator", "load_registry": "scripting.discovery",
    "crossed_above": "scripting.signals", "crossed_below": "scripting.signals",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
