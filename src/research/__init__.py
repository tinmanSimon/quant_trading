"""Public local research workflow, imported only when requested.

Analysis can be used independently of dataset services, strategies, and the
backtesting engine. The established ``from research import Research`` API is
preserved through lazy exports.
"""

from importlib import import_module


_EXPORTS = {
    "Research": (".api", "Research"),
    "ResearchRun": (".runs", "ResearchRun"),
    "ResearchError": (".errors", "ResearchError"),
    "PreflightError": (".errors", "PreflightError"),
    "PreflightReport": (".datasets", "PreflightReport"),
    "DataIssue": (".datasets", "DataIssue"),
    "Instrument": (".instruments", "Instrument"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
