"""Compatibility alias for :mod:`research.analysis.trend.selection`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("research.analysis.trend.selection")
