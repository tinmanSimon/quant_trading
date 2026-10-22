"""Compatibility alias for :mod:`research.analysis.core.input`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("research.analysis.core.input")
