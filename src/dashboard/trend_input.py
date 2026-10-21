"""Compatibility alias for :mod:`research.analysis.input`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("research.analysis.input")
