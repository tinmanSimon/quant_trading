"""Compatibility alias for :mod:`dashboard.state.workspace`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("dashboard.state.workspace")
