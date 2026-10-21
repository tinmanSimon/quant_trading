"""Compatibility alias for :mod:`dashboard.state.trend_settings`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("dashboard.state.trend_settings")
