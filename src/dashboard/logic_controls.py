"""Compatibility alias for :mod:`dashboard.market_data.analysis_controls`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("dashboard.market_data.analysis_controls")
