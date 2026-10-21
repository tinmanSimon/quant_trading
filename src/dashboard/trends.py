"""Compatibility alias for :mod:`dashboard.market_data.legacy_trends`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("dashboard.market_data.legacy_trends")
