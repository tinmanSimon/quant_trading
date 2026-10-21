"""Compatibility alias for :mod:`dashboard.state.migrations`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("dashboard.state.migrations")
