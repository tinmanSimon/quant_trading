"""Compatibility alias for :mod:`dashboard.pages.deletion`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("dashboard.pages.deletion")
