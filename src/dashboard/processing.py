"""Compatibility alias for :mod:`dashboard.pages.processing`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("dashboard.pages.processing")
