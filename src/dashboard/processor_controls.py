"""Compatibility alias for :mod:`dashboard.components.schema_controls`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("dashboard.components.schema_controls")
