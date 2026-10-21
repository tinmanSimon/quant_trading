"""Compatibility alias for :mod:`dashboard.components.session_schedules`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("dashboard.components.session_schedules")
