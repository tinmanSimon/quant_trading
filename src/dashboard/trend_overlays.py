"""Compatibility alias for :mod:`dashboard.charts.trend_overlays`."""

import sys
from .charts import trend_overlays as _implementation

sys.modules[__name__] = _implementation
