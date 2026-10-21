"""Compatibility alias for :mod:`dashboard.charts.series`."""

import sys
from .charts import series as _implementation

sys.modules[__name__] = _implementation
