"""Compatibility alias for :mod:`dashboard.charts.cache`."""

import sys
from .charts import cache as _implementation

sys.modules[__name__] = _implementation
