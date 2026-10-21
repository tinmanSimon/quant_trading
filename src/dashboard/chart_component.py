"""Compatibility alias for :mod:`dashboard.charts.component`."""

import sys
from .charts import component as _implementation

sys.modules[__name__] = _implementation
