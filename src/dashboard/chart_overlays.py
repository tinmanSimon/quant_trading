"""Compatibility alias for :mod:`dashboard.charts.overlays`."""

import sys
from .charts import overlays as _implementation

sys.modules[__name__] = _implementation
