"""Linear-time longest bounded wick envelopes for every causal endpoint."""

from collections import deque
from typing import Callable

import numpy as np

from .models import FlatConfig


def envelope(lower: float, upper: float) -> tuple[float, float]:
    """Midpoint and total width percent without overflowing finite prices."""
    width = upper - lower
    center = lower + width / 2
    return center, 100 * (width / center)


class EnvelopeWindows:
    """Numeric endpoint records; no period objects or candidate-interval matrix.

    Adding a candle can only enlarge the range relative to its midpoint. Once
    an earlier start fails it stays invalid at later endpoints. Each index is
    inserted and removed from each extrema deque once, giving O(N) work.
    """

    def __init__(self, highs: np.ndarray, lows: np.ndarray, config: FlatConfig,
                 progress: Callable[[int, int], None] | None = None):
        count = len(highs)
        self.starts = np.full(count, -1, dtype=np.int64)
        self.lowers = np.full(count, np.nan)
        self.uppers = np.full(count, np.nan)
        total = max(0, count - config.min_duration + 1)
        if progress is not None:
            progress(0, total)
        high_indices, low_indices = deque(), deque()
        start = 0
        # Fixed per-configuration tolerance preserves monotonic eligibility.
        limit = config.max_range_pct * (1 + 32 * np.finfo(float).eps)
        for end in range(count):
            while high_indices and highs[high_indices[-1]] <= highs[end]:
                high_indices.pop()
            high_indices.append(end)
            while low_indices and lows[low_indices[-1]] >= lows[end]:
                low_indices.pop()
            low_indices.append(end)
            start = max(start, end - config.max_duration + 1)
            while high_indices and high_indices[0] < start:
                high_indices.popleft()
            while low_indices and low_indices[0] < start:
                low_indices.popleft()
            while start <= end:
                lower, upper = float(lows[low_indices[0]]), float(highs[high_indices[0]])
                if envelope(lower, upper)[1] <= limit:
                    break
                start += 1
                if high_indices[0] < start:
                    high_indices.popleft()
                if low_indices[0] < start:
                    low_indices.popleft()
            if end - start + 1 >= config.min_duration:
                self.starts[end] = start
                self.lowers[end] = lows[low_indices[0]]
                self.uppers[end] = highs[high_indices[0]]
            if progress is not None and end + 1 >= config.min_duration:
                progress(end - config.min_duration + 2, total)
