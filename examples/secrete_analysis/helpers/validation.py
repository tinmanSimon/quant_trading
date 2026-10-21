"""Example of a reusable local helper, included in code fingerprints."""


def require_ordered_periods(fast, slow):
    if fast >= slow:
        raise ValueError("Fast MA period must be smaller than slow MA period.")
