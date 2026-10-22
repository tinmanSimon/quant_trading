"""Entry signals whose fixed stop/target exits require portfolio execution."""

from abc import abstractmethod

import polars as pl

from .base import Strategy


class BracketStrategy(Strategy):
    """Request next-open entries with stop and target price multipliers.

    The engine owns position sizing and exits, including checks on the entry
    bar. A strategy receives only completed bars when deciding an entry.
    Implementations provide finite multipliers satisfying
    ``0 < stop_percentage < 1 < win_percentage``.
    """

    stop_percentage: float
    win_percentage: float

    @abstractmethod
    def entry_signal(self, history: pl.DataFrame) -> bool:
        """Whether the latest completed bar requests entry at the next open."""

    def prepare_entry_signals(self, history: pl.DataFrame) -> pl.Series | None:
        """Optionally calculate all entry decisions in one causal pass.

        Return one non-null boolean per row, decided using only that row and
        earlier rows. Appending future data must never alter an earlier value.
        The engine uses row i-1 for an entry at row i. Returning None retains
        the completed-history entry_signal interface.
        """
        return None

    def target_weight(self, history: pl.DataFrame) -> float:
        raise ValueError(
            "Bracket strategies require shared portfolio execution; "
            "they do not provide target allocations."
        )
