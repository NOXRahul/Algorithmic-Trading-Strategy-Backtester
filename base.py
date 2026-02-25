"""
strategy/base.py — Abstract Strategy interface.

Every strategy must subclass BaseStrategy and implement `on_bar`.
The framework guarantees:
  • `feed.history(symbol, up_to=ts)` never exposes future data.
  • Each strategy instance carries a unique strategy_id for attribution.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

from algotrader.core import MarketEvent, SignalEvent, SignalDirection
from algotrader.data.loader import BarFeed


class BaseStrategy(ABC):
    """
    Abstract base class for all strategies.

    Subclasses implement `on_bar` and call `self.emit_signal(...)` to
    generate trading signals.  No order management here — that belongs
    to the portfolio/risk layer.
    """

    def __init__(self, strategy_id: str, symbols: List[str]):
        self.strategy_id = strategy_id
        self.symbols     = symbols
        self._signals:   List[SignalEvent] = []
        self._feed:      Optional[BarFeed] = None

    def attach_feed(self, feed: BarFeed) -> None:
        """Called by the engine before the backtest loop starts."""
        self._feed = feed

    @abstractmethod
    def on_bar(
        self,
        timestamp: pd.Timestamp,
        market_events: Dict[str, MarketEvent],
    ) -> None:
        """
        Called once per bar.  Implement your signal logic here.
        Use self.history(...) to access past data safely.
        """

    # ── helpers ──────────────────────────────────────────────────

    def history(self, symbol: str, up_to: pd.Timestamp, n: Optional[int] = None) -> pd.DataFrame:
        """Safe historical data access — no lookahead."""
        assert self._feed is not None, "Feed not attached. Call attach_feed first."
        return self._feed.history(symbol, up_to=up_to, n=n)

    def emit_signal(
        self,
        timestamp:   pd.Timestamp,
        symbol:      str,
        direction:   SignalDirection,
        strength:    float = 1.0,
        stop_loss:   Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> None:
        self._signals.append(SignalEvent(
            timestamp=timestamp,
            symbol=symbol,
            strategy_id=self.strategy_id,
            direction=direction,
            strength=max(0.0, min(1.0, strength)),
            stop_loss=stop_loss,
            take_profit=take_profit,
        ))

    def flush_signals(self) -> List[SignalEvent]:
        """Called by the engine to collect signals after each bar."""
        out, self._signals = self._signals, []
        return out

    def on_fill(self, fill) -> None:  # noqa: ANN001
        """Optional hook — strategies can track their own positions."""
