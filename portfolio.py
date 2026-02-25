"""
portfolio/portfolio.py — Portfolio accounting layer.

Tracks:
  • Cash balance
  • Holdings (symbol → quantity)
  • Per-position metadata (entry price, stop, take-profit)
  • Mark-to-market equity
  • Realized and unrealized P&L
  • Full equity curve for analytics

Design: immutable-style updates — each bar appends a new equity snapshot
rather than mutating in place, making the equity curve naturally correct.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from algotrader.core import FillEvent, MarketEvent, OrderSide

log = logging.getLogger(__name__)


@dataclass
class Position:
    symbol:      str
    quantity:    float          # positive = long, negative = short
    avg_entry:   float
    stop_loss:   Optional[float] = None
    take_profit: Optional[float] = None
    realized_pnl: float = 0.0

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_entry

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.avg_entry) * self.quantity

    def market_value(self, current_price: float) -> float:
        return self.quantity * current_price


@dataclass
class TradeRecord:
    timestamp:   pd.Timestamp
    symbol:      str
    side:        str
    quantity:    float
    fill_price:  float
    commission:  float
    slippage:    float
    pnl:         float = 0.0   # realized PnL (populated on close)
    order_id:    str = ""
    strategy_id: str = ""


class Portfolio:
    """
    Central accounting ledger for the backtest.

    Usage pattern (called by the engine each bar):
      1. on_fill(fill)         — update holdings on execution
      2. mark_to_market(bars)  — snapshot equity at bar close
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        self.cash            = initial_capital

        self._positions:  Dict[str, Position]    = {}
        self._equity_curve: List[Dict]            = []
        self._trades:       List[TradeRecord]     = []
        self._total_commission: float             = 0.0
        self._total_slippage:   float             = 0.0

    # ─────────────────────────────────────────────
    # Fill processing
    # ─────────────────────────────────────────────

    def on_fill(self, fill: FillEvent) -> None:
        sym   = fill.symbol
        cost  = fill.fill_price * fill.quantity + fill.commission

        if fill.side == OrderSide.BUY:
            self._process_buy(fill)
            self.cash -= cost
        else:
            realized = self._process_sell(fill)
            self.cash += fill.fill_price * fill.quantity - fill.commission

        self._total_commission += fill.commission
        self._total_slippage   += fill.slippage
        log.debug("Fill: %s %s x %.0f @ %.4f  cash=%.2f",
                  fill.side.value, sym, fill.quantity, fill.fill_price, self.cash)

    def _process_buy(self, fill: FillEvent) -> None:
        sym = fill.symbol
        if sym in self._positions and self._positions[sym].quantity > 0:
            # Average up
            pos = self._positions[sym]
            total_qty  = pos.quantity + fill.quantity
            avg_entry  = (pos.quantity * pos.avg_entry + fill.quantity * fill.fill_price) / total_qty
            pos.quantity  = total_qty
            pos.avg_entry = avg_entry
        else:
            self._positions[sym] = Position(
                symbol=sym,
                quantity=fill.quantity,
                avg_entry=fill.fill_price,
            )
        self._record_trade(fill, 0.0)

    def _process_sell(self, fill: FillEvent) -> float:
        sym = fill.symbol
        realized = 0.0
        if sym in self._positions:
            pos = self._positions[sym]
            realized = (fill.fill_price - pos.avg_entry) * fill.quantity
            pos.quantity -= fill.quantity
            pos.realized_pnl += realized
            if abs(pos.quantity) < 1e-9:
                del self._positions[sym]
        self._record_trade(fill, realized)
        return realized

    def _record_trade(self, fill: FillEvent, pnl: float) -> None:
        self._trades.append(TradeRecord(
            timestamp=fill.timestamp,
            symbol=fill.symbol,
            side=fill.side.value,
            quantity=fill.quantity,
            fill_price=fill.fill_price,
            commission=fill.commission,
            slippage=fill.slippage,
            pnl=pnl,
            order_id=fill.order_id,
            strategy_id=fill.strategy_id,
        ))

    def attach_stop_tp(self, order_id: str, symbol: str,
                       stop_loss: Optional[float], take_profit: Optional[float]) -> None:
        """Called after fill to store SL/TP on the position."""
        if symbol in self._positions:
            self._positions[symbol].stop_loss   = stop_loss
            self._positions[symbol].take_profit = take_profit

    # ─────────────────────────────────────────────
    # Mark-to-market
    # ─────────────────────────────────────────────

    def mark_to_market(
        self,
        timestamp:     pd.Timestamp,
        market_events: Dict[str, MarketEvent],
    ) -> float:
        """Compute current equity and append to equity curve."""
        holdings_value = 0.0
        unrealized_pnl = 0.0

        for sym, pos in self._positions.items():
            if sym in market_events:
                price = market_events[sym].close
            else:
                price = pos.avg_entry   # stale if no bar

            mv = pos.market_value(price)
            holdings_value += mv
            unrealized_pnl += pos.unrealized_pnl(price)

        equity = self.cash + holdings_value
        realized_pnl = sum(t.pnl for t in self._trades)

        self._equity_curve.append({
            "timestamp":      timestamp,
            "cash":           self.cash,
            "holdings_value": holdings_value,
            "equity":         equity,
            "realized_pnl":   realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "drawdown":       0.0,   # computed post-hoc
        })
        return equity

    # ─────────────────────────────────────────────
    # Accessors
    # ─────────────────────────────────────────────

    @property
    def equity(self) -> float:
        return self._equity_curve[-1]["equity"] if self._equity_curve else self.initial_capital

    @property
    def open_positions(self) -> Dict[str, float]:
        return {s: p.quantity for s, p in self._positions.items()}

    @property
    def open_positions_detail(self) -> Dict[str, Dict]:
        return {
            s: {
                "qty":         p.quantity,
                "entry":       p.avg_entry,
                "stop_loss":   p.stop_loss,
                "take_profit": p.take_profit,
            }
            for s, p in self._positions.items()
        }

    def equity_curve(self) -> pd.DataFrame:
        df = pd.DataFrame(self._equity_curve).set_index("timestamp")
        if len(df):
            rolling_max = df["equity"].cummax()
            df["drawdown"] = (df["equity"] - rolling_max) / rolling_max
        return df

    def trade_log(self) -> pd.DataFrame:
        return pd.DataFrame([vars(t) for t in self._trades])

    def summary_stats(self) -> Dict:
        return {
            "initial_capital":   self.initial_capital,
            "final_equity":      self.equity,
            "total_return_pct":  (self.equity / self.initial_capital - 1) * 100,
            "total_commission":  self._total_commission,
            "total_slippage":    self._total_slippage,
            "n_trades":          len(self._trades),
            "n_positions_open":  len(self._positions),
        }
