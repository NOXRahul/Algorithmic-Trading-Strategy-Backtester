"""
risk/manager.py — Risk Management Layer.

Responsibilities:
  • Convert SignalEvents → OrderEvents (with quantity)
  • Apply stop-loss and take-profit levels
  • ATR-based position sizing (volatility-adjusted)
  • Per-trade and portfolio-level risk limits
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from algotrader.core import (
    MarketEvent, OrderEvent, OrderSide, OrderType, SignalEvent, SignalDirection
)
from algotrader.data.loader import BarFeed

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ATR Calculation (no lookahead — uses history up_to bar)
# ─────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Compute ATR from the last `period` bars.
    Uses True Range = max(H-L, |H-Cprev|, |L-Cprev|)
    """
    if len(df) < period + 1:
        return np.nan

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    tr = np.maximum.reduce([
        highs[1:]  - lows[1:],
        np.abs(highs[1:]  - closes[:-1]),
        np.abs(lows[1:]   - closes[:-1]),
    ])
    return float(np.mean(tr[-period:]))


# ─────────────────────────────────────────────
# Position Sizer
# ─────────────────────────────────────────────

class ATRPositionSizer:
    """
    Risk a fixed fraction of equity per trade.
    Position size = (equity × risk_pct) / (ATR × atr_multiplier)

    This keeps dollar-risk per trade constant regardless of volatility.
    """

    def __init__(
        self,
        risk_pct:        float = 0.01,   # 1% of equity per trade
        atr_period:      int   = 14,
        atr_multiplier:  float = 2.0,    # stop distance in ATR units
        max_position_pct: float = 0.20,  # never exceed 20% of equity in one name
    ):
        self.risk_pct         = risk_pct
        self.atr_period       = atr_period
        self.atr_multiplier   = atr_multiplier
        self.max_position_pct = max_position_pct

    def size(
        self,
        equity:    float,
        price:     float,
        atr:       float,
        signal_strength: float = 1.0,
    ) -> float:
        if np.isnan(atr) or atr <= 0 or price <= 0:
            return 0.0

        stop_distance = atr * self.atr_multiplier
        dollar_risk   = equity * self.risk_pct * signal_strength
        raw_qty       = dollar_risk / stop_distance

        # Cap by max_position_pct
        max_qty = (equity * self.max_position_pct) / price
        qty = min(raw_qty, max_qty)
        return max(0.0, np.floor(qty))   # whole shares


class FixedFractionSizer:
    """Simple Kelly-lite: bet a fixed fraction of equity."""

    def __init__(self, fraction: float = 0.05):
        self.fraction = fraction

    def size(self, equity, price, atr=None, signal_strength=1.0):
        if price <= 0:
            return 0.0
        return max(0.0, np.floor((equity * self.fraction * signal_strength) / price))


# ─────────────────────────────────────────────
# Risk Manager
# ─────────────────────────────────────────────

class RiskManager:
    """
    Translates SignalEvents into OrderEvents, applying position sizing
    and risk controls.  Works as a stateless transformer — all state
    (cash, holdings) lives in the Portfolio.
    """

    def __init__(
        self,
        sizer:              ATRPositionSizer = None,
        atr_period:         int   = 14,
        stop_atr_multiple:  float = 2.0,
        tp_atr_multiple:    float = 4.0,
        max_open_positions: int   = 10,
        allow_short:        bool  = False,
    ):
        self.sizer              = sizer or ATRPositionSizer()
        self.atr_period         = atr_period
        self.stop_atr_mult      = stop_atr_multiple
        self.tp_atr_mult        = tp_atr_multiple
        self.max_open_positions = max_open_positions
        self.allow_short        = allow_short

    def process_signals(
        self,
        signals:          List[SignalEvent],
        market_events:    Dict[str, MarketEvent],
        feed:             BarFeed,
        equity:           float,
        open_positions:   Dict[str, float],   # symbol → qty held
    ) -> List[OrderEvent]:
        orders: List[OrderEvent] = []

        for sig in signals:
            bar = market_events.get(sig.symbol)
            if bar is None:
                continue

            price = bar.close   # sizing off current close; fill on next open

            # ── ATR for sizing and stops ──────────────────
            hist = feed.history(sig.symbol, up_to=sig.timestamp, n=self.atr_period + 5)
            atr  = compute_atr(hist, self.atr_period)

            # ── Determine action ──────────────────────────
            held_qty = open_positions.get(sig.symbol, 0.0)

            if sig.direction == SignalDirection.LONG:
                if held_qty > 0:
                    continue   # already long
                if not self.allow_short and len([s for s, q in open_positions.items() if q > 0]) >= self.max_open_positions:
                    log.debug("Max open positions reached — skipping %s", sig.symbol)
                    continue

                qty = self.sizer.size(equity, price, atr, sig.strength)
                if qty <= 0:
                    continue

                sl = sig.stop_loss   or (price - atr * self.stop_atr_mult if not np.isnan(atr) else None)
                tp = sig.take_profit or (price + atr * self.tp_atr_mult   if not np.isnan(atr) else None)

                orders.append(OrderEvent(
                    timestamp=sig.timestamp,
                    symbol=sig.symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.BUY,
                    quantity=qty,
                    stop_loss=sl,
                    take_profit=tp,
                ))

            elif sig.direction == SignalDirection.FLAT:
                if held_qty > 0:
                    orders.append(OrderEvent(
                        timestamp=sig.timestamp,
                        symbol=sig.symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,
                        quantity=held_qty,
                    ))
                elif held_qty < 0 and self.allow_short:
                    orders.append(OrderEvent(
                        timestamp=sig.timestamp,
                        symbol=sig.symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,
                        quantity=abs(held_qty),
                    ))

            elif sig.direction == SignalDirection.SHORT and self.allow_short:
                if held_qty < 0:
                    continue
                qty = self.sizer.size(equity, price, atr, sig.strength)
                if qty <= 0:
                    continue
                orders.append(OrderEvent(
                    timestamp=sig.timestamp,
                    symbol=sig.symbol,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL,
                    quantity=qty,
                ))

        return orders

    def check_stop_conditions(
        self,
        open_positions_detail: Dict[str, Dict],   # symbol → {qty, entry, sl, tp}
        market_events:         Dict[str, MarketEvent],
    ) -> List[OrderEvent]:
        """
        Check existing positions for stop-loss / take-profit triggers
        intrabar.  Uses the HIGH and LOW of the current bar.
        """
        orders: List[OrderEvent] = []

        for symbol, pos in open_positions_detail.items():
            bar = market_events.get(symbol)
            if bar is None or pos["qty"] == 0:
                continue

            sl = pos.get("stop_loss")
            tp = pos.get("take_profit")
            qty = pos["qty"]

            if qty > 0:   # long
                if sl and bar.low <= sl:
                    orders.append(OrderEvent(
                        timestamp=bar.timestamp,
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,
                        quantity=qty,
                    ))
                elif tp and bar.high >= tp:
                    orders.append(OrderEvent(
                        timestamp=bar.timestamp,
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,
                        quantity=qty,
                    ))

        return orders
