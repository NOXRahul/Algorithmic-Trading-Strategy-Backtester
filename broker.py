"""
execution/broker.py — Simulated order execution engine.

Slippage models:
  • FixedSlippage  — constant bps per trade
  • VolumeSlippage — market-impact proportional to order size vs. ADV

Commission models:
  • PerShareCommission
  • PercentCommission
  • TieredCommission  — realistic broker tiering

All fills use the NEXT bar's open price (or intrabar high/low for stops)
to avoid execution at the signal bar's close — a common source of bias.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import uuid

import numpy as np
import pandas as pd

from algotrader.core import (
    FillEvent, MarketEvent, OrderEvent, OrderSide, OrderStatus, OrderType
)


# ─────────────────────────────────────────────
# Slippage Models
# ─────────────────────────────────────────────

class BaseSlippage(ABC):
    @abstractmethod
    def apply(self, side: OrderSide, price: float, quantity: float,
               bar: MarketEvent) -> float:
        """Return the slippage cost (always positive)."""


class FixedSlippage(BaseSlippage):
    """Constant slippage in basis points."""

    def __init__(self, bps: float = 5.0):
        self.bps = bps / 10_000

    def apply(self, side, price, quantity, bar):
        return price * self.bps


class VolumeSlippage(BaseSlippage):
    """
    Market-impact model: slippage grows with participation rate.
    impact = spread + k * sqrt(quantity / adv)
    """

    def __init__(self, spread_bps: float = 3.0, impact_coeff: float = 0.1):
        self.spread   = spread_bps / 10_000
        self.k        = impact_coeff

    def apply(self, side, price, quantity, bar):
        adv = max(bar.volume * bar.close, 1)
        participation = quantity * price / adv
        return price * (self.spread + self.k * np.sqrt(participation))


# ─────────────────────────────────────────────
# Commission Models
# ─────────────────────────────────────────────

class BaseCommission(ABC):
    @abstractmethod
    def calculate(self, quantity: float, fill_price: float) -> float:
        """Return total commission in currency."""


class PerShareCommission(BaseCommission):
    def __init__(self, rate: float = 0.005, min_fee: float = 1.0):
        self.rate    = rate
        self.min_fee = min_fee

    def calculate(self, quantity, fill_price):
        return max(quantity * self.rate, self.min_fee)


class PercentCommission(BaseCommission):
    def __init__(self, pct: float = 0.001):   # 10 bps
        self.pct = pct

    def calculate(self, quantity, fill_price):
        return quantity * fill_price * self.pct


class TieredCommission(BaseCommission):
    """
    Tiered by notional value — common in prime-brokerage agreements.
    tiers: list of (notional_threshold, rate_pct)
    """

    DEFAULT_TIERS = [
        (0,           0.0015),
        (10_000,      0.0010),
        (100_000,     0.0007),
        (1_000_000,   0.0005),
    ]

    def __init__(self, tiers=None):
        self.tiers = sorted(tiers or self.DEFAULT_TIERS)

    def calculate(self, quantity, fill_price):
        notional = quantity * fill_price
        rate = self.tiers[0][1]
        for threshold, r in self.tiers:
            if notional >= threshold:
                rate = r
        return notional * rate


# ─────────────────────────────────────────────
# Simulated Broker
# ─────────────────────────────────────────────

class SimulatedBroker:
    """
    Executes pending orders against the next available bar.

    Execution rules (anti-bias):
      MARKET orders → fill at next bar's OPEN ± slippage
      LIMIT  orders → fill at limit_price if bar crosses it (open or intrabar)
      STOP   orders (embedded in FillEvent) → checked against bar high/low

    Pending orders carry over if the next bar doesn't trigger them.
    Orders expire after `max_bars_pending` bars.
    """

    def __init__(
        self,
        slippage:          BaseSlippage   = None,
        commission:        BaseCommission = None,
        max_bars_pending:  int = 1,
    ):
        self.slippage         = slippage  or FixedSlippage(bps=5)
        self.commission       = commission or PercentCommission(pct=0.001)
        self.max_bars_pending = max_bars_pending
        self._pending: List[Dict] = []   # {"order": OrderEvent, "bars_waited": int}

    def submit(self, order: OrderEvent) -> None:
        self._pending.append({"order": order, "bars_waited": 0})

    def process_bar(
        self,
        market_events: Dict[str, MarketEvent],
        strategy_id_map: Dict[str, str],   # order_id → strategy_id
    ) -> List[FillEvent]:
        """
        Attempt to fill all pending orders against the current bar.
        Returns a list of FillEvents.
        """
        fills: List[FillEvent] = []
        still_pending = []

        for item in self._pending:
            order: OrderEvent = item["order"]
            item["bars_waited"] += 1

            bar = market_events.get(order.symbol)
            if bar is None:
                still_pending.append(item)
                continue

            fill_price = self._try_fill(order, bar)

            if fill_price is not None:
                slip = self.slippage.apply(order.side, fill_price, order.quantity, bar)
                if order.side == OrderSide.BUY:
                    fill_price += slip
                else:
                    fill_price -= slip

                comm = self.commission.calculate(order.quantity, fill_price)
                order.status = OrderStatus.FILLED

                fills.append(FillEvent(
                    timestamp=bar.timestamp,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    fill_price=fill_price,
                    commission=comm,
                    slippage=slip * order.quantity,
                    order_id=order.order_id,
                    strategy_id=strategy_id_map.get(order.order_id, ""),
                ))
            elif item["bars_waited"] >= self.max_bars_pending:
                order.status = OrderStatus.CANCELLED
            else:
                still_pending.append(item)

        self._pending = still_pending
        return fills

    def _try_fill(self, order: OrderEvent, bar: MarketEvent) -> Optional[float]:
        if order.order_type == OrderType.MARKET:
            return bar.open   # next bar open

        if order.order_type == OrderType.LIMIT:
            lp = order.limit_price
            if order.side == OrderSide.BUY and bar.low <= lp:
                return min(lp, bar.open)
            if order.side == OrderSide.SELL and bar.high >= lp:
                return max(lp, bar.open)

        return None
