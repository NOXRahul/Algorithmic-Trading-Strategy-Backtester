"""
core.py — Shared enums, dataclasses, and event types.
All layers communicate through typed events to avoid tight coupling.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import pandas as pd


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class OrderSide(Enum):
    BUY  = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"

class OrderStatus(Enum):
    PENDING   = auto()
    FILLED    = auto()
    CANCELLED = auto()
    REJECTED  = auto()

class SignalDirection(Enum):
    LONG  = 1
    SHORT = -1
    FLAT  = 0


# ─────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────

@dataclass
class MarketEvent:
    """Fired once per bar for each symbol."""
    timestamp: pd.Timestamp
    symbol:    str
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float


@dataclass
class SignalEvent:
    """Emitted by a Strategy when it detects an opportunity."""
    timestamp:   pd.Timestamp
    symbol:      str
    strategy_id: str
    direction:   SignalDirection
    strength:    float = 1.0          # 0–1 scalar for position sizing
    stop_loss:   Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class OrderEvent:
    """Created by the Portfolio / RiskManager from a SignalEvent."""
    timestamp:  pd.Timestamp
    symbol:     str
    order_type: OrderType
    side:       OrderSide
    quantity:   float
    limit_price: Optional[float] = None
    stop_loss:   Optional[float] = None
    take_profit: Optional[float] = None
    order_id:   str = field(default_factory=lambda: _next_id())
    status:     OrderStatus = OrderStatus.PENDING


@dataclass
class FillEvent:
    """Returned by the Execution Layer after an order is filled."""
    timestamp:   pd.Timestamp
    symbol:      str
    side:        OrderSide
    quantity:    float
    fill_price:  float
    commission:  float
    slippage:    float
    order_id:    str
    strategy_id: str = ""


# ─────────────────────────────────────────────
# Simple auto-increment ID
# ─────────────────────────────────────────────

_order_counter = 0

def _next_id() -> str:
    global _order_counter
    _order_counter += 1
    return f"ORD-{_order_counter:06d}"
