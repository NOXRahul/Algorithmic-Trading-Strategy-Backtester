"""
strategies/examples.py — Plug-and-play example strategies.

1. MovingAverageCrossover  — Classic dual-MA trend follower
2. RSIMeanReversion        — Buy oversold, sell overbought
3. BollingerBandBreakout   — Breakout on band expansion
4. MomentumStrategy        — Rate-of-change momentum with ATR filter
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd

from algotrader.core import MarketEvent, SignalDirection
from algotrader.strategy.base import BaseStrategy


# ─────────────────────────────────────────────
# 1. Moving Average Crossover
# ─────────────────────────────────────────────

class MovingAverageCrossover(BaseStrategy):
    """
    Go long when fast SMA crosses above slow SMA.
    Exit (go flat) when fast SMA crosses below slow SMA.

    Anti-lookahead: uses history strictly up_to current timestamp.
    """

    def __init__(
        self,
        symbols:    List[str],
        fast_period: int = 20,
        slow_period: int = 50,
        strategy_id: str = "MA_Cross",
    ):
        super().__init__(strategy_id, symbols)
        self.fast = fast_period
        self.slow = slow_period
        self._prev_fast: Dict[str, float] = {}
        self._prev_slow: Dict[str, float] = {}

    def on_bar(self, timestamp: pd.Timestamp, market_events: Dict[str, MarketEvent]) -> None:
        for symbol in self.symbols:
            if symbol not in market_events:
                continue

            hist = self.history(symbol, up_to=timestamp, n=self.slow + 5)
            if len(hist) < self.slow:
                continue

            closes = hist["close"]
            fast_ma = closes.iloc[-self.fast:].mean()
            slow_ma = closes.iloc[-self.slow:].mean()

            prev_fast = self._prev_fast.get(symbol, fast_ma)
            prev_slow = self._prev_slow.get(symbol, slow_ma)

            # Golden cross
            if prev_fast <= prev_slow and fast_ma > slow_ma:
                self.emit_signal(timestamp, symbol, SignalDirection.LONG)

            # Death cross
            elif prev_fast >= prev_slow and fast_ma < slow_ma:
                self.emit_signal(timestamp, symbol, SignalDirection.FLAT)

            self._prev_fast[symbol] = fast_ma
            self._prev_slow[symbol] = slow_ma


# ─────────────────────────────────────────────
# 2. RSI Mean Reversion
# ─────────────────────────────────────────────

class RSIMeanReversion(BaseStrategy):
    """
    Buy when RSI < oversold_level; sell when RSI > overbought_level.
    Signal strength is inversely proportional to RSI (deeper = stronger).
    """

    def __init__(
        self,
        symbols:          List[str],
        rsi_period:       int   = 14,
        oversold_level:   float = 30.0,
        overbought_level: float = 70.0,
        strategy_id:      str   = "RSI_MR",
    ):
        super().__init__(strategy_id, symbols)
        self.period     = rsi_period
        self.oversold   = oversold_level
        self.overbought = overbought_level

    def _compute_rsi(self, closes: pd.Series) -> float:
        delta  = closes.diff().dropna()
        gain   = delta.clip(lower=0).ewm(com=self.period - 1, adjust=False).mean()
        loss   = (-delta.clip(upper=0)).ewm(com=self.period - 1, adjust=False).mean()
        rs     = gain / loss.replace(0, np.nan)
        return float(100 - 100 / (1 + rs.iloc[-1]))

    def on_bar(self, timestamp: pd.Timestamp, market_events: Dict[str, MarketEvent]) -> None:
        for symbol in self.symbols:
            if symbol not in market_events:
                continue

            hist = self.history(symbol, up_to=timestamp, n=self.period * 3)
            if len(hist) < self.period + 1:
                continue

            rsi = self._compute_rsi(hist["close"])

            if rsi < self.oversold:
                strength = (self.oversold - rsi) / self.oversold   # deeper = stronger
                self.emit_signal(timestamp, symbol, SignalDirection.LONG, strength=strength)

            elif rsi > self.overbought:
                self.emit_signal(timestamp, symbol, SignalDirection.FLAT)


# ─────────────────────────────────────────────
# 3. Bollinger Band Breakout
# ─────────────────────────────────────────────

class BollingerBandBreakout(BaseStrategy):
    """
    Enter long on close above upper band (momentum breakout variant).
    Exit when price reverts below the middle band.
    """

    def __init__(
        self,
        symbols:     List[str],
        period:      int   = 20,
        n_std:       float = 2.0,
        strategy_id: str   = "BB_Breakout",
    ):
        super().__init__(strategy_id, symbols)
        self.period = period
        self.n_std  = n_std

    def on_bar(self, timestamp: pd.Timestamp, market_events: Dict[str, MarketEvent]) -> None:
        for symbol in self.symbols:
            if symbol not in market_events:
                continue

            bar  = market_events[symbol]
            hist = self.history(symbol, up_to=timestamp, n=self.period + 5)
            if len(hist) < self.period:
                continue

            closes = hist["close"]
            mid    = closes.rolling(self.period).mean().iloc[-1]
            std    = closes.rolling(self.period).std(ddof=1).iloc[-1]
            upper  = mid + self.n_std * std

            if bar.close > upper:
                self.emit_signal(timestamp, symbol, SignalDirection.LONG)
            elif bar.close < mid:
                self.emit_signal(timestamp, symbol, SignalDirection.FLAT)


# ─────────────────────────────────────────────
# 4. Momentum (Rate of Change)
# ─────────────────────────────────────────────

class MomentumStrategy(BaseStrategy):
    """
    Buy when N-day ROC is positive AND above its own moving average.
    Uses ATR filter to avoid low-volatility chop.
    """

    def __init__(
        self,
        symbols:       List[str],
        roc_period:    int   = 20,
        ma_period:     int   = 10,
        atr_period:    int   = 14,
        min_atr_pct:   float = 0.005,   # minimum ATR/price ratio
        strategy_id:   str   = "Momentum",
    ):
        super().__init__(strategy_id, symbols)
        self.roc_period  = roc_period
        self.ma_period   = ma_period
        self.atr_period  = atr_period
        self.min_atr_pct = min_atr_pct

    def on_bar(self, timestamp: pd.Timestamp, market_events: Dict[str, MarketEvent]) -> None:
        from algotrader.risk.manager import compute_atr

        for symbol in self.symbols:
            if symbol not in market_events:
                continue

            bar  = market_events[symbol]
            n    = max(self.roc_period, self.atr_period) + self.ma_period + 5
            hist = self.history(symbol, up_to=timestamp, n=n)
            if len(hist) < n // 2:
                continue

            closes = hist["close"]
            roc    = (closes.iloc[-1] / closes.iloc[-self.roc_period] - 1) * 100
            roc_ma = pd.Series(
                [(closes.iloc[i] / closes.iloc[i - self.roc_period] - 1) * 100
                 for i in range(self.roc_period, len(closes))]
            ).rolling(self.ma_period).mean().iloc[-1]

            atr      = compute_atr(hist, self.atr_period)
            atr_pct  = atr / bar.close if bar.close > 0 else 0

            if atr_pct < self.min_atr_pct:
                continue   # avoid choppy, low-vol regimes

            if roc > 0 and roc > roc_ma:
                strength = min(1.0, roc / 10)   # scale by momentum magnitude
                self.emit_signal(timestamp, symbol, SignalDirection.LONG, strength=strength)
            elif roc < 0 and roc < roc_ma:
                self.emit_signal(timestamp, symbol, SignalDirection.FLAT)
