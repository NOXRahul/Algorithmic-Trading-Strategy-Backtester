"""
data/loader.py — Historical OHLCV data layer.

Design decisions:
  • Data is loaded ONCE and iterated bar-by-bar to prevent lookahead bias.
  • Resampling happens BEFORE the backtest loop so the strategy never sees
    future bars from a finer timeframe.
  • Validation raises early rather than silently corrupting results.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from algotrader.core import MarketEvent

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

REQUIRED_COLS = {"open", "high", "low", "close", "volume"}

def validate_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Validate and clean a raw OHLCV DataFrame.
    Raises ValueError on unrecoverable issues.
    """
    df.columns = [c.lower() for c in df.columns]
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"[{symbol}] Missing columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"[{symbol}] Index must be DatetimeIndex")

    if not df.index.is_monotonic_increasing:
        log.warning("[%s] Index not sorted — sorting now.", symbol)
        df = df.sort_index()

    # Drop exact duplicates
    n_dup = df.index.duplicated().sum()
    if n_dup:
        log.warning("[%s] Dropping %d duplicate timestamps.", symbol, n_dup)
        df = df[~df.index.duplicated(keep="last")]

    # OHLC sanity
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl:
        log.warning("[%s] %d bars with high < low — clamping.", symbol, bad_hl)
        df["high"] = df[["high", "low"]].max(axis=1)
        df["low"]  = df[["high", "low"]].min(axis=1)

    # Forward-fill small gaps (≤ 5 bars), drop leading NaNs
    df = df.ffill(limit=5).dropna()

    # Ensure numeric types
    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(REQUIRED_COLS))
    log.info("[%s] Validated: %d bars from %s to %s",
             symbol, len(df), df.index[0].date(), df.index[-1].date())
    return df[list(REQUIRED_COLS)]


# ─────────────────────────────────────────────
# Resampler
# ─────────────────────────────────────────────

RESAMPLE_RULES = {
    "weekly":    "W",
    "monthly":   "ME",
    "quarterly": "QE",
}

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Downsample OHLCV data.  rule can be a pandas offset alias ('W', 'ME', ...)
    or a friendly name from RESAMPLE_RULES.
    """
    rule = RESAMPLE_RULES.get(rule.lower(), rule)
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    resampled = df.resample(rule).agg(agg).dropna()
    log.info("Resampled %d bars → %d bars (%s)", len(df), len(resampled), rule)
    return resampled


# ─────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────

class CSVLoader:
    """Load OHLCV data from CSV files."""

    def __init__(
        self,
        data_dir: str = ".",
        date_col: str = "date",
        resample_to: Optional[str] = None,
    ):
        self.data_dir   = Path(data_dir)
        self.date_col   = date_col
        self.resample_to = resample_to

    def load(self, symbol: str) -> pd.DataFrame:
        path = self.data_dir / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No data file for {symbol} at {path}")

        df = pd.read_csv(path, parse_dates=[self.date_col], index_col=self.date_col)
        df = validate_ohlcv(df, symbol)

        if self.resample_to:
            df = resample_ohlcv(df, self.resample_to)

        return df

    def load_many(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        return {s: self.load(s) for s in symbols}


class DataFrameLoader:
    """Use pre-existing DataFrames (useful for testing / live feeds)."""

    def __init__(self, frames: Dict[str, pd.DataFrame], resample_to: Optional[str] = None):
        self._frames     = frames
        self.resample_to = resample_to

    def load(self, symbol: str) -> pd.DataFrame:
        df = validate_ohlcv(self._frames[symbol].copy(), symbol)
        if self.resample_to:
            df = resample_ohlcv(df, self.resample_to)
        return df

    def load_many(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        return {s: self.load(s) for s in symbols}


# ─────────────────────────────────────────────
# Bar Iterator  (the core anti-lookahead engine)
# ─────────────────────────────────────────────

class BarFeed:
    """
    Merges multiple symbol DataFrames and yields one MarketEvent at a time
    in strict chronological order.

    Only bars UP TO (and including) the current timestamp are ever visible,
    making lookahead bias structurally impossible.
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self._data = data
        # Align on a common index
        self._index: pd.DatetimeIndex = (
            pd.concat([df["close"] for df in data.values()], axis=1)
            .sort_index()
            .index
        )

    @property
    def symbols(self) -> List[str]:
        return list(self._data.keys())

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._index

    def __iter__(self) -> Iterator[Tuple[pd.Timestamp, Dict[str, MarketEvent]]]:
        """
        Yields (timestamp, {symbol: MarketEvent}) for every bar.
        Symbols with no data on a given date are omitted from the dict.
        """
        for ts in self._index:
            events: Dict[str, MarketEvent] = {}
            for symbol, df in self._data.items():
                if ts in df.index:
                    row = df.loc[ts]
                    events[symbol] = MarketEvent(
                        timestamp=ts,
                        symbol=symbol,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                    )
            if events:
                yield ts, events

    def history(self, symbol: str, up_to: pd.Timestamp, n: Optional[int] = None) -> pd.DataFrame:
        """
        Return historical bars for a symbol strictly up to `up_to`.
        Strategies MUST use this method — never index the raw DataFrame directly.
        """
        df = self._data[symbol]
        hist = df[df.index <= up_to]
        if n is not None:
            hist = hist.iloc[-n:]
        return hist
