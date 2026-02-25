"""
analytics/performance.py — Performance analytics engine.

All metrics are computed from the equity curve DataFrame produced by
Portfolio.equity_curve().  No market data required post-backtest.

Metrics:
  CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Rolling Sharpe,
  Win Rate, Profit Factor, Exposure %, Avg Trade, Best/Worst Trade
"""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


# ─────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────

def cagr(equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound Annual Growth Rate."""
    if len(equity_curve) < 2:
        return 0.0
    n_periods = len(equity_curve)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    return float(total_return ** (periods_per_year / n_periods) - 1)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    excess = returns - risk_free_rate / periods_per_year
    if excess.std(ddof=1) == 0:
        return 0.0
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    excess    = returns - risk_free_rate / periods_per_year
    downside  = excess[excess < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if downside_std == 0:
        return np.inf if excess.mean() > 0 else 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Returns max drawdown as a negative fraction (e.g. -0.25 = -25%)."""
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(
    equity_curve: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    md = max_drawdown(equity_curve)
    if md == 0:
        return np.inf
    return cagr(equity_curve, periods_per_year) / abs(md)


def rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Rolling Sharpe over a trailing window."""
    excess = returns - risk_free_rate / periods_per_year
    roll_mean = excess.rolling(window).mean()
    roll_std  = excess.rolling(window).std(ddof=1)
    return (roll_mean / roll_std) * np.sqrt(periods_per_year)


def exposure(returns: pd.Series, equity_curve: pd.Series = None) -> float:
    """
    Percentage of time the strategy is in the market.
    Proxy: fraction of periods with non-zero returns.
    """
    return float((returns != 0).mean())


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    rolling_max = equity_curve.cummax()
    return (equity_curve - rolling_max) / rolling_max


# ─────────────────────────────────────────────
# Trade-level stats
# ─────────────────────────────────────────────

def trade_statistics(trade_log: pd.DataFrame) -> Dict:
    sells = trade_log[trade_log["side"] == "SELL"].copy()
    if sells.empty:
        return {}

    wins   = sells[sells["pnl"] > 0]["pnl"]
    losses = sells[sells["pnl"] <= 0]["pnl"]

    gross_profit = wins.sum()
    gross_loss   = abs(losses.sum())

    return {
        "n_trades":        len(sells),
        "n_wins":          len(wins),
        "n_losses":        len(losses),
        "win_rate":        len(wins) / len(sells) if len(sells) else 0,
        "profit_factor":   gross_profit / gross_loss if gross_loss else np.inf,
        "avg_win":         wins.mean()   if len(wins)   else 0,
        "avg_loss":        losses.mean() if len(losses) else 0,
        "best_trade":      sells["pnl"].max(),
        "worst_trade":     sells["pnl"].min(),
        "avg_trade_pnl":   sells["pnl"].mean(),
        "total_pnl":       sells["pnl"].sum(),
        "gross_profit":    gross_profit,
        "gross_loss":      gross_loss,
        "expectancy":      sells["pnl"].mean(),
    }


# ─────────────────────────────────────────────
# Full performance report
# ─────────────────────────────────────────────

def full_report(
    equity_df:  pd.DataFrame,
    trade_log:  pd.DataFrame,
    initial_capital: float,
    risk_free_rate:  float = 0.0,
    periods_per_year: int  = TRADING_DAYS_PER_YEAR,
) -> Dict:
    ec      = equity_df["equity"]
    returns = ec.pct_change().dropna()

    report = {
        "Initial Capital":   f"${initial_capital:,.2f}",
        "Final Equity":      f"${ec.iloc[-1]:,.2f}",
        "Total Return":      f"{(ec.iloc[-1]/ec.iloc[0]-1)*100:.2f}%",
        "CAGR":              f"{cagr(ec, periods_per_year)*100:.2f}%",
        "Sharpe Ratio":      f"{sharpe_ratio(returns, risk_free_rate, periods_per_year):.3f}",
        "Sortino Ratio":     f"{sortino_ratio(returns, risk_free_rate, periods_per_year):.3f}",
        "Calmar Ratio":      f"{calmar_ratio(ec, periods_per_year):.3f}",
        "Max Drawdown":      f"{max_drawdown(ec)*100:.2f}%",
        "Exposure":          f"{exposure(returns)*100:.1f}%",
        "Volatility (Ann)":  f"{returns.std()*np.sqrt(periods_per_year)*100:.2f}%",
        "Total Commission":  f"${equity_df.get('commission', pd.Series([0])).sum():,.2f}" if "commission" in equity_df else "N/A",
        "Start Date":        str(ec.index[0].date()),
        "End Date":          str(ec.index[-1].date()),
        "N Bars":            len(ec),
    }

    if not trade_log.empty:
        ts = trade_statistics(trade_log)
        report.update({
            "N Trades":       ts.get("n_trades", 0),
            "Win Rate":       f"{ts.get('win_rate',0)*100:.1f}%",
            "Profit Factor":  f"{ts.get('profit_factor', 0):.2f}",
            "Avg Trade PnL":  f"${ts.get('avg_trade_pnl', 0):,.2f}",
            "Best Trade":     f"${ts.get('best_trade', 0):,.2f}",
            "Worst Trade":    f"${ts.get('worst_trade', 0):,.2f}",
        })

    return report
