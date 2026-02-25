"""
reporting/report.py — Backtest report generation.

Produces:
  1. Tabular performance summary (terminal + CSV)
  2. Equity curve chart
  3. Drawdown chart
  4. Trade PnL distribution histogram
  5. Rolling Sharpe chart

All charts saved to a single multi-panel PNG.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from algotrader.analytics.performance import (
    drawdown_series, rolling_sharpe, full_report, trade_statistics
)


DARK_BG   = "#0d1117"
GREEN     = "#2ea043"
RED       = "#f85149"
BLUE      = "#58a6ff"
AMBER     = "#e3b341"
GRID_COL  = "#21262d"
TEXT_COL  = "#c9d1d9"
SPINE_COL = "#30363d"


def _style_ax(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.7)


def _pct_fmt(x, pos):
    return f"{x*100:.0f}%"

def _dollar_fmt(x, pos):
    return f"${x:,.0f}"


class BacktestReport:
    """
    Renders a full backtest report to stdout and PNG.

    Parameters
    ----------
    equity_df     : DataFrame with columns [equity, cash, drawdown, ...]
                    indexed by timestamp
    trade_log     : DataFrame of trade records
    initial_capital : float
    strategy_name : str
    output_dir    : where to save the PNG (defaults to current dir)
    """

    def __init__(
        self,
        equity_df:       pd.DataFrame,
        trade_log:       pd.DataFrame,
        initial_capital: float,
        strategy_name:   str  = "Strategy",
        risk_free_rate:  float = 0.0,
        output_dir:      str  = ".",
    ):
        self.equity_df       = equity_df
        self.trade_log       = trade_log
        self.initial_capital = initial_capital
        self.strategy_name   = strategy_name
        self.rfr             = risk_free_rate
        self.output_dir      = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._report = full_report(equity_df, trade_log, initial_capital, risk_free_rate)

    # ─────────────────────────────────────────────
    # Text summary
    # ─────────────────────────────────────────────

    def print_summary(self) -> None:
        w = 52
        print("\n" + "═" * w)
        print(f"  {self.strategy_name:^{w-4}}")
        print("═" * w)
        for k, v in self._report.items():
            print(f"  {k:<24} {str(v):>24}")
        print("═" * w + "\n")

    def to_csv(self, filename: str = "performance_summary.csv") -> Path:
        path = self.output_dir / filename
        pd.Series(self._report).to_csv(path, header=False)
        return path

    # ─────────────────────────────────────────────
    # Charts
    # ─────────────────────────────────────────────

    def plot(self, filename: str = "backtest_report.png", dpi: int = 150) -> Path:
        ec      = self.equity_df["equity"]
        returns = ec.pct_change().dropna()
        dd      = drawdown_series(ec)
        roll_sh = rolling_sharpe(returns, window=min(63, max(10, len(returns)//4)))

        sells = self.trade_log[self.trade_log["side"] == "SELL"] if not self.trade_log.empty else pd.DataFrame()

        fig = plt.figure(figsize=(16, 12), facecolor=DARK_BG)
        fig.suptitle(
            f"{self.strategy_name} — Backtest Report",
            color=TEXT_COL, fontsize=14, fontweight="bold", y=0.98
        )

        gs = gridspec.GridSpec(
            3, 2,
            figure=fig,
            hspace=0.45, wspace=0.30,
            left=0.07, right=0.96, top=0.93, bottom=0.06,
        )

        # ── Panel 1: Equity Curve ─────────────────
        ax1 = fig.add_subplot(gs[0, :])
        _style_ax(ax1)
        ax1.plot(ec.index, ec.values, color=BLUE, linewidth=1.5, label="Portfolio Equity")
        ax1.fill_between(ec.index, self.initial_capital, ec.values,
                         where=(ec.values >= self.initial_capital),
                         alpha=0.15, color=GREEN)
        ax1.fill_between(ec.index, self.initial_capital, ec.values,
                         where=(ec.values < self.initial_capital),
                         alpha=0.15, color=RED)
        ax1.axhline(self.initial_capital, color=SPINE_COL, linewidth=0.8, linestyle="--")
        ax1.yaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
        ax1.set_title("Equity Curve", fontsize=10)
        ax1.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)

        # ── Panel 2: Drawdown ─────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        _style_ax(ax2)
        ax2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.5)
        ax2.plot(dd.index, dd.values, color=RED, linewidth=0.8)
        ax2.yaxis.set_major_formatter(FuncFormatter(_pct_fmt))
        ax2.set_title("Drawdown", fontsize=10)

        # ── Panel 3: Rolling Sharpe ───────────────
        ax3 = fig.add_subplot(gs[1, 1])
        _style_ax(ax3)
        ax3.plot(roll_sh.index, roll_sh.values, color=AMBER, linewidth=1.2)
        ax3.axhline(0, color=SPINE_COL, linewidth=0.8, linestyle="--")
        ax3.axhline(1, color=GREEN,     linewidth=0.6, linestyle=":", alpha=0.6)
        ax3.set_title("Rolling Sharpe (63-day)", fontsize=10)

        # ── Panel 4: Trade PnL Distribution ──────
        ax4 = fig.add_subplot(gs[2, 0])
        _style_ax(ax4)
        if not sells.empty and "pnl" in sells.columns:
            pnl = sells["pnl"].dropna()
            bins = min(50, max(10, len(pnl)//3))
            colors = [GREEN if v >= 0 else RED for v in pnl]
            ax4.hist(pnl, bins=bins, color=BLUE, edgecolor=DARK_BG, linewidth=0.3, alpha=0.8)
            ax4.axvline(0, color=TEXT_COL, linewidth=0.8, linestyle="--")
            ax4.axvline(pnl.mean(), color=AMBER, linewidth=1.0, linestyle="-", label=f"Mean: ${pnl.mean():.0f}")
            ax4.xaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
            ax4.set_title("Trade PnL Distribution", fontsize=10)
            ax4.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_COL, framealpha=0.5)
        else:
            ax4.text(0.5, 0.5, "No trade data", ha="center", va="center",
                     color=TEXT_COL, transform=ax4.transAxes)
            ax4.set_title("Trade PnL Distribution", fontsize=10)

        # ── Panel 5: Monthly Returns Heatmap ──────
        ax5 = fig.add_subplot(gs[2, 1])
        _style_ax(ax5)
        self._monthly_heatmap(ax5, ec)

        path = self.output_dir / filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        return path

    def _monthly_heatmap(self, ax, equity_curve: pd.Series) -> None:
        monthly = equity_curve.resample("ME").last().pct_change().dropna()
        if monthly.empty:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    color=TEXT_COL, transform=ax.transAxes)
            ax.set_title("Monthly Returns", fontsize=10)
            return

        df = pd.DataFrame({
            "year":  monthly.index.year,
            "month": monthly.index.month,
            "ret":   monthly.values,
        })
        pivot = df.pivot_table(index="year", columns="month", values="ret", aggfunc="sum")
        pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]

        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                       vmin=-0.10, vmax=0.10)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=6, color=TEXT_COL)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=6, color=TEXT_COL)
        ax.set_title("Monthly Returns Heatmap", fontsize=10)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                            fontsize=5, color="white", fontweight="bold")
