"""
engine.py — The Backtest Engine.

The engine is the orchestrator. It wires every layer together and drives
the event loop.  The loop order per bar is:

  1. Emit MarketEvents for the current bar
  2. Check stop-loss / take-profit on existing positions (risk manager)
  3. Call each strategy's on_bar() (strategy layer)
  4. Collect signals → risk manager → orders (risk/execution layer)
  5. Submit orders to broker
  6. Process bar fills (execution layer)
  7. Update portfolio on fills (portfolio layer)
  8. Mark-to-market equity snapshot (portfolio layer)
  9. Advance to next bar

This strictly prevents lookahead bias:
  • Strategies only see bars up_to the CURRENT timestamp.
  • Orders generated at bar T are filled at bar T+1 open (or later).
  • Stop/TP checks use intrabar high/low (not the next bar).
"""

from __future__ import annotations
import logging
import time
from typing import Dict, List, Optional, Type

import pandas as pd

from algotrader.core import FillEvent, OrderEvent, OrderSide
from algotrader.data.loader import BarFeed
from algotrader.execution.broker import SimulatedBroker, FixedSlippage, PercentCommission
from algotrader.portfolio.portfolio import Portfolio
from algotrader.risk.manager import RiskManager
from algotrader.strategy.base import BaseStrategy
from algotrader.analytics.performance import full_report
from algotrader.reporting.report import BacktestReport

log = logging.getLogger(__name__)


class BacktestEngine:
    """
    Event-driven backtest engine.

    Parameters
    ----------
    feed            : BarFeed wrapping historical OHLCV data
    strategies      : list of BaseStrategy subclass instances
    initial_capital : starting cash
    broker          : SimulatedBroker (optional, uses defaults)
    risk_manager    : RiskManager (optional, uses defaults)
    """

    def __init__(
        self,
        feed:             BarFeed,
        strategies:       List[BaseStrategy],
        initial_capital:  float            = 100_000.0,
        broker:           SimulatedBroker  = None,
        risk_manager:     RiskManager      = None,
        risk_free_rate:   float            = 0.0,
        verbose:          bool             = True,
    ):
        self.feed            = feed
        self.strategies      = strategies
        self.portfolio       = Portfolio(initial_capital)
        self.broker          = broker    or SimulatedBroker()
        self.risk_manager    = risk_manager or RiskManager()
        self.risk_free_rate  = risk_free_rate
        self.verbose         = verbose

        # order_id → strategy_id mapping (for fill attribution)
        self._order_strategy_map: Dict[str, str] = {}

        # Attach feed to all strategies
        for strat in strategies:
            strat.attach_feed(feed)

    # ─────────────────────────────────────────────
    # Main event loop
    # ─────────────────────────────────────────────

    def run(self) -> "BacktestResult":
        log.info("Starting backtest with %d strategy/ies on %d symbol(s)",
                 len(self.strategies), len(self.feed.symbols))
        t0 = time.time()
        bar_count = 0

        for timestamp, market_events in self.feed:
            bar_count += 1

            # ── 1. Check existing stop-loss / take-profit ──────────────
            stop_orders = self.risk_manager.check_stop_conditions(
                self.portfolio.open_positions_detail, market_events
            )
            for order in stop_orders:
                self._submit(order, strategy_id="__risk__")

            # ── 2. Run strategies ──────────────────────────────────────
            all_signals = []
            for strat in self.strategies:
                strat.on_bar(timestamp, market_events)
                all_signals.extend(strat.flush_signals())

            # ── 3. Risk manager: signals → orders ─────────────────────
            orders = self.risk_manager.process_signals(
                signals=all_signals,
                market_events=market_events,
                feed=self.feed,
                equity=self.portfolio.equity,
                open_positions=self.portfolio.open_positions,
            )
            for order in orders:
                sig = next((s for s in all_signals if s.symbol == order.symbol), None)
                self._submit(order, strategy_id=sig.strategy_id if sig else "")

            # ── 4. Execute pending orders against this bar ────────────
            fills = self.broker.process_bar(market_events, self._order_strategy_map)

            # ── 5. Apply fills to portfolio ───────────────────────────
            for fill in fills:
                self.portfolio.on_fill(fill)
                # Notify strategies
                for strat in self.strategies:
                    strat.on_fill(fill)

            # ── 6. Mark-to-market ─────────────────────────────────────
            equity = self.portfolio.mark_to_market(timestamp, market_events)

            if self.verbose and bar_count % 252 == 0:
                log.info("  %s  equity=$%.0f  positions=%d",
                         timestamp.date(), equity, len(self.portfolio.open_positions))

        elapsed = time.time() - t0
        log.info("Backtest complete: %d bars in %.2fs (%.0f bars/s)",
                 bar_count, elapsed, bar_count / max(elapsed, 1e-6))

        return BacktestResult(
            portfolio=self.portfolio,
            feed=self.feed,
            risk_free_rate=self.risk_free_rate,
        )

    def _submit(self, order: OrderEvent, strategy_id: str = "") -> None:
        self._order_strategy_map[order.order_id] = strategy_id
        self.broker.submit(order)


# ─────────────────────────────────────────────
# Result wrapper
# ─────────────────────────────────────────────

class BacktestResult:
    """Holds all post-run artifacts and generates reports."""

    def __init__(
        self,
        portfolio:       Portfolio,
        feed:            BarFeed,
        risk_free_rate:  float = 0.0,
    ):
        self.portfolio      = portfolio
        self.feed           = feed
        self.risk_free_rate = risk_free_rate

        self.equity_curve = portfolio.equity_curve()
        self.trade_log    = portfolio.trade_log()
        self.stats        = portfolio.summary_stats()

    def report(
        self,
        strategy_name: str = "Backtest",
        output_dir:    str = ".",
        save_chart:    bool = True,
    ) -> BacktestReport:
        r = BacktestReport(
            equity_df=self.equity_curve,
            trade_log=self.trade_log,
            initial_capital=self.portfolio.initial_capital,
            strategy_name=strategy_name,
            risk_free_rate=self.risk_free_rate,
            output_dir=output_dir,
        )
        r.print_summary()
        if save_chart:
            path = r.plot()
            log.info("Chart saved → %s", path)
        return r

    def performance(self) -> Dict:
        from algotrader.analytics.performance import full_report
        return full_report(
            self.equity_curve, self.trade_log,
            self.portfolio.initial_capital, self.risk_free_rate,
        )
