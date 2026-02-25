"""
run_demo.py — End-to-end demo of the AlgoTrader backtesting framework.

Generates synthetic OHLCV data for 3 tickers, runs 2 strategies simultaneously
(MA Crossover + RSI Mean Reversion), and produces a full performance report.

Usage:
    python run_demo.py

No external data files required — synthetic data is generated inline.
"""

import sys
import logging
import numpy as np
import pandas as pd

# ── Setup ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")

sys.path.insert(0, "/home/claude")   # add parent so 'algotrader' is importable


# ─────────────────────────────────────────────
# Synthetic Data Generator
# ─────────────────────────────────────────────

def generate_synthetic_ohlcv(
    n_days:     int   = 1500,
    start_price: float = 100.0,
    annual_vol: float = 0.25,
    annual_drift: float = 0.08,
    seed:       int   = 42,
    symbol:     str   = "SYN",
) -> pd.DataFrame:
    """
    Geometric Brownian Motion with realistic OHLC structure.
    High = close * (1 + |ε_h| * vol_intraday)
    Low  = close * (1 - |ε_l| * vol_intraday)
    Open = prev_close * (1 + ε_open * vol_overnight)
    """
    rng   = np.random.default_rng(seed)
    dt    = 1 / 252
    mu    = (annual_drift - 0.5 * annual_vol**2) * dt
    sigma = annual_vol * np.sqrt(dt)

    log_returns = rng.normal(mu, sigma, n_days)
    closes      = start_price * np.exp(np.cumsum(log_returns))

    intraday_vol  = annual_vol * np.sqrt(dt) * 0.5
    overnight_vol = annual_vol * np.sqrt(dt) * 0.3

    opens  = np.empty(n_days)
    opens[0] = start_price
    opens[1:] = closes[:-1] * (1 + rng.normal(0, overnight_vol, n_days - 1))

    highs  = np.maximum(closes, opens) * (1 + np.abs(rng.normal(0, intraday_vol, n_days)))
    lows   = np.minimum(closes, opens) * (1 - np.abs(rng.normal(0, intraday_vol, n_days)))
    volume = np.abs(rng.normal(1_000_000, 200_000, n_days)).astype(int)

    dates = pd.bdate_range("2018-01-02", periods=n_days)
    return pd.DataFrame({
        "open":   np.round(opens,  4),
        "high":   np.round(highs,  4),
        "low":    np.round(lows,   4),
        "close":  np.round(closes, 4),
        "volume": volume,
    }, index=dates)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    # ── 1. Generate synthetic data ────────────────────────────────
    symbols_cfg = {
        "AAPL_SYN": dict(start_price=150, annual_drift=0.12, annual_vol=0.28, seed=1),
        "MSFT_SYN": dict(start_price=300, annual_drift=0.10, annual_vol=0.22, seed=2),
        "TSLA_SYN": dict(start_price=200, annual_drift=0.15, annual_vol=0.55, seed=3),
    }

    frames = {sym: generate_synthetic_ohlcv(symbol=sym, **cfg)
              for sym, cfg in symbols_cfg.items()}
    symbols = list(frames.keys())
    log.info("Generated synthetic data for: %s", symbols)

    # ── 2. Build data layer ───────────────────────────────────────
    from algotrader.data.loader import DataFrameLoader, BarFeed

    loader = DataFrameLoader(frames)
    data   = loader.load_many(symbols)
    feed   = BarFeed(data)

    # ── 3. Configure strategies ───────────────────────────────────
    from algotrader.strategies.examples import MovingAverageCrossover, RSIMeanReversion

    strategies = [
        MovingAverageCrossover(
            symbols=symbols,
            fast_period=20,
            slow_period=50,
            strategy_id="MA_Cross",
        ),
        RSIMeanReversion(
            symbols=symbols,
            rsi_period=14,
            oversold_level=35,
            overbought_level=65,
            strategy_id="RSI_MR",
        ),
    ]

    # ── 4. Configure execution ────────────────────────────────────
    from algotrader.execution.broker import SimulatedBroker, FixedSlippage, PercentCommission

    broker = SimulatedBroker(
        slippage=FixedSlippage(bps=5),
        commission=PercentCommission(pct=0.001),
        max_bars_pending=1,
    )

    # ── 5. Configure risk management ──────────────────────────────
    from algotrader.risk.manager import RiskManager, ATRPositionSizer

    risk = RiskManager(
        sizer=ATRPositionSizer(
            risk_pct=0.01,
            atr_period=14,
            atr_multiplier=2.0,
            max_position_pct=0.20,
        ),
        stop_atr_multiple=2.0,
        tp_atr_multiple=4.0,
        max_open_positions=6,
        allow_short=False,
    )

    # ── 6. Run backtest ───────────────────────────────────────────
    from algotrader.engine import BacktestEngine

    engine = BacktestEngine(
        feed=feed,
        strategies=strategies,
        initial_capital=100_000.0,
        broker=broker,
        risk_manager=risk,
        risk_free_rate=0.04,
        verbose=True,
    )

    result = engine.run()

    # ── 7. Generate report ────────────────────────────────────────
    report = result.report(
        strategy_name="MA Crossover + RSI Mean Reversion (Multi-Asset)",
        output_dir="/home/claude/output",
        save_chart=True,
    )

    # Save trade log and equity curve
    result.equity_curve.to_csv("/home/claude/output/equity_curve.csv")
    if not result.trade_log.empty:
        result.trade_log.to_csv("/home/claude/output/trade_log.csv", index=False)
    report.to_csv("/home/claude/output/performance_summary.csv")

    log.info("All outputs saved to /home/claude/output/")


if __name__ == "__main__":
    main()
