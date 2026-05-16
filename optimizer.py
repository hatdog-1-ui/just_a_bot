"""Grid-search optimizer: finds best strategy + risk parameters via backtest."""

import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from backtester import Backtester, BacktestResult
from config import BotConfig


@dataclass
class OptimResult:
    rsi_oversold:    float
    rsi_overbought:  float
    bb_std:          float
    stop_loss_pct:   float
    take_profit_pct: float
    sharpe:          float
    total_return:    float
    win_rate:        float
    num_trades:      int


# Parameter search space
_GRID = {
    "rsi_oversold":    [25, 30, 35, 40],
    "rsi_overbought":  [60, 65, 70, 75],
    "bb_std":          [1.5, 2.0, 2.5],
    "stop_loss_pct":   [0.01, 0.02, 0.03],
    "take_profit_pct": [0.02, 0.04, 0.06],
}

_MIN_TRADES = 5  # discard combos with too few trades to be meaningful


def optimize(df: pd.DataFrame, initial_capital: float = 10_000.0) -> OptimResult:
    """
    Exhaustive grid search over _GRID. Returns the combo with the best
    Sharpe ratio among runs that produced at least _MIN_TRADES trades.
    """
    keys   = list(_GRID.keys())
    values = list(_GRID.values())
    best: Optional[OptimResult] = None

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))

        # Skip nonsensical RSI combos
        if params["rsi_oversold"] >= params["rsi_overbought"]:
            continue
        # Require at least 1:1 R/R
        if params["take_profit_pct"] <= params["stop_loss_pct"]:
            continue

        cfg = BotConfig()
        cfg.strategy.rsi_oversold    = params["rsi_oversold"]
        cfg.strategy.rsi_overbought  = params["rsi_overbought"]
        cfg.strategy.bb_std          = params["bb_std"]
        cfg.risk.stop_loss_pct       = params["stop_loss_pct"]
        cfg.risk.take_profit_pct     = params["take_profit_pct"]

        result = Backtester(cfg).run(df.copy(), initial_capital=initial_capital)

        if result.num_trades < _MIN_TRADES:
            continue

        candidate = OptimResult(
            rsi_oversold    = params["rsi_oversold"],
            rsi_overbought  = params["rsi_overbought"],
            bb_std          = params["bb_std"],
            stop_loss_pct   = params["stop_loss_pct"],
            take_profit_pct = params["take_profit_pct"],
            sharpe          = result.sharpe_ratio,
            total_return    = result.total_return_pct,
            win_rate        = result.win_rate_pct,
            num_trades      = result.num_trades,
        )

        if best is None or candidate.sharpe > best.sharpe:
            best = candidate

    if best is None:
        # Fallback to sensible defaults if no combo had enough trades
        best = OptimResult(
            rsi_oversold=35, rsi_overbought=65, bb_std=2.0,
            stop_loss_pct=0.02, take_profit_pct=0.04,
            sharpe=0.0, total_return=0.0, win_rate=0.0, num_trades=0,
        )

    return best
