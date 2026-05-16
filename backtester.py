"""Backtest the RSI+MACD+BB strategy against historical OHLCV data."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from config import BotConfig
from risk_manager import RiskManager, OrderPlan
from strategy import Signal, compute_indicators, generate_signal

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_bar: int
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    side: str  # "long" | "short"
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None  # quote-currency P&L
    pnl_pct: Optional[float] = None


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.Series
    total_return_pct: float
    win_rate_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int
    num_wins: int
    num_losses: int


class Backtester:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.risk_mgr = RiskManager(cfg.risk)

    def run(self, df: pd.DataFrame, initial_capital: float = 10_000.0) -> BacktestResult:
        """
        Walk through *df* bar by bar, generating signals and simulating trades.
        Exits are triggered by SL/TP on the *next* bar open (conservative fill).
        """
        df = compute_indicators(df, self.cfg.strategy)
        df = df.dropna().reset_index(drop=True)

        capital = initial_capital
        equity = []
        trades: List[Trade] = []
        active: Optional[Trade] = None

        for i in range(1, len(df)):
            bar = df.iloc[i]
            equity.append(capital + (self._unrealised(active, bar["close"]) if active else 0))

            # --- check exit conditions on active trade ---
            if active is not None:
                exit_price, reason = self._check_exit(active, bar)
                if exit_price is not None:
                    pnl = (exit_price - active.entry_price) * active.quantity
                    if active.side == "short":
                        pnl = -pnl
                    active.exit_bar = i
                    active.exit_price = exit_price
                    active.pnl = pnl
                    active.pnl_pct = pnl / (active.entry_price * active.quantity) * 100
                    capital += active.quantity * active.entry_price + pnl  # return cost basis + profit
                    logger.debug("Exit [%s] @ %.4f  PnL=%.2f  reason=%s", active.side, exit_price, pnl, reason)
                    trades.append(active)
                    active = None
                else:
                    continue  # stay in trade, no new entries while position open

            # --- look for new entry (need at least 2 bars for crossover) ---
            if i < 2:
                continue

            window = df.iloc[: i + 1]
            result = generate_signal(window, self.cfg.strategy)

            if result.signal == Signal.BUY:
                plan = self.risk_mgr.plan_long(bar["close"], capital)
                if plan.position_value > capital:
                    continue  # not enough capital
                capital -= plan.position_value
                active = Trade(
                    entry_bar=i,
                    entry_price=plan.entry_price,
                    stop_loss=plan.stop_loss,
                    take_profit=plan.take_profit,
                    quantity=plan.quantity,
                    side="long",
                )
                logger.debug("Enter LONG @ %.4f  qty=%.6f  SL=%.4f  TP=%.4f",
                             plan.entry_price, plan.quantity, plan.stop_loss, plan.take_profit)

            elif result.signal == Signal.SELL:
                # Short side (only if exchange supports it; safe to model here)
                plan = self.risk_mgr.plan_short(bar["close"], capital)
                if plan.position_value > capital:
                    continue
                capital -= plan.position_value
                active = Trade(
                    entry_bar=i,
                    entry_price=plan.entry_price,
                    stop_loss=plan.stop_loss,
                    take_profit=plan.take_profit,
                    quantity=plan.quantity,
                    side="short",
                )

        # Close any open trade at last price
        if active is not None:
            last_price = df.iloc[-1]["close"]
            pnl = (last_price - active.entry_price) * active.quantity
            if active.side == "short":
                pnl = -pnl
            active.exit_bar = len(df) - 1
            active.exit_price = last_price
            active.pnl = pnl
            active.pnl_pct = pnl / (active.entry_price * active.quantity) * 100
            capital += active.quantity * active.entry_price + pnl
            trades.append(active)

        equity.append(capital)
        equity_series = pd.Series(equity)

        return self._compute_metrics(trades, equity_series, initial_capital)

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _unrealised(trade: Trade, current_price: float) -> float:
        if trade.side == "long":
            return (current_price - trade.entry_price) * trade.quantity
        return (trade.entry_price - current_price) * trade.quantity

    @staticmethod
    def _check_exit(trade: Trade, bar: pd.Series):
        """
        Return (exit_price, reason) if SL or TP is triggered by this bar,
        else (None, None).  We use bar high/low to check intrabar touches.
        """
        if trade.side == "long":
            if bar["low"] <= trade.stop_loss:
                return trade.stop_loss, "SL"
            if bar["high"] >= trade.take_profit:
                return trade.take_profit, "TP"
        else:
            if bar["high"] >= trade.stop_loss:
                return trade.stop_loss, "SL"
            if bar["low"] <= trade.take_profit:
                return trade.take_profit, "TP"
        return None, None

    @staticmethod
    def _compute_metrics(trades: List[Trade], equity: pd.Series, initial_capital: float) -> BacktestResult:
        pnls = [t.pnl for t in trades if t.pnl is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_return = (equity.iloc[-1] - initial_capital) / initial_capital * 100
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0.0

        # Sharpe ratio (annualised, assume hourly bars)
        returns = equity.pct_change().dropna()
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(8760)
        else:
            sharpe = 0.0

        # Max drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        max_dd = drawdown.min()

        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            total_return_pct=round(total_return, 2),
            win_rate_pct=round(win_rate, 2),
            sharpe_ratio=round(sharpe, 3),
            max_drawdown_pct=round(max_dd, 2),
            num_trades=len(pnls),
            num_wins=len(wins),
            num_losses=len(losses),
        )

    @staticmethod
    def print_report(result: BacktestResult) -> None:
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"  Trades:         {result.num_trades}")
        print(f"  Wins / Losses:  {result.num_wins} / {result.num_losses}")
        print(f"  Win Rate:       {result.win_rate_pct:.1f}%")
        print(f"  Total Return:   {result.total_return_pct:.2f}%")
        print(f"  Sharpe Ratio:   {result.sharpe_ratio:.3f}")
        print(f"  Max Drawdown:   {result.max_drawdown_pct:.2f}%")
        print("=" * 50)
