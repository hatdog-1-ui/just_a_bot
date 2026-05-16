"""
Main entry point.

Usage
-----
Paper trading (default):
    python bot.py

Backtest:
    python bot.py --backtest

Live trading (real orders):
    python bot.py --live
"""

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Optional

from config import CONFIG
from exchange import ExchangeClient
from risk_manager import RiskManager, OrderPlan
from strategy import Signal, compute_indicators, generate_signal
from backtester import Backtester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- paper state

@dataclass
class PaperPosition:
    side: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float


class PaperPortfolio:
    def __init__(self, initial_capital: float = 10_000.0):
        self.capital = initial_capital
        self.position: Optional[PaperPosition] = None
        self.trades = []

    def open_long(self, plan: OrderPlan) -> None:
        if self.position:
            return
        if plan.position_value > self.capital:
            logger.warning("Not enough capital for trade.")
            return
        self.capital -= plan.position_value
        self.position = PaperPosition(
            side="long",
            entry_price=plan.entry_price,
            quantity=plan.quantity,
            stop_loss=plan.stop_loss,
            take_profit=plan.take_profit,
        )
        logger.info(
            "[PAPER] OPEN LONG  qty=%.6f  entry=%.4f  SL=%.4f  TP=%.4f",
            plan.quantity, plan.entry_price, plan.stop_loss, plan.take_profit,
        )

    def close_position(self, exit_price: float, reason: str) -> None:
        if not self.position:
            return
        pos = self.position
        pnl = (exit_price - pos.entry_price) * pos.quantity
        if pos.side == "short":
            pnl = -pnl
        self.capital += pos.quantity * pos.entry_price + pnl
        self.trades.append({"reason": reason, "pnl": pnl})
        logger.info(
            "[PAPER] CLOSE %s  exit=%.4f  PnL=%.4f USDT  capital=%.2f  reason=%s",
            pos.side.upper(), exit_price, pnl, self.capital, reason,
        )
        self.position = None

    def check_exit(self, current_price: float) -> None:
        if not self.position:
            return
        pos = self.position
        if pos.side == "long":
            if current_price <= pos.stop_loss:
                self.close_position(pos.stop_loss, "STOP_LOSS")
            elif current_price >= pos.take_profit:
                self.close_position(pos.take_profit, "TAKE_PROFIT")

    @property
    def equity(self) -> float:
        return self.capital


# --------------------------------------------------------------------- loops

def paper_trading_loop(client: ExchangeClient) -> None:
    portfolio = PaperPortfolio(initial_capital=10_000.0)
    risk_mgr = RiskManager(CONFIG.risk)
    logger.info("Paper trading started. Symbol=%s  TF=%s", CONFIG.symbol, CONFIG.timeframe)

    while True:
        try:
            df = client.fetch_ohlcv(CONFIG.symbol, CONFIG.timeframe, limit=CONFIG.warmup_bars + 10)
            df = compute_indicators(df, CONFIG.strategy)
            df = df.dropna()

            if len(df) < 2:
                logger.warning("Not enough bars after indicator warmup.")
                time.sleep(CONFIG.poll_interval)
                continue

            current_price = df.iloc[-1]["close"]
            portfolio.check_exit(current_price)

            if portfolio.position is None:
                result = generate_signal(df, CONFIG.strategy)
                logger.info(
                    "Signal=%-4s  price=%.4f  RSI=%.1f  MACD_hist=%.6f  vol_ok=%s",
                    result.signal.value, result.close, result.rsi, result.macd_hist, result.volume_ok,
                )
                if result.signal == Signal.BUY:
                    plan = risk_mgr.plan_long(current_price, portfolio.equity)
                    portfolio.open_long(plan)

            time.sleep(CONFIG.poll_interval)

        except KeyboardInterrupt:
            logger.info("Shutting down. Final equity=%.2f USDT", portfolio.equity)
            break
        except Exception as exc:
            logger.exception("Unexpected error: %s", exc)
            time.sleep(CONFIG.poll_interval)


def live_trading_loop(client: ExchangeClient) -> None:
    """
    Executes real orders. Enable only after thorough backtesting.
    Currently mirrors paper logic but calls the real exchange API.
    """
    risk_mgr = RiskManager(CONFIG.risk)
    open_position = None
    logger.warning("LIVE MODE — real orders will be placed.")

    while True:
        try:
            df = client.fetch_ohlcv(CONFIG.symbol, CONFIG.timeframe, limit=CONFIG.warmup_bars + 10)
            df = compute_indicators(df, CONFIG.strategy)
            df = df.dropna()

            current_price = df.iloc[-1]["close"]

            if open_position:
                pos = open_position
                hit_sl = current_price <= pos.stop_loss
                hit_tp = current_price >= pos.take_profit
                if hit_sl or hit_tp:
                    client.place_market_sell(CONFIG.symbol, pos.quantity)
                    open_position = None
            else:
                result = generate_signal(df, CONFIG.strategy)
                balance = client.fetch_balance()
                quote = CONFIG.symbol.split("/")[1]
                available = balance.get("free", {}).get(quote, 0.0)
                if result.signal == Signal.BUY and available > 0:
                    plan = risk_mgr.plan_long(current_price, available)
                    order = client.place_market_buy(CONFIG.symbol, plan.quantity)
                    if order:
                        open_position = PaperPosition(
                            side="long",
                            entry_price=current_price,
                            quantity=plan.quantity,
                            stop_loss=plan.stop_loss,
                            take_profit=plan.take_profit,
                        )

            time.sleep(CONFIG.poll_interval)

        except KeyboardInterrupt:
            logger.info("Live loop stopped by user.")
            break
        except Exception as exc:
            logger.exception("Live loop error: %s", exc)
            time.sleep(CONFIG.poll_interval)


def run_backtest(client: ExchangeClient) -> None:
    logger.info("Fetching historical data for backtest…")
    df = client.fetch_ohlcv(CONFIG.symbol, CONFIG.timeframe, limit=1000)
    backtester = Backtester(CONFIG)
    result = backtester.run(df)
    Backtester.print_report(result)


# ----------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto trading bot")
    parser.add_argument("--backtest", action="store_true", help="Run backtest and exit")
    parser.add_argument("--live", action="store_true", help="Enable live order execution")
    args = parser.parse_args()

    client = ExchangeClient(CONFIG.exchange)

    if args.backtest:
        run_backtest(client)
    elif args.live:
        if CONFIG.paper_trading:
            logger.error("Set paper_trading=False in config before going live.")
            return
        live_trading_loop(client)
    else:
        paper_trading_loop(client)


if __name__ == "__main__":
    main()
