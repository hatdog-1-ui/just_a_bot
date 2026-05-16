import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExchangeConfig:
    exchange_id: str = "binance"
    api_key: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("EXCHANGE_API_SECRET", ""))
    sandbox: bool = True  # Use exchange testnet when available


@dataclass
class StrategyConfig:
    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # Volume filter: price bar volume must exceed N-period average
    volume_ma_period: int = 20

    # How close to band edge counts as "near" (fraction of band width)
    bb_proximity: float = 0.1


@dataclass
class RiskConfig:
    # Maximum fraction of capital risked per trade
    max_risk_per_trade: float = 0.01  # 1%

    # Stop-loss / take-profit relative to entry price
    stop_loss_pct: float = 0.02   # 2% below entry
    take_profit_pct: float = 0.04  # 4% above entry (2:1 R/R)

    # Never commit more than this fraction of portfolio to a single position
    max_position_pct: float = 0.20


@dataclass
class BotConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    paper_trading: bool = True

    # How many closed candles to seed indicators on startup
    warmup_bars: int = 100

    # Seconds to sleep between polling cycles (used in paper-trade loop)
    poll_interval: int = 60

    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


# Singleton used throughout the project
CONFIG = BotConfig()
