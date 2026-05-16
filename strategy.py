"""Signal generation: RSI + MACD + Bollinger Bands."""

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import ta.momentum as ta_momentum
import ta.trend as ta_trend
import ta.volatility as ta_volatility

from config import StrategyConfig


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class SignalResult:
    signal: Signal
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_upper: float
    bb_lower: float
    bb_mid: float
    close: float
    volume_ok: bool


def compute_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = df.copy()

    df["rsi"] = ta_momentum.RSIIndicator(
        close=df["close"], window=cfg.rsi_period
    ).rsi()

    macd = ta_trend.MACD(
        close=df["close"],
        window_fast=cfg.macd_fast,
        window_slow=cfg.macd_slow,
        window_sign=cfg.macd_signal,
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    bb = ta_volatility.BollingerBands(
        close=df["close"], window=cfg.bb_period, window_dev=cfg.bb_std
    )
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()

    df["volume_ma"] = df["volume"].rolling(cfg.volume_ma_period).mean()

    return df


def _macd_bullish_cross(prev_hist: float, curr_hist: float) -> bool:
    """Histogram crosses from negative to positive (momentum flip)."""
    return prev_hist < 0 and curr_hist >= 0


def _macd_bearish_cross(prev_hist: float, curr_hist: float) -> bool:
    return prev_hist > 0 and curr_hist <= 0


def generate_signal(df: pd.DataFrame, cfg: StrategyConfig) -> SignalResult:
    if len(df) < 2:
        raise ValueError("Need at least 2 bars to detect crossovers.")

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    volume_ok = curr["volume"] > curr["volume_ma"]

    band_width = curr["bb_upper"] - curr["bb_lower"]
    near_lower = curr["close"] <= curr["bb_lower"] + cfg.bb_proximity * band_width
    near_upper = curr["close"] >= curr["bb_upper"] - cfg.bb_proximity * band_width

    bullish_cross = _macd_bullish_cross(prev["macd_hist"], curr["macd_hist"])
    bearish_cross = _macd_bearish_cross(prev["macd_hist"], curr["macd_hist"])

    if (
        curr["rsi"] < cfg.rsi_oversold
        and bullish_cross
        and near_lower
        and volume_ok
    ):
        signal = Signal.BUY
    elif (
        curr["rsi"] > cfg.rsi_overbought
        and bearish_cross
        and near_upper
        and volume_ok
    ):
        signal = Signal.SELL
    else:
        signal = Signal.HOLD

    return SignalResult(
        signal=signal,
        rsi=curr["rsi"],
        macd=curr["macd"],
        macd_signal=curr["macd_signal"],
        macd_hist=curr["macd_hist"],
        bb_upper=curr["bb_upper"],
        bb_lower=curr["bb_lower"],
        bb_mid=curr["bb_mid"],
        close=curr["close"],
        volume_ok=volume_ok,
    )
