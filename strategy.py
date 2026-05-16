"""Signal generation: RSI + MACD + Bollinger Bands."""

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import pandas_ta as ta

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
    """
    Attach RSI, MACD, Bollinger Bands, and volume MA columns to *df*.
    Expects columns: open, high, low, close, volume (all lowercase).
    """
    df = df.copy()

    df["rsi"] = ta.rsi(df["close"], length=cfg.rsi_period)

    macd_df = ta.macd(
        df["close"],
        fast=cfg.macd_fast,
        slow=cfg.macd_slow,
        signal=cfg.macd_signal,
    )
    df["macd"] = macd_df[f"MACD_{cfg.macd_fast}_{cfg.macd_slow}_{cfg.macd_signal}"]
    df["macd_signal"] = macd_df[f"MACDs_{cfg.macd_fast}_{cfg.macd_slow}_{cfg.macd_signal}"]
    df["macd_hist"] = macd_df[f"MACDh_{cfg.macd_fast}_{cfg.macd_slow}_{cfg.macd_signal}"]

    bb_df = ta.bbands(df["close"], length=cfg.bb_period, std=cfg.bb_std)
    df["bb_upper"] = bb_df[f"BBU_{cfg.bb_period}_{cfg.bb_std}"]
    df["bb_mid"] = bb_df[f"BBM_{cfg.bb_period}_{cfg.bb_std}"]
    df["bb_lower"] = bb_df[f"BBL_{cfg.bb_period}_{cfg.bb_std}"]

    df["volume_ma"] = df["volume"].rolling(cfg.volume_ma_period).mean()

    return df


def _macd_bullish_cross(prev_hist: float, curr_hist: float) -> bool:
    """Histogram crosses from negative to positive (momentum flip)."""
    return prev_hist < 0 and curr_hist >= 0


def _macd_bearish_cross(prev_hist: float, curr_hist: float) -> bool:
    return prev_hist > 0 and curr_hist <= 0


def generate_signal(df: pd.DataFrame, cfg: StrategyConfig) -> SignalResult:
    """
    Evaluate the two most recent complete bars and return a trading signal.
    *df* must already have indicator columns (call compute_indicators first).
    """
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
