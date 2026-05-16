"""Exchange wrapper: REST data fetching and order placement via ccxt."""

import logging
from typing import Optional

import ccxt
import pandas as pd

from config import ExchangeConfig

logger = logging.getLogger(__name__)


class ExchangeClient:
    def __init__(self, cfg: ExchangeConfig):
        exchange_class = getattr(ccxt, cfg.exchange_id)
        params = {
            "apiKey": cfg.api_key,
            "secret": cfg.api_secret,
            "enableRateLimit": True,
        }
        self.exchange: ccxt.Exchange = exchange_class(params)

        if cfg.sandbox and self.exchange.has.get("sandbox"):
            self.exchange.set_sandbox_mode(True)
            logger.info("Sandbox (testnet) mode enabled.")

    # ------------------------------------------------------------------ data

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Return a DataFrame with columns: timestamp, open, high, low, close, volume."""
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    def fetch_ticker(self, symbol: str) -> dict:
        return self.exchange.fetch_ticker(symbol)

    def fetch_balance(self) -> dict:
        return self.exchange.fetch_balance()

    # ----------------------------------------------------------------- orders

    def place_market_buy(self, symbol: str, amount: float) -> Optional[dict]:
        """Place a market buy; returns the order dict or None on failure."""
        try:
            order = self.exchange.create_market_buy_order(symbol, amount)
            logger.info("Market BUY placed: %s", order)
            return order
        except ccxt.BaseError as exc:
            logger.error("Market BUY failed: %s", exc)
            return None

    def place_market_sell(self, symbol: str, amount: float) -> Optional[dict]:
        try:
            order = self.exchange.create_market_sell_order(symbol, amount)
            logger.info("Market SELL placed: %s", order)
            return order
        except ccxt.BaseError as exc:
            logger.error("Market SELL failed: %s", exc)
            return None

    def place_limit_buy(self, symbol: str, amount: float, price: float) -> Optional[dict]:
        try:
            order = self.exchange.create_limit_buy_order(symbol, amount, price)
            logger.info("Limit BUY placed @ %.4f: %s", price, order)
            return order
        except ccxt.BaseError as exc:
            logger.error("Limit BUY failed: %s", exc)
            return None

    def place_limit_sell(self, symbol: str, amount: float, price: float) -> Optional[dict]:
        try:
            order = self.exchange.create_limit_sell_order(symbol, amount, price)
            logger.info("Limit SELL placed @ %.4f: %s", price, order)
            return order
        except ccxt.BaseError as exc:
            logger.error("Limit SELL failed: %s", exc)
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except ccxt.BaseError as exc:
            logger.warning("Could not cancel order %s: %s", order_id, exc)
            return False
