"""
WebSocket price feed using Binance public streams.

Runs in a background thread and maintains the latest trade price so the
paper-trading loop can consume it without blocking on REST calls.

Usage
-----
    feed = PriceFeed("BTC/USDT")
    feed.start()
    price = feed.latest_price   # None until first message arrives
    feed.stop()
"""

import json
import logging
import threading
import time
from typing import Optional

import websockets
import asyncio

logger = logging.getLogger(__name__)

# Public Binance WebSocket endpoint (no API key required)
_BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"


def _ccxt_symbol_to_stream(symbol: str) -> str:
    """Convert 'BTC/USDT' -> 'btcusdt@trade'."""
    return symbol.replace("/", "").lower() + "@trade"


class PriceFeed:
    """
    Subscribes to the Binance trade stream for *symbol* and exposes the
    latest executed price via the ``latest_price`` attribute.

    Reconnects automatically on disconnect with exponential back-off.
    """

    def __init__(self, symbol: str, max_reconnect_delay: float = 60.0):
        self.symbol = symbol
        self.max_reconnect_delay = max_reconnect_delay
        self._stream = _ccxt_symbol_to_stream(symbol)
        self._url = f"{_BINANCE_WS_BASE}/{self._stream}"

        self._latest_price: Optional[float] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ public

    @property
    def latest_price(self) -> Optional[float]:
        with self._lock:
            return self._latest_price

    def start(self) -> None:
        """Start the background WebSocket thread (non-blocking)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ws-feed")
        self._thread.start()
        logger.info("PriceFeed started for %s", self.symbol)

    def stop(self) -> None:
        """Signal the background thread to exit and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("PriceFeed stopped.")

    # --------------------------------------------------------------- internals

    def _run(self) -> None:
        """Entry point for the background thread; runs an asyncio event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._listen())
        finally:
            loop.close()

    async def _listen(self) -> None:
        delay = 1.0
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self._url, ping_interval=20) as ws:
                    logger.info("WebSocket connected: %s", self._url)
                    delay = 1.0  # reset back-off on successful connect
                    async for raw in ws:
                        if self._stop_event.is_set():
                            return
                        self._handle_message(raw)
            except Exception as exc:
                if self._stop_event.is_set():
                    return
                logger.warning("WebSocket error (%s). Reconnecting in %.1fs…", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.max_reconnect_delay)

    def _handle_message(self, raw: str) -> None:
        """Parse a Binance trade message and update latest_price."""
        try:
            msg = json.loads(raw)
            # Binance trade stream payload: {"e":"trade","p":"<price>",...}
            if msg.get("e") == "trade":
                price = float(msg["p"])
                with self._lock:
                    self._latest_price = price
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            logger.debug("Unparseable WS message: %s — %s", raw[:120], exc)
