"""Alpaca client construction, rate limiting and feed selection.

One module-level cache per client so a command never builds more than one, and
a token-bucket throttle shared by every data call so the Basic plan's 200
requests/minute is respected without the caller thinking about it.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

from . import config

_RETRYABLE_STATUS = (429, 500, 502, 503, 504)
_MAX_RETRIES = 5


class RateLimiter:
    """Sliding-window throttle. Blocks until a request slot is free."""

    def __init__(self, per_minute: int) -> None:
        self._per_minute = per_minute
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._calls and now - self._calls[0] >= 60.0:
                self._calls.popleft()
            if len(self._calls) >= self._per_minute:
                sleep_for = 60.0 - (now - self._calls[0]) + 0.01
                time.sleep(max(sleep_for, 0.0))
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
            self._calls.append(time.monotonic())


_limiter = RateLimiter(config.RATE_LIMIT_PER_MINUTE)


def _status_of(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def call(fn, *args, **kwargs):
    """Invoke an Alpaca SDK method under the rate limit, retrying transients.

    Retries 429s and 5xx with exponential backoff. Client errors (403 for an
    unsubscribed feed, 404 for an unknown symbol) are raised immediately --
    they will not succeed on a second attempt.
    """
    delay = 1.0
    for attempt in range(_MAX_RETRIES):
        _limiter.acquire()
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 -- re-raised below unless retryable
            status = _status_of(exc)
            if status not in _RETRYABLE_STATUS or attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


@lru_cache(maxsize=1)
def _creds() -> tuple[str, str]:
    # The CLI preflights this and surfaces any permissions warning, so by the
    # time a command reaches here the file has already been validated once.
    creds, _warning = config.load_credentials()
    return creds.key_id, creds.secret_key


@lru_cache(maxsize=1)
def trading() -> TradingClient:
    key, secret = _creds()
    return TradingClient(key, secret, paper=True)


@lru_cache(maxsize=1)
def market_data() -> StockHistoricalDataClient:
    key, secret = _creds()
    return StockHistoricalDataClient(key, secret)


@lru_cache(maxsize=1)
def news() -> NewsClient:
    key, secret = _creds()
    return NewsClient(key, secret)


def data_end() -> datetime:
    """Latest timestamp we may request.

    The Basic plan rejects SIP queries whose `end` falls inside the last 15
    minutes, so everything is clamped behind that window.
    """
    return datetime.now(timezone.utc) - timedelta(minutes=config.DATA_DELAY_MINUTES)


def _is_subscription_error(exc: Exception) -> bool:
    if _status_of(exc) in (401, 403):
        return True
    # Alpaca returns 'subscription does not permit querying recent SIP data'
    # for a real-time SIP query on the Basic plan, and the SDK does not always
    # surface a status code alongside it.
    return "subscription does not permit" in str(exc).lower()


@lru_cache(maxsize=1)
def feed() -> DataFeed:
    """The feed for *historical* bars. Probes SIP once, falls back to IEX.

    SIP covers 100% of consolidated volume; IEX is a single venue carrying a
    small fraction of it, which matters because the liquidity screen ranks on
    dollar volume. Worth one request to find out.
    """
    probe = StockBarsRequest(
        symbol_or_symbols=["AAPL"],
        timeframe=TimeFrame.Day,
        start=data_end() - timedelta(days=7),
        end=data_end(),
        feed=DataFeed.SIP,
    )
    try:
        call(market_data().get_stock_bars, probe)
        return DataFeed.SIP
    except Exception as exc:  # noqa: BLE001
        if _is_subscription_error(exc):
            print("note: no historical SIP entitlement -- using the IEX feed.")
            return DataFeed.IEX
        raise


@lru_cache(maxsize=1)
def realtime_feed() -> DataFeed:
    """The feed for *latest trade/quote* calls -- a separate entitlement.

    The Basic plan serves historical SIP data outside the last 15 minutes but
    refuses SIP for anything real-time, so an account can legitimately be SIP
    for bars and IEX for quotes. Probing `feed()` alone gets this wrong.
    """
    from alpaca.data.requests import StockLatestTradeRequest  # noqa: PLC0415

    try:
        call(
            market_data().get_stock_latest_trade,
            StockLatestTradeRequest(symbol_or_symbols="AAPL", feed=DataFeed.SIP),
        )
        return DataFeed.SIP
    except Exception as exc:  # noqa: BLE001
        if _is_subscription_error(exc):
            return DataFeed.IEX
        raise
