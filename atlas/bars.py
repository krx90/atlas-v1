"""Daily OHLCV cache.

First run seeds `HISTORY_YEARS` of history for the universe; every run after
that fetches only the sessions since each symbol's last cached date. Bars are
requested with `adjustment=all`, so prices are split- and dividend-adjusted and
therefore directly comparable across the whole window -- which is what the
model needs.

The catch with adjusted data is that a corporate action rewrites history
retroactively, so an incrementally-topped-up cache slowly drifts out of line
with the adjusted series. `reseed=True` (used by the 30-day universe rebuild)
discards and refetches everything to correct that.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import Adjustment
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from . import alpaca_client, config, db

_COLUMNS = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]


def _session_dates(index: pd.DatetimeIndex) -> pd.Index:
    """Map bar timestamps to their exchange-local session date.

    Alpaca stamps daily bars at the session's opening instant in UTC, so a
    naive `.date()` lands on the previous calendar day for part of the year.
    """
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert("America/New_York").date


def fetch(symbols: list[str], start: datetime, end: datetime | None = None) -> dict[str, pd.DataFrame]:
    """Fetch daily bars for `symbols`, chunked to keep request URLs sane.

    alpaca-py follows `next_page_token` internally, so a chunk returns its full
    range in one call from our point of view.
    """
    end = end or alpaca_client.data_end()
    client = alpaca_client.market_data()
    data_feed = alpaca_client.feed()
    out: dict[str, pd.DataFrame] = {}

    for i in range(0, len(symbols), config.BARS_SYMBOLS_PER_REQUEST):
        chunk = symbols[i : i + config.BARS_SYMBOLS_PER_REQUEST]
        request = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment=Adjustment.ALL,
            feed=data_feed,
        )
        barset = alpaca_client.call(client.get_stock_bars, request)
        frame = barset.df
        if frame is None or frame.empty:
            continue

        for symbol in frame.index.get_level_values(0).unique():
            sub = frame.xs(symbol, level=0).copy()
            sub.index = _session_dates(sub.index)
            sub.index.name = "date"
            for column in _COLUMNS:
                if column not in sub.columns:
                    sub[column] = None
            out[str(symbol)] = sub[_COLUMNS]

    return out


def _upsert(conn: sqlite3.Connection, symbol: str, frame: pd.DataFrame) -> int:
    rows = [
        (
            symbol,
            str(date),
            float(r.open),
            float(r.high),
            float(r.low),
            float(r.close),
            float(r.volume),
            None if pd.isna(r.trade_count) else float(r.trade_count),
            None if pd.isna(r.vwap) else float(r.vwap),
        )
        for date, r in frame.iterrows()
        if not any(pd.isna(getattr(r, c)) for c in ("open", "high", "low", "close", "volume"))
    ]
    conn.executemany(
        "INSERT INTO bars (symbol, date, open, high, low, close, volume, trade_count, vwap) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(symbol, date) DO UPDATE SET "
        "  open=excluded.open, high=excluded.high, low=excluded.low, "
        "  close=excluded.close, volume=excluded.volume, "
        "  trade_count=excluded.trade_count, vwap=excluded.vwap",
        rows,
    )
    return len(rows)


def last_dates(conn: sqlite3.Connection, symbols: list[str]) -> dict[str, str]:
    placeholders = ",".join("?" * len(symbols))
    rows = conn.execute(
        f"SELECT symbol, MAX(date) AS last FROM bars WHERE symbol IN ({placeholders}) "
        "GROUP BY symbol",
        symbols,
    ).fetchall()
    return {r["symbol"]: r["last"] for r in rows}


def sync(
    conn: sqlite3.Connection,
    symbols: list[str],
    *,
    reseed: bool = False,
    progress=None,
) -> tuple[int, int]:
    """Bring the cache up to date. Returns (rows written, symbols fetched).

    Symbols are grouped by their last cached date so that everything needing
    the same window travels in one multi-symbol request -- on a delta run that
    is usually a single group covering the whole universe.
    """
    end = alpaca_client.data_end()
    seed_start = end - timedelta(days=int(365.25 * config.HISTORY_YEARS))

    if reseed:
        groups: dict[datetime, list[str]] = {seed_start: list(symbols)}
    else:
        cached = last_dates(conn, symbols)
        groups = {}
        for symbol in symbols:
            last = cached.get(symbol)
            if last is None:
                start = seed_start
            else:
                start = datetime.fromisoformat(last).replace(tzinfo=timezone.utc) + timedelta(days=1)
                if start >= end:
                    continue  # already current
            groups.setdefault(start, []).append(symbol)

    rows_written = 0
    fetched = 0
    for start, group in groups.items():
        frames = fetch(group, start, end)
        for symbol, frame in frames.items():
            rows_written += _upsert(conn, symbol, frame)
            fetched += 1
            if progress is not None:
                progress(symbol)
    return rows_written, fetched


def load(conn: sqlite3.Connection, symbol: str, limit: int | None = None) -> pd.DataFrame:
    """Load a symbol's cached bars, oldest first, indexed by a UTC timestamp."""
    query = "SELECT date, open, high, low, close, volume, trade_count, vwap FROM bars WHERE symbol = ?"
    params: list = [symbol]
    if limit:
        # Take the newest `limit` rows, then restore chronological order.
        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)
    else:
        query += " ORDER BY date ASC"

    frame = pd.read_sql_query(query, conn, params=params)
    if frame.empty:
        return frame
    if limit:
        frame = frame.iloc[::-1].reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")


def bar_counts(conn: sqlite3.Connection, symbols: list[str]) -> dict[str, int]:
    placeholders = ",".join("?" * len(symbols))
    rows = conn.execute(
        f"SELECT symbol, COUNT(*) AS n FROM bars WHERE symbol IN ({placeholders}) GROUP BY symbol",
        symbols,
    ).fetchall()
    return {r["symbol"]: r["n"] for r in rows}
