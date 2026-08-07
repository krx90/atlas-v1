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

import math
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


# --------------------------------------------------------------------------
# Intraday
# --------------------------------------------------------------------------
#
# Kronos was pretrained predominantly on intraday K-lines, and the daily
# backtest found no signal (mean IC -0.008, t = -0.14). Testing it on the data
# it was actually trained on is the natural next experiment, so intraday bars
# get their own cache alongside the daily one.

#: Supported intraday timeframes and their regular-session bar count, as
#: measured against live Alpaca data -- not as arithmetic on session length.
#:
#: `1Hour` is 6, not the 7 that 09:30-16:00 suggests, because Alpaca aligns
#: hourly bars to the *clock hour*: a session returns 09:00, 10:00 ... 15:00.
#: The 09:00 bar spans 09:00-10:00, so half of it is premarket, and
#: `regular_session` drops it. That loses the opening 09:30-10:00 window
#: entirely -- a real cost of this timeframe, and the reason 30Min (which lands
#: exactly on 09:30) is the better-behaved intraday choice.
INTRADAY_TIMEFRAMES = {"5Min": 78, "15Min": 26, "30Min": 13, "1Hour": 6}

#: Extra calendar days fetched beyond the arithmetic minimum, to absorb market
#: holidays and half-days. Roughly 7/5 covers weekends; this is on top.
INTRADAY_FETCH_SLACK = 1.25

#: Regular US session in exchange-local time. Alpaca returns extended-hours
#: bars by default and they are ~59% of the rows -- thin, wide-spread, and
#: nothing like session bars. Feeding them to the model would mean most of its
#: context is noise, so they are filtered out on the way in.
SESSION_OPEN = "09:30"
SESSION_LAST = "15:55"


def _timeframe(label: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # noqa: PLC0415

    if label == "1Hour":
        return TimeFrame.Hour
    minutes = int(label.replace("Min", ""))
    return TimeFrame(minutes, TimeFrameUnit.Minute)


def fetch_days_for(timeframe: str, bars_needed: int) -> int:
    """Calendar days to request so `bars_needed` regular-session bars come back.

    Fetching is specified in calendar days but the model needs a count of
    *session* bars, and the two differ by weekends and holidays. Getting this
    wrong is quiet: too short a window returns fewer bars than the lookback
    needs and every symbol is skipped as "insufficient history" without ever
    saying the fetch was the problem.
    """
    per_session = INTRADAY_TIMEFRAMES[timeframe]
    sessions = math.ceil(bars_needed / per_session)
    return int(math.ceil(sessions * (7 / 5) * INTRADAY_FETCH_SLACK)) + 5


def bars_per_day(frame: pd.DataFrame) -> float:
    """Mean regular-session bars per trading day in `frame`.

    The cheapest check that session filtering actually happened: compare it
    against `INTRADAY_TIMEFRAMES`. Unfiltered data runs 04:00-20:00 and returns
    roughly 2.5x as many bars, which otherwise just looks like generous history.
    """
    if frame.empty:
        return 0.0
    index = frame.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    local = index.tz_convert("America/New_York")
    return len(frame) / max(1, len(set(local.date)))


def regular_session(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-session bars, indexed in UTC."""
    if frame.empty:
        return frame
    idx = frame.index
    if idx.tz is None:
        frame = frame.tz_localize("UTC")
    local = frame.tz_convert("America/New_York")
    return local.between_time(SESSION_OPEN, SESSION_LAST).tz_convert("UTC")


def fetch_intraday(
    symbols: list[str], timeframe: str, start: datetime, end: datetime | None = None
) -> dict[str, pd.DataFrame]:
    """Fetch intraday bars, regular session only."""
    if timeframe not in INTRADAY_TIMEFRAMES:
        raise ValueError(f"unsupported timeframe {timeframe!r}")
    end = end or alpaca_client.data_end()
    client = alpaca_client.market_data()
    data_feed = alpaca_client.feed()
    out: dict[str, pd.DataFrame] = {}

    # Fewer symbols per request than daily: an intraday range returns two
    # orders of magnitude more rows, and the SDK paginates each one.
    chunk_size = max(1, config.BARS_SYMBOLS_PER_REQUEST // 10)
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i : i + chunk_size]
        request = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=_timeframe(timeframe),
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
            sub = regular_session(frame.xs(symbol, level=0).copy())
            if sub.empty:
                continue
            for column in _COLUMNS:
                if column not in sub.columns:
                    sub[column] = None
            out[str(symbol)] = sub[_COLUMNS]
    return out


def sync_intraday(
    conn: sqlite3.Connection,
    symbols: list[str],
    timeframe: str,
    *,
    days: int,
    progress=None,
) -> tuple[int, int]:
    """Fill the intraday cache for `symbols`. Returns (rows written, symbols)."""
    end = alpaca_client.data_end()
    start = end - timedelta(days=days)
    rows_written = fetched = 0

    for symbol in symbols:
        row = conn.execute(
            "SELECT MAX(ts) AS last FROM intraday_bars WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe),
        ).fetchone()
        symbol_start = start
        if row and row["last"]:
            cached = datetime.fromisoformat(row["last"])
            if cached.tzinfo is None:
                cached = cached.replace(tzinfo=timezone.utc)
            symbol_start = max(start, cached + timedelta(minutes=1))
        if symbol_start >= end:
            if progress is not None:
                progress(symbol)
            continue

        frames = fetch_intraday([symbol], timeframe, symbol_start, end)
        frame = frames.get(symbol)
        if frame is not None and not frame.empty:
            rows = [
                (
                    symbol,
                    timeframe,
                    ts.isoformat(),
                    float(r.open),
                    float(r.high),
                    float(r.low),
                    float(r.close),
                    float(r.volume),
                    None if pd.isna(r.trade_count) else float(r.trade_count),
                    None if pd.isna(r.vwap) else float(r.vwap),
                )
                for ts, r in frame.iterrows()
                if not any(pd.isna(getattr(r, c)) for c in ("open", "high", "low", "close"))
            ]
            conn.executemany(
                "INSERT INTO intraday_bars (symbol, timeframe, ts, open, high, low, close, "
                "volume, trade_count, vwap) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(symbol, timeframe, ts) DO UPDATE SET "
                "  open=excluded.open, high=excluded.high, low=excluded.low, "
                "  close=excluded.close, volume=excluded.volume, "
                "  trade_count=excluded.trade_count, vwap=excluded.vwap",
                rows,
            )
            rows_written += len(rows)
            fetched += 1
        conn.commit()
        if progress is not None:
            progress(symbol)

    return rows_written, fetched


def load_intraday(
    conn: sqlite3.Connection, symbol: str, timeframe: str, limit: int | None = None
) -> pd.DataFrame:
    """Load cached intraday bars, oldest first."""
    query = (
        "SELECT ts, open, high, low, close, volume, trade_count, vwap "
        "FROM intraday_bars WHERE symbol = ? AND timeframe = ?"
    )
    params: list = [symbol, timeframe]
    if limit:
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
    else:
        query += " ORDER BY ts ASC"

    frame = pd.read_sql_query(query, conn, params=params)
    if frame.empty:
        return frame
    if limit:
        frame = frame.iloc[::-1].reset_index(drop=True)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame.set_index("ts")


def intraday_counts(conn: sqlite3.Connection, symbols: list[str], timeframe: str) -> dict[str, int]:
    placeholders = ",".join("?" * len(symbols))
    rows = conn.execute(
        f"SELECT symbol, COUNT(*) AS n FROM intraday_bars "
        f"WHERE timeframe = ? AND symbol IN ({placeholders}) GROUP BY symbol",
        [timeframe, *symbols],
    ).fetchall()
    return {r["symbol"]: r["n"] for r in rows}
