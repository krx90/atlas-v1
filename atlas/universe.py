"""Investable universe construction.

Three stages, so we never pull five years of history for eleven thousand
symbols:

1. Sweep every active, tradable US equity from Alpaca and drop the structurally
   untradeable ones (OTC venues, warrants, units, rights, preferred lines).
2. Fetch a month of recent bars for the survivors -- cheap, roughly 30
   multi-symbol requests -- and measure median dollar volume.
3. Keep the top `UNIVERSE_SIZE` by dollar volume, subject to floors on price
   and turnover.

The result is cached and rebuilt when it is older than `UNIVERSE_MAX_AGE_DAYS`.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from . import alpaca_client, bars, config, db

_BUILT_AT_KEY = "universe_built_at"

#: Suffixes Alpaca appends for warrants, units, rights and preferred shares.
#: These trade thinly, have their own price dynamics, and are not what a
#: candlestick model pretrained on ordinary equities is good at.
_EXCLUDED_SUFFIX = re.compile(r"(?:\.(?:WS|U|R|RT|W)|[-.]P[A-Z]?)$", re.IGNORECASE)


def _is_plain_equity(symbol: str) -> bool:
    if not symbol or len(symbol) > 5:
        return False
    if any(ch in symbol for ch in "./ +"):
        return False
    return not _EXCLUDED_SUFFIX.search(symbol)


def sweep_assets() -> list:
    """Every active, tradable US equity on a major exchange."""
    request = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    assets = alpaca_client.call(alpaca_client.trading().get_all_assets, request)
    return [
        a
        for a in assets
        if a.tradable
        and str(getattr(a.exchange, "value", a.exchange)) in config.ALLOWED_EXCHANGES
        and _is_plain_equity(a.symbol)
    ]


def _liquidity(symbols: list[str]) -> dict[str, tuple[float, float]]:
    """Median dollar volume and last close over the recent window."""
    end = alpaca_client.data_end()
    start = end - timedelta(days=config.LIQUIDITY_LOOKBACK_DAYS * 2)  # calendar vs trading days
    frames = bars.fetch(symbols, start, end)

    out: dict[str, tuple[float, float]] = {}
    for symbol, frame in frames.items():
        frame = frame.dropna(subset=["close", "volume"])
        if frame.empty:
            continue
        dollar_volume = float(np.median(frame["close"].to_numpy() * frame["volume"].to_numpy()))
        out[symbol] = (dollar_volume, float(frame["close"].iloc[-1]))
    return out


def build(conn: sqlite3.Connection, *, verbose: bool = True) -> list[sqlite3.Row]:
    """Run the full sweep and screen, replacing the cached universe."""
    if verbose:
        print("building universe: sweeping Alpaca assets...")
    assets = sweep_assets()
    by_symbol = {a.symbol: a for a in assets}
    if verbose:
        print(f"  {len(assets)} active tradable US equities on major exchanges")
        print(f"  measuring liquidity over the last {config.LIQUIDITY_LOOKBACK_DAYS} sessions...")

    liquidity = _liquidity(sorted(by_symbol))

    eligible = [
        (symbol, dv, price)
        for symbol, (dv, price) in liquidity.items()
        if dv >= config.MIN_DOLLAR_VOLUME and price >= config.MIN_PRICE
    ]
    eligible.sort(key=lambda row: row[1], reverse=True)
    selected = eligible[: config.UNIVERSE_SIZE]

    conn.execute("DELETE FROM universe")
    conn.executemany(
        "INSERT INTO universe (symbol, name, exchange, fractionable, shortable, "
        "easy_to_borrow, dollar_volume, last_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                symbol,
                by_symbol[symbol].name,
                str(getattr(by_symbol[symbol].exchange, "value", by_symbol[symbol].exchange)),
                int(bool(by_symbol[symbol].fractionable)),
                int(bool(by_symbol[symbol].shortable)),
                int(bool(by_symbol[symbol].easy_to_borrow)),
                dv,
                price,
            )
            for symbol, dv, price in selected
        ],
    )
    db.set_meta(conn, _BUILT_AT_KEY, datetime.now(timezone.utc).isoformat())
    conn.commit()

    if verbose:
        print(
            f"  {len(eligible)} passed the screen "
            f"(>= {config.MIN_DOLLAR_VOLUME/1e6:.0f}M median $ volume, >= ${config.MIN_PRICE:.0f}); "
            f"kept the top {len(selected)}"
        )
    return load(conn)


def load(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT symbol, name, exchange, fractionable, shortable, easy_to_borrow, "
        "dollar_volume, last_price FROM universe ORDER BY dollar_volume DESC"
    ).fetchall()


def borrowable(conn: sqlite3.Connection) -> set[str]:
    """Symbols that can actually be shorted.

    Both flags are required: `shortable` says Alpaca permits it at all,
    `easy_to_borrow` says there is inventory today. A name failing either is
    not a short candidate, however good the forecast looks.
    """
    return {
        r["symbol"]
        for r in conn.execute(
            "SELECT symbol FROM universe WHERE shortable = 1 AND easy_to_borrow = 1"
        )
    }


def age_days(conn: sqlite3.Connection) -> float | None:
    built_at = db.get_meta(conn, _BUILT_AT_KEY)
    if not built_at:
        return None
    delta = datetime.now(timezone.utc) - datetime.fromisoformat(built_at)
    return delta.total_seconds() / 86400.0


def is_stale(conn: sqlite3.Connection) -> bool:
    age = age_days(conn)
    return age is None or age > config.UNIVERSE_MAX_AGE_DAYS


def get(conn: sqlite3.Connection, *, force_rebuild: bool = False, verbose: bool = True) -> tuple[list, bool]:
    """Return the universe, rebuilding it if forced or stale.

    The second element reports whether a rebuild happened -- the caller uses it
    to decide whether history also needs reseeding, since a rebuild is the
    moment we correct for retroactive adjustment drift.
    """
    if force_rebuild or is_stale(conn):
        return build(conn, verbose=verbose), True

    rows = load(conn)
    if not rows:
        return build(conn, verbose=verbose), True
    if verbose:
        age = age_days(conn) or 0.0
        print(f"universe: {len(rows)} symbols (built {age:.1f} days ago)")
    return rows, False
