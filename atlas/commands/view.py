"""`atlas view XYZ [period]` -- interactive candlestick chart in the browser."""

from __future__ import annotations

import re
import tempfile
import webbrowser
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .. import alpaca_client, bars, db, ui

_PERIOD_DAYS = {"1m": 30, "3m": 91, "6m": 182, "1y": 365, "2y": 730, "3y": 1095, "5y": 1826}
_DEFAULT_PERIOD = "6m"


class BadPeriod(ValueError):
    pass


def parse_period(period: str | None) -> int:
    """`1m 3m 6m 1y 2y 3y 5y` or a plain day count. Returns calendar days."""
    if period is None:
        return _PERIOD_DAYS[_DEFAULT_PERIOD]
    key = period.strip().lower()
    if key in _PERIOD_DAYS:
        return _PERIOD_DAYS[key]
    if re.fullmatch(r"\d+", key):
        days = int(key)
        if days > 0:
            return days
    raise BadPeriod(
        f"unrecognised period {period!r}. Use one of "
        f"{', '.join(_PERIOD_DAYS)} or a number of days."
    )


def _load(conn, symbol: str, days: int):
    """Prefer the cache; fall back to a live fetch for uncovered symbols."""
    frame = bars.load(conn, symbol)
    if frame.empty:
        start = alpaca_client.data_end() - timedelta(days=days + 10)
        fetched = bars.fetch([symbol], start)
        if symbol not in fetched or fetched[symbol].empty:
            return None
        frame = fetched[symbol]
        frame.index = pd.to_datetime(frame.index)
    cutoff = frame.index.max() - timedelta(days=days)
    return frame[frame.index >= cutoff]


def _figure(symbol: str, frame, days: int):
    import plotly.graph_objects as go  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    ma20 = frame["close"].rolling(20).mean()
    ma50 = frame["close"].rolling(50).mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.76, 0.24],
        vertical_spacing=0.03,
    )
    fig.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name=symbol,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=frame.index, y=ma20, name="MA20", line=dict(width=1.2, color="#42a5f5")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=frame.index, y=ma50, name="MA50", line=dict(width=1.2, color="#ab47bc")),
        row=1,
        col=1,
    )

    colors = [
        "#26a69a" if c >= o else "#ef5350"
        for o, c in zip(frame["open"], frame["close"], strict=False)
    ]
    fig.add_trace(
        go.Bar(x=frame.index, y=frame["volume"], name="Volume", marker_color=colors, opacity=0.6),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"{symbol} -- {len(frame)} sessions ({days}d)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        margin=dict(l=50, r=30, t=70, b=40),
        height=760,
    )
    # Daily bars only exist for sessions, so hide weekends rather than drawing gaps.
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def run(args) -> int:
    symbol = args.symbol.upper()
    try:
        days = parse_period(args.period)
    except BadPeriod as exc:
        ui.error(str(exc))
        return 2

    with db.connect() as conn:
        frame = _load(conn, symbol, days)

    if frame is None or frame.empty:
        ui.error(f"No bars available for {symbol}.")
        return 1

    fig = _figure(symbol, frame, days)
    out = Path(tempfile.gettempdir()) / f"atlas_{symbol}_{days}d.html"
    fig.write_html(out, include_plotlyjs="cdn", auto_open=False)
    webbrowser.open_new(out.as_uri())
    ui.info(f"opened {symbol} ({len(frame)} sessions) -> {out}")
    return 0
