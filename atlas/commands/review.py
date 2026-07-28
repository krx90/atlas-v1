"""`atlas review` -- which open positions deserve attention.

Reports only. Closing stays a deliberate `atlas close XYZ`, consistent with
there being no bulk buy or sell.
"""

from __future__ import annotations

import time

import numpy as np

from .. import alpaca_client, bars, config, db, forecast, review, scoring, ui, universe
from .portfolio import entry_dates


def _daily_vol(frame) -> float:
    """Standard deviation of daily log returns, over the cached window."""
    closes = frame["close"].to_numpy()
    if len(closes) < 3:
        return 0.0
    returns = np.diff(np.log(closes[-config.REVIEW_VOL_WINDOW - 1 :]))
    return float(np.std(returns)) if returns.size >= 2 else 0.0


def _forecasts(requests, quiet: bool) -> dict[str, tuple[float, float]]:
    """Forecast the holdings, returning symbol -> (mu, |mu|/realized_vol).

    Uses the KV cache: `lookback + horizon <= max_context` holds by construction
    here because the lookback is chosen as `max_context - horizon`.
    """
    if not requests:
        return {}
    try:
        engine = forecast.KronosEngine(use_cache=True, quiet=quiet)
    except forecast.KronosUnavailable as exc:
        ui.warn(f"skipping the forecast signal -- {exc}")
        return {}

    out: dict[str, tuple[float, float]] = {}
    for paths in engine.forecast(requests, horizon=config.HORIZON, paths=config.PATHS):
        try:
            score = scoring.score_paths(paths)
        except scoring.InvalidForecast:
            continue  # same guards as a scan; an unusable forecast is no signal
        out[paths.symbol] = (score.mu, score.mu_vol_ratio)
    return out


def _gather(conn, positions, opened, universe_rows, forecasts):
    holdings = []
    for position in positions:
        symbol = position.symbol
        row = universe_rows.get(symbol)
        frame = bars.load(conn, symbol, limit=config.REVIEW_VOL_WINDOW + 1)
        entered = opened.get(symbol)
        mu, ratio = forecasts.get(symbol, (None, None))
        holdings.append(
            review.Holding(
                symbol=symbol,
                side=str(getattr(position.side, "value", position.side)),
                pnl_pct=float(position.unrealized_plpc),
                market_value=float(position.market_value),
                days_held=None if entered is None else max((_now() - entered).days, 0),
                daily_vol=_daily_vol(frame) if not frame.empty else 0.0,
                dollar_volume=row["dollar_volume"] if row else None,
                in_universe=row is not None,
                easy_to_borrow=bool(row["easy_to_borrow"]) if row else False,
                forecast_mu=mu,
                forecast_vol_ratio=ratio,
            )
        )
    return holdings


def _now():
    from datetime import datetime, timezone  # noqa: PLC0415

    return datetime.now(timezone.utc)


def _render(reviews) -> None:
    table = ui.table(
        "Positions worth reviewing",
        [
            ("Symbol", "left"),
            ("Side", "left"),
            ("P&L %", "right"),
            ("Sigma", "right"),
            ("Held", "right"),
            ("Score", "right"),
            ("Flags", "left", {"no_wrap": False}),
        ],
    )
    for r in reviews:
        h = r.holding
        sigma = r.pnl_sigma
        table.add_row(
            h.symbol,
            ui.Text(h.side.upper(), style="cyan" if h.side == "long" else "magenta"),
            ui.pct_text(h.pnl_pct),
            ui.Text(f"{sigma:+.2f}", style="red" if sigma <= -config.REVIEW_STOP_SIGMA else ""),
            "-" if h.days_held is None else (f"{h.days_held}d" if h.days_held else "today"),
            ui.Text(f"{r.score:.2f}", style="bold red" if r.score > 0 else "dim"),
            "; ".join(f.detail for f in r.findings) or "-",
        )
    ui.console.print()
    ui.console.print(table)


def run(args) -> int:
    started = time.perf_counter()
    positions = alpaca_client.call(alpaca_client.trading().get_all_positions)
    if not positions:
        ui.console.print()
        ui.info("No open positions.")
        return 0

    symbols = [p.symbol for p in positions]
    opened = entry_dates(symbols)

    with db.connect() as conn:
        rows = {r["symbol"]: r for r in universe.load(conn)}

        # A holding need not be in the universe -- bought manually, or dropped
        # out since. Fetch whatever history is missing rather than skipping it.
        counts = bars.bar_counts(conn, symbols)
        lookback = config.LOOKBACK - config.HORIZON
        missing = [s for s in symbols if counts.get(s, 0) < lookback]
        if missing:
            ui.info(f"fetching history for {len(missing)} uncached holding(s): {', '.join(missing)}")
            bars.sync(conn, missing)
            counts = bars.bar_counts(conn, symbols)

        requests = []
        if not args.no_forecast:
            for symbol in symbols:
                if counts.get(symbol, 0) < lookback:
                    continue
                frame = bars.load(conn, symbol, limit=lookback)
                history = forecast.prepare_history(frame)
                if history.isnull().values.any() or len(history) != lookback:
                    continue
                closes = history["close"].to_numpy()
                requests.append(
                    forecast.ForecastRequest(
                        symbol,
                        history,
                        float(closes[-1]),
                        realized_vol=forecast.realized_vol(closes, config.HORIZON),
                    )
                )

        forecasts = _forecasts(requests, quiet=False)
        holdings = _gather(conn, positions, opened, rows, forecasts)

    ranked = review.rank(holdings)
    flagged = [r for r in ranked if r.score > 0]

    _render(ranked)

    ui.console.print()
    if not flagged:
        ui.info(f"  Nothing flagged across {len(ranked)} positions.")
    else:
        ui.info(f"  {len(flagged)} of {len(ranked)} positions flagged. To act:")
        for r in flagged[: config.REVIEW_SUGGEST]:
            ui.info(f"    atlas close {r.holding.symbol}")
    if any(f.kind == "forecast*" for r in ranked for f in r.findings):
        ui.info(
            "  [dim]* forecast signal -- no demonstrated skill "
            "(backtest IC -0.008 / +0.026), see docs/implementation.md[/dim]"
        )
    ui.info(f"  reviewed in {time.perf_counter() - started:.1f}s")
    return 0
