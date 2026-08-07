"""`atlas scan` -- build the universe, forecast it, rank it, write the CSV.

Coverage is tracked explicitly. Symbols are partitioned up front into eligible
and skipped-with-a-reason, every eligible symbol must come back scored, and the
run reconciles the two at the end. If they disagree the command exits non-zero
and names the missing symbols, so the ranking can never quietly shrink.
"""

from __future__ import annotations

import time
from collections import Counter

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from .. import bars, config, db, forecast, results, scoring, ui, universe


def _collect_requests(
    conn, rows, limit: int | None, horizon: int, lookback: int, timeframe: str | None = None
):
    """Partition the universe into model-ready inputs and recorded skips.

    `timeframe` of None means daily bars; otherwise an intraday timeframe, read
    from the intraday cache. Also returns the mean regular-session bars per
    trading day actually seen, which is how the caller verifies that session
    filtering was applied -- see `_check_session_filter`.
    """
    symbols = [r["symbol"] for r in rows]
    if limit:
        symbols = symbols[:limit]

    counts = (
        bars.bar_counts(conn, symbols)
        if timeframe is None
        else bars.intraday_counts(conn, symbols, timeframe)
    )
    requests: list[forecast.ForecastRequest] = []
    skipped: list[tuple[str, str]] = []
    per_day: list[float] = []

    for symbol in symbols:
        available = counts.get(symbol, 0)
        if available < lookback:
            skipped.append((symbol, f"only {available} bars cached, need {lookback}"))
            continue

        frame = (
            bars.load(conn, symbol, limit=lookback)
            if timeframe is None
            else bars.load_intraday(conn, symbol, timeframe, limit=lookback)
        )
        if timeframe is not None and not frame.empty:
            per_day.append(bars.bars_per_day(frame))
        history = forecast.prepare_history(frame)
        if history.isnull().values.any():
            skipped.append((symbol, "NaNs in cached bars"))
            continue
        if len(history) != lookback:
            skipped.append((symbol, f"loaded {len(history)} bars, need {lookback}"))
            continue
        closes = history["close"].to_numpy()
        last_close = float(closes[-1])
        if not last_close > 0:
            skipped.append((symbol, f"non-positive last close {last_close}"))
            continue

        requests.append(
            forecast.ForecastRequest(
                symbol,
                history,
                last_close,
                realized_vol=forecast.realized_vol(closes, horizon),
            )
        )

    observed = sum(per_day) / len(per_day) if per_day else 0.0
    return requests, skipped, observed


def _check_session_filter(timeframe: str, observed: float) -> bool:
    """Is the cached intraday data regular-session only?

    Extended-hours bars are ~59% of Alpaca's intraday rows and nothing warns
    you -- the count just looks generous. Unfiltered data would put thin
    pre/post-market noise through most of the model's context, which makes
    every number downstream a measurement of the wrong thing. So this is
    checked before the forecast rather than trusted.

    The yardstick is the measured per-session count, not arithmetic on session
    length: Alpaca aligns hourly bars to the clock hour, so 1Hour legitimately
    yields 6 bars and not the 7 that 09:30-16:00 implies.
    """
    expected = bars.INTRADAY_TIMEFRAMES[timeframe]
    if observed <= expected * 1.25:
        ui.info(
            f"  session filter OK: {observed:.1f} bars/day against {expected} expected "
            f"for {timeframe}"
        )
        return True
    ui.error(
        f"extended-hours bars are in the cache: {observed:.1f} bars/day against "
        f"{expected} expected for {timeframe}. The cached rows predate the session "
        f"filter. Clear them and refetch:\n"
        f"    sqlite3 data/atlas.db \"DELETE FROM intraday_bars WHERE timeframe='{timeframe}';\""
    )
    return False


def _render_health(summary, timeframe: str | None, horizon: int) -> None:
    """The three diagnostics, against the reference points they are read by."""
    unit = timeframe or "1Day"
    table = ui.table(
        f"Forecast health -- {unit} bars, {horizon}-bar horizon, {summary.n} symbols",
        [
            ("Diagnostic", "left"),
            ("Measured", "right"),
            ("Healthy", "right"),
            ("Broken", "right"),
        ],
    )
    table.add_row(
        "rejection rate", f"{summary.rejection_rate * 100:.0f}%", "~18%", "~68%"
    )
    table.add_row("median |mu|/vol", f"{summary.median_mu_vol:.2f}", "~1.06", "~4.80")
    table.add_row("sigma/realized", f"{summary.median_sigma_vol:.2f}", "~1.00", "~0.35")
    ui.console.print()
    ui.console.print(table)
    ui.console.print()
    ui.console.print(f"  [bold]verdict:[/bold] {summary.verdict}")


def _render(ranked, names, shown: int, side: str) -> None:
    """One table per side, with that side's own risk columns.

    A long is judged on P(up), the 5th-percentile outcome and the worst low; a
    short on P(down), the 95th percentile and the worst high. Showing a long's
    columns for a short would make the risk look like the opposite of what it is.
    """
    is_long = side == scoring.LONG
    header = "Top %d long candidates" if is_long else "Top %d short candidates"
    table = ui.table(
        header % shown,
        [
            ("#", "right"),
            ("Symbol", "left"),
            ("Name", "left", {"no_wrap": True, "max_width": 22}),
            ("Last", "right"),
            ("Score", "right"),
            ("E[ret]", "right"),
            ("P(up)" if is_long else "P(down)", "right"),
            ("Sigma", "right"),
            ("5% tail" if is_long else "95% tail", "right"),
            ("Max DD" if is_long else "Max run-up", "right"),
            ("Signal", "left"),
        ],
    )
    for i, s in enumerate(ranked[:shown], start=1):
        table.add_row(
            str(i),
            s.symbol,
            names.get(s.symbol, "")[:28],
            ui.money(s.last_close),
            f"{s.score:.2f}",
            ui.pct_text(s.mu),
            f"{(s.p_up if is_long else s.p_down) * 100:.0f}%",
            f"{s.sigma * 100:.2f}%",
            ui.pct_text(s.q05 if is_long else s.q95),
            ui.pct_text(s.mdd if is_long else s.runup),
            s.signal,
        )
    ui.console.print()
    ui.console.print(table)


def run(args) -> int:
    started = time.perf_counter()
    horizon = args.horizon or config.HORIZON
    paths = args.paths or config.PATHS

    # Resolve the checkpoint first: the lookback follows the model's context
    # window, and requests are built before the engine exists.
    choice = getattr(args, "model", None) or "small"
    model_name, tokenizer_name, context = config.MODEL_CHOICES[choice]
    # The KV cache needs room to generate inside the context window, so the
    # lookback is `context - horizon` when caching. Also capped: five years of
    # daily bars is ~1250, so mini's 2048-bar context cannot be filled.
    cached = not getattr(args, "no_cache", False)
    lookback = config.data_lookback(context, horizon, cached=cached)
    timeframe = getattr(args, "timeframe", None)
    unit = "day" if timeframe is None else timeframe

    with db.connect() as conn:
        rows, rebuilt = universe.get(conn, force_rebuild=args.rebuild_universe)
        names = {r["symbol"]: (r["name"] or "") for r in rows}
        symbols = [r["symbol"] for r in rows]
        if args.limit:
            symbols = symbols[: args.limit]

        # A rebuild is also when we correct for adjusted history being rewritten
        # retroactively by corporate actions.
        if timeframe is None:
            label = "seeding history" if rebuilt else "updating history"
        else:
            # Intraday history is fetched per symbol over a window sized from
            # the lookback, so the daily sync is not on this run's path.
            days = bars.fetch_days_for(timeframe, lookback)
            label = f"fetching {days}d of {timeframe} history"
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=ui.console,
            transient=True,
        ) as progress:
            task = progress.add_task(label, total=len(symbols))
            if timeframe is None:
                written, fetched = bars.sync(
                    conn,
                    symbols,
                    reseed=rebuilt,
                    progress=lambda _s: progress.advance(task),
                )
            else:
                written, fetched = bars.sync_intraday(
                    conn,
                    symbols,
                    timeframe,
                    days=days,
                    progress=lambda _s: progress.advance(task),
                )
        ui.info(f"{label}: {written:,} bars written across {fetched} symbols")

        requests, skipped, observed = _collect_requests(
            conn, rows, args.limit, horizon, lookback, timeframe
        )

    # Verified before the forecast, not after: an unfiltered cache makes every
    # diagnostic below a measurement of pre/post-market noise, and there is no
    # point spending minutes of model time to produce that.
    if timeframe is not None and requests and not _check_session_filter(timeframe, observed):
        return 1

    if not requests:
        ui.error(
            f"no symbols have {lookback} cached {unit} bars to forecast from."
        )
        results.write_skipped(skipped)
        return 1

    try:
        engine = forecast.KronosEngine(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            max_context=context,
            use_cache=cached,
        )
    except forecast.KronosUnavailable as exc:
        ui.error(str(exc))
        return 2

    # Both sides are read off the same sampled paths, so scoring long and short
    # costs one forward pass, not two.
    scores: dict[str, list[scoring.Score]] = {scoring.LONG: [], scoring.SHORT: []}
    # Health is recorded for every symbol the model returns, before the guards
    # decide anything, so the diagnostics measure the whole population rather
    # than the survivors of the filter they are meant to be judging.
    health: list[scoring.Health] = []
    rejected: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []
    with db.connect() as conn:
        can_borrow = universe.borrowable(conn)

    def on_failure(symbols_failed, exc):
        for symbol in symbols_failed:
            failures.append((symbol, f"forecast failed: {type(exc).__name__}: {exc}"))

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=ui.console,
    ) as progress:
        task = progress.add_task(
            f"forecasting {horizon} x {unit} bars, {paths} paths", total=len(requests)
        )
        for symbol_paths in engine.forecast(
            requests,
            horizon=horizon,
            paths=paths,
            on_batch=lambda group: progress.advance(task, len(group)),
            on_failure=on_failure,
        ):
            health.append(scoring.health(symbol_paths))
            try:
                scores[scoring.LONG].append(scoring.score_paths(symbol_paths))
                # A name that cannot be borrowed is not a short candidate,
                # however good the forecast. Excluded here rather than shown
                # and then rejected at order time.
                if symbol_paths.symbol in can_borrow:
                    scores[scoring.SHORT].append(
                        scoring.score_paths(symbol_paths, side=scoring.SHORT)
                    )
            except scoring.InvalidForecast as exc:
                # Accounted for, not lost: the model returned something, and we
                # judged it unusable. Distinct from a symbol going missing. The
                # guards are direction-neutral, so this rejects both sides.
                rejected.append((symbol_paths.symbol, f"rejected: {exc}"))

    ranked = {side: scoring.rank(s) for side, s in scores.items()}
    written, archive = results.write(
        ranked, names, horizon=horizon, paths=paths, model=engine.model_name
    )
    all_skipped = skipped + rejected + failures
    results.write_skipped(all_skipped)
    health_path = results.write_health(health, timeframe=timeframe, horizon=horizon)
    summary = scoring.summarize_health(health)

    # No argument means show everything; a side narrows what is printed, never
    # what is computed or written.
    for side in (args.side,) if args.side else (scoring.LONG, scoring.SHORT):
        _render(ranked[side], names, min(config.TOP_N, len(ranked[side])), side)

    _render_health(summary, timeframe, horizon)

    eligible = len(requests)
    scored = len(scores[scoring.LONG])
    ui.console.print()
    ui.info(
        f"scored {scored} / eligible {eligible} / universe {len(symbols)} "
        f"in {time.perf_counter() - started:.1f}s"
    )
    if rejected:
        # Covers both rejection kinds: too few numerically valid paths, and a
        # mean move implausible against the symbol's own realized volatility.
        ui.info(f"  {len(rejected)} rejected as unusable model output")
    if failures:
        ui.info(f"  {len(failures)} failed to forecast")
    if skipped:
        # Group by reason kind, not the exact message, so "only 29 bars cached"
        # and "only 362 bars cached" collapse instead of each taking a slot.
        reasons = Counter(
            "insufficient history" if "bars cached" in reason else reason
            for _s, reason in skipped
        )
        detail = ", ".join(f"{n} x {r}" for r, n in reasons.most_common(4))
        ui.info(f"  {len(skipped)} skipped before forecasting: {detail}")
    if all_skipped:
        ui.info(f"  detail: {config.SKIPPED_CSV}")
    ui.info(f"  per-symbol health: {health_path.name}")
    n_long = min(config.TOP_N, len(ranked[scoring.LONG]))
    n_short = min(config.TOP_N, len(ranked[scoring.SHORT]))
    ui.info(
        f"  wrote {written[scoring.LONG].name} ({n_long} rows) and "
        f"{written[scoring.SHORT].name} ({n_short} rows), archived {archive.name}"
    )
    not_borrowable = scored - len(scores[scoring.SHORT])
    if not_borrowable > 0:
        ui.info(f"  {not_borrowable} scored symbols excluded from shorts (not borrowable)")

    # Every eligible symbol must be accounted for as scored, rejected or failed.
    # Anything else means a symbol vanished silently, which is the one outcome
    # this command must never allow.
    accounted = scored + len(rejected) + len(failures)
    if accounted != eligible:
        seen = (
            {s.symbol for s in scores[scoring.LONG]}
            | {s for s, _ in rejected}
            | {s for s, _ in failures}
        )
        missing = sorted({r.symbol for r in requests} - seen)
        ui.error(
            f"{len(missing)} eligible symbol(s) unaccounted for: "
            f"{', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}"
        )
        return 1
    return 0
