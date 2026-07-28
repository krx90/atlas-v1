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


def _collect_requests(conn, rows, limit: int | None, horizon: int, lookback: int):
    """Partition the universe into model-ready inputs and recorded skips."""
    symbols = [r["symbol"] for r in rows]
    if limit:
        symbols = symbols[:limit]

    counts = bars.bar_counts(conn, symbols)
    requests: list[forecast.ForecastRequest] = []
    skipped: list[tuple[str, str]] = []

    for symbol in symbols:
        available = counts.get(symbol, 0)
        if available < lookback:
            skipped.append((symbol, f"only {available} bars cached, need {lookback}"))
            continue

        frame = bars.load(conn, symbol, limit=lookback)
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

    return requests, skipped


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

    with db.connect() as conn:
        rows, rebuilt = universe.get(conn, force_rebuild=args.rebuild_universe)
        names = {r["symbol"]: (r["name"] or "") for r in rows}
        symbols = [r["symbol"] for r in rows]
        if args.limit:
            symbols = symbols[: args.limit]

        # A rebuild is also when we correct for adjusted history being rewritten
        # retroactively by corporate actions.
        label = "seeding history" if rebuilt else "updating history"
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=ui.console,
            transient=True,
        ) as progress:
            task = progress.add_task(label, total=len(symbols))
            written, fetched = bars.sync(
                conn,
                symbols,
                reseed=rebuilt,
                progress=lambda _s: progress.advance(task),
            )
        ui.info(f"{label}: {written:,} bars written across {fetched} symbols")

        requests, skipped = _collect_requests(conn, rows, args.limit, horizon, lookback)

    if not requests:
        ui.error("no symbols have enough cached history to forecast.")
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
            f"forecasting {horizon}d x {paths} paths", total=len(requests)
        )
        for symbol_paths in engine.forecast(
            requests,
            horizon=horizon,
            paths=paths,
            on_batch=lambda group: progress.advance(task, len(group)),
            on_failure=on_failure,
        ):
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

    # No argument means show everything; a side narrows what is printed, never
    # what is computed or written.
    for side in (args.side,) if args.side else (scoring.LONG, scoring.SHORT):
        _render(ranked[side], names, min(config.TOP_N, len(ranked[side])), side)

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
