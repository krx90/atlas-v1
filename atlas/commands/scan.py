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


def _collect_requests(conn, rows, limit: int | None, horizon: int):
    """Partition the universe into model-ready inputs and recorded skips."""
    symbols = [r["symbol"] for r in rows]
    if limit:
        symbols = symbols[:limit]

    counts = bars.bar_counts(conn, symbols)
    requests: list[forecast.ForecastRequest] = []
    skipped: list[tuple[str, str]] = []

    for symbol in symbols:
        available = counts.get(symbol, 0)
        if available < config.LOOKBACK:
            skipped.append((symbol, f"only {available} bars cached, need {config.LOOKBACK}"))
            continue

        frame = bars.load(conn, symbol, limit=config.LOOKBACK)
        history = forecast.prepare_history(frame)
        if history.isnull().values.any():
            skipped.append((symbol, "NaNs in cached bars"))
            continue
        if len(history) != config.LOOKBACK:
            skipped.append((symbol, f"loaded {len(history)} bars, need {config.LOOKBACK}"))
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


def _render(ranked, names, shown: int) -> None:
    table = ui.table(
        f"Top {shown} by score",
        [
            ("#", "right"),
            ("Symbol", "left"),
            ("Name", "left", {"no_wrap": True, "max_width": 22}),
            ("Last", "right"),
            ("Score", "right"),
            ("E[ret]", "right"),
            ("P(up)", "right"),
            ("Sigma", "right"),
            ("5% tail", "right"),
            ("Signal", "left"),
        ],
    )
    for i, s in enumerate(ranked[:shown], start=1):
        name = names.get(s.symbol, "")
        table.add_row(
            str(i),
            s.symbol,
            name[:28],
            ui.money(s.last_close),
            f"{s.score:.2f}",
            ui.pct_text(s.mu),
            f"{s.p_up * 100:.0f}%",
            f"{s.sigma * 100:.2f}%",
            ui.pct_text(s.q05),
            s.signal,
        )
    ui.console.print()
    ui.console.print(table)


def run(args) -> int:
    started = time.perf_counter()
    horizon = args.horizon or config.HORIZON
    paths = args.paths or config.PATHS

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

        requests, skipped = _collect_requests(conn, rows, args.limit, horizon)

    if not requests:
        ui.error("no symbols have enough cached history to forecast.")
        results.write_skipped(skipped)
        return 1

    try:
        engine = forecast.KronosEngine()
    except forecast.KronosUnavailable as exc:
        ui.error(str(exc))
        return 2

    scores: list[scoring.Score] = []
    rejected: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []

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
                scores.append(scoring.score_paths(symbol_paths))
            except scoring.InvalidForecast as exc:
                # Accounted for, not lost: the model returned something, and we
                # judged it unusable. Distinct from a symbol going missing.
                rejected.append((symbol_paths.symbol, f"rejected: {exc}"))

    ranked = scoring.rank(scores)
    top_path, archive = results.write(
        ranked, names, horizon=horizon, paths=paths, model=engine.model_name
    )
    all_skipped = skipped + rejected + failures
    results.write_skipped(all_skipped)

    _render(ranked, names, min(config.TOP_N, len(ranked)))

    eligible = len(requests)
    scored = len(scores)
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
    ui.info(
        f"  wrote {top_path.name} ({min(config.TOP_N, len(ranked))} rows), "
        f"archived {archive.name}"
    )

    # Every eligible symbol must be accounted for as scored, rejected or failed.
    # Anything else means a symbol vanished silently, which is the one outcome
    # this command must never allow.
    accounted = scored + len(rejected) + len(failures)
    if accounted != eligible:
        seen = {s.symbol for s in scores} | {s for s, _ in rejected} | {s for s, _ in failures}
        missing = sorted({r.symbol for r in requests} - seen)
        ui.error(
            f"{len(missing)} eligible symbol(s) unaccounted for: "
            f"{', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}"
        )
        return 1
    return 0
