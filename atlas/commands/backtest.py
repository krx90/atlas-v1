"""`atlas backtest` -- walk-forward test of whether the ranking predicts returns."""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from .. import backtest, config, db, forecast, scoring, ui, universe


def _render(result: backtest.Result, side: str, horizon: int, unit: str = "day") -> None:
    table = ui.table(
        f"Backtest -- {side} ranking, {horizon} x {unit} forward window",
        [("Metric", "left"), ("Value", "right"), ("Reading", "left")],
    )
    table.add_row("as-of dates", str(result.dates), "")
    table.add_row("observations", f"{result.observations:,}", "symbol-dates scored")
    table.add_row(
        "mean rank IC",
        f"{result.mean_ic:+.4f}",
        "correlation between score and what happened",
    )
    table.add_row(
        "IC t-statistic",
        f"{result.ic_t_stat:+.2f}",
        "|t| < 2 means indistinguishable from luck",
    )
    table.add_row("top decile", ui.signed_pct(result.top_return), "mean forward return")
    table.add_row("bottom decile", ui.signed_pct(result.bottom_return), "mean forward return")
    table.add_row("spread", ui.signed_pct(result.mean_spread), "top minus bottom, before costs")
    table.add_row("hit rate", f"{result.hit_rate * 100:.0f}%", "dates where top beat bottom")
    ui.console.print()
    ui.console.print(table)

    ui.console.print()
    ui.console.print(f"  [bold]verdict:[/bold] {result.verdict}")


def run(args) -> int:
    started = time.perf_counter()
    horizon = args.horizon or config.HORIZON
    paths = args.paths or config.PATHS
    side = args.side or scoring.LONG
    cached = not getattr(args, "no_cache", False)
    lookback = config.data_lookback(config.LOOKBACK, horizon, cached=cached)
    timeframe = getattr(args, "timeframe", None)
    unit = "day" if timeframe is None else timeframe

    with db.connect() as conn:
        rows = universe.load(conn)
        if not rows:
            ui.error("no universe cached -- run `atlas scan` first.")
            return 1
        symbols = [r["symbol"] for r in rows][: args.symbols]

        ui.info(
            f"slicing {len(symbols)} symbols x {args.dates} as-of points "
            f"({args.spacing} bars apart, {horizon}-bar horizon, {unit} bars)..."
        )
        windows, dates = backtest.build(
            conn,
            symbols,
            dates=args.dates,
            spacing=args.spacing,
            lookback=lookback,
            horizon=horizon,
            timeframe=timeframe,
        )

    if not windows:
        ui.error(
            f"no symbol has {lookback + horizon + 1} cached {unit} bars -- "
            "seed history first (`atlas scan`, or fetch intraday bars)."
        )
        return 1

    total = sum(len(v) for v in windows.values())
    ui.info(
        f"{total:,} symbol-dates across {len(dates)} dates "
        f"({dates[0]} .. {dates[-1]}) -- roughly {total * 2.5 / 60:.0f} min"
    )

    # A heartbeat file, so a run backgrounded or redirected to a log stays
    # observable. Rich suppresses its live bar when stdout is not a terminal,
    # which leaves an 80-minute job indistinguishable from a hung one.
    # `scripts/watch-progress.sh` renders a bar from this.
    #
    # The pid is in the filename because a single fixed path breaks badly with
    # two concurrent runs: they overwrite each other's counts, and whichever
    # finishes first deletes the file, so the watcher reports "done" while the
    # other job is still going.
    heartbeat = config.DATA_DIR / f"progress.{os.getpid()}.json"
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    last_write = 0.0

    def beat(stamp: str, n: int) -> None:
        nonlocal done, last_write
        done += n
        now = time.perf_counter()
        # Throttle: one file write a second is plenty for a 2s-refresh watcher,
        # and avoids thousands of writes over a long run. Always write the
        # first and last beat so the file appears immediately and ends correct.
        if n and done < total and now - last_write < 1.0:
            return
        last_write = now
        elapsed = now - started
        rate = done / elapsed if elapsed > 0 else 0.0
        heartbeat.write_text(
            json.dumps(
                {
                    "task": f"backtest {side}",
                    "pid": os.getpid(),
                    "done": done,
                    "total": total,
                    "last_date": stamp,
                    "elapsed_s": round(elapsed, 1),
                    "eta_s": round((total - done) / rate) if rate > 0 else None,
                    "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )
        )

    # Write 0/total before the model loads. Loading takes seconds and the first
    # forecast longer; without this the watcher reports "no run in progress"
    # while the job is in fact starting up.
    beat(dates[0], 0)

    try:
        engine = forecast.KronosEngine(use_cache=cached)
    except forecast.KronosUnavailable as exc:
        ui.error(str(exc))
        heartbeat.unlink(missing_ok=True)
        return 2

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=ui.console,
    ) as progress:
        task = progress.add_task(f"backtesting {side}", total=total)

        def on_progress(stamp: str, n: int) -> None:
            progress.advance(task, n)
            beat(stamp, n)

        obs = backtest.observations(
            engine,
            windows,
            horizon=horizon,
            paths=paths,
            side=side,
            on_progress=on_progress,
        )
    heartbeat.unlink(missing_ok=True)

    if not obs:
        ui.error("no usable observations -- every forecast was rejected by the guards.")
        return 1

    result = backtest.evaluate(obs)
    _render(result, side, horizon, unit)

    tag = "day" if timeframe is None else timeframe
    out = config.DATA_DIR / f"backtest_{side}_{tag}_{horizon}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "date", "symbol", "score", "mu", "forward_return",
            "sigma", "q05", "q95", "p_up",
        ])
        for o in obs:
            writer.writerow([
                o.date, o.symbol, f"{o.score:.6f}", f"{o.mu:.6f}",
                f"{o.forward_return:.6f}", f"{o.sigma:.6f}",
                f"{o.q05:.6f}", f"{o.q95:.6f}", f"{o.p_up:.3f}",
            ])

    ui.console.print()
    ui.info(f"  per-date IC: " + ", ".join(f"{d[5:]} {v:+.2f}" for d, v, _n in result.ic_by_date))
    ui.info(f"  {len(obs):,} observations written to {out}")
    ui.info(f"  completed in {(time.perf_counter() - started) / 60:.1f} min")
    ui.console.print()
    ui.console.print(
        "  [dim]Survivorship-biased (universe is today's liquid names), no transaction\n"
        "  costs, one market regime. Treat as a smoke test, not a validation.[/dim]"
    )
    return 0
