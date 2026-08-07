"""`atlas accuracy` -- read a saved backtest CSV and measure whether it was right.

Deliberately a separate command from `atlas backtest`. Producing observations
costs eighty minutes of model time; judging them costs milliseconds, and the two
should not be welded together. Re-analysing a run you already have -- with more
permutations, or after a metric changes -- should never mean re-running Kronos.

`atlas backtest` calls `render` here at the end of its own run, so the same
tables appear either way and there is one implementation of them.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .. import accuracy, backtest, config, ui

#: Columns added after the first backtests were run. A CSV written before them
#: loads with these defaulted, and the metrics that need them report NaN rather
#: than a fabricated zero.
_OPTIONAL = {"sigma": 0.0, "q05": 0.0, "q95": 0.0, "p_up": 0.0, "prior_return": 0.0}


def load(path: Path) -> list[backtest.Observation]:
    """Rebuild observations from a backtest CSV."""

    def number(row: dict, key: str, default: float = 0.0) -> float:
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            return default

    with path.open(newline="", encoding="utf-8") as fh:
        return [
            backtest.Observation(
                date=row["date"],
                symbol=row["symbol"],
                score=number(row, "score"),
                mu=number(row, "mu"),
                # The side is not in the CSV; it is in the filename. Only the
                # bucket labels depend on it, and those carry it themselves.
                side=row.get("side", ""),
                forward_return=number(row, "forward_return"),
                signal=row.get("signal", ""),
                **{key: number(row, key, default) for key, default in _OPTIONAL.items()},
            )
            for row in csv.DictReader(fh)
        ]


def _fmt(value: float, spec: str = "+.4f") -> str:
    """NaN prints as a dash rather than 'nan', which reads as a broken number."""
    return "-" if value != value else format(value, spec)


def render(result: accuracy.Accuracy, cover: backtest.Coverage | None = None) -> None:
    """The accuracy tables: direction, magnitude, baselines, buckets, coverage."""
    table = ui.table(
        "Accuracy -- was the forecast right?",
        [("Measure", "left"), ("Value", "right"), ("Reading", "left")],
    )

    d = result.direction
    table.add_row(
        "directional hit rate",
        f"{d.hit_rate * 100:.1f}%" if d.n else "-",
        f"{d.hits:,}/{d.n:,} signs correct, p={_fmt(d.p_value, '.3f')}",
    )

    m = result.size
    table.add_row("mu MAE", _fmt(m.mae, ".4f"), "mean absolute error of the point forecast")
    table.add_row("mu RMSE", _fmt(m.rmse, ".4f"), f"vs {_fmt(m.baseline_rmse, '.4f')} forecasting zero")
    table.add_row("skill score", _fmt(m.skill), "<= 0 means no better than 'no move'")
    table.add_row(
        "mean |mu| vs |actual|",
        f"{_fmt(m.mean_abs_mu, '.4f')} / {_fmt(m.mean_abs_return, '.4f')}",
        "ratio far above 1 is overconfidence",
    )

    n = result.null
    table.add_row("mean rank IC", _fmt(n.mean_ic), f"over {n.dates} dates")
    table.add_row(
        "permutation p",
        _fmt(n.p_value, ".3f"),
        f"vs shuffled scores (null sd {_fmt(n.null_sd, '.4f')}, {n.permutations:,} draws)",
    )
    table.add_row(
        "momentum IC",
        _fmt(result.momentum),
        "trailing return, the free baseline to beat",
    )
    ui.console.print()
    ui.console.print(table)

    if result.buckets:
        buckets = ui.table(
            "By signal -- what happened to what you would have traded",
            [("Signal", "left"), ("N", "right"), ("Mean return", "right"), ("Hit rate", "right")],
        )
        for b in result.buckets:
            buckets.add_row(
                b.signal, f"{b.n:,}", ui.signed_pct(b.mean_return), f"{b.hit_rate * 100:.1f}%"
            )
        ui.console.print()
        ui.console.print(buckets)

    if cover is not None and cover.n:
        calib = ui.table(
            "Calibration -- was the distribution the right width?",
            [("Band", "left"), ("Actual", "right"), ("Target", "right")],
        )
        calib.add_row("below q05", f"{cover.below_q05 * 100:.1f}%", "5.0%")
        calib.add_row("above q95", f"{cover.above_q95 * 100:.1f}%", "5.0%")
        calib.add_row("inside", f"{cover.inside * 100:.1f}%", "90.0%")
        ui.console.print()
        ui.console.print(calib)
        ui.console.print()
        ui.console.print(f"  [bold]calibration:[/bold] {cover.verdict}")

    ui.console.print()
    ui.console.print(f"  [bold]direction:[/bold]  {d.verdict}")
    ui.console.print(f"  [bold]magnitude:[/bold]  {m.verdict}")
    ui.console.print(f"  [bold]vs shuffle:[/bold] {n.verdict}")

    # The bar, printed beside the result. A verdict read without the threshold
    # it was judged against invites moving the threshold after the fact.
    bar = ui.table(
        f"Pre-registered bar (config.py, fixed {config.ACCURACY_FIXED_ON})",
        [("Condition", "left"), ("Required", "left"), ("Observed", "right"), ("", "left")],
    )
    passes = result.passes
    required = {
        "direction": f"p < {config.ACCURACY_MAX_DIRECTION_P} and hit rate > 50%",
        "magnitude": "skill score > 0",
        "ranking": f"permutation p < {config.ACCURACY_MAX_PERMUTATION_P} and IC > 0",
        "vs momentum": "mean IC > momentum IC",
    }
    observed = {
        "direction": f"p={_fmt(d.p_value, '.3f')}, {d.hit_rate * 100:.1f}%" if d.n else "-",
        "magnitude": _fmt(m.skill),
        "ranking": f"p={_fmt(n.p_value, '.3f')}, IC {_fmt(n.mean_ic)}",
        "vs momentum": _fmt(result.momentum),
    }
    for name, ok in passes.items():
        mark = "[green]pass[/green]" if ok else "[red]fail[/red]"
        bar.add_row(name, required[name], observed[name], mark)
    ui.console.print()
    ui.console.print(bar)

    ui.console.print()
    ui.console.print(f"  [bold]verdict:[/bold] {result.verdict}")


def run(args) -> int:
    path = Path(args.csv)
    if not path.exists():
        # The bare filename is the common case, so look in data/ before failing.
        candidate = config.DATA_DIR / path.name
        if not candidate.exists():
            ui.error(f"no such file: {path}")
            available = sorted(config.DATA_DIR.glob("backtest_*.csv"))
            if available:
                ui.info("  available runs: " + ", ".join(p.name for p in available))
            return 1
        path = candidate

    obs = load(path)
    if not obs:
        ui.error(f"{path} has no observations.")
        return 1

    ui.info(f"{len(obs):,} observations from {path.name}")
    render(
        accuracy.evaluate(obs, permutations=args.permutations),
        backtest.coverage(obs),
    )

    if all(o.prior_return == 0 for o in obs):
        ui.console.print()
        ui.warn(
            "this CSV predates the prior_return column, so the momentum baseline "
            "could not be computed -- re-run `atlas backtest` to fill it."
        )
    ui.console.print()
    ui.console.print(
        "  [dim]Same biases as the backtest that produced this file: survivorship,\n"
        "  no transaction costs, one market regime.[/dim]"
    )
    return 0
