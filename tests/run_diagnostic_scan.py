#!/usr/bin/env python
"""Step 2 -- the diagnostic scan. Is Kronos coping with intraday bars?

Runs the same symbol sample twice, once on an intraday timeframe and once on
daily bars, and reads the three forecast-health diagnostics off both.

    python tests/run_diagnostic_scan.py                    30 symbols, 1Hour vs daily
    python tests/run_diagnostic_scan.py --symbols 100      widen an ambiguous result
    python tests/run_diagnostic_scan.py --timeframe 30Min  the fallback comparison
    python tests/run_diagnostic_scan.py --skip-scan        re-read the newest results

**The daily run is the point of the pair.** The reference figures in the session
log came from specific runs on specific samples weeks ago, and sample
composition moves these numbers around. A same-day daily baseline on the *same*
symbols is what makes the intraday numbers mean something -- if both come back
equally bad, the problem is not the bar size.

Both runs use the same `--limit`, and the universe is ordered by dollar volume,
so both see the same symbols. Both use the same horizon *in bars*, which keeps
sigma and realized volatility on comparable footing.

Not a pytest module -- it costs minutes and hits the Alpaca API. It lives here
because it is a measurement harness, not a command.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas import config, scoring  # noqa: E402


def newest_health(tag: str, horizon: int, after: float = 0.0) -> Path | None:
    """The most recent health CSV for a timeframe, optionally written after `after`."""
    matches = [
        p
        for p in config.DATA_DIR.glob(f"scan_health_{tag}_{horizon}bar_*.csv")
        if p.stat().st_mtime >= after
    ]
    return max(matches, key=lambda p: p.stat().st_mtime, default=None)


def load_health(path: Path) -> list[scoring.Health]:
    """Rebuild Health rows from a written CSV."""

    def number(value: str) -> float:
        try:
            return float(value)
        except ValueError:
            return float("nan")

    with path.open(newline="", encoding="utf-8") as fh:
        return [
            scoring.Health(
                symbol=row["symbol"],
                mu=number(row["mu_pct"]) / 100.0,
                sigma=number(row["sigma_pct"]) / 100.0,
                realized_vol=number(row["realized_vol_pct"]) / 100.0,
                mu_vol_ratio=number(row["mu_vol_ratio"]),
                sigma_vol_ratio=number(row["sigma_vol_ratio"]),
                paths_used=int(row["paths_used"]),
                paths_total=int(row["paths_total"]),
                rejected=row["rejected"] == "1",
                reason=row["reason"],
            )
            for row in csv.DictReader(fh)
        ]


def run_scan(*, limit: int, horizon: int, paths: int, timeframe: str | None) -> Path | None:
    """Run one scan and return the health CSV it wrote."""
    started = time.time()
    cmd = ["atlas", "scan", "--limit", str(limit), "--horizon", str(horizon),
           "--paths", str(paths)]
    if timeframe:
        cmd += ["--timeframe", timeframe]

    label = timeframe or "1Day"
    print(f"\n{'=' * 72}\n  {label} bars -- {' '.join(cmd)}\n{'=' * 72}", flush=True)
    result = subprocess.run(cmd, check=False)

    tag = timeframe or "1Day"
    written = newest_health(tag, horizon, after=started - 1)
    if result.returncode != 0 and written is None:
        print(f"  ! {label} scan exited {result.returncode} and wrote no health file")
    return written


def compare(runs: dict[str, list[scoring.Health]], horizon: int) -> None:
    """Print the three diagnostics for every run side by side."""
    summaries = {label: scoring.summarize_health(rows) for label, rows in runs.items()}
    labels = list(summaries)
    width = max(12, max(len(l) for l in labels) + 2)

    def row(name: str, fmt, healthy: str, broken: str) -> str:
        cells = "".join(f"{fmt(summaries[l]):>{width}}" for l in labels)
        return f"  {name:<18}{cells}{healthy:>12}{broken:>10}"

    print(f"\n\n{'=' * 72}")
    print(f"  DIAGNOSTIC COMPARISON -- {horizon}-bar horizon")
    print("=" * 72)
    header = "".join(f"{l:>{width}}" for l in labels)
    print(f"  {'':<18}{header}{'healthy':>12}{'broken':>10}")
    print(f"  {'-' * (18 + width * len(labels) + 22)}")
    print(row("symbols", lambda s: str(s.n), "", ""))
    print(row("rejection rate", lambda s: f"{s.rejection_rate * 100:.0f}%", "~18%", "~68%"))
    print(row("median |mu|/vol", lambda s: f"{s.median_mu_vol:.2f}", "~1.06", "~4.80"))
    print(row("sigma/realized", lambda s: f"{s.median_sigma_vol:.2f}", "~1.00", "~0.35"))

    print()
    for label in labels:
        print(f"  {label:<10} {summaries[label].verdict}")

    # Same-symbol overlap, so a difference between the runs is not just a
    # difference in which names each one happened to measure.
    if len(labels) == 2:
        a, b = (set(h.symbol for h in runs[l]) for l in labels)
        shared = a & b
        print(f"\n  {len(shared)} symbols measured in both runs", end="")
        print(f" ({len(a - b)} only in {labels[0]}, {len(b - a)} only in {labels[1]})")

    print(
        "\n  Read these as a pattern, not individually. All three drifting bad\n"
        "  together means the model is genuinely struggling on this input. One\n"
        "  number off on a small sample is probably noise -- a correlation of\n"
        "  -0.73 on 17 symbols once collapsed to -0.09 on 113.\n"
        "\n  Weight sigma/realized well below 1 most heavily: that is the\n"
        "  confidently-wrong signature, dangerous because the forecasts look\n"
        "  clean enough to trade.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--symbols", type=int, default=30, help="symbols per run (default 30)")
    parser.add_argument("--horizon", type=int, default=2, help="bars ahead (default 2)")
    parser.add_argument("--paths", type=int, default=25, help="paths per symbol (default 25)")
    parser.add_argument("--timeframe", default="1Hour",
                        choices=("5Min", "15Min", "30Min", "1Hour"))
    parser.add_argument("--skip-scan", action="store_true",
                        help="compare the newest existing results, run nothing")
    args = parser.parse_args()

    runs: dict[str, list[scoring.Health]] = {}
    for timeframe in (args.timeframe, None):
        tag = timeframe or "1Day"
        if args.skip_scan:
            path = newest_health(tag, args.horizon)
        else:
            path = run_scan(
                limit=args.symbols,
                horizon=args.horizon,
                paths=args.paths,
                timeframe=timeframe,
            )
        if path is None:
            print(f"  ! no health results for {tag} at horizon {args.horizon}")
            continue
        runs[tag] = load_health(path)
        print(f"  read {len(runs[tag])} symbols from {path.name}")

    if len(runs) < 2:
        print("\n  Need both runs to compare. Re-run without --skip-scan.")
        return 1

    compare(runs, args.horizon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
