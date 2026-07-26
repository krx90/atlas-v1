"""Reading and writing the scan result CSVs.

`top30_assets.csv` is replaced wholesale on every run -- never appended to,
never merged with the previous run. It is written to a temp file in the same
directory and then `os.replace`d into position, which is atomic on the same
filesystem, so the file on disk is always either the complete new ranking or
the untouched old one even if a scan is interrupted mid-write.

The full scored universe is archived separately per run, so history is retained
without the top-30 file ever growing.
"""

from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .scoring import Score

FIELDS = [
    "rank",
    "symbol",
    "name",
    "last_close",
    "score",
    "signal",
    "mu_pct",
    "p_up",
    "sigma_pct",
    "q05_pct",
    "mdd_pct",
    "sharpe",
    "mu_vol_ratio",
    "horizon_days",
    "paths",
    "paths_used",
    "model",
    "scanned_at",
]


def _row(index: int, score: Score, name: str, horizon: int, paths: int, model: str, stamp: str) -> dict:
    return {
        "rank": index,
        "symbol": score.symbol,
        "name": name,
        "last_close": f"{score.last_close:.4f}",
        "score": f"{score.score:.4f}",
        "signal": score.signal,
        "mu_pct": f"{score.mu * 100:.3f}",
        "p_up": f"{score.p_up:.3f}",
        "sigma_pct": f"{score.sigma * 100:.3f}",
        "q05_pct": f"{score.q05 * 100:.3f}",
        "mdd_pct": f"{score.mdd * 100:.3f}",
        "sharpe": f"{score.sharpe:.4f}",
        "mu_vol_ratio": f"{score.mu_vol_ratio:.2f}",
        "horizon_days": horizon,
        "paths": paths,
        "paths_used": score.paths_used,
        "model": model,
        "scanned_at": stamp,
    }


def _write_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    try:
        with handle as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        # tempfile creates at 0600; scan results are not secret, and inheriting
        # owner-only permissions surprises anything else that reads the file.
        os.chmod(handle.name, 0o644)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def write(
    ranked: list[Score],
    names: dict[str, str],
    *,
    horizon: int,
    paths: int,
    model: str,
    top_n: int = config.TOP_N,
) -> tuple[Path, Path]:
    """Overwrite the top-N CSV and archive the full ranking. Returns both paths."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [
        _row(i, s, names.get(s.symbol, ""), horizon, paths, model, stamp)
        for i, s in enumerate(ranked, start=1)
    ]

    _write_atomic(config.TOP_ASSETS_CSV, rows[:top_n])

    archive_name = f"scan_full_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    archive = config.DATA_DIR / archive_name
    _write_atomic(archive, rows)

    return config.TOP_ASSETS_CSV, archive


def write_skipped(entries: list[tuple[str, str]]) -> Path | None:
    """Record every symbol that did not get scored, with its reason."""
    if not entries:
        config.SKIPPED_CSV.unlink(missing_ok=True)
        return None
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with config.SKIPPED_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["symbol", "reason"])
        writer.writerows(sorted(entries))
    return config.SKIPPED_CSV


def read() -> list[dict]:
    """Read the current top-N CSV. Raises FileNotFoundError if absent."""
    with config.TOP_ASSETS_CSV.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def age_hours() -> float | None:
    """Hours since the scan that produced the current CSV, from its own data."""
    try:
        rows = read()
    except FileNotFoundError:
        return None
    if not rows or not rows[0].get("scanned_at"):
        return None
    scanned = datetime.fromisoformat(rows[0]["scanned_at"])
    if scanned.tzinfo is None:
        scanned = scanned.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - scanned).total_seconds() / 3600.0
