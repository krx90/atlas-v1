"""Reading and writing the scan result CSVs.

`top30_long.csv` and `top30_short.csv` are each replaced wholesale on every run
-- never appended to, never merged with the previous run. Each is written to a
temp file in the same directory and then `os.replace`d into position, which is
atomic on the same filesystem, so the file on disk is always either the complete
new ranking or the untouched old one even if a scan is interrupted mid-write.

Both are written on every scan whichever side was requested: the two rankings
come from the same sampled paths, so producing both costs nothing beyond the one
forward pass, and discarding one would mean a full re-scan to see it.

The full scored universe is archived separately per run, carrying *both* sides'
statistics, so history is retained and either ranking can be re-derived without
re-running the model.
"""

from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .scoring import LONG, SHORT, Score

#: Columns shared by both sides.
_COMMON = [
    "rank",
    "symbol",
    "name",
    "last_close",
    "score",
    "signal",
    "mu_pct",
    "sigma_pct",
    "sharpe",
    "mu_vol_ratio",
    "realized_vol_pct",
    "sigma_vol_ratio",
]
_TRAILER = ["horizon_days", "paths", "paths_used", "model", "scanned_at"]

#: Side-specific risk columns. A long is hurt by the downside tail and the worst
#: low; a short by the upside tail and the worst high.
_SIDE_FIELDS = {
    LONG: ["p_up", "q05_pct", "mdd_pct"],
    SHORT: ["p_down", "q95_pct", "runup_pct"],
}

FIELDS = {side: _COMMON + extra + _TRAILER for side, extra in _SIDE_FIELDS.items()}
#: The archive keeps everything, so nothing computed is thrown away.
ARCHIVE_FIELDS = _COMMON + ["side"] + _SIDE_FIELDS[LONG] + _SIDE_FIELDS[SHORT] + _TRAILER


def _row(index: int, score: Score, name: str, horizon: int, paths: int, model: str, stamp: str) -> dict:
    """Every column for one symbol, both sides. Callers project what they need."""
    return {
        "rank": index,
        "symbol": score.symbol,
        "name": name,
        "last_close": f"{score.last_close:.4f}",
        "score": f"{score.score:.4f}",
        "signal": score.signal,
        "side": score.side,
        "mu_pct": f"{score.mu * 100:.3f}",
        "sigma_pct": f"{score.sigma * 100:.3f}",
        "sharpe": f"{score.sharpe:.4f}",
        "mu_vol_ratio": f"{score.mu_vol_ratio:.2f}",
        "realized_vol_pct": f"{score.realized_vol * 100:.3f}",
        "sigma_vol_ratio": f"{score.sigma_vol_ratio:.2f}",
        "p_up": f"{score.p_up:.3f}",
        "q05_pct": f"{score.q05 * 100:.3f}",
        "mdd_pct": f"{score.mdd * 100:.3f}",
        "p_down": f"{score.p_down:.3f}",
        "q95_pct": f"{score.q95 * 100:.3f}",
        "runup_pct": f"{score.runup * 100:.3f}",
        "horizon_days": horizon,
        "paths": paths,
        "paths_used": score.paths_used,
        "model": model,
        "scanned_at": stamp,
    }


def _write_atomic(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    try:
        with handle as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        # tempfile creates at 0600; scan results are not secret, and inheriting
        # owner-only permissions surprises anything else that reads the file.
        os.chmod(handle.name, 0o644)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def path_for(side: str) -> Path:
    return config.TOP_LONG_CSV if side == LONG else config.TOP_SHORT_CSV


def write(
    ranked: dict[str, list[Score]],
    names: dict[str, str],
    *,
    horizon: int,
    paths: int,
    model: str,
    top_n: int = config.TOP_N,
) -> tuple[dict[str, Path], Path]:
    """Overwrite both top-N CSVs and archive the full ranking.

    `ranked` maps side -> scores already sorted best-first for that side.
    Returns the per-side paths and the archive path.
    """
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    written: dict[str, Path] = {}
    archive_rows: list[dict] = []

    for side in (LONG, SHORT):
        scores = ranked.get(side, [])
        rows = [
            _row(i, s, names.get(s.symbol, ""), horizon, paths, model, stamp)
            for i, s in enumerate(scores, start=1)
        ]
        target = path_for(side)
        _write_atomic(target, rows[:top_n], FIELDS[side])
        written[side] = target
        archive_rows.extend(rows)

    archive_name = f"scan_full_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    archive = config.DATA_DIR / archive_name
    _write_atomic(archive, archive_rows, ARCHIVE_FIELDS)

    return written, archive


#: Per-symbol diagnostic columns, written for every symbol the model returned --
#: rejected ones included, which is the whole point of the file.
HEALTH_FIELDS = [
    "symbol", "mu_pct", "sigma_pct", "realized_vol_pct",
    "mu_vol_ratio", "sigma_vol_ratio", "paths_used", "paths_total",
    "rejected", "reason",
]


def write_health(entries, *, timeframe: str | None, horizon: int) -> Path:
    """Archive per-symbol forecast health for one run.

    Named by timeframe and timestamp so a 1Hour run and its daily comparison
    sit side by side and can be diffed, rather than one overwriting the other.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = timeframe or "1Day"
    path = config.DATA_DIR / f"scan_health_{tag}_{horizon}bar_{stamp}.csv"
    rows = [
        {
            "symbol": e.symbol,
            "mu_pct": f"{e.mu * 100:.3f}",
            "sigma_pct": f"{e.sigma * 100:.3f}",
            "realized_vol_pct": f"{e.realized_vol * 100:.3f}",
            "mu_vol_ratio": f"{e.mu_vol_ratio:.3f}",
            "sigma_vol_ratio": f"{e.sigma_vol_ratio:.3f}",
            "paths_used": e.paths_used,
            "paths_total": e.paths_total,
            "rejected": int(e.rejected),
            "reason": e.reason,
        }
        for e in sorted(entries, key=lambda e: e.symbol)
    ]
    _write_atomic(path, rows, HEALTH_FIELDS)
    return path


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


def read(side: str = LONG) -> list[dict]:
    """Read a side's top-N CSV. Raises FileNotFoundError if absent."""
    with path_for(side).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def age_hours(side: str = LONG) -> float | None:
    """Hours since the scan that produced a side's CSV, from its own data."""
    try:
        rows = read(side)
    except FileNotFoundError:
        return None
    if not rows or not rows[0].get("scanned_at"):
        return None
    scanned = datetime.fromisoformat(rows[0]["scanned_at"])
    if scanned.tzinfo is None:
        scanned = scanned.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - scanned).total_seconds() / 3600.0
