"""Does the ranking predict anything?

Everything else in Atlas -- the scoring weights, the horizon, the lookback, the
guard thresholds, the short side -- rests on reasoning rather than evidence.
This module supplies the evidence, or the absence of it.

The method is a walk-forward rank test. At each as-of date the model sees only
bars up to and including that date, produces a ranking, and is then judged
against what actually happened over the following `horizon` sessions:

    rank IC   Spearman correlation between score and realized forward return,
              computed per date. This is the headline number -- a positive mean
              IC with a t-statistic above ~2 is weak evidence of skill, and
              anything near zero means the ranking carries no information.

    spread    mean forward return of the top decile minus the bottom decile,
              which is what a long/short book would actually have earned before
              costs.

    hit rate  fraction of dates on which the top decile beat the bottom decile.

**Lookahead is the failure mode that makes a backtest lie**, so the slicing is
deliberately explicit: `history` ends at index `i` inclusive, and the realized
return is measured from `i` to `i + horizon`, which the model never saw. See
`slice_at` and its tests.

Known biases, stated because they inflate results rather than deflate them:

- **Survivorship.** The universe is built from symbols liquid *today*. Names
  that were liquid at an as-of date but have since delisted are absent, and
  those are disproportionately the losers. Alpaca's asset list is current-state
  only, so this cannot be corrected here.
- **No costs.** Spread, slippage and borrow fees are not modelled. At a 5-day
  horizon these are a material fraction of any edge.
- **One market regime.** A few months of one market is not a test of a strategy.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import bars, config, forecast, scoring


@dataclass
class Observation:
    """One symbol, one as-of date: what was predicted and what happened."""

    date: str
    symbol: str
    score: float
    mu: float
    side: str
    forward_return: float


@dataclass
class Window:
    """A model input and its future, sliced without lookahead."""

    history: pd.DataFrame
    last_close: float
    realized_vol: float
    forward_return: float


def slice_at(frame: pd.DataFrame, i: int, lookback: int, horizon: int) -> Window | None:
    """Build the as-of window ending at row `i` and the return that follows it.

    `i` is the last bar the model may see. The forward return spans `i` to
    `i + horizon`, which is strictly in the future relative to the history.
    Returns None when there is not enough data on either side.
    """
    if i - lookback + 1 < 0 or i + horizon >= len(frame):
        return None

    history = frame.iloc[i - lookback + 1 : i + 1]
    if len(history) != lookback:
        return None

    closes = history["close"].to_numpy()
    last_close = float(closes[-1])
    future_close = float(frame["close"].iloc[i + horizon])
    if not last_close > 0 or not future_close > 0:
        return None

    return Window(
        history=forecast.prepare_history(history),
        last_close=last_close,
        realized_vol=forecast.realized_vol(closes, horizon),
        forward_return=future_close / last_close - 1.0,
    )


def as_of_indices(n_bars: int, count: int, spacing: int, lookback: int, horizon: int) -> list[int]:
    """Evenly spaced as-of rows, newest first, that leave room on both sides.

    The newest usable row is `n_bars - horizon - 1`, because a later one would
    have no realized outcome to be judged against.
    """
    newest = n_bars - horizon - 1
    oldest = lookback - 1
    indices = [newest - k * spacing for k in range(count)]
    return sorted(i for i in indices if i >= oldest)


def build(
    conn: sqlite3.Connection,
    symbols: list[str],
    *,
    dates: int,
    spacing: int,
    lookback: int,
    horizon: int,
    timeframe: str | None = None,
) -> tuple[dict[str, list[tuple[str, Window]]], list[str]]:
    """Slice every (symbol, as-of point) pair up front.

    `timeframe` of None means daily bars; otherwise an intraday timeframe such
    as '5Min'. Everything downstream is timeframe-agnostic -- an as-of point is
    a row index either way, and the horizon is a bar count, not a day count.

    Returns windows keyed by symbol, plus the sorted list of as-of stamps. Doing
    this before any forecasting keeps the expensive loop free of data handling.
    """
    windows: dict[str, list[tuple[str, Window]]] = {}
    seen_dates: set[str] = set()

    for symbol in symbols:
        frame = (
            bars.load(conn, symbol)
            if timeframe is None
            else bars.load_intraday(conn, symbol, timeframe)
        )
        if len(frame) < lookback + horizon + 1:
            continue
        rows: list[tuple[str, Window]] = []
        for i in as_of_indices(len(frame), dates, spacing, lookback, horizon):
            window = slice_at(frame, i, lookback, horizon)
            if window is None:
                continue
            # Daily as-of points are a date; intraday ones need the time too,
            # or every point in a session would collide into one group.
            moment = frame.index[i]
            stamp = str(moment.date()) if timeframe is None else moment.strftime("%Y-%m-%d %H:%M")
            rows.append((stamp, window))
            seen_dates.add(stamp)
        if rows:
            windows[symbol] = rows

    return windows, sorted(seen_dates)


def observations(
    engine: forecast.KronosEngine,
    windows: dict[str, list[tuple[str, Window]]],
    *,
    horizon: int,
    paths: int,
    side: str = scoring.LONG,
    on_progress=None,
) -> list[Observation]:
    """Forecast every (symbol, date) pair and pair it with what happened.

    Grouped by date so each batch is one point in time -- it keeps the progress
    readable and makes a partial run still cover whole dates.
    """
    by_date: dict[str, list[tuple[str, Window]]] = {}
    for symbol, rows in windows.items():
        for stamp, window in rows:
            by_date.setdefault(stamp, []).append((symbol, window))

    out: list[Observation] = []
    for stamp in sorted(by_date):
        entries = by_date[stamp]
        requests = [
            forecast.ForecastRequest(sym, w.history, w.last_close, realized_vol=w.realized_vol)
            for sym, w in entries
        ]
        futures = {sym: w.forward_return for sym, w in entries}

        for symbol_paths in engine.forecast(requests, horizon=horizon, paths=paths):
            # Report per symbol, not per date. A date takes minutes, and a
            # progress signal that only moves that often is indistinguishable
            # from a stalled run.
            if on_progress is not None:
                on_progress(stamp, 1)
            try:
                score = scoring.score_paths(symbol_paths, side=side)
            except scoring.InvalidForecast:
                continue  # same guards as a live scan; excluded, not scored 0
            out.append(
                Observation(
                    date=stamp,
                    symbol=symbol_paths.symbol,
                    score=score.score,
                    mu=score.mu,
                    side=side,
                    forward_return=futures[symbol_paths.symbol],
                )
            )

    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without a scipy dependency in the main package."""
    if len(a) < 3:
        return float("nan")
    ar = pd.Series(a).rank().to_numpy()
    br = pd.Series(b).rank().to_numpy()
    if np.std(ar) == 0 or np.std(br) == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


@dataclass
class Result:
    dates: int
    observations: int
    mean_ic: float
    ic_t_stat: float
    ic_by_date: list[tuple[str, float, int]]
    mean_spread: float
    hit_rate: float
    top_return: float
    bottom_return: float

    @property
    def verdict(self) -> str:
        """A blunt reading, so the numbers are not over-interpreted."""
        if not np.isfinite(self.ic_t_stat):
            return "inconclusive -- too few dates to say anything"
        if abs(self.ic_t_stat) < 2.0:
            return (
                "no detectable skill -- the mean IC is not distinguishable from zero "
                "at this sample size"
            )
        return "positive" if self.mean_ic > 0 else "negative -- the ranking is inverted"


def evaluate(obs: list[Observation], *, decile: float = 0.1) -> Result:
    """Rank IC per date, plus the top-minus-bottom decile spread."""
    frame = pd.DataFrame([o.__dict__ for o in obs])
    ics: list[tuple[str, float, int]] = []
    spreads: list[float] = []
    tops: list[float] = []
    bottoms: list[float] = []

    for stamp, group in frame.groupby("date"):
        if len(group) < 5:
            continue
        ic = _spearman(group["score"].to_numpy(), group["forward_return"].to_numpy())
        if np.isfinite(ic):
            ics.append((stamp, ic, len(group)))

        ordered = group.sort_values("score", ascending=False)
        k = max(1, int(round(len(ordered) * decile)))
        top = float(ordered["forward_return"].head(k).mean())
        bottom = float(ordered["forward_return"].tail(k).mean())
        tops.append(top)
        bottoms.append(bottom)
        spreads.append(top - bottom)

    ic_values = np.array([v for _s, v, _n in ics])
    if len(ic_values) > 1 and np.std(ic_values, ddof=1) > 0:
        t_stat = float(np.mean(ic_values) / (np.std(ic_values, ddof=1) / np.sqrt(len(ic_values))))
    else:
        t_stat = float("nan")

    return Result(
        dates=len(ics),
        observations=len(frame),
        mean_ic=float(np.mean(ic_values)) if len(ic_values) else float("nan"),
        ic_t_stat=t_stat,
        ic_by_date=ics,
        mean_spread=float(np.mean(spreads)) if spreads else float("nan"),
        hit_rate=float(np.mean([s > 0 for s in spreads])) if spreads else float("nan"),
        top_return=float(np.mean(tops)) if tops else float("nan"),
        bottom_return=float(np.mean(bottoms)) if bottoms else float("nan"),
    )
