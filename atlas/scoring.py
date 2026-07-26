"""Turn sampled price paths into a ranked, auditable score.

The composite is expected return, penalised by the downside tail, per unit of
forecast dispersion:

    score = (mu - LAMBDA * max(0, -q05)) / max(sigma, eps)

`mu` alone would rank a coin-flip with a fat right tail above a steady grinder.
Dividing by `sigma` fixes that; subtracting the tail penalty stops a symbol
whose 5th-percentile path is a collapse from scoring well just because its mean
is positive. The score is absolute rather than z-scored across the universe, so
today's 2.4 means the same thing as last week's 2.4.

Every component is written to the CSV alongside the score, so the ranking can
be re-derived or the weights retuned without re-running the model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from . import config
from .forecast import SymbolPaths


class InvalidForecast(ValueError):
    """The model's output for this symbol is not usable as a forecast."""


@dataclass
class Score:
    symbol: str
    last_close: float
    score: float
    signal: str
    mu: float  # expected return over the horizon, as a fraction
    p_up: float  # fraction of paths ending above the last close
    sigma: float  # dispersion of terminal returns
    q05: float  # 5th-percentile terminal return
    mdd: float  # mean worst intra-path drawdown from the last close
    sharpe: float  # mu / sigma
    paths_used: int = 0  # valid paths the statistics were computed from
    #: |mu| as a multiple of the symbol's realized volatility over the horizon.
    #: A plausibility yardstick: above ~2.5 the forecast is straining against
    #: what the symbol has historically been capable of in five sessions.
    mu_vol_ratio: float = 0.0

    def as_row(self) -> dict:
        return asdict(self)


def classify(mu: float, p_up: float) -> str:
    if mu >= config.BUY_MIN_MU and p_up >= config.BUY_MIN_P_UP:
        return "BUY"
    if mu <= config.AVOID_MAX_MU:
        return "AVOID"
    return "HOLD"


def score_paths(paths: SymbolPaths, *, lam: float = config.LAMBDA_DOWNSIDE) -> Score:
    """Reduce sampled paths to statistics, rejecting unusable model output.

    Two guards, both raising `InvalidForecast` so the caller can record a
    reason rather than let a broken number into the ranking:

    1. Paths containing a non-positive or non-finite price are discarded --
       see `SymbolPaths.valid_mask`. If too few survive, the symbol is rejected.
    2. A mean move exceeding `MAX_MU_VOL_MULTIPLE` times the symbol's own
       realized volatility over the same horizon is treated as an artifact of
       Kronos's window normalization, not a forecast. Empirically this fires on
       strongly-trending names, whose multi-year window standard deviation is
       dominated by trend rather than by short-term volatility, so one unit of
       sampling noise in normalized space becomes an implausible price move.
    """
    p0 = paths.last_close
    if not np.isfinite(p0) or p0 <= 0:
        raise InvalidForecast(f"invalid last close {p0!r}")

    valid = paths.valid_mask()
    n_valid = int(valid.sum())
    required = max(
        config.MIN_VALID_PATHS,
        math.ceil(config.MIN_VALID_PATH_FRACTION * len(valid)),
    )
    if n_valid < required:
        raise InvalidForecast(
            f"only {n_valid} of {len(valid)} sampled paths were numerically valid"
        )

    closes = paths.closes[valid]
    lows = paths.lows[valid]

    terminal = closes[:, -1] / p0 - 1.0
    mu = float(np.mean(terminal))
    sigma = float(np.std(terminal))
    p_up = float(np.mean(terminal > 0.0))
    q05 = float(np.percentile(terminal, 5))
    mdd = float(np.mean(np.min(lows, axis=1) / p0 - 1.0))
    sharpe = mu / (sigma + 1e-6)

    if paths.realized_vol >= config.MIN_VOL_FOR_GUARD:
        limit = max(
            config.MIN_PLAUSIBLE_MU,
            config.MAX_MU_VOL_MULTIPLE * paths.realized_vol,
        )
        if abs(mu) > limit:
            raise InvalidForecast(
                f"implausible mean move {mu * 100:+.1f}% against realized "
                f"{paths.realized_vol * 100:.2f}% volatility over the horizon"
            )

    downside_penalty = lam * max(0.0, -q05)
    score = (mu - downside_penalty) / max(sigma, 1e-4)

    return Score(
        symbol=paths.symbol,
        last_close=p0,
        score=score,
        signal=classify(mu, p_up),
        mu=mu,
        p_up=p_up,
        sigma=sigma,
        q05=q05,
        mdd=mdd,
        sharpe=sharpe,
        paths_used=n_valid,
        mu_vol_ratio=(abs(mu) / paths.realized_vol if paths.realized_vol > 0 else 0.0),
    )


def rank(scores: list[Score]) -> list[Score]:
    """Best first. Ties break on symbol so the order is stable across runs."""
    return sorted(scores, key=lambda s: (-s.score, s.symbol))
