"""Turn sampled price paths into a ranked, auditable score, long or short.

The composite is expected return in the direction of the trade, penalised by
the tail that would hurt that trade, per unit of forecast dispersion:

    long  score = ( mu - LAMBDA * max(0, -q05)) / max(sigma, eps)
    short score = (-mu - LAMBDA * max(0,  q95)) / max(sigma, eps)

`mu` alone would rank a coin-flip with a fat tail above a steady grinder.
Dividing by `sigma` fixes that; subtracting the tail penalty stops a symbol
whose worst-case path is a disaster from scoring well just because its mean
points the right way. The score is absolute rather than z-scored across the
universe, so today's 2.4 means the same thing as last week's 2.4.

The two sides are mirror images because their risk is: a long is hurt by the
5th-percentile outcome and by the worst low along the way, a short by the
95th-percentile outcome and the worst high. Both sides are read off the *same*
sampled paths, so scoring both costs nothing beyond the one forward pass.

A `Score` always carries the full distribution -- both sides' statistics -- and
only `score`, `signal` and `side` depend on the direction asked for. That keeps
the archive complete enough to re-rank either way without re-running the model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from . import config
from .forecast import SymbolPaths


class InvalidForecast(ValueError):
    """The model's output for this symbol is not usable as a forecast."""


LONG = "long"
SHORT = "short"


@dataclass
class Score:
    symbol: str
    last_close: float
    score: float
    signal: str
    mu: float  # expected return over the horizon, as a fraction
    p_up: float  # fraction of paths ending above the last close
    sigma: float  # dispersion of terminal returns
    q05: float  # 5th-percentile terminal return -- a long's bad case
    mdd: float  # mean worst intra-path drawdown -- a long's worst moment
    sharpe: float  # mu / sigma
    paths_used: int = 0  # valid paths the statistics were computed from
    #: |mu| as a multiple of the symbol's realized volatility over the horizon.
    #: A plausibility yardstick: above ~2.5 the forecast is straining against
    #: what the symbol has historically been capable of in five sessions.
    mu_vol_ratio: float = 0.0
    side: str = LONG
    p_down: float = 0.0  # fraction of paths ending below the last close
    q95: float = 0.0  # 95th-percentile terminal return -- a short's bad case
    runup: float = 0.0  # mean worst intra-path run-up -- a short's worst moment

    def as_row(self) -> dict:
        return asdict(self)


def classify(mu: float, p_favourable: float, side: str = LONG) -> str:
    """BUY/SHORT when the edge and the odds both clear their floors.

    Mirrored for shorts: the edge is `-mu` and the odds are `p_down`, so a
    symbol expected to fall with consistent agreement is a SHORT exactly as its
    inverse would be a BUY.
    """
    edge = mu if side == LONG else -mu
    entry = "BUY" if side == LONG else "SHORT"
    if edge >= config.BUY_MIN_MU and p_favourable >= config.BUY_MIN_P_UP:
        return entry
    if edge <= config.AVOID_MAX_MU:
        return "AVOID"
    return "HOLD"


def score_paths(
    paths: SymbolPaths, *, side: str = LONG, lam: float = config.LAMBDA_DOWNSIDE
) -> Score:
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
    highs = paths.highs[valid]

    terminal = closes[:, -1] / p0 - 1.0
    mu = float(np.mean(terminal))
    sigma = float(np.std(terminal))
    sharpe = mu / (sigma + 1e-6)

    # Both sides' statistics, always. p_down is computed rather than derived as
    # 1 - p_up: a path landing exactly flat is favourable to neither.
    p_up = float(np.mean(terminal > 0.0))
    p_down = float(np.mean(terminal < 0.0))
    q05 = float(np.percentile(terminal, 5))
    q95 = float(np.percentile(terminal, 95))
    mdd = float(np.mean(np.min(lows, axis=1) / p0 - 1.0))
    runup = float(np.mean(np.max(highs, axis=1) / p0 - 1.0))

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

    # Edge in the direction of the trade, less the tail that would hurt it.
    if side == LONG:
        edge, adverse_tail, p_favourable = mu, -q05, p_up
    elif side == SHORT:
        edge, adverse_tail, p_favourable = -mu, q95, p_down
    else:
        raise ValueError(f"side must be {LONG!r} or {SHORT!r}, got {side!r}")

    score = (edge - lam * max(0.0, adverse_tail)) / max(sigma, 1e-4)

    return Score(
        symbol=paths.symbol,
        last_close=p0,
        score=score,
        signal=classify(mu, p_favourable, side),
        mu=mu,
        p_up=p_up,
        sigma=sigma,
        q05=q05,
        mdd=mdd,
        sharpe=sharpe,
        paths_used=n_valid,
        mu_vol_ratio=(abs(mu) / paths.realized_vol if paths.realized_vol > 0 else 0.0),
        side=side,
        p_down=p_down,
        q95=q95,
        runup=runup,
    )


def rank(scores: list[Score]) -> list[Score]:
    """Best first. Ties break on symbol so the order is stable across runs."""
    return sorted(scores, key=lambda s: (-s.score, s.symbol))
