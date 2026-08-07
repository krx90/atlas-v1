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
    #: The symbol's own realized volatility over the horizon, from its history.
    realized_vol: float = 0.0
    #: Forecast dispersion as a multiple of realized volatility. ~1.0 means the
    #: model's stated uncertainty matches how much the symbol actually moves.
    #: Well below 1 is the dangerous failure: confident, tight and wrong.
    sigma_vol_ratio: float = 0.0
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


@dataclass
class Health:
    """What the model produced for one symbol, measured *before* the guards.

    Deliberately guard-independent, because the diagnostic quantities and the
    rejection criterion are the same number. `mu_vol_ratio` is exactly what the
    3x guard rejects on, so a median taken over scored symbols alone is
    censored at 3.0 and cannot rise above it however badly the model behaves --
    the reference "broken" value of 4.80 is unreachable by construction. Only a
    median over every symbol the model returned can move.

    `rejected` records what the guards would decide, so the rejection rate and
    the ratios come from one pass over one population.
    """

    symbol: str
    mu: float
    sigma: float
    realized_vol: float
    mu_vol_ratio: float
    sigma_vol_ratio: float
    paths_used: int
    paths_total: int
    rejected: bool
    reason: str = ""


def _validity_reason(n_valid: int, n_total: int) -> str | None:
    """Too few numerically usable paths to compute statistics from."""
    required = max(
        config.MIN_VALID_PATHS,
        math.ceil(config.MIN_VALID_PATH_FRACTION * n_total),
    )
    if n_valid < required:
        return f"only {n_valid} of {n_total} sampled paths were numerically valid"
    return None


def _plausibility_reason(mu: float, realized_vol: float) -> str | None:
    """A mean move too large to be a forecast of this particular symbol."""
    if realized_vol < config.MIN_VOL_FOR_GUARD:
        return None
    limit = max(config.MIN_PLAUSIBLE_MU, config.MAX_MU_VOL_MULTIPLE * realized_vol)
    if abs(mu) > limit:
        return (
            f"implausible mean move {mu * 100:+.1f}% against realized "
            f"{realized_vol * 100:.2f}% volatility over the horizon"
        )
    return None


def _ratio(numerator: float, denominator: float) -> float:
    return abs(numerator) / denominator if denominator > 0 else 0.0


def health(paths: SymbolPaths) -> Health:
    """Diagnostic statistics for one symbol, whatever the guards would decide.

    Never raises. A symbol whose paths are too broken to measure comes back
    with NaN statistics and `rejected=True`, so it still counts in the
    rejection rate rather than vanishing from both numerator and denominator.
    """
    p0 = paths.last_close
    valid = paths.valid_mask()
    n_valid, n_total = int(valid.sum()), int(valid.size)
    rv = paths.realized_vol

    nan = float("nan")
    if not np.isfinite(p0) or p0 <= 0:
        return Health(paths.symbol, nan, nan, rv, nan, nan, n_valid, n_total, True,
                      f"invalid last close {p0!r}")

    reason = _validity_reason(n_valid, n_total)
    if n_valid < 2:
        # Fewer than two paths has no standard deviation to speak of.
        return Health(paths.symbol, nan, nan, rv, nan, nan, n_valid, n_total, True,
                      reason or "fewer than two valid paths")

    terminal = paths.closes[valid][:, -1] / p0 - 1.0
    mu = float(np.mean(terminal))
    sigma = float(np.std(terminal))
    reason = reason or _plausibility_reason(mu, rv)

    return Health(
        symbol=paths.symbol,
        mu=mu,
        sigma=sigma,
        realized_vol=rv,
        mu_vol_ratio=_ratio(mu, rv),
        sigma_vol_ratio=_ratio(sigma, rv),
        paths_used=n_valid,
        paths_total=n_total,
        rejected=reason is not None,
        reason=reason or "",
    )


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
    invalid = _validity_reason(n_valid, len(valid))
    if invalid:
        raise InvalidForecast(invalid)

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

    implausible = _plausibility_reason(mu, paths.realized_vol)
    if implausible:
        raise InvalidForecast(implausible)

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
        mu_vol_ratio=_ratio(mu, paths.realized_vol),
        realized_vol=paths.realized_vol,
        sigma_vol_ratio=_ratio(sigma, paths.realized_vol),
        side=side,
        p_down=p_down,
        q95=q95,
        runup=runup,
    )


#: Thresholds at which each diagnostic is "on the broken side". Set between the
#: measured healthy and broken reference points rather than at either, so a
#: number has to move most of the way across before it counts.
#:
#:   metric              healthy   broken   threshold
#:   rejection rate       ~0.18     ~0.68        0.40
#:   median |mu|/vol      ~1.06     ~4.80        2.50
#:   median sigma/vol     ~1.00     ~0.35        0.60
CONCERN_REJECTION_RATE = 0.40
CONCERN_MU_VOL = 2.50
CONCERN_SIGMA_VOL = 0.60
#: Below this many measured symbols, one bad number is more likely sample noise
#: than signal -- a 17-symbol correlation of -0.73 once collapsed to -0.09 on 113.
MIN_TRUSTWORTHY_SAMPLE = 50


@dataclass
class HealthSummary:
    """The three diagnostics, read as a pattern rather than individually."""

    n: int
    rejected: int
    rejection_rate: float
    median_mu_vol: float
    median_sigma_vol: float

    @property
    def concerns(self) -> list[str]:
        """Which diagnostics are on the broken side of their threshold."""
        out = []
        if self.rejection_rate > CONCERN_REJECTION_RATE:
            out.append("rejection rate")
        if self.median_mu_vol > CONCERN_MU_VOL:
            out.append("median |mu|/vol")
        # Weighted most heavily: dispersion far below realized volatility is the
        # base-model failure signature -- forecasts that look clean enough to
        # trade and are confidently wrong.
        if self.median_sigma_vol < CONCERN_SIGMA_VOL:
            out.append("sigma/realized")
        return out

    @property
    def verdict(self) -> str:
        concerns = self.concerns
        under = self.n < MIN_TRUSTWORTHY_SAMPLE
        caveat = f" -- but {self.n} symbols is small; widen to ~100 before acting" if under else ""

        if not concerns:
            return "healthy -- all three diagnostics are in the normal range"
        if len(concerns) == 3:
            return "broken -- all three diagnostics are on the failure side" + caveat
        if concerns == ["sigma/realized"]:
            return (
                "suspect -- dispersion is far below realized volatility, which is the "
                "confidently-wrong signature; treat as broken unless a wider sample "
                "disagrees" + caveat
            )
        return (
            f"ambiguous -- {len(concerns)} of 3 on the failure side ({', '.join(concerns)})"
            + (caveat or " -- widen to ~100 symbols before concluding")
        )


def summarize_health(entries: list[Health]) -> HealthSummary:
    """Reduce per-symbol health to the three headline numbers.

    The medians are taken over every symbol the model returned, including
    rejected ones -- see `Health`. Symbols too broken to yield statistics at all
    still count toward the rejection rate but carry NaN, so they are excluded
    from the medians rather than dragging them to NaN.
    """
    if not entries:
        return HealthSummary(0, 0, float("nan"), float("nan"), float("nan"))

    rejected = sum(1 for e in entries if e.rejected)
    mu_ratios = [e.mu_vol_ratio for e in entries if np.isfinite(e.mu_vol_ratio)]
    sigma_ratios = [e.sigma_vol_ratio for e in entries if np.isfinite(e.sigma_vol_ratio)]

    return HealthSummary(
        n=len(entries),
        rejected=rejected,
        rejection_rate=rejected / len(entries),
        median_mu_vol=float(np.median(mu_ratios)) if mu_ratios else float("nan"),
        median_sigma_vol=float(np.median(sigma_ratios)) if sigma_ratios else float("nan"),
    )


def rank(scores: list[Score]) -> list[Score]:
    """Best first. Ties break on symbol so the order is stable across runs."""
    return sorted(scores, key=lambda s: (-s.score, s.symbol))
