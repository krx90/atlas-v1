"""Score open positions on how much they warrant attention.

`atlas portfolio` sorts by P&L, which answers "what is down" but not "what has
moved further than it should have, given what this stock normally does". A -8%
move is a catastrophe in a utility and an ordinary day in a leveraged ETF.

So every threshold here is expressed in the symbol's *own* volatility:

    expected_move = daily_vol * sqrt(max(days_held, 1))
    pnl_sigma     = unrealised_pnl_pct / expected_move

Measured across one real portfolio, daily volatility ran 0.90% to 3.60% -- a 4x
spread -- so a fixed -8% stop meant -8.9 sigma for the quietest holding and -2.3
for the noisiest. The same number, four times stricter for the calm name.

The `sqrt(days_held)` term is what keeps a long-held position from flagging
merely for having had time to drift: volatility compounds with the square root
of time, so ten days should see ~3.2x the move of one day.

Each signal returns a `Finding` with a severity >= 0, and a position's score is
the **sum** -- so failing several checks outranks failing one badly. Four of the
five signals are mechanical; only the forecast depends on the model, which has
no demonstrated skill and is labelled accordingly wherever it appears.

Nothing here closes anything. It ranks attention; acting stays a deliberate
`atlas close XYZ`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import config


@dataclass(frozen=True)
class Finding:
    """One triggered signal: how bad, and why in words."""

    kind: str
    severity: float
    detail: str


@dataclass
class Holding:
    """Everything the signals need about one position, already gathered."""

    symbol: str
    side: str  # "long" | "short"
    pnl_pct: float  # unrealised, as a fraction
    market_value: float  # absolute dollars
    days_held: int | None
    daily_vol: float  # from cached bars
    dollar_volume: float | None = None  # median daily, from the universe
    in_universe: bool = True
    easy_to_borrow: bool = True
    forecast_mu: float | None = None  # expected return, direction-neutral
    forecast_vol_ratio: float | None = None  # |mu| / realized vol


@dataclass
class Review:
    holding: Holding
    findings: list[Finding] = field(default_factory=list)

    @property
    def score(self) -> float:
        return sum(f.severity for f in self.findings)

    @property
    def pnl_sigma(self) -> float:
        return pnl_sigma(self.holding)


def expected_move(holding: Holding) -> float:
    """How far this symbol would typically travel over the holding period.

    Floored so a barely-moving instrument cannot produce a divide-by-zero or an
    absurd sigma -- one real symbol had a realized volatility that rounded to
    0.00%, and any move at all against it would otherwise read as infinite.
    """
    days = max(holding.days_held or 1, 1)
    return max(holding.daily_vol, config.REVIEW_MIN_VOL) * math.sqrt(days)


def pnl_sigma(holding: Holding) -> float:
    """Unrealised P&L in units of the symbol's own expected move.

    Positive is in the position's favour for either side, because Alpaca already
    reports a short's P&L with the sign inverted relative to price.
    """
    return holding.pnl_pct / expected_move(holding)


# --- signals ------------------------------------------------------------------
#
# Each returns a Finding or None. Pure functions over a Holding, so they are
# testable without an Alpaca account or a model.


def adverse_excursion(holding: Holding) -> Finding | None:
    """Losing by more than the symbol's normal move warrants."""
    sigma = pnl_sigma(holding)
    excess = -sigma - config.REVIEW_STOP_SIGMA
    if excess <= 0:
        return None
    return Finding(
        "stop",
        excess,
        f"down {sigma:.1f}s, past the {config.REVIEW_STOP_SIGMA:.1f}s stop",
    )


def profit_target(holding: Holding) -> Finding | None:
    """Winning by more than expected -- a candidate to bank, not a problem."""
    sigma = pnl_sigma(holding)
    excess = sigma - config.REVIEW_TARGET_SIGMA
    if excess <= 0:
        return None
    return Finding(
        "target",
        excess,
        f"up {sigma:.1f}s, past the {config.REVIEW_TARGET_SIGMA:.1f}s target",
    )


def liquidity_exit(holding: Holding) -> Finding | None:
    """No longer investable, or -- worse -- no longer borrowable.

    A short that loses `easy_to_borrow` is the one genuinely forced exit here:
    the borrow can be recalled and the position closed *for* you, at a price you
    do not choose. It is scored above an ordinary universe exit for that reason.
    """
    if holding.side == "short" and not holding.easy_to_borrow:
        return Finding(
            "borrow",
            config.REVIEW_BORROW_SEVERITY,
            "short is no longer easy to borrow -- recall risk",
        )
    if not holding.in_universe:
        return Finding(
            "liquidity",
            config.REVIEW_UNIVERSE_SEVERITY,
            "no longer passes the liquidity screen",
        )
    return None


def costly_to_exit(holding: Holding) -> Finding | None:
    """Position large relative to what trades in a day.

    Deliberately not folded into the stop level: volume does not say whether to
    exit, it says what exiting will cost. That is information you want before
    acting, not a reason to act.
    """
    if not holding.dollar_volume or holding.dollar_volume <= 0:
        return None
    share = abs(holding.market_value) / holding.dollar_volume
    if share < config.REVIEW_ADV_SHARE:
        return None
    return Finding(
        "illiquid",
        share / config.REVIEW_ADV_SHARE,
        f"{share * 100:.1f}% of median daily volume -- costly to exit",
    )


def forecast_against(holding: Holding) -> Finding | None:
    """The model now expects the position to move against you.

    The weakest signal in this module. Two backtests found no detectable skill
    (IC -0.008 and +0.026, both p > 0.4), so this is reported with a marker and
    a footnote rather than mixed in silently.
    """
    if holding.forecast_mu is None or holding.forecast_vol_ratio is None:
        return None
    # mu is direction-neutral; a short profits when it is negative.
    against = -holding.forecast_mu if holding.side == "short" else holding.forecast_mu
    if against >= 0:
        return None
    severity = min(holding.forecast_vol_ratio, config.REVIEW_FORECAST_CAP)
    if severity < config.REVIEW_FORECAST_MIN:
        return None
    return Finding(
        "forecast*",
        severity,
        f"forecast {holding.forecast_mu * 100:+.1f}% opposes a {holding.side}",
    )


SIGNALS = (
    adverse_excursion,
    profit_target,
    liquidity_exit,
    costly_to_exit,
    forecast_against,
)


def review(holding: Holding) -> Review:
    findings = [f for signal in SIGNALS if (f := signal(holding)) is not None]
    return Review(holding, findings)


def rank(holdings: list[Holding]) -> list[Review]:
    """Most in need of attention first. Ties break on symbol for stability."""
    reviews = [review(h) for h in holdings]
    return sorted(reviews, key=lambda r: (-r.score, r.holding.symbol))
