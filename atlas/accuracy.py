"""Is the forecast *accurate*, as distinct from usefully ordered?

`backtest.evaluate` answers one narrow question: does the score rank symbols in
the order their forward returns arrive in? That is the right question for a
long/short book, and it is deliberately blind to three others that decide
whether the model is worth running at all:

    direction     does the sign of `mu` match the sign of what happened?
    magnitude     is `mu` closer to the truth than forecasting no move at all?
    the label     does BUY actually outperform AVOID, which is what a user acts on?

A model can rank well and be directionally useless -- ordering ten symbols
correctly says nothing about whether any of them went up. It can also produce a
respectable IC while every `mu` is an order of magnitude too large, which is
exactly the failure the plausibility guard in `scoring` was built to catch.

**Two baselines, because a bare number cannot be read.** An IC of +0.026 is
meaningless on its own; the question is always "compared to what".

    skill score       `mu`'s mean squared error against the zero forecast. A
                      value <= 0 means predicting "no move" was at least as
                      accurate, which is the honest null for a return forecast.
    permutation p     the mean IC recomputed on scores shuffled *within each
                      date*, a thousand times. This preserves the number of
                      symbols per date and the distribution of forward returns
                      and destroys only the pairing, so the resulting spread is
                      what luck alone produces on this exact sample. It makes no
                      normality assumption, unlike the IC t-statistic.
    momentum IC       the same rank test run on the trailing return. The model
                      costs 2.5 seconds a symbol-date; trailing return costs a
                      subtraction. Beating zero is not the bar -- beating this is.

Every function here is pure and takes `list[Observation]`, so the whole module
runs on a saved backtest CSV in milliseconds with no model, no GPU and no
network. That is the point: `atlas accuracy` re-derives every claim in seconds,
where producing the observations took eighty minutes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .backtest import Observation, spearman

#: Signals in the order a reader wants them: the two entries, then the two
#: non-entries. Sorting alphabetically would put AVOID first, which reads as
#: though it were the headline.
SIGNAL_ORDER = ("BUY", "SHORT", "HOLD", "AVOID")


def _two_sided_binomial(hits: int, n: int) -> float:
    """P(a fair coin is at least this lopsided over `n` flips).

    Computed in log space via `lgamma` rather than `math.comb`, which builds
    400-digit integers at the sample sizes a real backtest produces. Exact
    either way, and no scipy.
    """
    if n <= 0:
        return float("nan")
    tail = min(hits, n - hits)
    log_norm = math.lgamma(n + 1) + n * math.log(0.5)
    total = sum(
        math.exp(log_norm - math.lgamma(k + 1) - math.lgamma(n - k + 1))
        for k in range(tail + 1)
    )
    return min(1.0, 2.0 * total)


# --- direction ----------------------------------------------------------------


@dataclass
class Directional:
    """How often the forecast pointed the right way.

    Flat outcomes are excluded rather than counted as misses: a forward return
    of exactly zero is neither a hit nor a miss, and folding it into the
    denominator would drag the rate toward 50% for a reason unrelated to skill.
    """

    n: int
    hits: int
    hit_rate: float
    p_value: float  # two-sided, against a fair coin

    @property
    def verdict(self) -> str:
        if self.n < 30:
            return "too few observations to judge"
        if self.p_value > 0.05:
            return "no better than a coin flip"
        return (
            "directionally informative"
            if self.hit_rate > 0.5
            else "directionally inverted -- the sign is systematically wrong"
        )


def directional(obs: list[Observation]) -> Directional:
    """Sign agreement between `mu` and the realized return.

    Side-independent by construction: `mu` is direction-neutral, so a short book
    whose `mu` is correctly negative scores a hit exactly as a long's positive
    `mu` does.
    """
    pairs = [(o.mu, o.forward_return) for o in obs if o.mu != 0 and o.forward_return != 0]
    if not pairs:
        return Directional(0, 0, float("nan"), float("nan"))

    hits = sum(1 for mu, fwd in pairs if (mu > 0) == (fwd > 0))
    n = len(pairs)
    return Directional(n, hits, hits / n, _two_sided_binomial(hits, n))


# --- magnitude ----------------------------------------------------------------


@dataclass
class Magnitude:
    """Is `mu` a better point forecast than forecasting nothing?

    `skill` is the standard skill score, `1 - MSE_model / MSE_zero`. It is 1 for
    a perfect forecast, exactly 0 for one no better than assuming the price does
    not move, and negative -- unboundedly -- for one that is worse. For a 5-day
    equity return the zero forecast is a genuinely strong baseline, so a
    negative skill score here is the expected result rather than a bug.
    """

    n: int
    mae: float
    rmse: float
    baseline_rmse: float  # RMSE of forecasting zero
    skill: float
    mean_abs_mu: float
    mean_abs_return: float

    @property
    def verdict(self) -> str:
        if self.n < 30:
            return "too few observations to judge"
        if self.skill <= 0:
            return (
                f"worse than forecasting no move ({self.skill:+.3f} skill) -- "
                "mu carries no usable magnitude"
            )
        return f"beats the zero forecast by {self.skill * 100:.1f}% of variance"


def magnitude(obs: list[Observation]) -> Magnitude:
    """Point-forecast error of `mu`, against the zero-forecast baseline."""
    if not obs:
        nan = float("nan")
        return Magnitude(0, nan, nan, nan, nan, nan, nan)

    mu = np.array([o.mu for o in obs])
    actual = np.array([o.forward_return for o in obs])
    error = actual - mu

    mse = float(np.mean(error**2))
    baseline_mse = float(np.mean(actual**2))
    return Magnitude(
        n=len(obs),
        mae=float(np.mean(np.abs(error))),
        rmse=math.sqrt(mse),
        baseline_rmse=math.sqrt(baseline_mse),
        skill=1.0 - mse / baseline_mse if baseline_mse > 0 else float("nan"),
        # Printed together because their ratio is the diagnosis when skill is
        # bad: a mean |mu| several times the mean |return| is an overconfident
        # model, not an unlucky one.
        mean_abs_mu=float(np.mean(np.abs(mu))),
        mean_abs_return=float(np.mean(np.abs(actual))),
    )


# --- the label a user acts on -------------------------------------------------


@dataclass
class Bucket:
    """What actually happened to everything carrying one signal."""

    signal: str
    n: int
    mean_return: float
    hit_rate: float  # moved in the direction the signal implied


def by_signal(obs: list[Observation]) -> list[Bucket]:
    """Forward returns grouped by the BUY/HOLD/AVOID label.

    The most directly actionable measurement in this module, because the signal
    is what `atlas scan` prints and what a user trades on. `hit_rate` is
    relative to the signal's own claim: a BUY hits when the symbol rose, a SHORT
    when it fell, and AVOID -- which is a claim that the symbol falls -- when it
    fell. HOLD asserts nothing, so its hit rate is reported against "rose"
    purely as a reference point for the other buckets.
    """
    rows = [o for o in obs if o.signal]
    if not rows:
        return []

    wants_fall = {"SHORT", "AVOID"}
    out: list[Bucket] = []
    for signal in SIGNAL_ORDER:
        group = [o for o in rows if o.signal == signal]
        if not group:
            continue
        moved = [
            (o.forward_return < 0) if signal in wants_fall else (o.forward_return > 0)
            for o in group
        ]
        out.append(
            Bucket(
                signal=signal,
                n=len(group),
                mean_return=float(np.mean([o.forward_return for o in group])),
                hit_rate=float(np.mean(moved)),
            )
        )
    return out


# --- baselines ----------------------------------------------------------------


def _ic_per_date(frame: pd.DataFrame, column: str, min_symbols: int = 5) -> np.ndarray:
    """Rank IC of `column` against the forward return, one value per date."""
    values = []
    for _stamp, group in frame.groupby("date"):
        if len(group) < min_symbols:
            continue
        ic = spearman(group[column].to_numpy(), group["forward_return"].to_numpy())
        if np.isfinite(ic):
            values.append(ic)
    return np.array(values)


@dataclass
class Null:
    """The mean IC set against what shuffling alone produces.

    `p_value` is the fraction of permutations whose |mean IC| reached the
    observed one, so it answers "how often would luck do this well" directly
    rather than through a distributional assumption. Reported alongside
    `null_sd` because a p-value near 1 on a wide null and one on a narrow null
    mean different things about how much data there is.
    """

    mean_ic: float
    null_mean: float
    null_sd: float
    p_value: float
    permutations: int
    dates: int

    @property
    def verdict(self) -> str:
        if self.dates < 5:
            return "too few dates to permute meaningfully"
        if self.p_value > 0.05:
            return "indistinguishable from shuffled scores"
        return "beats its own shuffle" if self.mean_ic > 0 else "inverted, but consistently so"


def permutation_ic(obs: list[Observation], *, permutations: int = 1000, seed: int = 0) -> Null:
    """Empirical null for the mean rank IC, by shuffling within each date.

    Shuffling *within* a date and not across the whole sample is what makes this
    a valid null. It preserves how many symbols each date carries and the entire
    cross-section of forward returns, destroying only which score was attached
    to which outcome. Shuffling globally would additionally destroy the date
    structure and produce a null that is too narrow, flattering the model.
    """
    if not obs:
        nan = float("nan")
        return Null(nan, nan, nan, nan, permutations, 0)

    frame = pd.DataFrame([o.__dict__ for o in obs])
    observed = _ic_per_date(frame, "score")
    if len(observed) < 2:
        return Null(
            float(np.mean(observed)) if len(observed) else float("nan"),
            float("nan"), float("nan"), float("nan"), permutations, len(observed),
        )

    mean_ic = float(np.mean(observed))
    rng = np.random.default_rng(seed)
    groups = [g for _s, g in frame.groupby("date") if len(g) >= 5]

    null = np.empty(permutations)
    for i in range(permutations):
        ics = []
        for group in groups:
            shuffled = rng.permutation(group["score"].to_numpy())
            ic = spearman(shuffled, group["forward_return"].to_numpy())
            if np.isfinite(ic):
                ics.append(ic)
        null[i] = np.mean(ics) if ics else np.nan

    finite = null[np.isfinite(null)]
    # >= rather than >: a permutation that ties the observed value is evidence
    # for the null, not against it. With few dates, ties are common enough to
    # matter.
    p = float(np.mean(np.abs(finite) >= abs(mean_ic))) if len(finite) else float("nan")
    return Null(
        mean_ic=mean_ic,
        null_mean=float(np.mean(finite)) if len(finite) else float("nan"),
        null_sd=float(np.std(finite)) if len(finite) else float("nan"),
        p_value=p,
        permutations=permutations,
        dates=len(observed),
    )


def momentum_ic(obs: list[Observation]) -> float:
    """Mean rank IC of the trailing return -- the free baseline.

    NaN when the observations carry no `prior_return`, which is the case for any
    CSV written before that column existed. Silently returning 0.0 would read as
    "momentum has no edge" rather than "this was never measured".
    """
    if not obs or all(o.prior_return == 0 for o in obs):
        return float("nan")
    frame = pd.DataFrame([o.__dict__ for o in obs])
    ics = _ic_per_date(frame, "prior_return")
    return float(np.mean(ics)) if len(ics) else float("nan")


# --- the whole picture --------------------------------------------------------


@dataclass
class Accuracy:
    """Every measurement, and one blunt reading of the set.

    Deliberately conservative about calling anything skilful: three of these
    four measurements are one-sided tests of things a null model passes by
    accident often enough to matter, so the verdict requires agreement rather
    than a single number clearing a threshold.
    """

    n: int
    direction: Directional
    size: Magnitude
    null: Null
    momentum: float
    buckets: list[Bucket]

    @property
    def passes(self) -> dict[str, bool]:
        """Each pre-registered condition in `config`, and whether it was met.

        Exposed rather than folded straight into `verdict` so the output can
        print the bar next to the number it is judging, and so the tests assert
        against the same structure the user reads.
        """
        # `>=`, not `>`: a tie goes against the model. Trailing return is a
        # subtraction and the model is ~2.5 seconds a symbol-date, so matching
        # the free baseline is not a result worth reporting as one. An unmeasured
        # momentum baseline (NaN) cannot be beaten, so it does not block a pass.
        beats_momentum = not (
            np.isfinite(self.momentum) and self.momentum >= self.null.mean_ic
        )
        out = {
            "direction": (
                self.direction.p_value < config.ACCURACY_MAX_DIRECTION_P
                and self.direction.hit_rate > 0.5
            ),
            "magnitude": self.size.skill > 0,
            "ranking": (
                self.null.p_value < config.ACCURACY_MAX_PERMUTATION_P
                and self.null.mean_ic > 0
            ),
        }
        if config.ACCURACY_MUST_BEAT_MOMENTUM:
            out["vs momentum"] = beats_momentum
        return out

    @property
    def verdict(self) -> str:
        if self.n < config.ACCURACY_MIN_OBSERVATIONS:
            return (
                f"inconclusive -- fewer than {config.ACCURACY_MIN_OBSERVATIONS} observations"
            )

        passes = self.passes
        met = sum(passes.values())
        total = len(passes)

        if met == total:
            return f"clears the pre-registered bar on all {total} conditions"
        if met == 0:
            return (
                "no measurable accuracy -- direction is a coin flip, mu is worse "
                "than forecasting no move, and the IC is inside its own shuffle"
            )
        failed = ", ".join(name for name, ok in passes.items() if not ok)
        return (
            f"does not clear the bar -- {met} of {total} conditions met, "
            f"failing on {failed}"
        )


def evaluate(obs: list[Observation], *, permutations: int = 1000, seed: int = 0) -> Accuracy:
    """Run every accuracy measurement over one set of observations."""
    return Accuracy(
        n=len(obs),
        direction=directional(obs),
        size=magnitude(obs),
        null=permutation_ic(obs, permutations=permutations, seed=seed),
        momentum=momentum_ic(obs),
        buckets=by_signal(obs),
    )
