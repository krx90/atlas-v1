"""Scoring statistics against paths with known distributions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atlas.forecast import SymbolPaths
from atlas.scoring import InvalidForecast, Score, classify, rank, score_paths


def paths_from_terminal(terminal_returns, *, p0=100.0, lows=None, vol=0.0) -> SymbolPaths:
    """Build a SymbolPaths whose terminal closes give exactly these returns.

    `vol` defaults to 0, which disables the plausibility guard, so tests of the
    statistics themselves are not entangled with it.
    """
    terminal = np.asarray(terminal_returns, dtype="float64")
    closes = np.column_stack([np.full_like(terminal, p0), p0 * (1.0 + terminal)])
    low_array = (
        np.asarray(lows, dtype="float64").reshape(-1, 1) * np.ones((1, 2))
        if lows is not None
        else closes
    )
    return SymbolPaths("TEST", p0, closes, low_array, realized_vol=vol)


def test_statistics_match_the_sample():
    returns = [-0.02, 0.00, 0.01, 0.03, 0.05]
    result = score_paths(paths_from_terminal(returns))

    assert result.mu == pytest.approx(np.mean(returns))
    assert result.sigma == pytest.approx(np.std(returns))
    assert result.q05 == pytest.approx(np.percentile(returns, 5))
    # Three of five strictly above zero; the flat path does not count as up.
    assert result.p_up == pytest.approx(0.6)


def test_sharpe_is_mu_over_sigma():
    result = score_paths(paths_from_terminal([0.01, 0.02, 0.03]))
    assert result.sharpe == pytest.approx(result.mu / (result.sigma + 1e-6), rel=1e-6)


def test_downside_penalty_only_applies_to_a_negative_tail():
    # Same mean and spread, but one distribution's 5th percentile is positive.
    all_positive = score_paths(paths_from_terminal([0.02, 0.03, 0.04, 0.05, 0.06]))
    assert all_positive.q05 > 0
    assert all_positive.score == pytest.approx(all_positive.mu / all_positive.sigma, rel=1e-6)

    with_tail = score_paths(paths_from_terminal([-0.06, 0.03, 0.04, 0.05, 0.14]))
    assert with_tail.q05 < 0
    expected = (with_tail.mu - 0.5 * abs(with_tail.q05)) / with_tail.sigma
    assert with_tail.score == pytest.approx(expected, rel=1e-6)


def test_tighter_dispersion_scores_higher_at_equal_mean():
    tight = score_paths(paths_from_terminal([0.019, 0.020, 0.021]))
    loose = score_paths(paths_from_terminal([-0.06, 0.020, 0.10]))
    assert tight.mu == pytest.approx(loose.mu, abs=1e-9)
    assert tight.score > loose.score


def test_drawdown_uses_the_worst_low_per_path():
    result = score_paths(paths_from_terminal([0.05, 0.05], p0=100.0, lows=[90.0, 80.0]))
    # mean of (-10%, -20%)
    assert result.mdd == pytest.approx(-0.15)


def test_zero_dispersion_does_not_divide_by_zero():
    result = score_paths(paths_from_terminal([0.02, 0.02, 0.02]))
    assert np.isfinite(result.score)
    assert result.sigma == pytest.approx(0.0)


def test_invalid_last_close_is_rejected():
    bad = SymbolPaths("TEST", 0.0, np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(InvalidForecast, match="invalid last close"):
        score_paths(bad)


# --- guards against unusable model output -----------------------------------
#
# Kronos denormalizes with the input window's own mean/std, so a high-dispersion
# window can produce negative prices or implausible mean moves. Both were
# observed on real data: a -388% mean return, and 22 of 113 symbols with
# |mu| > 15% at a 512-bar lookback.


def test_non_positive_paths_are_excluded_from_the_statistics():
    returns = [0.01, 0.02, 0.03] * 5  # 15 valid paths
    good = paths_from_terminal(returns)
    clean = score_paths(good)

    # Add one path that collapses through zero. It must not move the mean.
    closes = np.vstack([good.closes, np.array([100.0, -271.0])])
    lows = np.vstack([good.lows, np.array([100.0, -271.0])])
    contaminated = SymbolPaths("TEST", 100.0, closes, lows)

    result = score_paths(contaminated)
    assert result.paths_used == len(returns)
    assert result.mu == pytest.approx(clean.mu)


def test_symbol_is_rejected_when_too_few_paths_survive():
    closes = np.column_stack([np.full(12, 100.0), np.full(12, -5.0)])
    with pytest.raises(InvalidForecast, match="numerically valid"):
        score_paths(SymbolPaths("TEST", 100.0, closes, closes))


def test_the_valid_path_requirement_is_a_fraction_not_a_fixed_count():
    """`--paths 5` must behave like `--paths 25`, not reject everything."""
    for n in (5, 25):
        result = score_paths(paths_from_terminal([0.01, 0.02] * (n // 2) + [0.03] * (n % 2)))
        assert result.paths_used == n

    # Half the paths invalid leaves fewer than the required 60%, at either count.
    for n in (5, 25):
        bad = math.ceil(n / 2)
        closes = np.column_stack(
            [np.full(n, 100.0), np.concatenate([np.full(bad, -1.0), np.full(n - bad, 101.0)])]
        )
        with pytest.raises(InvalidForecast, match="numerically valid"):
            score_paths(SymbolPaths("TEST", 100.0, closes, closes))


def test_a_single_bad_path_does_not_reject_the_symbol():
    """One artifact in 25 paths is dropped; the symbol still gets scored."""
    closes = np.column_stack([np.full(25, 100.0), np.concatenate([[-1.0], np.full(24, 101.0)])])
    result = score_paths(SymbolPaths("TEST", 100.0, closes, closes))
    assert result.paths_used == 24
    assert result.mu == pytest.approx(0.01)


def test_non_finite_paths_are_treated_as_invalid():
    returns = [0.01] * 12
    good = paths_from_terminal(returns)
    closes = np.vstack([good.closes, np.array([100.0, np.nan])])
    lows = np.vstack([good.lows, np.array([100.0, np.inf])])

    result = score_paths(SymbolPaths("TEST", 100.0, closes, lows))
    assert result.paths_used == 12
    assert np.isfinite(result.mu)


def test_mean_move_far_beyond_realized_volatility_is_rejected():
    # A -60% mean against 5% realized volatility is 12x -- an artifact.
    paths = paths_from_terminal([-0.60] * 12, vol=0.05)
    with pytest.raises(InvalidForecast, match="implausible mean move"):
        score_paths(paths)


def test_a_move_within_the_volatility_budget_is_accepted():
    # 2x realized volatility is large but inside the 3x limit.
    result = score_paths(paths_from_terminal([0.09, 0.10, 0.11] * 4, vol=0.05))
    assert result.mu == pytest.approx(0.10)


def test_the_plausibility_guard_is_disabled_without_a_volatility_estimate():
    result = score_paths(paths_from_terminal([-0.60] * 12, vol=0.0))
    assert result.mu == pytest.approx(-0.60)


def test_a_barely_moving_instrument_is_not_rejected_for_a_small_move():
    """Regression: '-0.4% against realized 0.0% volatility' was a real rejection.

    Below MIN_VOL_FOR_GUARD the multiple-based bound is meaningless -- 3x of
    almost nothing rejects any forecast at all.
    """
    result = score_paths(paths_from_terminal([-0.004] * 12, vol=0.0005))
    assert result.mu == pytest.approx(-0.004)


def test_an_absolute_floor_permits_a_small_move_at_low_volatility():
    # 0.8% forecast against 0.3% vol is 2.7x, under the 1% absolute floor.
    result = score_paths(paths_from_terminal([0.008] * 12, vol=0.003))
    assert result.mu == pytest.approx(0.008)

    # 2% against the same volatility clears the floor and is rejected. This was
    # a real escape: at a 2% floor a bond ETF passed at a 6.5x ratio.
    with pytest.raises(InvalidForecast, match="implausible mean move"):
        score_paths(paths_from_terminal([0.02] * 12, vol=0.003))


def test_mu_vol_ratio_is_reported_for_filtering():
    result = score_paths(paths_from_terminal([0.05] * 12, vol=0.05))
    assert result.mu_vol_ratio == pytest.approx(1.0)

    # Zero volatility yields 0 rather than a division by zero.
    assert score_paths(paths_from_terminal([0.05] * 12, vol=0.0)).mu_vol_ratio == 0.0


def test_paths_used_reports_the_valid_count():
    assert score_paths(paths_from_terminal([0.01] * 25)).paths_used == 25


@pytest.mark.parametrize(
    ("mu", "p_up", "expected"),
    [
        (0.02, 0.80, "BUY"),
        (0.02, 0.50, "HOLD"),  # strong mean, but a coin flip
        (0.001, 0.90, "HOLD"),  # confident, but the edge is below the floor
        (-0.01, 0.30, "AVOID"),
        (0.0, 0.90, "AVOID"),  # no expected edge at all
    ],
)
def test_signal_thresholds(mu, p_up, expected):
    assert classify(mu, p_up) == expected


def test_rank_is_descending_and_breaks_ties_stably():
    def stub(symbol, score):
        return Score(symbol, 10.0, score, "HOLD", 0.0, 0.5, 0.01, 0.0, 0.0, 0.0)

    ordered = rank([stub("BBB", 1.0), stub("AAA", 2.0), stub("AAA2", 1.0)])
    assert [s.symbol for s in ordered] == ["AAA", "AAA2", "BBB"]
