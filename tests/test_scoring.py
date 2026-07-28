"""Scoring statistics against paths with known distributions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atlas.forecast import SymbolPaths
from atlas.scoring import InvalidForecast, Score, classify, rank, score_paths


def _bracket(values, shape):
    """Broadcast a per-path price into a (paths, horizon) array."""
    return np.asarray(values, dtype="float64").reshape(-1, 1) * np.ones((1, shape))


def paths_from_terminal(
    terminal_returns, *, p0=100.0, lows=None, highs=None, vol=0.0
) -> SymbolPaths:
    """Build a SymbolPaths whose terminal closes give exactly these returns.

    `vol` defaults to 0, which disables the plausibility guard, so tests of the
    statistics themselves are not entangled with it. `lows`/`highs` default to
    the closes, which keeps drawdown and run-up out of the way unless a test is
    specifically about them.
    """
    terminal = np.asarray(terminal_returns, dtype="float64")
    closes = np.column_stack([np.full_like(terminal, p0), p0 * (1.0 + terminal)])
    low_array = _bracket(lows, 2) if lows is not None else closes
    high_array = _bracket(highs, 2) if highs is not None else closes
    return SymbolPaths("TEST", p0, closes, low_array, high_array, realized_vol=vol)


def symbol_paths(closes, lows=None, highs=None, p0=100.0, vol=0.0) -> SymbolPaths:
    """Construct directly from a (paths, horizon) close array."""
    closes = np.asarray(closes, dtype="float64")
    return SymbolPaths(
        "TEST",
        p0,
        closes,
        closes if lows is None else np.asarray(lows, dtype="float64"),
        closes if highs is None else np.asarray(highs, dtype="float64"),
        realized_vol=vol,
    )


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
    bad = symbol_paths(np.ones((2, 2)), p0=0.0)
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
    contaminated = symbol_paths(closes, lows=lows, highs=closes)

    result = score_paths(contaminated)
    assert result.paths_used == len(returns)
    assert result.mu == pytest.approx(clean.mu)


def test_symbol_is_rejected_when_too_few_paths_survive():
    closes = np.column_stack([np.full(12, 100.0), np.full(12, -5.0)])
    with pytest.raises(InvalidForecast, match="numerically valid"):
        score_paths(symbol_paths(closes))


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
            score_paths(symbol_paths(closes))


def test_a_single_bad_path_does_not_reject_the_symbol():
    """One artifact in 25 paths is dropped; the symbol still gets scored."""
    closes = np.column_stack([np.full(25, 100.0), np.concatenate([[-1.0], np.full(24, 101.0)])])
    result = score_paths(symbol_paths(closes))
    assert result.paths_used == 24
    assert result.mu == pytest.approx(0.01)


def test_non_finite_paths_are_treated_as_invalid():
    returns = [0.01] * 12
    good = paths_from_terminal(returns)
    closes = np.vstack([good.closes, np.array([100.0, np.nan])])
    lows = np.vstack([good.lows, np.array([100.0, np.inf])])

    result = score_paths(symbol_paths(closes, lows=lows, highs=closes))
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


# --- short side ---------------------------------------------------------------
#
# The two sides read the same sampled paths. A short's risk is the mirror of a
# long's: the 95th-percentile outcome and the worst intra-path high, rather than
# the 5th percentile and the worst low.


def test_a_falling_symbol_scores_well_short_and_badly_long():
    falling = paths_from_terminal([-0.04, -0.05, -0.06] * 4)
    assert score_paths(falling, side="short").score > 0
    assert score_paths(falling, side="long").score < 0


def test_the_two_sides_are_mirror_images_absent_tail_penalties():
    """With symmetric tails and no penalty, short score = -long score."""
    paths = paths_from_terminal([-0.02, -0.01, 0.01, 0.02] * 3)
    long_s = score_paths(paths, lam=0.0)
    short_s = score_paths(paths, side="short", lam=0.0)
    assert short_s.score == pytest.approx(-long_s.score, rel=1e-9)


def test_short_is_penalised_by_the_upside_tail_not_the_downside():
    # Mostly falls, but one path rips upward -- that is the short's disaster.
    paths = paths_from_terminal([-0.05] * 11 + [0.30])
    s = score_paths(paths, side="short")
    assert s.q95 > 0
    expected = (-s.mu - 0.5 * s.q95) / s.sigma
    assert s.score == pytest.approx(expected, rel=1e-6)


def test_run_up_uses_the_worst_high_per_path():
    result = score_paths(
        paths_from_terminal([-0.05, -0.05], p0=100.0, highs=[110.0, 120.0]), side="short"
    )
    assert result.runup == pytest.approx(0.15)  # mean of +10%, +20%


def test_both_sides_statistics_are_always_populated():
    """A Score describes the whole distribution regardless of the side asked for."""
    for side in ("long", "short"):
        s = score_paths(paths_from_terminal([-0.05, -0.02, 0.03] * 4), side=side)
        assert s.side == side
        assert s.p_up > 0 and s.p_down > 0
        assert s.q05 < s.q95
        assert s.mdd <= 0 <= s.runup


def test_p_down_is_not_one_minus_p_up_when_a_path_is_flat():
    s = score_paths(paths_from_terminal([-0.01, 0.0, 0.01]))
    assert s.p_up == pytest.approx(1 / 3)
    assert s.p_down == pytest.approx(1 / 3)


def test_mu_is_direction_neutral_across_sides():
    """mu is the raw expected return, not the trade's edge -- same either way."""
    paths = paths_from_terminal([-0.05, -0.03, -0.01] * 4)
    assert score_paths(paths, side="short").mu == pytest.approx(
        score_paths(paths, side="long").mu
    )


@pytest.mark.parametrize(
    ("mu", "p_fav", "side", "expected"),
    [
        (-0.02, 0.80, "short", "SHORT"),  # falls hard, consistently
        (-0.02, 0.50, "short", "HOLD"),   # falls, but a coin flip
        (-0.001, 0.90, "short", "HOLD"),  # consistent, edge under the floor
        (0.01, 0.10, "short", "AVOID"),   # rising -- wrong way for a short
        (0.02, 0.80, "long", "BUY"),
        (-0.01, 0.30, "long", "AVOID"),
    ],
)
def test_signal_thresholds_mirror_across_sides(mu, p_fav, side, expected):
    assert classify(mu, p_fav, side) == expected


def test_the_plausibility_guard_applies_to_shorts_too():
    with pytest.raises(InvalidForecast, match="implausible mean move"):
        score_paths(paths_from_terminal([-0.60] * 12, vol=0.05), side="short")


def test_an_unknown_side_is_rejected():
    with pytest.raises(ValueError, match="side must be"):
        score_paths(paths_from_terminal([0.01] * 12), side="sideways")
