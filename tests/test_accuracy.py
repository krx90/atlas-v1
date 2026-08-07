"""Forecast accuracy: direction, magnitude, the traded label, and the baselines.

`test_backtest` covers whether the ranking machinery is honest. This covers
whether the forecast is *right*, which is a different question -- a model can
rank ten symbols perfectly without any of them going the way it said.

The tests that matter most here are the ones pinning the baselines, because a
baseline that flatters the model is worse than no baseline at all. So the zero
forecast is asserted to score exactly 0.0 skill, and the permutation null is
asserted to sit at zero and to be *wide enough* that a lucky sample does not
clear it.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from atlas import accuracy, config
from atlas.backtest import Observation


def observation(date, symbol, *, score=0.0, mu=0.0, forward=0.0, signal="", prior=0.0):
    return Observation(
        date=date,
        symbol=symbol,
        score=score,
        mu=mu,
        side="long",
        forward_return=forward,
        signal=signal,
        prior_return=prior,
    )


def spread_over_dates(n_dates, n_symbols, make, start=1):
    """`make(date_index, symbol_index)` -> Observation, over a grid of dates."""
    rows = []
    for d in range(n_dates):
        stamp = f"2024-01-{d + start:02d}"
        rows += [make(stamp, i) for i in range(n_symbols)]
    return rows


# --- direction ----------------------------------------------------------------


def test_a_forecast_that_always_points_the_right_way_hits_every_time():
    rows = [
        observation("2024-01-01", f"S{i}", mu=0.01 if i % 2 else -0.01,
                    forward=0.05 if i % 2 else -0.05)
        for i in range(40)
    ]
    result = accuracy.directional(rows)

    assert result.hit_rate == pytest.approx(1.0)
    assert result.n == 40
    assert result.p_value < 1e-9
    assert result.verdict == "directionally informative"


def test_a_systematically_inverted_sign_is_called_out_not_praised():
    rows = [
        observation("2024-01-01", f"S{i}", mu=0.01, forward=-0.05) for i in range(40)
    ]
    result = accuracy.directional(rows)

    assert result.hit_rate == pytest.approx(0.0)
    assert result.p_value < 1e-9
    assert "inverted" in result.verdict


def test_a_coin_flip_forecast_is_not_mistaken_for_skill():
    rng = np.random.default_rng(0)
    rows = [
        observation("2024-01-01", f"S{i}", mu=rng.normal(), forward=rng.normal())
        for i in range(400)
    ]
    result = accuracy.directional(rows)

    assert result.hit_rate == pytest.approx(0.5, abs=0.08)
    assert result.p_value > 0.05
    assert result.verdict == "no better than a coin flip"


def test_flat_outcomes_are_excluded_rather_than_scored_as_misses():
    """A zero forward return is neither a hit nor a miss.

    Counting it would pull the rate toward 50% for a reason unrelated to skill.
    """
    rows = [observation("2024-01-01", f"S{i}", mu=0.01, forward=0.05) for i in range(10)]
    rows += [observation("2024-01-01", f"F{i}", mu=0.01, forward=0.0) for i in range(90)]
    result = accuracy.directional(rows)

    assert result.n == 10
    assert result.hit_rate == pytest.approx(1.0)


def test_a_short_forecast_scores_on_the_same_footing_as_a_long():
    """mu is direction-neutral, so a correctly negative mu is a hit."""
    longs = [observation("2024-01-01", f"L{i}", mu=0.02, forward=0.03) for i in range(20)]
    shorts = [observation("2024-01-01", f"S{i}", mu=-0.02, forward=-0.03) for i in range(20)]

    assert accuracy.directional(longs + shorts).hit_rate == pytest.approx(1.0)


def test_the_binomial_p_value_matches_a_hand_checkable_case():
    # Ten heads out of ten: 2 * 0.5**10.
    assert accuracy._two_sided_binomial(10, 10) == pytest.approx(2 * 0.5**10)
    # Dead even is the least surprising outcome there is.
    assert accuracy._two_sided_binomial(50, 100) == pytest.approx(1.0)


def test_the_p_value_survives_a_sample_large_enough_to_overflow_exact_integers():
    """Log-space, so a realistic backtest size does not build 400-digit ints."""
    p = accuracy._two_sided_binomial(5200, 10000)
    assert 0.0 < p < 1.0


# --- magnitude ----------------------------------------------------------------


def test_forecasting_zero_scores_exactly_zero_skill():
    """The definition of the baseline, pinned so it cannot drift."""
    rng = np.random.default_rng(1)
    rows = [
        observation("2024-01-01", f"S{i}", mu=0.0, forward=float(rng.normal(0, 0.03)))
        for i in range(200)
    ]
    result = accuracy.magnitude(rows)

    assert result.skill == pytest.approx(0.0)
    assert result.rmse == pytest.approx(result.baseline_rmse)
    assert "worse than forecasting no move" in result.verdict


def test_a_perfect_point_forecast_scores_skill_of_one():
    rng = np.random.default_rng(2)
    rows = []
    for i in range(200):
        actual = float(rng.normal(0, 0.03))
        rows.append(observation("2024-01-01", f"S{i}", mu=actual, forward=actual))
    result = accuracy.magnitude(rows)

    assert result.skill == pytest.approx(1.0)
    assert result.rmse == pytest.approx(0.0, abs=1e-12)


def test_an_overconfident_forecast_scores_worse_than_predicting_nothing():
    """The failure the plausibility guard exists to catch, measured end to end."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(200):
        actual = float(rng.normal(0, 0.03))
        # Right direction, five times too large.
        rows.append(observation("2024-01-01", f"S{i}", mu=actual * 5, forward=actual))
    result = accuracy.magnitude(rows)

    assert result.skill < 0
    assert result.mean_abs_mu > result.mean_abs_return
    assert "worse than forecasting no move" in result.verdict


def test_mae_and_rmse_are_the_error_against_the_realized_return():
    rows = [
        observation("2024-01-01", "A", mu=0.01, forward=0.03),  # error 0.02
        observation("2024-01-01", "B", mu=0.05, forward=0.01),  # error -0.04
    ]
    result = accuracy.magnitude(rows)

    assert result.mae == pytest.approx(0.03)
    assert result.rmse == pytest.approx(np.sqrt((0.02**2 + 0.04**2) / 2))


# --- the traded label ---------------------------------------------------------


def test_buckets_report_what_happened_to_each_signal():
    rows = [observation("2024-01-01", f"B{i}", signal="BUY", forward=0.04) for i in range(10)]
    rows += [observation("2024-01-01", f"H{i}", signal="HOLD", forward=0.00) for i in range(5)]
    rows += [observation("2024-01-01", f"A{i}", signal="AVOID", forward=-0.02) for i in range(8)]
    buckets = {b.signal: b for b in accuracy.by_signal(rows)}

    assert buckets["BUY"].n == 10
    assert buckets["BUY"].mean_return == pytest.approx(0.04)
    assert buckets["AVOID"].mean_return == pytest.approx(-0.02)


def test_a_signal_hits_when_the_symbol_moves_the_way_it_claimed():
    """AVOID is a claim that the symbol falls, so a fall is a hit, not a miss."""
    rows = [observation("2024-01-01", f"A{i}", signal="AVOID", forward=-0.02) for i in range(8)]
    rows += [observation("2024-01-01", f"S{i}", signal="SHORT", forward=-0.03) for i in range(8)]
    rows += [observation("2024-01-01", f"B{i}", signal="BUY", forward=-0.03) for i in range(8)]
    buckets = {b.signal: b for b in accuracy.by_signal(rows)}

    assert buckets["AVOID"].hit_rate == pytest.approx(1.0)
    assert buckets["SHORT"].hit_rate == pytest.approx(1.0)
    assert buckets["BUY"].hit_rate == pytest.approx(0.0)  # claimed a rise, fell


def test_buckets_come_back_entry_signals_first():
    rows = [observation("2024-01-01", "A", signal="AVOID")]
    rows += [observation("2024-01-01", "B", signal="BUY")]
    rows += [observation("2024-01-01", "H", signal="HOLD")]

    assert [b.signal for b in accuracy.by_signal(rows)] == ["BUY", "HOLD", "AVOID"]


def test_observations_without_a_signal_produce_no_buckets():
    """A CSV written before the column existed must not fabricate one."""
    rows = [observation("2024-01-01", f"S{i}", forward=0.01) for i in range(10)]

    assert accuracy.by_signal(rows) == []


# --- the permutation null -----------------------------------------------------


def test_a_perfect_ranking_beats_every_shuffle():
    rows = spread_over_dates(
        10, 15, lambda d, i: observation(d, f"S{i}", score=float(i), forward=float(i) / 100)
    )
    result = accuracy.permutation_ic(rows, permutations=200)

    assert result.mean_ic == pytest.approx(1.0)
    assert result.p_value == pytest.approx(0.0)
    assert result.verdict == "beats its own shuffle"


def test_the_null_sits_at_zero_and_is_wide_enough_to_be_honest():
    """A null centred anywhere but zero, or too narrow, would flatter the model."""
    rng = np.random.default_rng(4)
    rows = spread_over_dates(
        12,
        20,
        lambda d, i: observation(d, f"S{i}", score=float(rng.normal()),
                                 forward=float(rng.normal()) / 100),
    )
    result = accuracy.permutation_ic(rows, permutations=300)

    assert result.null_mean == pytest.approx(0.0, abs=0.02)
    assert result.null_sd > 0.02  # a real spread, not a degenerate one
    assert result.p_value > 0.05
    assert result.verdict == "indistinguishable from shuffled scores"


def test_a_weak_but_consistent_edge_clears_the_shuffle():
    rng = np.random.default_rng(5)

    def make(d, i):
        signal = float(i)
        return observation(
            d, f"S{i}",
            score=signal + rng.normal(0, 3.0),
            forward=signal / 100 + rng.normal(0, 0.02),
        )

    result = accuracy.permutation_ic(spread_over_dates(15, 20, make), permutations=300)

    assert 0.1 < result.mean_ic < 0.9  # weak, not a giveaway
    assert result.p_value < 0.05


def test_a_single_date_cannot_be_permuted_into_a_conclusion():
    rows = [observation("2024-01-01", f"S{i}", score=float(i), forward=float(i)) for i in range(10)]
    result = accuracy.permutation_ic(rows, permutations=50)

    assert result.dates == 1
    assert np.isnan(result.p_value)
    assert "too few dates" in result.verdict


def test_the_permutation_is_reproducible_under_a_seed():
    rng = np.random.default_rng(6)
    rows = spread_over_dates(
        8, 12,
        lambda d, i: observation(d, f"S{i}", score=float(rng.normal()),
                                 forward=float(rng.normal())),
    )
    a = accuracy.permutation_ic(rows, permutations=100, seed=7)
    b = accuracy.permutation_ic(rows, permutations=100, seed=7)

    assert a.p_value == b.p_value
    assert a.null_sd == pytest.approx(b.null_sd)


# --- the momentum baseline ----------------------------------------------------


def test_momentum_ic_recovers_a_planted_trailing_return_edge():
    rows = spread_over_dates(
        10, 15,
        lambda d, i: observation(d, f"S{i}", prior=float(i) / 100, forward=float(i) / 100),
    )

    assert accuracy.momentum_ic(rows) == pytest.approx(1.0)


def test_momentum_is_nan_when_it_was_never_recorded():
    """Not 0.0 -- an unmeasured baseline must not read as a measured null."""
    rows = spread_over_dates(
        5, 10, lambda d, i: observation(d, f"S{i}", score=float(i), forward=float(i) / 100)
    )

    assert np.isnan(accuracy.momentum_ic(rows))


# --- the combined reading -----------------------------------------------------


def make_null_model(n_dates=20, n_symbols=30, seed=9):
    """Observations with no relationship between forecast and outcome.

    The seed is fixed deliberately. A 5%-level test fires on roughly one null
    sample in twenty by construction -- seed 8 draws a permutation p of 0.045 --
    so leaving this to chance would make the suite flaky for a reason that is
    the threshold working correctly rather than the code failing.
    """
    rng = np.random.default_rng(seed)
    return spread_over_dates(
        n_dates,
        n_symbols,
        lambda d, i: observation(
            d, f"S{i}",
            score=float(rng.normal()),
            mu=float(rng.normal(0, 0.03)),
            forward=float(rng.normal(0, 0.03)),
            signal="BUY" if i % 2 else "AVOID",
            prior=float(rng.normal(0, 0.03)),
        ),
    )


def make_accurate_model(n_dates=20, n_symbols=30, seed=9, prior_multiple=None):
    """Observations where the forecast is right on every measure."""
    rng = np.random.default_rng(seed)

    def make(d, i):
        actual = float(rng.normal(0, 0.03))
        return observation(
            d, f"S{i}",
            score=actual,
            mu=actual * 0.9,  # right direction, slightly conservative
            forward=actual,
            signal="BUY" if actual > 0 else "AVOID",
            prior=actual * prior_multiple if prior_multiple else 0.0,
        )

    return spread_over_dates(n_dates, n_symbols, make)


def test_a_model_with_no_skill_fails_every_substantive_condition():
    result = accuracy.evaluate(make_null_model(), permutations=200)

    assert not result.passes["direction"]
    assert not result.passes["magnitude"]
    assert not result.passes["ranking"]
    assert "clears the pre-registered bar" not in result.verdict


def test_a_model_accurate_on_every_measure_clears_the_bar():
    result = accuracy.evaluate(make_accurate_model(), permutations=200)

    assert result.direction.hit_rate == pytest.approx(1.0)
    assert result.size.skill > 0
    assert result.null.p_value < 0.05
    assert all(result.passes.values())
    assert result.verdict.startswith("clears the pre-registered bar")


def test_a_model_beaten_by_trailing_return_fails_even_when_it_is_accurate():
    """Beating zero is not the bar. Beating a free baseline is."""
    # Momentum ranks exactly as well as the model -- and costs a subtraction.
    result = accuracy.evaluate(make_accurate_model(prior_multiple=2), permutations=200)

    assert result.momentum >= result.null.mean_ic
    assert not result.passes["vs momentum"]
    assert "vs momentum" in result.verdict


def test_an_unmeasured_momentum_baseline_does_not_block_a_pass():
    """NaN means never measured, which is not the same as being beaten."""
    result = accuracy.evaluate(make_accurate_model(), permutations=200)

    assert np.isnan(result.momentum)
    assert result.passes["vs momentum"]


def test_a_small_sample_refuses_to_reach_a_verdict():
    result = accuracy.evaluate(make_null_model(n_dates=2, n_symbols=10), permutations=50)

    assert result.n < config.ACCURACY_MIN_OBSERVATIONS
    assert result.verdict.startswith("inconclusive")


def test_a_partial_result_is_named_by_what_it_failed_on():
    """Direction right, magnitude useless -- the realistic in-between case."""
    rng = np.random.default_rng(11)

    def make(d, i):
        actual = float(rng.normal(0, 0.03))
        return observation(
            d, f"S{i}",
            score=float(rng.normal()),  # ranking carries nothing
            mu=np.sign(actual) * 0.5,  # sign right, magnitude absurd
            forward=actual,
        )

    result = accuracy.evaluate(spread_over_dates(20, 30, make), permutations=200)

    assert result.passes["direction"]
    assert not result.passes["magnitude"]
    assert not result.passes["ranking"]
    assert result.verdict.startswith("does not clear the bar")
    assert "magnitude" in result.verdict and "ranking" in result.verdict


# --- the bar is the config, not a literal -------------------------------------


def test_the_direction_threshold_comes_from_config_not_a_hardcoded_p():
    """The whole point of pre-registering: loosening the bar is a visible edit.

    A p of 0.03 clears the conventional 0.05 and fails the stricter 0.01 this
    project committed to, so it is exactly the case that distinguishes them.
    """
    rows = [observation("2024-01-01", f"S{i}", mu=0.01, forward=0.01) for i in range(60)]
    rows += [observation("2024-01-01", f"M{i}", mu=0.01, forward=-0.01) for i in range(40)]
    result = accuracy.evaluate(rows, permutations=50)

    assert result.direction.p_value == pytest.approx(0.0569, abs=0.001)
    assert not result.passes["direction"]  # fails at 0.01

    with mock.patch.object(config, "ACCURACY_MAX_DIRECTION_P", 0.10):
        assert result.passes["direction"]  # would have passed at 0.10


def test_the_momentum_condition_can_be_switched_off_in_config():
    result = accuracy.evaluate(make_accurate_model(prior_multiple=2), permutations=200)

    assert "vs momentum" in result.passes
    with mock.patch.object(config, "ACCURACY_MUST_BEAT_MOMENTUM", False):
        assert "vs momentum" not in result.passes
        assert result.verdict.startswith("clears the pre-registered bar")


def test_an_empty_run_does_not_divide_by_zero():
    result = accuracy.evaluate([], permutations=10)

    assert result.n == 0
    assert np.isnan(result.size.skill)
    assert result.buckets == []
