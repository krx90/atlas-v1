"""Exit signals for `atlas review`.

The point of the module is that thresholds scale with each symbol's own
volatility, so most of these tests are about the sigma arithmetic being right
and each signal firing at -- and not before -- its boundary.
"""

from __future__ import annotations

import math

import pytest

from atlas import config
from atlas.review import (
    Holding,
    adverse_excursion,
    costly_to_exit,
    expected_move,
    forecast_against,
    liquidity_exit,
    pnl_sigma,
    profit_target,
    rank,
    review,
)


def holding(**kw) -> Holding:
    base = dict(
        symbol="AAA",
        side="long",
        pnl_pct=0.0,
        market_value=1000.0,
        days_held=1,
        daily_vol=0.02,
        dollar_volume=50_000_000.0,
    )
    base.update(kw)
    return Holding(**base)


# --- sigma arithmetic ---------------------------------------------------------


def test_expected_move_is_the_daily_volatility_for_one_day():
    assert expected_move(holding(daily_vol=0.02, days_held=1)) == pytest.approx(0.02)


def test_expected_move_grows_with_the_square_root_of_time():
    """Ten days should be ~3.16x one day, not 10x."""
    one = expected_move(holding(days_held=1))
    ten = expected_move(holding(days_held=10))
    assert ten / one == pytest.approx(math.sqrt(10))


def test_a_long_held_position_is_not_flagged_merely_for_drifting():
    """-6% over 9 days is under 2 sigma; over 1 day it is well past."""
    long_held = holding(pnl_pct=-0.06, days_held=9, daily_vol=0.02)
    fresh = holding(pnl_pct=-0.06, days_held=1, daily_vol=0.02)

    assert pnl_sigma(long_held) == pytest.approx(-1.0)
    assert adverse_excursion(long_held) is None
    assert adverse_excursion(fresh) is not None


def test_the_same_percentage_means_different_things_at_different_volatility():
    """The reason fixed thresholds were rejected, pinned as a test."""
    quiet = holding(pnl_pct=-0.08, daily_vol=0.009)   # XLC-like
    noisy = holding(pnl_pct=-0.08, daily_vol=0.036)   # TLN-like

    assert pnl_sigma(quiet) == pytest.approx(-8.89, abs=0.01)
    assert pnl_sigma(noisy) == pytest.approx(-2.22, abs=0.01)
    assert adverse_excursion(quiet).severity > adverse_excursion(noisy).severity


def test_an_unknown_holding_period_is_treated_as_one_day():
    assert expected_move(holding(days_held=None)) == pytest.approx(0.02)


def test_a_barely_moving_symbol_does_not_divide_by_zero():
    """One real symbol had realized volatility rounding to 0.00%."""
    result = pnl_sigma(holding(pnl_pct=-0.01, daily_vol=0.0))
    assert math.isfinite(result)
    assert expected_move(holding(daily_vol=0.0)) == pytest.approx(config.REVIEW_MIN_VOL)


# --- individual signals -------------------------------------------------------


def test_the_stop_fires_only_past_its_boundary():
    just_inside = holding(pnl_pct=-0.02 * config.REVIEW_STOP_SIGMA * 0.99)
    just_outside = holding(pnl_pct=-0.02 * config.REVIEW_STOP_SIGMA * 1.5)

    assert adverse_excursion(just_inside) is None
    finding = adverse_excursion(just_outside)
    assert finding.kind == "stop"
    assert finding.severity == pytest.approx(1.0, abs=0.01)


def test_the_target_fires_only_past_its_boundary():
    assert profit_target(holding(pnl_pct=0.02 * 2.9)) is None
    finding = profit_target(holding(pnl_pct=0.02 * 4.0))
    assert finding.kind == "target"
    assert finding.severity == pytest.approx(1.0, abs=0.01)


def test_a_gain_never_trips_the_stop_and_a_loss_never_the_target():
    winner = holding(pnl_pct=0.20)
    loser = holding(pnl_pct=-0.20)
    assert adverse_excursion(winner) is None
    assert profit_target(loser) is None


def test_a_short_profits_when_alpaca_reports_positive_pnl():
    """Alpaca already inverts a short's P&L sign, so no extra handling."""
    assert profit_target(holding(side="short", pnl_pct=0.02 * 4.0)) is not None
    assert adverse_excursion(holding(side="short", pnl_pct=0.02 * 4.0)) is None


def test_leaving_the_universe_is_flagged():
    finding = liquidity_exit(holding(in_universe=False))
    assert finding.kind == "liquidity"
    assert finding.severity == config.REVIEW_UNIVERSE_SEVERITY


def test_an_unborrowable_short_outranks_an_ordinary_universe_exit():
    """It is the one forced exit -- the borrow can be recalled."""
    borrow = liquidity_exit(holding(side="short", easy_to_borrow=False))
    universe = liquidity_exit(holding(in_universe=False))
    assert borrow.kind == "borrow"
    assert borrow.severity > universe.severity


def test_borrowability_is_irrelevant_to_a_long():
    assert liquidity_exit(holding(side="long", easy_to_borrow=False)) is None


def test_a_healthy_holding_trips_nothing():
    assert liquidity_exit(holding()) is None
    assert costly_to_exit(holding()) is None
    assert review(holding()).findings == []


def test_a_large_position_relative_to_daily_volume_is_flagged():
    small = holding(market_value=100_000.0, dollar_volume=50_000_000.0)   # 0.2%
    large = holding(market_value=5_000_000.0, dollar_volume=50_000_000.0)  # 10%

    assert costly_to_exit(small) is None
    finding = costly_to_exit(large)
    assert finding.kind == "illiquid"
    assert "10.0%" in finding.detail


def test_a_short_position_uses_its_absolute_size():
    finding = costly_to_exit(holding(side="short", market_value=-5_000_000.0))
    assert finding is not None


def test_unknown_volume_is_not_a_flag():
    assert costly_to_exit(holding(dollar_volume=None)) is None
    assert costly_to_exit(holding(dollar_volume=0.0)) is None


# --- the model signal ---------------------------------------------------------


def test_a_negative_forecast_opposes_a_long():
    finding = forecast_against(holding(forecast_mu=-0.03, forecast_vol_ratio=1.5))
    assert finding.kind == "forecast*", "must stay marked as the unvalidated signal"
    assert finding.severity == pytest.approx(1.5)


def test_a_negative_forecast_favours_a_short():
    assert forecast_against(
        holding(side="short", forecast_mu=-0.03, forecast_vol_ratio=1.5)
    ) is None


def test_a_positive_forecast_opposes_a_short():
    assert forecast_against(
        holding(side="short", forecast_mu=0.03, forecast_vol_ratio=1.5)
    ) is not None


def test_the_forecast_signal_is_capped_so_it_cannot_dominate():
    finding = forecast_against(holding(forecast_mu=-0.5, forecast_vol_ratio=99.0))
    assert finding.severity == config.REVIEW_FORECAST_CAP


def test_a_weak_forecast_is_ignored():
    assert forecast_against(holding(forecast_mu=-0.001, forecast_vol_ratio=0.1)) is None


def test_no_forecast_is_not_a_flag():
    assert forecast_against(holding(forecast_mu=None)) is None
    assert forecast_against(holding(forecast_mu=-0.03, forecast_vol_ratio=None)) is None


# --- scoring and ordering -----------------------------------------------------


def test_the_score_sums_every_triggered_signal():
    both = holding(pnl_pct=-0.02 * 3.5, in_universe=False)
    result = review(both)

    assert {f.kind for f in result.findings} == {"stop", "liquidity"}
    assert result.score == pytest.approx(1.5 + config.REVIEW_UNIVERSE_SEVERITY, abs=0.01)


def test_failing_several_checks_outranks_failing_one_badly():
    many = holding(symbol="MANY", pnl_pct=-0.02 * 3.0, in_universe=False)
    one = holding(symbol="ONE", pnl_pct=-0.02 * 3.4)

    order = rank([one, many])
    assert [r.holding.symbol for r in order] == ["MANY", "ONE"]


def test_ranking_is_worst_first_and_ties_break_on_symbol():
    order = rank([
        holding(symbol="CLEAN"),
        holding(symbol="BAD", pnl_pct=-0.02 * 5.0),
        holding(symbol="ALSO"),
    ])
    assert [r.holding.symbol for r in order] == ["BAD", "ALSO", "CLEAN"]
    assert order[-1].score == 0.0


def test_an_empty_portfolio_ranks_to_nothing():
    assert rank([]) == []
