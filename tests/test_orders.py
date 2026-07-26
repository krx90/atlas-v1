"""Share/notional arithmetic for order placement."""

from __future__ import annotations

import pytest

from atlas.commands.buy import shares_for


def test_fractional_order_spends_the_full_amount():
    shares, dollars = shares_for(500.0, 45.23, fractionable=True)
    assert dollars == pytest.approx(500.0)
    assert shares == pytest.approx(500.0 / 45.23)
    # The example from docs/commands.md.
    assert round(shares, 2) == 11.05


def test_whole_share_order_rounds_down_never_up():
    shares, dollars = shares_for(1000.0, 21.01, fractionable=False)
    assert shares == 47
    assert dollars == pytest.approx(987.47)
    assert dollars <= 1000.0


def test_rounding_down_holds_just_below_a_share_boundary():
    # 2 shares would cost 100.00 exactly, but we only have 99.99.
    shares, dollars = shares_for(99.99, 50.0, fractionable=False)
    assert shares == 1
    assert dollars == pytest.approx(50.0)


def test_exact_multiple_is_not_rounded_off():
    shares, dollars = shares_for(100.0, 50.0, fractionable=False)
    assert shares == 2
    assert dollars == pytest.approx(100.0)


def test_unaffordable_whole_share_yields_zero_for_the_caller_to_reject():
    shares, dollars = shares_for(100.0, 250.0, fractionable=False)
    assert shares == 0
    assert dollars == 0


def test_non_positive_price_is_rejected():
    with pytest.raises(ValueError, match="invalid price"):
        shares_for(100.0, 0.0, fractionable=True)
