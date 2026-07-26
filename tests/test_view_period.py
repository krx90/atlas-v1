"""Period parsing for `atlas view`."""

from __future__ import annotations

import pytest

from atlas.commands.view import BadPeriod, parse_period


@pytest.mark.parametrize(
    ("period", "days"),
    [("1m", 30), ("3m", 91), ("6m", 182), ("1y", 365), ("2y", 730), ("3y", 1095), ("5y", 1826)],
)
def test_named_periods(period, days):
    assert parse_period(period) == days


def test_default_is_six_months():
    assert parse_period(None) == 182


def test_case_and_whitespace_are_tolerated():
    assert parse_period("  1Y  ") == 365


def test_plain_day_counts_pass_through():
    assert parse_period("45") == 45


@pytest.mark.parametrize("bad", ["0", "-5", "6months", "m6", "", "1w", "1.5y"])
def test_rejected_inputs(bad):
    with pytest.raises(BadPeriod):
        parse_period(bad)
