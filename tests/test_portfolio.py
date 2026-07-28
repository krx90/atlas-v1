"""Reconstructing position entry dates from fill history.

Alpaca's Position model carries no timestamp, so the entry date is replayed from
filled orders. The subtlety is what "entry" means when a position was scaled into,
or closed and later reopened -- the answer has to be the fill that took it off
zero and which it never returned to zero after.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from atlas.commands.portfolio import _held_for, entry_dates

DAY = timedelta(days=1)
BASE = datetime(2026, 7, 1, 14, 30, tzinfo=timezone.utc)


def order(symbol, side, qty, day, status="filled"):
    return SimpleNamespace(
        symbol=symbol,
        side=SimpleNamespace(value=side),
        filled_qty=str(qty),
        filled_at=BASE + day * DAY,
        status=SimpleNamespace(value=status),
    )


def with_orders(orders):
    """Patch the orders call to return `orders`."""
    return patch("atlas.alpaca_client.call", lambda _fn, *_a, **_k: orders)


def test_a_single_buy_is_the_entry():
    with with_orders([order("AAA", "buy", 10, 0)]):
        assert entry_dates(["AAA"])["AAA"] == BASE


def test_scaling_in_keeps_the_original_entry():
    """Adding to a winner must not reset the clock."""
    orders = [order("AAA", "buy", 10, 0), order("AAA", "buy", 5, 3), order("AAA", "buy", 5, 9)]
    with with_orders(orders):
        assert entry_dates(["AAA"])["AAA"] == BASE


def test_partial_selling_keeps_the_original_entry():
    orders = [order("AAA", "buy", 10, 0), order("AAA", "sell", 4, 5)]
    with with_orders(orders):
        assert entry_dates(["AAA"])["AAA"] == BASE


def test_closing_and_reopening_reports_the_reopening():
    """The current position started when it last came off zero, not in January."""
    orders = [
        order("AAA", "buy", 10, 0),
        order("AAA", "sell", 10, 4),   # flat here
        order("AAA", "buy", 7, 20),    # this is the current position
    ]
    with with_orders(orders):
        assert entry_dates(["AAA"])["AAA"] == BASE + 20 * DAY


def test_a_fully_closed_symbol_reports_nothing():
    orders = [order("AAA", "buy", 10, 0), order("AAA", "sell", 10, 4)]
    with with_orders(orders):
        assert entry_dates(["AAA"]) == {}


def test_a_short_is_handled_the_same_way():
    """Opening short is a SELL; the sign convention must not break the reset."""
    orders = [order("AAA", "sell", 12, 2), order("AAA", "sell", 3, 6)]
    with with_orders(orders):
        assert entry_dates(["AAA"])["AAA"] == BASE + 2 * DAY


def test_a_short_closed_then_reopened_long():
    orders = [
        order("AAA", "sell", 5, 0),    # short
        order("AAA", "buy", 5, 3),     # covered, flat
        order("AAA", "buy", 8, 11),    # now long
    ]
    with with_orders(orders):
        assert entry_dates(["AAA"])["AAA"] == BASE + 11 * DAY


def test_orders_are_replayed_in_time_order_not_api_order():
    """Alpaca's ordering must not decide the answer."""
    orders = [order("AAA", "buy", 5, 6), order("AAA", "buy", 10, 0)]
    with with_orders(orders):
        assert entry_dates(["AAA"])["AAA"] == BASE


def test_unfilled_and_zero_quantity_orders_are_ignored():
    cancelled = order("AAA", "buy", 0, 0, status="canceled")
    cancelled.filled_at = None
    orders = [cancelled, order("AAA", "buy", 0, 1), order("AAA", "buy", 10, 5)]
    with with_orders(orders):
        assert entry_dates(["AAA"])["AAA"] == BASE + 5 * DAY


def test_several_symbols_are_resolved_from_one_request():
    orders = [
        order("AAA", "buy", 10, 0),
        order("BBB", "sell", 4, 2),
        order("CCC", "buy", 1, 1),
        order("CCC", "sell", 1, 3),   # closed
    ]
    with with_orders(orders):
        result = entry_dates(["AAA", "BBB", "CCC"])
    assert result["AAA"] == BASE
    assert result["BBB"] == BASE + 2 * DAY
    assert "CCC" not in result


def test_no_symbols_makes_no_request():
    called = False

    def fake(*_a, **_k):
        nonlocal called
        called = True
        return []

    with patch("atlas.alpaca_client.call", fake):
        assert entry_dates([]) == {}
    assert not called


def test_an_api_failure_degrades_to_no_dates():
    """A missing entry date is cosmetic and must never break `atlas portfolio`."""
    def boom(*_a, **_k):
        raise RuntimeError("alpaca is down")

    with patch("atlas.alpaca_client.call", boom):
        assert entry_dates(["AAA"]) == {}


def test_fractional_fills_still_reach_exactly_flat():
    """Float noise must not leave a phantom 1e-16 position open."""
    orders = [
        order("AAA", "buy", 0.1, 0),
        order("AAA", "buy", 0.2, 1),
        order("AAA", "sell", 0.3, 2),
    ]
    with with_orders(orders):
        assert entry_dates(["AAA"]) == {}


def test_held_for_renders_a_duration():
    now = datetime.now(timezone.utc)
    assert _held_for(None) == "-"
    assert _held_for(now) == "today"
    assert _held_for(now - 1 * DAY) == "1d"
    assert _held_for(now - 47 * DAY) == "47d"
