"""The symbol filter that runs before any bars are fetched."""

from __future__ import annotations

import pytest

from atlas.universe import _is_plain_equity


@pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "F", "SPY", "GOOGL", "BRKB"])
def test_ordinary_tickers_are_kept(symbol):
    assert _is_plain_equity(symbol)


@pytest.mark.parametrize(
    "symbol",
    [
        "BRK.A",  # class share, dotted
        "BRK/A",  # the slash form of the same
        "ABCD.WS",  # warrant
        "ABCD.U",  # unit
        "ABCD.R",  # right
        "ABC-PA",  # preferred series A
        "ABC.PB",  # preferred, dotted form
        "TOOLONG",  # not a US equity ticker
        "",
    ],
)
def test_non_common_lines_are_dropped(symbol):
    assert not _is_plain_equity(symbol)


def test_five_characters_is_the_limit():
    assert _is_plain_equity("ABCDE")
    assert not _is_plain_equity("ABCDEF")
