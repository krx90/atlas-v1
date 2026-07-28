"""Argument parsing and side selection for `atlas buy`.

    atlas buy XYZ [AMOUNT] <l|s>

Amount and side are both optional tokens in either order, so they are
identified by shape rather than position. The side has no default: guessing
would risk opening the opposite of the intended position.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from atlas import ui
from atlas.commands.buy import BadArguments, ask_side, parse_args, shares_for


@pytest.fixture
def tty():
    with patch.object(sys.stdin, "isatty", return_value=True):
        yield


# --- parsing ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tokens", "amount", "side"),
    [
        ([], None, None),
        (["500"], 500.0, None),
        (["l"], None, "long"),
        (["s"], None, "short"),
        (["500", "l"], 500.0, "long"),
        (["500", "s"], 500.0, "short"),
        (["s", "500"], 500.0, "short"),  # order-insensitive
        (["long"], None, "long"),
        (["short"], None, "short"),
        (["L"], None, "long"),  # case-insensitive
        (["S", "1000"], 1000.0, "short"),
        (["12.50", "l"], 12.5, "long"),
    ],
)
def test_tokens_are_identified_by_shape_not_position(tokens, amount, side):
    assert parse_args(tokens) == (amount, side)


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        (["500", "1000"], "amount given twice"),
        (["l", "s"], "side given twice"),
        (["l", "long"], "side given twice"),
        (["banana"], "unrecognised argument"),
        (["0", "l"], "amount must be positive"),
        (["-50", "l"], "amount must be positive"),
    ],
)
def test_bad_arguments_are_rejected_with_a_reason(tokens, message):
    with pytest.raises(BadArguments, match=message):
        parse_args(tokens)


# --- side selection -----------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("l", "long"), ("s", "short"), ("L", "long"), (" s ", "short")],
)
def test_the_side_prompt_accepts_l_and_s(tty, answer, expected):
    with patch.object(ui.console, "input", lambda *_a, **_k: answer):
        assert ask_side() == expected


@pytest.mark.parametrize("answer", ["", "x", "long?", "y", "1", "b"])
def test_an_unrecognised_answer_selects_nothing(tty, answer):
    """No guessing: an unclear answer yields None, and the caller aborts."""
    with patch.object(ui.console, "input", lambda *_a, **_k: answer):
        assert ask_side() is None


def test_the_side_prompt_renders_its_brackets(tty):
    """Rich eats "[l/s]" unless markup is disabled -- same trap as confirm()."""
    seen = {}

    def fake(prompt, **kwargs):
        seen.update(prompt=prompt, kwargs=kwargs)
        return "l"

    with patch.object(ui.console, "input", fake):
        ask_side()

    assert seen["prompt"] == "Long or short? [l/s]: "
    assert seen["kwargs"].get("markup") is False


def test_a_non_interactive_stdin_cannot_choose_a_side():
    """A piped invocation must abort, never pick a direction on its own."""
    called = False

    def fake(*_a, **_k):
        nonlocal called
        called = True
        return "l"

    with patch.object(sys.stdin, "isatty", return_value=False), \
         patch.object(ui.console, "input", fake):
        assert ask_side() is None
    assert not called, "must not prompt when stdin is not a terminal"


@pytest.mark.parametrize("boom", [EOFError, KeyboardInterrupt])
def test_interrupting_the_side_prompt_selects_nothing(tty, boom):
    with patch.object(ui.console, "input", side_effect=boom):
        assert ask_side() is None


# --- share arithmetic ---------------------------------------------------------


def test_shorts_are_whole_shares_even_on_a_fractionable_asset():
    """Alpaca has no fractional shorts, so the caller passes fractionable=False."""
    shares, dollars = shares_for(1000.0, 333.33, fractionable=False)
    assert shares == 3
    assert dollars == pytest.approx(999.99)
    assert dollars <= 1000.0


def test_a_short_too_small_for_one_share_yields_zero():
    shares, dollars = shares_for(50.0, 333.33, fractionable=False)
    assert shares == 0
    assert dollars == 0
