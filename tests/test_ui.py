"""Rendering of the confirmation prompt and order summaries.

These exist because the earlier verification never rendered the interactive
prompt: `--dry-run` returns before it, and a non-TTY stdin declines before it.
So the one string `docs/commands.md` specifies verbatim -- `Confirm? [y/n]:` --
went out broken, because rich parses square brackets as style tags and silently
dropped the `[y/n]`.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from atlas import ui


@pytest.fixture
def tty():
    """Pretend stdin is interactive so confirm() reaches the prompt."""
    with patch.object(sys.stdin, "isatty", return_value=True):
        yield


def captured_input(answer="y"):
    """Patch console.input, recording the prompt it was asked to render."""
    seen = {}

    def fake(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["kwargs"] = kwargs
        return answer

    return fake, seen


def test_prompt_renders_the_y_n_brackets(tty):
    """The literal string from docs/commands.md must survive rich."""
    fake, seen = captured_input()
    with patch.object(ui.console, "input", fake):
        ui.confirm()

    assert seen["prompt"] == "Confirm? [y/n]: "
    # The mechanism that makes it survive. Without this rich renders "Confirm? : ".
    assert seen["kwargs"].get("markup") is False


def test_rich_would_eat_the_brackets_without_markup_false():
    """Guards the reason for the fix, so it is not 'simplified' away later."""
    with ui.console.capture() as cap:
        ui.console.print("Confirm? [y/n]: ", end="")
    assert cap.get() == "Confirm? : "

    with ui.console.capture() as cap:
        ui.console.print("Confirm? [y/n]: ", end="", markup=False)
    assert cap.get() == "Confirm? [y/n]: "


@pytest.mark.parametrize("answer", ["y", "Y", "yes", "YES", " y ", "Yes"])
def test_affirmative_answers(tty, answer):
    fake, _ = captured_input(answer)
    with patch.object(ui.console, "input", fake):
        assert ui.confirm() is True


@pytest.mark.parametrize("answer", ["n", "no", "", " ", "maybe", "ye", "yolo", "1", "yy"])
def test_everything_else_declines(tty, answer):
    """A stray keypress must not place an order."""
    fake, _ = captured_input(answer)
    with patch.object(ui.console, "input", fake):
        assert ui.confirm() is False


def test_non_interactive_stdin_declines_without_prompting():
    """A piped or scripted invocation must never trade unattended."""
    called = False

    def fake(*_a, **_k):
        nonlocal called
        called = True
        return "y"

    with patch.object(sys.stdin, "isatty", return_value=False), \
         patch.object(ui.console, "input", fake):
        assert ui.confirm() is False
    assert not called, "confirm() must not even reach the prompt on a non-TTY"


@pytest.mark.parametrize("boom", [EOFError, KeyboardInterrupt])
def test_interrupting_the_prompt_declines(tty, boom):
    with patch.object(ui.console, "input", side_effect=boom):
        assert ui.confirm() is False


def test_a_custom_prompt_is_used_verbatim(tty):
    fake, seen = captured_input()
    with patch.object(ui.console, "input", fake):
        ui.confirm("Close this position?")
    assert seen["prompt"] == "Close this position? [y/n]: "


def test_buy_summary_matches_the_documented_format():
    with ui.console.capture() as cap:
        ui.buy_summary("RING", 500.0, 11.05, 45.23, fractionable=True)
    out = cap.get()

    assert "ORDER SUMMARY:" in out
    assert "BUY $500.00 of RING  (11.05 shares)" in out
    assert "at $45.23/share" in out
    assert "Non-fractionable" not in out


def test_non_fractionable_summary_shows_the_warning_and_whole_shares():
    with ui.console.capture() as cap:
        ui.buy_summary("LMTL", 987.45, 47.0, 21.01, fractionable=False)
    out = cap.get()

    assert "! Non-fractionable!" in out
    assert "BUY ~$987.45 of LMTL  (47 shares)" in out
    assert "47.00 shares" not in out  # whole shares, not a fractional count


def test_sell_summary_includes_proceeds_and_pnl():
    with ui.console.capture() as cap:
        ui.sell_summary("RING", 11.05, 45.23, 499.79, 12.34, 0.0253)
    out = cap.get()

    assert "SELL entire position in RING  (11.05 shares)" in out
    assert "at $45.23/share" in out
    assert "$499.79" in out
    assert "+$12.34" in out
    assert "+2.53%" in out


def test_a_loss_renders_with_a_negative_sign():
    with ui.console.capture() as cap:
        ui.sell_summary("PCT", 130.889, 5.85, 765.70, -234.29, -0.2343)
    out = cap.get()

    assert "-$234.29" in out
    assert "-23.43%" in out
