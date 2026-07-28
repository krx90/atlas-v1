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


def test_long_order_summary_matches_the_documented_format():
    with ui.console.capture() as cap:
        ui.order_summary("RING", 500.0, 11.05, 45.23, fractionable=True)
    out = cap.get()

    assert "ORDER SUMMARY:" in out
    assert "BUY $500.00 of RING  (11.05 shares)" in out
    assert "at $45.23/share" in out
    assert "Non-fractionable" not in out


def test_non_fractionable_summary_shows_the_warning_and_whole_shares():
    with ui.console.capture() as cap:
        ui.order_summary("LMTL", 987.45, 47.0, 21.01, fractionable=False)
    out = cap.get()

    assert "! Non-fractionable!" in out
    assert "BUY ~$987.45 of LMTL  (47 shares)" in out
    assert "47.00 shares" not in out  # whole shares, not a fractional count


def test_close_summary_includes_proceeds_and_pnl():
    with ui.console.capture() as cap:
        ui.close_summary("RING", 11.05, 45.23, 499.79, 12.34, 0.0253)
    out = cap.get()

    assert "SELL entire long position in RING  (11.05 shares)" in out
    assert "at $45.23/share" in out
    assert "$499.79" in out
    assert "+$12.34" in out
    assert "+2.53%" in out


def test_a_loss_renders_with_a_negative_sign():
    with ui.console.capture() as cap:
        ui.close_summary("PCT", 130.889, 5.85, 765.70, -234.29, -0.2343)
    out = cap.get()

    assert "-$234.29" in out
    assert "-23.43%" in out


# --- short and close summaries ------------------------------------------------


def test_a_short_summary_names_the_direction_and_the_risk():
    with ui.console.capture() as cap:
        ui.order_summary("XYZ", 487.20, 12.0, 40.60, fractionable=False, side="short")
    out = cap.get()

    assert "SELL SHORT ~$487.20 of XYZ  (12 shares)" in out
    # Alpaca has no fractional shorts, so the rounding is always visible.
    assert "! Non-fractionable!" in out
    assert "Losses on a short are unbounded." in out
    assert "BUY " not in out


def test_a_long_summary_never_warns_about_unbounded_loss():
    """A long can lose at most its cost basis; the warning would be wrong."""
    with ui.console.capture() as cap:
        ui.order_summary("RING", 500.0, 11.05, 45.23, fractionable=True, side="long")
    assert "unbounded" not in cap.get()


def test_closing_a_short_buys_to_cover():
    with ui.console.capture() as cap:
        ui.close_summary("XYZ", -12.0, 40.60, -487.20, 25.00, 0.05, side="short")
    out = cap.get()

    assert "BUY TO COVER entire short position in XYZ  (12 shares)" in out
    # Magnitudes, not the negative qty and value Alpaca reports for a short.
    assert "-12" not in out
    assert "$487.20" in out
    assert "+$25.00" in out


def test_closing_a_long_sells():
    with ui.console.capture() as cap:
        ui.close_summary("RING", 11.05, 45.23, 499.79, 12.34, 0.0253, side="long")
    out = cap.get()
    assert "SELL entire long position in RING" in out
    assert "COVER" not in out


# --- the generic chooser ------------------------------------------------------


def test_ask_maps_answers_through_its_choices(tty):
    with patch.object(ui.console, "input", lambda *_a, **_k: "b"):
        assert ui.ask("Pick", {"a": "apple", "b": "banana"}) == "banana"


def test_ask_returns_none_for_an_answer_not_offered(tty):
    with patch.object(ui.console, "input", lambda *_a, **_k: "z"):
        assert ui.ask("Pick", {"a": "apple", "b": "banana"}) is None


def test_ask_lists_the_choices_in_the_prompt(tty):
    seen = {}

    def fake(prompt, **kwargs):
        seen.update(prompt=prompt, kwargs=kwargs)
        return "a"

    with patch.object(ui.console, "input", fake):
        ui.ask("Pick", {"a": "apple", "b": "banana"})

    assert seen["prompt"] == "Pick [a/b]: "
    assert seen["kwargs"].get("markup") is False


def test_a_single_whole_share_is_not_pluralised():
    with ui.console.capture() as cap:
        ui.order_summary("AAPL", 333.93, 1.0, 333.93, fractionable=False, side="short")
    out = cap.get()
    assert "(1 share)" in out
    assert "1 shares" not in out
