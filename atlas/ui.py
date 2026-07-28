"""Terminal output: tables, the order summary block, and confirmations."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()


def money(value: float, decimals: int = 2) -> str:
    return f"${value:,.{decimals}f}"


def signed_money(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"


def signed_pct(value: float) -> str:
    """`value` is a fraction, not a percentage."""
    return f"{value * 100:+.2f}%"


def pnl_text(value: float) -> Text:
    return Text(signed_money(value), style="green" if value >= 0 else "red")


def pct_text(value: float) -> Text:
    return Text(signed_pct(value), style="green" if value >= 0 else "red")


def table(title: str, columns: list[tuple], *, show_header: bool = True) -> Table:
    """Build a table.

    `columns` holds (header, justify) pairs, optionally with a third element of
    extra `add_column` kwargs -- e.g. `{"no_wrap": True, "max_width": 22}` to
    keep a long name on one line instead of letting it wrap over four rows.
    """
    t = Table(
        title=title or None,
        title_justify="left",
        header_style="bold",
        show_header=show_header,
        box=None,
        pad_edge=False,
    )
    for column in columns:
        header, justify = column[0], column[1]
        options = column[2] if len(column) > 2 else {}
        t.add_column(header, justify=justify, **options)
    return t


def warn(message: str) -> None:
    console.print(f"[yellow]warning:[/yellow] {message}", soft_wrap=True)


def error(message: str) -> None:
    # soft_wrap keeps file paths, URLs and copy-pasteable commands on one line
    # instead of letting rich break them mid-token at the console width.
    console.print(f"[red]error:[/red] {message}", highlight=False, soft_wrap=True)


def info(message: str) -> None:
    console.print(message, highlight=False, soft_wrap=True)


def order_summary(
    symbol: str,
    dollars: float,
    shares: float,
    price: float,
    *,
    fractionable: bool,
    side: str = "long",
) -> None:
    """Print the order summary block from docs/commands.md.

    Shorts always show the non-fractionable marker, because Alpaca has no
    fractional shorts and the rounding should be visible, and carry an explicit
    note that the loss is unbounded -- a long can lose at most its cost basis.
    """
    opening_short = side == "short"
    console.print()
    console.print("ORDER SUMMARY:")
    if fractionable:
        console.print(f"  BUY {money(dollars)} of {symbol}  ({shares:.2f} shares)")
    else:
        console.print("  [yellow]! Non-fractionable![/yellow]")
        verb = "SELL SHORT" if opening_short else "BUY"
        whole = int(shares)
        unit = "share" if whole == 1 else "shares"
        console.print(f"  {verb} ~{money(dollars)} of {symbol}  ({whole} {unit})")
    console.print(f"  at {money(price)}/share")
    if opening_short:
        console.print("  [red]! Losses on a short are unbounded.[/red]")
    console.print()


def close_summary(
    symbol: str,
    shares: float,
    price: float,
    proceeds: float,
    unrealized_pl: float,
    unrealized_plpc: float,
    *,
    side: str = "long",
) -> None:
    """Closing a long sells; closing a short buys to cover."""
    action = "SELL" if side == "long" else "BUY TO COVER"
    console.print()
    console.print("ORDER SUMMARY:")
    console.print(f"  {action} entire {side} position in {symbol}  ({abs(shares):g} shares)")
    console.print(f"  at {money(price)}/share  ->  ~{money(abs(proceeds))}")
    console.print(
        Text.assemble(
            "  P&L: ",
            pnl_text(unrealized_pl),
            " (",
            pct_text(unrealized_plpc),
            ")",
        )
    )
    console.print()


def confirm(prompt: str = "Confirm?") -> bool:
    """Ask for [y/n]. Anything other than y/yes is a no.

    A non-interactive stdin declines rather than hanging or defaulting to yes,
    so a piped or scripted invocation can never place an order unattended.
    """
    if not sys.stdin.isatty():
        warn("stdin is not a terminal -- declining automatically.")
        return False
    try:
        # markup=False is required, not cosmetic: rich parses square brackets as
        # style tags, so "[y/n]" is silently swallowed and the prompt renders as
        # "Confirm? : ". Escaping as "\\[y/n]" also works but is easy to lose in
        # a later edit.
        answer = console.input(f"{prompt} [y/n]: ", markup=False).strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False
    return answer in {"y", "yes"}


def ask(prompt: str, choices: dict[str, str]) -> str | None:
    """Ask the user to pick one of `choices`. Returns None if they cannot.

    Used where there is no safe default -- notably the long/short side of an
    order, where guessing would risk opening the opposite position. Same
    non-interactive guard as `confirm()`: a piped invocation aborts rather than
    choosing on the user's behalf.
    """
    keys = "/".join(choices)
    if not sys.stdin.isatty():
        warn(f"stdin is not a terminal -- cannot ask '{prompt}'.")
        return None
    try:
        # markup=False for the same reason as confirm(): rich would eat "[l/s]".
        answer = console.input(f"{prompt} [{keys}]: ", markup=False).strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return None
    return choices.get(answer)
