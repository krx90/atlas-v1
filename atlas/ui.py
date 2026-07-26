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


def buy_summary(
    symbol: str,
    dollars: float,
    shares: float,
    price: float,
    fractionable: bool,
) -> None:
    """Print the order summary block from docs/commands.md verbatim."""
    console.print()
    console.print("ORDER SUMMARY:")
    if fractionable:
        console.print(f"  BUY {money(dollars)} of {symbol}  ({shares:.2f} shares)")
    else:
        console.print("  [yellow]! Non-fractionable![/yellow]")
        console.print(f"  BUY ~{money(dollars)} of {symbol}  ({int(shares)} shares)")
    console.print(f"  at {money(price)}/share")
    console.print()


def sell_summary(
    symbol: str,
    shares: float,
    price: float,
    proceeds: float,
    unrealized_pl: float,
    unrealized_plpc: float,
) -> None:
    console.print()
    console.print("ORDER SUMMARY:")
    console.print(f"  SELL entire position in {symbol}  ({shares:g} shares)")
    console.print(f"  at {money(price)}/share  ->  ~{money(proceeds)}")
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
