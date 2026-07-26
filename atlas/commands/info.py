"""`atlas info XYZ` -- company fundamentals, via yfinance."""

from __future__ import annotations

import logging
import textwrap

from .. import ui


def _human(value) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(number) >= threshold:
            return f"${number / threshold:,.2f}{suffix}"
    return f"${number:,.2f}"


def run(args) -> int:
    symbol = args.symbol.upper()

    try:
        import yfinance  # noqa: PLC0415
    except ImportError:
        ui.error("`atlas info` requires yfinance:  pip install yfinance")
        return 1

    # yfinance logs upstream HTTP errors itself; we report them our own way.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    try:
        data = yfinance.Ticker(symbol).info or {}
    except Exception as exc:  # noqa: BLE001
        ui.error(f"{symbol}: yfinance lookup failed ({exc})")
        return 1

    if not data.get("symbol") and not data.get("shortName"):
        ui.error(f"{symbol}: no company information found.")
        return 1

    low = data.get("fiftyTwoWeekLow")
    high = data.get("fiftyTwoWeekHigh")
    week52 = f"{ui.money(low)} - {ui.money(high)}" if low and high else "-"
    pe = data.get("trailingPE")
    employees = data.get("fullTimeEmployees")

    ui.console.print()
    ui.console.print(f"[bold]{data.get('longName') or data.get('shortName') or symbol}[/bold]  ({symbol})")
    ui.console.print()

    table = ui.table("", [("", "left"), ("", "left")], show_header=False)
    table.add_row("Sector", data.get("sector") or "-")
    table.add_row("Industry", data.get("industry") or "-")
    table.add_row("Market cap", _human(data.get("marketCap")))
    table.add_row("P/E (trailing)", f"{pe:.2f}" if isinstance(pe, (int, float)) else "-")
    table.add_row("52-week range", week52)
    table.add_row("Employees", f"{employees:,}" if isinstance(employees, int) else "-")
    table.add_row("Website", data.get("website") or "-")
    ui.console.print(table)

    summary = data.get("longBusinessSummary")
    if summary:
        # Wrap once, ourselves, and tell rich not to wrap the result again.
        width = min(88, ui.console.width)
        ui.console.print()
        # markup=False: this is arbitrary prose from yfinance, and rich would
        # read any "[...]" in it as a style tag and drop it.
        ui.console.print(
            textwrap.fill(summary[:900], width=width), soft_wrap=True, markup=False
        )
    return 0
