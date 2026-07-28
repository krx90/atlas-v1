"""`atlas portfolio` -- account balance and open positions."""

from __future__ import annotations

from datetime import datetime, timezone

from .. import alpaca_client, ui

#: Orders scanned when reconstructing entry dates. Alpaca returns them newest
#: first; a position older than this window reports "-" rather than a wrong date.
_ORDER_HISTORY_LIMIT = 500


def entry_dates(symbols: list[str]) -> dict[str, datetime]:
    """When each current position was opened, derived from fill history.

    Alpaca's Position model carries no timestamp, so this replays filled orders
    oldest-first and tracks the running signed quantity. The entry is the fill
    that took the position off zero and which it never returned to zero after --
    so scaling in keeps the *original* entry, while a symbol that was closed and
    later reopened reports the reopening, not the first time you ever traded it.

    One request covers every held symbol. Symbols whose opening fill predates the
    history window are simply absent from the result.
    """
    from alpaca.trading.enums import QueryOrderStatus  # noqa: PLC0415
    from alpaca.trading.requests import GetOrdersRequest  # noqa: PLC0415

    if not symbols:
        return {}
    try:
        orders = alpaca_client.call(
            alpaca_client.trading().get_orders,
            GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=symbols,
                limit=_ORDER_HISTORY_LIMIT,
                direction="asc",
            ),
        )
    except Exception:  # noqa: BLE001 -- a missing date is cosmetic, never fatal
        return {}

    fills: dict[str, list] = {}
    for order in orders:
        if not order.filled_at or not order.filled_qty:
            continue
        if float(order.filled_qty) <= 0:
            continue
        fills.setdefault(order.symbol, []).append(order)

    out: dict[str, datetime] = {}
    for symbol, symbol_orders in fills.items():
        running = 0.0
        opened_at = None
        for order in sorted(symbol_orders, key=lambda o: o.filled_at):
            qty = float(order.filled_qty)
            signed = qty if str(order.side.value) == "buy" else -qty
            if abs(running) < 1e-9:
                opened_at = order.filled_at
            running += signed
            if abs(running) < 1e-9:  # flat again -- the next fill starts anew
                opened_at = None
        if opened_at is not None:
            out[symbol] = opened_at
    return out


def _held_for(opened: datetime | None) -> str:
    if opened is None:
        return "-"
    days = (datetime.now(timezone.utc) - opened).days
    return f"{days}d" if days else "today"


def run(args) -> int:
    client = alpaca_client.trading()
    account = alpaca_client.call(client.get_account)
    positions = alpaca_client.call(client.get_all_positions)

    equity = float(account.equity)
    last_equity = float(account.last_equity)
    day_change = equity - last_equity
    day_pct = day_change / last_equity if last_equity else 0.0

    ui.console.print()
    ui.console.print(
        f"[bold]Paper account[/bold]  {account.account_number}", highlight=False
    )
    ui.console.print(
        f"  equity {ui.money(equity)}   cash {ui.money(float(account.cash))}   "
        f"buying power {ui.money(float(account.buying_power))}",
        highlight=False,
    )
    ui.console.print(
        ui.Text.assemble(
            "  today  ", ui.pnl_text(day_change), " (", ui.pct_text(day_pct), ")"
        )
    )

    if not positions:
        ui.console.print()
        ui.info("No open positions.")
        return 0

    opened = entry_dates([p.symbol for p in positions])

    table = ui.table(
        "Positions",
        [
            ("Symbol", "left"),
            ("Side", "left"),
            ("Entry", "left"),
            ("Held", "right"),
            ("Qty", "right"),
            ("Avg entry", "right"),
            ("Current", "right"),
            ("Value", "right"),
            ("P&L", "right"),
            ("P&L %", "right"),
        ],
    )
    rows = sorted(positions, key=lambda p: float(p.unrealized_plpc))
    total_value = 0.0
    total_pl = 0.0
    for p in rows:
        # A short reports negative qty and market value. Exposure is what the
        # position is worth either way, so totals use magnitudes -- summing the
        # signed values would net a hedged book to nearly nothing and read as
        # though there were no positions at all.
        side = str(getattr(p.side, "value", p.side))
        value = float(p.market_value)
        pl = float(p.unrealized_pl)
        total_value += abs(value)
        total_pl += pl
        entered = opened.get(p.symbol)
        table.add_row(
            p.symbol,
            ui.Text(side.upper(), style="cyan" if side == "long" else "magenta"),
            entered.astimezone().strftime("%Y-%m-%d") if entered else "-",
            _held_for(entered),
            f"{abs(float(p.qty)):g}",
            ui.money(float(p.avg_entry_price)),
            ui.money(float(p.current_price)),
            ui.money(abs(value)),
            ui.pnl_text(pl),
            ui.pct_text(float(p.unrealized_plpc)),
        )

    cost_basis = total_value - total_pl
    table.add_section()
    table.add_row(
        f"{len(rows)} positions",
        "",
        "",
        "",
        "",
        "",
        "",
        ui.money(total_value),
        ui.pnl_text(total_pl),
        ui.pct_text(total_pl / cost_basis if cost_basis else 0.0),
        style="bold",
    )

    ui.console.print()
    ui.console.print(table)
    return 0
