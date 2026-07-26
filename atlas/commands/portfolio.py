"""`atlas portfolio` -- account balance and open positions."""

from __future__ import annotations

from .. import alpaca_client, ui


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

    table = ui.table(
        "Positions",
        [
            ("Symbol", "left"),
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
        value = float(p.market_value)
        pl = float(p.unrealized_pl)
        total_value += value
        total_pl += pl
        table.add_row(
            p.symbol,
            f"{float(p.qty):g}",
            ui.money(float(p.avg_entry_price)),
            ui.money(float(p.current_price)),
            ui.money(value),
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
        ui.money(total_value),
        ui.pnl_text(total_pl),
        ui.pct_text(total_pl / cost_basis if cost_basis else 0.0),
        style="bold",
    )

    ui.console.print()
    ui.console.print(table)
    return 0
