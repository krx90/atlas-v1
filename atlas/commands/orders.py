"""`atlas orders` -- currently open orders on the paper account."""

from __future__ import annotations

from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from .. import alpaca_client, ui


def _size(order) -> str:
    if order.qty is not None:
        return f"{float(order.qty):g} sh"
    if order.notional is not None:
        return ui.money(float(order.notional))
    return "-"


def run(args) -> int:
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
    open_orders = alpaca_client.call(alpaca_client.trading().get_orders, request)

    if not open_orders:
        ui.console.print()
        ui.info("No open orders.")
        return 0

    table = ui.table(
        f"Open orders ({len(open_orders)})",
        [
            ("Submitted", "left"),
            ("Symbol", "left"),
            ("Side", "left"),
            ("Type", "left"),
            ("Size", "right"),
            ("Filled", "right"),
            ("TIF", "left"),
            ("Status", "left"),
            ("ID", "left"),
        ],
    )
    for order in sorted(open_orders, key=lambda o: o.submitted_at, reverse=True):
        side = str(order.side.value).upper()
        table.add_row(
            order.submitted_at.astimezone().strftime("%m-%d %H:%M"),
            order.symbol,
            ui.Text(side, style="green" if side == "BUY" else "red"),
            str(order.order_type.value),
            _size(order),
            f"{float(order.filled_qty or 0):g}",
            str(order.time_in_force.value).upper(),
            str(order.status.value),
            str(order.id)[:8],
        )

    ui.console.print()
    ui.console.print(table)
    return 0
