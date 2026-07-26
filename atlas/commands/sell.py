"""`atlas sell XYZ` -- close an entire position.

Selling is by symbol only. There is deliberately no `atlas sell N`: ranking
positions worst-to-best and closing the bottom few is a bulk action that is
easy to fire by accident and hard to undo.
"""

from __future__ import annotations

from .. import alpaca_client, ui


def run(args) -> int:
    symbol = args.symbol.upper()
    client = alpaca_client.trading()

    try:
        position = alpaca_client.call(client.get_open_position, symbol)
    except Exception:  # noqa: BLE001 -- Alpaca 404s for anything not held
        ui.error(f"No open position in {symbol}.")
        return 1

    shares = float(position.qty)
    price = float(position.current_price)
    proceeds = float(position.market_value)

    ui.sell_summary(
        symbol,
        shares,
        price,
        proceeds,
        float(position.unrealized_pl),
        float(position.unrealized_plpc),
    )

    if args.dry_run:
        ui.info("dry run -- not submitted.")
        return 0
    if not ui.confirm():
        ui.info("cancelled.")
        return 0

    try:
        order = alpaca_client.call(client.close_position, symbol)
    except Exception as exc:  # noqa: BLE001
        ui.error(f"{symbol}: close rejected ({exc})")
        return 1

    ui.console.print(f"[green]submitted[/green] {order.id} -- {order.status.value}")
    return 0
