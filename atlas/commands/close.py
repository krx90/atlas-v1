"""`atlas close XYZ` -- close an entire position, in either direction.

Closing is by symbol only. There is deliberately no bulk form: ranking positions
and closing the worst few is easy to fire by accident and hard to undo.

Alpaca's `close_position` handles both directions itself -- it sells a long and
buys to cover a short -- so the work here is reporting accurately which of those
is about to happen, since the two are opposite trades.
"""

from __future__ import annotations

from .. import alpaca_client, ui


def _mention_pending_orders(symbol: str) -> None:
    """Explain an unfilled order rather than leaving 'no position' looking wrong.

    An order placed while the market is closed sits as `new` until the open, so
    there is genuinely no position yet. Saying only 'no open position' seconds
    after the user watched an order get submitted reads like a bug.
    """
    from alpaca.trading.enums import QueryOrderStatus  # noqa: PLC0415
    from alpaca.trading.requests import GetOrdersRequest  # noqa: PLC0415

    try:
        pending = alpaca_client.call(
            alpaca_client.trading().get_orders,
            GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol], limit=10),
        )
    except Exception:  # noqa: BLE001 -- this is a courtesy, never the main path
        return
    if not pending:
        return

    order = pending[0]
    size = f"{float(order.qty):g} sh" if order.qty else ui.money(float(order.notional or 0))
    ui.info(
        f"  {len(pending)} unfilled order for {symbol} "
        f"({str(order.side.value).upper()} {size}, {order.status.value}) -- "
        "a position appears once it fills."
    )
    try:
        clock = alpaca_client.call(alpaca_client.trading().get_clock)
        if not clock.is_open:
            ui.info(f"  The market is closed; it next opens {clock.next_open.astimezone():%H:%M on %d %b}.")
    except Exception:  # noqa: BLE001
        pass


def run(args) -> int:
    symbol = args.symbol.upper()
    client = alpaca_client.trading()

    try:
        position = alpaca_client.call(client.get_open_position, symbol)
    except Exception:  # noqa: BLE001 -- Alpaca 404s for anything not held
        ui.error(f"No open position in {symbol}.")
        _mention_pending_orders(symbol)
        return 1

    # A short reports a negative qty and market value; the summary shows
    # magnitudes and names the direction rather than printing a confusing
    # negative share count.
    side = str(getattr(position.side, "value", position.side))
    shares = float(position.qty)
    price = float(position.current_price)
    proceeds = float(position.market_value)

    ui.close_summary(
        symbol,
        shares,
        price,
        proceeds,
        float(position.unrealized_pl),
        float(position.unrealized_plpc),
        side=side,
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
