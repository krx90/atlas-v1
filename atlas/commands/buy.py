"""`atlas buy` -- place paper market orders by dollar amount.

Two forms, distinguished by whether the first argument is a number:

    atlas buy XYZ [amount]    a single symbol
    atlas buy N   [amount]    the top N from the latest scan, `amount` each

Alpaca accepts a notional (dollar) order only for fractionable assets, and only
as a market order with day time-in-force. For everything else the order has to
be whole shares, so Atlas detects that upfront and rounds down rather than
letting the API reject the order.
"""

from __future__ import annotations

import math

from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from .. import alpaca_client, config, results, ui


def latest_price(symbol: str) -> float:
    # realtime_feed(), not feed(): quotes are a separate entitlement from bars.
    request = StockLatestTradeRequest(
        symbol_or_symbols=symbol, feed=alpaca_client.realtime_feed()
    )
    trades = alpaca_client.call(alpaca_client.market_data().get_stock_latest_trade, request)
    if symbol not in trades:
        # Alpaca's asset lookup is more permissive than its market data, so a
        # symbol can resolve as an asset and still have no trades on this feed.
        raise LookupError(f"no recent trade on the {alpaca_client.realtime_feed().value} feed")
    return float(trades[symbol].price)


def shares_for(dollars: float, price: float, fractionable: bool) -> tuple[float, float]:
    """Return (shares, actual dollars) for an order.

    Fractional orders spend the full amount. Whole-share orders round *down*,
    so the order never costs more than asked.
    """
    if price <= 0:
        raise ValueError(f"invalid price {price}")
    if fractionable:
        return dollars / price, dollars
    shares = math.floor(dollars / price)
    return float(shares), shares * price


def _place(symbol: str, dollars: float, *, dry_run: bool) -> bool:
    """Quote, confirm and submit one order. Returns True if it was submitted."""
    client = alpaca_client.trading()
    try:
        asset = alpaca_client.call(client.get_asset, symbol)
    except Exception as exc:  # noqa: BLE001
        ui.error(f"{symbol}: could not look up asset ({exc})")
        return False

    if not asset.tradable:
        ui.error(f"{symbol} is not tradable on Alpaca.")
        return False

    try:
        price = latest_price(symbol)
    except Exception as exc:  # noqa: BLE001
        ui.error(f"{symbol}: could not get a price ({exc})")
        return False

    fractionable = bool(asset.fractionable)
    shares, actual = shares_for(dollars, price, fractionable)

    if shares < 1 and not fractionable:
        ui.error(
            f"{symbol} is non-fractionable and one share costs {ui.money(price)} -- "
            f"{ui.money(dollars)} is not enough for a whole share."
        )
        return False

    ui.buy_summary(symbol, actual, shares, price, fractionable)

    if dry_run:
        ui.info("dry run -- not submitted.")
        return False
    if not ui.confirm():
        ui.info("skipped.")
        return False

    if fractionable:
        request = MarketOrderRequest(
            symbol=symbol,
            notional=round(actual, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
    else:
        request = MarketOrderRequest(
            symbol=symbol,
            qty=int(shares),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )

    try:
        order = alpaca_client.call(client.submit_order, request)
    except Exception as exc:  # noqa: BLE001
        ui.error(f"{symbol}: order rejected ({exc})")
        return False

    ui.console.print(f"[green]submitted[/green] {order.id} -- {order.status.value}")
    return True


def _scan_symbols(count: int) -> list[str] | None:
    try:
        rows = results.read()
    except FileNotFoundError:
        ui.error(
            f"No scan results at {config.TOP_ASSETS_CSV}. Run `atlas scan` first."
        )
        return None

    age = results.age_hours()
    if age is not None and age > config.SCAN_STALE_HOURS:
        ui.warn(
            f"scan results are {age:.0f} hours old -- run `atlas scan` to refresh them."
        )

    if count > len(rows):
        ui.warn(f"scan has only {len(rows)} rows; buying all of them.")
    return [row["symbol"] for row in rows[:count]]


def run(args) -> int:
    dollars = args.amount if args.amount is not None else config.DEFAULT_ORDER_DOLLARS
    if dollars <= 0:
        ui.error("amount must be positive.")
        return 2

    target = args.target
    if target.isdigit():
        symbols = _scan_symbols(int(target))
        if symbols is None:
            return 1
        ui.info(
            f"Buying the top {len(symbols)} from the latest scan at "
            f"{ui.money(dollars)} each: {', '.join(symbols)}"
        )
    else:
        symbols = [target.upper()]

    account = alpaca_client.call(alpaca_client.trading().get_account)
    buying_power = float(account.buying_power)
    if not args.dry_run and dollars * len(symbols) > buying_power:
        ui.warn(
            f"{ui.money(dollars * len(symbols))} of orders exceeds "
            f"{ui.money(buying_power)} buying power -- later orders may be rejected."
        )

    placed = sum(_place(symbol, dollars, dry_run=args.dry_run) for symbol in symbols)

    if len(symbols) > 1:
        ui.console.print()
        ui.info(f"{placed} of {len(symbols)} orders submitted.")
    return 0 if placed or args.dry_run else 1
