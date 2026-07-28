"""`atlas buy` -- open one position, long or short.

    atlas buy XYZ [AMOUNT] <l|s>

One symbol at a time. There is no bulk form: buying a list in one command makes
it easy to take positions you did not individually look at.

The side is **required**, and asked for rather than defaulted when omitted.
Defaulting to long would mean a mistyped command silently opens the opposite of
the intended position, which is the one mistake here that cannot be undone by
reading the summary.

Alpaca accepts a notional (dollar) order only for a fractionable asset bought
long, and only as a market order with day time-in-force. Shorts are always whole
shares. Both cases round *down* so an order never costs more than asked.
"""

from __future__ import annotations

import math

from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from .. import alpaca_client, config, ui
from ..scoring import LONG, SHORT

_SIDE_WORDS = {"l": LONG, "long": LONG, "s": SHORT, "short": SHORT}


class BadArguments(ValueError):
    pass


def parse_args(tokens: list[str]) -> tuple[float | None, str | None]:
    """Pull an optional amount and an optional side out of trailing tokens.

    Both are optional and either order reads naturally, so they are identified
    by shape -- a number is the amount, l/s is the side -- rather than by
    position. Returns (amount, side), each None if not given.
    """
    amount: float | None = None
    side: str | None = None

    for token in tokens:
        key = token.strip().lower()
        if key in _SIDE_WORDS:
            if side is not None:
                raise BadArguments("side given twice")
            side = _SIDE_WORDS[key]
            continue
        try:
            value = float(key)
        except ValueError:
            raise BadArguments(
                f"unrecognised argument {token!r} -- expected a dollar amount or l/s"
            ) from None
        if amount is not None:
            raise BadArguments("amount given twice")
        if value <= 0:
            raise BadArguments("amount must be positive")
        amount = value

    return amount, side


def ask_side() -> str | None:
    """Prompt for the side. Returns None if the user declines or cannot answer."""
    answer = ui.ask("Long or short?", {"l": LONG, "s": SHORT})
    return answer


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


def _place(symbol: str, dollars: float, side: str, *, dry_run: bool) -> bool:
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

    if side == SHORT:
        # Checked here rather than left to a rejection: an unborrowable symbol
        # is not a candidate at all, and the reason is worth stating.
        if not asset.shortable:
            ui.error(f"{symbol} is not shortable on Alpaca.")
            return False
        if not asset.easy_to_borrow:
            ui.error(f"{symbol} is not easy to borrow -- Atlas will not short it.")
            return False

    try:
        price = latest_price(symbol)
    except Exception as exc:  # noqa: BLE001
        ui.error(f"{symbol}: could not get a price ({exc})")
        return False

    # Alpaca has no fractional shorts, so a short is whole-share regardless of
    # what the asset itself supports.
    fractionable = bool(asset.fractionable) and side == LONG
    shares, actual = shares_for(dollars, price, fractionable)

    if shares < 1 and not fractionable:
        ui.error(
            f"{symbol} needs whole shares and one costs {ui.money(price)} -- "
            f"{ui.money(dollars)} is not enough."
        )
        return False

    ui.order_summary(symbol, actual, shares, price, fractionable=fractionable, side=side)

    if dry_run:
        ui.info("dry run -- not submitted.")
        return False
    if not ui.confirm():
        ui.info("skipped.")
        return False

    order_side = OrderSide.BUY if side == LONG else OrderSide.SELL
    if fractionable:
        request = MarketOrderRequest(
            symbol=symbol,
            notional=round(actual, 2),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
    else:
        request = MarketOrderRequest(
            symbol=symbol,
            qty=int(shares),
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )

    try:
        order = alpaca_client.call(client.submit_order, request)
    except Exception as exc:  # noqa: BLE001
        ui.error(f"{symbol}: order rejected ({exc})")
        return False

    ui.console.print(f"[green]submitted[/green] {order.id} -- {order.status.value}")
    return True


def run(args) -> int:
    symbol = args.symbol.strip()
    if symbol.isdigit():
        ui.error(
            "`atlas buy` takes a symbol, not a count -- buying the top N from a scan "
            "was removed. Buy one symbol at a time, e.g. `atlas buy AAPL 500 l`."
        )
        return 2

    try:
        amount, side = parse_args(args.rest)
    except BadArguments as exc:
        ui.error(str(exc))
        return 2

    if side is None:
        side = ask_side()
        if side is None:
            ui.error("no side given -- specify `l` for long or `s` for short.")
            return 2

    dollars = amount if amount is not None else config.DEFAULT_ORDER_DOLLARS

    account = alpaca_client.call(alpaca_client.trading().get_account)
    buying_power = float(account.buying_power)
    if not args.dry_run and dollars > buying_power:
        ui.warn(
            f"{ui.money(dollars)} exceeds {ui.money(buying_power)} buying power -- "
            "the order may be rejected."
        )

    placed = _place(symbol.upper(), dollars, side, dry_run=args.dry_run)
    return 0 if placed or args.dry_run else 1
