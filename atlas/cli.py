"""Argument parsing and dispatch for the `atlas` command."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="Kronos-scored asset lookup and paper trading on the Alpaca API.",
    )
    parser.add_argument("--version", action="version", version=f"atlas {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    scan = sub.add_parser(
        "scan", help="score the universe for long and short candidates"
    )
    # Both sides are always computed and written; this only narrows what prints.
    scan.add_argument(
        "side",
        nargs="?",
        choices=("long", "short"),
        help="which table to print (default: both)",
    )
    scan.add_argument("--limit", type=int, help="only scan the N most liquid symbols")
    scan.add_argument("--paths", type=int, help="Monte Carlo paths per symbol (default 25)")
    scan.add_argument("--horizon", type=int, help="trading days to forecast (default 5)")
    scan.add_argument(
        "--rebuild-universe",
        action="store_true",
        help="force a full asset sweep and history reseed",
    )
    scan.add_argument(
        "--model",
        choices=("mini", "small", "base"),
        help="Kronos checkpoint (default small). base is ~4x slower per symbol",
    )

    bt = sub.add_parser(
        "backtest", help="walk-forward test of whether the ranking predicts returns"
    )
    bt.add_argument("side", nargs="?", choices=("long", "short"), help="ranking to test")
    bt.add_argument("--dates", type=int, default=20, help="as-of dates (default 20)")
    bt.add_argument(
        "--spacing", type=int, default=5, help="sessions between as-of dates (default 5)"
    )
    bt.add_argument("--symbols", type=int, default=100, help="symbols to test (default 100)")
    bt.add_argument("--horizon", type=int, help="forward window in sessions (default 5)")
    bt.add_argument("--paths", type=int, help="Monte Carlo paths (default 25)")
    bt.add_argument(
        "--timeframe",
        choices=("5Min", "15Min", "1Hour"),
        help="intraday bars instead of daily -- what Kronos was pretrained on",
    )

    sub.add_parser("portfolio", help="show account balance and open positions")
    sub.add_parser("orders", help="show open orders")

    buy = sub.add_parser("buy", help="open a long or short position in one symbol")
    buy.add_argument("symbol")
    # amount and side are both optional and order-insensitive, so they are
    # parsed by shape rather than position -- see commands/buy.py.
    buy.add_argument(
        "rest",
        nargs="*",
        metavar="[AMOUNT] [l|s]",
        help="dollars (default 100) and side; side is asked for if omitted",
    )
    buy.add_argument("--dry-run", action="store_true", help="show the summary, submit nothing")

    rev = sub.add_parser("review", help="flag open positions worth closing")
    rev.add_argument(
        "--no-forecast",
        action="store_true",
        help="skip the model signal (the four mechanical checks still run)",
    )

    close = sub.add_parser("close", help="close an entire position, long or short")
    close.add_argument("symbol")
    close.add_argument("--dry-run", action="store_true", help="show the summary, submit nothing")

    info = sub.add_parser("info", help="company information for a symbol")
    info.add_argument("symbol")

    view = sub.add_parser("view", help="open an interactive candlestick chart")
    view.add_argument("symbol")
    view.add_argument("period", nargs="?", help="1m 3m 6m 1y 2y 3y 5y, or a day count")

    news = sub.add_parser("news", help="recent financial news")
    news.add_argument("target", nargs="?", help="a symbol, `all`, or omit for scan results")

    # Removed, but kept as a stub so muscle memory gets a pointer rather than
    # argparse's bare "invalid choice".
    removed = sub.add_parser("sell", add_help=False)
    removed.add_argument("args", nargs="*")

    return parser


def _removed_sell() -> int:
    from . import ui  # noqa: PLC0415

    ui.error(
        "`atlas sell` was replaced by `atlas close`, which closes a position in "
        "either direction (selling a long, or buying to cover a short)."
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0
    if args.command == "sell":
        return _removed_sell()

    # Imported lazily: `atlas view` should not pay for torch, and `atlas --help`
    # should not pay for anything.
    from .commands import (  # noqa: PLC0415
        backtest, buy, close, info, news, orders, portfolio, review, scan, view,
    )

    handlers = {
        "scan": scan.run,
        "backtest": backtest.run,
        "portfolio": portfolio.run,
        "orders": orders.run,
        "buy": buy.run,
        "close": close.run,
        "review": review.run,
        "info": info.run,
        "view": view.run,
        "news": news.run,
    }

    from . import config, ui  # noqa: PLC0415

    try:
        # Preflight the credentials so a command fails on the missing file
        # rather than partway through a universe sweep. `info` and `view` are
        # exempt: yfinance and the local bar cache work without an account.
        if args.command not in {"info", "view"}:
            _, warning = config.load_credentials()
            if warning:
                ui.warn(warning)
        return handlers[args.command](args)
    except config.ConfigError as exc:
        ui.error(str(exc))
        return 2
    except KeyboardInterrupt:
        ui.console.print()
        ui.info("interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
