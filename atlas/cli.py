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

    scan = sub.add_parser("scan", help="score the universe and write top30_assets.csv")
    scan.add_argument("--limit", type=int, help="only scan the N most liquid symbols")
    scan.add_argument("--paths", type=int, help="Monte Carlo paths per symbol (default 25)")
    scan.add_argument("--horizon", type=int, help="trading days to forecast (default 5)")
    scan.add_argument(
        "--rebuild-universe",
        action="store_true",
        help="force a full asset sweep and history reseed",
    )

    sub.add_parser("portfolio", help="show account balance and open positions")
    sub.add_parser("orders", help="show open orders")

    buy = sub.add_parser("buy", help="buy a symbol, or the top N from the latest scan")
    buy.add_argument("target", help="a symbol (XYZ) or a count (5)")
    buy.add_argument("amount", nargs="?", type=float, help="dollars per order (default 100)")
    buy.add_argument("--dry-run", action="store_true", help="show the summary, submit nothing")

    sell = sub.add_parser("sell", help="close an entire position")
    sell.add_argument("symbol")
    sell.add_argument("--dry-run", action="store_true", help="show the summary, submit nothing")

    info = sub.add_parser("info", help="company information for a symbol")
    info.add_argument("symbol")

    view = sub.add_parser("view", help="open an interactive candlestick chart")
    view.add_argument("symbol")
    view.add_argument("period", nargs="?", help="1m 3m 6m 1y 2y 3y 5y, or a day count")

    news = sub.add_parser("news", help="recent financial news")
    news.add_argument("target", nargs="?", help="a symbol, `all`, or omit for scan results")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    # Imported lazily: `atlas view` should not pay for torch, and `atlas --help`
    # should not pay for anything.
    from .commands import buy, info, news, orders, portfolio, scan, sell, view  # noqa: PLC0415

    handlers = {
        "scan": scan.run,
        "portfolio": portfolio.run,
        "orders": orders.run,
        "buy": buy.run,
        "sell": sell.run,
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
