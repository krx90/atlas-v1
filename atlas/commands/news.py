"""`atlas news [XYZ|all]` -- recent financial news.

No argument reads the symbols from the latest scan; `all` widens to general
market news from Alpaca plus a handful of RSS feeds.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from alpaca.data.requests import NewsRequest

from .. import alpaca_client, config, results, ui


def _since() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=config.NEWS_LOOKBACK_HOURS)


def _fetch_alpaca(symbols: list[str] | None, limit: int = 50) -> list:
    request = NewsRequest(
        symbols=",".join(symbols) if symbols else None,
        start=_since(),
        limit=limit,
        include_content=False,
    )
    try:
        response = alpaca_client.call(alpaca_client.news().get_news, request)
    except Exception as exc:  # noqa: BLE001
        ui.warn(f"Alpaca news unavailable ({exc})")
        return []
    return list(response.data.get("news", []))


def _fetch_rss() -> list[tuple[datetime, str, str, str]]:
    try:
        import feedparser  # noqa: PLC0415
    except ImportError:
        ui.warn("feedparser is not installed -- skipping RSS feeds.")
        return []

    cutoff = _since()
    items: list[tuple[datetime, str, str, str]] = []
    for source, url in config.RSS_FEEDS:
        try:
            parsed = feedparser.parse(url)
        except Exception:  # noqa: BLE001 -- one dead feed should not sink the command
            continue
        for entry in parsed.entries:
            stamp = entry.get("published_parsed") or entry.get("updated_parsed")
            if not stamp:
                continue
            when = datetime(*stamp[:6], tzinfo=timezone.utc)
            if when < cutoff:
                continue
            items.append((when, source, entry.get("title", "(untitled)"), entry.get("link", "")))
    return items


def _print(rows: list[tuple[datetime, str, str, str]], title: str) -> None:
    if not rows:
        ui.info(f"No news in the last {config.NEWS_LOOKBACK_HOURS} hours.")
        return

    table = ui.table(
        title,
        [("When", "left"), ("Source", "left"), ("Headline", "left")],
    )
    for when, source, headline, link in sorted(rows, key=lambda r: r[0], reverse=True):
        text = ui.Text(headline[:110])
        if link:
            text.stylize(f"link {link}")
        table.add_row(when.astimezone().strftime("%m-%d %H:%M"), source[:18], text)

    ui.console.print()
    ui.console.print(table)


def run(args) -> int:
    target = (args.target or "").strip()

    if target.lower() == "all":
        rows = [
            (a.created_at, a.source or "Alpaca", a.headline, a.url or "")
            for a in _fetch_alpaca(None)
        ]
        rows += _fetch_rss()
        _print(rows, f"Market news, last {config.NEWS_LOOKBACK_HOURS}h")
        return 0

    if target:
        symbols = [target.upper()]
    else:
        try:
            scan_rows = results.read()
        except FileNotFoundError:
            ui.error(
                f"No scan results at {config.TOP_LONG_CSV}. "
                "Run `atlas scan`, or pass a symbol / `all`."
            )
            return 1
        # Alpaca's news query takes a symbol list; keep it to the top slice so
        # the results stay readable rather than returning a wall of headlines.
        symbols = [row["symbol"] for row in scan_rows[:10]]
        ui.info(f"News for the top {len(symbols)} scanned: {', '.join(symbols)}")

    articles = _fetch_alpaca(symbols)
    rows = [
        (a.created_at, ",".join(a.symbols[:3]) or (a.source or "Alpaca"), a.headline, a.url or "")
        for a in articles
    ]
    _print(rows, f"News for {', '.join(symbols)}, last {config.NEWS_LOOKBACK_HOURS}h")
    return 0
