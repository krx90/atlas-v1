# Atlas CLI Commands

All commands run inside the `atlas` conda environment (`conda activate atlas`).
See [implementation.md](implementation.md) for setup and design notes.

## Scan

Run the scanner and write the top assets to `top30_assets.csv`.

```
atlas scan                 full universe, 25 Monte Carlo paths per symbol
```

The first run builds the investable universe (a full Alpaca asset sweep) and
fetches 5 years of daily bars for ~600 symbols. Every run after that is a delta
fetch of a few days. The universe is rebuilt automatically every 30 days.

Each symbol is forecast 5 trading days ahead with the Kronos candlestick model,
25 sampled paths per symbol, and ranked on expected return penalised by the
downside tail, per unit of forecast dispersion. `top30_assets.csv` is fully
overwritten on every run; the complete scored universe is archived to
`data/scan_full_<timestamp>.csv`. A full scan takes roughly 25 minutes.

**Read the ranking, not the expected return.** The model's absolute `mu_pct` is
not calibrated — it carries a systematic downward tilt, and `signal` is the
weakest column in the output. Relative order is what the scan is for. Symbols
whose forecasts fail the plausibility checks are excluded and listed with a
reason in `data/scan_skipped.csv`, so check there before concluding a symbol was
never considered. See [implementation.md](implementation.md) §6.

Flags:

```
--limit N            only scan the N most liquid symbols
--paths N            Monte Carlo paths per symbol (default 25)
--horizon N          trading days to forecast (default 5)
--rebuild-universe   force a full asset sweep and history reseed
```



## Portfolio

Show open positions and account balance from the paper account.

```
atlas portfolio
```

## Orders

Show all currently open orders on the paper account.

```
atlas orders
```

## Buy

Place paper-account market orders by dollar amount. Before every order an order
summary is shown and a `[y/n]` confirmation is required.

```
atlas buy XYZ             buy $100 of symbol XYZ
atlas buy XYZ 500         buy $500 of symbol XYZ

atlas buy 5               buy top 5 from the latest scan, $100 each
atlas buy 5 500           buy top 5 from the latest scan, $500 each
```

**Order summary** — shown before every order:
```
ORDER SUMMARY:
  BUY $500.00 of RING  (11.05 shares)
  at $45.23/share

Confirm? [y/n]:
```

**Non-fractionable stocks** — Alpaca only allows whole-share orders for some
assets. Atlas detects this upfront and rounds the order to the nearest whole share:
```
ORDER SUMMARY:
  ! Non-fractionable!
  BUY ~$987.45 of LMTL  (47 shares)
  at $21.01/share

Confirm? [y/n]:
```

Scanner buys read the existing `top30_assets.csv`. If the file is older than 24
hours a warning is printed — run a scanner first to refresh it.

Add `--dry-run` to any buy to see the order summary without submitting anything.

## Sell

Sell the entire position in a symbol. Requires a `[y/n]` confirmation before
executing.

```
atlas sell XYZ            sell the entire position in symbol XYZ
```

Selling is by symbol only — there is no `atlas sell N`. Closing the bottom X
positions in one command is easy to fire by accident and hard to undo.

```
ORDER SUMMARY:
  SELL entire position in RING  (11.05 shares)
  at $45.23/share  ->  ~$499.79
  P&L: +$12.34 (+2.53%)

Confirm? [y/n]:
```

## Info

Show company information for a symbol: sector, market cap, P/E, 52-week range,
employees, website, and a short description.

```
atlas info XYZ
```

Requires `yfinance`

## View

Open an interactive candlestick chart for an instrument in the browser. Shows
candlesticks, MA20, MA50, volume bars, and a hover tooltip with OHLC data.
Always opens in a new window.

```
atlas view XYZ            open chart for XYZ (default: 6 months)
atlas view XYZ 6m         custom period: 1m 3m 6m 1y 2y 3y 5y or number of days
```

## News

Fetch and display recent financial news (last 48 hours).

```
atlas news                show news for all assets in the current scan results
atlas news XYZ            show news for a specific symbol
atlas news all            show all general market news (Alpaca + RSS)
```
