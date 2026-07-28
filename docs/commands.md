# Atlas CLI Commands

All commands run inside the `atlas` conda environment (`conda activate atlas`).
See [implementation.md](implementation.md) for setup and design notes.

## Scan

Score the universe for both long and short candidates.

```
atlas scan                 writes both files, prints both tables
atlas scan long            writes both files, prints the long table
atlas scan short           writes both files, prints the short table
```

Both sides come from the same 25 sampled paths, so scoring both costs nothing
beyond the single forward pass. `top30_long.csv` and `top30_short.csv` are
therefore **always both written**; the argument only narrows what is printed.
Short candidates are additionally filtered to symbols Alpaca reports as
`shortable` and `easy_to_borrow` — a name you cannot borrow is not a candidate,
however good the forecast looks.

The first run builds the investable universe (a full Alpaca asset sweep) and
fetches 5 years of daily bars for ~600 symbols. Every run after that is a delta
fetch of a few days. The universe is rebuilt automatically every 30 days.

Each symbol is forecast 5 trading days ahead with the Kronos candlestick model,
25 sampled paths per symbol, and ranked on expected return penalised by the
tail that would hurt that trade, per unit of forecast dispersion. Both files are
fully overwritten on every run; the complete scored universe, carrying both
sides' statistics, is archived to `data/scan_full_<timestamp>.csv`. A full scan
takes roughly 25 minutes.

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



## Backtest

Walk-forward test of whether the ranking actually predicts returns.

```
atlas backtest                    20 as-of dates, 100 symbols, long ranking
atlas backtest short              test the short ranking instead
atlas backtest --dates 40         more dates = more statistical power
atlas backtest --symbols 200      more symbols = less noisy per-date IC
atlas backtest --horizon 10       test a different forward window
```

At each as-of date the model sees **only** the bars up to that day, produces a
ranking, and is scored against the following `--horizon` sessions. Default
`--spacing 5` with a 5-day horizon makes the forward windows non-overlapping,
which is what keeps the per-date results independent and the t-statistic honest.

Three numbers matter:

| | |
|---|---|
| **mean rank IC** | correlation between score and realized forward return |
| **IC t-statistic** | `\|t\| < 2` means indistinguishable from luck |
| **spread** | top decile minus bottom decile, before costs |

Every observation is written to `data/backtest_<side>_<horizon>d.csv` so you can
re-analyse without re-running the model.

A run takes roughly 2.5 seconds per symbol-date — the default is about 80
minutes. It writes `data/progress.<pid>.json` once a second, so you can watch it
from another terminal:

```
bash scripts/watch-progress.sh
```

**Three biases inflate the result, and none are correctable here.** The universe
is today's liquid names, so symbols that have since delisted are missing and
those skew to losers (survivorship). Transaction costs are not modelled, and at a
5-day horizon spread and slippage are a material fraction of any edge. And it
covers one market regime. Treat a positive result as weak evidence; a null result
is the more trustworthy outcome.

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

Open one position, long or short, by dollar amount. An order summary and a
`[y/n]` confirmation come before every order.

```
atlas buy XYZ 500 l       open a $500 long in XYZ
atlas buy XYZ 500 s       open a $500 short in XYZ
atlas buy XYZ l           $100 is the default amount
atlas buy XYZ 500         asks "Long or short? [l/s]:"
```

One symbol at a time. There is no bulk form — buying a list in one command makes
it easy to take positions you never individually looked at.

**The side is required.** If you omit it Atlas asks rather than assuming, because
defaulting to long would let a mistyped command silently open the opposite of
the position you intended. A piped invocation aborts instead of choosing.

Amount and side can be given in either order; they are recognised by shape.

**Order summary** — shown before every order:
```
ORDER SUMMARY:
  BUY $500.00 of RING  (11.05 shares)
  at $45.23/share

Confirm? [y/n]:
```

**Non-fractionable stocks** — Alpaca only allows whole-share orders for some
assets. Atlas detects this upfront and rounds *down* to whole shares, so the
order never costs more than you asked:
```
ORDER SUMMARY:
  ! Non-fractionable!
  BUY ~$987.45 of LMTL  (47 shares)
  at $21.01/share

Confirm? [y/n]:
```

**Shorts** are always whole shares — Alpaca has no fractional shorts — and are
refused unless the asset is both `shortable` and `easy_to_borrow`:
```
ORDER SUMMARY:
  ! Non-fractionable!
  SELL SHORT ~$487.20 of XYZ  (12 shares)
  at $40.60/share
  ! Losses on a short are unbounded.

Confirm? [y/n]:
```

Add `--dry-run` to any buy to see the summary without submitting anything.

## Close

Close an entire position, in either direction. Requires a `[y/n]` confirmation.

```
atlas close XYZ           close the whole position in XYZ
```

Closing a long sells; closing a short buys to cover. Atlas reads the direction
from the position and says which is about to happen — they are opposite trades:

```
ORDER SUMMARY:
  BUY TO COVER entire short position in XYZ  (12 shares)
  at $40.60/share  ->  ~$487.20
  P&L: +$25.00 (+5.00%)

Confirm? [y/n]:
```

Closing is by symbol only. Ranking positions and closing the worst few is easy
to fire by accident and hard to undo.

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
