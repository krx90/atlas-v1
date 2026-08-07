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
--horizon N          bars to forecast (default 5; days unless --timeframe)
--timeframe TF       5Min | 15Min | 30Min | 1Hour -- intraday bars, not daily
--rebuild-universe   force a full asset sweep and history reseed
--no-cache           slower uncached decoder, 512-bar lookback (reference path)
```

### Intraday scans

`--timeframe` scores on intraday bars instead of daily, which is the data Kronos
was actually pretrained on. History is fetched into a separate intraday cache,
sized automatically from the lookback, and `--horizon` becomes a count of *bars*
rather than days.

**Extended-hours bars are filtered out, and the scan verifies it before
forecasting.** Alpaca returns pre/post-market bars by default and they are ~59%
of intraday rows — thin, wide-spread, and nothing like session bars. Left in,
most of the model's context would be noise and every number below would be
measuring the wrong thing. The scan prints what it saw:

```
session filter OK: 6.0 bars/day against 6 expected for 1Hour
```

Expected regular-session bars per day, measured against live data:

| timeframe | bars/day | unfiltered |
|---|---|---|
| 5Min | 78 | 191 |
| 15Min | 26 | 64 |
| 30Min | 13 | 32 |
| 1Hour | **6** | 16 |

**1Hour is 6, not the 7 that 09:30–16:00 suggests.** Alpaca aligns hourly bars
to the clock hour, so a session returns 09:00, 10:00 … 15:00; the 09:00 bar is
half premarket and is dropped, which loses the 09:30–10:00 window entirely. That
is a real cost of the timeframe and a reason to prefer 30Min, which lands
exactly on 09:30.

### Forecast health

Every scan ends with three diagnostics that say whether the model is coping with
the input it was given:

```
Diagnostic       Measured  Healthy  Broken
rejection rate         7%     ~18%    ~68%
median |mu|/vol      0.76    ~1.06   ~4.80
sigma/realized       1.42    ~1.00   ~0.35
```

| | |
|---|---|
| **rejection rate** | share of symbols the plausibility guards threw out |
| **median \|mu\|/vol** | forecast mean against the symbol's own realized volatility |
| **sigma/realized** | forecast *dispersion* against realized volatility |

The reference columns are measured points: healthy is Kronos-small on daily
bars, broken is Kronos-base on the same sample. See
[implementation.md](implementation.md) §6.

**Read them as a pattern, not individually.** All three drifting bad together
means the model is genuinely struggling. One number off on a small sample is
probably noise — a correlation of −0.73 on 17 symbols once collapsed to −0.09 on
113. Below 50 symbols the verdict says so.

**`sigma/realized` well below 1 is the one to weight most heavily.** It is the
confidently-wrong signature: the model states far less uncertainty than the
symbol actually has, which is dangerous precisely because such forecasts look
clean enough to trade.

These are measured over **every** symbol the model returned, rejected ones
included. That is deliberate — `|mu|/vol` is exactly what the 3× guard rejects
on, so a median over surviving symbols alone is censored at 3.0 and could never
show the broken value at all. Per-symbol detail goes to
`data/scan_health_<timeframe>_<horizon>bar_<timestamp>.csv`.

To compare a timeframe against a same-day daily baseline on the same symbols:

```
python tests/run_diagnostic_scan.py --symbols 30 --timeframe 1Hour
```



## Backtest

Walk-forward test of whether the ranking actually predicts returns.

```
atlas backtest                    20 as-of dates, 100 symbols, long ranking
atlas backtest short              test the short ranking instead
atlas backtest --dates 40         more dates = more statistical power
atlas backtest --symbols 200      more symbols = less noisy per-date IC
atlas backtest --horizon 10       test a different forward window
atlas backtest --timeframe 30Min  intraday bars instead of daily
```

`--timeframe` accepts `5Min`, `15Min`, `30Min` or `1Hour` and reads the intraday
cache, which `atlas scan --timeframe` fills. With it, `--horizon` and `--spacing`
count *bars* rather than sessions. Results are written to
`data/backtest_<side>_<timeframe>_<horizon>.csv`, so an intraday run never
overwrites the daily one.

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

## Accuracy

Measure whether the forecast was *right*, as distinct from usefully ordered.
Reads a CSV `atlas backtest` already wrote, so it needs no model, no GPU, no
credentials and no network — it returns in about a second.

```
atlas accuracy backtest_long_5d.csv         a saved daily run
atlas accuracy data/backtest_long_5Min_6.csv   a path works too
atlas accuracy backtest_long_5d.csv --permutations 5000
```

A bare filename is looked up in `data/`. The backtest prints these same tables at
the end of its own run; this command exists so you can re-derive them without
spending eighty minutes again.

Rank IC answers one narrow question — does the score sort symbols in the order
their returns arrive? A model can do that and still be useless, so four more
questions are asked here:

| | |
|---|---|
| **directional hit rate** | does the sign of `mu` match what happened? With a binomial p-value against a coin flip |
| **skill score** | `mu`'s squared error against forecasting *no move*. `<= 0` means the zero forecast was at least as accurate |
| **permutation p** | the mean IC recomputed on scores shuffled within each date, 1000 times. Makes no normality assumption, unlike the t-statistic |
| **momentum IC** | the same rank test on the trailing return. A subtraction, versus 2.5 seconds a symbol-date |
| **by signal** | mean forward return of everything labelled BUY, HOLD, AVOID or SHORT — the thing you actually trade |

Plus tail calibration: a well-formed forecast puts 5% of outcomes below its own
`q05` and 5% above `q95`. Materially more means the distribution is too narrow —
confident and wrong, the failure worth catching.

Two baselines exist because a bare number cannot be read. An IC of +0.026 means
nothing until you know what luck produces on the same sample.

### The pre-registered bar

Every verdict is judged against four conditions fixed in `config.py` on
2026-08-07 — **before** the run they would judge — and printed beside the numbers
they are applied to:

| Condition | Required |
|---|---|
| direction | `p < 0.01` and hit rate > 50% |
| magnitude | skill score > 0 |
| ranking | permutation `p < 0.05` and IC > 0 |
| vs momentum | mean IC > momentum IC |

The direction threshold is deliberately stricter than the conventional 0.05:
every bias in the backtest pushes results upward, so a marginal pass is more
likely bias than skill. **Beating zero is not the bar; beating momentum is** — a
tie with trailing return counts as a failure, because momentum costs a
subtraction and the model costs ~2.5 seconds a symbol-date. An unmeasured
momentum baseline (`-`) cannot be beaten, so it does not block a pass.

The ordering is the point. Three runs had already returned null before the bar
was set; with the threshold chosen afterwards, a fourth run is not evidence, it
is re-rolling. Loosening any of these is a legitimate decision, but it should be
a visible commit with an argument attached, not a quiet edit after seeing a
result. `ACCURACY_FIXED_ON` is printed with every verdict so the claim is
checkable against the commit date.

### Current results

Neither saved run clears it:

| | daily, 5d | 5-minute, 6-bar |
|---|---|---|
| directional hit rate | 48.7% (p=0.33) | 52.9% (p=0.017) |
| skill score | −1.10 | −0.18 |
| mean rank IC | −0.0076 | +0.0258 |
| permutation p | 0.74 | 0.29 |
| **conditions met** | 1 of 4 | 1 of 4 |

The intraday direction result is the one the bar changes: 52.9% at `p = 0.017`
clears a conventional 0.05 and fails the 0.01 committed to here. That is the
threshold doing its job rather than a defect.

CSVs written before this command existed lack the `prior_return` column, so the
momentum baseline shows as `-` until you re-run the backtest.

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

## Review

Flag open positions worth closing, worst first.

```
atlas review                  score every holding
atlas review --no-forecast    mechanical signals only
```

**Reports only.** Closing stays a deliberate `atlas close XYZ`.

**Thresholds scale with each symbol's own volatility, not fixed percentages.**
Across one real portfolio daily volatility ran 0.90% to 3.60% — a 4x spread — so
a fixed -8% stop meant -8.9 sigma for the quietest holding and -2.3 for the
noisiest. Instead:

```
expected_move = daily_vol x sqrt(days_held)
sigma         = unrealised P&L% / expected_move
```

The `sqrt(days_held)` term stops a long-held position flagging merely for having
had time to drift — volatility compounds with the square root of time.

Five signals, each contributing a severity; the score is their sum, so failing
several outranks failing one badly:

| signal | fires when |
|---|---|
| **stop** | losing more than 2 sigma |
| **target** | winning more than 3 sigma |
| **liquidity** | no longer passes the liquidity screen |
| **borrow** | a short is no longer easy-to-borrow — it can be recalled and closed *for* you |
| **forecast\*** | the model now expects the position to move against you |

Thresholds are asymmetric on purpose: cut losses sooner than gains are banked.
All are tunable in `config.py` (`REVIEW_STOP_SIGMA`, `REVIEW_TARGET_SIGMA`, …).

**\* The forecast signal has no demonstrated skill.** Two backtests returned
IC -0.008 and +0.026, neither distinguishable from zero. It is capped so it
cannot dominate, marked wherever it appears, and `--no-forecast` removes it — the
other four signals are mechanical and do not depend on the model working.

```
Symbol  Side    P&L %  Sigma   Held  Score  Flags
TLN     LONG   -8.36%  -2.32  today   0.32  down -2.3s, past the 2.0s stop
KR      LONG   +2.63%  +1.49  today   0.00  -

  1 of 10 positions flagged. To act:
    atlas close TLN
```

Volume deliberately does **not** move the thresholds. It tells you what exiting
will cost, not whether to exit, so it appears as its own `costly to exit` flag
when a position is large relative to the symbol's median daily dollar volume.

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
