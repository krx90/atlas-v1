# Atlas — full session log, 2026-07-26/27

A complete record of building Atlas: what was made, every measurement taken, every
bug found, every decision and its reasoning, and what remains open. Written to be
read standalone — you should be able to ask questions about any part of it without
the code in front of you.

**Bottom line up front:** the tool is built, verified and documented (198 tests).
The *signal it ranks on has not been shown to work*. Two backtests both came back
statistically null, and both were too small to have detected a realistic edge. The
decisive experiment is identified, sized, and affordable on the user's other machine.

---

## 1. What Atlas is

A CLI that scans US equities, ranks them with the [Kronos](https://github.com/shiyu-coder/Kronos)
candlestick foundation model, and manages an Alpaca **paper** account.

```
atlas scan [long|short]        score the universe, write both CSVs
atlas backtest [long|short]    walk-forward test of whether the ranking predicts
atlas portfolio                account + open positions
atlas orders                   open orders
atlas buy XYZ [AMT] <l|s>      open one position, side required
atlas close XYZ                close a position, either direction
atlas info XYZ                 company fundamentals (yfinance)
atlas view XYZ [period]        plotly candlestick chart
atlas news [XYZ|all]           Alpaca news + RSS
```

**Stack:** conda env `atlas`, Python 3.12, torch 2.13 on MPS (Apple Silicon),
alpaca-py, SQLite cache, rich terminal output.

**Scale:** 27 tracked files, 198 tests, 717k daily bars + 285k intraday bars
across a 600-symbol universe.

---

## 2. Kronos: what it is and how it had to be used

A two-stage model: a tokenizer quantises OHLCV bars into discrete tokens, and a
decoder-only Transformer predicts the next token autoregressively. It is
**generative** — it samples plausible futures rather than emitting a point
forecast, which is what makes a distribution (and therefore risk statistics)
possible.

### Available checkpoints

| model | params | context | notes |
|---|---|---|---|
| Kronos-mini | 4.1M | 2048 | pairs with Tokenizer-2k |
| **Kronos-small** | **24.7M** | **512** | the default here |
| Kronos-base | 102.3M | 512 | measured worse, see §6 |
| Kronos-large | 499M | — | **not published** — HF returns 401 |

### Weights come from HuggingFace; code does not

This trips people up. `Kronos.from_pretrained("NeoQuasar/Kronos-small")` downloads
weights automatically. But the HF repos contain **only** `config.json` and
`model.safetensors` — no modeling file, no `auto_map`, so `trust_remote_code`
has nothing to load. Kronos subclasses `PyTorchModelHubMixin`, not a
`transformers` auto-class, so the class definition must already exist locally.

It cannot come from pip either: the repo has no `setup.py`/`pyproject.toml`, so
`pip install git+…` fails. **The PyPI package named `kronos` is an unrelated
Django cron library — do not install it.**

Resolution: shallow clone into `vendor/kronos` (gitignored), added to `sys.path`.
Only 3 files are used — `model/{__init__,kronos,module}.py`, ~30 KB of the repo's
16 MB.

### `sample_count` averages — the trap that would have broken everything

`KronosPredictor.predict(sample_count=25)` generates 25 samples and **averages
them internally**, returning one mean path (`auto_regressive_inference`,
`kronos.py:389`). Every symbol would have come back with `sigma = 0` and the
entire risk-based scoring would have been meaningless.

Fix: replicate each symbol's history 25× through `predict_batch` at
`sample_count=1`, which yields 25 independently sampled futures. Verified: 25 of
25 terminal closes distinct.

---

## 3. Architecture

```
Alpaca /v2/assets   -> universe.py  -> ~600 liquid symbols -> SQLite `universe`
Alpaca /v2/bars     -> bars.py      -> 5y daily OHLCV      -> SQLite `bars`
                                            |
                                            v
                       forecast.py  -> KronosEngine (loaded ONCE)
                                            |  25 paths x horizon per symbol
                                            v
                       scoring.py   -> mu, sigma, p_up/p_down, q05/q95, mdd/runup
                                            |
                                            v
                       results.py   -> top30_long.csv + top30_short.csv
                                       data/scan_full_<ts>.csv (archive)
```

| module | role |
|---|---|
| `cli.py` | argparse dispatch, credential preflight, lazy imports |
| `config.py` | paths, every tunable, paper-only guard |
| `alpaca_client.py` | cached clients, rate limiting, retries, feed probes |
| `db.py` | SQLite schema + idempotent migrations |
| `bars.py` | daily + intraday caches |
| `universe.py` | asset sweep, liquidity screen, borrowability |
| `forecast.py` | Kronos load-once + batched Monte Carlo sampling |
| `scoring.py` | paths -> statistics -> score, long or short |
| `backtest.py` | walk-forward as-of slicing, forward returns, rank IC |
| `results.py` | atomic CSV overwrite, archives |
| `ui.py` | rich tables, order summaries, confirmations |

---

## 4. Market data (Alpaca)

### Feed entitlements — historical and real-time are separate

The Basic (free) plan serves **full historical SIP** data provided the query's
`end` is ≥15 minutes old, but refuses SIP for anything real-time. So this account
is legitimately **SIP for bars and IEX for quotes**.

This cost a bug: the feed was probed with a bars request and the answer reused for
`get_stock_latest_trade`, producing `subscription does not permit querying recent
SIP data` on every buy. Now `realtime_feed()` probes the latest-trade endpoint
separately. Alpaca returns this error *without a reliable status code*, so the
check matches message text as well as 401/403.

`data_end()` clamps every request to `now − 16 min`.

### Universe construction (3 stages, so 5y is never pulled for 11k symbols)

1. `get_all_assets(ACTIVE, US_EQUITY)`, keep `tradable` on NYSE/NASDAQ/ARCA/AMEX/BATS,
   drop >5 chars, `.`/`/`/`+`/space, and warrant/unit/right/preferred suffixes.
   → ~12,457 symbols.
2. Fetch ~30 recent daily bars for all of them (~30 requests), compute median
   dollar volume. → 4,003 pass (≥$5M median $ volume, ≥$5 price).
3. Keep the top **600** by dollar volume.

Cached with `built_at`; rebuilt after 30 days. 578/600 are `shortable AND
easy_to_borrow`.

### Intraday

All timeframes work on the same account, back to 2016.

**Extended-hours bars are 59% of intraday rows** — Alpaca returns 191/session for
5-min instead of the 78 regular-session bars. Nothing warns you; the count just
looks generous. Unfiltered, most of the model's 512-bar context would be thin
pre/post-market noise. `regular_session()` filters to 09:30–15:55 ET.

---

## 5. Scoring

```
r_i     = path_i.close[-1] / p0 - 1      per-path horizon return
mu      = mean(r)      sigma = std(r)
p_up    = frac(r > 0)  p_down = frac(r < 0)
q05/q95 = 5th / 95th percentile of r
mdd     = mean min(low)/p0 - 1           a long's worst moment
runup   = mean max(high)/p0 - 1          a short's worst moment

long  score = ( mu - LAMBDA*max(0,-q05)) / max(sigma, 1e-4)
short score = (-mu - LAMBDA*max(0, q95)) / max(sigma, 1e-4)
```

`LAMBDA = 0.5`. Signals: BUY/SHORT when edge ≥0.5% and favourable probability
≥60%; AVOID when edge ≤0; else HOLD.

### Guards against unusable model output

Kronos normalises each window by its own mean/std and denormalises the same way.
On a **daily** series spanning years, that std is dominated by *trend*, not
volatility — so sampling noise denormalises into enormous moves. Two failure modes,
both observed:

**Non-positive prices.** One symbol produced a mean "return" of **−388%** — a
negative predicted price. Measured incidence 27/2,825 paths (0.96%), concentrated
in leveraged ETFs (SOXS alone 22). Paths with non-positive or non-finite prices
are dropped; a symbol needs 60% of its paths valid (a *fraction*, not a count — a
fixed count would reject every symbol under `--paths 5`).

**Implausible means.** Filtering wasn't enough: at a 512-bar lookback **22 of 113
symbols still produced |mu| > 15%** over five days — SOXL −65%, DELL −62%, AMD
−61%, WDC −55%, AMAT −37%, KLAC −32%, LRCX −28%. Almost entirely semiconductors
and hardware: the strongest trenders, exactly as the mechanism predicts.

So a forecast whose |mu| exceeds **3×** the symbol's own realized volatility is
rejected. Calibration among passing symbols: |mu|/vol p50 0.73, p90 2.49, p99 3.73.
An initial 4× barely bit (a −62.45% forecast survived). Two refinements: a
volatility floor (0.2%) because 3× of almost nothing rejects everything — one real
case was `-0.4% against realized 0.00% volatility` — and an absolute 1% allowance.
At a 2% allowance a 0.31%-volatility bond ETF passed at a 6.5× ratio.

Rejections are **recorded**, not silently dropped: `data/scan_skipped.csv` plus a
reconciliation line. Every eligible symbol ends as scored, rejected or failed, and
the command exits non-zero if those don't sum.

### A ruled-out hypothesis

An initial 17-symbol sample showed `corr(trend, mu) = −0.73`, suggesting a clean
linear anti-momentum bias correctable by cross-sectional neutralization. **On 113
symbols that collapsed to −0.09, R² = 0.01.** The original was small-sample noise;
neutralizing would have been fitting to nothing. The effect is real but lives in
the tail, which is why a tail guard is the right instrument.

### The systematic tilt

Across 531 scored symbols: median `mu` **−0.58%**, 56% negative. On a 104-symbol
sample it was −1.99%/67%. Mild systematic downward tilt from the same mechanism
acting on everything.

Consequence: **use the ranking, not the point estimate.** Dividing by `sigma`
makes the score robust to a common additive tilt; `mu_pct` and the `signal` column
are not. Both documented as such.

### Long vs short are the same opinion

**Spearman(long score, short score) = −0.9946**; top-10 shorts overlap bottom-10
longs **10/10**. Tail asymmetry reorders 16 of 29 symbols by one or two places and
never changes which names you'd act on.

Inherent: both are functions of the same `mu`/`sigma`, differing only by
`λ(q05−q95)/σ`. The short table is a *presentation* of the same ranking, not a
second opinion. It still earns its place — borrowable-only filtering, correct risk
columns, no mental inversion.

**Not modelled:** a long's worst case is capped at −100%, a short's is unbounded.
A larger `LAMBDA` for shorts would reflect that. Both sides currently use 0.5.

---

## 6. Measurements

### Model loading and batching

| quantity | measured |
|---|---|
| model load | 8.4 s cold, ~1 s warm, **once** per run |
| throughput | 2.48 s/symbol (25 paths, 5-day horizon) |
| full universe | ~25 min for 600 symbols |

**Bigger batches are slower on MPS** — Kronos has no KV cache and recomputes
attention over the full context every step, so a larger batch only multiplies
memory traffic:

```
symbols/pass:      1      2      5     10     20
sec/symbol:     2.48   2.53   2.70   2.72   4.96
```

Default moved from 5 to **1 symbol per pass** (~9% faster), which also makes the
RNG seed exactly per-symbol, removing a documented reproducibility caveat.

### Kronos-base is worse, not better

40 identical symbols, 25 paths, 5-day horizon:

| | small (24.7M) | base (102.3M) |
|---|---|---|
| scored | 33/40 | **13/40** |
| rejected | 18% | **68%** |
| median raw `mu` | −3.34% | **−27.25%** |
| median \|mu\|/vol | 1.06 | **4.80** |
| sec/symbol | 2.43 | 9.67 |
| 600 symbols | 24m | 97m |

Four times slower *and* two-thirds of its output unusable. Rank agreement where
both survive is ρ = +0.62 — it agrees when it works, it just fails far more often.

**Temperature cannot rescue it.** Across T = 1.0 → 0.3 the median forecast moved
only −15.01% → −14.24% while dispersion collapsed 0.75 → 0.35× realized. At T=0.3
base is *confidently* predicting a 14% five-day drop. The bias is in the learned
weights, not the sampling.

Likely cause: base ships `attn_dropout_p: 0.0`, `token_dropout_p: 0.0` versus
small's `0.1`/`0.1`. Sharper, less regularised, so on out-of-distribution daily
bars it is more *confidently* wrong. Scale amplifies the pathology.

### Temperature on `small` (unresolved, promising)

| T | median mu | sigma/realized |
|---|---|---|
| **1.0 (current)** | **−1.67%** | 0.85 |
| 0.7 | −0.52% | 0.77 |
| 0.5 | −0.10% | 0.65 |
| 0.3 | −0.84% | 0.31 |

Lowering T largely removes the tilt at modest dispersion cost. **Measured on only
15 symbols — never verified at scale, never adopted.** Open item.

### Lookback (unresolved)

| lookback | \|mu\|>15% rate | sigma/realized |
|---|---|---|
| 128 | 0.9% | 0.59 (under-dispersed) |
| 512 (current) | 19.5% | 1.02 (well calibrated) |

A real trade-off, never resolved. Left at 512 (Kronos-small's limit).

---

## 7. The backtests — the central result

### Method

Walk-forward rank test. At each as-of point the model sees **only** bars up to and
including it, produces a ranking, and is judged on the next `horizon` bars.
Spacing ≥ horizon keeps forward windows non-overlapping, which is what makes the
per-date ICs independent and the t-statistic honest.

- **rank IC** — Spearman(score, realized forward return) per date. The headline.
- **t-statistic** — |t| < 2 means indistinguishable from luck.
- **spread** — top decile minus bottom decile, before costs.

Lookahead was the failure mode guarded hardest: 17 tests, including hand-checked
cases proving a crash the day after the as-of point cannot leak backwards.

### Results

| run | n | mean IC | t | 95% CI | spread | spread t |
|---|---|---|---|---|---|---|
| daily, 5-day horizon | 20 | −0.0076 | −0.14 | [−0.121, +0.106] | +0.688% | +0.62 |
| 5Min, 30-min horizon | 60 | +0.0258 | +0.72 | [−0.045, +0.097] | +0.043% | +0.41 |

**Both null.** The intraday sign flipped positive and is 3× larger, but the
difference between the two runs is **t = 0.52, p = 0.61** — nothing.

Daily detail: the apparent +0.69% spread is **two lucky dates** — drop the best two
of twenty and it goes **negative (−0.37%)**. The top decile returned +0.93% while
the average name returned **+1.38%**, so the model's favourites *lagged the market*
— though at p = 0.68 that is not distinguishable from random either.

Important precision: this is **"indistinguishable from random", not "worse than
random"**. A reliably-inverted signal could be traded backwards; a null cannot.

Intraday detail: the late positive streak (07-07 +0.69, 07-15 +0.68) is noise —
first half +0.036 vs second half +0.015, p = 0.77. And the +0.043% spread over 30
minutes would be consumed by a 0.02–0.10% round-trip bid-ask before reaching you.

### Both tests were underpowered

`sd(IC) ≈ 0.276`. Requirements to reach t = 2:

```
mean IC +0.02  ->  ~759 as-of points   (17 h at current rate)
mean IC +0.03  ->  ~337 as-of points   (7.5 h)
mean IC +0.05  ->  ~121 as-of points   (2.7 h)
```

A typical good equity signal is IC 0.02–0.05. **Neither run could have detected
one.** 20 and 60 points against a requirement of several hundred.

### Known biases, all inflating rather than deflating

- **Survivorship** — the universe is today's liquid names; symbols that have since
  delisted are absent and skew to losers. Alpaca's asset list is current-state only.
- **No transaction costs.**
- **One market regime** — a few months of 2026.
- **No execution lag** — the backtest measures from the as-of bar, implicitly
  assuming you can trade at its close. With the 16-min data delay you cannot. Not
  lookahead in the backtest's own terms, but it overstates tradability by 3–4
  intraday bars. A `lag` parameter was designed but not implemented.

---

## 8. Actionability — the timeframe economics

Forecast cost scales with **horizon in bars**, not wall-clock time. So coarser
bars buy the same wall-clock horizon far more cheaply.

```
512-bar context, and cost to forecast ONE TRADING DAY ahead (600 symbols)
   bars      context   steps/day   s/symbol   600 symbols
   5Min   7 sessions          78      41.0s        410 min
  15Min  20 sessions          26      13.7s        136 min
  30Min  39 sessions          13       6.8s         68 min
  1Hour  73 sessions           7       3.7s         37 min
   1Day     2.0 years           1       0.5s          5 min
```

Alpaca Basic's **16-minute data delay** is the binding constraint at short
horizons:

```
30-min horizon, 600 symbols: 16m delay + 31.5m scan = 47.5m of 30m -> CLOSED
30-min horizon,  30 symbols: 16m delay +  1.6m scan = 17.6m of 30m -> usable
 1-hour horizon (30Min bars): delay is 27% of the window
  1-day horizon (30Min bars): delay is  4% of the window, scan 84m
```

**30-minute bars are the sweet spot**: ~2 months of context (short enough to avoid
the trend-domination pathology, long enough for real structure), 6× cheaper than
5-min for the same wall-clock horizon, ~10M rows for 600 symbols × 5 years versus
59M for 5-min, and a horizon where the data delay is a rounding error.

Unverified assumption: how much 30-min data was in Kronos's pretraining mix. Its
examples are 5-min.

---

## 9. Bugs found, and what they have in common

1. **Rich ate `[y/n]`.** The confirmation rendered as `Confirm? : ` because rich
   parses square brackets as style tags. The one string `commands.md` specifies
   verbatim shipped broken. Found by the user placing a real order — every
   automated path skipped it (`--dry-run` returns before the prompt, non-TTY
   declines before it). Fixed with `markup=False`.
2. **Feed entitlement** — historical vs real-time SIP are separate; reusing one
   probe broke every buy.
3. **`MIN_VALID_PATHS` as an absolute count** would have rejected every symbol
   under `--paths 5` — the exact flag the smoke test used. Now a fraction.
4. **`.gitignore` stale after the rename** — `top30_assets.csv` no longer matched,
   so the renamed scan output was about to be committed as source. Found because
   the user asked where files were being put.
5. **Progress watcher, four defects**: no initial heartbeat (file absent for 4
   minutes, reported "no run in progress"); one shared filename (concurrent runs
   clobbered each other, first to finish deleted it and announced "done"
   mid-run); fixed bar width (line exceeded terminal width, wrapped, so `\r` could
   not overwrite and every refresh printed a new bar); whitespace-delimited
   parsing (intraday `last_date` contains a space, shifting every later field so
   the liveness check failed on a healthy run).
6. **`pkill -f "backtest --dates 2"` matched `--dates 20`** and killed the long
   run — 8 minutes lost. Should have used the recorded PID.
7. **`backtest` undocumented** — built and never added to `commands.md`.
8. **As-of spacing** would have crammed 60 intraday points into a single week
   (Jul 20–27) while reporting six months. Caught before launch.

**The pattern:** almost every one appeared when real data hit real usage, and none
would have failed a test. Several were things a rename or an addition left behind
— a gitignore entry, a docs section, a module table row. The watcher bugs each
needed a specific situation (narrow terminal, two concurrent runs, intraday
timestamps) that testing a single rendered frame never produced.

---

## 10. Decisions taken, and why

| decision | reasoning |
|---|---|
| conda env, Python 3.12 | Kronos needs 3.10+; system Python is 3.9.6 |
| pip inside conda | the `defaults` channel lacks arm64 torch and alpaca-py |
| Kronos shallow-cloned, `vendor/` gitignored | not on PyPI; avoids the embedded-repo gitlink problem |
| secrets in gitignored `creds.env`, `dotenv_values` | an ambient-env fallback makes it impossible to know which account you're trading |
| paper-only, asserted | a live key (`AK…`) is refused outright |
| Kronos-small default | measured better than base on this task |
| 1 symbol per forward pass | measured fastest on MPS; exact per-symbol seeding |
| both CSVs written every scan | same paths, zero extra compute; discarding one means a re-scan |
| side required on `buy`, prompted if omitted | defaulting to long lets a typo open the opposite position |
| no bulk `buy N` / `sell N` | easy to fire by accident, hard to undo |
| non-TTY declines every confirmation | a piped invocation must never trade unattended |
| guards record rather than drop | a rejected symbol is accounted for, never silently lost |

---

## 11. Current state

- **198 tests passing.** Self-test `scripts/selftest.sh` 24/24 (read-only, dry-run).
- All 9 commands verified against the live paper account.
- A live short round-trip was submitted (`SELL SHORT 6 shares AAL`) and queued —
  market was closed, so the fill and the `BUY TO COVER` close remain unverified.
- Data: 717k daily bars (600 symbols, 5y), 285k 5-min bars (30 symbols, 6 months).
- Docs: `docs/commands.md` (user-facing), `docs/implementation.md` (design + every
  measurement), this file.

### Open items

- **The decisive backtest.** ~350 as-of points at 30Min bars with a 2-bar (1-hour)
  horizon: ~3.5 h on this Mac, plausibly under an hour on the user's 4070 laptop.
  This is the experiment that would actually answer whether the signal exists.
- **Execution lag** in the backtest (`i+lag -> i+lag+horizon`, lag ≈ 4 bars).
- **Temperature 0.7** — verify at scale, then adopt or reject.
- **Lookback 128 vs 512** — unresolved trade-off.
- **Asymmetric LAMBDA** for shorts (unbounded loss).
- **KV cache** — ~5× at a 5-day daily horizon, 20–90× at long intraday horizons.
  Three hazards: window rolling invalidates the cache (fix: `LOOKBACK = 512 −
  horizon` so the buffer never rolls), RoPE needs a position offset, and
  `is_causal=True` is wrong for a cached single query. Gate on cached-vs-uncached
  logits matching bit-for-bit.
- **fp16** — ~2×, a day's work, no model surgery, composes with everything.
- **Finetuning** — `vendor/kronos/finetune_csv/` finetunes tokenizer + predictor
  from a plain OHLCV CSV. `train_sequential.py:30` has **no MPS branch**, so on
  the Mac it falls back to CPU (weeks). On the 4070 it is the supported path,
  roughly overnight, though 8 GB VRAM may force a smaller batch than the shipped
  32. Recommended only *after* evidence the model works on equities at all.

### The recommendation on record

Do not build the KV cache, switch to base, or finetune yet. All three optimise a
signal with no demonstrated edge. Run the properly-powered 30-minute backtest
first — it is the cheapest experiment that can produce a real answer, and every
other decision depends on it.
