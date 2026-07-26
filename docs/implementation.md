# Atlas — Implementation

Atlas is a command-line tool that scans a universe of US equities, ranks them with the
[Kronos](https://github.com/shiyu-coder/Kronos) candlestick foundation model, and manages an
Alpaca **paper** trading account around the result. `docs/commands.md` is the user-facing
command reference; this document explains how it is built and why.

---

## 1. Overview

The core idea is that Kronos is a *generative* model. Given a window of OHLCV bars it samples
plausible continuations rather than emitting a single number, so running it 25 times gives a
distribution of five-day outcomes per symbol. A distribution supports things a point forecast
cannot: probability of a gain, dispersion, the 5th-percentile tail, expected drawdown. The
ranking is built from those statistics, not from a predicted price.

Data flow for `atlas scan`:

```
Alpaca /v2/assets        ──►  universe.py   ──►  ~600 liquid symbols   ──►  SQLite `universe`
Alpaca /v2/stocks/bars   ──►  bars.py       ──►  5y daily OHLCV         ──►  SQLite `bars`
                                                        │
                                                        ▼
                              forecast.py  ──►  KronosEngine (loaded once)
                                                        │  25 sampled paths x 5 days per symbol
                                                        ▼
                              scoring.py   ──►  mu, sigma, p_up, q05, mdd, score
                                                        │
                                                        ▼
                              results.py   ──►  top30_assets.csv  (atomic overwrite)
                                                  data/scan_full_<ts>.csv  (archive)
```

The other seven commands operate on the paper account or inspect individual instruments; none
of them touch the model.

---

## 2. Layout

| Path | Role |
|---|---|
| `atlas/cli.py` | argparse dispatch, credential preflight, lazy command import |
| `atlas/config.py` | paths, every tunable, credential loading, paper-only guard |
| `atlas/alpaca_client.py` | cached SDK clients, rate limiting, retries, SIP→IEX probe |
| `atlas/db.py` | SQLite schema and connection handling |
| `atlas/bars.py` | daily bar cache: seed, delta top-up, load |
| `atlas/universe.py` | asset sweep and liquidity screen |
| `atlas/forecast.py` | Kronos loading and batched Monte Carlo sampling |
| `atlas/scoring.py` | paths → statistics → composite score |
| `atlas/results.py` | scan CSV read/write, atomic overwrite, staleness |
| `atlas/ui.py` | rich tables, order summary block, confirmations |
| `atlas/commands/` | one module per verb |
| `vendor/kronos/` | cloned Kronos source (gitignored) |
| `data/atlas.db` | SQLite cache (gitignored) |

---

## 3. Environment

Conda supplies the interpreter; pip supplies the packages. The `defaults` channel does not
carry a current arm64 build of torch or `alpaca-py`, and the official PyTorch wheels from PyPI
are the supported path for Apple-Silicon MPS. Mixing the two is safe in this direction.

```bash
conda create -n atlas python=3.12 -y
conda activate atlas
git clone --depth 1 https://github.com/shiyu-coder/Kronos vendor/kronos
pip install -r requirements.txt
pip install -e .
```

Python 3.12 because Kronos requires 3.10+. `environment.yml` reproduces the same env in one
step. Verified working: torch 2.13.0 with `torch.backends.mps.is_available() == True`.

### Why Kronos comes from git rather than pip

Kronos splits across two sources, and this trips people up:

- **Weights** come from HuggingFace automatically. `Kronos.from_pretrained("NeoQuasar/Kronos-small")`
  downloads and caches them under `~/.cache/huggingface` on first run. You never handle a
  checkpoint by hand.
- **Model code** cannot come from HuggingFace. Both HF repos contain only `config.json` and
  `model.safetensors` — no modeling file and no `auto_map`, so `trust_remote_code=True` has
  nothing to load. Kronos subclasses `PyTorchModelHubMixin`, not a `transformers` auto-class,
  and that mixin fetches weights into a class definition that must already exist locally.
  `config.json` is hyperparameters (`d_model: 512, n_layers: 8, n_heads: 8`), not architecture.

It cannot come from pip either: the Kronos repo has no `setup.py`, `pyproject.toml` or
`setup.cfg`, so `pip install git+…` fails. **The PyPI package named `kronos` is an unrelated
Django cron-scheduling library — do not install it.**

So the code is shallow-cloned into `vendor/kronos`, which `forecast.py` puts on `sys.path`.
`vendor/` is gitignored, which also sidesteps the embedded-repository problem a nested `.git`
would otherwise create. Only three files are used —
`model/{__init__,kronos,module}.py`, about 30 KB of the repo's 16 MB. Upstream tip when this
was built: `67b630e6` (2026-04-13), MIT licensed.

### Credentials

Secrets live in `creds.env` at the project root — gitignored, `chmod 600`, with
`creds.env.example` committed as a template.

```
APCA_API_KEY_ID=PK...
APCA_API_SECRET_KEY=...
APCA_API_PAPER=true
```

`config.load_credentials` reads the file with `dotenv_values`, deliberately **not**
`load_dotenv`. A silent fallback to ambient environment variables would make it impossible to
tell which account you are about to trade against. Three guards apply:

- `APCA_API_PAPER` must be true.
- A key id starting with `AK` (live) is refused outright, whatever the flag says.
- Group- or world-readable permissions produce a warning naming the `chmod` fix.

`cli.py` preflights all of this before dispatch, so a command fails on the missing file rather
than partway through a universe sweep.

---

## 4. Market data

### Feed selection

Alpaca's Basic (free) plan serves full historical SIP daily bars provided the query's `end` is
at least 15 minutes old; only recent timestamps are gated. `alpaca_client.data_end()` clamps
every request to `now − 16 minutes` to stay clear of that boundary.

`alpaca_client.feed()` probes SIP once and falls back to IEX on a subscription error, caching
the result. The distinction matters more than it looks: IEX is a single venue carrying a small
fraction of consolidated volume, and the liquidity screen ranks on dollar volume, so an
IEX-only account gets a materially different universe.

**Historical and real-time are separate entitlements**, which cost a bug during verification.
The Basic plan serves historical SIP outside the 15-minute window but refuses SIP for anything
real-time, so this account is legitimately SIP for bars and IEX for quotes. Probing with a bars
request and reusing that answer for `get_stock_latest_trade` produced
`subscription does not permit querying recent SIP data` on every buy. `realtime_feed()` now
probes the latest-trade endpoint separately. Note that Alpaca returns this as a message without
a reliable status code, so the check matches on the message text as well as 401/403.

Requests are throttled through a sliding-window `RateLimiter` at 200/minute (the Basic limit).
429s and 5xx retry with exponential backoff; 4xx errors raise immediately, since an unknown
symbol or an unentitled feed will not succeed on a second attempt.

### Universe construction

Three stages, so five years of history is never pulled for eleven thousand symbols:

1. **Sweep.** `get_all_assets(status=ACTIVE, asset_class=US_EQUITY)`, keeping `tradable`
   assets on NYSE, NASDAQ, ARCA, AMEX or BATS. Symbols longer than five characters, or
   containing `.` `/` `+` or a space, are dropped, as are warrant/unit/right/preferred
   suffixes (`.WS`, `.U`, `.R`, `-PA`, …). These trade thinly and have their own price
   dynamics, which is not what a model pretrained on ordinary equities is good at.
2. **Measure.** Fetch ~30 recent sessions for the survivors — roughly 30 multi-symbol requests
   — and compute median dollar volume and last close.
3. **Screen.** Require median dollar volume ≥ $5M and price ≥ $5, then keep the top 600 by
   dollar volume.

Tunables: `UNIVERSE_SIZE`, `MIN_DOLLAR_VOLUME`, `MIN_PRICE`, `LIQUIDITY_LOOKBACK_DAYS`,
`ALLOWED_EXCHANGES`. The universe is cached in SQLite with a `built_at` stamp and rebuilt when
older than `UNIVERSE_MAX_AGE_DAYS` (30).

### Bar cache

```sql
bars(symbol, date, open, high, low, close, volume, trade_count, vwap)
  PRIMARY KEY (symbol, date)
universe(symbol, name, exchange, fractionable, dollar_volume, last_price)
meta(key, value)
```

First run seeds five years; later runs fetch only from each symbol's `last_date + 1` and
upsert. Symbols are grouped by their last cached date so everything needing the same window
travels in one request — on a delta run that is usually a single group covering the universe.

Bars are requested with `adjustment=all`, so prices are split- and dividend-adjusted and
comparable across the whole window. The catch is that adjusted history is rewritten
retroactively by corporate actions, so an incrementally-topped-up cache slowly drifts. The
30-day universe rebuild therefore also reseeds full history (`sync(reseed=True)`), which is
what corrects the drift.

Alpaca stamps daily bars at the session's opening instant in UTC, so `bars._session_dates`
converts to `America/New_York` before taking the date — a naive `.date()` lands on the
previous calendar day for part of the year.

---

## 5. Forecasting

### Checkpoint choice: bigger is worse here

`atlas scan --model {mini,small,base}` selects the checkpoint; the lookback follows the model's
context window. `Kronos-large` is listed upstream but is **not published** (its HF repo returns
401), so `base` is the ceiling.

`small` is the default on measured evidence, not convenience. On 40 identical symbols at 25
paths and a 5-day horizon:

| | Kronos-small (24.7M) | Kronos-base (102.3M) |
|---|---|---|
| scored | 33/40 | **13/40** |
| rejected | 18% | **68%** |
| median raw `mu` | −3.34% | **−27.25%** |
| median &#124;mu&#124; / realized vol | **1.06** | **4.80** |
| `sigma` / realized vol | 1.16 | 0.88 |
| seconds per symbol | **2.43** | 9.67 |
| projected 600 symbols | **24m** | 97m |

Base is four times slower *and* two-thirds of its output is unusable. Its median forecast is a
−27% five-day move at 4.8× realized volatility — the central tendency, not a tail. Rank
agreement where both survive is Spearman ρ = +0.62, so base broadly agrees with small when it
works; it just fails far more often.

The likely cause: base ships with `attn_dropout_p: 0.0` and `token_dropout_p: 0.0` against
small's `0.1`/`0.1`. It is a sharper, less-regularised model, so on daily bars — out of
distribution for a model pretrained mostly on intraday K-lines — it is more *confidently* wrong.
Scale amplifies the normalization pathology of §6 rather than overcoming it.

`base` remains available behind the flag, since this is one market snapshot at one horizon and
one sampling temperature, not a general claim about the checkpoint.

### One load per process

`KronosEngine.__init__` calls `from_pretrained` exactly once, sets `eval()`, and hands both
modules to `KronosPredictor`, which moves them to the device. That instance then serves every
batch for the rest of the run. There is no `from_pretrained` inside any loop, no per-symbol
helper that builds a predictor, and no MPS↔CPU round-trip mid-scan — that transfer is the
usual cause of load/unload thrash. The whole scan runs inside `torch.inference_mode()`.

The engine logs exactly one line at startup:

```
loaded Kronos-small on mps in 8.4s
```

**A second such line during a scan is a bug**, and the verification below asserts on it.

No training happens anywhere in Atlas. Kronos is used purely for pretrained inference and the
weights are read-only for the entire run.

### Sampling distinct paths

This is the subtle part. `KronosPredictor.predict(sample_count=N)` generates N samples and
then **averages them internally**, returning a single mean path
(`auto_regressive_inference`, `vendor/kronos/model/kronos.py:389`). That destroys exactly the
dispersion the scoring depends on — every symbol would come back with `sigma = 0`.

Instead, each symbol's history is replicated 25 times into `predict_batch` at
`sample_count=1`. One batched forward pass then yields 25 independently sampled futures that
stay separate. Verified: 25 of 25 terminal closes distinct.

Inputs per symbol: the last 512 daily bars as `[open, high, low, close, volume, amount]`.
`amount` is turnover — Kronos will synthesise it as volume × mean price, but vwap × volume is
the real figure and vwap is cached, so Atlas supplies it. Sampling uses `T=1.0, top_p=0.9`.

`y_timestamp` is the next five NYSE sessions from Alpaca's own `get_calendar`, avoiding a
market-calendar dependency. If that call fails it falls back to weekdays: a wrong holiday only
slightly perturbs Kronos's time features, which is not worth aborting a scan over.

### Batching and determinism

Batches hold a whole number of symbols, so a symbol's paths never straddle two forward passes.
`SYMBOLS_PER_BATCH` defaults to **1** — one symbol, 25 sequences, per pass.

That default is measured, not assumed. Bigger batches are *slower* on MPS, because Kronos has
no KV cache: `auto_regressive_inference` recomputes attention over the full 512-bar context at
every step, so a larger batch only multiplies memory traffic against unified-memory bandwidth.

| symbols/pass | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| sec/symbol | **2.48** | 2.53 | 2.70 | 2.72 | 4.96 |
| projected 600 | **24.8m** | 25.3m | 27.0m | 49.6m at 20 | |

An earlier default of 5 symbols per pass (from a `BATCH_SIZE // paths` calculation, chosen by
reasoning about memory rather than by measurement) cost about 9%. Twenty per pass is twice as
slow as one.

One symbol per pass has a second benefit: the RNG seed becomes **exactly per-symbol**, so a
re-scan reproduces regardless of universe ordering. Raising `SYMBOLS_PER_BATCH` makes
reproducibility conditional on the grouping instead, since the seed then depends on which
symbols share a pass.

### Coverage

A scan must not quietly shrink. Symbols are partitioned up front into eligible and
skipped-with-a-reason (too few cached bars, NaNs, non-positive close). A batch that raises is
retried once, one symbol at a time, so a single bad series cannot take its neighbours down.
Every skip, rejection and failure is written to `data/scan_skipped.csv`.

Each eligible symbol ends in exactly one of three states — **scored**, **rejected** (the model
returned something and the guards above judged it unusable), or **failed** (the forward pass
raised). Rejected is not the same as lost, which is why they are counted separately. The run
reconciles them:

```
scored 574 / eligible 600 / universe 600 in 1524.3s
  22 rejected as implausible model output
  skipped: 4 x only 341 bars cached
```

If `scored + rejected + failed != eligible` a symbol vanished silently, so the command prints
the missing symbols and **exits non-zero**.

### Measured performance

On an M-series Mac (MPS, Kronos-small, 512-bar context, 25 paths, 5-day horizon):

| Quantity | Measured |
|---|---|
| Model load | 8.4 s cold, ~1 s warm — **once** per run |
| Throughput | 2.48 s per symbol at `SYMBOLS_PER_BATCH = 1` |
| 40-symbol scan | 98 s end to end |
| Projected full universe | **~25 minutes** for 600 symbols |

First run adds the five-year bar download and the ~120 MB weight download on top.
`--limit` and `--paths` cut this down for iteration.

---

## 6. Scoring

For each symbol, with `p0` the last actual close and 25 sampled paths:

```
r_i    = path_i.close[-1] / p0 - 1        five-day return of path i
mu     = mean(r)                          expected return
sigma  = std(r)                           dispersion
p_up   = fraction(r_i > 0)                probability of a gain
q05    = 5th percentile of r              downside tail
mdd    = mean over paths of min(low_t)/p0 - 1
sharpe = mu / (sigma + 1e-6)

score  = (mu - LAMBDA * max(0, -q05)) / max(sigma, 1e-4)
```

Expected return, penalised by the downside tail, per unit of forecast dispersion.

The reasoning: `mu` alone ranks a coin-flip with a fat right tail above a steady grinder, so
dividing by `sigma` is what makes the two comparable. But a symbol can have positive `mu` and
still have a 5th-percentile path that is a collapse, so the tail penalty subtracts that
separately. `LAMBDA` defaults to 0.5 and is a `config.py` tunable; setting it to 0 gives a
plain risk-adjusted return.

The score is absolute rather than z-scored across the universe, so today's 2.4 means the same
thing as last week's 2.4. Ranking is descending, ties broken on symbol for stability.

Signals: **BUY** when `mu ≥ 0.5%` and `p_up ≥ 60%`; **AVOID** when `mu ≤ 0`; **HOLD**
otherwise.

### Guards against unusable model output

Kronos normalizes each input window by its own mean and standard deviation, and denormalizes
its output the same way. On a *daily* series spanning years, that standard deviation is
dominated by the trend rather than by short-term volatility — so one unit of sampling noise in
normalized space denormalizes into an enormous price move. Two failure modes follow, both
observed on real data during verification, and both guarded:

**Non-positive prices.** A high-dispersion window can map a sampled token below zero. One
symbol produced a mean "return" of **−388%** — a negative predicted price. Such a path is a
numerical failure, not a forecast, so `SymbolPaths.valid_mask` drops any path containing a
non-positive or non-finite price. Measured incidence: 27 of 2,825 paths (0.96%) across 113
symbols, concentrated in leveraged ETFs (`SOXS` alone accounted for 22).

A symbol needs `MIN_VALID_PATH_FRACTION` (60%) of its paths to survive, with an absolute floor
of 2. Expressed as a fraction deliberately: a fixed count would reject every symbol under
`--paths 5`.

**Implausible means.** Filtering invalid paths was not sufficient. At a 512-bar lookback,
**22 of 113 symbols still produced |mu| > 15%** over five days, and the affected names were
almost entirely semiconductors and hardware — `SOXL` −65%, `DELL` −62%, `AMD` −61%, `WDC` −55%,
`AMAT` −37%, `KLAC` −32%, `LRCX` −28%. These are the strongest trenders in the sample, which is
exactly what the normalization mechanism predicts.

So any forecast whose mean move exceeds `MAX_MU_VOL_MULTIPLE` (3×) the symbol's own realized
volatility over the same horizon is rejected as an artifact. Rejections are **recorded**, not
silently dropped: they appear in `data/scan_skipped.csv` with a reason and in the run's
reconciliation line.

Two refinements, both from real failures on the first guarded scan:

- **The multiple was calibrated, not guessed.** An initial 4× barely bit — among accepted
  symbols `|mu|/realized_vol` came out p50 0.73, p90 2.49, p99 3.73, max 3.79 — so a −62.45%
  forecast on a high-volatility name passed. 3× rejects the indisputable cases and caps the
  worst surviving mean near −29%.
- **The bound is meaningless at very low volatility.** One symbol was rejected for
  `-0.4% against realized 0.00% volatility`: 3× of almost nothing rejects any forecast at all.
  The guard now requires `MIN_VOL_FOR_GUARD` (0.2%) before applying, and always permits a move
  of at least `MIN_PLAUSIBLE_MU` (2%) whatever the volatility.

Deliberately, the threshold catches artifacts and no more — it is a numerical guard, not a
strategy filter. The ambiguous middle is exposed as the `mu_vol_ratio` column so it can be
filtered on judgement rather than silently excluded. Above roughly 2.5 the forecast is straining
against what the symbol has historically managed in five sessions.

A note on what was *ruled out*. An initial 17-symbol sample showed `corr(trend, mu) = −0.73`,
suggesting a clean linear anti-momentum bias correctable by cross-sectional neutralization. On
113 symbols that correlation collapsed to **−0.09, R² = 0.01** — the original figure was
small-sample noise, and neutralizing on trend would have been fitting to nothing. The effect is
real but concentrated in the tail, which is why a tail guard is the right instrument and a
linear correction is not.

Forecast *dispersion*, by contrast, is well calibrated and needs no adjustment: median forecast
sigma is **1.02–1.04×** realized volatility over the same horizon at a 512-bar lookback.

### The means carry a systematic tilt; the ranking is the usable output

Across 104 scored symbols in a real scan, `mu` had median **−1.99%** and only **33%** of
symbols were positive. That is the same normalization mechanism operating mildly on everything
rather than extremely on a few: most symbols in this market have risen, so most windows have
their last close above the window mean, so most forecasts revert slightly downward.

The practical consequence: **treat the cross-sectional ranking as the signal and the absolute
`mu_pct` as unreliable.** A symbol at rank 3 is genuinely more attractive than one at rank 80
by the model's own reckoning, but "+7.4% expected in five days" should not be read as a
calibrated return estimate. The score's division by `sigma` is unaffected by a common
additive tilt, so ranking survives what the point estimate does not.

This is also why `signal` (BUY/HOLD/AVOID) uses absolute thresholds on `mu` and should be
treated as the weakest column in the output.

### Output

`top30_assets.csv` columns:

```
rank, symbol, name, last_close, score, signal, mu_pct, p_up, sigma_pct,
q05_pct, mdd_pct, sharpe, mu_vol_ratio, horizon_days, paths, paths_used,
model, scanned_at
```

`paths_used` is the number of numerically valid paths the statistics came from; below `paths`
means the validity guard discarded some. `mu_vol_ratio` is `|mu|` as a multiple of realized
volatility — the plausibility yardstick described above, exposed so you can tighten it without
re-running the model.

Every component is written, so the ranking is auditable and the weights can be retuned without
re-running the model.

**Overwrite semantics.** Each run replaces the file wholesale — never appends, never merges. It
is written to a temp file in the same directory and `os.replace`d into position, which is
atomic on one filesystem, so the file on disk is always either the complete new ranking or the
untouched old one even if a scan is interrupted mid-write. `scanned_at` carries the run
timestamp and is what `atlas buy N` reads for its 24-hour staleness warning. The full scored
universe is archived to `data/scan_full_<timestamp>.csv`, so history is retained without the
top-30 file ever growing.

---

## 7. Trading

`atlas buy` distinguishes its two forms by whether the first argument is a number:
`atlas buy AAPL 500` versus `atlas buy 5 500`.

Alpaca accepts a **notional** (dollar) order only for fractionable assets, and only as a market
order with `day` time-in-force. For everything else the order must be whole shares, so Atlas
calls `get_asset` upfront, reads `fractionable`, and rounds down — `floor(dollars / price)` —
so an order never costs more than asked. If a single share exceeds the budget the order is
refused with an explanation rather than sent to be rejected.

Every order prints the summary block from `docs/commands.md` and requires `[y/n]`. A
non-interactive stdin **declines** rather than defaulting to yes, so a piped or scripted
invocation can never trade unattended. `--dry-run` renders the summary and submits nothing.

`atlas sell XYZ` closes an entire position via `close_position`. There is deliberately no
`atlas sell N`: ranking positions worst-to-best and closing the bottom few is a bulk action
that is easy to fire by accident and hard to undo.

---

## 8. Verification

```bash
conda activate atlas
pytest                                    # 69 tests
atlas scan --limit 20 --paths 5           # end-to-end smoke test
atlas scan --limit 20                     # delta path + overwrite behaviour
```

Automated coverage (`tests/`): scoring statistics against synthetic paths with known
distributions, the downside-penalty branch, dispersion ordering, drawdown, and signal
thresholds; fractional versus whole-share order arithmetic including the boundary cases;
period parsing; CSV overwrite, archiving, atomic-failure rollback and staleness; credential
validation including the paper-only guard and the no-ambient-fallback rule; the symbol filter.

Manual invariant checks after a scan:

- exactly one `loaded Kronos-small` line in the output;
- `scored N == eligible M` in the reconciliation line;
- `top30_assets.csv` stays at 30 rows across runs while `scanned_at` advances;
- corrupting one symbol's bars puts it in `data/scan_skipped.csv` with a reason rather than
  making it disappear.

---

## 9. Known limitations

- **Pretraining mismatch.** Kronos was trained predominantly on intraday K-lines. Daily bars
  are in distribution (upstream ships a daily example) but are not what the model saw most of.
  The normalization pathology in §6 is downstream of this: over an intraday window the window
  standard deviation really is short-term volatility, whereas over a multi-year daily window it
  is mostly trend.
- **The plausibility guard is a guard, not a correction.** Rejected symbols are *excluded* from
  the ranking, not repaired — so a strongly-trending name may simply be absent from the scan
  rather than ranked. Check `data/scan_skipped.csv` before concluding a symbol was not
  considered. Roughly 8% of eligible symbols were rejected at 3× in testing.
- **Absolute expected returns are not calibrated.** Median `mu` was −1.99% across a real scan
  with only 33% of symbols positive. Use the ranking, not the point estimate, and treat
  `signal` as the weakest column. See §6.
- **The lookback is not tuned.** 512 bars is Kronos-small's context limit and is what the model
  gets. A 128-bar window measured a far lower artifact rate (0.9% of symbols with |mu| > 15%
  versus 19.5%) but under-dispersed forecasts at 0.59× realized volatility. No principled
  choice between the two was made, and none should be inferred — this needs a backtest, which
  does not exist yet.
- **No transaction costs.** The score models gross price movement. Spread, slippage and market
  impact are absent, which flatters short-horizon signals in particular.
- **`Kronos-large` is unreleased.** `Kronos-small` (24.7M) is the speed/quality choice here;
  `Kronos-base` (102.3M) is configurable via `KRONOS_MODEL` at roughly 4× the runtime.
- **512-bar context.** The model sees about two years of daily history regardless of the five
  years cached. `Kronos-mini` has a 2048-bar context but far fewer parameters.
- **Reproducibility** holds per-symbol at the default `SYMBOLS_PER_BATCH = 1`. Raising it makes
  reproducibility depend on batch grouping, and is also slower — see §5.
- **Adjusted-history drift.** Corrected only on the 30-day rebuild, so between rebuilds the
  cache can lag a corporate action by up to a month.
- **Free-tier feed.** Without SIP entitlement the universe is screened on IEX volumes, which
  are a fraction of consolidated volume.
- **Not investment advice.** This is a paper-trading research tool. The scoring formula is a
  reasonable default, not a validated strategy — there is no backtest behind it.

---

## 10. Build log

**2026-07-26** — Initial implementation.

- Researched the Kronos API directly from source; established that `predict(sample_count=N)`
  averages internally and designed the path-replication workaround.
- Confirmed the HuggingFace repos ship weights only, that the repo has no packaging files, and
  that the PyPI `kronos` name belongs to an unrelated project.
- Confirmed Alpaca Basic serves historical SIP data outside the 15-minute window.
- Built the package, 69 tests, and this document.
- Verified in the `atlas` conda env: torch 2.13.0 on MPS; model loads once (8.4 s); 25 of 25
  sampled paths distinct; ~2.5 s per symbol, projecting to ~25 minutes for 600 symbols.
- Decisions taken with the user: single `top30_assets.csv`; 5-day horizon; `Kronos-small`;
  `atlas sell N` removed in favour of symbol-only selling; conda environment; secrets in a
  gitignored `creds.env`; shallow clone with `vendor/` gitignored.

**2026-07-26, later** — verification against the live paper account, and two real bugs.

- `portfolio`, `orders`, `news`, `info`, `view` verified working. `buy` verified for the
  fractionable path (`AAPL`), the non-fractionable path (`SNDQ`, 14 whole shares from $500),
  and the too-small-for-one-share rejection. `sell` verified against a held position and a
  non-held symbol.
- **Bug 1: feed entitlement.** `buy` failed on every symbol with
  `subscription does not permit querying recent SIP data`. Historical and real-time SIP are
  separate entitlements; the probe validated bars and the result was wrongly reused for quotes.
  Fixed by adding `realtime_feed()`. See §4.
- **Bug 2: unusable model output.** The first real scan produced forecasts like `AMD −54%` and
  `SOXL −65%` over five days. Investigated across 113 symbols: Kronos's per-window
  normalization produces both negative prices (0.96% of paths) and implausible means (19.5% of
  symbols at a 512-bar lookback, concentrated in semiconductors). Added a validity guard and a
  volatility-relative plausibility guard, both recorded rather than silent. An initial
  17-symbol reading suggested a linear anti-momentum bias; that did not survive the wider
  sample and no neutralization was applied. See §6.
- **Guard recalibration**, after measuring the first guarded scan rather than trusting it: 4×
  barely bit (a −62.45% forecast survived) and the bound spuriously rejected a near-zero-
  volatility instrument. Tightened to 3× with a volatility floor and an absolute 2% allowance,
  and added the `mu_vol_ratio` column so the ambiguous middle is filterable instead of silently
  excluded.
- **Measured the mean tilt**: median `mu` −1.99%, 33% positive across 104 symbols. Documented
  that the ranking is the usable output and the absolute expected return is not.
- Also fixed: `MIN_VALID_PATHS` as an absolute count would have rejected every symbol under
  `--paths 5`; it is now a fraction of requested paths. Table column wrapping, `info` summary
  double-wrapping, an opaque `KeyError` on symbols that resolve as assets but have no quotes,
  a skip-reason summary that truncated to four slots and hid the total, and the results CSV
  inheriting `tempfile`'s 0600 permissions.
- **Benchmarked the batch size** instead of leaving it at a reasoned guess. Larger forward
  passes turned out *slower* on MPS (Kronos has no KV cache), so the default moved from 5
  symbols per pass to 1 — about 9% faster, and it makes the RNG seed exactly per-symbol, which
  removed a documented reproducibility caveat. `BATCH_SIZE` was replaced by the clearer
  `SYMBOLS_PER_BATCH`.
- Tightened `MIN_PLAUSIBLE_MU` from 2% to 1% after observing a `mu_vol_ratio` of 6.47 slip past
  a 3× guard: at 2% a 0.31%-volatility bond ETF passed. Max ratio is now 2.67.
- Added `scripts/selftest.sh` — read-only and dry-run, asserting the single-load and coverage
  invariants. Its own first run failed on a `pipefail` + `grep -q` SIGPIPE race in the harness,
  not in Atlas.
- Test suite grew from 69 to 81.
