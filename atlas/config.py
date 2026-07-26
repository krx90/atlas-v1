"""Paths, tunables and credential loading.

Credentials live in a gitignored `creds.env` at the project root and nowhere
else. They are read with `dotenv_values` rather than `load_dotenv` so that a
stray shell variable can never silently stand in for the file -- if the file is
missing or incomplete you get a message naming the file and the key.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDS_FILE = PROJECT_ROOT / "creds.env"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "atlas.db"
KRONOS_DIR = PROJECT_ROOT / "vendor" / "kronos"

#: Scan output. Fully overwritten every run -- never appended to.
TOP_ASSETS_CSV = PROJECT_ROOT / "top30_assets.csv"
#: Rows written to TOP_ASSETS_CSV. The full scored universe is archived separately.
TOP_N = 30
#: Age past which `atlas buy N` warns that the scan is stale.
SCAN_STALE_HOURS = 24

SKIPPED_CSV = DATA_DIR / "scan_skipped.csv"

# --------------------------------------------------------------------------
# Universe construction
# --------------------------------------------------------------------------

#: Exchanges we consider investable. Excludes OTC.
ALLOWED_EXCHANGES = {"NYSE", "NASDAQ", "ARCA", "AMEX", "BATS"}
#: Symbols surviving the liquidity screen, ranked by median dollar volume.
UNIVERSE_SIZE = 600
#: Liquidity screen thresholds, measured over LIQUIDITY_LOOKBACK_DAYS.
MIN_DOLLAR_VOLUME = 5_000_000.0
MIN_PRICE = 5.0
LIQUIDITY_LOOKBACK_DAYS = 30
#: The universe is rebuilt from scratch when the cached one is older than this.
UNIVERSE_MAX_AGE_DAYS = 30

# --------------------------------------------------------------------------
# Bars
# --------------------------------------------------------------------------

#: Years of daily history seeded on first run. Later runs fetch only the delta.
HISTORY_YEARS = 5
#: Symbols per multi-symbol bars request.
BARS_SYMBOLS_PER_REQUEST = 200
#: Alpaca's Basic plan rejects SIP queries whose `end` is inside the last 15
#: minutes. One extra minute of slack absorbs clock skew.
DATA_DELAY_MINUTES = 16
#: Requests per minute. Basic plan allows 200.
RATE_LIMIT_PER_MINUTE = 200

# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------

KRONOS_MODEL = "NeoQuasar/Kronos-small"
KRONOS_TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
#: Kronos-small's context window, and therefore the required bar history.
LOOKBACK = 512
#: Trading days ahead to forecast.
HORIZON = 5
#: Monte Carlo paths sampled per symbol.
PATHS = 25
#: Symbols per forward pass. Each contributes `PATHS` sequences, so the default
#: of 1 means 25 sequences of 512 bars per pass.
#:
#: Measured on MPS (40 symbols, 25 paths, 5-day horizon) -- bigger is *slower*,
#: because Kronos has no KV cache and recomputes attention over the full context
#: at every autoregressive step, so a large batch only multiplies memory traffic:
#:
#:   symbols/pass:      1      2      5     10     20
#:   sec/symbol:     2.48   2.53   2.70   2.72   4.96
#:
#: One symbol per pass also makes the RNG seed exactly per-symbol, so re-scans
#: reproduce regardless of universe ordering.
SYMBOLS_PER_BATCH = 1
#: Kronos sampling parameters.
TEMPERATURE = 1.0
TOP_P = 0.9
#: Base RNG seed. Each symbol derives its own from this, so re-scans reproduce.
SEED = 20260726

# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

#: Weight on the downside tail penalty in the composite score.
LAMBDA_DOWNSIDE = 0.5
#: Fraction of the requested paths that must be numerically valid for a symbol
#: to be scored. Kronos can emit non-positive prices when the input window has
#: high dispersion; those paths are discarded rather than averaged in. Expressed
#: as a fraction, not a count, so `--paths 5` and `--paths 25` behave alike.
MIN_VALID_PATH_FRACTION = 0.6
#: Absolute floor -- two paths is the minimum for a standard deviation.
MIN_VALID_PATHS = 2
#: Reject a forecast whose mean move exceeds this multiple of the symbol's own
#: realized volatility over the same horizon. Kronos normalizes by the input
#: window's mean and standard deviation, and on a strongly-trending daily series
#: that standard deviation is dominated by trend rather than by short-term
#: volatility -- so sampling noise denormalizes into implausible moves. Measured
#: on 113 symbols: at a 512-bar lookback 19.5% of symbols produced |mu| > 15%,
#: concentrated in semiconductors and hardware.
#:
#: Calibration: among symbols that passed, |mu|/realized_vol had p50 0.73, p90
#: 2.49, p99 3.73. This is deliberately set to catch indisputable artifacts and
#: no more -- it is a numerical guard, not a strategy filter. The ambiguous
#: middle is exposed as the `mu_vol_ratio` CSV column so it can be filtered on
#: judgement rather than silently excluded here.
MAX_MU_VOL_MULTIPLE = 3.0
#: Below this realized volatility the multiple-based bound is meaningless -- a
#: barely-moving instrument would reject any forecast at all. One real case:
#: "-0.4% against realized 0.0% volatility".
MIN_VOL_FOR_GUARD = 0.002
#: And always permit a move at least this large, whatever the volatility. This
#: floor is why `mu_vol_ratio` can legitimately exceed MAX_MU_VOL_MULTIPLE for
#: very low-volatility instruments. Keep it small: at 2% a bond ETF with 0.3%
#: weekly volatility passed at a 6.5x ratio, which defeats the point.
MIN_PLAUSIBLE_MU = 0.01
#: A symbol is signalled BUY only if it clears both thresholds.
BUY_MIN_MU = 0.005
BUY_MIN_P_UP = 0.60
#: Below these it is signalled AVOID; in between, HOLD.
AVOID_MAX_MU = 0.0

# --------------------------------------------------------------------------
# Trading
# --------------------------------------------------------------------------

DEFAULT_ORDER_DOLLARS = 100.0

# --------------------------------------------------------------------------
# News
# --------------------------------------------------------------------------

NEWS_LOOKBACK_HOURS = 48
RSS_FEEDS = [
    ("CNBC Markets", "https://www.cnbc.com/id/20910258/device/rss/rss.html"),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
]


class ConfigError(RuntimeError):
    """Raised for a missing or malformed credentials file."""


@dataclass(frozen=True)
class Credentials:
    key_id: str
    secret_key: str
    paper: bool


def _permissions_warning(path: Path) -> str | None:
    """Return a warning if the creds file is readable by anyone but its owner."""
    mode = path.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
        return f"{path.name} is readable by other users. Run: chmod 600 {path}"
    return None


def load_credentials() -> tuple[Credentials, str | None]:
    """Read `creds.env`. Returns the credentials and an optional warning.

    Deliberately does not fall back to the ambient environment: secrets belong
    in the file, and a silent fallback makes it impossible to tell which
    account you are about to trade against.
    """
    if not CREDS_FILE.exists():
        raise ConfigError(
            f"No credentials file at {CREDS_FILE}.\n"
            f"  cp creds.env.example creds.env && chmod 600 creds.env\n"
            f"then fill in your paper keys from "
            f"https://app.alpaca.markets/paper/dashboard/overview"
        )

    values = dotenv_values(CREDS_FILE)
    missing = [k for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY") if not values.get(k)]
    if missing:
        raise ConfigError(f"{CREDS_FILE} is missing: {', '.join(missing)}")

    key_id = str(values["APCA_API_KEY_ID"]).strip()
    secret_key = str(values["APCA_API_SECRET_KEY"]).strip()
    paper = str(values.get("APCA_API_PAPER", "true")).strip().lower() in {"1", "true", "yes"}

    # Atlas places real orders. It is paper-only by construction, and a live key
    # is the one mistake that cannot be undone after the fact.
    if not paper:
        raise ConfigError(
            "APCA_API_PAPER is not true. Atlas only trades paper accounts; "
            "refusing to start against live trading."
        )
    if key_id.startswith("AK"):
        raise ConfigError(
            f"{key_id[:6]}... looks like a live-trading key (paper keys start with 'PK'). "
            "Refusing to start."
        )

    return Credentials(key_id, secret_key, paper), _permissions_warning(CREDS_FILE)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def configure_torch_env() -> None:
    """Set torch/HF environment variables before torch is imported.

    `PYTORCH_ENABLE_MPS_FALLBACK` lets any op without an MPS kernel fall back to
    CPU instead of raising. `HF_HUB_OFFLINE` is not set -- weights need to
    download once -- but the cache is pinned so repeated loads resolve locally.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
