"""Kronos inference: load the model once, sample Monte Carlo paths in batches.

Two things this module exists to get right.

**One load per process.** `KronosEngine` calls `from_pretrained` exactly once in
its constructor, moves the weights to the device once, and serves every batch
from the same `KronosPredictor`. Nothing reloads, and nothing migrates between
MPS and CPU mid-run -- that round-trip is the usual cause of load/unload
thrash. The engine logs a single line at startup; a second one means a bug.

**Distinct sampled paths.** `KronosPredictor.predict(sample_count=N)` averages
its N samples internally and hands back one mean path, which destroys exactly
the dispersion the scoring depends on. To keep the paths separate we replicate
each symbol's history N times through `predict_batch` at `sample_count=1`, so
one batched forward pass yields N independently sampled futures.

No training happens here. Kronos is used purely for pretrained inference and
the weights are read-only for the whole run.
"""

from __future__ import annotations

import sys
import time
import zlib
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from . import alpaca_client, config

_PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]


class KronosUnavailable(RuntimeError):
    """The vendored Kronos source is missing."""


@dataclass
class ForecastRequest:
    """One symbol's model input: exactly `LOOKBACK` bars, oldest first."""

    symbol: str
    history: pd.DataFrame  # indexed by timestamp, columns _PRICE_COLS
    last_close: float
    #: Realized volatility over the horizon, from history. Used by scoring as a
    #: plausibility yardstick for the model's output.
    realized_vol: float = 0.0


@dataclass
class SymbolPaths:
    """Sampled futures for one symbol. Arrays are (paths, horizon).

    `lows` and `highs` are both kept because the two sides of a trade have
    opposite risk: a long is hurt by the worst low, a short by the worst high.
    """

    symbol: str
    last_close: float
    closes: np.ndarray
    lows: np.ndarray
    highs: np.ndarray
    realized_vol: float = 0.0

    def valid_mask(self) -> np.ndarray:
        """Which paths are numerically usable.

        Kronos denormalizes with the input window's own mean and standard
        deviation, so a high-dispersion window can map a sampled token to a
        negative price. Such a path is not a forecast, it is a numerical
        failure, and averaging it into the mean corrupts the whole symbol --
        one observed case produced a mean 'return' of -388%.
        """
        series = (self.closes, self.lows, self.highs)
        mask = np.ones(self.closes.shape[0], dtype=bool)
        for arr in series:
            mask &= np.isfinite(arr).all(axis=1) & (arr > 0).all(axis=1)
        return mask


def _stamp_tensor(torch, timestamps, paths: int, device: str):
    """Kronos's five time features, replicated across sampled paths."""
    from model.kronos import calc_time_stamps  # noqa: PLC0415 -- vendored

    frame = calc_time_stamps(timestamps)
    arr = np.repeat(frame.to_numpy(dtype="float32")[None], paths, axis=0)
    return torch.from_numpy(arr).to(device)


def realized_vol(closes: np.ndarray, horizon: int, window: int = 250) -> float:
    """Annualization-free realized volatility scaled to `horizon` sessions."""
    log_returns = np.diff(np.log(closes[-(window + 1) :]))
    if log_returns.size < 2:
        return 0.0
    return float(np.std(log_returns) * np.sqrt(horizon))


def _is_cached(repo_id: str) -> bool:
    """Are this repo's weights already in the local HuggingFace cache?

    Checked so loading can pass `local_files_only=True` when there is nothing to
    download. That skips a hub round-trip on every run -- worth ~0.9s of startup,
    and it silences the `unauthenticated requests to the HF Hub` warning, which
    is emitted purely because the check happens at all. A fresh machine with an
    empty cache still downloads normally.
    """
    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
    except ImportError:
        return False
    try:
        return isinstance(try_to_load_from_cache(repo_id, "model.safetensors"), str)
    except Exception:  # noqa: BLE001 -- a cache probe must never block a load
        return False


def _import_kronos():
    """Import the vendored Kronos package, with an actionable error if absent."""
    model_dir = config.KRONOS_DIR
    if not (model_dir / "model" / "kronos.py").exists():
        raise KronosUnavailable(
            f"Kronos source not found at {model_dir}.\n"
            f"The weights come from HuggingFace automatically, but the model code has to be "
            f"cloned (it is not on PyPI):\n"
            f"  git clone --depth 1 https://github.com/shiyu-coder/Kronos {model_dir}"
        )
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: PLC0415

    return Kronos, KronosTokenizer, KronosPredictor


def prepare_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Shape cached bars into Kronos's expected input.

    `amount` is turnover. Kronos will synthesise it as volume x mean price if
    absent, but vwap x volume is the real figure and we have vwap cached.
    """
    out = frame[["open", "high", "low", "close", "volume"]].astype("float64").copy()
    vwap = frame["vwap"] if "vwap" in frame.columns else None
    if vwap is not None and vwap.notna().all():
        out["amount"] = out["volume"] * vwap.astype("float64")
    else:
        out["amount"] = out["volume"] * out[["open", "high", "low", "close"]].mean(axis=1)
    return out[_PRICE_COLS]


def next_sessions(after: date, count: int) -> list[pd.Timestamp]:
    """The next `count` NYSE sessions strictly after `after`.

    Uses Alpaca's own calendar rather than adding a market-calendar dependency,
    and falls back to weekdays if the calendar call fails -- a wrong holiday
    only perturbs Kronos's time features slightly, so it is not worth aborting
    a scan over.
    """
    from alpaca.trading.requests import GetCalendarRequest  # noqa: PLC0415

    horizon_end = after + timedelta(days=count * 3 + 14)
    try:
        calendar = alpaca_client.call(
            alpaca_client.trading().get_calendar,
            GetCalendarRequest(start=after + timedelta(days=1), end=horizon_end),
        )
        sessions = [pd.Timestamp(day.date) for day in calendar][:count]
        if len(sessions) == count:
            return sessions
    except Exception:  # noqa: BLE001 -- fall through to the weekday approximation
        pass

    sessions, cursor = [], pd.Timestamp(after)
    while len(sessions) < count:
        cursor += pd.Timedelta(days=1)
        if cursor.weekday() < 5:
            sessions.append(cursor)
    return sessions


class KronosEngine:
    """Owns the model for the lifetime of a command. Construct once."""

    def __init__(
        self,
        *,
        model_name: str = config.KRONOS_MODEL,
        tokenizer_name: str = config.KRONOS_TOKENIZER,
        device: str | None = None,
        max_context: int = config.LOOKBACK,
        quiet: bool = False,
        use_cache: bool = False,
    ) -> None:
        config.configure_torch_env()
        import torch  # noqa: PLC0415  -- imported after the env vars are set

        Kronos, KronosTokenizer, KronosPredictor = _import_kronos()

        self.model_name = model_name
        self.max_context = max_context
        # Opt-in: the cache requires lookback + horizon <= max_context so the
        # context window never rolls, which not every caller satisfies.
        self.use_cache = use_cache
        self._torch = torch

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"
        self.device = device

        started = time.perf_counter()
        # Only when both are already cached: mixing a cached model with an
        # uncached tokenizer would fail the load rather than fetch the missing
        # half, so the two are decided together.
        offline = _is_cached(model_name) and _is_cached(tokenizer_name)
        opts = {"local_files_only": True} if offline else {}
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name, **opts)
        model = Kronos.from_pretrained(model_name, **opts)
        tokenizer.eval()
        model.eval()
        # KronosPredictor moves both to the device. This is the only transfer;
        # nothing returns to CPU until the process exits.
        self.predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
        elapsed = time.perf_counter() - started

        if not quiet:
            short = model_name.split("/")[-1]
            print(f"loaded {short} on {device} in {elapsed:.1f}s")

    def _seed(self, symbols: tuple[str, ...]) -> None:
        """Seed deterministically from the batch's symbols.

        At the default of one symbol per batch this is an exact per-symbol seed,
        so a re-scan reproduces regardless of universe ordering. Raise
        `SYMBOLS_PER_BATCH` and reproducibility becomes conditional on the
        grouping instead, since the seed then depends on which symbols share a
        pass.

        **Not `hash()`**: Python randomises string hashing per process, so
        `hash(("NVDA",))` differs on every run. This silently made scans
        irreproducible while the docstring claimed otherwise -- two identical
        runs came out at rank correlation 0.94 with zero identical scores. CRC32
        is stable across processes and machines.
        """
        digest = zlib.crc32("|".join(symbols).encode()) & 0x7FFFFFFF
        self._torch.manual_seed(config.SEED ^ digest)
        if self.device.startswith("cuda"):
            self._torch.cuda.manual_seed_all(config.SEED ^ digest)

    def forecast(
        self,
        requests: list[ForecastRequest],
        *,
        horizon: int = config.HORIZON,
        paths: int = config.PATHS,
        symbols_per_batch: int = config.SYMBOLS_PER_BATCH,
        on_batch: Callable[[list[str]], None] | None = None,
        on_failure: Callable[[list[str], Exception], None] | None = None,
    ) -> Iterator[SymbolPaths]:
        """Yield sampled paths per symbol.

        Batches hold a whole number of symbols so a symbol's paths never
        straddle two forward passes. A batch that raises is retried once, one
        symbol at a time, so a single bad series cannot take its neighbours
        down with it.

        See `config.SYMBOLS_PER_BATCH` for why the default is 1 -- larger passes
        measured slower on MPS, not faster.
        """
        if not requests:
            return

        symbols_per_batch = max(1, symbols_per_batch)
        torch = self._torch

        with torch.inference_mode():
            for start in range(0, len(requests), symbols_per_batch):
                group = requests[start : start + symbols_per_batch]
                try:
                    yield from self._run_batch(group, horizon, paths)
                except Exception as exc:  # noqa: BLE001
                    recovered = self._retry_individually(group, horizon, paths, on_failure, exc)
                    yield from recovered
                if on_batch is not None:
                    on_batch([r.symbol for r in group])

    def _retry_individually(
        self,
        group: list[ForecastRequest],
        horizon: int,
        paths: int,
        on_failure: Callable[[list[str], Exception], None] | None,
        batch_error: Exception,
    ) -> list[SymbolPaths]:
        recovered: list[SymbolPaths] = []
        for request in group:
            try:
                recovered.extend(self._run_batch([request], horizon, paths))
            except Exception as exc:  # noqa: BLE001
                if on_failure is not None:
                    on_failure([request.symbol], exc)
        if len(recovered) < len(group) and on_failure is None:
            raise batch_error
        return recovered

    def _cached_batch(
        self, group: list[ForecastRequest], horizon: int, paths: int
    ) -> list[SymbolPaths]:
        """Generate with a KV cache instead of re-reading the context each step.

        Verified bit-identical to the uncached path under the same seed, and
        3-7x faster depending on horizon. Requires that the context window never
        rolls, which `kv_cache.check_fits` enforces.
        """
        from . import kv_cache  # noqa: PLC0415

        torch = self._torch
        out: list[SymbolPaths] = []
        for request in group:
            self._seed((request.symbol,))
            history = request.history
            lookback = len(history)
            kv_cache.check_fits(lookback, horizon, self.max_context)

            values = history[_PRICE_COLS].to_numpy(dtype="float64")
            mean, std = values.mean(axis=0), values.std(axis=0)
            normed = np.clip((values - mean) / (std + 1e-5), -self.predictor.clip, self.predictor.clip)
            x = torch.from_numpy(
                np.repeat(normed[None].astype("float32"), paths, axis=0)
            ).to(self.device)

            last = history.index[-1].date() if hasattr(history.index[-1], "date") else None
            future = pd.Series(next_sessions(last, horizon)) if last else None
            x_stamp = _stamp_tensor(torch, pd.Series(history.index), paths, self.device)
            y_stamp = _stamp_tensor(torch, future, paths, self.device)

            preds = kv_cache.generate(
                self.predictor.tokenizer,
                self.predictor.model,
                x,
                x_stamp,
                y_stamp,
                horizon,
                max_context=self.max_context,
                clip=self.predictor.clip,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
            )
            arr = preds.detach().cpu().numpy() * (std + 1e-5) + mean
            out.append(
                SymbolPaths(
                    symbol=request.symbol,
                    last_close=request.last_close,
                    closes=arr[:, :, _PRICE_COLS.index("close")],
                    lows=arr[:, :, _PRICE_COLS.index("low")],
                    highs=arr[:, :, _PRICE_COLS.index("high")],
                    realized_vol=request.realized_vol,
                )
            )
        return out

    def _run_batch(
        self, group: list[ForecastRequest], horizon: int, paths: int
    ) -> list[SymbolPaths]:
        if self.use_cache:
            return self._cached_batch(group, horizon, paths)
        self._seed(tuple(r.symbol for r in group))

        # Replicate each symbol `paths` times: predict_batch with sample_count=1
        # returns one independently sampled future per list entry, whereas
        # sample_count=paths would average them into a single mean path.
        df_list: list[pd.DataFrame] = []
        x_stamps: list[pd.Series] = []
        y_stamps: list[pd.Series] = []

        for request in group:
            last_session = request.history.index[-1].date()
            future = pd.Series(next_sessions(last_session, horizon))
            x_stamp = pd.Series(request.history.index)
            for _ in range(paths):
                df_list.append(request.history)
                x_stamps.append(x_stamp)
                y_stamps.append(future)

        predictions = self.predictor.predict_batch(
            df_list=df_list,
            x_timestamp_list=x_stamps,
            y_timestamp_list=y_stamps,
            pred_len=horizon,
            T=config.TEMPERATURE,
            top_p=config.TOP_P,
            sample_count=1,
            verbose=False,
        )

        out: list[SymbolPaths] = []
        for i, request in enumerate(group):
            block = predictions[i * paths : (i + 1) * paths]
            out.append(
                SymbolPaths(
                    symbol=request.symbol,
                    last_close=request.last_close,
                    closes=np.stack([p["close"].to_numpy(dtype="float64") for p in block]),
                    lows=np.stack([p["low"].to_numpy(dtype="float64") for p in block]),
                    highs=np.stack([p["high"].to_numpy(dtype="float64") for p in block]),
                    realized_vol=request.realized_vol,
                )
            )
        return out
