"""The KV cache must be a pure optimisation.

The headline test loads the real model and asserts the cached decoder produces
**byte-identical** output to the uncached one under the same seed. That is the
gate: a cache that is merely "close" is worse than no cache, because every one of
its failure modes produces plausible wrong numbers rather than an error.

It is marked slow and skipped when `vendor/kronos` is absent, so the fast suite
stays fast and a fresh checkout without the vendored source still passes.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas import config
from atlas.kv_cache import Cache, ContextTooLong, LayerCache, check_fits

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# --- the window-rolling guard -------------------------------------------------
#
# Kronos slides its context buffer once the sequence exceeds max_context, which
# shifts every cached key's position and invalidates the cache silently. This is
# the one constraint the caller must satisfy.


def test_a_configuration_that_fits_is_accepted():
    check_fits(507, 5, 512)
    check_fits(100, 10, 512)


def test_exactly_filling_the_context_is_accepted():
    check_fits(507, 5, 512)  # 507 + 5 == 512


def test_overflowing_the_context_is_refused():
    with pytest.raises(ContextTooLong, match="would roll"):
        check_fits(512, 5, 512)


def test_the_refusal_names_the_lookback_that_would_work():
    with pytest.raises(ContextTooLong, match="lookback <= 507"):
        check_fits(512, 5, 512)


def test_the_default_scan_configuration_fits():
    """Regression: scan used to sit exactly one horizon over the limit."""
    lookback = config.data_lookback(512, config.HORIZON, cached=True)
    check_fits(lookback, config.HORIZON, 512)
    assert lookback == 512 - config.HORIZON


def test_the_uncached_lookback_fills_the_whole_context():
    assert config.data_lookback(512, config.HORIZON, cached=False) == 512


def test_the_lookback_is_capped_for_a_long_context_model():
    """mini has a 2048-bar context but five years of dailies is only ~1250."""
    assert config.data_lookback(2048, 5, cached=True) == config.MAX_LOOKBACK


# --- cache bookkeeping --------------------------------------------------------


def test_an_empty_layer_cache_has_no_length():
    assert LayerCache().length == 0


def test_appending_grows_the_cache_along_the_sequence_axis():
    layer = LayerCache()
    k1 = np.zeros((2, 4, 3, 8))  # batch, heads, seq, head_dim
    import torch

    layer.append(torch.zeros(2, 4, 3, 8), torch.zeros(2, 4, 3, 8))
    assert layer.length == 3
    layer.append(torch.zeros(2, 4, 1, 8), torch.zeros(2, 4, 1, 8))
    assert layer.length == 4
    assert k1.shape[2] == 3  # untouched


def test_a_cache_reports_the_length_of_its_layers():
    import torch

    cache = Cache.empty(3)
    assert cache.length == 0
    for layer in cache.layers:
        layer.append(torch.zeros(1, 2, 5, 4), torch.zeros(1, 2, 5, 4))
    assert cache.length == 5


def test_an_empty_cache_has_one_entry_per_layer():
    assert len(Cache.empty(8).layers) == 8


# --- the gate -----------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not (config.KRONOS_DIR / "model" / "kronos.py").exists(),
    reason="vendored Kronos source not present",
)
@pytest.mark.parametrize("horizon", [5, 13])
def test_cached_output_is_identical_to_uncached(horizon):
    """The gate. Anything short of identical means this does not ship.

    Both paths sample (upstream hardcodes `sample_logits=True`), so under one
    seed an exact cache must reproduce every draw -- comparing against greedy
    decoding would be comparing two different algorithms.
    """
    import sys

    import torch

    from atlas import forecast, kv_cache

    sys.path.insert(0, str(config.KRONOS_DIR))
    from model.kronos import auto_regressive_inference  # noqa: PLC0415

    engine = forecast.KronosEngine(quiet=True)
    predictor = engine.predictor
    lookback, paths, seed = 512 - horizon, 8, 4321

    rng = np.random.default_rng(0)
    series = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, lookback)))
    features = np.column_stack([series, series * 1.01, series * 0.99, series,
                                np.full(lookback, 1e6), series * 1e6]).astype("float32")
    normed = (features - features.mean(0)) / (features.std(0) + 1e-5)
    x = torch.from_numpy(np.repeat(np.clip(normed, -5, 5)[None], paths, 0)).to(predictor.device)
    xs = torch.zeros(paths, lookback, 5, device=predictor.device)
    ys = torch.zeros(paths, horizon, 5, device=predictor.device)

    torch.manual_seed(seed)
    uncached = auto_regressive_inference(
        predictor.tokenizer, predictor.model, x, xs, ys, 512, horizon,
        clip=5, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=False,
    )
    torch.manual_seed(seed)
    cached = kv_cache.generate(
        predictor.tokenizer, predictor.model, x, xs, ys, horizon,
        max_context=512, temperature=1.0, top_p=0.9, sample=True,
    )

    ref = np.asarray(uncached)[:, -horizon:, :]
    got = cached.detach().cpu().numpy()
    assert ref.shape == got.shape
    assert np.array_equal(ref, got), (
        f"cached output diverged by {np.abs(ref - got).max():.3e} -- "
        "a KV cache must be a pure optimisation"
    )


# --- seed stability -----------------------------------------------------------
#
# Regression: the seed was derived from Python's `hash()`, which randomises
# string hashing per process. Scans were irreproducible while the docstring
# claimed the opposite -- two identical runs came out at rank correlation 0.94
# with zero identical scores.


def test_the_seed_digest_is_stable_across_processes():
    import subprocess
    import sys

    code = (
        "import zlib;"
        "print(zlib.crc32('|'.join(('NVDA',)).encode()) & 0x7FFFFFFF)"
    )
    seen = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
        for _ in range(3)
    }
    assert len(seen) == 1, f"digest varies between processes: {seen}"


def test_pythons_hash_is_not_stable_and_must_not_be_used():
    """Pins the reason for the fix, so it is not 'simplified' back."""
    import subprocess
    import sys

    code = "print(abs(hash(('NVDA',))) % (2**31))"
    seen = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
        for _ in range(3)
    }
    assert len(seen) > 1, "hash() appears stable here; the guard above still matters"


def test_different_symbols_get_different_seeds():
    import zlib

    digest = lambda s: zlib.crc32("|".join(s).encode()) & 0x7FFFFFFF
    assert digest(("NVDA",)) != digest(("TSLA",))
    assert digest(("NVDA", "TSLA")) != digest(("TSLA", "NVDA"))
