"""Intraday bar handling.

The session filter is the one that matters. Alpaca returns extended-hours bars
by default and they are ~59% of the rows -- thin, wide-spread, and nothing like
regular-session bars. Nothing warns you: the bar count just looks generous. Left
unfiltered, most of the model's 512-bar context would be noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas.bars import INTRADAY_TIMEFRAMES, _timeframe, regular_session


def intraday_frame(day="2026-07-20", start="04:00", end="19:55", freq="5min"):
    """A full extended-hours session in UTC, as Alpaca returns it."""
    local = pd.date_range(f"{day} {start}", f"{day} {end}", freq=freq, tz="America/New_York")
    closes = np.linspace(100.0, 101.0, len(local))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": np.full(len(local), 1000.0),
            "vwap": closes,
        },
        index=local.tz_convert("UTC"),
    )


def test_extended_hours_bars_are_dropped():
    frame = intraday_frame()
    kept = regular_session(frame)

    # 04:00-19:55 is 192 five-minute bars; the regular session is 78.
    assert len(frame) == 192
    assert len(kept) == 78


def test_the_kept_window_is_exactly_the_regular_session():
    kept = regular_session(intraday_frame()).tz_convert("America/New_York")
    assert kept.index.min().strftime("%H:%M") == "09:30"
    assert kept.index.max().strftime("%H:%M") == "15:55"


def test_pre_and_post_market_prices_are_actually_excluded():
    """Not just counted out -- the values must be gone."""
    frame = intraday_frame()
    local = frame.tz_convert("America/New_York")
    premarket_close = float(local.between_time("04:00", "09:25")["close"].iloc[0])

    kept = regular_session(frame)
    assert premarket_close not in set(kept["close"].to_numpy())


def test_the_result_stays_in_utc():
    kept = regular_session(intraday_frame())
    assert str(kept.index.tz) == "UTC"


def test_a_naive_index_is_treated_as_utc_not_local():
    """Alpaca sends UTC; assuming local time would shift the session by hours."""
    frame = intraday_frame()
    naive = frame.tz_convert("UTC").tz_localize(None)

    assert len(regular_session(naive)) == len(regular_session(frame))


def test_an_empty_frame_passes_through():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])
    assert regular_session(empty).empty


def test_multiple_sessions_each_keep_their_own_window():
    frames = [intraday_frame(day=d) for d in ("2026-07-20", "2026-07-21", "2026-07-22")]
    combined = pd.concat(frames)
    kept = regular_session(combined)

    assert len(kept) == 78 * 3
    local = kept.tz_convert("America/New_York")
    assert set(local.index.strftime("%H:%M")) == set(
        pd.date_range("09:30", "15:55", freq="5min").strftime("%H:%M")
    )


@pytest.mark.parametrize(
    ("label", "freq", "expected_per_session"),
    [("5Min", "5min", 78), ("15Min", "15min", 26), ("30Min", "30min", 13), ("1Hour", "60min", 6)],
)
def test_declared_bar_counts_match_a_real_session(label, freq, expected_per_session):
    """Exact, not approximate. The count is what verifies the session filter ran.

    An earlier version of this test allowed a tolerance of one, which is what
    let `1Hour` be declared as 7 when both this fixture and live Alpaca data
    yield 6 -- hourly bars are aligned to the clock hour, so the 09:00 bar
    straddles the open and is dropped. A tolerance here hides exactly the kind
    of off-by-one the constant exists to catch.
    """
    assert INTRADAY_TIMEFRAMES[label] == expected_per_session
    assert len(regular_session(intraday_frame(freq=freq))) == expected_per_session


def test_the_hourly_open_bar_is_dropped_because_it_straddles_the_open():
    """09:00-10:00 is half premarket, so the session's first kept bar is 10:00."""
    kept = regular_session(intraday_frame(freq="60min")).tz_convert("America/New_York")
    assert kept.index.min().strftime("%H:%M") == "10:00"
    assert "09:00" not in set(kept.index.strftime("%H:%M"))


def test_timeframe_labels_map_to_alpaca_objects():
    assert _timeframe("5Min").amount == 5
    assert _timeframe("15Min").amount == 15
    assert _timeframe("30Min").amount == 30
    assert _timeframe("1Hour").amount == 1


def test_fetch_windows_cover_the_bars_they_promise():
    """Too short a window skips every symbol as 'insufficient history'."""
    from atlas.bars import fetch_days_for

    for label, per_session in INTRADAY_TIMEFRAMES.items():
        days = fetch_days_for(label, 510)
        sessions = days * (5 / 7)  # calendar days -> trading days
        assert sessions * per_session >= 510, f"{label}: {days}d is too short"


def test_bars_per_day_detects_an_unfiltered_cache():
    """The one cheap check that extended-hours rows did not get in."""
    from atlas.bars import bars_per_day

    frame = intraday_frame(freq="60min")
    assert bars_per_day(regular_session(frame)) == 6
    # Unfiltered, 04:00-19:55 hourly is 16 bars -- far above the expected 6.
    assert bars_per_day(frame) == 16


def test_an_unknown_timeframe_is_refused():
    from atlas.bars import fetch_intraday

    with pytest.raises(ValueError, match="unsupported timeframe"):
        fetch_intraday(["AAPL"], "2Min", start=None)


# --- HuggingFace cache probe --------------------------------------------------
#
# Loading passes local_files_only=True when the weights are already cached, which
# skips a hub round-trip (~0.9s per run) and silences the "unauthenticated
# requests to the HF Hub" warning. A fresh machine must still be able to download.


def test_a_cached_repo_is_detected():
    from unittest.mock import patch
    with patch("huggingface_hub.try_to_load_from_cache", return_value="/some/path/model.safetensors"):
        from atlas.forecast import _is_cached
        assert _is_cached("NeoQuasar/Kronos-small") is True


def test_an_uncached_repo_is_not_detected():
    from unittest.mock import patch
    from atlas.forecast import _is_cached
    with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
        assert _is_cached("NeoQuasar/Kronos-nonexistent") is False


def test_the_sentinel_for_a_known_missing_file_is_not_a_hit():
    """try_to_load_from_cache returns a sentinel object, not a path, for those."""
    from unittest.mock import patch
    from atlas.forecast import _is_cached
    with patch("huggingface_hub.try_to_load_from_cache", return_value=object()):
        assert _is_cached("NeoQuasar/Kronos-small") is False


def test_a_failing_probe_falls_back_to_online():
    """The probe is an optimisation -- it must never block a model load."""
    from unittest.mock import patch
    from atlas.forecast import _is_cached
    with patch("huggingface_hub.try_to_load_from_cache", side_effect=OSError("cache unreadable")):
        assert _is_cached("NeoQuasar/Kronos-small") is False
