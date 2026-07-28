"""Backtest slicing and metrics.

The point of most of these is lookahead: a backtest that lets the model see
even one bar of the future produces beautiful numbers and tells you nothing.
`slice_at` is therefore tested against explicit, hand-checkable series rather
than only for self-consistency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas.backtest import Observation, as_of_indices, evaluate, slice_at


def make_frame(closes, start="2020-01-01"):
    """A bar frame whose close on row i is closes[i]."""
    closes = np.asarray(closes, dtype="float64")
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(len(closes), 1e6),
            "vwap": closes,
        },
        index=idx,
    )


# --- lookahead ----------------------------------------------------------------


def test_history_never_includes_a_bar_after_the_as_of_row():
    frame = make_frame(np.arange(1, 101, dtype="float64"))
    window = slice_at(frame, i=60, lookback=20, horizon=5)

    # Row 60 has close 61.0; the window must end exactly there.
    assert window.last_close == pytest.approx(61.0)
    assert window.history["close"].iloc[-1] == pytest.approx(61.0)
    assert len(window.history) == 20
    # ...and start 20 rows earlier, not one bar sooner or later.
    assert window.history["close"].iloc[0] == pytest.approx(42.0)
    assert window.history["close"].max() == pytest.approx(61.0)


def test_the_forward_return_is_measured_strictly_after_the_history():
    frame = make_frame([100.0] * 50 + [110.0] * 50)
    # As-of row 49 is the last 100.0; five rows later is 110.0.
    window = slice_at(frame, i=49, lookback=20, horizon=5)

    assert window.last_close == pytest.approx(100.0)
    assert window.forward_return == pytest.approx(0.10)
    # The jump to 110 must be invisible to the model.
    assert window.history["close"].max() == pytest.approx(100.0)


def test_a_crash_after_the_as_of_date_does_not_leak_into_the_window():
    closes = [100.0] * 60 + [10.0] * 40
    frame = make_frame(closes)
    window = slice_at(frame, i=59, lookback=30, horizon=5)

    assert window.forward_return == pytest.approx(-0.90)
    assert window.history["close"].min() == pytest.approx(100.0)
    assert window.realized_vol == pytest.approx(0.0, abs=1e-9)


def test_slicing_refuses_when_there_is_no_future_to_judge_against():
    frame = make_frame(np.arange(1, 51, dtype="float64"))
    assert slice_at(frame, i=49, lookback=20, horizon=5) is None  # runs off the end
    assert slice_at(frame, i=45, lookback=20, horizon=5) is None
    assert slice_at(frame, i=44, lookback=20, horizon=5) is not None


def test_slicing_refuses_when_there_is_not_enough_history():
    frame = make_frame(np.arange(1, 51, dtype="float64"))
    assert slice_at(frame, i=10, lookback=20, horizon=5) is None
    assert slice_at(frame, i=19, lookback=20, horizon=5) is not None


def test_non_positive_prices_are_refused():
    frame = make_frame([100.0] * 30 + [0.0] + [100.0] * 20)
    assert slice_at(frame, i=30, lookback=20, horizon=5) is None  # as-of close is 0
    assert slice_at(frame, i=25, lookback=20, horizon=5) is None  # future close is 0


# --- as-of date selection -----------------------------------------------------


def test_as_of_indices_are_evenly_spaced_and_leave_room_on_both_sides():
    idx = as_of_indices(n_bars=1000, count=5, spacing=10, lookback=512, horizon=5)
    assert idx == [954, 964, 974, 984, 994]
    assert min(idx) >= 512 - 1
    assert max(idx) <= 1000 - 5 - 1


def test_as_of_indices_drop_dates_with_insufficient_history():
    idx = as_of_indices(n_bars=530, count=10, spacing=5, lookback=512, horizon=5)
    assert all(i >= 511 for i in idx)
    assert len(idx) < 10  # some requested dates fall before the lookback


def test_the_newest_as_of_date_still_has_a_realized_outcome():
    n, horizon = 1000, 5
    idx = as_of_indices(n_bars=n, count=1, spacing=5, lookback=512, horizon=horizon)
    assert idx == [n - horizon - 1]
    assert idx[0] + horizon < n  # the outcome bar exists


# --- metrics ------------------------------------------------------------------


def obs(date, pairs, side="long"):
    return [
        Observation(date=date, symbol=s, score=sc, mu=0.0, side=side, forward_return=fr)
        for s, sc, fr in pairs
    ]


def test_a_perfect_ranking_scores_ic_of_one():
    rows = obs("2024-01-01", [(f"S{i}", float(i), float(i) / 100) for i in range(10)])
    result = evaluate(rows)
    assert result.mean_ic == pytest.approx(1.0)
    assert result.mean_spread > 0
    assert result.hit_rate == 1.0


def test_an_inverted_ranking_scores_ic_of_minus_one():
    rows = obs("2024-01-01", [(f"S{i}", float(i), -float(i) / 100) for i in range(10)])
    result = evaluate(rows)
    assert result.mean_ic == pytest.approx(-1.0)
    assert result.mean_spread < 0


def test_a_consistently_inverted_ranking_is_called_out_as_inverted():
    """Needs several dates: one date cannot produce a t-statistic."""
    rng = np.random.default_rng(1)
    rows = []
    for d in range(12):
        # Score is the negative of what happens, plus a little noise.
        pairs = [
            (f"S{i}", float(i) + rng.normal(0, 0.3), -float(i) / 100 + rng.normal(0, 0.002))
            for i in range(15)
        ]
        rows += obs(f"2024-01-{d + 1:02d}", pairs)

    result = evaluate(rows)
    assert result.mean_ic < -0.5
    assert result.ic_t_stat < -2
    assert "inverted" in result.verdict


def test_consistent_skill_is_called_positive():
    rng = np.random.default_rng(2)
    rows = []
    for d in range(12):
        pairs = [
            (f"S{i}", float(i) + rng.normal(0, 0.3), float(i) / 100 + rng.normal(0, 0.002))
            for i in range(15)
        ]
        rows += obs(f"2024-01-{d + 1:02d}", pairs)

    result = evaluate(rows)
    assert result.ic_t_stat > 2
    assert result.verdict == "positive"


def test_a_random_ranking_gives_an_ic_near_zero_and_no_skill_verdict():
    rng = np.random.default_rng(0)
    rows = []
    for d in range(20):
        pairs = [(f"S{i}", float(rng.normal()), float(rng.normal()) / 100) for i in range(30)]
        rows += obs(f"2024-01-{d + 1:02d}", pairs)

    result = evaluate(rows)
    assert abs(result.mean_ic) < 0.2
    assert "no detectable skill" in result.verdict


def test_the_spread_is_top_decile_minus_bottom_decile():
    pairs = [(f"S{i}", float(i), 0.10 if i >= 9 else -0.02) for i in range(10)]
    result = evaluate(obs("2024-01-01", pairs), decile=0.1)
    assert result.top_return == pytest.approx(0.10)
    assert result.bottom_return == pytest.approx(-0.02)
    assert result.mean_spread == pytest.approx(0.12)


def test_the_t_statistic_needs_more_than_one_date():
    result = evaluate(obs("2024-01-01", [(f"S{i}", float(i), float(i)) for i in range(10)]))
    assert np.isnan(result.ic_t_stat)
    assert "inconclusive" in result.verdict


def test_dates_with_too_few_symbols_are_skipped():
    rows = obs("2024-01-01", [("A", 1.0, 0.01), ("B", 2.0, 0.02)])
    rows += obs("2024-01-02", [(f"S{i}", float(i), float(i) / 100) for i in range(10)])
    result = evaluate(rows)
    assert result.dates == 1  # the two-symbol date carries no information
