"""Forecast-health diagnostics -- the three numbers that say whether the model
is coping with a given timeframe, and the censoring trap they exist to avoid.

The diagnostics are:

    rejection rate     scanned vs rejected
    median |mu|/vol    the ratio the 3x plausibility guard rejects on
    median sigma/vol   forecast dispersion against realized volatility

They are read as a pattern, not individually. `sigma/realized` well below 1 is
weighted most heavily: it is the confidently-wrong signature, and the dangerous
one because such forecasts look clean enough to trade.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas import config, scoring
from atlas.scoring import (
    Health,
    InvalidForecast,
    health,
    score_paths,
    summarize_health,
)

from test_scoring import paths_from_terminal


def _health(n, *, mu_ratio, sigma_ratio, rejected=False):
    """A Health row with the two ratios set directly."""
    return [
        Health(
            symbol=f"S{i}",
            mu=0.01,
            sigma=0.01,
            realized_vol=0.01,
            mu_vol_ratio=mu_ratio,
            sigma_vol_ratio=sigma_ratio,
            paths_used=25,
            paths_total=25,
            rejected=rejected,
        )
        for i in range(n)
    ]


# --- the censoring trap -------------------------------------------------------


def test_the_median_ratio_counts_rejected_symbols():
    """The whole reason health is measured before the guards.

    |mu|/vol is exactly what the 3x guard rejects on, so a median over scored
    symbols alone cannot exceed 3.0 however badly the model behaves -- the
    reference "broken" value of 4.80 would be unreachable by construction.
    """
    entries = _health(5, mu_ratio=1.0, sigma_ratio=1.0) + _health(
        15, mu_ratio=6.0, sigma_ratio=1.0, rejected=True
    )
    summary = summarize_health(entries)

    assert summary.median_mu_vol == pytest.approx(6.0)
    assert summary.median_mu_vol > config.MAX_MU_VOL_MULTIPLE


def test_a_scored_only_median_would_have_hidden_it():
    """Contrast: filtering to survivors first caps the metric below the guard."""
    entries = _health(5, mu_ratio=1.0, sigma_ratio=1.0) + _health(
        15, mu_ratio=6.0, sigma_ratio=1.0, rejected=True
    )
    survivors = summarize_health([e for e in entries if not e.rejected])

    assert survivors.median_mu_vol == pytest.approx(1.0)


# --- health agrees with the guards -------------------------------------------


def test_health_marks_exactly_what_scoring_rejects():
    """One guard definition, two callers -- they must not drift apart."""
    cases = [
        paths_from_terminal([0.01, 0.02, 0.03], vol=0.02),  # plausible
        paths_from_terminal([0.30, 0.31, 0.32], vol=0.02),  # implausible mean
        paths_from_terminal([0.01, 0.02, 0.03], vol=0.0),  # guard disabled
    ]
    for paths in cases:
        try:
            score_paths(paths)
            rejected_by_scoring = False
        except InvalidForecast:
            rejected_by_scoring = True
        assert health(paths).rejected is rejected_by_scoring


def test_health_never_raises_on_unusable_paths():
    """A symbol too broken to measure still has to land in the denominator."""
    broken = paths_from_terminal([-2.0, -2.0, -2.0], vol=0.02)  # negative prices
    result = health(broken)

    assert result.rejected is True
    assert result.reason
    assert np.isnan(result.mu_vol_ratio)


def test_unmeasurable_symbols_count_as_rejected_but_not_in_the_medians():
    entries = _health(4, mu_ratio=1.0, sigma_ratio=1.0) + [
        Health("BAD", float("nan"), float("nan"), 0.01, float("nan"), float("nan"),
               0, 25, True, "no valid paths")
    ]
    summary = summarize_health(entries)

    assert summary.n == 5
    assert summary.rejection_rate == pytest.approx(0.2)
    assert summary.median_mu_vol == pytest.approx(1.0)  # NaN excluded, not propagated


# --- the ratios themselves ----------------------------------------------------


def test_sigma_vol_ratio_is_dispersion_over_realized_volatility():
    paths = paths_from_terminal([-0.02, 0.0, 0.02], vol=0.02)
    result = health(paths)

    assert result.sigma_vol_ratio == pytest.approx(np.std([-0.02, 0.0, 0.02]) / 0.02)


def test_score_carries_the_same_ratios_as_health():
    paths = paths_from_terminal([-0.01, 0.0, 0.01], vol=0.02)
    scored, measured = score_paths(paths), health(paths)

    assert scored.sigma_vol_ratio == pytest.approx(measured.sigma_vol_ratio)
    assert scored.mu_vol_ratio == pytest.approx(measured.mu_vol_ratio)
    assert scored.realized_vol == pytest.approx(0.02)


def test_a_zero_volatility_symbol_gets_a_zero_ratio_not_an_infinity():
    result = health(paths_from_terminal([0.01, 0.02, 0.03], vol=0.0))

    assert result.sigma_vol_ratio == 0.0
    assert result.mu_vol_ratio == 0.0


# --- the verdict --------------------------------------------------------------


def test_healthy_reference_numbers_read_as_healthy():
    """The measured `small`-on-daily figures from the session log."""
    entries = _health(82, mu_ratio=1.06, sigma_ratio=1.0) + _health(
        18, mu_ratio=1.06, sigma_ratio=1.0, rejected=True
    )
    summary = summarize_health(entries)

    assert summary.rejection_rate == pytest.approx(0.18)
    assert summary.concerns == []
    assert summary.verdict.startswith("healthy")


def test_broken_reference_numbers_read_as_broken():
    """The measured `base`-on-daily figures: 68% rejected, 4.80, 0.35."""
    entries = _health(32, mu_ratio=4.80, sigma_ratio=0.35) + _health(
        68, mu_ratio=4.80, sigma_ratio=0.35, rejected=True
    )
    summary = summarize_health(entries)

    assert len(summary.concerns) == 3
    assert summary.verdict.startswith("broken")


def test_one_bad_number_is_ambiguous_not_broken():
    """Small samples lie -- a lone outlier must not read as a conclusion."""
    entries = _health(90, mu_ratio=5.0, sigma_ratio=1.0) + _health(
        10, mu_ratio=5.0, sigma_ratio=1.0, rejected=True
    )
    summary = summarize_health(entries)

    assert summary.concerns == ["median |mu|/vol"]
    assert summary.verdict.startswith("ambiguous")


def test_tight_dispersion_alone_is_called_out_as_the_dangerous_case():
    """sigma/realized well below 1 is weighted most heavily on its own."""
    entries = _health(90, mu_ratio=1.0, sigma_ratio=0.35) + _health(
        10, mu_ratio=1.0, sigma_ratio=0.35, rejected=True
    )
    summary = summarize_health(entries)

    assert summary.concerns == ["sigma/realized"]
    assert summary.verdict.startswith("suspect")


def test_a_small_sample_carries_a_caveat():
    entries = _health(10, mu_ratio=4.8, sigma_ratio=0.35) + _health(
        20, mu_ratio=4.8, sigma_ratio=0.35, rejected=True
    )
    summary = summarize_health(entries)

    assert summary.n < scoring.MIN_TRUSTWORTHY_SAMPLE
    assert "widen to ~100" in summary.verdict


def test_an_empty_run_does_not_divide_by_zero():
    summary = summarize_health([])

    assert summary.n == 0
    assert np.isnan(summary.rejection_rate)
