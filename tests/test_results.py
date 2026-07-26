"""Scan CSV: overwrite semantics, archiving, and staleness detection."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import pytest

from atlas import config, results
from atlas.scoring import Score


@pytest.fixture
def csv_paths(tmp_path, monkeypatch):
    top = tmp_path / "top30_assets.csv"
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(config, "TOP_ASSETS_CSV", top)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SKIPPED_CSV", data / "scan_skipped.csv")
    return top, data


def make_scores(symbols):
    return [
        Score(s, 10.0 + i, 5.0 - i, "BUY", 0.02, 0.7, 0.01, -0.01, -0.02, 2.0)
        for i, s in enumerate(symbols)
    ]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_writes_only_top_n_rows(csv_paths):
    top, _ = csv_paths
    scores = make_scores([f"S{i:03d}" for i in range(50)])
    results.write(scores, {}, horizon=5, paths=25, model="m", top_n=30)

    rows = read_rows(top)
    assert len(rows) == 30
    assert rows[0]["rank"] == "1"
    assert rows[-1]["rank"] == "30"


def test_second_run_replaces_rather_than_appends(csv_paths):
    top, _ = csv_paths
    results.write(make_scores(["AAA", "BBB", "CCC"]), {}, horizon=5, paths=25, model="m", top_n=30)
    first = read_rows(top)

    results.write(make_scores(["XXX", "YYY"]), {}, horizon=5, paths=25, model="m", top_n=30)
    second = read_rows(top)

    assert len(first) == 3
    assert len(second) == 2  # replaced, not 3 + 2
    assert [r["symbol"] for r in second] == ["XXX", "YYY"]
    assert second[0]["scanned_at"] >= first[0]["scanned_at"]


def test_full_universe_is_archived_separately(csv_paths):
    top, data = csv_paths
    scores = make_scores([f"S{i:03d}" for i in range(40)])
    _, archive = results.write(scores, {}, horizon=5, paths=25, model="m", top_n=30)

    assert len(read_rows(top)) == 30
    assert len(read_rows(archive)) == 40
    assert archive.parent == data
    assert archive.name.startswith("scan_full_")


def test_a_failed_write_leaves_the_previous_file_intact(csv_paths, monkeypatch):
    top, _ = csv_paths
    results.write(make_scores(["AAA", "BBB"]), {}, horizon=5, paths=25, model="m", top_n=30)
    before = top.read_text()

    def explode(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(results.os, "replace", explode)
    with pytest.raises(OSError, match="disk full"):
        results.write(make_scores(["CCC"]), {}, horizon=5, paths=25, model="m", top_n=30)

    assert top.read_text() == before
    # No partial temp files left behind.
    assert [p.name for p in top.parent.iterdir() if p.name.startswith(".top30")] == []


def test_components_round_trip_as_percentages(csv_paths):
    top, _ = csv_paths
    score = Score("AAA", 45.23, 2.5, "BUY", 0.0234, 0.72, 0.0155, -0.0089, -0.0201, 1.51)
    results.write([score], {"AAA": "Alpha Inc"}, horizon=5, paths=25, model="Kronos-small", top_n=30)

    row = read_rows(top)[0]
    assert row["symbol"] == "AAA"
    assert row["name"] == "Alpha Inc"
    assert float(row["mu_pct"]) == pytest.approx(2.34)
    assert float(row["q05_pct"]) == pytest.approx(-0.89)
    assert float(row["last_close"]) == pytest.approx(45.23)
    assert row["horizon_days"] == "5"
    assert row["model"] == "Kronos-small"


def test_age_is_none_without_a_file(csv_paths):
    assert results.age_hours() is None


def test_age_is_read_from_the_row_timestamp(csv_paths, monkeypatch):
    top, _ = csv_paths
    results.write(make_scores(["AAA"]), {}, horizon=5, paths=25, model="m", top_n=30)

    rows = read_rows(top)
    stale = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat(timespec="seconds")
    rows[0]["scanned_at"] = stale
    with top.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=results.FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    age = results.age_hours()
    assert age == pytest.approx(30.0, abs=0.1)
    assert age > config.SCAN_STALE_HOURS


def test_fresh_scan_is_not_stale(csv_paths):
    results.write(make_scores(["AAA"]), {}, horizon=5, paths=25, model="m", top_n=30)
    assert results.age_hours() < config.SCAN_STALE_HOURS


def test_skipped_file_is_removed_when_nothing_is_skipped(csv_paths):
    _, data = csv_paths
    results.write_skipped([("AAA", "only 12 bars cached")])
    assert config.SKIPPED_CSV.exists()

    results.write_skipped([])
    assert not config.SKIPPED_CSV.exists()
