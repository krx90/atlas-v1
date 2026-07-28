"""Scan CSV: overwrite semantics, archiving, and staleness detection."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import pytest

from atlas import config, results
from atlas.scoring import Score


@pytest.fixture
def csv_paths(tmp_path, monkeypatch):
    top = tmp_path / "top30_long.csv"
    short = tmp_path / "top30_short.csv"
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(config, "TOP_LONG_CSV", top)
    monkeypatch.setattr(config, "TOP_SHORT_CSV", short)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SKIPPED_CSV", data / "scan_skipped.csv")
    return top, short, data


def make_scores(symbols):
    return [
        Score(s, 10.0 + i, 5.0 - i, "BUY", 0.02, 0.7, 0.01, -0.01, -0.02, 2.0)
        for i, s in enumerate(symbols)
    ]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_writes_only_top_n_rows(csv_paths):
    top, _short, _data = csv_paths
    scores = make_scores([f"S{i:03d}" for i in range(50)])
    results.write({'long': scores}, {}, horizon=5, paths=25, model="m", top_n=30)

    rows = read_rows(top)
    assert len(rows) == 30
    assert rows[0]["rank"] == "1"
    assert rows[-1]["rank"] == "30"


def test_second_run_replaces_rather_than_appends(csv_paths):
    top, _short, _data = csv_paths
    results.write({'long': make_scores(["AAA", "BBB", "CCC"])}, {}, horizon=5, paths=25, model="m", top_n=30)
    first = read_rows(top)

    results.write({'long': make_scores(["XXX", "YYY"])}, {}, horizon=5, paths=25, model="m", top_n=30)
    second = read_rows(top)

    assert len(first) == 3
    assert len(second) == 2  # replaced, not 3 + 2
    assert [r["symbol"] for r in second] == ["XXX", "YYY"]
    assert second[0]["scanned_at"] >= first[0]["scanned_at"]


def test_full_universe_is_archived_separately(csv_paths):
    top, _short, data = csv_paths
    scores = make_scores([f"S{i:03d}" for i in range(40)])
    _written, archive = results.write({'long': scores}, {}, horizon=5, paths=25, model="m", top_n=30)

    assert len(read_rows(top)) == 30
    assert len(read_rows(archive)) == 40
    assert archive.parent == data
    assert archive.name.startswith("scan_full_")


def test_a_failed_write_leaves_the_previous_file_intact(csv_paths, monkeypatch):
    top, _short, _data = csv_paths
    results.write({'long': make_scores(["AAA", "BBB"])}, {}, horizon=5, paths=25, model="m", top_n=30)
    before = top.read_text()

    def explode(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(results.os, "replace", explode)
    with pytest.raises(OSError, match="disk full"):
        results.write({'long': make_scores(["CCC"])}, {}, horizon=5, paths=25, model="m", top_n=30)

    assert top.read_text() == before
    # No partial temp files left behind.
    assert [p.name for p in top.parent.iterdir() if p.name.startswith(".top30")] == []


def test_components_round_trip_as_percentages(csv_paths):
    top, _short, _data = csv_paths
    score = Score("AAA", 45.23, 2.5, "BUY", 0.0234, 0.72, 0.0155, -0.0089, -0.0201, 1.51)
    results.write({"long": [score]}, {"AAA": "Alpha Inc"}, horizon=5, paths=25, model="Kronos-small", top_n=30)

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
    top, _short, _data = csv_paths
    results.write({'long': make_scores(["AAA"])}, {}, horizon=5, paths=25, model="m", top_n=30)

    rows = read_rows(top)
    stale = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat(timespec="seconds")
    rows[0]["scanned_at"] = stale
    with top.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=results.FIELDS["long"])
        writer.writeheader()
        writer.writerows(rows)

    age = results.age_hours()
    assert age == pytest.approx(30.0, abs=0.1)
    assert age > config.SCAN_STALE_HOURS


def test_fresh_scan_is_not_stale(csv_paths):
    results.write({'long': make_scores(["AAA"])}, {}, horizon=5, paths=25, model="m", top_n=30)
    assert results.age_hours() < config.SCAN_STALE_HOURS


def test_skipped_file_is_removed_when_nothing_is_skipped(csv_paths):
    _top, _short, data = csv_paths
    results.write_skipped([("AAA", "only 12 bars cached")])
    assert config.SKIPPED_CSV.exists()

    results.write_skipped([])
    assert not config.SKIPPED_CSV.exists()


# --- both sides ---------------------------------------------------------------
#
# Both files are written on every scan whichever side was asked for: the two
# rankings come off the same sampled paths, so producing both is free and
# discarding one would mean a full re-scan to see it.


def short_scores(symbols):
    return [
        Score(
            s, 10.0, 3.0 - i, "SHORT", -0.03, 0.2, 0.02, -0.05, -0.06, -1.5,
            side="short", p_down=0.8, q95=0.01, runup=0.04,
        )
        for i, s in enumerate(symbols)
    ]


def test_both_files_are_written_even_when_one_side_is_empty(csv_paths):
    top, short, _ = csv_paths
    results.write({"long": make_scores(["AAA"]), "short": []}, {},
                  horizon=5, paths=25, model="m", top_n=30)

    assert len(read_rows(top)) == 1
    assert short.exists(), "the short file must exist even with no candidates"
    assert read_rows(short) == []


def test_each_side_gets_its_own_risk_columns(csv_paths):
    top, short, _ = csv_paths
    results.write({"long": make_scores(["AAA"]), "short": short_scores(["BBB"])}, {},
                  horizon=5, paths=25, model="m", top_n=30)

    long_row = read_rows(top)[0]
    short_row = read_rows(short)[0]

    # A long is judged on the downside; a short on the upside.
    assert {"p_up", "q05_pct", "mdd_pct"} <= long_row.keys()
    assert not {"p_down", "q95_pct", "runup_pct"} & long_row.keys()
    assert {"p_down", "q95_pct", "runup_pct"} <= short_row.keys()
    assert not {"p_up", "q05_pct", "mdd_pct"} & short_row.keys()


def test_short_rows_carry_the_mirrored_statistics(csv_paths):
    _top, short, _ = csv_paths
    results.write({"long": [], "short": short_scores(["BBB"])}, {},
                  horizon=5, paths=25, model="m", top_n=30)

    row = read_rows(short)[0]
    assert row["signal"] == "SHORT"
    assert float(row["p_down"]) == pytest.approx(0.8)
    assert float(row["q95_pct"]) == pytest.approx(1.0)
    assert float(row["runup_pct"]) == pytest.approx(4.0)


def test_the_archive_keeps_both_sides_and_every_column(csv_paths):
    _top, _short, data = csv_paths
    _w, archive = results.write(
        {"long": make_scores(["AAA", "BBB"]), "short": short_scores(["CCC"])}, {},
        horizon=5, paths=25, model="m", top_n=30,
    )

    rows = read_rows(archive)
    assert len(rows) == 3
    assert {r["side"] for r in rows} == {"long", "short"}
    # Nothing computed is discarded, so either ranking can be re-derived.
    assert {"p_up", "q05_pct", "mdd_pct", "p_down", "q95_pct", "runup_pct"} <= rows[0].keys()
    assert archive.parent == data


def test_each_side_is_overwritten_independently(csv_paths):
    top, short, _ = csv_paths
    results.write({"long": make_scores(["A", "B", "C"]), "short": short_scores(["X", "Y"])},
                  {}, horizon=5, paths=25, model="m", top_n=30)
    results.write({"long": make_scores(["D"]), "short": short_scores(["Z"])},
                  {}, horizon=5, paths=25, model="m", top_n=30)

    assert [r["symbol"] for r in read_rows(top)] == ["D"]
    assert [r["symbol"] for r in read_rows(short)] == ["Z"]


def test_age_can_be_read_per_side(csv_paths):
    results.write({"long": make_scores(["AAA"]), "short": short_scores(["BBB"])}, {},
                  horizon=5, paths=25, model="m", top_n=30)
    assert results.age_hours("long") < config.SCAN_STALE_HOURS
    assert results.age_hours("short") < config.SCAN_STALE_HOURS
