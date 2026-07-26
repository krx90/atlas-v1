"""Credential loading: the paper-only guard and the no-silent-fallback rule."""

from __future__ import annotations

import pytest

from atlas import config
from atlas.config import ConfigError, load_credentials

PAPER = "APCA_API_KEY_ID=PK123\nAPCA_API_SECRET_KEY=abc\nAPCA_API_PAPER=true\n"


@pytest.fixture
def creds_file(tmp_path, monkeypatch):
    path = tmp_path / "creds.env"
    monkeypatch.setattr(config, "CREDS_FILE", path)
    return path


def test_valid_paper_credentials(creds_file):
    creds_file.write_text(PAPER)
    creds_file.chmod(0o600)

    creds, warning = load_credentials()
    assert creds.key_id == "PK123"
    assert creds.secret_key == "abc"
    assert creds.paper is True
    assert warning is None


def test_missing_file_names_the_path_and_the_fix(creds_file):
    with pytest.raises(ConfigError) as exc:
        load_credentials()
    assert "creds.env" in str(exc.value)
    assert "cp creds.env.example" in str(exc.value)


def test_missing_key_is_reported_by_name(creds_file):
    creds_file.write_text("APCA_API_KEY_ID=PK123\n")
    with pytest.raises(ConfigError, match="APCA_API_SECRET_KEY"):
        load_credentials()


def test_ambient_environment_cannot_substitute_for_the_file(creds_file, monkeypatch):
    """A shell variable must not stand in for a missing creds file.

    Otherwise you cannot tell which account you are about to trade against.
    """
    monkeypatch.setenv("APCA_API_KEY_ID", "PKfromenv")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secretfromenv")
    with pytest.raises(ConfigError):
        load_credentials()


def test_live_trading_flag_is_refused(creds_file):
    creds_file.write_text("APCA_API_KEY_ID=PK123\nAPCA_API_SECRET_KEY=abc\nAPCA_API_PAPER=false\n")
    with pytest.raises(ConfigError, match="only trades paper accounts"):
        load_credentials()


def test_live_key_prefix_is_refused_even_when_the_flag_says_paper(creds_file):
    creds_file.write_text("APCA_API_KEY_ID=AK123\nAPCA_API_SECRET_KEY=abc\nAPCA_API_PAPER=true\n")
    with pytest.raises(ConfigError, match="live-trading key"):
        load_credentials()


def test_loose_permissions_produce_a_warning_not_a_failure(creds_file):
    creds_file.write_text(PAPER)
    creds_file.chmod(0o644)

    creds, warning = load_credentials()
    assert creds.key_id == "PK123"
    assert warning is not None
    assert "chmod 600" in warning


def test_whitespace_around_values_is_stripped(creds_file):
    creds_file.write_text("APCA_API_KEY_ID= PK123 \nAPCA_API_SECRET_KEY= abc \n")
    creds_file.chmod(0o600)

    creds, _ = load_credentials()
    assert creds.key_id == "PK123"
    assert creds.secret_key == "abc"
