"""Settings survive restarts without corrupting or replacing unreadable files."""

import json

import pytest

from dashboard.trend_settings import SCHEMA_VERSION, load_trend_settings, save_trend_settings
from trend_analysis import TrendConfig


def test_missing_settings_use_defaults_without_writing(tmp_path):
    path = tmp_path / "missing" / "trend_config.json"
    loaded = load_trend_settings(path)
    assert loaded.config == TrendConfig()
    assert loaded.warning is None
    assert not path.exists()


def test_round_trip_and_added_fields_receive_defaults(tmp_path):
    path = tmp_path / "local" / "trend_config.json"
    config = TrendConfig(max_history=480, clarity_threshold=0.75,
                         redundancy_percentage=12.5, redundancy_boundary_rule="or", redundancy_min_bars=9)
    save_trend_settings(config, path)
    assert load_trend_settings(path).config == config
    document = json.loads(path.read_text())
    assert document["schema_version"] == SCHEMA_VERSION
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "config": {"max_history": 400}}))
    loaded = load_trend_settings(path)
    assert loaded.config == TrendConfig(max_history=400)
    assert loaded.config.redundancy_percentage == 5.0
    assert loaded.config.redundancy_boundary_rule == "and"
    assert loaded.config.redundancy_min_bars == 5
    assert loaded.warning is None


def test_legacy_bar_setting_migrates_without_changing_other_settings_or_file(tmp_path):
    path = tmp_path / "trend_config.json"
    legacy = {"schema_version": SCHEMA_VERSION,
              "config": {"max_history": 480, "clarity_threshold": .75,
                         "redundancy_threshold_bars": 12, "dedup_ratio": 1.5}}
    payload = json.dumps(legacy)
    path.write_text(payload)
    loaded = load_trend_settings(path)
    assert loaded.config == TrendConfig(max_history=480, clarity_threshold=.75, dedup_ratio=1.5)
    assert "5%" in loaded.warning and "Both boundaries (AND)" in loaded.warning
    assert path.read_text() == payload
    save_trend_settings(loaded.config, path)
    saved = json.loads(path.read_text())["config"]
    assert "redundancy_threshold_bars" not in saved
    assert saved["redundancy_percentage"] == 5.0
    assert saved["redundancy_boundary_rule"] == "and"
    assert load_trend_settings(path).warning is None


def test_explicit_percentage_settings_take_precedence_over_retired_bar_setting(tmp_path):
    path = tmp_path / "trend_config.json"
    config = TrendConfig(redundancy_percentage=7.5, redundancy_boundary_rule="or")
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION,
                                "config": {**config.to_dict(), "redundancy_threshold_bars": 12}}))
    loaded = load_trend_settings(path)
    assert loaded.config == config
    assert "7.5%" in loaded.warning and "Either boundary (OR)" in loaded.warning


@pytest.mark.parametrize("payload", [
    "{broken", "[]", '{"schema_version":999,"config":{}}',
    '{"schema_version":true,"config":{}}',
    '{"schema_version":1,"config":{"clarity_threshold":2}}',
    '{"schema_version":1,"config":{"unknown_field":1}}',
    '{"schema_version":1,"config":[]}',
    '{"schema_version":1,"config":{"penalty_multiplier":NaN}}',
    '{"schema_version":1,"config":{"redundancy_percentage":101}}',
    '{"schema_version":1,"config":{"redundancy_boundary_rule":"xor"}}',
    '{"schema_version":1,"config":{"redundancy_min_bars":-1}}',
    '{"schema_version":1,"config":{"redundancy_threshold_bars":true}}',
])
def test_invalid_file_warns_and_remains_unchanged(tmp_path, payload):
    path = tmp_path / "trend_config.json"
    path.write_text(payload)
    loaded = load_trend_settings(path)
    assert loaded.config == TrendConfig()
    assert "saved file has not been changed" in loaded.warning
    assert path.read_text() == payload


def test_interrupted_replace_keeps_previous_valid_file(tmp_path, monkeypatch):
    from dashboard import trend_settings
    path = tmp_path / "trend_config.json"
    original = TrendConfig(clarity_threshold=0.8)
    save_trend_settings(original, path)

    def reject_replace(source, destination):
        # The staged file is a complete valid document before replacement.
        assert json.loads(source.read_text())["config"]["clarity_threshold"] == 0.9
        raise OSError("simulated interrupted replacement")

    monkeypatch.setattr(trend_settings.os, "replace", reject_replace)
    with pytest.raises(OSError, match="interrupted"):
        save_trend_settings(TrendConfig(clarity_threshold=0.9), path)
    assert load_trend_settings(path).config == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("value", [-1, 1.5, True, float("nan"), float("inf"), "5"])
def test_minimum_resolution_validated(value):
    with pytest.raises(ValueError, match="redundancy_min_bars"):
        TrendConfig(redundancy_min_bars=value)
