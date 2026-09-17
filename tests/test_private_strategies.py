"""Private strategies use normal registries without becoming repository assets."""

from datetime import UTC, date, datetime
import json
from pathlib import Path
import subprocess
import sys

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from data_pipeline import DataRequest
from research import Research
from research.strategies import PrivateStrategyError, StrategyRegistry, load_registry
from research.strategies import loader


STRATEGY_SOURCE = '''
from research.strategies import Strategy

class Personal(Strategy):
    name, version = "personal", "1"

    def __init__(self, window=2, target=0.5, enabled=True, mode="long", note="", details=None):
        self.lookback = window
        self.options = dict(window=window, target=target, enabled=enabled, mode=mode, note=note)
        if details is not None:
            self.options["details"] = details

    @property
    def config(self):
        return dict(self.options)

    def target_weight(self, history):
        return self.options["target"] if self.options["enabled"] else 0.0
'''

REGISTER_SOURCE = '''
from .implementation import Personal

def register_strategies(registry):
    registry.register("personal", "1", lambda config: Personal(**config),
        label="Personal test", default_config={"window": 2, "target": 0.5, "enabled": True, "mode": "long", "note": ""},
        parameters={
            "window": {"type": "integer", "minimum": 1, "label": "History bars"},
            "target": {"type": "number", "minimum": 0, "maximum": 1, "label": "Allocation"},
            "enabled": {"type": "boolean", "label": "Enabled"},
            "mode": {"type": "string", "choices": ["long", "cash"], "label": "Mode"},
            "note": {"type": "string", "label": "Note"},
        })
'''


@pytest.fixture
def private_project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    package = root / "private_strategies"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "implementation.py").write_text(STRATEGY_SOURCE, encoding="utf-8")
    (package / "register.py").write_text(REGISTER_SOURCE, encoding="utf-8")
    monkeypatch.setattr(loader, "project_root", lambda: root)
    return root


def spec(config=None):
    return {"name": "personal", "version": "1", "config": config or {"window": 2, "target": 0.5}}


def seed(research):
    days = (2, 3, 4, 5, 8, 9)
    frame = pl.DataFrame({"timestamp": [datetime(2024, 1, day, tzinfo=UTC) for day in days],
                          "symbol": ["AAPL"] * 6, "open": [100.] * 6, "high": [102.] * 6,
                          "low": [99.] * 6, "close": [101.] * 6, "volume": [1000.] * 6})
    research.pipeline.ingest_frame(DataRequest(symbol="AAPL", timeframe="1d",
        start=datetime(2024, 1, 2, tzinfo=UTC), end=datetime(2024, 1, 10, tzinfo=UTC)), frame)


def test_absent_private_package_keeps_builtins(tmp_path):
    assert [item.name for item in load_registry(project_dir=tmp_path).definitions()] == [
        "moving_average_cross", "momentum", "weighted"]


def test_private_relative_import_discovery_independent_of_cwd(private_project, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = list(sys.path)
    registry = load_registry()
    assert registry.load(spec()).lookback == 2
    assert next(item for item in registry.definitions() if item.name == "personal").label == "Personal test"
    assert sys.path == before


def test_import_once_but_registries_are_independent(private_project):
    package = private_project / "private_strategies"
    (package / "__init__.py").write_text('from pathlib import Path\nwith Path(__file__).with_name("imports.txt").open("a") as f:\n    f.write("imported\\n")\n')
    first = load_registry()
    # Source edits intentionally require restarting the process.
    (package / "register.py").write_text('raise RuntimeError("edited after import")\n')
    second = load_registry()
    assert first is not second
    assert first.load(spec()).config == second.load(spec()).config
    assert (package / "imports.txt").read_text() == "imported\n"


@pytest.mark.parametrize("source,message", [
    ('import nonexistent_private_dependency\n', "nonexistent_private_dependency"),
    ('def register_strategies(registry):\n    raise RuntimeError("broken hook")\n', "broken hook"),
    ('value = 1\n', "register_strategies"),
    ('def register_strategies(registry):\n    registry.register("momentum", "1", lambda config: None)\n', "already registered"),
])
def test_present_broken_private_package_is_not_silently_ignored(private_project, source, message):
    (private_project / "private_strategies/register.py").write_text(source)
    with pytest.raises(PrivateStrategyError, match=message):
        load_registry()


def test_missing_registration_file_is_an_error(private_project):
    (private_project / "private_strategies/register.py").unlink()
    with pytest.raises(PrivateStrategyError, match="register.py"):
        load_registry()


def test_failed_hook_cannot_leave_partial_registry(private_project):
    path = private_project / "private_strategies/register.py"
    path.write_text(REGISTER_SOURCE + '\noriginal = register_strategies\ndef register_strategies(registry):\n    original(registry)\n    raise RuntimeError("fail after registering")\n')
    for _ in range(2):
        with pytest.raises(PrivateStrategyError, match="fail after registering"):
            load_registry()


def test_different_roots_do_not_share_private_modules(private_project, tmp_path):
    other = tmp_path / "other"
    package = other / "private_strategies"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "implementation.py").write_text(STRATEGY_SOURCE.replace('"1"', '"2"', 1))
    (package / "register.py").write_text(REGISTER_SOURCE.replace('"personal", "1"', '"personal", "2"'))
    assert load_registry().load(spec()).version == "1"
    assert load_registry(project_dir=other).load(spec() | {"version": "2"}).version == "2"


def test_metadata_is_detached_and_config_bounds_apply(private_project):
    registry = load_registry()
    definition = next(item for item in registry.definitions() if item.name == "personal")
    definition.default_config["window"] = 999
    definition.parameters["window"]["minimum"] = 999
    actual = next(item for item in registry.definitions() if item.name == "personal")
    assert actual.default_config["window"] == 2
    with pytest.raises(ValueError, match="at least"):
        registry.load(spec({"window": 0}))
    with pytest.raises(ValueError, match="choices"):
        registry.load(spec({"mode": "invalid"}))
    with pytest.raises(ValueError, match="integer"):
        registry.load(spec({"window": True}))


@pytest.mark.parametrize("kwargs", [
    {"default_config": {"x": float("nan")}},
    {"default_config": {"x": 1}, "parameters": {"x": {"type": "unknown"}}},
    {"parameters": {"x": {"type": "integer"}}},
    {"default_config": {"x": 1}, "parameters": {"x": {"type": "integer", "minimum": 2}}},
    {"default_config": {"x": 1}, "parameters": {"x": {"type": "integer", "step": 0}}},
    {"default_config": {"x": True}, "parameters": {"x": {"type": "integer"}}},
    {"default_config": {"x": 1}, "parameters": {"x": {"type": "integer", "choices": []}}},
])
def test_invalid_metadata_never_registers_a_partial_entry(kwargs):
    registry = StrategyRegistry()
    with pytest.raises(ValueError):
        registry.register("example", "1", lambda config: None, **kwargs)
    assert registry.definitions() == ()


def test_weighted_strategy_can_construct_private_children(private_project):
    registry = load_registry()
    composite = registry.load({"name": "weighted", "version": "1", "config": {
        "strategies": [spec({"target": 1.}), spec({"target": 0.})], "weights": [0.3, 0.7]}})
    assert composite.target_weight(pl.DataFrame({"close": [1., 2.]})) == 0.3


def test_real_cli_loads_private_strategy_and_saved_run_needs_no_import(private_project, tmp_path, monkeypatch, capsys):
    from research.cli import main
    import research.api as api

    app = Research(tmp_path / "data", tmp_path / "runs")
    seed(app)
    config = tmp_path / "strategies.json"
    config.write_text(json.dumps([spec()]))
    assert main(["--data-dir", str(app.pipeline.store.data_dir), "--runs-dir", str(app.runs_dir),
                 "backtest", "--tickers", "AAPL", "--start", "2024-01-05", "--end", "2024-01-10",
                 "--strategies", str(config)]) == 0
    assert "Saved run:" in capsys.readouterr().out

    def unavailable():
        raise AssertionError("Saved results must not import private code")

    monkeypatch.setattr(api, "load_registry", unavailable)
    fresh = Research(app.pipeline.store.data_dir, app.runs_dir)
    saved = fresh.load_run(fresh.list_runs()[0]["run_id"])
    assert saved.results[0].strategy_spec["name"] == "personal"
    assert saved.results[0].trades.height > 0
    assert all(path.name in {"manifest.json", "manifest.sha256"} or path.suffix == ".parquet" for path in saved.path.iterdir())


def test_dashboard_discovers_configures_combines_and_runs_private_strategy(private_project, tmp_path, monkeypatch):
    import dashboard.app as dashboard
    research = Research(tmp_path / "data", tmp_path / "runs")
    seed(research)
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    view = AppTest.from_string("from dashboard.app import main\nmain()").run()
    view.sidebar.radio[0].set_value("Backtest").run()
    selection = next(item for item in view.multiselect if item.label == "Strategies")
    assert any("Personal test" in option for option in selection.options)
    selection.set_value([("personal", "1"), ("momentum", "1")]).run()
    next(item for item in view.number_input if item.label == "Momentum lookback (bars)").set_value(1)
    next(item for item in view.number_input if item.label == "Allocation").set_value(0.25)
    next(item for item in view.text_input if item.label == "Note").set_value("custom parameters")
    next(item for item in view.checkbox if item.label == "Also test a weighted combination").check().run()
    next(item for item in view.date_input if item.label.startswith("Start date")).set_value(date(2024, 1, 5))
    next(item for item in view.date_input if item.label.startswith("End date")).set_value(date(2024, 1, 10))
    next(item for item in view.button if item.label == "Validate all data and run").click().run(timeout=20)
    assert not view.exception
    assert not view.error
    run = research.load_run(research.list_runs()[0]["run_id"])
    assert len(run.results) == 3
    personal = next(item for item in run.results if item.strategy_spec["name"] == "personal")
    assert personal.strategy_spec["config"]["target"] == 0.25
    assert personal.strategy_spec["config"]["note"] == "custom parameters"
    assert run.results[-1].strategy_spec["config"]["weights"] == [0.5, 0.5]


def test_complex_json_controls_preserve_configuration_and_block_invalid_input(private_project):
    path = private_project / "private_strategies/register.py"
    path.write_text('from .implementation import Personal\ndef register_strategies(registry):\n'
                    '    registry.register("personal", "1", lambda config: Personal(**config), '
                    'default_config={"details": {"levels": [1, 2]}})\n')
    view = AppTest.from_string('from research.strategies import load_registry\n'
        'from dashboard.strategy_controls import strategy_specs\nimport streamlit as st\n'
        'specs = strategy_specs(load_registry())\nst.session_state["specs"] = specs\n').run()
    view.multiselect[0].set_value([("personal", "1")]).run()
    assert json.loads(view.text_area[0].value) == {"details": {"levels": [1, 2]}}
    view.text_area[0].set_value('{"details": {"levels": [3, 4]}, "window": 2}').run()
    assert view.session_state["specs"][0]["config"]["details"] == {"levels": [3, 4]}
    view.text_area[0].set_value("not json").run()
    assert view.error
    assert view.session_state["specs"] is None
    assert not view.exception


def test_dashboard_reports_private_import_error(private_project, tmp_path, monkeypatch):
    import dashboard.app as dashboard
    (private_project / "private_strategies/register.py").write_text('raise RuntimeError("private dependency failed")\n')
    research = Research(tmp_path / "data", tmp_path / "runs")
    monkeypatch.setattr(dashboard, "Research", lambda **kwargs: research)
    view = AppTest.from_string("from dashboard.app import main\nmain()").run()
    assert not view.error  # Browsing data never imports strategy implementations.
    view.sidebar.radio[0].set_value("Backtest").run()
    assert "private dependency failed" in view.error[0].value
    assert research.list_runs() == []


def test_private_sources_are_ignored_and_untracked():
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(["git", "check-ignore", "private_strategies/anything.py"],
                            cwd=repository, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "private_strategies/anything.py"
    tracked = subprocess.run(["git", "ls-files", "private_strategies"], cwd=repository,
                             capture_output=True, text=True, check=True)
    assert tracked.stdout == ""
