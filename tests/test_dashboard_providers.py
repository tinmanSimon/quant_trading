"""Provider menus route requests and local reads without mixing vendors."""
from datetime import UTC, date, datetime

import polars as pl
from streamlit.testing.v1 import AppTest

from data_pipeline import DataRequest
from data_pipeline.providers import BaseDataProvider, ProviderRegistry, YFinanceProvider
from research import Research


class OtherProvider(BaseDataProvider):
    supported_timeframes = frozenset({'1d'})

    def fetch(self, request):
        self.requests.append(request)
        return daily_frame(request.symbol)

    def __init__(self):
        self.requests = []


def daily_frame(symbol, offset=0):
    days = (2, 3, 4, 5, 8, 9)
    return pl.DataFrame({
        'timestamp': [datetime(2024, 1, day, tzinfo=UTC) for day in days],
        'symbol': [symbol] * len(days),
        **{column: [offset + base + i for i in range(len(days))]
           for column, base in [('open', 100.), ('high', 102.), ('low', 99.), ('close', 101.)]},
        'volume': [1000.] * len(days),
    })


def open_app(monkeypatch, research):
    import dashboard.app as dashboard
    monkeypatch.setattr(dashboard, 'Research', lambda **kwargs: research)
    return AppTest.from_string('from dashboard.app import main\nmain()').run()


def test_registry_names_do_not_construct_factories():
    def broken():
        raise AssertionError('Must not construct a provider to list it')
    registry = ProviderRegistry({' Z ': broken, 'a': broken})
    assert registry.names() == ('a', 'z')


def test_fetch_menu_uses_selected_capabilities_and_preserves_symbol_case(tmp_path, monkeypatch):
    other = OtherProvider()
    registry = ProviderRegistry({'yahoo': YFinanceProvider, 'other': other})
    research = Research(tmp_path / 'data', tmp_path / 'runs', providers=registry)
    app = open_app(monkeypatch, research)
    app.sidebar.radio[0].set_value('Fetch').run()
    assert app.selectbox(key='fetch-provider').options == ['other', 'yahoo']
    assert app.selectbox(key='fetch-timeframe').options == ['1d']
    assert not app.checkbox
    app.text_area[0].set_value('aBc')
    app.date_input(key='fetch-start').set_value(date(2024, 1, 2))
    app.date_input(key='fetch-end').set_value(date(2024, 1, 10))
    next(b for b in app.button if b.label == 'Fetch and save').click().run()
    assert not app.exception and not app.error
    assert [(r.provider, r.symbol) for r in other.requests] == [('other', 'aBc')]
    assert research.pipeline.list_datasets()[0].request.provider == 'other'
    app.selectbox(key='fetch-provider').set_value('yahoo').run()
    assert '1h' in app.selectbox(key='fetch-timeframe').options
    assert len(app.checkbox) == 1
    assert 'fetch-report' not in app.session_state


def test_browse_and_backtest_select_exact_local_provider(tmp_path, monkeypatch):
    research = Research(tmp_path / 'data', tmp_path / 'runs')
    for provider, offset in [('other', 1000), ('yahoo', 0)]:
        research.pipeline.ingest_frame(DataRequest(
            symbol='AAPL', provider=provider, timeframe='1d',
            start=datetime(2024, 1, 2, tzinfo=UTC), end=datetime(2024, 1, 10, tzinfo=UTC)),
            daily_frame('AAPL', offset))
    app = open_app(monkeypatch, research)
    assert app.selectbox(key='browse-provider').options == ['other', 'yahoo']
    # Inspect the actual rows rendered alongside the chart, not just menu labels.
    rows = [item.value for item in app.dataframe if 'close' in item.value.columns]
    assert rows[0]['close'].iloc[0] == 1101.
    app.selectbox(key='browse-provider').set_value('yahoo').run()
    rows = [item.value for item in app.dataframe if 'close' in item.value.columns]
    assert rows[0]['close'].iloc[0] == 101.
    app.sidebar.radio[0].set_value('Backtest').run()
    app.selectbox(key='backtest-provider').set_value('other').run()
    app.date_input(key='backtest-start').set_value(date(2024, 1, 5))
    app.date_input(key='backtest-end').set_value(date(2024, 1, 10))
    next(n for n in app.number_input if n.label == 'Fast moving average (bars)').set_value(1)
    next(n for n in app.number_input if n.label == 'Slow moving average (bars)').set_value(2)
    next(b for b in app.button if b.label == 'Validate all data and run').click().run()
    assert not app.exception and not app.error
    run = app.session_state['backtest-run']
    assert run.manifest['provider'] == 'other'
    app.selectbox(key='backtest-provider').set_value('yahoo').run()
    assert 'backtest-run' not in app.session_state


def test_empty_registry_has_no_fetch_form(tmp_path, monkeypatch):
    research = Research(tmp_path / 'data', tmp_path / 'runs', providers=ProviderRegistry())
    app = open_app(monkeypatch, research)
    app.sidebar.radio[0].set_value('Fetch').run()
    assert not app.exception and not app.error
    assert not app.text_area
    assert any('No providers' in info.value for info in app.info)


def test_provider_without_declared_intervals_cannot_submit_fetch(tmp_path, monkeypatch):
    other = OtherProvider()
    other.supported_timeframes = frozenset()
    research = Research(tmp_path / 'data', tmp_path / 'runs',
                        providers=ProviderRegistry({'other': other}))
    app = open_app(monkeypatch, research)
    app.sidebar.radio[0].set_value('Fetch').run()
    assert not app.exception and not app.error
    assert any('no fetch intervals' in warning.value for warning in app.warning)
    assert not app.text_area
    assert other.requests == []
