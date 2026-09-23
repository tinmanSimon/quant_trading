"""Provider menus route requests and local reads without mixing vendors."""
from datetime import UTC, date, datetime

import polars as pl
from polars.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest

from data_pipeline import DataQuery, DataRequest, FetchQuality, FetchResult
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


def test_massive_is_discovered_without_credentials_or_network(tmp_path, monkeypatch):
    from data_pipeline.providers import MassiveProvider

    monkeypatch.delenv('MASSIVE_API_KEY', raising=False)
    research = Research(tmp_path / 'data', tmp_path / 'runs')
    assert 'massive' in research.pipeline.providers.names()
    assert isinstance(research.pipeline.providers.get('massive'), MassiveProvider)
    app = open_app(monkeypatch, research)
    app.sidebar.radio[0].set_value('Fetch').run()
    app.selectbox(key='fetch-provider').set_value('massive').run()
    assert not app.exception and not app.error
    assert {'massive', 'yahoo'} <= set(app.selectbox(key='fetch-provider').options)
    assert set(app.selectbox(key='fetch-timeframe').options) == {
        '1m', '2m', '5m', '15m', '30m', '1h', '90m', '1d',
    }
    assert not app.checkbox  # The missing-OHLC option belongs to Yahoo only.
    assert app.number_input(key='fetch-massive-request-interval').value == 0.0
    assert any('Massive only' in caption.value and '13 seconds' in caption.value
               for caption in app.caption)
    app.selectbox(key='fetch-provider').set_value('yahoo').run()
    assert not any(field.key == 'fetch-massive-request-interval' for field in app.number_input)
    assert not any('Massive only' in caption.value for caption in app.caption)


def test_massive_dashboard_fetch_and_browse_keep_vendor_data_separate(tmp_path, monkeypatch):
    from data_pipeline.providers import MassiveProvider

    monkeypatch.delenv('MASSIVE_API_KEY', raising=False)
    research = Research(tmp_path / 'data', tmp_path / 'runs')
    start, end = datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 10, tzinfo=UTC)
    research.pipeline.ingest_frame(DataRequest('AAPL', start, end, provider='yahoo', timeframe='1d'),
                                   daily_frame('AAPL'))
    calls = []
    source = daily_frame('AAPL', offset=1000).with_columns(
        # Provider results use canonical session-date labels at UTC midnight.
        pl.col('timestamp').cast(pl.Datetime('ms', 'UTC')))

    def fetch_result(self, request):
        calls.append((request, self.request_interval_seconds))
        return FetchResult(source, FetchQuality(status='reported'))

    monkeypatch.setattr(MassiveProvider, 'fetch_result', fetch_result)
    app = open_app(monkeypatch, research)
    app.sidebar.radio[0].set_value('Fetch').run()
    app.selectbox(key='fetch-provider').set_value('massive').run()
    app.selectbox(key='fetch-timeframe').set_value('1d').run()
    app.number_input(key='fetch-massive-request-interval').set_value(13.0)
    app.text_area[0].set_value('AAPL')
    app.date_input(key='fetch-start').set_value(start.date())
    app.date_input(key='fetch-end').set_value(end.date())
    next(b for b in app.button if b.label == 'Fetch and save').click().run()
    assert not app.exception and not app.error
    assert [(r.provider, r.symbol, r.timeframe, interval) for r, interval in calls] == [
        ('massive', 'AAPL', '1d', 13.0),
    ]
    assert research.pipeline.providers.get('massive').request_interval_seconds == 0.0
    report = app.session_state['fetch-report']
    assert report.ok
    massive_id = report.outcomes[0].dataset_ids[0]
    assert_frame_equal(research.pipeline.read_dataset(massive_id), source)
    assert research.pipeline.read(DataQuery(provider='yahoo', symbol='AAPL'))['close'][0] == 101.

    app.sidebar.radio[0].set_value('Market data').run()
    app.selectbox(key='browse-provider').set_value('massive').run()
    rows = [item.value for item in app.dataframe if 'close' in item.value.columns]
    assert rows[0]['close'].iloc[0] == 1101.
    app.selectbox(key='browse-provider').set_value('yahoo').run()
    rows = [item.value for item in app.dataframe if 'close' in item.value.columns]
    assert rows[0]['close'].iloc[0] == 101.

    app.sidebar.radio[0].set_value('Backtest').run()
    app.selectbox(key='backtest-provider').set_value('massive').run()
    assert not app.exception and not app.error
    assert app.multiselect[0].options == ['AAPL']
    assert len(calls) == 1  # Local-data pages do not fetch again.
    assert_frame_equal(research.pipeline.read_dataset(massive_id), source)
