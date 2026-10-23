"""Offline reference-data normalization, immutable storage and coverage reporting."""
from dataclasses import replace
from datetime import UTC, date, datetime
import json
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from data_pipeline.corporate_actions import ActionCoverage, ActionSnapshot, CorporateAction, SnapshotStore, other_events_report
from data_pipeline.corporate_actions.providers import DatabentoActionProvider, import_snapshot
from data_pipeline.corporate_actions.providers.databento import normalize_records

START = date(2020, 1, 1)
END = date(2020, 12, 31)
FETCHED = datetime(2021, 1, 10, tzinfo=UTC)
NY = ZoneInfo('America/New_York')


def action(**kwargs):
    return CorporateAction(**dict(event_id='split-1', symbol='TSLA', event_type='forward_split',
        effective_at=datetime(2020, 8, 31, tzinfo=NY), share_ratio=5, security_id='123', **kwargs))


def snapshot(events=None):
    return ActionSnapshot('verified-test', END, FETCHED,
        (ActionCoverage('TSLA', START, END, source_basis='unadjusted', security_id='123'),),
        (action(),) if events is None else events)


def row(**kwargs):
    fields = dict(event_unique_id='record-1', event_id='event-1', symbol='TSLA', security_id='123',
        listing_id='listing-1', exchange='USNASD', primary_exchange='USNASD', trading_currency='USD',
        ts_record='2020-08-10T00:00:00+00:00', event_action='I', event='FSPLT', event_subtype='SD',
        ex_date='2020-08-31', event_date='2020-08-31', ratio_old=1, ratio_new=5,
        rate_info={}, event_info={}, payment_type='S')
    fields.update(kwargs)
    return fields


def normalize(rows, factors=()):
    return normalize_records(rows, factors, symbol='TSLA', security_id='123', listing_id='listing-1',
        exchange='USNASD', start=START, reference_date=END, fetched_at=FETCHED)


def dividend(**kwargs):
    return row(**dict(dict(event='DIV', event_subtype=None, payment_type='C', rate_currency='USD',
               rate_info={'gross_dividend': 2}), **kwargs))


def factor(**kwargs):
    fields = dict(event_id='event-1', security_id='123', exchange='USNASD', ex_date='2020-08-31',
        ts_created='2020-08-31T20:00:00+00:00', reason=17, dividend_currency='USD',
        option=1, factor=.98, status='A')
    fields.update(kwargs)
    return fields


def test_snapshot_roundtrip_checksum_order_and_immutability(tmp_path):
    snap = snapshot()
    assert ActionSnapshot.from_dict(snap.to_dict()) == snap
    assert len(snap.checksum) == 64
    with pytest.raises(AttributeError):
        snap.source = 'changed'
    store = SnapshotStore(tmp_path)
    path = store.save(snap)
    before = path.stat().st_mtime_ns
    assert store.save(snap) == path
    assert path.stat().st_mtime_ns == before
    assert store.load(path) == snap
    data = json.loads(path.read_text())
    data['snapshot']['events'][0]['share_ratio'] = 10
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='checksum'):
        store.load(path)
    with pytest.raises(ValueError, match='checksum'):
        store.save(snap)


def test_import_plain_and_checked_envelope():
    snap = snapshot()
    assert import_snapshot(json.dumps(snap.to_dict()).encode()) == snap
    assert import_snapshot({'snapshot': snap.to_dict(), 'checksum': snap.checksum}) == snap
    with pytest.raises(ValueError, match='checksum'):
        import_snapshot({'snapshot': snap.to_dict(), 'checksum': 'bad'})
    with pytest.raises(ValueError, match='Invalid'):
        import_snapshot({'source': 'incomplete'})


def test_legacy_snapshot_checksum_is_unchanged_with_optional_coverage_scope():
    snap = snapshot()
    # Recorded before introducing other_events_status: existing stored snapshots
    # must still validate, and processor identities must not change on load.
    assert snap.checksum == 'b2bd322fe232fcc9bec17d8a295c6c5099ce5bd182db950c853ae6fa638bd16f'
    assert 'other_events_status' not in snap.to_dict()['coverage'][0]
    restored = import_snapshot({'snapshot': snap.to_dict(), 'checksum': snap.checksum})
    assert restored.coverage[0].other_events_status is None
    assert restored.checksum == snap.checksum


def test_scoped_coverage_roundtrip_and_checksum():
    legacy = snapshot()
    snap = replace(legacy, coverage=(replace(legacy.coverage[0], other_events_status='unknown'),))
    assert snap.to_dict()['coverage'][0]['other_events_status'] == 'unknown'
    assert ActionSnapshot.from_dict(snap.to_dict()) == snap
    assert snap.checksum != legacy.checksum
    with pytest.raises(ValueError, match='other-events coverage'):
        replace(snap.coverage[0], other_events_status='invalid')


@pytest.mark.parametrize('has_known_event', [False, True])
def test_report_distinguishes_supported_coverage_from_unknown_other_events(has_known_event):
    snap = snapshot()
    events = (replace(action(), event_id='spin', event_type='spin_off'),) if has_known_event else ()
    snap = replace(snap, events=events,
                   coverage=(replace(snap.coverage[0], other_events_status='unknown'),))
    rows = other_events_report(snap, ['TSLA'])
    assert rows[-1]['event_type'] == 'coverage_unknown'
    assert all(row['coverage'] == 'unknown' for row in rows)
    assert all(row['supported_events_coverage'] == 'complete' for row in rows)
    assert not any(row['event_type'] == 'no_other_events' for row in rows)
    if has_known_event:
        assert rows[0]['event_type'] == 'spin_off'


def test_model_validation_and_revision_duplicates():
    with pytest.raises(ValueError, match='timezone'):
        replace(action(), effective_at=datetime(2020, 8, 31))
    with pytest.raises(ValueError, match='positive'):
        replace(action(), price_factor=float('nan'))
    with pytest.raises(ValueError, match='Duplicate event'):
        snapshot((action(), action()))
    with pytest.raises(ValueError, match='Duplicate coverage'):
        replace(snapshot(), coverage=snapshot().coverage * 2)


def test_report_covers_missing_and_failed_tickers_and_excludes_supported_events():
    snap = replace(snapshot(), events=(action(), replace(action(), event_id='other', event_type='spin_off')),
        coverage=snapshot().coverage + (ActionCoverage('BAD', START, END, 'error', 'Lookup unavailable'),
                                      ActionCoverage('CLEAN', START, END)))
    report = other_events_report(snap, ['TSLA', 'BAD', 'CLEAN', 'MISSING', 'tsla'])
    assert len(report) == 4
    assert [r['event_type'] for r in report] == ['spin_off', 'coverage_unknown', 'no_other_events', 'coverage_unknown']
    assert report[0]['effective_date'] == '2020-08-31'
    assert report[-1]['coverage'] == 'unknown'


def test_latest_revision_retains_nulls_and_cancellation():
    original = row()
    updated = row(ts_record='2020-08-20T00:00:00+00:00', event_action='U', ratio_new=3)
    events = normalize([updated, original, updated])
    assert events[0].share_ratio == 3
    cancelled = row(ts_record='2020-09-01T00:00:00+00:00', event_action='C', ratio_new=None)
    event = normalize([original, cancelled])[0]
    assert event.status == 'cancelled'
    assert event.share_ratio is None
    assert '2020-09-01' in event.details


def test_split_effective_midnight_and_split_ratios():
    event = normalize([row()])[0]
    assert event.price_factor == .2
    assert event.effective_at.isoformat() == '2020-08-31T00:00:00-04:00'
    reverse = normalize([row(event='RSPLT', ratio_old=10, ratio_new=1)])[0]
    assert reverse.event_type == 'reverse_split'
    assert reverse.share_ratio == .1
    assert reverse.price_factor == 10


def test_cash_special_dividend_factor_and_capital_distribution_classification():
    event = normalize([dividend(event_info={'marker': 'SPL'})], [factor()])[0]
    assert event.event_type == 'special_cash_dividend'
    assert event.price_factor == .98 and event.cash_amount == 2
    assert event.share_ratio is None
    assert normalize([dividend(event_subtype='CAPDIST')])[0].event_type == 'return_of_capital'


@pytest.mark.parametrize('changes', [{'factor': 1}, {'factor': -1}, {'reason': 10}, {'dividend_currency': 'EUR'}])
def test_bad_cash_factors_rejected(changes):
    with pytest.raises(ValueError):
        normalize([dividend()], [factor(**changes)])


def test_pending_cancelled_and_reissued_factor():
    pending = normalize([dividend()], [factor(status='P')])[0]
    assert pending.status == 'pending' and pending.price_factor is None
    cancelled = normalize([dividend()], [factor(status='R')])[0]
    assert cancelled.status == 'pending'
    factors = [factor(), factor(status='R', ts_created='2020-09-01T00:00:00+00:00'),
               factor(factor=.97, ts_created='2020-09-01T00:01:00+00:00')]
    assert normalize([dividend()], factors)[0].price_factor == .97


def test_elections_and_conflicts_rejected():
    with pytest.raises(ValueError, match='multiple benefits'):
        normalize([row(), row(event_unique_id='alternative', option_id='2')])
    with pytest.raises(ValueError, match='multiple factor options'):
        normalize([dividend()], [factor(), factor(option=2)])
    with pytest.raises(ValueError, match='Conflicting'):
        normalize([row(), row(ratio_new=10)])
    with pytest.raises(ValueError, match='Ambiguous'):
        normalize([dividend()], [factor(), factor(factor=.97)])


def test_missing_factor_cannot_be_fabricated_from_adjustment_close():
    with pytest.raises(ValueError, match='Missing verified'):
        normalize([dividend()])


def test_unrelated_listings_and_future_revisions_do_not_apply():
    events = normalize([row(), row(listing_id='secondary', ratio_new=100),
                        row(ts_record='2022-01-01T00:00:00+00:00', ratio_new=10)])
    assert events[0].share_ratio == 5


def fake_client(corporate=None):
    return SimpleNamespace(
        security_master=SimpleNamespace(get_range=Mock(return_value=[row()])),
        corporate_actions=SimpleNamespace(get_range=Mock(return_value=[] if corporate is None else corporate)),
        adjustment_factors=SimpleNamespace(get_range=Mock(return_value=[])))


def test_fetch_explicit_sdk_calls_and_empty_verified_coverage():
    client = fake_client()
    snap = DatabentoActionProvider(client=client).fetch(['TSLA'], START, END, source_basis='unadjusted')
    assert snap.coverage[0].status == 'complete'
    assert snap.coverage[0].security_id == '123'
    assert snap.events == ()
    assert client.corporate_actions.get_range.call_count == 2
    kwargs = client.corporate_actions.get_range.call_args.kwargs
    assert kwargs['pit'] is True and kwargs['flatten'] is False
    assert kwargs['end'] == date(2021, 1, 1)
    assert 'events' not in kwargs


def test_fetch_reports_per_ticker_errors_without_sdk_exception_secrets():
    client = fake_client()
    client.corporate_actions.get_range.side_effect = ValueError('secret-key')
    snap = DatabentoActionProvider(client=client).fetch(['TSLA', 'BAD'], START, END)
    assert all(c.status == 'error' for c in snap.coverage)
    assert 'secret-key' not in json.dumps(snap.to_dict())
    assert len(other_events_report(snap, ['TSLA', 'BAD'])) == 2


def test_pre_dataset_history_is_not_claimed_complete():
    client = fake_client()
    snap = DatabentoActionProvider(client=client).fetch(['TSLA'], date(2017, 1, 1), END)
    assert snap.coverage[0].status == 'partial'
    client.corporate_actions.get_range.assert_not_called()


def test_missing_credentials_fail_before_sdk_import(monkeypatch):
    monkeypatch.delenv('DATABENTO_API_KEY', raising=False)
    with pytest.raises(ValueError, match='DATABENTO_API_KEY'):
        DatabentoActionProvider().fetch(['TSLA'], START, END)


def test_supported_events_enforce_exchange_midnight_including_dst():
    with pytest.raises(ValueError, match='midnight'):
        replace(action(), effective_at=datetime(2020, 8, 31, 9, 30, tzinfo=NY))
    assert replace(action(), effective_at=datetime(2020, 8, 31, 4, tzinfo=UTC))
    assert replace(action(), effective_at=datetime(2020, 12, 1, 5, tzinfo=UTC))
    with pytest.raises(ValueError, match='midnight'):
        replace(action(), effective_at=datetime(2020, 12, 1, 4, tzinfo=UTC))


def test_missing_factor_retains_other_events_for_batch_report():
    client = fake_client([dividend(), row(event_unique_id='spin', event_id='spin', event='DMRGR')])
    snap = DatabentoActionProvider(client=client).fetch(['TSLA'], START, END, source_basis='unadjusted')
    assert snap.coverage[0].status == 'complete'
    assert len(snap.events) == 2
    cash_event = next(e for e in snap.events if e.event_type == 'cash_dividend')
    assert cash_event.status == 'pending'
    assert 'Missing verified' in cash_event.details
    assert other_events_report(snap, ['TSLA'])[0]['event_type'] == 'spin_off'


def test_factor_entitlement_error_preserves_known_other_events():
    client = fake_client([dividend(), row(event_unique_id='spin', event_id='spin', event='DMRGR')])
    client.adjustment_factors.get_range.side_effect = ValueError('secret-key')
    snap = DatabentoActionProvider(client=client).fetch(['TSLA'], START, END)
    assert len(snap.events) == 2
    assert 'secret-key' not in json.dumps(snap.to_dict())
    assert other_events_report(snap, ['TSLA'])[0]['event_type'] == 'spin_off'


def test_subscription_failure_preserves_safe_diagnostic():
    from databento.common.error import BentoClientError
    from data_pipeline.corporate_actions.providers.databento import _request, ReferenceRequestError
    error = BentoClientError(403, json_body={'detail': {
        'case': 'license_reference_dataset_no_subscription',
        'message': 'secret-key', 'docs': 'https://secret-key',
        'payload': {'reference_dataset': 'security master'},
    }})
    api = SimpleNamespace(get_range=Mock(side_effect=error))
    with pytest.raises(ReferenceRequestError) as caught:
        _request(api)
    assert 'HTTP 403' in str(caught.value)
    assert 'subscription is required for the security master dataset' in str(caught.value)
    assert 'secret-key' not in str(caught.value)
