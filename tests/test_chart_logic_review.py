"""Extension and failure-boundary checks for the dynamic chart-logic framework."""

from datetime import UTC, datetime
from uuid import uuid4

import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from research.analysis import BaseChartLogic, LogicContext, LogicRegistry, LogicResult, LogicRunner, LogicSpec
from dashboard.workspace import MarketMemory


PLUGIN = '''
from research.analysis import BaseChartLogic, LogicResult

class CustomLevel(BaseChartLogic):
    name = 'review_level'
    label = 'Review level'
    config_schema = {
        'type': 'object', 'additionalProperties': False,
        'properties': {'offset': {'type': 'number', 'default': 5., 'title': 'Custom offset'}},
    }
    def calculate(self, context, config, progress=None):
        values = [value + config['offset'] for value in context.frame['open'].to_list()]
        return LogicResult(overlays=[{'type': 'line', 'id': 'level', 'name': 'Custom level',
                                      'x': list(range(len(values))), 'y': values}],
                           tables=[{'name': 'Levels', 'rows': [{'level': value} for value in values]}])
'''


def _script(package):
    return f'''
from datetime import UTC, datetime
import polars as pl
import streamlit as st
from research.analysis import LogicContext, discover_logics
from dashboard.chart_data import ChartSeries
from dashboard.chart_component import window_payload
from dashboard.logic_controls import render_logic_controls, render_logic_results
from dashboard.workspace import market_memory
registry = discover_logics({package!r})
frame = pl.DataFrame({{'timestamp': [datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)],
                      'open': [100., 102.], 'close': [101., 103.], 'high': [104., 105.],
                      'low': [98., 99.], 'volume': [1000., 1100.]}})
context = LogicContext(frame, '1d', 'review', datetime(2024, 2, 1, tzinfo=UTC))
memory = market_memory('review-panel')
with st.container():
    run = render_logic_controls(context, memory, registry)
overlays = run.overlays if run else []
payload = window_payload(ChartSeries(frame, timeframe='1d'), chart_id='review', timeframe='1d',
                         timezone='UTC', overlays=overlays)
st.session_state['review-payload'] = payload
render_logic_results(run, registry)
memory.save()
'''


def _package(tmp_path, monkeypatch):
    name = 'review_plugins_' + uuid4().hex
    package = tmp_path / name
    package.mkdir()
    (package / '__init__.py').write_text('')
    monkeypatch.syspath_prepend(str(tmp_path))
    return name, package


def test_one_new_module_is_discovered_configured_applied_and_rendered_without_other_edits(tmp_path, monkeypatch):
    name, package = _package(tmp_path, monkeypatch)
    app = AppTest.from_string(_script(name), default_timeout=15).run()
    assert not app.exception
    assert not any(item.key == 'chart-logic-add-choice' for item in app.selectbox)
    original_package_init = (package / '__init__.py').read_text()
    # This is the only code change: no __init__, registry, UI, or renderer edits.
    (package / 'new_idea.py').write_text(PLUGIN)
    app.run()
    assert not app.exception
    assert app.selectbox(key='chart-logic-add-choice').options == ['Review level']
    app.button(key='chart-logic-add').click().run()
    assert not app.exception
    assert app.number_input[0].label == 'Custom offset'
    app.number_input[0].set_value(7).run()
    assert app.session_state['review-payload']['overlays'] == []
    app.button(key='chart-logic-apply').click().run()
    assert not app.exception and not app.error
    payload = app.session_state['review-payload']
    assert payload['overlays'][0]['y'] == [107., 109.]
    trace = next(trace for trace in payload['figure']['data']
                 if trace.get('meta', {}).get('chart_overlay'))
    assert trace['name'] == 'Custom level' and trace['y'] == [107., 109.]
    assert len(app.dataframe) == 1
    assert (package / '__init__.py').read_text() == original_package_init


def test_plugin_validation_exception_is_reported_as_invalid_configuration():
    class BrokenValidation(BaseChartLogic):
        name = 'broken_validation'
        def validate(self, config):
            raise RuntimeError('validation exploded')
        def calculate(self, context, config, progress=None):
            raise AssertionError('Invalid settings must prevent execution')

    registry = LogicRegistry()
    registry.register_class(BrokenValidation)
    runner = LogicRunner(registry)
    context = LogicContext(pl.DataFrame({'timestamp': [datetime(2024, 1, 1, tzinfo=UTC)], 'close': [100.]}),
                           '1d', 'test', datetime(2024, 2, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match='validation exploded'):
        runner.apply([LogicSpec('bad', 'broken_validation', '1', {})], context)


def test_malformed_saved_numeric_setting_does_not_abort_the_chart(tmp_path, monkeypatch):
    name, package = _package(tmp_path, monkeypatch)
    (package / 'level.py').write_text(PLUGIN)
    memory = MarketMemory('review-panel')
    malformed = LogicSpec('bad-settings', 'review_level', '1', {'offset': {}}).to_dict()
    memory.values['chart_logics'] = {'version': 1, 'draft': [malformed], 'applied': []}
    memory.save()
    app = AppTest.from_string(_script(name), default_timeout=15).run()
    assert not app.exception
    assert app.error or app.warning
    assert app.session_state['review-payload']['overlays'] == []
    assert app.button(key='chart-logic-apply').disabled


@pytest.mark.parametrize('properties,config', [
    ({'offset': True}, {'offset': 7}),
    ({'offset': {'type': 'number', 'default': 5.},
      'extra': {'type': 'object', 'properties': {'free': True}, 'additionalProperties': False}},
     {'offset': 7, 'extra': {'free': ['arbitrary', 1]}}),
    ({'offset': {'type': 'number', 'default': 5.},
      'extra': {'anyOf': [False, {'type': 'integer'}]}}, {'offset': 7, 'extra': 2}),
])
def test_boolean_json_schemas_use_editable_json_and_validate(tmp_path, monkeypatch, properties, config):
    import json
    from research.analysis import discover_logics

    name, package = _package(tmp_path, monkeypatch)
    original_properties = "{'offset': {'type': 'number', 'default': 5., 'title': 'Custom offset'}}"
    (package / 'boolean_schema.py').write_text(PLUGIN.replace(original_properties, repr(properties)))
    registry = discover_logics(name)
    assert not registry.issues
    assert registry.configure('review_level', '1', config) == config
    app = AppTest.from_string(_script(name), default_timeout=15).run()
    app.button(key='chart-logic-add').click().run()
    assert not app.exception
    app.text_area[0].set_value(json.dumps(config)).run()
    app.button(key='chart-logic-apply').click().run()
    assert not app.exception and not app.error
    assert app.session_state['review-payload']['overlays'][0]['y'] == [107., 109.]


def test_malformed_nested_object_setting_does_not_abort_the_chart(tmp_path, monkeypatch):
    name, package = _package(tmp_path, monkeypatch)
    original_properties = "{'offset': {'type': 'number', 'default': 5., 'title': 'Custom offset'}}"
    properties = {'offset': {'type': 'number', 'default': 5.},
                  'nested': {'type': 'object', 'default': {}, 'additionalProperties': False,
                             'properties': {'size': {'type': 'integer', 'default': 1}}}}
    (package / 'nested_schema.py').write_text(PLUGIN.replace(original_properties, repr(properties)))
    memory = MarketMemory('review-panel')
    malformed = LogicSpec('bad-settings', 'review_level', '1', {'offset': 5., 'nested': 42}).to_dict()
    memory.values['chart_logics'] = {'version': 1, 'draft': [malformed], 'applied': []}
    memory.save()
    app = AppTest.from_string(_script(name), default_timeout=15).run()
    assert not app.exception
    assert app.error or app.warning
    assert app.session_state['review-payload']['overlays'] == []
    assert app.button(key='chart-logic-apply').disabled


def test_plugin_owned_groups_render_once_and_keep_unlisted_settings_editable(tmp_path, monkeypatch):
    name, package = _package(tmp_path, monkeypatch)
    original_properties = "{'offset': {'type': 'number', 'default': 5., 'title': 'Custom offset'}}"
    properties = {'offset': {'type': 'number', 'default': 5., 'title': 'Custom offset'},
                  'scale': {'type': 'number', 'default': 2., 'title': 'Scale'},
                  'note': {'type': 'string', 'default': 'Ungrouped', 'title': 'Note'}}
    groups = [{'title': 'Main settings', 'fields': ['offset'], 'expanded': True},
              {'title': 'Advanced settings', 'fields': ['scale'], 'expanded': False}]
    module = PLUGIN.replace(original_properties, repr(properties))
    module += '\nCustomLevel.config_schema["x-ui-groups"] = ' + repr(groups) + '\n'
    (package / 'grouped.py').write_text(module)
    app = AppTest.from_string(_script(name), default_timeout=15).run()
    app.button(key='chart-logic-add').click().run()
    assert not app.exception and not app.error
    assert next(item for item in app.expander if item.label == 'Main settings').proto.expanded
    assert not next(item for item in app.expander if item.label == 'Advanced settings').proto.expanded
    assert [item.label for item in app.number_input] == ['Custom offset', 'Scale']
    assert [item.label for item in app.text_input] == ['Note']
    next(item for item in app.number_input if item.label == 'Custom offset').set_value(9)
    app.text_input[0].set_value('A new unlisted setting').run()
    app.button(key='chart-logic-apply').click().run()
    assert not app.exception and not app.error
    assert app.session_state['review-payload']['overlays'][0]['y'] == [109., 111.]
    applied = MarketMemory('review-panel').values['chart_logics']['applied'][0]['config']
    assert applied == {'offset': 9., 'scale': 2., 'note': 'A new unlisted setting'}


@pytest.mark.parametrize('groups', [
    'invalid',
    [{'fields': ['offset']}],
    [{'title': 'Main', 'fields': ['offset'], 'expanded': 'yes'}],
    [{'title': 'Main', 'fields': ['offset', 'offset']}],
    [{'title': 'Main', 'fields': ['missing']}],
])
def test_malformed_group_metadata_reports_error_without_hiding_chart(tmp_path, monkeypatch, groups):
    name, package = _package(tmp_path, monkeypatch)
    module = PLUGIN + '\nCustomLevel.config_schema["x-ui-groups"] = ' + repr(groups) + '\n'
    (package / 'bad_groups.py').write_text(module)
    app = AppTest.from_string(_script(name), default_timeout=15).run()
    app.button(key='chart-logic-add').click().run()
    assert not app.exception
    assert any('Invalid saved settings' in item.value for item in app.error)
    assert app.button(key='chart-logic-apply').disabled
    assert app.session_state['review-payload']['overlays'] == []


def test_result_table_render_failure_keeps_chart_available(tmp_path, monkeypatch):
    import streamlit as st

    name, package = _package(tmp_path, monkeypatch)
    (package / 'level.py').write_text(PLUGIN)

    def failed_table(*args, **kwargs):
        raise ValueError('Table conversion failed')

    monkeypatch.setattr(st, 'dataframe', failed_table)
    app = AppTest.from_string(_script(name), default_timeout=15).run()
    app.button(key='chart-logic-add').click().run()
    app.button(key='chart-logic-apply').click().run()
    assert not app.exception
    assert any('Table conversion failed' in item.value for item in app.error)
    assert app.session_state['review-payload']['overlays'][0]['y'] == [105., 107.]


def test_new_module_supplies_live_suggested_filters_without_dashboard_edits(tmp_path, monkeypatch):
    from research.analysis import discover_logics

    name, package = _package(tmp_path, monkeypatch)
    plugin = PLUGIN.replace("        values = [value + config['offset']", "        type(self).calls += 1\n        values = [value + config['offset']") + '''
    calls = 0
    display_schema = {
        'type': 'object', 'additionalProperties': False,
        'properties': {'levels': {'type': 'array', 'default': [],
                                  'items': {'type': 'number'}, 'title': 'Visible levels'}},
    }
    def display_schema_for(self, result):
        from copy import deepcopy
        schema = deepcopy(self.display_schema)
        schema['properties']['levels']['x-ui-options'] = sorted(set(result.overlays[0]['y']))
        return schema
    def present(self, result, config):
        if config['levels']:
            for overlay in result.overlays:
                overlay['y'] = [value if value in config['levels'] else None for value in overlay['y']]
        return result
'''
    (package / 'new_idea.py').write_text(plugin)
    script = _script(name).replace(
        'from dashboard.logic_controls import render_logic_controls, render_logic_results',
        'from dashboard.logic_controls import render_logic_controls, render_logic_results, render_logic_display_controls',
    ).replace('overlays = run.overlays if run else []',
              'run = render_logic_display_controls(run, memory, registry)\noverlays = run.overlays if run else []')
    app = AppTest.from_string(script, default_timeout=15).run()
    app.button(key='chart-logic-add').click().run()
    assert not app.multiselect
    app.button(key='chart-logic-apply').click().run()
    assert not app.exception and not app.error
    logic_class = type(discover_logics(name).create('review_level', '1'))
    assert logic_class.calls == 1
    assert [float(value) for value in app.multiselect[0].options] == [105., 107.]
    app.multiselect[0].set_value([107.]).run()
    assert not app.exception and not app.error
    assert app.session_state['review-payload']['overlays'][0]['y'] == [None, 107.]
    assert logic_class.calls == 1
    assert not any(item.value == 'Unapplied changes' for item in app.caption)
    visibility = next(item for item in app.checkbox if item.label == 'Show on chart')
    app.checkbox(key=visibility.key).uncheck().run()
    assert app.session_state['review-payload']['overlays'] == []
    app.checkbox(key=visibility.key).check().run()
    app.multiselect[0].set_value([]).run()
    assert app.session_state['review-payload']['overlays'][0]['y'] == [105., 107.]
    assert logic_class.calls == 1
