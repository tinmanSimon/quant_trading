"""Local Market data workspace snapshots; no executable objects are deserialized."""

from dataclasses import asdict
from datetime import date, datetime, time
from hashlib import sha256
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
import zlib

import streamlit as st
from polars.exceptions import PolarsError

from trend_analysis import ALGORITHM_VERSION, TrendConfig
from trend_analysis.models import AnalysisResult, CandidateDiagnostic, Trend


DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / ".local" / "market_workspaces"


def _encode(value):
    if isinstance(value, (datetime, date, time)):
        return {"type": type(value).__name__, "value": value.isoformat()}
    if isinstance(value, tuple):
        return {"type": "tuple", "value": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"type", "value"}:
            kind, content = value["type"], value["value"]
            if kind == "tuple":
                return tuple(_decode(item) for item in content)
            return {"date": date, "datetime": datetime, "time": time}[kind].fromisoformat(content)
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite saved value")
    return value


def _atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".workspace-", delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class MarketMemory:
    def __init__(self, scope, directory=None):
        directory = Path(directory) if directory is not None else DEFAULT_DIRECTORY
        self.path = directory / (sha256(scope.encode()).hexdigest() + ".json")
        self.analysis_path = self.path.with_suffix(".analysis.json.gz")
        self.values = {"widgets": {}}
        self.warning = None
        self.blocked = False
        self._saved = None
        try:
            document = _decode(json.loads(self.path.read_text(encoding="utf-8")))
            if (type(document.get("version")) is not int or document["version"] != 1
                    or document.get("scope") != scope
                    or not isinstance(document.get("values"), dict)
                    or not isinstance(document["values"].get("widgets"), dict)):
                raise ValueError("invalid workspace format")
            self.values = document["values"]
        except FileNotFoundError:
            pass
        except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError) as exc:
            self.blocked = True
            self.warning = f"Could not restore Market data workspace: {exc}. The saved file was left unchanged."
        self.scope = scope
        self._saved = self._payload() if not self.blocked and self.path.exists() else None

    def _payload(self):
        return json.dumps(_encode({"version": 1, "scope": self.scope, "values": self.values}),
                          sort_keys=True, allow_nan=False).encode()

    def save(self):
        if self.blocked:
            return
        try:
            payload = self._payload()
            if payload != self._saved:
                _atomic_write(self.path, payload)
                self._saved = payload
            self.warning = None
        except (OSError, TypeError, ValueError) as exc:
            self.warning = f"Could not save Market data workspace: {exc}. Changes remain available in this session."

    def _capture(self, key):
        self.values["widgets"][key] = st.session_state[key]
        self.save()

    def widget(self, render, label, *, key, **kwargs):
        """Keep widget values outside Streamlit's page/conditional-widget cleanup."""
        saved = self.values["widgets"]
        present = key in saved
        value = saved.get(key)
        kind = render.__name__
        options = kwargs.get("options")
        if present and options is not None:
            if kind == "multiselect":
                value = [item for item in value if item in options] if isinstance(value, list) else []
            elif value not in options:
                present = False
        if present and kind == "number_input":
            present = type(value) in (int, float) and math.isfinite(value)
            if present:
                integer = type(kwargs.get("step", kwargs.get("value"))) is int
                if integer and int(value) != value:
                    present = False
                else:
                    value = int(value) if integer else float(value)
                value = max(kwargs.get("min_value", value), value) if kwargs.get("min_value") is not None else value
                value = min(kwargs.get("max_value", value), value) if kwargs.get("max_value") is not None else value
        if present and kind in ("checkbox", "expander", "text_input", "date_input"):
            expected = {"checkbox": bool, "expander": bool, "text_input": str, "date_input": date}[kind]
            present = type(value) is expected
        if present and kind == "date_input":
            # Explicit bounds keep older saved market dates selectable even when
            # Streamlit's default date range would be centered on today's date.
            default = kwargs.get("value", date.today())
            kwargs.setdefault("min_value", date.fromordinal(max(1, min(value, default).toordinal() - 3650)))
            kwargs.setdefault("max_value", date.fromordinal(min(date.max.toordinal(), max(value, default).toordinal() + 3650)))
        if present:
            st.session_state[key] = value
            kwargs.pop("value", None)
        else:
            st.session_state.pop(key, None)
        value = render(label, key=key, on_change=self._capture, args=(key,), **kwargs)
        saved[key] = value.open if kind == "expander" else value
        return value

    def save_analysis(self, key, result):
        document = {"version": 1, "key": key, "result": asdict(result)}
        content = json.dumps(_encode(document), allow_nan=False).encode()
        _atomic_write(self.analysis_path, gzip.compress(content))
        self.values["analysis_key"] = key

    def load_analysis(self, key, config):
        if self.values.get("analysis_key") != key:
            return None
        try:
            document = _decode(json.loads(gzip.decompress(self.analysis_path.read_bytes())))
            if document.get("version") != 1 or document.get("key") != key:
                raise ValueError("analysis snapshot does not match")
            values = document["result"]
            values["config"] = TrendConfig.from_dict(values["config"])
            if values["config"] != config or values["algorithm_version"] != ALGORITHM_VERSION:
                return None
            values["trends"] = tuple(Trend(**item) for item in values["trends"])
            values["diagnostics"] = tuple(CandidateDiagnostic(**item) for item in values["diagnostics"])
            result = AnalysisResult(**values)
            # Reject malformed snapshots before they reach the chart or tables.
            from dashboard.trend_overlays import trend_descriptors
            trend_descriptors(result.trends)
            result.to_frame()
            result.diagnostics_frame()
            return result
        except (OSError, EOFError, UnicodeError, ValueError, TypeError, KeyError, AttributeError, zlib.error, PolarsError):
            self.values["analysis_key"] = None
            st.warning("Saved trend analysis could not be restored. Apply and analyze to rebuild it.")
            return None


def market_memory(scope):
    key = "_market_workspace_" + sha256((str(DEFAULT_DIRECTORY.resolve()) + scope).encode()).hexdigest()
    if key not in st.session_state:
        st.session_state[key] = MarketMemory(scope)
    return st.session_state[key]
