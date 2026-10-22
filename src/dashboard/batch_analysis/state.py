"""Scoped preferences and a worker that never calls Streamlit."""

from hashlib import sha256
from pathlib import Path
from threading import Event, Thread

import streamlit as st

from dashboard.state.workspace import MarketMemory


def analysis_memory(research):
    scope = "batch-analysis:" + repr((str(Path(research.pipeline.store.data_dir).resolve()),
                                      str(Path(research.runs_dir).resolve())))
    key = "_batch_workspace_" + sha256(scope.encode()).hexdigest()
    if key not in st.session_state:
        st.session_state[key] = MarketMemory(scope, directory=Path(research.runs_dir) / "analysis" / "workspaces")
    return st.session_state[key]


def job_key(memory):
    return "_batch_job_" + sha256(memory.scope.encode()).hexdigest()


class BatchJob:
    """Own a frozen runner and cooperative stop signal for one session's run."""

    def __init__(self, runner, run_id, *, temporary=None):
        self.temporary = temporary
        self.saved = False
        self.runner = runner
        self.run_id = run_id
        self.cancel = Event()
        self.error = None
        self.notified = False
        self.thread = Thread(target=self._run, name="analysis-" + run_id[:12], daemon=True)

    def _run(self):
        try:
            self.runner.run(self.run_id, cancel=self.cancel)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def start(self):
        self.thread.start()

    @property
    def running(self):
        return self.thread.is_alive()
