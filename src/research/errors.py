"""Errors reported by research orchestration."""


class ResearchError(Exception):
    pass


class PreflightError(ResearchError):
    def __init__(self, report):
        self.report = report
        super().__init__("Backtest aborted before execution:\n" + "\n".join(
            f"{issue.ticker}: {issue.message}" for issue in report.issues
        ))
