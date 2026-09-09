"""Domain-specific exceptions raised by the data pipeline."""


class DataPipelineError(Exception):
    """Base class for all data-pipeline errors."""


class InvalidDataRequestError(DataPipelineError):
    """Raised when a provider-neutral data request is malformed or ambiguous."""


class OHLCVValidationError(DataPipelineError):
    """Base class for failures of the canonical OHLCV contract."""


class SchemaValidationError(OHLCVValidationError):
    """Raised when an OHLCV frame has incompatible columns or data types."""


class EmptyDataError(OHLCVValidationError):
    """Raised when an OHLCV operation receives an empty frame."""


class DuplicateBarError(OHLCVValidationError):
    """Raised when more than one bar has the same symbol and timestamp."""


class InvalidOHLCVError(OHLCVValidationError):
    """Raised when OHLCV values violate market-data invariants."""
