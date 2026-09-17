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


class StorageError(DataPipelineError):
    """Base class for failures while persisting or reading local data."""


class DataAlreadyExistsError(StorageError):
    """Raised when a write would replace an existing raw-data file or record."""


class DatasetNotFoundError(StorageError):
    """Raised when a requested local dataset is absent from the catalog."""


class DataIntegrityError(StorageError):
    """Raised when stored data no longer matches its catalog metadata."""


class RequestDataMismatchError(StorageError):
    """Raised when data does not match the request it is being stored under."""


class StorageWriteError(StorageError):
    """Raised when a local write cannot be completed safely."""


class OverlappingDataError(DataAlreadyExistsError):
    """Incoming coverage conflicts with an active dataset."""


class ConfirmationRequiredError(StorageError):
    """Replacement needs the exact target IDs as confirmation."""


class StorageBusyError(StorageError):
    """Another process is using the store; retry after it finishes."""


class ProviderError(DataPipelineError):
    """A vendor could not supply valid data."""


class UnsupportedRequestError(ProviderError):
    """The vendor does not support this request."""


class UnknownProviderError(ProviderError):
    """No provider is registered under the requested name."""


class ProcessingError(DataPipelineError):
    """A processor configuration or execution failed."""
