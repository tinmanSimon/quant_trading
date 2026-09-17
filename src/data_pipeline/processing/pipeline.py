"""Ordered OHLCV transformations with reproducible processor identities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Set
import hashlib
import json
from typing import Protocol, runtime_checkable

import polars as pl

from ..exceptions import ProcessingError
from ._config import JSONValue, canonical_json, require_identifier, snapshot_config
from .contracts import DataContract, ProcessingResult


@runtime_checkable
class Processor(Protocol):
    """A versioned Python transformation of canonical OHLCV bars.

    ``name`` and ``version`` must be explicit, non-empty strings. ``config``
    must contain only JSON values, with string mapping keys and finite numbers.
    Implementations need not inherit this protocol. Identity and config must
    remain unchanged while a pipeline uses the processor; bump the version
    whenever transformation semantics change. Implementations own their key
    validation and resolve their output contracts internally; the pipeline
    independently checks each returned ProcessingResult against its contract.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def config(self) -> Mapping[str, JSONValue]: ...

    def transform(self, frame: pl.DataFrame, input_contract: DataContract) -> ProcessingResult: ...


def _identity(processor: Processor) -> dict[str, JSONValue]:
    name = require_identifier(processor.name, "name")
    version = require_identifier(processor.version, "version")
    config = snapshot_config(processor.config)
    if not callable(processor.transform):
        raise TypeError("Processor transform must be callable.")
    return {"name": name, "version": version, "config": config}


class Pipeline:
    """Snapshot and execute processors in the supplied order.

    ``canonical_json`` is the ordered JSON list of ``name``, ``version`` and
    ``config`` objects, with sorted mapping keys, compact separators and ASCII
    escapes. ``fingerprint`` is the SHA256 hex digest of its UTF-8 bytes.
    Metadata access returns defensive copies. A processor whose identity or
    config changes after construction is rejected, so execution cannot silently
    use stale provenance. Custom processors remain responsible for describing
    their semantics completely in that identity.
    """

    def __init__(self, processors: Iterable[Processor] = ()) -> None:
        try:
            if isinstance(processors, (Mapping, Set, str, bytes)):
                raise TypeError("Processors must be supplied in an ordered iterable.")
            self._processors = tuple(processors)
        except Exception as exc:
            raise ProcessingError(f"Invalid processor sequence: {exc}") from exc

        identity_json = []
        for index, processor in enumerate(self._processors):
            label = f"Processor at index {index}"
            try:
                name = require_identifier(processor.name, "name")
                label = f"Processor {name!r} at index {index}"
                identity_json.append(canonical_json(_identity(processor)))
            except Exception as exc:
                raise ProcessingError(f"{label} has invalid identity/config: {exc}") from exc

        self._canonical_json = "[" + ",".join(identity_json) + "]"
        self._fingerprint = hashlib.sha256(self._canonical_json.encode("utf-8")).hexdigest()
        self._identity_json = tuple(identity_json)

    @property
    def identities(self) -> list[dict[str, JSONValue]]:
        """Return a fresh JSON-compatible copy of the ordered identity snapshot."""
        return json.loads(self._canonical_json)

    @property
    def canonical_json(self) -> str:
        return self._canonical_json

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def run(self, frame: pl.DataFrame, contract: DataContract) -> ProcessingResult:
        """Chain processor results; contracts are explicit even for an empty pipeline.

        Processors own output-contract resolution and their validation hooks.
        This runner checks input/result contracts independently and retains
        responsibility for order, identity consistency and error attribution.
        """
        if not isinstance(contract, DataContract):
            raise ProcessingError("run requires an explicit contract (DataContract).")
        if not isinstance(frame, pl.DataFrame):
            cause = TypeError("Pipeline input must be a Polars DataFrame.")
            raise ProcessingError(str(cause)) from cause
        current = frame.clone()
        if not self._processors:
            try:
                current = contract.validate(current)
            except Exception as exc:
                raise ProcessingError(f"Input failed contract validation: {exc}") from exc
        for index, (processor, identity_json) in enumerate(
            zip(self._processors, self._identity_json, strict=True)
        ):
            identity = json.loads(identity_json)
            label = (
                f"Processor {identity['name']!r} version {identity['version']!r} "
                f"at index {index}"
            )
            try:
                self._check_identity(processor, identity_json)
                current = contract.validate(current)
                result = processor.transform(current.clone(), contract)
                if not isinstance(result, ProcessingResult):
                    raise TypeError("Processor transform must return a ProcessingResult.")
                if not isinstance(result.contract, DataContract):
                    raise TypeError("ProcessingResult.contract must be a DataContract.")
                output = result.contract.validate(result.frame)
                self._check_identity(processor, identity_json)
                current = output
                contract = result.contract
            except Exception as exc:
                raise ProcessingError(f"{label} failed: {exc}") from exc
        return ProcessingResult(current, contract)

    def transform(self, frame: pl.DataFrame, *, contract: DataContract) -> pl.DataFrame:
        """Return only the data; use run() when the output contract is also needed."""
        return self.run(frame, contract).frame

    @staticmethod
    def _check_identity(processor: Processor, expected: str) -> None:
        if canonical_json(_identity(processor)) != expected:
            raise ValueError(
                "Processor identity/config changed after the pipeline snapshot; "
                "construct a new pipeline."
            )
