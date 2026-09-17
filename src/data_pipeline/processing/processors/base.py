"""Processor template: subclasses implement hooks, not the public transform wrapper."""

from abc import ABC, abstractmethod
from typing import final

import polars as pl

from ...exceptions import ProcessingError
from ..contracts import DataContract, ProcessingResult


class BaseProcessor(ABC):
    """Inherit transform; implement _validate_input, _transform, _validate_output.

    ``final`` documents this convention to type checkers, not runtime security.
    Hooks receive isolated frames. Validators raise on failure and return None.
    Metadata-changing processors must also override output_contract. The
    wrapper resolves it once, before computation, and returns it with the data.
    Contracts are invocation-local, never stored as mutable processor state.
    """

    def output_contract(self, input_contract: DataContract) -> DataContract:
        return input_contract

    @final
    def transform(self, frame: pl.DataFrame, input_contract: DataContract) -> ProcessingResult:
        if not isinstance(input_contract, DataContract):
            raise ProcessingError("An explicit input contract (DataContract) is required.")
        output_contract = self.output_contract(input_contract)
        if not isinstance(output_contract, DataContract):
            raise TypeError("Processor must declare a DataContract before execution.")
        original = input_contract.validate(frame).clone()
        self._validate_input(original.clone(), input_contract)
        output = self._transform(original.clone(), input_contract, output_contract)
        output = output_contract.validate(output)
        self._validate_output(original.clone(), output.clone(), input_contract, output_contract)
        return ProcessingResult(output, output_contract)

    @abstractmethod
    def _validate_input(self, frame: pl.DataFrame, input_contract: DataContract) -> None:
        ...

    @abstractmethod
    def _transform(self, frame: pl.DataFrame, input_contract: DataContract,
                   output_contract: DataContract) -> pl.DataFrame:
        ...

    @abstractmethod
    def _validate_output(self, input_frame: pl.DataFrame, output_frame: pl.DataFrame,
                         input_contract: DataContract, output_contract: DataContract) -> None:
        ...
