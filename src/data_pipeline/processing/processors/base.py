"""Processor template: subclasses implement hooks, not the public transform wrapper."""

from abc import ABC, abstractmethod
from typing import ClassVar, final

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

    # Discovery reads these declarations without constructing the processor.
    # Mark causal only when each output bar uses information available by its
    # completion time. Explicit metadata is required for research eligibility.
    causal: ClassVar[bool] = False
    price_basis: ClassVar[str] = "preserve"
    config_schema: ClassVar[dict | None] = None

    @property
    def config(self):
        """Infer JSON settings unless an implementation provides its own config."""
        if "_explicit_processor_config" in self.__dict__:
            return self.__dict__["_explicit_processor_config"]
        from ..configuration import instance_config
        return instance_config(self)

    @config.setter
    def config(self, value):
        # Retain compatibility with processors that set self.config explicitly.
        self._explicit_processor_config = value

    def output_contract(self, input_contract: DataContract) -> DataContract:
        return input_contract

    def validate_backtest_calendar(self, bars, *, input_contract: DataContract,
                                   output_contract: DataContract) -> None:
        """Validate the timing of output used by the research execution calendar.

        ``bars`` contains records with ``timestamp``, ``open``, and ``close``
        timezone-aware UTC datetimes, restricted to the saved output's coverage.
        This hook is independent of research implementation types. The default
        supports unchanged contracts: a causal processor promises output values
        are available by each output bar's completion time. A processor changing
        timeframes/timestamp conventions must override this hook and validate
        its configured aggregation or labeling against the supplied calendar.
        """
        if input_contract != output_contract:
            raise ProcessingError(
                "Processors changing the data contract must implement "
                "validate_backtest_calendar to validate execution calendar completion times."
            )

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
