"""Ordered, versioned processing of canonical OHLCV frames."""

from .pipeline import Pipeline, Processor
from .contracts import DataContract, ProcessingResult
from .processors import BaseProcessor, ScalePrices, ResampleOHLCV, TradingSession
from .registry import ProcessorDefinition, ProcessorRegistry, discover_processors, load_pipeline

__all__ = ["BaseProcessor", "DataContract", "ProcessingResult",
           "Pipeline", "Processor", "ProcessorDefinition", "ProcessorRegistry", "ScalePrices",
           "discover_processors", "load_pipeline",
           "ResampleOHLCV", "TradingSession"]
