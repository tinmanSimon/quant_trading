"""Public provider contract, vendor implementations and default registry."""

from .base import BaseDataProvider
from .registry import ProviderEntry, ProviderFactory, ProviderRegistry
from .yahoo import YFinanceProvider
from .massive import MassiveProvider


registry = ProviderRegistry({"yahoo": YFinanceProvider, "massive": MassiveProvider})
"""Default registry; providers are constructed lazily when requested."""


def register_provider(
    name: str, provider: ProviderEntry, *, replace: bool = False
) -> None:
    """Register an instance or zero-argument factory in the default registry."""
    registry.register(name, provider, replace=replace)


def get_provider(name: str) -> BaseDataProvider:
    """Resolve a provider by name; unknown names never use a fallback."""
    return registry.get(name)


__all__ = [
    "BaseDataProvider",
    "ProviderEntry",
    "ProviderFactory",
    "ProviderRegistry",
    "YFinanceProvider",
    "MassiveProvider",
    "get_provider",
    "register_provider",
    "registry",
]
