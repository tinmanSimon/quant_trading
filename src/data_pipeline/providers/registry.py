"""Explicit provider registration; lookups never fall back to another vendor."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ..exceptions import ProviderError, UnknownProviderError
from .base import BaseDataProvider


ProviderFactory = Callable[[], BaseDataProvider]
ProviderEntry = BaseDataProvider | ProviderFactory


class ProviderRegistry:
    """Map case-insensitive provider names to instances or zero-arg factories.

    Instances are reused by identity. Factories (including provider classes)
    are called on each lookup, so registration itself performs no vendor I/O.
    Existing registrations require an explicit ``replace=True`` to overwrite.
    """

    def __init__(self, providers: Mapping[str, ProviderEntry] | None = None) -> None:
        self._providers: dict[str, ProviderEntry] = {}
        for name, provider in (providers or {}).items():
            self.register(name, provider)

    def register(
        self, name: str, provider: ProviderEntry, *, replace: bool = False
    ) -> None:
        key = _provider_key(name)
        if not isinstance(provider, BaseDataProvider) and not callable(provider):
            raise TypeError("provider must be a BaseDataProvider instance or factory.")
        if key in self._providers and not replace:
            raise ValueError(f"Provider {key!r} is already registered.")
        self._providers[key] = provider

    def get(self, name: str) -> BaseDataProvider:
        key = _provider_key(name)
        try:
            entry = self._providers[key]
        except KeyError as exc:
            raise UnknownProviderError(f"Unknown provider {key!r}.") from exc
        if isinstance(entry, BaseDataProvider):
            return entry
        try:
            provider = entry()
        except Exception as exc:
            raise ProviderError(f"Could not construct provider {key!r}.") from exc
        if not isinstance(provider, BaseDataProvider):
            raise ProviderError(
                f"Factory for provider {key!r} did not return a BaseDataProvider."
            )
        return provider


def _provider_key(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Provider name must be a non-empty string.")
    return name.strip().lower()
