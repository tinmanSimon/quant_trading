"""Load a trusted, optional local strategy package without changing sys.path.

Package modules are imported once per process. Each registry is freshly built,
so registration on Streamlit reruns cannot accumulate duplicate entries.
"""

from functools import lru_cache
from hashlib import sha256
import importlib
import importlib.util
from pathlib import Path
import sys
from threading import RLock

from research.errors import ResearchError

from .registry import builtin_registry


class PrivateStrategyError(ResearchError):
    """The private package exists but could not be loaded or registered."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


_IMPORT_LOCK = RLock()


@lru_cache(maxsize=None)
def _import_hook(root: Path):
    package = root / "private_strategies"
    namespace = "_quant_private_" + sha256(str(root).encode()).hexdigest()
    try:
        for filename in ("__init__.py", "register.py"):
            if not (package / filename).is_file():
                raise ValueError(f"Missing {filename} in {package}.")
        spec = importlib.util.spec_from_file_location(namespace, package / "__init__.py",
                                                      submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = module
        spec.loader.exec_module(module)
        registration = importlib.import_module(namespace + ".register")
        hook = getattr(registration, "register_strategies", None)
        if not callable(hook):
            raise ValueError("register.py must define register_strategies(registry).")
        return hook
    except BaseException:
        for name in tuple(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                del sys.modules[name]
        raise


def load_registry(*, project_dir=None):
    """Combine built-ins with private registrations; absence is optional, errors are not.

    Private modules should use relative imports (``from .my_strategy import ...``).
    Restart the process after editing source. The source tree root is the default,
    independent of cwd and the selected market-data/results directories.
    """
    root = project_root() if project_dir is None else Path(project_dir).expanduser().resolve()
    registry = builtin_registry()
    package = root / "private_strategies"
    if not package.exists():
        return registry
    try:
        with _IMPORT_LOCK:
            hook = _import_hook(root)
            hook(registry)
    except Exception as exc:
        raise PrivateStrategyError(f"Cannot load private strategies from {package}: {exc}") from exc
    return registry
