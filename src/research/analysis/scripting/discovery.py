"""Discover trusted local analyses; isolate imports and fingerprint helper code.

The optional ignored package is never imported by importing research itself.
Each load uses a source snapshot so editing files cannot mix new helper code
with an old script. Reload handles Python's same-second bytecode-cache edge case.
"""

from hashlib import sha256
import importlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys
from threading import RLock

from ..core.registry import discover_logics
from .decorator import TOOLKIT_VERSION, is_analysis, script_class


_CACHE = {}
_LOCK = RLock()
_GENERATION = 0


def project_root():
    return Path(__file__).resolve().parents[4]


class _SnapshotImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, namespace, root, sources):
        self.namespace, self.root = namespace, root
        self.modules = {}
        for relative, source in sources.items():
            parts = list(relative.with_suffix("").parts)
            package = parts[-1] == "__init__"
            if package:
                parts.pop()
            fullname = ".".join([namespace, *parts])
            self.modules[fullname] = (root / relative, source, package)
        for fullname in list(self.modules):
            parts = fullname.split(".")
            for count in range(1, len(parts)):
                name = ".".join(parts[:count])
                self.modules.setdefault(name, (root.joinpath(*parts[1:count], "__init__.py"), b"", True))
        self.modules.setdefault(namespace, (root / "__init__.py", b"", True))

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.modules:
            return None
        filename, _, package = self.modules[fullname]
        return importlib.util.spec_from_loader(fullname, self, origin=str(filename), is_package=package)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        filename, source, package = self.modules[module.__name__]
        module.__file__ = str(filename)
        if package:
            module.__path__ = []
        exec(compile(source, str(filename), "exec"), module.__dict__)


def _private(root, reload):
    global _GENERATION
    from ..indicators import ALGORITHM_VERSION as indicator_version
    from ..trend import ALGORITHM_VERSION as trend_version

    sources = {path.relative_to(root): path.read_bytes() for path in sorted(root.rglob("*.py"))
               if "__pycache__" not in path.parts}
    digest = sha256(f"{TOOLKIT_VERSION}:{indicator_version}:{trend_version}".encode())
    for path, source in sources.items():
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(source)
        digest.update(b"\0")
    fingerprint = digest.hexdigest()
    previous = _CACHE.get(root)
    if previous and previous[0] == fingerprint and not reload:
        return previous[1:]
    _GENERATION += 1
    namespace = f"_quant_analysis_{sha256(str(root).encode()).hexdigest()[:16]}_{_GENERATION}"
    importer = _SnapshotImporter(namespace, root, sources)
    sys.meta_path.insert(0, importer)
    classes, issues, seen = [], [], set()
    for relative in sources:
        if not relative.parts or relative.parts[0] != "analyses":
            continue
        if relative.name == "__init__.py" or relative.name.startswith("_"):
            continue
        module_name = namespace + "." + ".".join(relative.with_suffix("").parts)
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            issues.append(f"Cannot import private analysis {relative}: {error}")
            continue
        for function in vars(module).values():
            if not is_analysis(function) or function.__module__ != module.__name__ or function in seen:
                continue
            seen.add(function)
            try:
                classes.append(script_class(function, fingerprint))
            except Exception as error:
                issues.append(f"Cannot register private analysis {relative}: {error}")
    _CACHE[root] = fingerprint, classes, issues
    return classes, issues


def load_registry(project_dir=None, *, reload=False):
    """Load built-ins and optional ``secrete_analysis/analyses/*.py`` scripts.

    Code or helper changes automatically invalidate calculation caches. Explicit
    reload also repeats module initialization after a fixed dependency/import.
    Import failures and duplicate IDs are reported through ``registry.issues``.
    """
    registry = discover_logics()
    root = (Path(project_dir) if project_dir is not None else project_root()) / "secrete_analysis"
    if not root.is_dir():
        return registry
    root = root.resolve()
    with _LOCK:
        try:
            classes, issues = _private(root, reload)
        except Exception as error:
            registry._issues.append(f"Cannot inspect private analyses: {error}")
            return registry
    registry._issues.extend(issues)
    identities = {(item.name, item.version) for item in registry.definitions()}
    for cls in classes:
        key = cls.name, cls.version
        if key in identities:
            registry._classes.pop(key, None)
            registry._definitions.pop(key, None)
            registry._issues.append(f"Duplicate analysis {key}; unavailable. Use a unique @analysis name.")
            continue
        identities.add(key)
        try:
            registry.register_class(cls)
        except Exception as error:
            registry._issues.append(f"Cannot register analysis {key}: {error}")
    return registry
