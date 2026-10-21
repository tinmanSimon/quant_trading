"""Apply snapshots, per-logic caching, and safe JSON persistence."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
import json

from .base import LogicContext, LogicResult, LogicSpec, json_copy


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _frame_key(frame):
    if frame is None:
        return None
    digest = sha256(repr(tuple(frame.schema.items())).encode())
    digest.update(str(frame.height).encode())
    digest.update(frame.hash_rows(seed=41, seed_1=73, seed_2=109, seed_3=151).to_numpy().tobytes())
    return digest.hexdigest()


def source_key(context: LogicContext):
    """Do not include wall-clock time: completion changes only at bar boundaries."""
    return _hash({"data": _frame_key(context.frame), "support": _frame_key(context.support_frame),
                  "identity": repr(context.identity), "timeframe": context.timeframe,
                  "session_timezone": context.session_timezone, "metadata": context.metadata,
                  "completed": context.completed_frame.height})


@dataclass
class AppliedRun:
    source_key: str
    specs: list[LogicSpec]
    results: dict[str, LogicResult]
    errors: dict[str, str]
    keys: dict[str, str]

    @property
    def overlays(self):
        outputs = []
        for spec in self.specs:
            result = self.results.get(spec.instance_id)
            if result is None:
                continue
            for overlay in result.overlays:
                item = deepcopy(overlay)
                item["id"] = f"{spec.instance_id}:{item['id']}"
                item["instance_id"] = spec.instance_id
                if item.get("panel") is not None:
                    namespace = (f"logic:{spec.name}:{spec.version}"
                                 if item["panel"].get("scope") == "logic" else spec.instance_id)
                    item["panel"]["id"] = f"{namespace}:{item['panel']['id']}"
                outputs.append(item)
        return outputs


class LogicRunner:
    def __init__(self, registry, cache_limit=16, output_validator=None):
        if type(cache_limit) is not int or cache_limit < 0:
            raise ValueError("cache_limit must be a nonnegative integer.")
        self.registry = registry
        self.cache_limit = cache_limit
        self.output_validator = output_validator
        self._cache = OrderedDict()

    def _key(self, spec, context, data_keys=None):
        logic = self.registry.create(spec.name, spec.version)
        if data_keys is None:
            data_keys = {"original": _frame_key(context.frame), "completed": _frame_key(context.completed_frame),
                         "support": _frame_key(context.support_frame)}
        implementation = ({"implementation_fingerprint": logic.implementation_fingerprint}
                          if logic.implementation_fingerprint else {})
        return _hash({**implementation, "name": spec.name, "version": spec.version, "result_version": logic.result_version, "config": spec.config,
                      "data": data_keys[logic.input_policy], "support": data_keys["support"],
                      "identity": repr(context.identity), "timeframe": context.timeframe,
                      "session_timezone": context.session_timezone, "metadata": context.metadata,
                      "cutoff": context.cutoff.isoformat() if logic.uses_cutoff else None})

    def _validate_output(self, result):
        result.validate()
        if self.output_validator is not None:
            for overlay in result.overlays:
                self.output_validator(overlay)

    def _remember(self, key, result):
        if self.cache_limit:
            self._cache[key] = result.to_dict()
            self._cache.move_to_end(key)
        while len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)

    def validate(self, specs):
        normalized, ids = [], set()
        for spec in specs:
            spec = LogicSpec.from_dict(spec.to_dict() if isinstance(spec, LogicSpec) else spec)
            if spec.instance_id in ids:
                raise ValueError("Each selected chart logic must have a unique instance ID.")
            ids.add(spec.instance_id)
            config = self.registry.configure(spec.name, spec.version, spec.config)
            normalized.append(LogicSpec(spec.instance_id, spec.name, spec.version, config))
        return normalized

    def apply(self, specs, context, progress=None):
        # Validate every configuration before starting any calculations.
        specs = self.validate(specs)
        context = replace(context, calculation_cache={})
        results, errors, keys = {}, {}, {}
        data_keys = {"original": _frame_key(context.frame), "completed": _frame_key(context.completed_frame),
                     "support": _frame_key(context.support_frame)}
        for spec in specs:
            try:
                logic = self.registry.create(spec.name, spec.version)
                missing = set(logic.required_columns) - set(context.frame.columns)
                if missing:
                    raise ValueError(f"Missing required data columns: {', '.join(sorted(missing))}.")
                if logic.supported_timeframes and context.timeframe not in logic.supported_timeframes:
                    raise ValueError(f"{context.timeframe} is not supported by this logic.")
                key = self._key(spec, context, data_keys)
                keys[spec.instance_id] = key
                if key in self._cache:
                    result = LogicResult.from_dict(self._cache[key])
                    self._cache.move_to_end(key)
                else:
                    callback = ((lambda done, total: progress(spec.instance_id, done, total))
                                if progress else None)
                    result = logic.calculate(context, deepcopy(spec.config), progress=callback)
                    if not isinstance(result, LogicResult):
                        raise ValueError("Chart logic must return a LogicResult.")
                    self._validate_output(result)
                    self._remember(key, result)
                results[spec.instance_id] = LogicResult.from_dict(result.to_dict())
            except Exception as error:
                errors[spec.instance_id] = str(error)
        return AppliedRun(source_key(context), specs, results, errors, keys)

    def present(self, run, display_configs=None, visibility=None):
        """Create disposable display results; calculation cache and archive stay intact."""
        display_configs = display_configs or {}
        visibility = visibility or {}
        results, errors = {}, dict(run.errors)
        for spec in run.specs:
            if spec.instance_id not in run.results:
                continue
            try:
                result = self.registry.present(spec.name, spec.version, run.results[spec.instance_id],
                                               display_configs.get(spec.instance_id, {}))
                self._validate_output(result)
                if not visibility.get(spec.instance_id, True):
                    result.overlays = []
                    for table in result.tables:
                        for row in table["rows"]:
                            if "on_chart" in row:
                                row["on_chart"] = False
                                if row.get("status") == "On chart":
                                    row["status"] = "Hidden"
                    result.messages.insert(0, "Hidden on chart. Enable Show on chart to display this result.")
                results[spec.instance_id] = result
            except Exception as error:
                errors[spec.instance_id] = f"Could not display the calculated result: {error}"
        return AppliedRun(run.source_key, list(run.specs), results, errors, dict(run.keys))

    def matches(self, run, context):
        if run.source_key != source_key(context):
            return False
        try:
            specs = self.validate(run.specs)
            data_keys = {"original": _frame_key(context.frame), "completed": _frame_key(context.completed_frame),
                         "support": _frame_key(context.support_frame)}
            return all(run.keys.get(spec.instance_id) == self._key(spec, context, data_keys)
                       for spec in specs if spec.instance_id in run.results)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def snapshot(run):
        return json_copy({"version": 2, "source_key": run.source_key,
                          "specs": [spec.to_dict() for spec in run.specs],
                          "results": {key: result.to_dict() for key, result in run.results.items()},
                          "errors": run.errors, "keys": run.keys})

    def restore(self, snapshot, context):
        try:
            if snapshot.get("version") != 2:
                return None
            specs = self.validate(snapshot["specs"])
            results = {key: LogicResult.from_dict(value) for key, value in snapshot["results"].items()}
            ids = {spec.instance_id for spec in specs}
            errors, keys = snapshot["errors"], snapshot["keys"]
            if (not isinstance(errors, dict) or not isinstance(keys, dict)
                    or set(results) & set(errors) or set(results) | set(errors) != ids
                    or not set(keys) <= ids
                    or not all(isinstance(value, str) for value in [*errors.values(), *keys.values()])):
                return None
            run = AppliedRun(snapshot["source_key"], specs, results, errors, keys)
            if not self.matches(run, context):
                return None
            for identifier, result in results.items():
                self._validate_output(result)
                self._remember(keys[identifier], result)
            return run
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
