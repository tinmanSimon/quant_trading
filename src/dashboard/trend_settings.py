"""Project-local persistence for applied trend settings, independent of Streamlit."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

from trend_analysis import TrendConfig


SCHEMA_VERSION = 1
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / ".local" / "trend_config.json"


@dataclass(frozen=True)
class LoadedTrendSettings:
    config: TrendConfig
    warning: str | None = None


def load_trend_settings(path: Path | str | None = None) -> LoadedTrendSettings:
    """Load valid settings; a bad file is retained for inspection, never rewritten."""
    path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return LoadedTrendSettings(TrendConfig())
    except (OSError, UnicodeError, ValueError) as exc:
        return LoadedTrendSettings(
            TrendConfig(), f"Could not load saved trend settings from {path}: {exc}. "
            "Using defaults; the saved file has not been changed.")
    try:
        if not isinstance(document, dict):
            raise ValueError("settings must be a JSON object")
        version = document.get("schema_version")
        if type(version) is not int or version != SCHEMA_VERSION:
            raise ValueError(f"unsupported settings schema version {version!r}")
        if not isinstance(document.get("config"), dict):
            raise ValueError("config must be a JSON object")
        config = TrendConfig.from_dict(document["config"])
    except (TypeError, ValueError) as exc:
        return LoadedTrendSettings(
            TrendConfig(), f"Saved trend settings in {path} are invalid: {exc}. "
            "Using defaults; the saved file has not been changed.")
    warning = None
    if "redundancy_threshold_bars" in document["config"]:
        rule = "Both boundaries (AND)" if config.redundancy_boundary_rule == "and" else "Either boundary (OR)"
        warning = (
            "The saved bar-based redundancy setting has been replaced by percentage matching. "
            f"Using {config.redundancy_percentage:g}% and {rule}; other saved settings are preserved. "
            "Apply and analyze to save the new format. The saved file has not been changed."
        )
    if "penalty_multiplier" in document["config"] and "penalty_grid_levels" not in document["config"]:
        grid_notice = (
            "Segmentation now uses a fixed λ grid. The saved penalty multiplier is used directly as Starting λ "
            f"({config.penalty_multiplier:g}), with {config.penalty_grid_levels} levels. "
            "Review the fixed λ grid preview and apply to save the new format."
        )
        warning = f"{warning} {grid_notice}" if warning else grid_notice
    return LoadedTrendSettings(config, warning)


def save_trend_settings(config: TrendConfig, path: Path | str | None = None) -> None:
    """Replace settings atomically, only after the complete new file is flushed."""
    path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    # Validate at this boundary too: persistence should only ever store a valid
    # configuration, even if a caller supplied an object from a stale session.
    values = TrendConfig.from_dict(config.to_dict()).to_dict()
    payload = json.dumps({"schema_version": SCHEMA_VERSION, "config": values},
                         indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
