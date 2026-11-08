# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration I/O, CLI overrides, reproducibility helpers, and runtime metadata."""

from __future__ import annotations

import datetime
import logging
import os
import random
import subprocess
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import yaml

if TYPE_CHECKING:
    from nids.config import Config


def resolve_device(device_str: str) -> str:
    """Resolve device string to 'gpu' or 'cpu' using TensorFlow."""
    import tensorflow as tf

    dev = device_str.lower()
    if dev == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        return "cpu"
    if dev == "gpu":
        if not tf.config.list_physical_devices("GPU"):
            raise ValueError("Device 'gpu' requested, but no GPU found by TensorFlow.")
        return "gpu"
    if dev == "auto":
        return "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
    raise ValueError(f"Invalid device '{device_str}'. Expected 'auto', 'gpu', or 'cpu'.")


def _coerce_type(expected_type: Any, val: Any, key: str) -> Any:
    """Coerce and validate value against expected field type."""
    if expected_type is bool or expected_type == "bool":
        if isinstance(val, bool):
            return val
        if isinstance(val, str) and val.lower() in ("true", "1", "yes"):
            return True
        if isinstance(val, str) and val.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"Expected bool for '{key}', got {type(val).__name__} ({val})")
    if expected_type in (float, "float"):
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        try:
            return float(val)
        except (ValueError, TypeError):
            raise ValueError(f"Expected float for '{key}', got {type(val).__name__} ({val})")
    if expected_type in (int, "int"):
        if isinstance(val, int) and not isinstance(val, bool):
            return val
        if isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                pass
        raise ValueError(f"Expected int for '{key}', got {type(val).__name__} ({val})")
    return str(val) if expected_type in (str, "str") else val


def apply_overrides(cfg_dict: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply repeated key=value CLI overrides parsed via yaml.safe_load (last wins)."""
    from nids.config import Config

    valid_fields = {f.name: f.type for f in fields(Config)}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override format: '{item}'. Expected 'key=value'.")
        key, val_str = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in override '{item}'.")
        if key not in valid_fields:
            raise KeyError(f"Unknown config key: '{key}'.")
        parsed_val = yaml.safe_load(val_str)
        cfg_dict[key] = _coerce_type(valid_fields[key], parsed_val, key)
    return cfg_dict


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[list[str]] = None,
    model_kind: Optional[str] = None,
) -> Config:
    """Load config from optional YAML path, overlaying overrides and validating fields."""
    from nids.config import Config, get_default_config, validate_config

    raw_dict: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"YAML content in {p} must be a mapping.")
            raw_dict = loaded

    kind = model_kind or raw_dict.get("model_kind", "closr")
    merged = asdict(get_default_config(kind))
    valid_names = {f.name for f in fields(Config)}
    for k, v in raw_dict.items():
        if k not in valid_names:
            raise KeyError(f"Unknown config key in YAML: '{k}'.")
        merged[k] = v

    if overrides:
        merged = apply_overrides(merged, overrides)

    cfg = Config(**merged)
    validate_config(cfg)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """Serialize Config dataclass to a YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python random, NumPy, and TensorFlow for deterministic execution."""
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def get_run_metadata(cfg: Config) -> dict[str, Any]:
    """Return runtime metadata dictionary including git commit, versions, and device."""
    import keras
    import pandas
    import sklearn
    import tensorflow as tf

    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_hash = "unknown"

    return {
        "seed": cfg.seed,
        "git_commit": git_hash,
        "device": resolve_device(cfg.device),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "versions": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "pandas": pandas.__version__,
            "sklearn": sklearn.__version__,
        },
    }


def setup_logging(level: int = logging.INFO) -> None:
    """Configure standard logging format across all package and script execution."""
    fmt = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")
