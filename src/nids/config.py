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
"""Configuration dataclasses, YAML loading, CLI overrides, and reproducibility helpers."""

from __future__ import annotations

import logging
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml


@dataclass
class Config:
    """Flat configuration dataclass holding all experiment hyperparameters and paths."""

    # Paths
    data_path: str = "data/lycos.csv"
    results_dir: str = "results"
    weights_dir: str = "weights"
    upstream_dir: str = "third_party/CLOSR"

    # Data
    split_seed: int = 39058032
    test_ratio: float = 0.5
    val_ratio: float = 0.0
    sample_thres: int = 100
    drop_cols: list[str] = field(
        default_factory=lambda: [
            "flow_id",
            "src_addr",
            "src_port",
            "dst_addr",
            "dst_port",
            "ip_prot",
            "timestamp",
        ]
    )
    class_zero: str = "benign"

    # Model & Loss
    model_kind: str = "closr"
    d_in: Optional[int] = None
    d_out: int = 64
    n_classes: int = 12
    hidden: list[int] = field(default_factory=lambda: [1024, 1024, 1024])
    dropout: float = 0.1
    residual: bool = False
    margin: float = 1.0
    squared: bool = True

    # Optimizer & Scheduler
    batch_size: int = 4096
    weight_decay: float = 0.0403709
    lr_init: float = 1e-6
    lr_peak: float = 1e-5
    lr_final: float = 1e-6
    warmup_frac: float = 0.1
    epochs: int = 200
    seed: int = 1
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    smoke: bool = False
    max_batches: Optional[int] = None

    # Inference & Calibration
    scorer: str = "upstream"
    grid_size: int = 200
    target_acceptance: list[float] = field(default_factory=lambda: [0.95, 0.99])
    f_beta: float = 1.0
    max_known_rejection: float = 0.05
    max_benign_fp: float = 0.01
    siamese_conf_grid: int = 10
    skip_if_confident_benign: bool = False

    # Siamese Stage-2
    support_mode: str = "prototype"
    k_exemplars: int = 5

    # Benchmarking
    warmup_runs: int = 20
    timed_runs: int = 200
    bench_batch_sizes: list[int] = field(default_factory=lambda: [1, 32, 1024])

    # Visualization
    tsne_max_points: int = 5000

    # Device
    device: str = "auto"


def get_default_config(model_kind: str = "closr") -> Config:
    """Return default Config dataclass initialized with per-kind defaults."""
    kind = model_kind.lower()
    if kind == "closr":
        return Config(
            model_kind="closr",
            d_out=64,
            hidden=[1024, 1024, 1024],
            dropout=0.1,
            weight_decay=0.0403709,
            lr_peak=1e-5,
        )
    if kind == "clad":
        return Config(
            model_kind="clad",
            d_out=8,
            hidden=[1024, 1024, 1024, 1024],
            dropout=0.0,
            weight_decay=3e-7,
            lr_peak=4.4e-6,
        )
    raise ValueError(f"Unknown model_kind: '{model_kind}'. Expected 'closr' or 'clad'.")


def resolve_device(device_str: str) -> str:
    """Resolve device string to 'cuda', 'mps', or 'cpu'."""
    dev = device_str.lower()
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dev in {"cuda", "cpu", "mps"}:
        return dev
    raise ValueError(f"Invalid device '{device_str}'. Expected 'auto', 'cuda', 'cpu', or 'mps'.")


def validate_config(cfg: Config) -> None:
    """Validate hyperparameter bounds and invariants."""
    if cfg.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {cfg.epochs}")
    if not (0.0 <= cfg.test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in [0.0, 1.0), got {cfg.test_ratio}")
    if not (0.0 <= cfg.val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0.0, 1.0), got {cfg.val_ratio}")
    if len(cfg.hidden) == 0:
        raise ValueError("hidden layers list cannot be empty")
    if any(h <= 0 for h in cfg.hidden):
        raise ValueError(f"hidden layer dimensions must be positive, got {cfg.hidden}")
    if cfg.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {cfg.batch_size}")
    if cfg.d_out <= 0:
        raise ValueError(f"d_out must be positive, got {cfg.d_out}")
    if cfg.n_classes <= 0:
        raise ValueError(f"n_classes must be positive, got {cfg.n_classes}")


def _coerce_type(expected_type: Any, val: Any, key: str) -> Any:
    """Coerce and validate value against expected field type."""
    if expected_type is bool or expected_type == "bool":
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            if val.lower() in ("true", "1", "yes"):
                return True
            if val.lower() in ("false", "0", "no"):
                return False
        raise ValueError(f"Expected bool for '{key}', got {type(val).__name__} ({val})")
    if expected_type is float or expected_type == "float":
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                pass
        raise ValueError(f"Expected float for '{key}', got {type(val).__name__} ({val})")
    if expected_type is int or expected_type == "int":
        if isinstance(val, int) and not isinstance(val, bool):
            return val
        if isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                pass
        raise ValueError(f"Expected int for '{key}', got {type(val).__name__} ({val})")
    if expected_type is str or expected_type == "str":
        return str(val)
    return val


def apply_overrides(cfg_dict: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply repeated key=value CLI overrides parsed via yaml.safe_load (last wins)."""
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
        coerced = _coerce_type(valid_fields[key], parsed_val, key)
        cfg_dict[key] = coerced
    return cfg_dict


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[list[str]] = None,
    model_kind: Optional[str] = None,
) -> Config:
    """Load config from optional YAML path, overlaying overrides and validating fields."""
    raw_dict: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if loaded is not None:
                if not isinstance(loaded, dict):
                    raise ValueError(f"YAML content in {p} must be a mapping.")
                raw_dict = loaded

    kind = model_kind or raw_dict.get("model_kind", "closr")
    base_cfg = get_default_config(kind)
    merged = asdict(base_cfg)

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


def set_seed(seed: int) -> None:
    """Seed Python random, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_run_metadata(cfg: Config) -> dict[str, Any]:
    """Return runtime metadata dictionary including git commit, versions, and device."""
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_hash = "unknown"

    import pandas
    import sklearn

    return {
        "seed": cfg.seed,
        "git_commit": git_hash,
        "device": resolve_device(cfg.device),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pandas.__version__,
            "sklearn": sklearn.__version__,
        },
    }


def setup_logging(level: int = logging.INFO) -> None:
    """Configure standard logging format across all package and script execution."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
