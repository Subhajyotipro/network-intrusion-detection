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
"""Configuration dataclasses, validation, and per-kind defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from nids.config_io import (
    apply_overrides,
    get_run_metadata,
    load_config,
    resolve_device,
    save_config,
    set_seed,
    setup_logging,
)

__all__ = [
    "Config",
    "get_default_config",
    "validate_config",
    "apply_overrides",
    "load_config",
    "save_config",
    "set_seed",
    "get_run_metadata",
    "resolve_device",
    "setup_logging",
]


@dataclass
class Config:
    """Flat configuration dataclass holding all experiment hyperparameters and paths."""

    # Paths
    data_path: str = "data/lycos.csv"
    data_source: str = "auto"
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
    init: str = "torch_default"
    margin: float = 1.0
    squared: bool = True

    # Optimizer & Scheduler
    batch_size: int = 4096
    weight_decay: float = 0.0403709
    adam_eps: float = 1e-8
    lr_init: float = 1e-6
    lr_peak: float = 1e-5
    lr_final: float = 1e-6
    warmup_frac: float = 0.1
    epochs: int = 200
    baseline_lr: float = 1e-3
    baseline_epochs: int = 10
    seed: int = 1
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    smoke: bool = False
    max_batches: Optional[int] = None
    log_every: int = 10
    checkpoint_every: int = 0

    # Inference & Calibration
    scorer: str = "upstream"
    eval_batch_size: int = 4096
    eval_max_rows: int = 0
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

    # Benchmarking & Visualization
    warmup_runs: int = 20
    timed_runs: int = 200
    bench_batch_sizes: list[int] = field(default_factory=lambda: [1, 32, 1024])
    tsne_max_points: int = 5000
    tsne_n_components: int = 2
    tsne_perplexity: float = 30.0
    tsne_early_exaggeration: float = 12.0
    tsne_learning_rate: str | float = "auto"
    tsne_max_iter: int = 1000
    tsne_n_iter_without_progress: int = 300
    tsne_min_grad_norm: float = 1e-7
    tsne_metric: str = "euclidean"
    tsne_init: str = "pca"
    tsne_method: str = "barnes_hut"
    tsne_angle: float = 0.5
    tsne_n_jobs: int = 1
    tsne_verbose: int = 0
    tsne_marker_size: float = 12.0
    tsne_alpha: float = 0.85
    tsne_axis_padding: float = 0.04
    tsne_tiny_group_threshold: int = 25
    tsne_figure_size: list[float] = field(default_factory=lambda: [9.0, 7.0])
    tsne_colors: dict[str, str] = field(
        default_factory=lambda: {
            "benign": "#00876c",
            "malicious": "#d64b3c",
            "zero_day": "#2878b5",
        }
    )

    # Device & Determinism
    device: str = "auto"
    deterministic: bool = False


def get_default_config(model_kind: str = "closr") -> Config:
    """Return default Config dataclass initialized with per-kind defaults."""
    kind = model_kind.lower()
    if kind == "closr":
        return Config()
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


def validate_config(cfg: Config) -> None:
    """Validate hyperparameter bounds and invariants."""
    if cfg.data_source not in ("auto", "real", "synthetic"):
        raise ValueError(
            f"data_source must be 'auto', 'real', or 'synthetic', got '{cfg.data_source}'"
        )
    if cfg.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {cfg.epochs}")
    if cfg.baseline_epochs <= 0:
        raise ValueError(f"baseline_epochs must be positive, got {cfg.baseline_epochs}")
    if cfg.baseline_lr <= 0.0:
        raise ValueError(f"baseline_lr must be positive, got {cfg.baseline_lr}")
    if not (0.0 <= cfg.test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in [0.0, 1.0), got {cfg.test_ratio}")
    if not (0.0 <= cfg.val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0.0, 1.0), got {cfg.val_ratio}")
    if cfg.init not in ("torch_default", "keras_default"):
        raise ValueError(f"init must be 'torch_default' or 'keras_default', got '{cfg.init}'")
    if len(cfg.hidden) == 0:
        raise ValueError("hidden layers list cannot be empty")
    if any(h <= 0 for h in cfg.hidden):
        raise ValueError(f"hidden layer dimensions must be positive, got {cfg.hidden}")
    for name, val in (
        ("batch_size", cfg.batch_size),
        ("d_out", cfg.d_out),
        ("n_classes", cfg.n_classes),
        ("eval_batch_size", cfg.eval_batch_size),
    ):
        if val <= 0:
            raise ValueError(f"{name} must be positive, got {val}")
    if cfg.eval_max_rows < 0:
        raise ValueError(f"eval_max_rows must be non-negative, got {cfg.eval_max_rows}")
    if cfg.tsne_max_points <= 0 or cfg.tsne_n_components != 2 or cfg.tsne_n_jobs != 1:
        raise ValueError("t-SNE requires positive max points, 2 components, and n_jobs=1")
    if cfg.tsne_perplexity <= 0.0 or cfg.tsne_max_iter < 250:
        raise ValueError("t-SNE perplexity must be positive and max_iter must be at least 250")
    if cfg.tsne_init != "pca":
        raise ValueError("tsne_init must be 'pca' for deterministic project visualizations")
