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
"""Visualization routines for t-SNE projections, ROC curves, and trade-off plots."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from nids.config import Config
from nids.infer import stratified_tsne_indices

logger = logging.getLogger(__name__)


def _prepare_tsne_input(
    embeddings: np.ndarray, labels: np.ndarray, cfg: Config, model_kind: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate embeddings, flatten CLOSR vectors, and select the deterministic sample."""
    if embeddings is None:
        raise ValueError("t-SNE embeddings are missing")
    vectors, labels = np.asarray(embeddings), np.asarray(labels)
    expected_ndim = 3 if model_kind == "closr" else 2
    if vectors.ndim != expected_ndim or vectors.shape[0] != labels.size or labels.ndim != 1:
        raise ValueError(f"{model_kind} embeddings and labels have incompatible shapes")
    if vectors.shape[0] == 0 or not np.isfinite(vectors).all():
        raise ValueError("t-SNE embeddings must be non-empty and finite")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("t-SNE labels must be integers")
    if model_kind == "closr":
        vectors = vectors.reshape(vectors.shape[0], -1)
    indices = stratified_tsne_indices(labels, cfg.tsne_max_points, cfg.seed)
    if len(indices) <= cfg.tsne_perplexity:
        raise ValueError("t-SNE sample size must exceed tsne_perplexity")
    return vectors[indices].astype(np.float32), labels[indices]


def _fit_tsne(vectors: np.ndarray, cfg: Config) -> np.ndarray:
    """Fit the configured deterministic two-dimensional t-SNE projection."""
    return TSNE(
        n_components=cfg.tsne_n_components,
        perplexity=cfg.tsne_perplexity,
        early_exaggeration=cfg.tsne_early_exaggeration,
        learning_rate=cfg.tsne_learning_rate,
        max_iter=cfg.tsne_max_iter,
        n_iter_without_progress=cfg.tsne_n_iter_without_progress,
        min_grad_norm=cfg.tsne_min_grad_norm,
        metric=cfg.tsne_metric,
        init=cfg.tsne_init,
        verbose=cfg.tsne_verbose,
        random_state=cfg.seed,
        method=cfg.tsne_method,
        angle=cfg.tsne_angle,
        n_jobs=cfg.tsne_n_jobs,
    ).fit_transform(vectors)


def _draw_tsne(
    coordinates: np.ndarray, labels: np.ndarray, cfg: Config, kind: str, dataset: str, path: Path
) -> tuple[dict[str, int], bool]:
    """Draw stable group styling, counts, axes, and legend to a PNG file."""
    groups = {
        "benign": labels == 0,
        "malicious": labels > 0,
        "zero_day": labels < 0,
    }
    counts = {name: int(mask.sum()) for name, mask in groups.items()}
    zero_day_tiny = counts["zero_day"] <= cfg.tsne_tiny_group_threshold
    names = ("benign", "malicious", "zero_day")
    display = ("Benign", "Malicious known", "Zero-day" + (" (TINY)" if zero_day_tiny else ""))
    fig, axis = plt.subplots(figsize=cfg.tsne_figure_size)
    for name, label, mask in zip(names, display, groups.values()):
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=cfg.tsne_marker_size,
            alpha=cfg.tsne_alpha,
            c=cfg.tsne_colors[name],
            label=f"{label} (n={counts[name]})",
            linewidths=0,
        )
    low, high = coordinates.min(axis=0), coordinates.max(axis=0)
    padding = np.maximum((high - low) * cfg.tsne_axis_padding, cfg.tsne_axis_padding)
    axis.set_xlim(low[0] - padding[0], high[0] + padding[0])
    axis.set_ylim(low[1] - padding[1], high[1] + padding[1])
    count_text = ", ".join(f"{name}={count}" for name, count in counts.items())
    axis.set_title(f"{kind.upper()} embeddings | {dataset} | {count_text}")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120, metadata={"Software": "matplotlib"})
    plt.close(fig)
    return counts, zero_day_tiny


def _write_tsne_sidecar(
    path: Path,
    cfg: Config,
    model_kind: str,
    dataset: str,
    reduced_run: bool,
    counts: dict[str, int],
    zero_day_tiny: bool,
    git_commit: str,
    versions: dict[str, Any],
) -> None:
    """Write stable run metadata and every configured t-SNE parameter to JSON."""
    param_names = (
        "n_components",
        "perplexity",
        "early_exaggeration",
        "learning_rate",
        "max_iter",
        "n_iter_without_progress",
        "min_grad_norm",
        "metric",
        "init",
        "method",
        "angle",
        "n_jobs",
        "verbose",
    )
    parameters = {name: getattr(cfg, f"tsne_{name}") for name in param_names}
    metadata = {
        "model": model_kind,
        "dataset": dataset,
        "seed": cfg.seed,
        "reduced_run": bool(reduced_run),
        "projection": "flattened_per_class_embeddings" if model_kind == "closr" else "embedding",
        "n_points_per_group": counts,
        "zero_day_tiny": zero_day_tiny,
        "tsne_parameters": {"seed": cfg.seed, "max_points": cfg.tsne_max_points, **parameters},
        "git_commit": git_commit,
        "versions": versions,
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_tsne_figure(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cfg: Config,
    model_kind: str,
    dataset: str,
    reduced_run: bool,
    git_commit: str,
    versions: dict[str, Any],
) -> tuple[np.ndarray, Path, Path]:
    """Save a deterministic, group-counted t-SNE PNG and its JSON metadata sidecar."""
    if model_kind not in ("clad", "closr") or dataset not in ("real", "synthetic"):
        raise ValueError("model_kind must be clad/closr and dataset must be real/synthetic")
    vectors, selected_labels = _prepare_tsne_input(embeddings, labels, cfg, model_kind)
    coordinates = _fit_tsne(vectors, cfg)
    output_dir = Path(cfg.results_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tsne_{model_kind}_seed{cfg.seed}"
    figure_path, sidecar_path = output_dir / f"{stem}.png", output_dir / f"{stem}.json"
    counts, zero_day_tiny = _draw_tsne(
        coordinates, selected_labels, cfg, model_kind, dataset, figure_path
    )
    _write_tsne_sidecar(
        sidecar_path,
        cfg,
        model_kind,
        dataset,
        reduced_run,
        counts,
        zero_day_tiny,
        git_commit,
        versions,
    )
    logger.info("Saved t-SNE figure and metadata: %s, %s", figure_path, sidecar_path)
    return coordinates, figure_path, sidecar_path
