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
"""Inference routines: embedding extraction, centroid computation, scorers, and streaming eval."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

__all__ = [
    "get_embeddings",
    "get_centroids",
    "get_clad_centroid",
    "calc_class_sims",
    "upstream_scorer",
    "predict_closed_set",
    "stream_evaluate_closr",
    "stream_evaluate_clad",
    "SCORERS",
]


def get_embeddings(model: tf.keras.Model, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Extract embeddings in chunked batches without gradient tracking."""
    embeddings = []
    n = len(x)

    @tf.function(reduce_retracing=True)
    def _fwd(batch: tf.Tensor) -> tf.Tensor:
        return model(batch, training=False)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        z = _fwd(x[start:end]).numpy()
        embeddings.append(z)
    if not embeddings:
        d_out = getattr(model, "d_out", 64)
        return np.empty((0, d_out), dtype=np.float32)
    return np.concatenate(embeddings, axis=0)


def get_centroids(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute normalized class centroids chunkwise from training embeddings."""
    n = len(x_train)
    d_out = getattr(model, "d_out", 64)
    class_sums = np.zeros((n_classes, d_out), dtype=np.float64)
    class_counts = np.zeros(n_classes, dtype=np.int64)

    @tf.function(reduce_retracing=True)
    def _fwd(batch: tf.Tensor) -> tf.Tensor:
        return model(batch, training=False)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        z = _fwd(x_train[start:end]).numpy()
        y_chunk = y_train[start:end]
        normed = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
        one_hot = (y_chunk[:, None] == np.arange(n_classes)).astype(np.float64)
        class_sums += np.einsum("bc,bcd->cd", one_hot, normed)
        class_counts += np.sum(one_hot, axis=0).astype(np.int64)

    centroids = class_sums / np.maximum(class_counts[:, None], 1)
    norm = np.maximum(np.linalg.norm(centroids, axis=-1, keepdims=True), 1e-12)
    return (centroids / norm).astype(np.float32)


def get_clad_centroid(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    benign_id: int = 0,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute normalized benign centroid chunkwise for CLAD model."""
    n = len(x_train)
    d_out = getattr(model, "d_out", 8)
    benign_sum = np.zeros(d_out, dtype=np.float64)
    benign_count = 0

    @tf.function(reduce_retracing=True)
    def _fwd(batch: tf.Tensor) -> tf.Tensor:
        return model(batch, training=False)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        mask = y_train[start:end] == benign_id
        if np.any(mask):
            z = _fwd(x_train[start:end][mask]).numpy()
            benign_sum += np.sum(z, axis=0)
            benign_count += int(np.sum(mask))

    mean_emb = benign_sum / max(benign_count, 1)
    norm = max(float(np.linalg.norm(mean_emb)), 1e-12)
    return (mean_emb / norm).astype(np.float32)


def calc_class_sims(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity of embeddings to centroids."""
    if embeddings.ndim == 3:
        e_norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-12)
        c_norm = centroids / np.maximum(np.linalg.norm(centroids, axis=-1, keepdims=True), 1e-12)
        return np.sum(e_norm * c_norm[None, :, :], axis=2).astype(np.float32)
    e_norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-12)
    c_norm = centroids / np.maximum(float(np.linalg.norm(centroids)), 1e-12)
    return np.sum(e_norm * c_norm[None, :], axis=1).astype(np.float32)


def upstream_scorer(sims: np.ndarray) -> np.ndarray:
    """Compute upstream quadratic softmax weighted open-set known-ness score."""
    s = sims.astype(np.float64)
    exp_s = np.exp(s - np.max(s, axis=-1, keepdims=True))
    probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    return np.sum((s**2) * probs, axis=-1).astype(np.float32)


SCORERS = {"upstream": upstream_scorer}


def predict_closed_set(sims: np.ndarray) -> np.ndarray:
    """Predict closed-set class via argmax of similarity vector."""
    return np.argmax(sims, axis=-1)


def stratified_tsne_indices(labels: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    """Keep every zero-day index and sample known classes proportionally and deterministically."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("t-SNE labels must be a one-dimensional integer array")
    if max_points <= 0:
        raise ValueError("tsne_max_points must be positive")
    zero_day = np.flatnonzero(labels < 0)
    known = np.flatnonzero(labels >= 0)
    if zero_day.size == 0:
        raise ValueError("t-SNE labels contain no zero-day samples")
    if zero_day.size > max_points:
        raise ValueError("tsne_max_points cannot retain every zero-day sample")
    if labels.size <= max_points:
        return np.arange(labels.size)
    classes, counts = np.unique(labels[known], return_counts=True)
    known_budget = max_points - zero_day.size
    if known_budget < classes.size:
        raise ValueError("tsne_max_points is too small to represent every known class")

    ideal = known_budget * counts / counts.sum()
    allocations = np.maximum(1, np.floor(ideal).astype(int))
    allocations = np.minimum(allocations, counts)
    while allocations.sum() > known_budget:
        candidates = np.flatnonzero(allocations > 1)
        allocations[candidates[np.argmax(allocations[candidates] - ideal[candidates])]] -= 1
    while allocations.sum() < known_budget:
        candidates = np.flatnonzero(allocations < counts)
        allocations[candidates[np.argmax(ideal[candidates] - allocations[candidates])]] += 1

    rng = np.random.default_rng(seed)
    sampled = [zero_day]
    for class_id, count in zip(classes, allocations):
        class_indices = np.flatnonzero(labels == class_id)
        sampled.append(rng.choice(class_indices, size=count, replace=False))
    return np.sort(np.concatenate(sampled))


def stream_evaluate_closr(
    model: tf.keras.Model,
    x: np.ndarray,
    centroids: np.ndarray,
    batch_size: int = 4096,
    tsne_max_points: int = 5000,
    seed: int = 42,
    sample_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream evaluate CLOSR chunkwise without accumulating full test embeddings."""
    n = len(x)
    if sample_labels is None or len(sample_labels) != n:
        raise ValueError("sample_labels must align with the evaluation rows")
    tsne_idx = stratified_tsne_indices(sample_labels, tsne_max_points, seed)
    tsne_set = set(tsne_idx.tolist())
    all_sims, tsne_embs = [], []

    @tf.function(reduce_retracing=True)
    def _fwd(batch: tf.Tensor) -> tf.Tensor:
        return model(batch, training=False)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        z = _fwd(x[start:end]).numpy()
        all_sims.append(calc_class_sims(z, centroids))
        chunk_tsne = [i - start for i in range(start, end) if i in tsne_set]
        if chunk_tsne:
            tsne_embs.append(z[chunk_tsne])

    n_c = len(centroids)
    sims = np.concatenate(all_sims, axis=0) if all_sims else np.empty((0, n_c), dtype=np.float32)
    scores = upstream_scorer(sims) if len(sims) > 0 else np.empty((0,), dtype=np.float32)
    preds = predict_closed_set(sims) if len(sims) > 0 else np.empty((0,), dtype=np.int64)
    d_out = getattr(model, "d_out", 64)
    embs = (
        np.concatenate(tsne_embs, axis=0)
        if tsne_embs
        else np.empty((0, n_c, d_out), dtype=np.float32)
    )
    return sims, scores, preds, embs, tsne_idx


def stream_evaluate_clad(
    model: tf.keras.Model,
    x: np.ndarray,
    centroid: np.ndarray,
    batch_size: int = 4096,
    tsne_max_points: int = 5000,
    seed: int = 42,
    sample_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream evaluate CLAD chunkwise without accumulating full test embeddings."""
    n = len(x)
    if sample_labels is None or len(sample_labels) != n:
        raise ValueError("sample_labels must align with the evaluation rows")
    tsne_idx = stratified_tsne_indices(sample_labels, tsne_max_points, seed)
    tsne_set = set(tsne_idx.tolist())
    all_sims, tsne_embs = [], []

    @tf.function(reduce_retracing=True)
    def _fwd(batch: tf.Tensor) -> tf.Tensor:
        return model(batch, training=False)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        z = _fwd(x[start:end]).numpy()
        all_sims.append(calc_class_sims(z, centroid))
        chunk_tsne = [i - start for i in range(start, end) if i in tsne_set]
        if chunk_tsne:
            tsne_embs.append(z[chunk_tsne])

    sims = np.concatenate(all_sims, axis=0) if all_sims else np.empty((0,), dtype=np.float32)
    d_out = getattr(model, "d_out", 8)
    embs = (
        np.concatenate(tsne_embs, axis=0) if tsne_embs else np.empty((0, d_out), dtype=np.float32)
    )
    return sims, embs, tsne_idx
