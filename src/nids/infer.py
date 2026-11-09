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


def _subsample_indices(n_total: int, max_points: int, seed: int) -> np.ndarray:
    """Select sorted seeded subsample indices for t-SNE visualization."""
    if n_total <= max_points or max_points <= 0:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=max_points, replace=False))


def stream_evaluate_closr(
    model: tf.keras.Model,
    x: np.ndarray,
    centroids: np.ndarray,
    batch_size: int = 4096,
    tsne_max_points: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream evaluate CLOSR chunkwise without accumulating full test embeddings."""
    n = len(x)
    tsne_idx = _subsample_indices(n, tsne_max_points, seed)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream evaluate CLAD chunkwise without accumulating full test embeddings."""
    n = len(x)
    tsne_idx = _subsample_indices(n, tsne_max_points, seed)
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
