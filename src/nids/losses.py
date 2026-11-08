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
"""Contrastive learning loss functions for CLAD and CLOSR anomaly and open-set recognition.

Adapted from upstream CLOSR (Apache-2.0, DOI 10.1109/TNSM.2026.3652529)
by Jack Wilkie et al. Re-implemented in pure TensorFlow operations, supporting
eager execution, tf.function graph compilation, and batch edge cases.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf


def _check_finite(tensor: tf.Tensor, name: str) -> None:
    """Raise ValueError if input tensor contains NaN or Inf values in eager execution."""
    if tf.executing_eagerly():
        if not tf.reduce_all(tf.math.is_finite(tensor)):
            raise ValueError(f"Tensor '{name}' contains non-finite (NaN or Inf) values.")


def pairwise_cosine_distance(
    x1: tf.Tensor, x2: Optional[tf.Tensor] = None, eps: float = 1e-8
) -> tf.Tensor:
    """Compute normalized pairwise cosine distance matrix in range [0, 1]."""
    u = tf.linalg.l2_normalize(x1, axis=-1, epsilon=eps)
    is_self = x2 is None
    v = tf.linalg.l2_normalize(x2, axis=-1, epsilon=eps) if not is_self else u
    sim = tf.matmul(u, v, transpose_b=True)
    dist = (1.0 - sim) / 2.0
    dist = tf.clip_by_value(dist, 0.0, 1.0)
    if is_self:
        batch_size = tf.shape(x1)[0]
        dist = tf.linalg.set_diag(dist, tf.zeros([batch_size], dtype=dist.dtype))
    return dist


def _contrastive_loss_calc(
    dists: tf.Tensor,
    y: tf.Tensor,
    target_class: int,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Calculate functional asymmetric contrastive loss for a single target class."""
    dists = tf.cast(dists, tf.float32)
    y = tf.cast(y, tf.int32)
    tgt = tf.cast(target_class, tf.int32)

    # Pair masks matching upstream logic
    y_col = tf.reshape(y, [-1, 1])
    y_row = tf.reshape(y, [1, -1])
    eq_pairs = tf.equal(y_col, y_row)

    sim_mask = tf.logical_and(eq_pairs, tf.equal(y_row, tgt))
    dissim_mask = tf.logical_and(tf.logical_not(eq_pairs), tf.equal(y_col, tgt))

    # Similar pair loss
    sim_dists = tf.where(sim_mask, dists, 0.0)
    n_sim = tf.reduce_sum(tf.cast(sim_dists > eps, tf.float32))
    sim_calc = tf.square(sim_dists) if squared else sim_dists
    sim_loss = tf.reduce_sum(sim_calc) / (n_sim + eps)

    # Dissimilar pair loss
    margin_val = tf.cast(margin, tf.float32)
    margin_box = tf.fill(tf.shape(dists), margin_val)
    dissim_dists = tf.where(dissim_mask, dists, margin_box)
    dissim_diff = tf.nn.relu(margin_val - dissim_dists)
    dissim_calc = tf.square(dissim_diff) if squared else dissim_diff
    n_dissim = tf.reduce_sum(tf.cast(dissim_calc > eps, tf.float32))
    dissim_loss = tf.reduce_sum(dissim_calc) / (n_dissim + eps)

    # Diagnostic fraction of margin violations
    total_dissim = tf.reduce_sum(tf.cast(dissim_mask, tf.float32))
    frac_pos = n_dissim / (total_dissim + eps)

    total_loss = (alpha * sim_loss) + ((1.0 - alpha) * dissim_loss)
    return total_loss, frac_pos


def clad_loss(
    x: tf.Tensor,
    y: tf.Tensor,
    target_class: int = 0,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> tf.Tensor:
    """Functional CLAD loss modeling benign traffic as a von Mises-Fisher distribution."""
    _check_finite(x, "x")
    dists = pairwise_cosine_distance(x, eps=eps)
    loss, _ = _contrastive_loss_calc(
        dists=dists,
        y=y,
        target_class=target_class,
        margin=margin,
        squared=squared,
        alpha=alpha,
        eps=eps,
    )
    return loss


def clad_fraction_pos(
    x: tf.Tensor,
    y: tf.Tensor,
    target_class: int = 0,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> tf.Tensor:
    """Compute fraction of positive (margin-violating) dissimilar pairs for CLAD."""
    _check_finite(x, "x")
    dists = pairwise_cosine_distance(x, eps=eps)
    _, frac_pos = _contrastive_loss_calc(
        dists=dists,
        y=y,
        target_class=target_class,
        margin=margin,
        squared=squared,
        alpha=alpha,
        eps=eps,
    )
    return frac_pos


def closr_loss(
    x: tf.Tensor,
    y: tf.Tensor,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> tf.Tensor:
    """Functional CLOSR loss modeling known classes as per-class vMF distributions."""
    _check_finite(x, "x")
    # x shape: [B, C, D]
    x_norm = tf.linalg.l2_normalize(x, axis=-1, epsilon=eps)
    x_t = tf.transpose(x_norm, [1, 0, 2])  # [C, B, D]
    sim_matrix = tf.matmul(x_t, x_t, transpose_b=True)  # [C, B, B]
    dist_matrix = tf.clip_by_value((1.0 - sim_matrix) / 2.0, 0.0, 1.0)

    n_classes = tf.shape(x)[1]
    batch_size = tf.shape(x)[0]
    dist_matrix = tf.linalg.set_diag(
        dist_matrix, tf.zeros([n_classes, batch_size], dtype=dist_matrix.dtype)
    )
    losses = tf.TensorArray(tf.float32, size=n_classes)
    for c in tf.range(n_classes):
        loss_c, _ = _contrastive_loss_calc(
            dists=dist_matrix[c],
            y=y,
            target_class=c,
            margin=margin,
            squared=squared,
            alpha=alpha,
            eps=eps,
        )
        losses = losses.write(c, loss_c)

    return tf.reduce_mean(losses.stack())


def closr_fraction_pos(
    x: tf.Tensor,
    y: tf.Tensor,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> tf.Tensor:
    """Compute mean fraction of positive dissimilar pairs across all known classes."""
    _check_finite(x, "x")
    x_norm = tf.linalg.l2_normalize(x, axis=-1, epsilon=eps)
    x_t = tf.transpose(x_norm, [1, 0, 2])
    sim_matrix = tf.matmul(x_t, x_t, transpose_b=True)
    dist_matrix = tf.clip_by_value((1.0 - sim_matrix) / 2.0, 0.0, 1.0)

    n_classes = tf.shape(x)[1]
    batch_size = tf.shape(x)[0]
    dist_matrix = tf.linalg.set_diag(
        dist_matrix, tf.zeros([n_classes, batch_size], dtype=dist_matrix.dtype)
    )
    fracs = tf.TensorArray(tf.float32, size=n_classes)
    for c in tf.range(n_classes):
        _, frac_c = _contrastive_loss_calc(
            dists=dist_matrix[c],
            y=y,
            target_class=c,
            margin=margin,
            squared=squared,
            alpha=alpha,
            eps=eps,
        )
        fracs = fracs.write(c, frac_c)

    return tf.reduce_mean(fracs.stack())
