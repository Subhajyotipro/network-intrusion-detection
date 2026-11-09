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
    m = tf.cast(tf.equal(y, tf.cast(target_class, tf.int32)), tf.float32)
    v = tf.expand_dims(m, axis=-1)
    v_not = tf.expand_dims(1.0 - m, axis=-1)

    diff = tf.nn.relu(margin - dists)
    diff_sq = tf.square(diff) if squared else diff
    dist_sq = tf.square(dists) if squared else dists

    sim_sum = tf.reduce_sum(v * tf.matmul(dist_sq, v))
    n_sim = tf.reduce_sum(v * tf.matmul(tf.cast(dists > eps, tf.float32), v))
    sim_loss = sim_sum / (n_sim + eps)

    dissim_sum = tf.reduce_sum(v * tf.matmul(diff_sq, v_not))
    n_dissim = tf.reduce_sum(v * tf.matmul(tf.cast(diff_sq > eps, tf.float32), v_not))
    dissim_loss = dissim_sum / (n_dissim + eps)

    n_tgt = tf.reduce_sum(m)
    total_dissim = n_tgt * (tf.cast(tf.shape(y)[0], tf.float32) - n_tgt)
    frac_pos = n_dissim / (total_dissim + eps)
    return (alpha * sim_loss) + ((1.0 - alpha) * dissim_loss), frac_pos


def clad_loss_and_frac(
    x: tf.Tensor,
    y: tf.Tensor,
    target_class: int = 0,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Calculate functional CLAD loss and fraction positive for target class."""
    _check_finite(x, "x")
    dists = pairwise_cosine_distance(x, eps=eps)
    return _contrastive_loss_calc(
        dists=dists,
        y=y,
        target_class=target_class,
        margin=margin,
        squared=squared,
        alpha=alpha,
        eps=eps,
    )


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
    loss, _ = clad_loss_and_frac(
        x,
        y,
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
    _, frac_pos = clad_loss_and_frac(
        x,
        y,
        target_class=target_class,
        margin=margin,
        squared=squared,
        alpha=alpha,
        eps=eps,
    )
    return frac_pos


def _closr_dist_matrix(x: tf.Tensor, eps: float) -> tf.Tensor:
    """Compute normalized pairwise cosine distance matrix with zeroed diagonal."""
    x_norm = tf.linalg.l2_normalize(x, axis=-1, epsilon=eps)
    x_t = tf.transpose(x_norm, [1, 0, 2])
    sim_matrix = tf.matmul(x_t, x_t, transpose_b=True)
    dist_matrix = tf.clip_by_value((1.0 - sim_matrix) / 2.0, 0.0, 1.0)
    n_classes, batch_size = tf.shape(x)[1], tf.shape(x)[0]
    return tf.linalg.set_diag(
        dist_matrix, tf.zeros([n_classes, batch_size], dtype=dist_matrix.dtype)
    )


def closr_loss_and_frac(
    x: tf.Tensor,
    y: tf.Tensor,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute vectorized CLOSR loss and mean fraction of positive pairs across classes."""
    _check_finite(x, "x")
    dist_matrix = _closr_dist_matrix(x, eps)
    n_classes, batch_size = tf.shape(x)[1], tf.shape(x)[0]

    one_hot = tf.one_hot(y, depth=n_classes, dtype=tf.float32)
    v_col = tf.expand_dims(tf.transpose(one_hot, [1, 0]), axis=-1)
    v_not_col = tf.expand_dims(tf.transpose(1.0 - one_hot, [1, 0]), axis=-1)

    diff = tf.nn.relu(margin - dist_matrix)
    diff_sq = tf.square(diff) if squared else diff
    dist_sq = tf.square(dist_matrix) if squared else dist_matrix

    sim_sum = tf.reduce_sum(v_col * tf.matmul(dist_sq, v_col), axis=[1, 2])
    n_sim = tf.reduce_sum(
        v_col * tf.matmul(tf.cast(dist_matrix > eps, tf.float32), v_col), axis=[1, 2]
    )
    dissim_sum = tf.reduce_sum(v_col * tf.matmul(diff_sq, v_not_col), axis=[1, 2])
    n_dissim = tf.reduce_sum(
        v_col * tf.matmul(tf.cast(diff_sq > eps, tf.float32), v_not_col), axis=[1, 2]
    )

    class_loss = (alpha * sim_sum / (n_sim + eps)) + ((1.0 - alpha) * dissim_sum / (n_dissim + eps))
    counts = tf.reduce_sum(one_hot, axis=0)
    total_dissim = counts * (tf.cast(batch_size, tf.float32) - counts)
    return tf.reduce_mean(class_loss), tf.reduce_mean(n_dissim / (total_dissim + eps))


def closr_loss(
    x: tf.Tensor,
    y: tf.Tensor,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> tf.Tensor:
    """Functional CLOSR loss modeling known classes as per-class vMF distributions."""
    loss, _ = closr_loss_and_frac(x, y, margin=margin, squared=squared, alpha=alpha, eps=eps)
    return loss


def closr_fraction_pos(
    x: tf.Tensor,
    y: tf.Tensor,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> tf.Tensor:
    """Compute mean fraction of positive dissimilar pairs across all known classes."""
    _, frac = closr_loss_and_frac(x, y, margin=margin, squared=squared, alpha=alpha, eps=eps)
    return frac
