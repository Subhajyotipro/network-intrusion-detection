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
"""NumPy-only numeric regressions for upstream CLOSR behavior (DV11)."""

import numpy as np
import tensorflow as tf

from nids.data import UNKNOWN_ID
from nids.infer import get_centroids, upstream_scorer
from nids.losses import clad_loss, closr_loss
from nids.metrics import fpr_at_recall, mean_auroc, open_auc
from nids.metrics_eval import compute_closr_metrics, compute_recalls
from nids.models import ResidualBlock
from tests.unit.np_reference import (
    np_clad_loss,
    np_closr_loss,
    np_closr_metrics,
    np_fpr_at_recall,
    np_get_centroids,
    np_mean_auroc,
    np_open_set_score,
)


def _to_upstream_labels(labels: np.ndarray, n_known: int) -> np.ndarray:
    """Adapt project UNKNOWN_ID labels to upstream's contiguous unknown-label layout."""
    mapped = np.asarray(labels, dtype=np.int64).copy()
    mapped[mapped == UNKNOWN_ID] = n_known
    return mapped


def test_clad_loss_matches_upstream_numpy_reference() -> None:
    """Regress CLAD loss against fixed-seed independent NumPy transcription."""
    rng = np.random.default_rng(413)
    features = rng.normal(size=(7, 5)).astype(np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
    actual = float(clad_loss(tf.constant(features), tf.constant(labels)))
    expected = np_clad_loss(features, labels)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_closr_loss_matches_upstream_numpy_reference() -> None:
    """Regress CLOSR loss against fixed-seed independent NumPy transcription."""
    rng = np.random.default_rng(414)
    features = rng.normal(size=(8, 3, 5)).astype(np.float32)
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int32)
    actual = float(closr_loss(tf.constant(features), tf.constant(labels)))
    expected = np_closr_loss(features, labels)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_open_set_score_matches_upstream_numpy_reference() -> None:
    """Regress the upstream quadratic softmax-weighted open-set score."""
    similarities = np.array([[0.9, 0.2, -0.1], [0.1, 0.7, 0.4]], dtype=np.float32)
    np.testing.assert_allclose(
        upstream_scorer(similarities), np_open_set_score(similarities), rtol=1e-6, atol=1e-7
    )


class IdentityEmbeddingModel(tf.keras.Model):
    """Identity model for testing class-centroid aggregation on fixed embeddings."""

    d_out = 3

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return inputs


def test_centroids_match_upstream_numpy_reference() -> None:
    """Regress normalized class centroids against upstream's NumPy transcription."""
    embeddings = np.array(
        [
            [[1, 0, 0], [0, 1, 0]],
            [[1, 1, 0], [0, 1, 1]],
            [[0, 1, 0], [1, 0, 0]],
            [[0, 1, 1], [1, 0, 1]],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    actual = get_centroids(IdentityEmbeddingModel(), embeddings, labels, 2, batch_size=2)
    expected = np_get_centroids(embeddings, labels, 2)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_fpr_at_recall_matches_upstream_after_unknown_adapter() -> None:
    """Regress zero-day-positive FPR@recall with project unknown-label adaptation."""
    labels = np.array([0, 1, 2, UNKNOWN_ID, 0, UNKNOWN_ID], dtype=np.int64)
    upstream_labels = _to_upstream_labels(labels, n_known=3)
    zero_day = (upstream_labels >= 3).astype(np.int64)
    scores = np.array([0.95, 0.85, 0.8, 0.2, 0.75, 0.1], dtype=np.float64)
    actual = fpr_at_recall(zero_day, -scores)
    expected = np_fpr_at_recall(zero_day, -scores)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_mean_auroc_matches_upstream_after_unknown_adapter() -> None:
    """Regress one-vs-benign mean AUROC with UNKNOWN_ID mapped to upstream layout."""
    labels = np.array([0, 1, 1, 2, 2, UNKNOWN_ID, UNKNOWN_ID], dtype=np.int64)
    upstream_labels = _to_upstream_labels(labels, n_known=3)
    scores = np.array([0.9, 0.7, 0.2, 0.8, 0.1, 0.15, 0.05])
    actual = mean_auroc(scores, upstream_labels)
    expected = np_mean_auroc(scores, upstream_labels)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_malicious_recall_and_open_auc_match_upstream_after_unknown_adapter() -> None:
    """Regress malicious recall and OpenAUC using the explicit unknown-label adapter."""
    labels = np.array([0, 0, 1, 1, 2, 2, UNKNOWN_ID, UNKNOWN_ID], dtype=np.int64)
    upstream_labels = _to_upstream_labels(labels, n_known=3)
    similarities = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.2, 0.8, 0.1],
            [0.3, 0.7, 0.1],
            [0.1, 0.2, 0.9],
            [0.1, 0.3, 0.8],
            [0.4, 0.3, 0.2],
            [0.3, 0.4, 0.1],
        ],
        dtype=np.float32,
    )
    predictions = np.argmax(similarities, axis=1)
    scores = upstream_scorer(similarities)
    actual = compute_closr_metrics(upstream_labels, predictions, similarities, scores, num_known=3)
    expected = np_closr_metrics(upstream_labels, similarities, num_known=3)
    np.testing.assert_allclose(actual["mal_recall"], expected["mal_recall"])
    np.testing.assert_allclose(actual["open_auc"], expected["open_auc"])
    np.testing.assert_allclose(
        open_auc(actual["closed_set_acc"], actual["open_set_auc"]), expected["open_auc"]
    )
    _, recall = compute_recalls(upstream_labels, predictions, 3, similarities)
    np.testing.assert_allclose(recall, expected["mal_recall"])


def test_keras_torch_linear_initializer_bounds() -> None:
    """Verify torch_default Keras kernels and biases stay within Linear fan-in bounds."""
    fan_in = 64
    block = ResidualBlock(fan_in, 32, residual=True, init="torch_default")
    block(tf.zeros((2, fan_in)), training=False)
    bound = 1.0 / np.sqrt(fan_in)
    for dense in (block.dense, block.resize):
        assert np.max(np.abs(dense.kernel.numpy())) <= bound
        assert np.max(np.abs(dense.bias.numpy())) <= bound
