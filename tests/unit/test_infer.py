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
"""Unit tests for inference routines: embedding extraction, centroids, similarity, and streaming."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from nids.config import Config
from nids.infer import (
    SCORERS,
    calc_class_sims,
    get_centroids,
    get_clad_centroid,
    get_embeddings,
    predict_closed_set,
    stream_evaluate_clad,
    stream_evaluate_closr,
    upstream_scorer,
)
from nids.models import get_model
from tests.unit.np_reference import (
    np_calc_class_sims,
    np_closed_set_pred,
    np_get_centroids,
    np_get_clad_centroid,
    np_open_set_score,
)


@pytest.fixture
def closr_toy_model() -> tf.keras.Model:
    """Fixture returning a built CLOSR model on toy dimensions."""
    cfg = Config(
        model_kind="closr",
        d_in=10,
        d_out=16,
        n_classes=4,
        hidden=[32, 32],
        dropout=0.0,
    )
    model = get_model(cfg, d_in=10)
    model(tf.zeros((2, 10)))
    return model


@pytest.fixture
def clad_toy_model() -> tf.keras.Model:
    """Fixture returning a built CLAD model on toy dimensions."""
    cfg = Config(
        model_kind="clad",
        d_in=10,
        d_out=8,
        n_classes=4,
        hidden=[32, 32],
        dropout=0.0,
    )
    model = get_model(cfg, d_in=10)
    model(tf.zeros((2, 10)))
    return model


def test_get_embeddings_chunk_independence(closr_toy_model: tf.keras.Model) -> None:
    """Verify get_embeddings gives identical results for chunk sizes 1, 7, and N."""
    rng = np.random.default_rng(12345)
    x = rng.standard_normal((35, 10), dtype=np.float32)

    emb_all = get_embeddings(closr_toy_model, x, batch_size=35)
    emb_7 = get_embeddings(closr_toy_model, x, batch_size=7)
    emb_1 = get_embeddings(closr_toy_model, x, batch_size=1)

    np.testing.assert_allclose(emb_7, emb_all, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(emb_1, emb_all, rtol=1e-5, atol=1e-6)


def test_get_embeddings_deterministic_eval(closr_toy_model: tf.keras.Model) -> None:
    """Verify inference is deterministic across repeated calls (training=False)."""
    rng = np.random.default_rng(23456)
    x = rng.standard_normal((20, 10), dtype=np.float32)

    run1 = get_embeddings(closr_toy_model, x, batch_size=10)
    run2 = get_embeddings(closr_toy_model, x, batch_size=10)
    np.testing.assert_array_equal(run1, run2)


def test_centroids_unit_norm_and_numpy_reference(closr_toy_model: tf.keras.Model) -> None:
    """Verify get_centroids outputs unit-norm vectors matching the NumPy reference."""
    rng = np.random.default_rng(34567)
    n_samples, n_classes = 50, 4
    x_train = rng.standard_normal((n_samples, 10), dtype=np.float32)
    y_train = rng.integers(0, n_classes, size=(n_samples,), dtype=np.int64)

    tf_centroids = get_centroids(closr_toy_model, x_train, y_train, n_classes, batch_size=16)

    # Unit norm check
    norms = np.linalg.norm(tf_centroids, axis=-1)
    np.testing.assert_allclose(norms, np.ones(n_classes), rtol=1e-5, atol=1e-6)

    # Reference check using full-batch embeddings
    full_embs = closr_toy_model(x_train, training=False).numpy()
    ref_centroids = np_get_centroids(full_embs, y_train, n_classes)
    np.testing.assert_allclose(tf_centroids, ref_centroids, rtol=1e-5, atol=1e-6)


def test_clad_centroid_unit_norm_and_numpy_reference(clad_toy_model: tf.keras.Model) -> None:
    """Verify get_clad_centroid outputs unit-norm vector matching the NumPy reference."""
    rng = np.random.default_rng(45678)
    n_samples = 40
    x_train = rng.standard_normal((n_samples, 10), dtype=np.float32)
    y_train = rng.choice([0, 1, 2], size=(n_samples,), p=[0.6, 0.2, 0.2])

    tf_centroid = get_clad_centroid(clad_toy_model, x_train, y_train, benign_id=0, batch_size=12)

    norm = np.linalg.norm(tf_centroid)
    assert np.isclose(norm, 1.0, atol=1e-6)

    full_embs = clad_toy_model(x_train, training=False).numpy()
    ref_centroid = np_get_clad_centroid(full_embs, y_train, benign_id=0)
    np.testing.assert_allclose(tf_centroid, ref_centroid, rtol=1e-5, atol=1e-6)


def test_calc_class_sims_matches_reference() -> None:
    """Verify calc_class_sims matches np_calc_class_sims for both CLOSR and CLAD."""
    rng = np.random.default_rng(56789)
    # CLOSR: [B, C, D] and [C, D]
    embs_3d = rng.standard_normal((15, 4, 16), dtype=np.float32)
    cents_2d = rng.standard_normal((4, 16), dtype=np.float32)
    sims_3d = calc_class_sims(embs_3d, cents_2d)
    ref_3d = np_calc_class_sims(embs_3d, cents_2d)
    np.testing.assert_allclose(sims_3d, ref_3d, rtol=1e-5, atol=1e-6)

    # CLAD: [B, D] and [D]
    embs_2d = rng.standard_normal((15, 8), dtype=np.float32)
    cent_1d = rng.standard_normal((8,), dtype=np.float32)
    sims_2d = calc_class_sims(embs_2d, cent_1d)
    ref_2d = np_calc_class_sims(embs_2d, cent_1d)
    np.testing.assert_allclose(sims_2d, ref_2d, rtol=1e-5, atol=1e-6)


def test_upstream_scorer_matches_reference() -> None:
    """Verify upstream_scorer and predict_closed_set match NumPy reference."""
    rng = np.random.default_rng(67890)
    sims = rng.uniform(-1.0, 1.0, size=(25, 4)).astype(np.float32)

    score = upstream_scorer(sims)
    ref_score = np_open_set_score(sims)
    np.testing.assert_allclose(score, ref_score, rtol=1e-5, atol=1e-6)
    assert "upstream" in SCORERS
    assert SCORERS["upstream"] is upstream_scorer

    preds = predict_closed_set(sims)
    ref_preds = np_closed_set_pred(sims)
    np.testing.assert_array_equal(preds, ref_preds)


def test_streaming_eval_memory_bounded_and_exact(closr_toy_model: tf.keras.Model) -> None:
    """Verify streaming eval produces exact same results as non-chunked, bounding memory."""
    rng = np.random.default_rng(78901)
    n_samples, n_classes = 80, 4
    x = rng.standard_normal((n_samples, 10), dtype=np.float32)
    cents = rng.standard_normal((n_classes, 16), dtype=np.float32)
    cents /= np.linalg.norm(cents, axis=-1, keepdims=True)

    sims, scores, preds, embs, idx = stream_evaluate_closr(
        closr_toy_model, x, cents, batch_size=17, tsne_max_points=25, seed=999
    )

    full_embs = closr_toy_model(x, training=False).numpy()
    expected_sims = calc_class_sims(full_embs, cents)
    expected_scores = upstream_scorer(expected_sims)
    expected_preds = predict_closed_set(expected_sims)

    np.testing.assert_allclose(sims, expected_sims, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(preds, expected_preds)

    assert len(embs) == 25
    assert len(idx) == 25
    assert embs.shape == (25, n_classes, 16)
    np.testing.assert_allclose(embs, full_embs[idx], rtol=1e-5, atol=1e-6)


def test_streaming_eval_clad(clad_toy_model: tf.keras.Model) -> None:
    """Verify streaming eval for CLAD model matches full-batch calculation."""
    rng = np.random.default_rng(89012)
    x = rng.standard_normal((60, 10), dtype=np.float32)
    cent = rng.standard_normal((8,), dtype=np.float32)
    cent /= np.linalg.norm(cent)

    sims, embs, idx = stream_evaluate_clad(
        clad_toy_model, x, cent, batch_size=15, tsne_max_points=20, seed=123
    )

    full_embs = clad_toy_model(x, training=False).numpy()
    expected_sims = calc_class_sims(full_embs, cent)

    np.testing.assert_allclose(sims, expected_sims, rtol=1e-5, atol=1e-6)
    assert len(embs) == 20
    assert len(idx) == 20
    np.testing.assert_allclose(embs, full_embs[idx], rtol=1e-5, atol=1e-6)


def test_batch_size_one(closr_toy_model: tf.keras.Model, clad_toy_model: tf.keras.Model) -> None:
    """Verify inference pipeline operates cleanly with batch size 1."""
    x_single = np.ones((1, 10), dtype=np.float32)
    emb_closr = get_embeddings(closr_toy_model, x_single, batch_size=1)
    assert emb_closr.shape == (1, 4, 16)

    emb_clad = get_embeddings(clad_toy_model, x_single, batch_size=1)
    assert emb_clad.shape == (1, 8)
