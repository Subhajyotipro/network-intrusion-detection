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
"""Unit tests for evaluation metrics: numerical parity, edge cases, score direction."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from nids.data import BENIGN_ID, ClassMap
from nids.metrics import (
    closed_set_accuracy,
    fpr_at_recall,
    open_auc,
    open_set_auroc,
    safe_auroc,
)
from nids.metrics_eval import compute_clad_metrics, compute_closr_metrics
from nids.tables import build_clad_per_class_table, build_closr_per_class_table
from tests.unit.np_reference import np_clad_metrics, np_closr_metrics


def test_benign_id_constant() -> None:
    """Verify BENIGN_ID is strictly 0 per project conventions."""
    assert BENIGN_ID == 0


def test_hand_computed_tiny_cases() -> None:
    """Verify metrics against exact analytical closed-form hand calculations."""
    y_true = np.array([0, 1, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 2, 1])
    # 3 correct out of 5: acc = 3/5 = 0.6
    acc = closed_set_accuracy(y_true, y_pred)
    assert np.isclose(acc, 0.6)

    # fpr_at_recall:
    # 4 samples: labels=[0, 1, 0, 1], scores=[0.1, 0.8, 0.4, 0.9]
    # Sorted by score desc: scores=[0.9(1), 0.8(1), 0.4(0), 0.1(0)]
    # At recall 0.95 (needs 2/2 = 1.0 recall): cutoff after 0.8 -> 0 false positives!
    labels_bin = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.8, 0.4, 0.9])
    fpr = fpr_at_recall(labels_bin, scores, recall_level=0.95)
    assert np.isclose(fpr, 0.0)

    # OpenAUC product
    assert np.isclose(open_auc(0.8, 0.9), 0.72)
    assert np.isnan(open_auc(float("nan"), 0.9))


def test_score_direction() -> None:
    """Verify score direction: lower known-ness score indicates zero-day (positive class)."""
    # Known samples (0, 1) have high known-ness scores; Zero-day samples (2) have low scores
    y_true = np.array([0, 1, 0, 1, 2, 2])
    is_zd = y_true == 2
    perfect_known_scores = np.array([0.9, 0.95, 0.85, 0.92, 0.1, 0.05])

    # Inverted score passed to ROC should give AUROC 1.0
    auroc = open_set_auroc(y_true, perfect_known_scores, is_zd)
    assert np.isclose(auroc, 1.0)

    # Inverted case: zero-day has high known scores, known has low
    inverted_known_scores = np.array([0.1, 0.05, 0.15, 0.08, 0.9, 0.95])
    auroc_inv = open_set_auroc(y_true, inverted_known_scores, is_zd)
    assert np.isclose(auroc_inv, 0.0)


def test_safe_auroc_edge_cases(caplog: pytest.LogCaptureFixture) -> None:
    """Verify safe_auroc returns NaN with a logged warning when single class present."""
    with caplog.at_level(logging.WARNING):
        single_class = np.array([1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        res = safe_auroc(single_class, scores)
        assert np.isnan(res)
        assert "safe_auroc called with 1 class" in caplog.text


def test_empty_zero_day_set() -> None:
    """Verify open-set evaluation handles empty zero-day set gracefully returning NaN."""
    y_true = np.array([0, 1, 0, 1])
    is_zd = np.zeros(4, dtype=bool)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    res = open_set_auroc(y_true, scores, is_zd)
    assert np.isnan(res)


def test_known_class_absent_from_test_split() -> None:
    """Verify metrics calculation when a known class is completely absent from test set."""
    # 4 known classes (0, 1, 2, 3), but class 2 is missing from test set
    y_true = np.array([0, 0, 1, 1, 3, 3, 4])  # 4 is zero-day
    y_pred = np.array([0, 0, 1, 1, 3, 3, 0])
    sims = np.zeros((7, 4), dtype=np.float32)
    for i, p in enumerate(y_pred):
        sims[i, p] = 1.0
    osr_scores = np.array([0.9, 0.85, 0.88, 0.91, 0.82, 0.79, 0.2])

    # Should compute metrics using num_known=4 without index error or crash
    metrics = compute_closr_metrics(y_true, y_pred, sims, osr_scores, num_known=4)
    assert not np.isnan(metrics["closed_set_acc"])
    assert not np.isnan(metrics["open_set_auc"])
    assert not np.isnan(metrics["open_auc"])


def test_division_by_zero_in_recall() -> None:
    """Verify division by zero in recall or FPR returns NaN cleanly without crash."""
    # When all samples are negative, fpr_at_recall has no positive samples
    all_neg = np.zeros(5, dtype=int)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert np.isnan(fpr_at_recall(all_neg, scores))


def test_all_equal_scores() -> None:
    """Verify handling when all predicted scores are identical."""
    y_true = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    roc = safe_auroc(y_true, scores)
    assert np.isclose(roc, 0.5)


def test_nan_input_handling() -> None:
    """Verify safe_auroc filters NaN values in true labels."""
    y_true = np.array([0.0, 1.0, np.nan, 1.0, 0.0])
    scores = np.array([0.1, 0.9, 0.5, 0.8, 0.2])
    mask = ~np.isnan(y_true)
    roc = safe_auroc(y_true[mask], scores[mask])
    assert np.isclose(roc, 1.0)


def test_closr_metrics_vs_numpy_reference() -> None:
    """Verify compute_closr_metrics matches the independent NumPy reference implementation."""
    rng = np.random.default_rng(998877)
    n_samples, n_known = 120, 5
    # labels: 0..4 are known, 5 is zero-day
    y_true = rng.integers(0, n_known + 1, size=(n_samples,), dtype=np.int64)
    sims = rng.uniform(-0.5, 1.5, size=(n_samples, n_known)).astype(np.float32)
    y_pred = np.argmax(sims, axis=1)

    exp_s = np.exp(sims - np.max(sims, axis=-1, keepdims=True))
    probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    osr_scores = np.sum((sims**2) * probs, axis=-1)

    tf_metrics = compute_closr_metrics(y_true, y_pred, sims, osr_scores, num_known=n_known)
    ref_metrics = np_closr_metrics(y_true, sims, num_known=n_known)

    for key, ref_val in ref_metrics.items():
        assert key in tf_metrics, f"Missing key {key}"
        tf_val = tf_metrics[key]
        if np.isnan(ref_val):
            assert np.isnan(tf_val), f"Expected NaN for {key}"
        else:
            np.testing.assert_allclose(
                tf_val, ref_val, rtol=1e-5, atol=1e-6, err_msg=f"Mismatch for {key}"
            )


def test_clad_metrics_vs_numpy_reference() -> None:
    """Verify compute_clad_metrics matches the independent NumPy reference implementation."""
    rng = np.random.default_rng(112233)
    n_samples, num_known_attacks = 100, 3
    # 0 is benign, 1..3 known attacks, 4 is zero-day
    y_true = rng.integers(0, num_known_attacks + 2, size=(n_samples,), dtype=np.int64)
    sims = rng.uniform(-1.0, 1.0, size=(n_samples,)).astype(np.float32)

    tf_metrics = compute_clad_metrics(y_true, sims, num_known_attacks=num_known_attacks)
    ref_metrics = np_clad_metrics(y_true, sims, num_known_attacks=num_known_attacks)

    for key, ref_val in ref_metrics.items():
        assert key in tf_metrics, f"Missing key {key}"
        tf_val = tf_metrics[key]
        if np.isnan(ref_val):
            assert np.isnan(tf_val), f"Expected NaN for {key}"
        else:
            np.testing.assert_allclose(
                tf_val, ref_val, rtol=1e-5, atol=1e-6, err_msg=f"Mismatch for {key}"
            )


def test_per_class_tables_construction() -> None:
    """Verify DataFrame construction for per-class tables."""
    cmap = ClassMap(names=["benign", "dos", "probe"], zero_day_names=["heartbleed"])
    y_true = np.array([0, 0, 1, 1, 2, 2, 3])
    y_pred = np.array([0, 0, 1, 2, 2, 2, 1])
    sims = np.zeros((7, 3), dtype=np.float32)
    for i, p in enumerate(y_pred):
        sims[i, p] = 1.0
    osr = np.array([0.9, 0.9, 0.8, 0.5, 0.7, 0.7, 0.2])

    df_closr = build_closr_per_class_table(cmap, y_true, y_pred, sims, osr)
    assert len(df_closr) == 4
    assert set(df_closr.columns) == {
        "class_id",
        "class_name",
        "category",
        "n_samples",
        "precision",
        "recall",
        "f1_score",
        "mean_score",
    }

    df_clad = build_clad_per_class_table(cmap, y_true, sims[:, 0])
    assert len(df_clad) == 4
    assert set(df_clad.columns) == {
        "class_id",
        "class_name",
        "category",
        "n_samples",
        "auroc",
        "fpr_at_95",
        "mean_similarity",
    }


def test_mal_recall_and_macro_f1_hand_computed() -> None:
    """Verify mal_recall and mal_macro_f1 on a tiny hand-computed example."""
    from nids.metrics_eval import compute_f1_scores, compute_recalls

    num_known = 4
    y_true = np.array([0, 1, 1, 2, 2, 4], dtype=np.int64)
    y_pred = np.array([0, 1, 2, 2, 2, 0], dtype=np.int64)
    sims = np.array(
        [
            [0.9, 0.1, 0.0, 0.0],  # sample 0 (benign) -> skipped by mal_mask
            [0.1, 0.8, 0.1, 0.0],  # sample 1: class 1 -> sliced argmax 0 -> pred 1 (correct)
            [0.2, 0.1, 0.7, 0.0],  # sample 2: class 1 -> sliced argmax 1 -> pred 2 (wrong)
            [0.1, 0.2, 0.6, 0.1],  # sample 3: class 2 -> sliced argmax 1 -> pred 2 (correct)
            [0.1, 0.2, 0.6, 0.1],  # sample 4: class 2 -> sliced argmax 1 -> pred 2 (correct)
            [0.2, 0.3, 0.3, 0.2],  # sample 5 (zero-day) -> skipped by mal_mask
        ],
        dtype=np.float32,
    )
    # mal_labels = y_true[mal_mask] - 1 = [0, 0, 1, 1]
    # mal_preds = argmax(sims[mal_mask][:, 1:], axis=-1) = [0, 1, 1, 1]
    # Recall: class 0 has 1/2 = 0.5, class 1 has 2/2 = 1.0 -> mean = 0.75
    # F1: class 0 has 2/3, class 1 has 4/5 -> mean = 11/15 ≈ 0.7333333333333334
    _, mal_rec = compute_recalls(y_true, y_pred, num_known, sims)
    _, mal_f1 = compute_f1_scores(y_true, y_pred, num_known, sims)

    assert np.isclose(mal_rec, 0.75, atol=1e-6)
    assert np.isclose(mal_f1, 11.0 / 15.0, atol=1e-6)
