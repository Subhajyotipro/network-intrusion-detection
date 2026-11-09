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
"""Evaluation aggregation routines for CLOSR and CLAD metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from nids.data import BENIGN_ID
from nids.metrics import (
    classwise_fpr_at_recall,
    closed_set_accuracy,
    compute_mean_ap_and_pr_auc,
    fpr_at_recall,
    mean_auroc,
    open_auc,
    open_set_auroc,
    to_zero_day_scores,
)

__all__ = [
    "compute_recalls",
    "compute_f1_scores",
    "compute_closr_metrics",
    "compute_clad_metrics",
]


def compute_recalls(
    y_true: np.ndarray, y_pred: np.ndarray, num_known: int, test_sims: np.ndarray
) -> tuple[float, float]:
    """Compute balanced recall on known classes and malicious-only known classes."""
    known_mask = y_true < num_known
    if not np.any(known_mask):
        return float("nan"), float("nan")
    mean_rec = float(balanced_accuracy_score(y_true[known_mask], y_pred[known_mask]))

    mal_mask = (y_true > BENIGN_ID) & (y_true < num_known)
    if not np.any(mal_mask):
        return mean_rec, float("nan")
    mal_labels = y_true[mal_mask] - 1
    mal_preds = np.argmax(test_sims[mal_mask][:, 1:], axis=-1)
    mal_rec = float(balanced_accuracy_score(mal_labels, mal_preds))
    return mean_rec, mal_rec


def compute_f1_scores(
    y_true: np.ndarray, y_pred: np.ndarray, num_known: int, test_sims: np.ndarray
) -> tuple[float, float]:
    """Compute macro-F1 on all known classes and malicious-only known classes."""
    known_mask = y_true < num_known
    if not np.any(known_mask):
        return float("nan"), float("nan")
    macro_f1 = float(f1_score(y_true[known_mask], y_pred[known_mask], average="macro"))

    mal_mask = (y_true > BENIGN_ID) & (y_true < num_known)
    if not np.any(mal_mask):
        return macro_f1, float("nan")
    mal_labels = y_true[mal_mask] - 1
    mal_preds = np.argmax(test_sims[mal_mask][:, 1:], axis=-1)
    mal_f1 = float(f1_score(mal_labels, mal_preds, average="macro"))
    return macro_f1, mal_f1


def _compute_closr_prob_metrics(
    test_sims: np.ndarray, y_true: np.ndarray, known_mask: np.ndarray, num_known: int
) -> tuple[float, float, float, float]:
    """Compute closed-set AUROC, mean AUROC, MAP, and PR-AUC from similarities."""
    exp_s = np.exp(test_sims - np.max(test_sims, axis=-1, keepdims=True))
    test_probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    mean_auc = float(mean_auroc(scores=test_probs[known_mask][:, 0], y_true=y_true[known_mask]))
    bin_y = (y_true[known_mask] > BENIGN_ID).astype(int)
    auc = float(mean_auroc(scores=test_probs[known_mask][:, 0], y_true=bin_y))
    mean_ap, mean_pr = compute_mean_ap_and_pr_auc(y_true, test_probs, num_known)
    return mean_auc, auc, mean_ap, mean_pr


def compute_closr_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_sims: np.ndarray,
    osr_scores: np.ndarray,
    num_known: int,
) -> dict[str, float]:
    """Compute all evaluation metrics specified in upstream eval_closr.py."""
    known_mask = y_true < num_known
    closed_acc = closed_set_accuracy(y_true, y_pred, mask=known_mask)
    is_zd = y_true >= num_known
    open_auc_val = open_set_auroc(y_true, osr_scores, is_zd)
    zd_scores = to_zero_day_scores(osr_scores)
    mean_rec, mal_rec = compute_recalls(y_true, y_pred, num_known, test_sims)
    macro_f1, mal_macro_f1 = compute_f1_scores(y_true, y_pred, num_known, test_sims)
    m_auc, auc, m_ap, m_pr = _compute_closr_prob_metrics(test_sims, y_true, known_mask, num_known)
    b_mask = y_true == BENIGN_ID
    fp_rate = (
        float(1.0 - (np.sum(b_mask & (y_pred == BENIGN_ID)) / np.sum(b_mask)))
        if np.any(b_mask)
        else float("nan")
    )
    return {
        "closed_set_acc": closed_acc,
        "open_set_auc": open_auc_val,
        "open_auc": open_auc(closed_acc, open_auc_val),
        "fpr_95": fpr_at_recall(is_zd.astype(int), zd_scores),
        "mean_fpr_95": classwise_fpr_at_recall(zd_scores, y_true, class_thres=num_known)[-1],
        "mean_recall": mean_rec,
        "mal_recall": mal_rec,
        "macro_f1": macro_f1,
        "mal_macro_f1": mal_macro_f1,
        "closed_set_mean_auroc": m_auc,
        "closed_set_auroc": auc,
        "fp_rate": fp_rate,
        "mean_average_precision_score": m_ap,
        "mean_precision_recall_auc": m_pr,
    }


def compute_clad_metrics(
    y_true: np.ndarray, sims: np.ndarray, num_known_attacks: int
) -> dict[str, float]:
    """Compute all evaluation metrics specified in upstream eval_clad.py."""
    auroc_scores = mean_auroc(scores=sims, y_true=y_true, return_class_level=True)[:-1]
    fpr_scores = classwise_fpr_at_recall(scores=-sims, y_true=y_true)[:-1]

    metrics = {}
    for i, auroc in enumerate(auroc_scores[:num_known_attacks]):
        metrics[f"known_class_{i + 1}_auroc"] = float(auroc)
    for i, auroc in enumerate(auroc_scores[num_known_attacks:]):
        metrics[f"zd_class_{i + 1}_auroc"] = float(auroc)
    metrics["known_mean_auroc"] = float(np.mean(auroc_scores[:num_known_attacks]))
    metrics["zd_mean_auroc"] = float(np.mean(auroc_scores[num_known_attacks:]))

    for i, fpr in enumerate(fpr_scores[:num_known_attacks]):
        metrics[f"known_class_{i + 1}_fpr@95"] = float(fpr)
    for i, fpr in enumerate(fpr_scores[num_known_attacks:]):
        metrics[f"zd_class_{i + 1}_fpr@95"] = float(fpr)
    metrics["known_mean_fpr_scores"] = float(np.mean(fpr_scores[:num_known_attacks]))
    metrics["zd_mean_fpr_scores"] = float(np.mean(fpr_scores[num_known_attacks:]))
    return metrics
