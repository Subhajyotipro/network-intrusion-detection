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
"""Evaluation metrics: AUROC, OpenAUC, FPR@recall, and metric calculation routines."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from nids.data import BENIGN_ID

logger = logging.getLogger("nids.metrics")

__all__ = [
    "to_zero_day_scores",
    "safe_auroc",
    "closed_set_accuracy",
    "open_set_auroc",
    "open_auc",
    "fpr_at_recall",
    "classwise_fpr_at_recall",
    "mean_auroc",
    "compute_mean_ap_and_pr_auc",
]


def to_zero_day_scores(known_scores: np.ndarray) -> np.ndarray:
    """Negate known-ness scores so higher values indicate zero-day (positive class)."""
    return -1.0 * np.asarray(known_scores, dtype=np.float64)


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC safely, returning NaN and logging a warning if single class."""
    y_arr = np.asarray(y_true)
    classes = np.unique(y_arr[~np.isnan(y_arr)])
    if len(classes) < 2:
        logger.warning(
            "safe_auroc called with %d class(es): %s; returning nan", len(classes), classes
        )
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError as err:
        logger.warning("safe_auroc failed with ValueError: %s; returning nan", err)
        return float("nan")


def closed_set_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    """Compute overall closed-set classification accuracy on known samples."""
    yt = y_true[mask] if mask is not None else y_true
    yp = y_pred[mask] if mask is not None else y_pred
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(yt == yp))


def open_set_auroc(
    y_true: np.ndarray, open_set_scores: np.ndarray, is_zd_mask: np.ndarray
) -> float:
    """Compute open-set zero-day detection AUROC (zero-day is positive class)."""
    y_bin = is_zd_mask.astype(int)
    zd_scores = to_zero_day_scores(open_set_scores)
    return safe_auroc(y_bin, zd_scores)


def open_auc(closed_acc: float, open_set_auc_score: float) -> float:
    """Compute joint OpenAUC metric as product of closed accuracy and open AUC."""
    if np.isnan(closed_acc) or np.isnan(open_set_auc_score):
        return float("nan")
    return float(closed_acc * open_set_auc_score)


def fpr_at_recall(y_true: np.ndarray, y_score: np.ndarray, recall_level: float = 0.95) -> float:
    """Compute false positive rate when true positive rate (recall) reaches target."""
    y_true_arr = (np.asarray(y_true) > 0).astype(bool)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    desc_idx = np.argsort(y_score_arr)[::-1]
    y_score_arr = y_score_arr[desc_idx]
    y_true_arr = y_true_arr[desc_idx]

    tps = np.cumsum(y_true_arr)
    fps = np.cumsum(~y_true_arr)
    total_pos = tps[-1]
    total_neg = np.sum(~y_true_arr)
    if total_pos == 0 or total_neg == 0:
        return float("nan")
    recall = tps / total_pos
    valid = recall >= recall_level
    if not np.any(valid):
        return 0.0
    cutoff = np.argmax(valid)
    return float(fps[cutoff] / total_neg)


def classwise_fpr_at_recall(
    scores: np.ndarray,
    y_true: np.ndarray,
    recall_level: float = 0.95,
    class_thres: Optional[int] = None,
) -> list[float]:
    """Compute class-wise FPR@recall against class 0, appending mean as last element."""
    y_true_arr = np.asarray(y_true).copy()
    if class_thres is not None:
        y_true_arr[y_true_arr < class_thres] = 0

    fpr_scores = []
    for c in np.unique(y_true_arr):
        if c == 0:
            continue
        class_mask = (y_true_arr == 0) | (y_true_arr == c)
        y = (y_true_arr[class_mask] > 0).astype(int)
        x = scores[class_mask]
        fpr_scores.append(fpr_at_recall(y, x, recall_level))
    mean_val = float(np.mean(fpr_scores)) if fpr_scores else float("nan")
    fpr_scores.append(mean_val)
    return fpr_scores


def mean_auroc(
    scores: np.ndarray,
    y_true: np.ndarray,
    return_class_level: bool = False,
    include_lower_thres: bool = True,
) -> list[float] | float:
    """Compute mean one-vs-benign AUROC across non-benign classes."""
    y_true_arr = np.asarray(y_true)
    scores_arr = np.asarray(scores)
    roc_scores = []
    for c in np.unique(y_true_arr):
        if c == BENIGN_ID:
            continue
        class_mask = (y_true_arr == BENIGN_ID) | (y_true_arr == c)
        y = (y_true_arr[class_mask] > 0).astype(int)
        roc = safe_auroc(y, scores_arr[class_mask])
        if not np.isnan(roc):
            roc = max(roc, 1.0 - roc) if include_lower_thres else roc
        roc_scores.append(roc)
    mean_val = float(np.mean(roc_scores)) if roc_scores else float("nan")
    roc_scores.append(mean_val)
    return roc_scores if return_class_level else roc_scores[-1]


def compute_mean_ap_and_pr_auc(
    test_labels: np.ndarray, test_probs: np.ndarray, num_known: int, reference_class: int = 0
) -> tuple[float, float]:
    """Compute macro-mean Average Precision and PR-AUC vs reference class."""
    ap_scores, pr_auc_scores = [], []
    for c in range(num_known):
        if c == reference_class:
            continue
        mask = np.isin(test_labels, [reference_class, c])
        y_true = (test_labels[mask] == c).astype(int)
        if len(np.unique(y_true)) < 2:
            continue
        y_score = test_probs[mask][:, c]
        ap_scores.append(float(average_precision_score(y_true, y_score)))
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        recall_unique, idx = np.unique(recall, return_index=True)
        pr_auc_scores.append(float(auc(recall_unique, precision[idx])))
    mean_ap = float(np.mean(ap_scores)) if ap_scores else float("nan")
    mean_pr = float(np.mean(pr_auc_scores)) if pr_auc_scores else float("nan")
    return mean_ap, mean_pr
