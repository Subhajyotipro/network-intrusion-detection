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
"""Independent pure-NumPy float64 reference implementation of CLAD and CLOSR loss formulas.

Transcribed directly from upstream CLOSR PyTorch source text:
- third_party/CLOSR/losses/clad_loss.py
- third_party/CLOSR/losses/closr_loss.py
- third_party/CLOSR/util/distance.py
Used strictly in the test tier for numerical equivalence verification without PyTorch execution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def np_cossim(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    """NumPy transcription of upstream cossim/cosdist in util/distance.py."""
    a = a.astype(np.float64)
    a_norm = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    if b is not None:
        b = b.astype(np.float64)
        b_norm = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    else:
        b_norm = a_norm.copy()
    similarity = a_norm @ b_norm.T
    dists = (1.0 - similarity) / 2.0
    return dists


def np_loss_calc(
    dists: np.ndarray,
    y: np.ndarray,
    target_class: int,
    margin: float = 1.0,
    eps: float = 1e-8,
    squared: bool = True,
    alpha: float = 0.5,
) -> tuple[float, float]:
    """NumPy transcription of upstream clad_calc / loss_calc in losses/."""
    dists = dists.astype(np.float64)
    y = np.asarray(y, dtype=np.int64)

    # In upstream:
    # eq_pairs = T.eq(T.unsqueeze(y, 0), T.unsqueeze(y, 1))
    # sim_mask = eq_pairs & (unsqueeze(y, 0) == target_class)
    # dissim_mask = (~eq_pairs) & (unsqueeze(y, 1) == target_class)
    y_row = y[None, :]  # shape [1, B]
    y_col = y[:, None]  # shape [B, 1]
    eq_pairs = y_row == y_col

    sim_mask = eq_pairs & (y_row == target_class)
    dissim_mask = (~eq_pairs) & (y_col == target_class)

    sim_dists = dists * sim_mask
    n_sim = np.sum((sim_dists > eps).astype(np.float64))

    sim_dists_calc = (sim_dists**2) if squared else sim_dists
    sim_loss = np.sum(sim_dists_calc) / (n_sim + eps)

    margin_tensor = np.full(dists.shape, margin, dtype=np.float64)
    dissim_dists = np.where(dissim_mask, dists, margin_tensor)

    dissim_diff = np.maximum(0.0, margin - dissim_dists)
    dissim_diff_calc = (dissim_diff**2) if squared else dissim_diff
    n_dissim = np.sum((dissim_diff_calc > eps).astype(np.float64))
    dissim_loss = np.sum(dissim_diff_calc) / (n_dissim + eps)

    total_dissim_pairs = np.sum(dissim_mask.astype(np.float64))
    frac_pos = n_dissim / (total_dissim_pairs + eps)

    loss = (alpha * sim_loss) + ((1.0 - alpha) * dissim_loss)
    return float(loss), float(frac_pos)


def np_clad_loss(
    x: np.ndarray,
    y: np.ndarray,
    target_class: int = 0,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> float:
    """Compute CLAD loss using pure NumPy float64 reference implementation."""
    dists = np_cossim(x)
    loss, _ = np_loss_calc(
        dists=dists,
        y=y,
        target_class=target_class,
        margin=margin,
        eps=eps,
        squared=squared,
        alpha=alpha,
    )
    return loss


def np_clad_fraction_pos(
    x: np.ndarray,
    y: np.ndarray,
    target_class: int = 0,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> float:
    """Compute fraction of positive dissimilar pairs using pure NumPy float64 reference."""
    dists = np_cossim(x)
    _, frac = np_loss_calc(
        dists=dists,
        y=y,
        target_class=target_class,
        margin=margin,
        eps=eps,
        squared=squared,
        alpha=alpha,
    )
    return frac


def np_closr_loss(
    x: np.ndarray,
    y: np.ndarray,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Compute CLOSR loss across all classes using pure NumPy float64 reference."""
    x = x.astype(np.float64)
    # x shape: [B, C, D]
    n_classes = x.shape[1]
    losses = []
    for c in range(n_classes):
        xc = x[:, c, :]
        dists_c = np_cossim(xc)
        l_c, _ = np_loss_calc(
            dists=dists_c,
            y=y,
            target_class=c,
            margin=margin,
            eps=eps,
            squared=squared,
            alpha=alpha,
        )
        losses.append(l_c)
    return float(np.mean(losses))


def np_closr_fraction_pos(
    x: np.ndarray,
    y: np.ndarray,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Compute CLOSR mean fraction of positive pairs across all classes using NumPy reference."""
    x = x.astype(np.float64)
    n_classes = x.shape[1]
    fracs = []
    for c in range(n_classes):
        xc = x[:, c, :]
        dists_c = np_cossim(xc)
        _, f_c = np_loss_calc(
            dists=dists_c,
            y=y,
            target_class=c,
            margin=margin,
            eps=eps,
            squared=squared,
            alpha=alpha,
        )
        fracs.append(f_c)
    return float(np.mean(fracs))


def np_warmup_cosine_schedule(
    step: int,
    start_val: float,
    end_val: float,
    ref_val: float,
    T_max: int,
    warmup_steps: int,
    plateau: bool = True,
) -> float:
    """Independent NumPy transcription of upstream WarmupCosineSchedule in util/schedules.py."""
    if step <= 0:
        return float(start_val)
    if plateau and step > T_max:
        return float(end_val)
    if step < warmup_steps:
        progress = float(step) / float(max(1, warmup_steps))
        val = start_val + progress * (ref_val - start_val)
    else:
        progress = float(step - warmup_steps) / float(max(1, T_max - warmup_steps))
        val = end_val + (ref_val - end_val) * 0.5 * (1.0 + np.cos(progress * np.pi))
        if ref_val <= end_val:
            val = min(end_val, val)
        else:
            val = max(end_val, val)
    return float(val)


def np_adamw_step(
    param: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    step: int,
    lr: float,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent NumPy transcription of one PyTorch AdamW step with decoupled weight decay."""
    param = param.astype(np.float64)
    grad = grad.astype(np.float64)
    m = m.astype(np.float64)
    v = v.astype(np.float64)

    # 1. Decoupled weight decay: w = w * (1 - lr * weight_decay)
    param_decayed = param * (1.0 - lr * weight_decay)

    # 2. Update biased 1st and 2nd moment estimates
    m_next = beta1 * m + (1.0 - beta1) * grad
    v_next = beta2 * v + (1.0 - beta2) * (grad**2)

    # 3. Compute bias-corrected estimates
    bias_correction1 = 1.0 - (beta1**step)
    bias_correction2 = 1.0 - (beta2**step)
    m_hat = m_next / bias_correction1
    v_hat = v_next / bias_correction2

    # 4. Update parameter using PyTorch epsilon placement: lr * m_hat / (sqrt(v_hat) + eps)
    denom = np.sqrt(v_hat) + eps
    param_next = param_decayed - lr * (m_hat / denom)

    return param_next, m_next, v_next


def np_get_centroids(
    train_embeddings: np.ndarray, train_labels: np.ndarray, n_classes: int
) -> np.ndarray:
    """NumPy transcription of upstream get_centroids in util/distance.py."""
    centroids = []
    for c in range(n_classes):
        ref_points = train_embeddings[:, c, :]
        ref_points = ref_points[train_labels == c]
        norm = np.maximum(np.linalg.norm(ref_points, axis=-1, keepdims=True), 1e-12)
        c_mean = np.mean(ref_points / norm, axis=0)
        c_norm = c_mean / np.maximum(np.linalg.norm(c_mean), 1e-12)
        centroids.append(c_norm)
    return np.stack(centroids, axis=0)


def np_get_clad_centroid(
    train_embeddings: np.ndarray, train_labels: np.ndarray, benign_id: int = 0
) -> np.ndarray:
    """NumPy transcription of upstream benign centroid in eval_clad.py."""
    pts = train_embeddings[train_labels == benign_id]
    c_mean = np.mean(pts, axis=0)
    c_norm = c_mean / np.maximum(np.linalg.norm(c_mean), 1e-12)
    return c_norm


def np_calc_class_sims(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """NumPy transcription of calc_class_sims_chunked and chunked_centroid_sims."""
    if embeddings.ndim == 3:
        e_norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-12)
        c_norm = centroids / np.maximum(np.linalg.norm(centroids, axis=-1, keepdims=True), 1e-12)
        return np.sum(e_norm * c_norm[None, :, :], axis=2)
    e_norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-12)
    c_norm = centroids / np.maximum(np.linalg.norm(centroids), 1e-12)
    return np.sum(e_norm * c_norm[None, :], axis=1)


def np_open_set_score(sims: np.ndarray) -> np.ndarray:
    """NumPy transcription of upstream open-set score in eval_closr.py."""
    sims = sims.astype(np.float64)
    exp_sims = np.exp(sims - np.max(sims, axis=-1, keepdims=True))
    probs = exp_sims / np.sum(exp_sims, axis=-1, keepdims=True)
    return np.sum((sims**2) * probs, axis=-1)


def np_closed_set_pred(sims: np.ndarray) -> np.ndarray:
    """NumPy transcription of closed-set argmax prediction."""
    return np.argmax(sims, axis=-1)


def np_fpr_at_recall(y_true: np.ndarray, y_score: np.ndarray, recall_level: float = 0.95) -> float:
    """NumPy transcription of upstream fpr_at_recall in util/metrics.py."""
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


def np_classwise_fpr_at_recall(
    scores: np.ndarray,
    y_true: np.ndarray,
    recall_level: float = 0.95,
    class_thres: Optional[int] = None,
) -> list[float]:
    """NumPy transcription of upstream classwise_fpr_at_recall in util/metrics.py."""
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
        fpr_scores.append(np_fpr_at_recall(y, x, recall_level))
    fpr_scores.append(float(np.mean(fpr_scores)))
    return fpr_scores


def np_mean_auroc(
    scores: np.ndarray,
    y_true: np.ndarray,
    return_class_level: bool = False,
    include_lower_thres: bool = True,
) -> list[float] | float:
    """NumPy transcription of upstream mean_auroc in util/metrics.py."""
    y_true_arr = np.asarray(y_true)
    scores_arr = np.asarray(scores)
    roc_scores = []
    for c in np.unique(y_true_arr):
        if c == 0:
            continue
        class_mask = (y_true_arr == 0) | (y_true_arr == c)
        y = (y_true_arr[class_mask] > 0).astype(int)
        x = scores_arr[class_mask]
        roc = float(roc_auc_score(y, x))
        if include_lower_thres:
            roc_scores.append(max(roc, 1.0 - roc))
        else:
            roc_scores.append(roc)
    roc_scores.append(float(np.mean(roc_scores)))
    if return_class_level:
        return roc_scores
    return roc_scores[-1]


def np_compute_mean_ap_and_pr_auc(
    test_labels: np.ndarray, test_probs: np.ndarray, num_known: int, reference_class: int = 0
) -> tuple[float, float]:
    """NumPy transcription of upstream compute_mean_ap_and_pr_auc in util/metrics.py."""
    ap_scores = []
    pr_auc_scores = []
    for c in range(num_known):
        if c == reference_class:
            continue
        mask = np.isin(test_labels, [reference_class, c])
        y_true = (test_labels[mask] == c).astype(int)
        y_score = test_probs[mask][:, c]

        ap = float(average_precision_score(y_true, y_score))
        ap_scores.append(ap)

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        recall_unique, idx = np.unique(recall, return_index=True)
        pr_auc = float(auc(recall_unique, precision[idx]))
        pr_auc_scores.append(pr_auc)
    return float(np.mean(ap_scores)), float(np.mean(pr_auc_scores))


def np_closr_metrics(
    test_labels: np.ndarray, test_sims: np.ndarray, num_known: int
) -> dict[str, float]:
    """NumPy transcription of upstream metrics in eval_closr.py."""
    test_preds = np.argmax(test_sims, axis=1)
    exp_sims = np.exp(test_sims - np.max(test_sims, axis=-1, keepdims=True))
    test_probs = exp_sims / np.sum(exp_sims, axis=-1, keepdims=True)
    osr_scores = np.sum((test_sims**2) * test_probs, axis=-1)

    known_mask = test_labels < num_known
    closed_acc = float(np.mean(test_labels[known_mask] == test_preds[known_mask]))

    ood_labels = (test_labels >= num_known).astype(int)
    open_set_auc = float(roc_auc_score(ood_labels, -1.0 * osr_scores))
    open_auc = closed_acc * open_set_auc
    fpr_95 = np_fpr_at_recall(ood_labels, -1.0 * osr_scores)
    mean_fpr_95 = np_classwise_fpr_at_recall(-1.0 * osr_scores, test_labels, class_thres=num_known)[
        -1
    ]

    mal_mask = (test_labels > 0) & (test_labels < num_known)
    mal_labels = test_labels[mal_mask] - 1
    mal_preds = np.argmax(test_sims[mal_mask][:, 1:], axis=-1)

    mean_recall = float(balanced_accuracy_score(test_labels[known_mask], test_preds[known_mask]))
    mal_recall = float(balanced_accuracy_score(mal_labels, mal_preds))
    macro_f1 = float(f1_score(test_labels[known_mask], test_preds[known_mask], average="macro"))
    mal_macro_f1 = float(f1_score(mal_labels, mal_preds, average="macro"))

    closed_mean_auroc = np_mean_auroc(
        scores=test_probs[known_mask][:, 0], y_true=test_labels[known_mask]
    )
    closed_auroc = np_mean_auroc(
        scores=test_probs[known_mask][:, 0], y_true=(test_labels[known_mask] > 0).astype(int)
    )

    benign_mask = test_labels == 0
    fp_rate = float(1.0 - (np.sum((benign_mask) & (test_preds == 0)) / np.sum(benign_mask)))

    ap_score, pr_score = np_compute_mean_ap_and_pr_auc(test_labels, test_probs, num_known)

    return {
        "closed_set_acc": closed_acc,
        "open_set_auc": open_set_auc,
        "open_auc": open_auc,
        "fpr_95": fpr_95,
        "mean_fpr_95": mean_fpr_95,
        "mean_recall": mean_recall,
        "mal_recall": mal_recall,
        "macro_f1": macro_f1,
        "mal_macro_f1": mal_macro_f1,
        "closed_set_mean_auroc": float(closed_mean_auroc),
        "closed_set_auroc": float(closed_auroc),
        "fp_rate": fp_rate,
        "mean_average_precision_score": ap_score,
        "mean_precision_recall_auc": pr_score,
    }


def np_clad_metrics(
    test_labels: np.ndarray, sims: np.ndarray, num_known_attacks: int
) -> dict[str, float]:
    """NumPy transcription of upstream metrics in eval_clad.py."""
    auroc_scores = np_mean_auroc(scores=sims, y_true=test_labels, return_class_level=True)[:-1]
    fpr_scores = np_classwise_fpr_at_recall(scores=-sims, y_true=test_labels)[:-1]

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
