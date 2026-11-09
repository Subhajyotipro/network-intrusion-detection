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
"""Per-class metric summary table generation for CLOSR and CLAD evaluation."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from nids.data import BENIGN_ID, ClassMap
from nids.metrics import fpr_at_recall, safe_auroc

__all__ = [
    "build_closr_per_class_table",
    "build_clad_per_class_table",
]


def _calc_prf(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> tuple[float, float, float]:
    """Calculate precision, recall, and f1 score for a single class."""
    n_true = np.sum(y_true == c)
    n_pred = np.sum(y_pred == c)
    tp = np.sum((y_true == c) & (y_pred == c))
    prec = float(tp / n_pred) if n_pred > 0 else float("nan")
    rec = float(tp / n_true) if n_true > 0 else float("nan")
    if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0.0:
        f1 = float("nan")
    else:
        f1 = float(2.0 * prec * rec / (prec + rec))
    return prec, rec, f1


def _build_closr_known_rows(
    class_map: ClassMap, y_true: np.ndarray, y_pred: np.ndarray, osr_scores: np.ndarray
) -> list[dict[str, Any]]:
    """Build known-class metric rows for CLOSR."""
    rows = []
    for c, name in enumerate(class_map.names):
        mask = y_true == c
        n_samples = int(np.sum(mask))
        prec, rec, f1 = _calc_prf(y_true, y_pred, c)
        mean_s = float(np.mean(osr_scores[mask])) if n_samples > 0 else float("nan")
        rows.append(
            {
                "class_id": c,
                "class_name": name,
                "category": "known",
                "n_samples": n_samples,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "mean_score": mean_s,
            }
        )
    return rows


def _build_closr_zd_rows(
    class_map: ClassMap,
    y_true: np.ndarray,
    osr_scores: np.ndarray,
    zd_labels: Optional[np.ndarray],
) -> list[dict[str, Any]]:
    """Build zero-day metric rows for CLOSR."""
    rows = []
    num_known = class_map.n_known
    for i, zd_name in enumerate(class_map.zero_day_names):
        zd_id = num_known + i
        mask = (zd_labels == zd_name) if zd_labels is not None else (y_true == zd_id)
        n_samples = int(np.sum(mask))
        mean_s = float(np.mean(osr_scores[mask])) if n_samples > 0 else float("nan")
        rows.append(
            {
                "class_id": zd_id,
                "class_name": zd_name,
                "category": "zero_day",
                "n_samples": n_samples,
                "precision": float("nan"),
                "recall": float("nan"),
                "f1_score": float("nan"),
                "mean_score": mean_s,
            }
        )
    return rows


def build_closr_per_class_table(
    class_map: ClassMap,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sims: np.ndarray,
    osr_scores: np.ndarray,
    zd_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Build per-class metric summary DataFrame for CLOSR model."""
    del sims
    rows = _build_closr_known_rows(class_map, y_true, y_pred, osr_scores)
    rows.extend(_build_closr_zd_rows(class_map, y_true, osr_scores, zd_labels))
    return pd.DataFrame(rows)


def _eval_clad_pair(
    benign_mask: np.ndarray,
    class_mask: np.ndarray,
    y_true: np.ndarray,
    sims: np.ndarray,
) -> tuple[float, float, float]:
    """Compute AUROC, FPR@95, and mean similarity for a class against benign."""
    n_samples = int(np.sum(class_mask))
    if n_samples == 0:
        return float("nan"), float("nan"), float("nan")
    pair_mask = benign_mask | class_mask
    y_bin = (y_true[pair_mask] > BENIGN_ID).astype(int)
    roc = safe_auroc(y_bin, sims[pair_mask])
    if not np.isnan(roc):
        roc = max(roc, 1.0 - roc)
    fpr = fpr_at_recall(y_bin, -sims[pair_mask], 0.95)
    mean_sim = float(np.mean(sims[class_mask]))
    return roc, fpr, mean_sim


def _build_clad_known_rows(
    class_map: ClassMap, y_true: np.ndarray, sims: np.ndarray, benign_mask: np.ndarray
) -> list[dict[str, Any]]:
    """Build known attack class metric rows for CLAD."""
    rows = []
    for c in range(1, class_map.n_known):
        name = class_map.names[c]
        mask = y_true == c
        roc, fpr, mean_sim = _eval_clad_pair(benign_mask, mask, y_true, sims)
        rows.append(
            {
                "class_id": c,
                "class_name": name,
                "category": "known_attack",
                "n_samples": int(np.sum(mask)),
                "auroc": roc,
                "fpr_at_95": fpr,
                "mean_similarity": mean_sim,
            }
        )
    return rows


def _build_clad_zd_rows(
    class_map: ClassMap,
    y_true: np.ndarray,
    sims: np.ndarray,
    benign_mask: np.ndarray,
    zd_labels: Optional[np.ndarray],
) -> list[dict[str, Any]]:
    """Build zero-day metric rows for CLAD."""
    rows = []
    num_known = class_map.n_known
    for i, zd_name in enumerate(class_map.zero_day_names):
        zd_id = num_known + i
        mask = (zd_labels == zd_name) if zd_labels is not None else (y_true == zd_id)
        roc, fpr, mean_sim = _eval_clad_pair(benign_mask, mask, y_true, sims)
        rows.append(
            {
                "class_id": zd_id,
                "class_name": zd_name,
                "category": "zero_day",
                "n_samples": int(np.sum(mask)),
                "auroc": roc,
                "fpr_at_95": fpr,
                "mean_similarity": mean_sim,
            }
        )
    return rows


def build_clad_per_class_table(
    class_map: ClassMap,
    y_true: np.ndarray,
    sims: np.ndarray,
    zd_labels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Build per-class metric summary DataFrame for CLAD model."""
    benign_mask = y_true == BENIGN_ID
    n_benign = int(np.sum(benign_mask))
    mean_b = float(np.mean(sims[benign_mask])) if n_benign > 0 else float("nan")
    rows = [
        {
            "class_id": BENIGN_ID,
            "class_name": class_map.names[BENIGN_ID],
            "category": "benign",
            "n_samples": n_benign,
            "auroc": float("nan"),
            "fpr_at_95": float("nan"),
            "mean_similarity": mean_b,
        }
    ]
    rows.extend(_build_clad_known_rows(class_map, y_true, sims, benign_mask))
    rows.extend(_build_clad_zd_rows(class_map, y_true, sims, benign_mask, zd_labels))
    return pd.DataFrame(rows)
