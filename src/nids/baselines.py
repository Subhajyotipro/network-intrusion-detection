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
"""Baseline models: MLP classifier, Autoencoder, and Isolation Forest."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest

from nids.config import Config, get_run_metadata
from nids.data import BENIGN_ID, DatasetSplits
from nids.metrics import closed_set_accuracy, mean_auroc, open_auc, open_set_auroc

logger = logging.getLogger("nids.baselines")


class MLPBaseline:
    """Supervised MLP classifier; Table 1 anomaly score: 1 - P(benign) (higher = anomalous)."""

    def __init__(self, d_in: int, n_classes: int, hidden: list[int], dropout: float = 0.1) -> None:
        inputs = tf.keras.Input(shape=(d_in,))
        x = inputs
        for h in hidden:
            x = tf.keras.layers.Dense(h, activation="relu")(x)
            if dropout > 0.0:
                x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.d_in = d_in
        self.n_classes = n_classes

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, lr: float) -> None:
        """Train classifier with AdamW optimizer and sparse categorical crossentropy."""
        opt = tf.keras.optimizers.AdamW(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model.compile(optimizer=opt, loss=loss)
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict_proba(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Compute softmax class probabilities."""
        return self.model.predict(x, batch_size=batch_size, verbose=0)

    def score_anomaly(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Table 1 anomaly score: 1 - P(benign) (higher = more anomalous)."""
        probs = self.predict_proba(x, batch_size=batch_size)
        return 1.0 - probs[:, BENIGN_ID]

    def score_known(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Table 2 supplementary score: max softmax probability (higher = known)."""
        probs = self.predict_proba(x, batch_size=batch_size)
        return np.max(probs, axis=-1)


class AutoencoderBaseline:
    """Symmetric Autoencoder; Table 1 anomaly score: reconstruction MSE (higher = anomalous)."""

    def __init__(self, d_in: int, hidden: list[int], latent_dim: int = 16) -> None:
        inputs = tf.keras.Input(shape=(d_in,))
        x = inputs
        for h in hidden:
            x = tf.keras.layers.Dense(h, activation="relu")(x)
        latent = tf.keras.layers.Dense(latent_dim, activation="relu")(x)
        x = latent
        for h in reversed(hidden):
            x = tf.keras.layers.Dense(h, activation="relu")(x)
        outputs = tf.keras.layers.Dense(d_in)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.d_in = d_in

    def fit(self, x: np.ndarray, epochs: int, batch_size: int, lr: float) -> None:
        """Fit reconstruction on normal/benign samples using AdamW and MSE loss."""
        opt = tf.keras.optimizers.AdamW(learning_rate=lr)
        self.model.compile(optimizer=opt, loss="mse")
        self.model.fit(x, x, epochs=epochs, batch_size=batch_size, verbose=0)

    def score_anomaly(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Table 1 anomaly score: MSE reconstruction error (higher = more anomalous)."""
        recon = self.model.predict(x, batch_size=batch_size, verbose=0)
        return np.mean((x - recon) ** 2, axis=-1)

    def score_known(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Negative reconstruction error for inlier scoring (higher = known)."""
        return -1.0 * self.score_anomaly(x, batch_size=batch_size)


class IsolationForestBaseline:
    """Isolation Forest; Table 1 anomaly score: -score_samples (higher = anomalous)."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        random_state: int = 42,
        max_samples: int | str = "auto",
    ) -> None:
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            max_samples=max_samples,
            n_jobs=-1,
        )

    def fit(self, x: np.ndarray) -> None:
        """Fit Isolation Forest on normal/benign training features."""
        self.model.fit(x)

    def score_anomaly(self, x: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        """Table 1 anomaly score: -score_samples (higher = more anomalous)."""
        return -1.0 * self.score_known(x, chunk_size=chunk_size)

    def score_known(self, x: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        """Chunked score_samples (higher = normal inlier)."""
        scores = []
        for i in range(0, len(x), chunk_size):
            chunk = x[i : i + chunk_size]
            scores.append(self.model.score_samples(chunk))
        return np.concatenate(scores, axis=0) if scores else np.empty(0, dtype=np.float64)


def _prepare_table1_eval_arrays(splits: DatasetSplits) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate test and zero-day samples mirroring upstream eval_clad.py:148-200."""
    x_parts = [splits.x_test]
    y_parts = [splits.y_test]
    if splits.x_zd is not None and splits.zd_names is not None:
        x_parts.append(splits.x_zd)
        zd_id_map = {
            name: splits.class_map.n_known + i
            for i, name in enumerate(splits.class_map.zero_day_names)
        }
        zd_labels = np.array([zd_id_map[name] for name in splits.zd_names], dtype=np.int64)
        y_parts.append(zd_labels)
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def _compute_table1_branch(
    scores: np.ndarray,
    y_eval: np.ndarray,
    splits: DatasetSplits,
    include_lower_thres: bool,
) -> dict[str, Any]:
    """Compute per-class and closed/open mean AUROCs for one include_lower_thres branch."""
    raw_res = mean_auroc(
        scores, y_eval, return_class_level=True, include_lower_thres=include_lower_thres
    )
    roc_list = list(raw_res) if isinstance(raw_res, list) else [float(raw_res)]
    n_known_attacks = splits.class_map.n_known - 1
    n_zd = len(splits.class_map.zero_day_names)

    closed_scores = roc_list[:n_known_attacks]
    open_scores = roc_list[n_known_attacks : n_known_attacks + n_zd]
    per_class: dict[str, float] = {}
    for i in range(1, splits.class_map.n_known):
        cname = splits.class_map.names[i]
        per_class[cname] = (
            float(closed_scores[i - 1]) if i - 1 < len(closed_scores) else float("nan")
        )
    for j, zname in enumerate(splits.class_map.zero_day_names):
        per_class[zname] = float(open_scores[j]) if j < len(open_scores) else float("nan")

    return {
        "closed_set_mean": float(np.mean(closed_scores)) if closed_scores else float("nan"),
        "open_set_mean": float(np.mean(open_scores)) if open_scores else float("nan"),
        "per_class": per_class,
    }


def _compute_supplementary_metrics(
    model_name: str,
    model: Any,
    splits: DatasetSplits,
) -> dict[str, Any]:
    """Compute Table 2 supplementary metrics for MLP only (B2)."""
    if model_name != "mlp":
        return {"closed_set_acc": "n/a", "open_set_auc": "n/a", "open_auc": "n/a"}
    probs = model.predict_proba(splits.x_test)
    preds = np.argmax(probs, axis=-1)
    acc = closed_set_accuracy(splits.y_test, preds)
    if splits.x_zd is not None and len(splits.x_zd) > 0:
        s_test = model.score_known(splits.x_test)
        s_zd = model.score_known(splits.x_zd)
        all_s = np.concatenate([s_test, s_zd], axis=0)
        is_zd = np.concatenate([np.zeros(len(s_test), dtype=bool), np.ones(len(s_zd), dtype=bool)])
        os_auc = float(open_set_auroc(is_zd.astype(int), all_s, is_zd))
    else:
        os_auc = float("nan")
    o_auc = open_auc(float(acc), float(os_auc)) if not np.isnan(os_auc) else "n/a"
    return {
        "closed_set_acc": acc,
        "open_set_auc": os_auc if not np.isnan(os_auc) else "n/a",
        "open_auc": o_auc,
    }


def _eval_baseline_model(
    model_name: str,
    model: Any,
    splits: DatasetSplits,
) -> dict[str, Any]:
    """Compute full Table 1 and supplementary baseline metrics."""
    x_eval, y_eval = _prepare_table1_eval_arrays(splits)
    scores = model.score_anomaly(x_eval)
    t1_true = _compute_table1_branch(scores, y_eval, splits, include_lower_thres=True)
    t1_false = _compute_table1_branch(scores, y_eval, splits, include_lower_thres=False)
    primary_branch = t1_false if model_name == "mlp" else t1_true

    supp = _compute_supplementary_metrics(model_name, model, splits)
    return {
        "closed_set_auc_vs_benign": primary_branch["closed_set_mean"],
        "open_set_auc_vs_benign": primary_branch["open_set_mean"],
        "table1_primary_setting": "include_lower_thres=False"
        if model_name == "mlp"
        else "include_lower_thres=True",
        "table1": {
            "include_lower_thres_true": t1_true,
            "include_lower_thres_false": t1_false,
        },
        **supp,
    }


def _fit_model(model_name: str, cfg: Config, splits: DatasetSplits) -> tuple[Any, bool]:
    """Fit baseline model and return model instance plus reduced_run flag."""
    d_in = splits.x_train.shape[-1]
    benign_mask = splits.y_train == BENIGN_ID
    x_benign = splits.x_train[benign_mask]
    epochs = 2 if cfg.smoke else cfg.baseline_epochs
    n_est = 10 if cfg.smoke else 100
    is_reduced = bool(
        cfg.smoke or cfg.baseline_epochs < 200 or (model_name == "iforest" and n_est != 100)
    )

    if model_name == "mlp":
        model = MLPBaseline(d_in, cfg.n_classes, cfg.hidden, cfg.dropout)
        model.fit(splits.x_train, splits.y_train, epochs, cfg.batch_size, cfg.baseline_lr)
        return model, is_reduced
    if model_name == "autoencoder":
        model = AutoencoderBaseline(d_in, hidden=[64, 32], latent_dim=16)
        model.fit(x_benign, epochs, cfg.batch_size, cfg.baseline_lr)
        return model, is_reduced
    if model_name == "iforest":
        model = IsolationForestBaseline(n_estimators=n_est, random_state=cfg.seed)
        model.fit(x_benign)
        return model, is_reduced
    raise ValueError(f"Unknown baseline model: {model_name}")


def run_baseline(
    model_name: str,
    cfg: Config,
    splits: DatasetSplits,
    output_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Train and evaluate baseline model, writing standardized D4.7 results record."""
    model, is_reduced = _fit_model(model_name, cfg, splits)
    metrics = _eval_baseline_model(model_name, model, splits)

    meta = get_run_metadata(cfg)
    out_dir = Path(output_dir or Path(cfg.results_dir) / "baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_synthetic" if cfg.data_source != "real" else ""
    out_path = out_dir / f"{model_name}_seed{cfg.seed}{suffix}.json"

    record = {
        "model": model_name,
        "metrics": metrics,
        "config": cfg.__dict__,
        "source": "re-run",
        "reduced_run": is_reduced,
        "seed": cfg.seed,
        "data_source": cfg.data_source,
        "versions": meta["versions"],
        "git_commit": meta["git_commit"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return {"record_path": str(out_path), "metrics": metrics}
