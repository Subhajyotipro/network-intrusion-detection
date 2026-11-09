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
from nids.metrics import closed_set_accuracy, open_auc, open_set_auroc

logger = logging.getLogger("nids.baselines")


class MLPBaseline:
    """Supervised MLP classifier; open-set score: max softmax probability (higher = known)."""

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

    def score_known(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Compute open-set score as max softmax probability (higher = more likely known)."""
        probs = self.predict_proba(x, batch_size=batch_size)
        return np.max(probs, axis=-1)


class AutoencoderBaseline:
    """Symmetric Autoencoder; open-set score: negative reconstruction error (higher = known)."""

    def __init__(self, d_in: int, hidden: list[int], latent_dim: int = 32) -> None:
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

    def score_known(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Compute open-set score as negative MSE reconstruction error (higher = known)."""
        recon = self.model.predict(x, batch_size=batch_size, verbose=0)
        mse = np.mean((x - recon) ** 2, axis=-1)
        return -1.0 * mse


class IsolationForestBaseline:
    """Isolation Forest; open-set score: sklearn score_samples (higher = known inlier)."""

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

    def score_known(self, x: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
        """Compute chunked score_samples (higher = more normal / likely known)."""
        scores = []
        for i in range(0, len(x), chunk_size):
            chunk = x[i : i + chunk_size]
            scores.append(self.model.score_samples(chunk))
        return np.concatenate(scores, axis=0) if scores else np.empty(0, dtype=np.float64)


def _eval_baseline_metrics(
    model_name: str,
    scores_test: np.ndarray,
    scores_zd: Optional[np.ndarray],
    y_test: np.ndarray,
    preds_test: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Compute standard metrics parity dictionary for re-run baseline."""
    has_classifier = preds_test is not None
    closed_acc = closed_set_accuracy(y_test, preds_test) if has_classifier else "n/a"

    if scores_zd is not None and len(scores_zd) > 0:
        all_scores = np.concatenate([scores_test, scores_zd], axis=0)
        is_zd = np.concatenate(
            [np.zeros(len(scores_test), dtype=bool), np.ones(len(scores_zd), dtype=bool)]
        )
        os_auc = open_set_auroc(is_zd.astype(int), all_scores, is_zd)
    else:
        os_auc = float("nan")

    o_auc = (
        open_auc(float(closed_acc), float(os_auc))
        if has_classifier and not np.isnan(float(os_auc))
        else "n/a"
    )
    return {
        "closed_set_acc": closed_acc,
        "open_set_auc": float(os_auc) if not np.isnan(os_auc) else "n/a",
        "open_auc": o_auc,
    }


def _fit_and_score_model(
    model_name: str, cfg: Config, splits: DatasetSplits
) -> tuple[Any, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit selected baseline model and return scores on test and zero-day splits."""
    d_in = splits.x_train.shape[-1]
    benign_mask = splits.y_train == BENIGN_ID
    x_benign = splits.x_train[benign_mask]
    epochs = 2 if cfg.smoke else cfg.epochs

    if model_name == "mlp":
        model = MLPBaseline(d_in, cfg.n_classes, cfg.hidden, cfg.dropout)
        model.fit(splits.x_train, splits.y_train, epochs, cfg.batch_size, cfg.lr_peak)
        s_test = model.score_known(splits.x_test)
        s_zd = model.score_known(splits.x_zd) if splits.x_zd is not None else None
        p_test = np.argmax(model.predict_proba(splits.x_test), axis=-1)
        return model, s_test, s_zd, p_test

    if model_name == "autoencoder":
        model = AutoencoderBaseline(d_in, hidden=[64, 32], latent_dim=16)
        model.fit(x_benign, epochs, cfg.batch_size, cfg.lr_peak)
        s_test = model.score_known(splits.x_test)
        s_zd = model.score_known(splits.x_zd) if splits.x_zd is not None else None
        return model, s_test, s_zd, None

    if model_name == "iforest":
        n_est = 10 if cfg.smoke else 100
        model = IsolationForestBaseline(n_estimators=n_est, random_state=cfg.seed)
        model.fit(x_benign)
        s_test = model.score_known(splits.x_test)
        s_zd = model.score_known(splits.x_zd) if splits.x_zd is not None else None
        return model, s_test, s_zd, None

    raise ValueError(f"Unknown baseline model: {model_name}")


def run_baseline(
    model_name: str,
    cfg: Config,
    splits: DatasetSplits,
    output_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Train and evaluate baseline model, writing standardized D4.7 results record."""
    _, s_test, s_zd, p_test = _fit_and_score_model(model_name, cfg, splits)
    metrics = _eval_baseline_metrics(model_name, s_test, s_zd, splits.y_test, p_test)

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
        "reduced_run": cfg.smoke,
        "seed": cfg.seed,
        "data_source": cfg.data_source,
        "versions": meta["versions"],
        "git_commit": meta["git_commit"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return {"record_path": str(out_path), "metrics": metrics}
