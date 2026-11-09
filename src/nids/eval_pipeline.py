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
"""Evaluation pipeline execution, array export, and result persistence."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from nids.checkpoint import load_checkpoint
from nids.config import Config, get_run_metadata
from nids.data import BENIGN_ID
from nids.infer import (
    get_centroids,
    get_clad_centroid,
    stream_evaluate_clad,
    stream_evaluate_closr,
)
from nids.metrics_eval import compute_clad_metrics, compute_closr_metrics
from nids.models import get_model
from nids.tables import build_clad_per_class_table, build_closr_per_class_table

logger = logging.getLogger("nids.eval_pipeline")

__all__ = ["resolve_checkpoint", "prepare_eval_data", "run_eval_pipeline"]


def resolve_checkpoint(cfg: Config, override: Optional[str]) -> Path:
    """Resolve path to checkpoint file or raise FileNotFoundError."""
    if override:
        p = Path(override)
        if not p.is_file() and not Path(str(p) + ".weights.h5").is_file():
            raise FileNotFoundError(f"Specified checkpoint not found: {p}")
        return p
    base = Path(cfg.weights_dir) / f"{cfg.model_kind}_seed{cfg.seed}"
    if Path(str(base) + ".weights.h5").is_file():
        return base
    raise FileNotFoundError(f"No checkpoint found for {cfg.model_kind} seed {cfg.seed} at {base}")


def prepare_eval_data(
    cfg: Config, data: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Prepare concatenated test and zero-day arrays with stratified subsampling if configured."""
    x_test, y_test = data.x_test, data.y_test
    reduced = False
    if cfg.eval_max_rows > 0 and len(x_test) > cfg.eval_max_rows:
        x_test, _, y_test, _ = train_test_split(
            x_test,
            y_test,
            train_size=cfg.eval_max_rows,
            stratify=y_test,
            random_state=cfg.split_seed,
        )
        reduced = True

    x_zd = data.x_zd if data.x_zd is not None else np.empty((0, data.d_in), dtype=np.float32)
    y_zd = data.y_zd if data.y_zd is not None else np.empty((0,), dtype=np.int64)
    zd_names = data.zd_names if data.zd_names is not None else np.empty((0,), dtype=object)

    x_eval = np.concatenate([x_test, x_zd], axis=0) if len(x_zd) > 0 else x_test
    y_eval = np.concatenate([y_test, y_zd], axis=0) if len(y_zd) > 0 else y_test
    zd_labels = (
        np.concatenate([np.array(["known"] * len(y_test)), zd_names])
        if len(x_zd) > 0
        else np.array(["known"] * len(y_test))
    )

    y_mapped = y_eval.copy()
    for i, zd_name in enumerate(data.class_map.zero_day_names):
        y_mapped[zd_labels == zd_name] = data.class_map.n_known + i

    return x_eval, y_eval, y_mapped, zd_labels, reduced


def _evaluate_model(
    cfg: Config,
    model: tf.keras.Model,
    data: Any,
    x_eval: np.ndarray,
    y_mapped: np.ndarray,
    zd_labels: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch evaluation by model kind and return metrics, table, and array outputs."""
    if cfg.model_kind == "closr":
        centroids = get_centroids(
            model, data.x_train, data.y_train, data.class_map.n_known, cfg.eval_batch_size
        )
        sims, scores, preds, embs, idx = stream_evaluate_closr(
            model, x_eval, centroids, cfg.eval_batch_size, cfg.tsne_max_points, cfg.seed
        )
        m = compute_closr_metrics(y_mapped, preds, sims, scores, data.class_map.n_known)
        df = build_closr_per_class_table(data.class_map, y_mapped, preds, sims, scores, zd_labels)
        return m, df, scores, preds, embs, idx

    centroid = get_clad_centroid(model, data.x_train, data.y_train, BENIGN_ID, cfg.eval_batch_size)
    sims, embs, idx = stream_evaluate_clad(
        model, x_eval, centroid, cfg.eval_batch_size, cfg.tsne_max_points, cfg.seed
    )
    preds = (sims < 0.5).astype(np.int64)
    m = compute_clad_metrics(y_mapped, sims, data.class_map.n_known - 1)
    df = build_clad_per_class_table(data.class_map, y_mapped, sims, zd_labels)
    return m, df, sims, preds, embs, idx


def _save_eval_artifacts(
    base: Path, record: dict[str, Any], df: pd.DataFrame, arrays: dict[str, Any]
) -> dict[str, str]:
    """Save JSON record, CSV summary, and NPZ arrays to disk."""
    json_path = base.with_name(f"{base.name}_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    csv_path = base.with_name(f"{base.name}_per_class.csv")
    df.to_csv(csv_path, index=False)

    npz_path = base.with_name(f"{base.name}_arrays.npz")
    np.savez_compressed(npz_path, **arrays)
    return {
        "eval_json": str(json_path),
        "per_class_csv": str(csv_path),
        "arrays_npz": str(npz_path),
    }


def _build_record(
    cfg: Config, m: dict[str, Any], meta: dict[str, Any], ds: str, ep: int, red: bool
) -> dict[str, Any]:
    """Construct metadata record for evaluation."""
    return {
        "metrics": m,
        "config": asdict(cfg),
        "seed": cfg.seed,
        "git_commit": meta["git_commit"],
        "versions": meta["versions"],
        "device": meta["device"],
        "dataset": ds,
        "epochs": ep,
        "reduced_run": bool(cfg.smoke or ep < 200),
        "reduced_eval": red,
    }


def _write_eval_record(
    cfg: Config,
    metrics: dict[str, Any],
    df: pd.DataFrame,
    arrays_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dataset_type: str,
    epochs_run: int,
    reduced_eval: bool,
    class_map: Any,
) -> dict[str, str]:
    """Assemble and write evaluation outputs to results/eval directory."""
    scores, preds, y_eval, embs, tsne_idx = arrays_tuple
    meta = get_run_metadata(cfg)
    exp_dir = Path(cfg.results_dir) / "eval"
    exp_dir.mkdir(parents=True, exist_ok=True)
    base = exp_dir / f"{cfg.model_kind}_seed{cfg.seed}"
    record = _build_record(cfg, metrics, meta, dataset_type, epochs_run, reduced_eval)
    tsne_y = y_eval[tsne_idx] if len(tsne_idx) > 0 else np.empty((0,), dtype=np.int64)
    arrays = {
        "scores": scores,
        "predictions": preds,
        "labels": y_eval,
        "zd_names": np.array(class_map.zero_day_names),
        "tsne_embeddings": embs,
        "tsne_labels": tsne_y,
    }
    return _save_eval_artifacts(base, record, df, arrays)


def run_eval_pipeline(
    cfg: Config,
    data: Any,
    dataset_type: str,
    checkpoint_override: Optional[str] = None,
) -> dict[str, str]:
    """Execute evaluation pipeline and write metrics JSON, CSV table, and arrays NPZ."""
    cfg.d_in = data.d_in
    model = get_model(cfg, d_in=data.d_in)
    ckpt_path = resolve_checkpoint(cfg, checkpoint_override)
    metadata = load_checkpoint(ckpt_path, model)

    x_eval, y_eval, y_mapped, zd_labels, red_eval = prepare_eval_data(cfg, data)
    res = _evaluate_model(cfg, model, data, x_eval, y_mapped, zd_labels)
    metrics, df, scores, preds, embs, tsne_idx = res

    epochs = metadata.get("epoch", cfg.epochs)
    arrays_tuple = (scores, preds, y_eval, embs, tsne_idx)
    paths = _write_eval_record(
        cfg, metrics, df, arrays_tuple, dataset_type, epochs, red_eval, data.class_map
    )
    logger.info("Evaluation complete (%s, seed %d):", cfg.model_kind, cfg.seed)
    for k, v in list(metrics.items())[:6]:
        logger.info("  %-25s: %.6f", k, v)
    return paths
