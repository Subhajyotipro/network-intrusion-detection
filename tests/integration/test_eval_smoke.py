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
"""End-to-end integration test verifying training and evaluation CLI workflows on synthetic data."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nids.config import load_config
from nids.data import generate_synthetic_data
from nids.eval_pipeline import run_eval_pipeline


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Execute command in subprocess and verify zero exit status."""
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Command failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return res


def test_closr_eval_smoke_subprocess(tmp_path: Path) -> None:
    """Train tiny CLOSR model and run evaluation CLI subcommand end-to-end."""
    weights_dir = tmp_path / "weights"
    results_dir = tmp_path / "results"
    py = sys.executable

    # 1. Train tiny model
    train_cmd = [
        py,
        "scripts/run.py",
        "train",
        "--config",
        "configs/default.yaml",
        "--set",
        "data_source=synthetic",
        "--set",
        "hidden=[64,64]",
        "--set",
        "lr_peak=1e-3",
        "--set",
        "epochs=10",
        "--set",
        "batch_size=256",
        "--set",
        f"weights_dir={weights_dir}",
        "--set",
        f"results_dir={results_dir}",
        "--set",
        "seed=1",
    ]
    _run_cmd(train_cmd)

    # 2. Evaluate model via CLI
    eval_cmd = [
        py,
        "scripts/run.py",
        "eval",
        "--config",
        "configs/default.yaml",
        "--set",
        "data_source=synthetic",
        "--set",
        "hidden=[64,64]",
        "--set",
        f"weights_dir={weights_dir}",
        "--set",
        f"results_dir={results_dir}",
        "--set",
        "seed=1",
        "--set",
        "eval_batch_size=512",
    ]
    _run_cmd(eval_cmd)

    # 3. Check JSON outputs
    eval_json = results_dir / "eval" / "closr_seed1_eval.json"
    assert eval_json.is_file()
    with open(eval_json, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    for req_key in [
        "metrics",
        "config",
        "seed",
        "git_commit",
        "versions",
        "device",
        "dataset",
        "epochs",
        "reduced_run",
        "reduced_eval",
    ]:
        assert req_key in data, f"Missing key {req_key} in eval JSON"

    metrics = data["metrics"]
    assert metrics["closed_set_acc"] > 0.20, "Closed accuracy should be well above chance (1/12)"
    assert metrics["open_set_auc"] > 0.50, "Open-set AUC should be > 0.5 on synthetic data"
    assert not np.isnan(metrics["open_auc"])

    # 4. Check CSV outputs
    per_class_csv = results_dir / "eval" / "closr_seed1_per_class.csv"
    assert per_class_csv.is_file()
    df = pd.read_csv(per_class_csv)
    assert len(df) == 14  # 12 known + 2 zero-day
    assert "class_name" in df.columns
    assert "mean_score" in df.columns

    # 5. Check NPZ array outputs
    arrays_npz = results_dir / "eval" / "closr_seed1_arrays.npz"
    assert arrays_npz.is_file()
    npz = np.load(arrays_npz)
    expected_keys = [
        "scores",
        "predictions",
        "labels",
        "zd_names",
        "tsne_embeddings",
        "tsne_labels",
    ]
    for arr_key in expected_keys:
        assert arr_key in npz.files

    assert len(npz["scores"]) == len(npz["labels"])
    assert len(npz["predictions"]) == len(npz["labels"])
    assert len(npz["tsne_embeddings"]) <= 5000

    # 6. Check checkpoint reload gives same metrics as in-process evaluation
    cfg = load_config(
        "configs/default.yaml",
        overrides=[
            "data_source=synthetic",
            "hidden=[64,64]",
            f"weights_dir={weights_dir}",
            f"results_dir={results_dir}",
            "seed=1",
        ],
    )
    dataset = generate_synthetic_data(cfg)
    in_proc = run_eval_pipeline(cfg, dataset, "synthetic")
    with open(in_proc["eval_json"], "r", encoding="utf-8") as f:
        in_proc_data = json.load(f)
    assert np.isclose(
        in_proc_data["metrics"]["closed_set_acc"],
        metrics["closed_set_acc"],
        atol=1e-5,
    )


def test_clad_eval_smoke_subprocess(tmp_path: Path) -> None:
    """Train tiny CLAD model and run evaluation CLI subcommand end-to-end."""
    weights_dir = tmp_path / "weights"
    results_dir = tmp_path / "results"
    py = sys.executable

    train_cmd = [
        py,
        "scripts/run.py",
        "train",
        "--config",
        "configs/clad.yaml",
        "--set",
        "data_source=synthetic",
        "--set",
        "hidden=[64,64]",
        "--set",
        "lr_peak=1e-3",
        "--set",
        "epochs=10",
        "--set",
        "batch_size=256",
        "--set",
        f"weights_dir={weights_dir}",
        "--set",
        f"results_dir={results_dir}",
        "--set",
        "seed=1",
    ]
    _run_cmd(train_cmd)

    eval_cmd = [
        py,
        "scripts/run.py",
        "eval",
        "--config",
        "configs/clad.yaml",
        "--set",
        "data_source=synthetic",
        "--set",
        "hidden=[64,64]",
        "--set",
        f"weights_dir={weights_dir}",
        "--set",
        f"results_dir={results_dir}",
        "--set",
        "seed=1",
        "--set",
        "eval_batch_size=512",
    ]
    _run_cmd(eval_cmd)

    eval_json = results_dir / "eval" / "clad_seed1_eval.json"
    assert eval_json.is_file()
    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["metrics"]["known_mean_auroc"] > 0.50
    assert data["metrics"]["zd_mean_auroc"] > 0.50

    per_class_csv = results_dir / "eval" / "clad_seed1_per_class.csv"
    assert per_class_csv.is_file()
    df = pd.read_csv(per_class_csv)
    assert len(df) == 14
    assert "auroc" in df.columns
    assert "mean_similarity" in df.columns

    arrays_npz = results_dir / "eval" / "clad_seed1_arrays.npz"
    assert arrays_npz.is_file()
