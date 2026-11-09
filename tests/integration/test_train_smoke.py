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
"""Integration test running CLI train subcommand in smoke mode via subprocess."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_train_cli_smoke_closr(tmp_path: Path):
    """Verify scripts/run.py train runs end-to-end and produces valid D4.7 results."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_py = repo_root / "scripts" / "run.py"
    res_dir = tmp_path / "results"
    weights_dir = tmp_path / "weights"

    env = os.environ.copy()
    env["KERAS_HOME"] = "/tmp/.keras"
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"

    cmd = [
        sys.executable,
        str(run_py),
        "train",
        "--set",
        "smoke=true",
        "--set",
        "data_source=synthetic",
        "--set",
        f"results_dir={res_dir}",
        "--set",
        f"weights_dir={weights_dir}",
        "--set",
        "seed=1",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(repo_root))
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    result_json = res_dir / "train" / "closr_seed1.json"
    assert result_json.is_file(), f"Result JSON not created at {result_json}"

    with open(result_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Verify D4.7 required schema
    req_keys = [
        "metrics",
        "config",
        "seed",
        "git_commit",
        "versions",
        "device",
        "dataset",
        "epochs",
        "reduced_run",
    ]
    for key in req_keys:
        assert key in data, f"Missing D4.7 key '{key}' in {result_json}"

    assert data["seed"] == 1
    assert data["dataset"] == "synthetic"
    assert data["epochs"] == 2
    assert data["reduced_run"] is True
    assert "final_loss" in data["metrics"]
    assert "epoch_losses" in data["metrics"]
    assert "final_frac_pos" in data["metrics"]

    # Verify checkpoint bundle created
    assert (weights_dir / "closr_seed1.weights.h5").is_file()
    assert (weights_dir / "closr_seed1_meta.json").is_file()
    assert (weights_dir / "closr_seed1_opt.npz").is_file()


def test_train_cli_smoke_clad(tmp_path: Path):
    """Verify scripts/run.py train runs end-to-end for CLAD model."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_py = repo_root / "scripts" / "run.py"
    res_dir = tmp_path / "results"
    weights_dir = tmp_path / "weights"

    env = os.environ.copy()
    env["KERAS_HOME"] = "/tmp/.keras"
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"

    cmd = [
        sys.executable,
        str(run_py),
        "train",
        "--set",
        "model_kind=clad",
        "--set",
        "smoke=true",
        "--set",
        "data_source=synthetic",
        "--set",
        f"results_dir={res_dir}",
        "--set",
        f"weights_dir={weights_dir}",
        "--set",
        "seed=2",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(repo_root))
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    result_json = res_dir / "train" / "clad_seed2.json"
    assert result_json.is_file()
    assert (weights_dir / "clad_seed2.weights.h5").is_file()
