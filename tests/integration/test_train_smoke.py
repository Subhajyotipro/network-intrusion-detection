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
from typing import Any


def _run_cli(args: list[str], env: dict[str, str], root: Path) -> subprocess.CompletedProcess:
    """Execute run.py subcommand with common env and verify returncode 0."""
    cmd = [sys.executable, str(root / "scripts" / "run.py")] + args
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(root))
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return res


def _assert_d47_record(data: dict[str, Any], seed: int) -> None:
    """Verify standard D4.7 fields and metrics are present in training record."""
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
        assert key in data
    assert data["seed"] == seed and data["dataset"] == "synthetic"
    assert data["epochs"] == 2 and data["reduced_run"] is True
    assert "final_loss" in data["metrics"] and "epoch_losses" in data["metrics"]


def test_train_cli_smoke_closr(tmp_path: Path):
    """Verify scripts/run.py train runs end-to-end and produces valid D4.7 results."""
    root = Path(__file__).resolve().parent.parent.parent
    res_dir, w_dir = tmp_path / "results", tmp_path / "weights"
    env = {
        **os.environ,
        "KERAS_HOME": "/tmp/.keras",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "PYTHONPATH": str(root / "src"),
    }
    args = [
        "train",
        "--set",
        "smoke=true",
        "--set",
        "data_source=synthetic",
        "--set",
        f"results_dir={res_dir}",
        "--set",
        f"weights_dir={w_dir}",
        "--set",
        "seed=1",
    ]
    _run_cli(args, env, root)

    result_json = res_dir / "train" / "closr_seed1.json"
    assert result_json.is_file(), f"Result JSON not created at {result_json}"
    with open(result_json, "r", encoding="utf-8") as f:
        _assert_d47_record(json.load(f), seed=1)

    assert (w_dir / "closr_seed1.weights.h5").is_file()
    assert (w_dir / "closr_seed1_meta.json").is_file()
    assert (w_dir / "closr_seed1_opt.npz").is_file()


def test_train_cli_smoke_clad(tmp_path: Path):
    """Verify scripts/run.py train runs end-to-end for CLAD model."""
    root = Path(__file__).resolve().parent.parent.parent
    res_dir, w_dir = tmp_path / "results", tmp_path / "weights"
    env = {
        **os.environ,
        "KERAS_HOME": "/tmp/.keras",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "PYTHONPATH": str(root / "src"),
    }
    args = [
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
        f"weights_dir={w_dir}",
        "--set",
        "seed=2",
    ]
    _run_cli(args, env, root)

    assert (res_dir / "train" / "clad_seed2.json").is_file()
    assert (w_dir / "clad_seed2.weights.h5").is_file()


def _stage_cli(args_str: str, base: list[str], env: dict[str, str], root: Path) -> None:
    """Run CLI with space-separated args appended to base."""
    _run_cli(base + args_str.split(), env, root)


def _run_stages(
    base: list[str], tmp_path: Path, env: dict[str, str], root: Path
) -> tuple[list[float], list[float]]:
    """Execute uninterrupted, partial, and resumed runs and return loss histories."""
    fw, fr = tmp_path / "f_w", tmp_path / "f_res"
    pw, pr = tmp_path / "p_w", tmp_path / "p_res"
    rw, rr = tmp_path / "r_w", tmp_path / "r_res"

    _stage_cli(f"--set epochs=3 --set results_dir={fr} --set weights_dir={fw}", base, env, root)
    _stage_cli(
        f"--set epochs=2 --set checkpoint_every=1 --set results_dir={pr} --set weights_dir={pw}",
        base,
        env,
        root,
    )
    ckpt = str(pw / "closr_seed42.weights.h5")
    _stage_cli(
        f"--resume {ckpt} --set epochs=3 --set results_dir={rr} --set weights_dir={rw}",
        base,
        env,
        root,
    )

    with open(fr / "train" / "closr_seed42.json") as f:
        l_full = json.load(f)["metrics"]["epoch_losses"]
    with open(rr / "train" / "closr_seed42.json") as f:
        l_res = json.load(f)["metrics"]["epoch_losses"]
    return l_full, l_res


def test_train_cli_resume_parity(tmp_path: Path):
    """Verify CLI resume reproduces uninterrupted loss history within 1e-5."""
    import numpy as np

    root = Path(__file__).resolve().parent.parent.parent
    env = {
        **os.environ,
        "KERAS_HOME": "/tmp/.keras",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "PYTHONPATH": str(root / "src"),
    }
    base = [
        "train",
        "--set",
        "smoke=false",
        "--set",
        "dropout=0.0",
        "--set",
        "data_source=synthetic",
        "--set",
        "hidden=[32]",
        "--set",
        "batch_size=64",
        "--set",
        "max_batches=2",
        "--set",
        "seed=42",
        "--set",
        "lr_init=1e-4",
        "--set",
        "lr_peak=1e-4",
        "--set",
        "lr_final=1e-4",
        "--set",
        "warmup_frac=0.0",
    ]
    l_full, l_res = _run_stages(base, tmp_path, env, root)
    assert len(l_full) == 3 and len(l_res) == 3
    assert np.allclose(l_full, l_res, atol=1e-5)
