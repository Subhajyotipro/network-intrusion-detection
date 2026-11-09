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
"""Integration test running baseline CLI subcommands in smoke mode on synthetic data."""

from pathlib import Path

from scripts.run import main as run_main


def _run_baseline_cli(model: str, tmp_path: Path) -> Path:
    """Helper running baseline CLI command into isolated tmp_path."""
    res_dir = tmp_path / "results"
    code = run_main(
        [
            "baseline",
            "--model",
            model,
            "--config",
            "configs/default.yaml",
            "--set",
            "smoke=true",
            "--set",
            "data_source=synthetic",
            "--set",
            f"results_dir={res_dir}",
            "--set",
            "seed=1",
        ]
    )
    assert code == 0
    record_file = res_dir / "baselines" / f"{model}_seed1_synthetic.json"
    assert record_file.is_file(), f"Expected record file {record_file} not found"
    return record_file


def test_baseline_cli_smoke_all_models(tmp_path: Path):
    """Verify CLI smoke for each of the 3 baselines and comparison table generation."""
    res_dir = tmp_path / "results"
    for model in ["mlp", "autoencoder", "iforest"]:
        _run_baseline_cli(model, tmp_path)

    # Now run compare CLI
    code = run_main(
        [
            "compare",
            "--config",
            "configs/default.yaml",
            "--set",
            "data_source=synthetic",
            "--set",
            f"results_dir={res_dir}",
        ]
    )
    assert code == 0
    assert (res_dir / "baselines" / "comparison_synthetic.csv").is_file()
    assert (res_dir / "baselines" / "comparison_synthetic.md").is_file()

