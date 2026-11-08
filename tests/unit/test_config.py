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
"""Unit tests for nids.config module: defaults, YAML loading, overrides, and helpers."""

from pathlib import Path
from unittest.mock import patch

import pytest

from nids.config import (
    get_default_config,
    get_run_metadata,
    load_config,
    resolve_device,
    save_config,
    set_seed,
)
from scripts.run import build_parser
from scripts.run import main as run_main


def test_closr_defaults_match_inventory():
    """Verify default CLOSR configuration values match upstream inventory table."""
    cfg = get_default_config("closr")
    assert cfg.model_kind == "closr"
    assert cfg.d_out == 64
    assert cfg.hidden == [1024, 1024, 1024]
    assert cfg.dropout == 0.1
    assert cfg.margin == 1.0
    assert cfg.squared is True
    assert cfg.residual is False
    assert cfg.batch_size == 4096
    assert abs(cfg.weight_decay - 0.0403709) < 1e-6
    assert cfg.epochs == 200
    assert cfg.lr_init == 1e-6
    assert cfg.lr_peak == 1e-5
    assert cfg.lr_final == 1e-6
    assert cfg.warmup_frac == 0.1
    assert cfg.sample_thres == 100
    assert cfg.split_seed == 39058032
    assert cfg.test_ratio == 0.5
    assert cfg.val_ratio == 0.0


def test_clad_defaults_match_inventory():
    """Verify default CLAD configuration values match upstream inventory table."""
    cfg = get_default_config("clad")
    assert cfg.model_kind == "clad"
    assert cfg.d_out == 8
    assert cfg.hidden == [1024, 1024, 1024, 1024]
    assert cfg.dropout == 0.0
    assert cfg.weight_decay == 3e-7
    assert cfg.lr_peak == 4.4e-6
    # Other values match global baseline
    assert cfg.squared is True
    assert cfg.residual is False
    assert cfg.batch_size == 4096


def test_boolean_flag_overrides():
    """Verify boolean flags parse correctly from string overrides (fixing argparse bug)."""
    cfg = load_config(overrides=["squared=false", "residual=true"])
    assert cfg.squared is False
    assert cfg.residual is True

    cfg2 = load_config(overrides=["squared=true", "residual=false"])
    assert cfg2.squared is True
    assert cfg2.residual is False

    with pytest.raises(ValueError, match="Expected bool"):
        load_config(overrides=["squared=invalid_bool"])


def test_type_coercion_and_overrides():
    """Verify float, list, and int coercion from string overrides."""
    cfg = load_config(overrides=["lr_peak=1e-5", "hidden=[512, 512]", "epochs=50"])
    assert isinstance(cfg.lr_peak, float)
    assert cfg.lr_peak == 1e-5
    assert cfg.hidden == [512, 512]
    assert cfg.epochs == 50

    with pytest.raises(KeyError, match="Unknown config key"):
        load_config(overrides=["nonexistent_param=123"])

    with pytest.raises(ValueError, match="Expected float"):
        load_config(overrides=["lr_peak=not_a_float"])

    with pytest.raises(ValueError, match="Expected int"):
        load_config(overrides=["epochs=3.14"])


def test_yaml_round_trip(tmp_path: Path):
    """Verify writing a Config to YAML and reloading it yields an identical Config."""
    cfg = get_default_config("closr")
    cfg.epochs = 42
    yaml_file = tmp_path / "test_cfg.yaml"
    save_config(cfg, yaml_file)

    loaded = load_config(path=yaml_file)
    assert loaded == cfg


def test_clad_yaml_overrides_default():
    """Verify loading configs/clad.yaml modifies only the differing CLAD fields."""
    default_cfg = load_config(path="configs/default.yaml")
    clad_cfg = load_config(path="configs/clad.yaml")

    assert clad_cfg.model_kind == "clad"
    assert clad_cfg.d_out == 8
    assert clad_cfg.hidden == [1024, 1024, 1024, 1024]
    assert clad_cfg.dropout == 0.0
    assert clad_cfg.weight_decay == 3e-7
    assert clad_cfg.lr_peak == 4.4e-6

    # Fields that should stay identical to default
    assert clad_cfg.batch_size == default_cfg.batch_size
    assert clad_cfg.margin == default_cfg.margin
    assert clad_cfg.squared == default_cfg.squared
    assert clad_cfg.epochs == default_cfg.epochs


def test_din_defaults_to_none_and_device_resolution():
    """Verify d_in defaults to None and device helper respects auto/overrides."""
    cfg = load_config()
    assert cfg.d_in is None

    cfg_din = load_config(overrides=["d_in=72"])
    assert cfg_din.d_in == 72

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    with patch("torch.cuda.is_available", return_value=False):
        assert resolve_device("auto") == "cpu"
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device("auto") == "cuda"

    with pytest.raises(ValueError, match="Invalid device"):
        resolve_device("tpu")


def test_get_run_metadata():
    """Verify metadata helper returns all required keys including git fallback."""
    cfg = get_default_config("closr")
    meta = get_run_metadata(cfg)

    assert "seed" in meta and meta["seed"] == cfg.seed
    assert "git_commit" in meta
    assert "device" in meta
    assert "versions" in meta
    assert "torch" in meta["versions"]
    assert "numpy" in meta["versions"]
    assert "pandas" in meta["versions"]
    assert "sklearn" in meta["versions"]

    # Test when git fails
    with patch("subprocess.check_output", side_effect=FileNotFoundError):
        meta_nogit = get_run_metadata(cfg)
        assert meta_nogit["git_commit"] == "unknown"


def test_cli_subcommands():
    """Verify that run.py parses subcommands and options correctly."""
    parser = build_parser()
    for cmd in ["train", "eval", "calibrate", "cascade", "bench", "demo"]:
        args = parser.parse_args([cmd, "--config", "configs/default.yaml", "--set", "epochs=5"])
        assert args.subcommand == cmd
        assert args.config == "configs/default.yaml"
        assert args.overrides == ["epochs=5"]

    # Test main execution
    exit_code = run_main(["train", "--config", "configs/default.yaml", "--set", "epochs=2"])
    assert exit_code == 0


def test_config_validation_and_bug_hunt(tmp_path: Path):
    """Deliberate bug-hunt tests: bounds, duplicate keys, spaces in paths, empty lists."""
    # Negative or zero epochs
    with pytest.raises(ValueError, match="epochs must be positive"):
        load_config(overrides=["epochs=0"])
    with pytest.raises(ValueError, match="epochs must be positive"):
        load_config(overrides=["epochs=-5"])

    # Test ratio out of bounds
    with pytest.raises(ValueError, match="test_ratio must be in"):
        load_config(overrides=["test_ratio=1.0"])
    with pytest.raises(ValueError, match="test_ratio must be in"):
        load_config(overrides=["test_ratio=-0.1"])

    # Val ratio out of bounds
    with pytest.raises(ValueError, match="val_ratio must be in"):
        load_config(overrides=["val_ratio=1.2"])

    # Empty hidden layers
    with pytest.raises(ValueError, match="hidden layers list cannot be empty"):
        load_config(overrides=["hidden=[]"])

    # Duplicate --set keys: last wins
    cfg_dup = load_config(overrides=["epochs=10", "epochs=25"])
    assert cfg_dup.epochs == 25

    # Empty override key
    with pytest.raises(ValueError, match="Empty key in override"):
        load_config(overrides=["=10"])

    # Missing = in override
    with pytest.raises(ValueError, match="Invalid override format"):
        load_config(overrides=["epochs10"])

    # Paths with spaces
    spaced_path = tmp_path / "path with spaces" / "cfg.yaml"
    cfg = get_default_config("closr")
    save_config(cfg, spaced_path)
    loaded = load_config(path=spaced_path)
    assert loaded.epochs == cfg.epochs

    # Unknown model_kind
    with pytest.raises(ValueError, match="Unknown model_kind"):
        get_default_config("transformer")


def test_set_seed():
    """Verify set_seed executes without crashing."""
    set_seed(42)
