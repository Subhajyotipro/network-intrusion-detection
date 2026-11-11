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
"""Unit tests for the unified training engine (CLAD and CLOSR)."""

import math

import numpy as np
import pytest

from nids.config import Config
from nids.data import ClassMap, DatasetSplits, generate_synthetic_data
from nids.train import train


def test_train_smoke_closr_end_to_end():
    """Verify smoke training runs end-to-end on synthetic data on CPU for CLOSR."""
    cfg = Config(
        model_kind="closr",
        d_in=8,
        hidden=[16, 16],
        d_out=8,
        n_classes=3,
        smoke=True,
        batch_size=8,
        max_batches=2,
        seed=42,
    )
    data = generate_synthetic_data(cfg, num_samples=32)
    res = train(cfg, kind="closr", data=data)

    assert "model" in res and "optimizer" in res
    assert len(res["epoch_losses"]) == 2
    assert math.isfinite(res["final_loss"]) and res["final_loss"] >= 0.0
    assert 0.0 <= res["final_frac_pos"] <= 1.0


def test_train_smoke_clad_end_to_end():
    """Verify smoke training runs end-to-end on synthetic data on CPU for CLAD."""
    cfg = Config(
        model_kind="clad",
        d_in=8,
        hidden=[16, 16],
        d_out=8,
        n_classes=2,
        smoke=True,
        batch_size=8,
        max_batches=2,
        seed=42,
    )
    data = generate_synthetic_data(cfg, num_samples=32)
    res = train(cfg, kind="clad", data=data)

    assert "model" in res and "optimizer" in res
    assert len(res["epoch_losses"]) == 2
    assert math.isfinite(res["final_loss"]) and res["final_loss"] >= 0.0


def test_train_loss_decreases_on_separable_problem():
    """Verify loss strictly decreases over a short run on a well-separated toy dataset."""
    rng = np.random.default_rng(42)
    n_per_class = 64
    x0 = rng.standard_normal((n_per_class, 8), dtype=np.float32) + 5.0
    x1 = rng.standard_normal((n_per_class, 8), dtype=np.float32) - 5.0
    x = np.vstack([x0, x1])
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)

    splits = DatasetSplits(
        x_train=x,
        y_train=y,
        x_val=None,
        y_val=None,
        x_test=None,
        y_test=None,
        x_zd=None,
        y_zd=None,
        zd_names=None,
        class_map=ClassMap(names=["benign", "attack"], zero_day_names=[]),
        split_indices={"train": np.arange(len(y))},
        d_in=8,
    )
    cfg = Config(
        model_kind="closr",
        d_in=8,
        hidden=[32, 32],
        d_out=8,
        n_classes=2,
        epochs=8,
        smoke=False,
        batch_size=32,
        lr_init=1e-3,
        lr_peak=1e-2,
        lr_final=1e-3,
        seed=42,
    )
    res = train(cfg, kind="closr", data=splits)
    assert res["epoch_losses"][-1] < res["epoch_losses"][0]


def test_train_seed_determinism_and_variation():
    """Verify identical seeds produce identical losses and different seeds vary."""
    cfg1 = Config(
        model_kind="closr",
        d_in=8,
        hidden=[16, 16],
        d_out=8,
        n_classes=3,
        dropout=0.0,
        epochs=2,
        smoke=False,
        batch_size=8,
        max_batches=2,
        seed=101,
    )
    data = generate_synthetic_data(cfg1, num_samples=32)

    res1_a = train(cfg1, kind="closr", data=data)
    res1_b = train(cfg1, kind="closr", data=data)
    np.testing.assert_allclose(res1_a["epoch_losses"], res1_b["epoch_losses"], rtol=1e-6)

    cfg2 = Config(**{**cfg1.__dict__, "seed": 202})
    res2 = train(cfg2, kind="closr", data=data)
    assert not np.allclose(res1_a["epoch_losses"], res2["epoch_losses"])


def test_train_robustness_small_dataset_and_class():
    """Verify dataset smaller than one batch runs safely without crashing."""
    x = np.ones((3, 4), dtype=np.float32)
    y = np.array([0, 0, 1], dtype=np.int64)
    data = DatasetSplits(
        x_train=x,
        y_train=y,
        x_val=None,
        y_val=None,
        x_test=None,
        y_test=None,
        x_zd=None,
        y_zd=None,
        zd_names=None,
        class_map=ClassMap(names=["benign", "attack"], zero_day_names=[]),
        split_indices={"train": np.arange(3)},
        d_in=4,
    )
    cfg = Config(
        model_kind="closr",
        d_in=4,
        hidden=[8],
        d_out=4,
        n_classes=2,
        batch_size=16,
        epochs=1,
        smoke=False,
    )
    res = train(cfg, kind="closr", data=data)
    assert math.isfinite(res["final_loss"])


def test_train_nan_input_raises_value_error():
    """Verify non-finite loss during training raises ValueError."""
    x = np.full((16, 4), np.nan, dtype=np.float32)
    y = np.zeros(16, dtype=np.int64)
    data = DatasetSplits(
        x_train=x,
        y_train=y,
        x_val=None,
        y_val=None,
        x_test=None,
        y_test=None,
        x_zd=None,
        y_zd=None,
        zd_names=None,
        class_map=ClassMap(names=["benign"], zero_day_names=[]),
        split_indices={"train": np.arange(16)},
        d_in=4,
    )
    cfg = Config(
        model_kind="closr",
        d_in=4,
        hidden=[8],
        d_out=4,
        n_classes=1,
        batch_size=8,
        epochs=1,
        smoke=False,
    )
    with pytest.raises(ValueError, match=r"(Non-finite loss|contains non-finite)"):
        train(cfg, kind="closr", data=data)


def test_train_inf_input_raises_value_error():
    """Verify positive infinity in training features fails before an optimizer update."""
    x = np.full((16, 4), np.inf, dtype=np.float32)
    y = np.zeros(16, dtype=np.int64)
    data = DatasetSplits(
        x_train=x,
        y_train=y,
        x_val=None,
        y_val=None,
        x_test=None,
        y_test=None,
        x_zd=None,
        y_zd=None,
        zd_names=None,
        class_map=ClassMap(names=["benign"], zero_day_names=[]),
        split_indices={"train": np.arange(16)},
        d_in=4,
    )
    cfg = Config(
        model_kind="closr",
        d_in=4,
        hidden=[8],
        d_out=4,
        n_classes=1,
        batch_size=8,
        epochs=1,
        smoke=False,
    )
    with pytest.raises(ValueError, match="Training batch contains non-finite"):
        train(cfg, kind="closr", data=data)


def test_train_unsupported_model_kind_raises_value_error():
    """Verify unsupported kind string raises ValueError."""
    cfg = Config(model_kind="closr", d_in=4, hidden=[8], d_out=4, n_classes=2)
    data = generate_synthetic_data(cfg, num_samples=16)
    with pytest.raises(ValueError, match="Unsupported model kind"):
        train(cfg, kind="unknown", data=data)
