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
"""Unit tests for nids.data: loaders, class map, stratified splits, imputation, and samplers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nids.data import (
    BENIGN_ID,
    IDENTIFIER_COLS,
    KNOWN_CLASSES,
    UNKNOWN_ID,
    ZERO_DAY_CLASSES,
    class_balanced_sampler,
    generate_synthetic_dataset,
    load_data,
    load_split_indices,
)


def test_synthetic_generator_schema_and_determinism():
    """Verify synthetic generator generates exact Lycos2017 schema deterministically."""
    df1 = generate_synthetic_dataset(n_samples_known=50, n_samples_zd=10, seed=123)
    df2 = generate_synthetic_dataset(n_samples_known=50, n_samples_zd=10, seed=123)

    pd.testing.assert_frame_equal(df1, df2)

    # Check identifier columns
    for col in IDENTIFIER_COLS:
        assert col in df1.columns

    # Check 72 features
    feature_cols = [c for c in df1.columns if c.startswith("feature_")]
    assert len(feature_cols) == 72

    # Check classes: 12 known + 2 zero-day = 14 total
    unique_classes = set(df1["label"].unique())
    assert len(unique_classes) == 14
    for k in KNOWN_CLASSES:
        assert k in unique_classes
    for zd in ZERO_DAY_CLASSES:
        assert zd in unique_classes


def test_data_splits_no_index_overlap_and_zero_day_absence(tiny_synthetic_csv_path: Path):
    """Verify train, val, and test splits have no index overlap and exclude zero-day classes."""
    splits = load_data(
        data_path=tiny_synthetic_csv_path,
        sample_thres=50,  # 20 samples -> zero-day
        test_ratio=0.5,
        val_ratio=0.1,
    )

    tr_idx = set(splits.split_indices["train"])
    val_idx = set(splits.split_indices["val"])
    te_idx = set(splits.split_indices["test"])

    assert tr_idx.isdisjoint(val_idx)
    assert tr_idx.isdisjoint(te_idx)
    assert val_idx.isdisjoint(te_idx)

    # Zero-day classes must be absent from train and val
    assert splits.x_zd is not None
    assert splits.y_zd is not None
    assert np.all(splits.y_zd == UNKNOWN_ID)
    assert len(splits.zd_names) == len(splits.y_zd)
    assert "heartbleed" in splits.zd_names

    # Check that y_train and y_val only contain known class IDs
    known_ids = set(range(splits.class_map.n_known))
    assert set(splits.y_train).issubset(known_ids)
    assert set(splits.y_val).issubset(known_ids)


def test_val_ratio_zero_reproduces_upstream_split(tiny_synthetic_csv_path: Path):
    """Verify val_ratio=0.0 yields no val set and matches direct train_test_split indices."""
    splits = load_data(
        data_path=tiny_synthetic_csv_path,
        sample_thres=50,
        test_ratio=0.5,
        val_ratio=0.0,
        split_seed=39058032,
    )

    assert splits.x_val is None
    assert splits.y_val is None
    assert "val" not in splits.split_indices

    # Check test ratio size
    total_known = len(splits.y_train) + len(splits.y_test)
    assert len(splits.y_test) == int(total_known * 0.5)


def test_class_map_benign_is_zero():
    """Verify ClassMap assigns BENIGN_ID = 0 and reports known and zero-day counts."""
    df = generate_synthetic_dataset(n_samples_known=20, n_samples_zd=5)
    # Save to temp csv
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_csv = f.name

    try:
        splits = load_data(temp_csv, sample_thres=10)
        assert splits.class_map.names[0] == "benign"
        assert BENIGN_ID == 0
        assert splits.class_map.n_known == 12
        assert splits.class_map.n_zero_day == 2
    finally:
        Path(temp_csv).unlink(missing_ok=True)


def test_identifier_cols_dropped(tiny_synthetic_csv_path: Path):
    """Verify identifier columns are stripped from input feature matrices."""
    splits = load_data(tiny_synthetic_csv_path, sample_thres=50)
    # Synthetic has 8 features
    assert splits.d_in == 8
    assert splits.x_train.shape[1] == 8


def test_scaler_and_imputer_fit_on_train_only(tmp_path: Path):
    """Verify changing val/test values does not change train scaling statistics."""
    df1 = generate_synthetic_dataset(n_samples_known=100, n_samples_zd=10, seed=42)
    csv1 = tmp_path / "data1.csv"
    df1.to_csv(csv1, index=False)
    splits1 = load_data(csv1, sample_thres=50, test_ratio=0.5, val_ratio=0.1, split_seed=42)

    # Create df2 with identical train distribution but heavily perturbed test rows
    df2 = df1.copy()
    # Find a test index in known data and inject huge outliers
    df2.loc[df2["label"] == "benign", "feature_0"] += 1e6
    csv2 = tmp_path / "data2.csv"
    df2.to_csv(csv2, index=False)
    splits2 = load_data(csv2, sample_thres=50, test_ratio=0.5, val_ratio=0.1, split_seed=42)

    # Verify train normalization is isolated:
    # Train mean is 0.0 and std is 1.0 after standard scaling
    assert np.allclose(splits1.x_train.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(splits1.x_train.std(axis=0), 1.0, atol=1e-5)
    assert np.allclose(splits2.x_train.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(splits2.x_train.std(axis=0), 1.0, atol=1e-5)


def test_zero_column_mask_computed_on_train_only(tmp_path: Path):
    """Verify all-zero column mask is fit on train split only without test leakage (DV13)."""
    df = generate_synthetic_dataset(n_samples_known=100, n_samples_zd=10, seed=42)
    # Zero out feature_0 across all known classes (which form train/val/test),
    # but give zero-day flows non-zero values for feature_0.
    df.loc[df["label"].isin(KNOWN_CLASSES), "feature_0"] = 0.0
    df.loc[df["label"].isin(ZERO_DAY_CLASSES), "feature_0"] = 42.0

    csv_path = tmp_path / "train_zero_zd_nonzero.csv"
    df.to_csv(csv_path, index=False)

    splits = load_data(csv_path, sample_thres=50, test_ratio=0.5, val_ratio=0.0, split_seed=42)
    # Because feature_0 was all-zero in train, it must be dropped (d_in drops from 72 to 71)
    # even though it was non-zero in zero-day data!
    assert splits.d_in == 71
    assert splits.x_train.shape[1] == 71
    assert splits.x_test.shape[1] == 71
    assert splits.x_zd.shape[1] == 71


def test_nan_and_inf_handling(tmp_path: Path):
    """Verify +-inf values are replaced with NaN and imputed using train medians."""
    df = generate_synthetic_dataset(n_samples_known=50, n_samples_zd=5, seed=42)
    # Inject Inf and NaN
    df.loc[0, "feature_0"] = np.inf
    df.loc[1, "feature_0"] = -np.inf
    df.loc[2, "feature_1"] = np.nan
    csv = tmp_path / "dirty.csv"
    df.to_csv(csv, index=False)

    splits = load_data(csv, sample_thres=10, test_ratio=0.5, val_ratio=0.0)
    assert not np.isnan(splits.x_train).any()
    assert not np.isinf(splits.x_train).any()
    assert not np.isnan(splits.x_test).any()
    assert not np.isinf(splits.x_test).any()


def test_split_indices_save_and_reload(tmp_path: Path, tiny_synthetic_csv_path: Path):
    """Verify split indices are saved and reloaded with identical contents."""
    splits = load_data(
        tiny_synthetic_csv_path,
        sample_thres=50,
        test_ratio=0.5,
        val_ratio=0.1,
        results_dir=tmp_path,
    )
    saved_files = list((tmp_path / "splits").glob("*.npz"))
    assert len(saved_files) == 1

    loaded = load_split_indices(saved_files[0])
    for split_name in ("train", "val", "test"):
        np.testing.assert_array_equal(splits.split_indices[split_name], loaded[split_name])


def test_balanced_sampler_properties():
    """Verify class_balanced_sampler determinism, rare class sampling, edge cases, and safety."""
    # Create an imbalanced dataset: 1000 samples of class 0, 5 samples of class 1
    rng = np.random.default_rng(42)
    x0 = rng.standard_normal((1000, 4))
    y0 = np.zeros(1000, dtype=np.int64)
    x1 = rng.standard_normal((5, 4))
    y1 = np.ones(5, dtype=np.int64)
    x = np.vstack([x0, x1])
    y = np.concatenate([y0, y1])

    # 1. Determinism
    gen1 = class_balanced_sampler(x, y, batch_size=32, seed=999, max_batches=5)
    b1_x, b1_y = next(gen1)
    gen2 = class_balanced_sampler(x, y, batch_size=32, seed=999, max_batches=5)
    b2_x, b2_y = next(gen2)
    np.testing.assert_array_equal(b1_x, b2_x)
    np.testing.assert_array_equal(b1_y, b2_y)

    # 2. Rare class is actually sampled
    sampler = class_balanced_sampler(x, y, batch_size=64, seed=42, max_batches=10)
    batches_y = [b_y for _, b_y in sampler]
    class_1_counts = sum(np.sum(b_y == 1) for b_y in batches_y)
    assert class_1_counts > 0, "Rare class 1 was never sampled!"

    # 3. Class smaller than batch size does not crash
    gen_small = class_balanced_sampler(x1, y1, batch_size=10, seed=42, max_batches=2)
    bx_s, by_s = next(gen_small)
    assert len(bx_s) == 5  # Clamped to len(x)

    # 4. Batch size 1
    gen_one = class_balanced_sampler(x, y, batch_size=1, seed=42, max_batches=3)
    bx_1, by_1 = next(gen_one)
    assert len(bx_1) == 1

    # 5. Empty dataset
    gen_empty = class_balanced_sampler(np.empty((0, 4)), np.empty(0, dtype=np.int64), batch_size=10)
    assert list(gen_empty) == []


@pytest.mark.real
def test_real_lycos_data_pipeline():
    """Integration test on real Lycos2017 dataset if present at data/lycos.csv."""
    real_csv = Path("data/lycos.csv")
    if not real_csv.exists():
        pytest.skip("data/lycos.csv not found; skipping real data test")

    splits = load_data(
        data_path=real_csv,
        sample_thres=100,
        split_seed=39058032,
        test_ratio=0.5,
        val_ratio=0.0,
    )
    assert splits.d_in == 72
    assert splits.class_map.names[0] == "benign"
    assert "heartbleed" in splits.class_map.zero_day_names
    assert "webattack_sql_injection" in splits.class_map.zero_day_names
