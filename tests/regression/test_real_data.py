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
"""Regression tests verifying pipeline execution on real Lycos2017 dataset."""

from pathlib import Path

import numpy as np
import pytest

from nids.data import UNKNOWN_ID, load_data

DATA_FILE = Path("data/lycos.csv")


@pytest.mark.real
def test_real_data_splits_and_zero_day_detection():
    """Verify loading real Lycos2017 dataset: splits, feature count, and zero-day isolation."""
    if not DATA_FILE.is_file():
        pytest.skip(f"Real data file not found at {DATA_FILE}")

    splits = load_data(
        DATA_FILE,
        test_ratio=0.5,
        val_ratio=0.1,
        sample_thres=100,
        split_seed=39058032,
    )

    # 1. Feature dimension
    assert splits.d_in == 72, f"Expected d_in=72, got {splits.d_in}"

    # 2. Known class and zero-day class structure
    assert splits.class_map.names[0] == "benign"
    assert splits.class_map.n_known == 12
    assert splits.class_map.n_zero_day == 2
    expected_zd = ["heartbleed", "webattack_sql_injection"]
    assert splits.class_map.zero_day_names == expected_zd

    # 3. Non-overlapping partitions
    tr_idx = splits.split_indices["train"]
    val_idx = splits.split_indices["val"]
    te_idx = splits.split_indices["test"]
    assert len(np.intersect1d(tr_idx, val_idx)) == 0
    assert len(np.intersect1d(tr_idx, te_idx)) == 0
    assert len(np.intersect1d(val_idx, te_idx)) == 0

    # 4. Zero-day classes completely isolated
    assert splits.x_zd is not None
    assert splits.y_zd is not None
    assert np.all(splits.y_zd == UNKNOWN_ID)
    assert set(splits.zd_names) == set(expected_zd)

    # 5. Every known class present across train, val, test
    partitions = [
        (splits.y_train, "train"),
        (splits.y_val, "val"),
        (splits.y_test, "test"),
    ]
    for split_y, name in partitions:
        unique_classes = set(np.unique(split_y))
        msg = f"Split {name} missing classes: expected 12, got {len(unique_classes)}"
        assert len(unique_classes) == 12, msg

    # 6. Finite values after scaling
    assert np.all(np.isfinite(splits.x_train))
    assert np.all(np.isfinite(splits.x_val))
    assert np.all(np.isfinite(splits.x_test))
