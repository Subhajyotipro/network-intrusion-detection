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
"""Unit tests verifying tri-state data_source configuration modes (auto, real, synthetic)."""

from pathlib import Path

import pytest

from nids.config import Config
from scripts.run import _load_train_dataset


def test_data_source_synthetic_forces_synthetic():
    """Verify data_source='synthetic' forces synthetic generation regardless of data_path."""
    cfg = Config(data_source="synthetic", d_in=8, seed=42)
    data, dataset_type = _load_train_dataset(cfg)
    assert dataset_type == "synthetic"
    assert data.d_in == 8


def test_data_source_real_missing_raises_file_not_found():
    """Verify data_source='real' raises FileNotFoundError when the dataset file does not exist."""
    cfg = Config(
        data_source="real",
        data_path="/tmp/definitely_nonexistent_dataset_path_xyz123.csv",
    )
    with pytest.raises(FileNotFoundError, match="data_source='real' requested, but file not found"):
        _load_train_dataset(cfg)


def test_data_source_auto_fallback(caplog):
    """Verify data_source='auto' logs a warning and falls back to synthetic when file missing."""
    cfg = Config(
        data_source="auto",
        data_path="/tmp/definitely_nonexistent_dataset_path_xyz123.csv",
        d_in=8,
    )
    data, dataset_type = _load_train_dataset(cfg)
    assert dataset_type == "synthetic"
    assert data.d_in == 8
    assert "PROMINENT WARNING" in caplog.text


def test_data_source_auto_with_existing_file(tmp_path: Path):
    """Verify data_source='auto' loads real dataset when the file exists."""
    from nids.data import generate_synthetic_dataset

    csv_file = tmp_path / "test_data.csv"
    df = generate_synthetic_dataset(n_samples_known=20, n_samples_zd=5, n_features=8, seed=42)
    df.to_csv(csv_file, index=False)

    cfg = Config(data_source="auto", data_path=str(csv_file), sample_thres=10)
    data, dataset_type = _load_train_dataset(cfg)
    assert dataset_type == "real"
    assert data.x_train is not None
