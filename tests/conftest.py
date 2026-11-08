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
"""Shared pytest fixtures, configuration, and synthetic data helpers."""

import os
from pathlib import Path

import pandas as pd
import pytest

# Set default Keras and Matplotlib homes to /tmp to prevent sandbox permission issues
os.environ.setdefault("KERAS_HOME", "/tmp/.keras")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from nids.data import generate_synthetic_dataset  # noqa: E402


@pytest.fixture
def tiny_synthetic_df() -> pd.DataFrame:
    """Fixture providing a tiny synthetic dataset (8 features, 4 known + 1 zero-day class)."""
    return generate_synthetic_dataset(
        n_samples_known=200,
        n_samples_zd=20,
        n_features=8,
        seed=42,
        known_classes=["benign", "Botnet", "DDoS", "Portscan"],
        zero_day_classes=["Heartbleed"],
    )


@pytest.fixture
def tiny_synthetic_csv_path(tmp_path: Path, tiny_synthetic_df: pd.DataFrame) -> Path:
    """Fixture providing the path to a saved tiny synthetic CSV file."""
    csv_path = tmp_path / "tiny_lycos.csv"
    tiny_synthetic_df.to_csv(csv_path, index=False)
    return csv_path
