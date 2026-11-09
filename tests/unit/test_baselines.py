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
"""Unit tests for baseline models and comparison tables."""

from pathlib import Path

import numpy as np

from nids.baselines import AutoencoderBaseline, IsolationForestBaseline, MLPBaseline
from nids.compare import COLUMNS, UPSTREAM_TARGETS, build_comparison_tables


def test_hand_computed_mlp_score():
    """Verify hand-computed max softmax probability score on deterministic probabilities.

    Hand calculation:
    Suppose sample 0 probabilities are [0.1, 0.7, 0.2] -> max is 0.7
    Suppose sample 1 probabilities are [0.4, 0.3, 0.3] -> max is 0.4
    Expected open-set scores: [0.7, 0.4]
    """
    model = MLPBaseline(d_in=2, n_classes=3, hidden=[4], dropout=0.0)
    probs = np.array([[0.1, 0.7, 0.2], [0.4, 0.3, 0.3]], dtype=np.float32)
    # Monkey-patch predict_proba to return deterministic test probabilities
    model.predict_proba = lambda x, batch_size=4096: probs

    scores = model.score_known(np.zeros((2, 2)))
    np.testing.assert_allclose(scores, np.array([0.7, 0.4]), atol=1e-6)


def test_hand_computed_autoencoder_score():
    """Verify hand-computed negative MSE reconstruction error.

    Hand calculation:
    Input x:
      sample 0 = [1.0, 2.0]
      sample 1 = [3.0, 4.0]
    Reconstruction x_rec:
      sample 0 = [1.0, 1.0] -> diff = [0.0, 1.0] -> MSE = (0^2 + 1^2)/2 = 0.5
      sample 1 = [1.0, 4.0] -> diff = [2.0, 0.0] -> MSE = (2^2 + 0^2)/2 = 2.0
    Negative MSE score (higher = known):
      sample 0 = -0.5
      sample 1 = -2.0
    """
    ae = AutoencoderBaseline(d_in=2, hidden=[4], latent_dim=2)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x_rec = np.array([[1.0, 1.0], [1.0, 4.0]], dtype=np.float32)
    ae.model.predict = lambda inp, batch_size=4096, verbose=0: x_rec

    scores = ae.score_known(x)
    np.testing.assert_allclose(scores, np.array([-0.5, -2.0]), atol=1e-6)


def test_hand_computed_iforest_sign_convention():
    """Verify Isolation Forest scores inliers higher than outliers.

    Cluster 20 points around [0.0, 0.0] and evaluate an inlier [0.0, 0.0]
    versus a distant outlier [100.0, 100.0]. The inlier score must be higher.
    """
    rng = np.random.default_rng(42)
    train_x = rng.normal(loc=0.0, scale=0.1, size=(50, 2)).astype(np.float32)
    model = IsolationForestBaseline(n_estimators=20, random_state=42)
    model.fit(train_x)

    test_inlier = np.array([[0.0, 0.0]], dtype=np.float32)
    test_outlier = np.array([[100.0, 100.0]], dtype=np.float32)
    s_in = model.score_known(test_inlier)[0]
    s_out = model.score_known(test_outlier)[0]
    assert s_in > s_out, f"Inlier score ({s_in}) must be higher than outlier ({s_out})"


def test_sign_conventions_on_toy_data():
    """Verify sign convention: baselines score normal samples higher than outliers."""
    rng = np.random.default_rng(123)
    # Class 0 at -2.0, Class 1 at +2.0
    x_c0 = rng.normal(loc=-2.0, scale=0.1, size=(30, 4)).astype(np.float32)
    x_c1 = rng.normal(loc=2.0, scale=0.1, size=(30, 4)).astype(np.float32)
    x_train = np.concatenate([x_c0, x_c1], axis=0)
    y_train = np.concatenate([np.zeros(30, dtype=np.int32), np.ones(30, dtype=np.int32)], axis=0)

    # 1. MLP: confident in-distribution sample vs ambiguous boundary sample
    mlp = MLPBaseline(d_in=4, n_classes=2, hidden=[16], dropout=0.0)
    mlp.fit(x_train, y_train, epochs=15, batch_size=16, lr=1e-2)
    confident_known = np.array([[2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
    ambiguous = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    assert mlp.score_known(confident_known)[0] > mlp.score_known(ambiguous)[0]

    # 2. Autoencoder: benign-like sample vs outlier
    x_benign = x_c0
    ae = AutoencoderBaseline(d_in=4, hidden=[8], latent_dim=4)
    ae.fit(x_benign, epochs=20, batch_size=16, lr=1e-2)
    outlier = np.array([[10.0, 10.0, 10.0, 10.0]], dtype=np.float32)
    assert ae.score_known(x_benign[:1])[0] > ae.score_known(outlier)[0]

    # 3. Isolation Forest: inlier vs outlier
    ifo = IsolationForestBaseline(n_estimators=30, random_state=42)
    ifo.fit(x_benign)
    assert ifo.score_known(x_benign[:1])[0] > ifo.score_known(outlier)[0]


def test_comparison_table_structure_and_isolation(tmp_path: Path):
    """Verify comparison table columns, source values, NA handling, and synthetic safety."""
    res_dir = tmp_path / "results"
    out_dir = tmp_path / "baselines_out"

    # 1. Real-data comparison table (contains upstream targets)
    tables_real = build_comparison_tables(results_dir=res_dir, output_dir=out_dir)
    assert Path(tables_real["csv"]).name == "comparison.csv"
    assert Path(tables_real["md"]).name == "comparison.md"

    with open(tables_real["csv"]) as f:
        header = f.readline().strip().split(",")
        assert header == COLUMNS

    # 2. Synthetic comparison table (never writes to real comparison.csv, contains banner)
    tables_syn = build_comparison_tables(
        results_dir=res_dir, output_dir=out_dir, synthetic_mode=True
    )
    assert Path(tables_syn["csv"]).name == "comparison_synthetic.csv"
    assert Path(tables_syn["md"]).name == "comparison_synthetic.md"
    assert not (out_dir / "comparison_synthetic.csv").samefile(out_dir / "comparison.csv")

    with open(tables_syn["md"]) as f:
        content = f.read()
        assert "SYNTHETIC DATA RESULTS ONLY" in content

    # 3. Validate sources in targets
    for target in UPSTREAM_TARGETS:
        assert target["source"] in ("re-run", "upstream-reported")
        assert target["data_source"] == "real"
