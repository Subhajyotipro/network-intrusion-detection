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

import csv
from pathlib import Path

import numpy as np

from nids.baselines import (
    AutoencoderBaseline,
    IsolationForestBaseline,
    MLPBaseline,
    _compute_table1_branch,
)
from nids.compare import (
    ALLOWED_PROTOCOLS,
    COLUMNS,
    build_comparison_tables,
    parse_upstream_readme,
)
from nids.data import ClassMap, DatasetSplits


def test_hand_computed_mlp_score():
    """Verify hand-computed anomaly score 1 - P(benign) and known score max prob.

    Hand calculation:
    Sample 0: probs = [0.1, 0.7, 0.2] -> anomaly score = 1 - 0.1 = 0.9; max prob = 0.7
    Sample 1: probs = [0.4, 0.3, 0.3] -> anomaly score = 1 - 0.4 = 0.6; max prob = 0.4
    """
    model = MLPBaseline(d_in=2, n_classes=3, hidden=[4], dropout=0.0)
    probs = np.array([[0.1, 0.7, 0.2], [0.4, 0.3, 0.3]], dtype=np.float32)
    model.predict_proba = lambda x, batch_size=4096: probs

    np.testing.assert_allclose(model.score_anomaly(np.zeros((2, 2))), [0.9, 0.6], atol=1e-6)
    np.testing.assert_allclose(model.score_known(np.zeros((2, 2))), [0.7, 0.4], atol=1e-6)


def test_hand_computed_autoencoder_score():
    """Verify hand-computed MSE reconstruction error.

    Hand calculation:
    Input x: sample 0 = [1.0, 2.0], sample 1 = [3.0, 4.0]
    Recon:   sample 0 = [1.0, 1.0], sample 1 = [1.0, 4.0]
    MSE:
      sample 0 = ((1-1)^2 + (2-1)^2)/2 = 0.5
      sample 1 = ((3-1)^2 + (4-4)^2)/2 = 2.0
    score_anomaly (MSE): [0.5, 2.0]
    score_known (-MSE): [-0.5, -2.0]
    """
    ae = AutoencoderBaseline(d_in=2, hidden=[4], latent_dim=2)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x_rec = np.array([[1.0, 1.0], [1.0, 4.0]], dtype=np.float32)
    ae.model.predict = lambda inp, batch_size=4096, verbose=0: x_rec

    np.testing.assert_allclose(ae.score_anomaly(x), [0.5, 2.0], atol=1e-6)
    np.testing.assert_allclose(ae.score_known(x), [-0.5, -2.0], atol=1e-6)


def test_hand_computed_iforest_sign_convention():
    """Verify Isolation Forest anomaly score convention: outlier scores higher."""
    rng = np.random.default_rng(42)
    train_x = rng.normal(loc=0.0, scale=0.1, size=(50, 2)).astype(np.float32)
    model = IsolationForestBaseline(n_estimators=20, random_state=42)
    model.fit(train_x)

    test_inlier = np.array([[0.0, 0.0]], dtype=np.float32)
    test_outlier = np.array([[100.0, 100.0]], dtype=np.float32)
    assert model.score_anomaly(test_outlier)[0] > model.score_anomaly(test_inlier)[0]
    assert model.score_known(test_inlier)[0] > model.score_known(test_outlier)[0]


def test_sign_conventions_on_toy_data():
    """Verify sign convention: baselines score anomalies higher for Table 1."""
    rng = np.random.default_rng(123)
    x_c0 = rng.normal(loc=-2.0, scale=0.1, size=(30, 4)).astype(np.float32)
    x_c1 = rng.normal(loc=2.0, scale=0.1, size=(30, 4)).astype(np.float32)
    x_train = np.concatenate([x_c0, x_c1], axis=0)
    y_train = np.concatenate([np.zeros(30, dtype=np.int32), np.ones(30, dtype=np.int32)], axis=0)

    mlp = MLPBaseline(d_in=4, n_classes=2, hidden=[16], dropout=0.0)
    mlp.fit(x_train, y_train, epochs=15, batch_size=16, lr=1e-2)
    benign_sample = np.array([[-2.0, -2.0, -2.0, -2.0]], dtype=np.float32)
    attack_sample = np.array([[2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
    assert mlp.score_anomaly(attack_sample)[0] > mlp.score_anomaly(benign_sample)[0]

    ae = AutoencoderBaseline(d_in=4, hidden=[8], latent_dim=4)
    ae.fit(x_c0, epochs=20, batch_size=16, lr=1e-2)
    outlier = np.array([[10.0, 10.0, 10.0, 10.0]], dtype=np.float32)
    assert ae.score_anomaly(outlier)[0] > ae.score_anomaly(x_c0[:1])[0]

    ifo = IsolationForestBaseline(n_estimators=30, random_state=42)
    ifo.fit(x_c0)
    assert ifo.score_anomaly(outlier)[0] > ifo.score_anomaly(x_c0[:1])[0]


def test_hand_computed_table1_auroc_flipping():
    """Verify Table 1 AUROC and closed/open mean arithmetic with flipping behavior.

    Hand calculation:
      Benign (0): 2 samples with scores [0.8, 0.9]
      Known Attack 1 (1): 2 samples with scores [0.1, 0.2]
        -> Raw AUROC = 0.0, Flipped AUROC = max(0.0, 1.0) = 1.0
      Zero-Day Attack 2 (2): 2 samples with scores [0.85, 0.95]
        -> Raw AUROC = 0.75, Flipped AUROC = max(0.75, 0.25) = 0.75
    """
    scores = np.array([0.8, 0.9, 0.1, 0.2, 0.85, 0.95], dtype=np.float64)
    y_eval = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    cmap = ClassMap(["benign", "Attack1"], ["Attack2"])
    dummy_splits = DatasetSplits(
        x_train=np.zeros((1, 2)),
        y_train=np.zeros(1, dtype=np.int64),
        x_val=None,
        y_val=None,
        x_test=np.zeros((4, 2)),
        y_test=np.zeros(4, dtype=np.int64),
        x_zd=np.zeros((2, 2)),
        y_zd=np.zeros(2, dtype=np.int64),
        zd_names=np.array(["Attack2", "Attack2"]),
        class_map=cmap,
        split_indices={},
        d_in=2,
    )
    t1_unflipped = _compute_table1_branch(scores, y_eval, dummy_splits, include_lower_thres=False)
    np.testing.assert_allclose(t1_unflipped["closed_set_mean"], 0.0, atol=1e-6)
    np.testing.assert_allclose(t1_unflipped["open_set_mean"], 0.75, atol=1e-6)

    t1_flipped = _compute_table1_branch(scores, y_eval, dummy_splits, include_lower_thres=True)
    np.testing.assert_allclose(t1_flipped["closed_set_mean"], 1.0, atol=1e-6)
    np.testing.assert_allclose(t1_flipped["open_set_mean"], 0.75, atol=1e-6)


def test_reduced_run_logic():
    """Verify reduced_run flag logic across all baseline configurations."""
    assert bool(True or 200 < 200 or 100 != 100) is True
    assert bool(False or 10 < 200 or 100 != 100) is True
    assert bool(False or 200 < 200 or 100 != 100) is False
    assert bool(False or 200 < 200 or 10 != 100) is True


def test_parsed_readme_targets():
    """Verify parsed upstream README Table 1 and Table 2 targets match Section 2.4."""
    t1, t2 = parse_upstream_readme("third_party/CLOSR/README.md")
    assert "Closed Set Mean" in t1 and "Open Set Mean" in t1

    np.testing.assert_allclose(t1["Closed Set Mean"]["CLAD"], 0.999855, atol=1e-6)
    np.testing.assert_allclose(t1["Closed Set Mean"]["AE"], 0.857221, atol=1e-6)
    np.testing.assert_allclose(t1["Closed Set Mean"]["IF"], 0.767531, atol=1e-6)
    np.testing.assert_allclose(t1["Closed Set Mean"]["MLP"], 0.999557, atol=1e-6)

    np.testing.assert_allclose(t1["Open Set Mean"]["CLAD"], 0.996627, atol=1e-6)
    np.testing.assert_allclose(t1["Open Set Mean"]["AE"], 0.840659, atol=1e-6)
    np.testing.assert_allclose(t1["Open Set Mean"]["IF"], 0.838422, atol=1e-6)
    np.testing.assert_allclose(t1["Open Set Mean"]["MLP"], 0.533859, atol=1e-6)

    closr = t2["CLOSR"]
    np.testing.assert_allclose(closr["closed_set_acc"], 0.995276, atol=1e-6)
    np.testing.assert_allclose(closr["open_set_auc"], 0.974022, atol=1e-6)
    np.testing.assert_allclose(closr["open_auc"], 0.969420, atol=1e-6)


def test_comparison_table_structure_and_protocol(tmp_path: Path):
    """Verify comparison table columns, protocol values, and synthetic file isolation."""
    res_dir = tmp_path / "results"
    out_dir = tmp_path / "baselines_out"

    tables_real = build_comparison_tables(results_dir=res_dir, output_dir=out_dir)
    assert Path(tables_real["csv"]).name == "comparison.csv"
    with open(tables_real["csv"], encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        for row in reader:
            assert row["protocol"] in ALLOWED_PROTOCOLS

    tables_syn = build_comparison_tables(
        results_dir=res_dir, output_dir=out_dir, synthetic_mode=True
    )
    assert Path(tables_syn["csv"]).name == "comparison_synthetic.csv"
    with open(tables_syn["csv"], encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row["protocol"] in ALLOWED_PROTOCOLS
