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
"""Unit tests for CLAD and CLOSR loss functions, verifying NumPy equivalence."""

import numpy as np
import pytest
import tensorflow as tf

from nids.losses import (
    clad_fraction_pos,
    clad_loss,
    clad_loss_and_frac,
    closr_fraction_pos,
    closr_loss,
    closr_loss_and_frac,
)
from tests.unit.np_reference import (
    np_clad_fraction_pos,
    np_clad_loss,
    np_closr_fraction_pos,
    np_closr_loss,
)


@pytest.mark.parametrize("seed", [42, 101, 2024])
@pytest.mark.parametrize("margin", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("squared", [True, False])
def test_clad_loss_matches_numpy_reference(seed: int, margin: float, squared: bool):
    """Verify TF clad_loss matches independent NumPy float64 reference within tight tolerance."""
    rng = np.random.default_rng(seed)
    bsz, dim = 16, 8
    x_np = rng.standard_normal((bsz, dim)).astype(np.float32)
    # Binary labels: 0 (benign) and 1 (malicious)
    y_np = rng.integers(0, 2, size=bsz).astype(np.int32)

    expected_loss = np_clad_loss(x_np, y_np, target_class=0, margin=margin, squared=squared)
    expected_frac = np_clad_fraction_pos(x_np, y_np, target_class=0, margin=margin, squared=squared)

    x_tf = tf.convert_to_tensor(x_np)
    y_tf = tf.convert_to_tensor(y_np)

    actual_loss = clad_loss(x_tf, y_tf, target_class=0, margin=margin, squared=squared).numpy()
    actual_frac = clad_fraction_pos(
        x_tf, y_tf, target_class=0, margin=margin, squared=squared
    ).numpy()

    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual_frac, expected_frac, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("seed", [42, 123])
@pytest.mark.parametrize("margin", [0.8, 1.0])
@pytest.mark.parametrize("squared", [True, False])
def test_closr_loss_matches_numpy_reference(seed: int, margin: float, squared: bool):
    """Verify TF closr_loss matches independent NumPy reference across classes and heads."""
    rng = np.random.default_rng(seed)
    bsz, n_classes, dim = 8, 4, 16
    x_np = rng.standard_normal((bsz, n_classes, dim)).astype(np.float32)
    y_np = rng.integers(0, n_classes, size=bsz).astype(np.int32)

    expected_loss = np_closr_loss(x_np, y_np, margin=margin, squared=squared)
    expected_frac = np_closr_fraction_pos(x_np, y_np, margin=margin, squared=squared)

    x_tf = tf.convert_to_tensor(x_np)
    y_tf = tf.convert_to_tensor(y_np)

    actual_loss = closr_loss(x_tf, y_tf, margin=margin, squared=squared).numpy()
    actual_frac = closr_fraction_pos(x_tf, y_tf, margin=margin, squared=squared).numpy()

    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual_frac, expected_frac, rtol=1e-5, atol=1e-6)


def test_hand_computed_clad_case():
    """Verify CLAD loss on an analytically derived 3-sample case.

    Setup:
      Sample 0: y=0 (benign),    u0 = [1, 0]
      Sample 1: y=0 (benign),    u1 = [0, 1]
      Sample 2: y=1 (malicious), u2 = [1, 0]
      Margin = 1.0, squared = True, alpha = 0.5.

    Derivation:
      Cosine similarities: S01=0, S02=1, S12=0
      Distances Dij = (1 - S)/2:
        D01 = D10 = 0.5
        D02 = 0.0
        D12 = 0.5
      Similar pairs (both y=0): (0,1) and (1,0) have D=0.5.
        sim_loss = (0.5^2 + 0.5^2) / 2 = 0.25.
      Dissimilar pairs (row y=0, col y=1):
        (0,2): D02 = 0.0 -> relu(1.0 - 0.0)^2 = 1.0^2 = 1.0
        (1,2): D12 = 0.5 -> relu(1.0 - 0.5)^2 = 0.5^2 = 0.25
        dissim_loss = (1.0 + 0.25) / 2 = 0.625.
      Total loss = 0.5 * 0.25 + 0.5 * 0.625 = 0.125 + 0.3125 = 0.4375.
    """
    x = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=tf.float32)
    y = tf.constant([0, 0, 1], dtype=tf.int32)

    loss = clad_loss(x, y, target_class=0, margin=1.0, squared=True, alpha=0.5).numpy()
    frac = clad_fraction_pos(x, y, target_class=0, margin=1.0, squared=True, alpha=0.5).numpy()

    assert np.isclose(loss, 0.4375, atol=1e-6)
    assert np.isclose(frac, 1.0, atol=1e-6)


def test_hand_computed_closr_case():
    """Verify CLOSR loss on an analytically derived 2-sample, 2-class case.

    Setup:
      Sample 0: y=0, class-0 head=[1, 0], class-1 head=[0, 1]
      Sample 1: y=1, class-0 head=[0, 1], class-1 head=[0, 1]
      Margin = 1.0, squared = True, alpha = 0.5.

    Derivation:
      Class 0:
        Sample 0 has y=0 (target). Sample 1 has y=1 (dissimilar).
        D01 = (1 - [1,0]·[0,1]) / 2 = 0.5.
        sim_loss = 0.0 (only 1 sample of class 0).
        dissim_loss = relu(1.0 - 0.5)^2 = 0.25 / 1 = 0.25.
        loss_0 = 0.5 * 0.0 + 0.5 * 0.25 = 0.125.
      Class 1:
        Sample 1 has y=1 (target). Sample 0 has y=0 (dissimilar).
        Both heads are [0, 1] -> similarity = 1.0 -> D10 = 0.0.
        sim_loss = 0.0 (only 1 sample of class 1).
        dissim_loss = relu(1.0 - 0.0)^2 = 1.0^2 = 1.0.
        loss_1 = 0.5 * 0.0 + 0.5 * 1.0 = 0.5.
      Mean CLOSR loss = (0.125 + 0.5) / 2 = 0.3125.
    """
    # Shape [2, 2, 2]: batch 2, classes 2, dim 2
    x = tf.constant(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=tf.float32,
    )
    y = tf.constant([0, 1], dtype=tf.int32)

    loss = closr_loss(x, y, margin=1.0, squared=True, alpha=0.5).numpy()
    assert np.isclose(loss, 0.3125, atol=1e-6)


def test_asymmetric_property_of_clad():
    """Verify that CLAD treats benign vs malicious asymmetrically (attracts only benign)."""
    # Three samples where sample 1 and 2 have non-zero distance between them
    u = np.array([[1.0, 0.0], [0.0, 1.0], [np.sqrt(0.5), np.sqrt(0.5)]], dtype=np.float32)
    x = tf.convert_to_tensor(u)

    # When samples 1 & 2 are benign (class 0), their mutual attraction term is activated
    y_benign = tf.constant([1, 0, 0], dtype=tf.int32)
    loss_benign = clad_loss(x, y_benign, target_class=0).numpy()

    # When samples 1 & 2 are malicious (class 1), no mutual attraction occurs
    y_malicious = tf.constant([0, 1, 1], dtype=tf.int32)
    loss_malicious = clad_loss(x, y_malicious, target_class=0).numpy()

    # Because benign pairs are pulled together while malicious are not, the losses MUST differ
    assert not np.isclose(loss_benign, loss_malicious, atol=1e-3)


def test_margin_and_squared_sensitivity():
    """Verify margin and squared flags alter loss value monotonically as predicted."""
    rng = np.random.default_rng(42)
    x = tf.convert_to_tensor(rng.standard_normal((10, 8), dtype=np.float32))
    y = tf.convert_to_tensor(rng.integers(0, 2, size=10, dtype=np.int32))

    l_m05 = clad_loss(x, y, margin=0.5, squared=True).numpy()
    l_m10 = clad_loss(x, y, margin=1.0, squared=True).numpy()
    l_m15 = clad_loss(x, y, margin=1.5, squared=True).numpy()

    # Larger margin forces larger dissimilar penalty -> strictly increasing loss
    assert l_m05 < l_m10 < l_m15

    l_sq = clad_loss(x, y, margin=1.0, squared=True).numpy()
    l_lin = clad_loss(x, y, margin=1.0, squared=False).numpy()
    assert l_sq != l_lin


def test_gradient_flow_and_tf_function():
    """Verify gradients flow to inputs and tf.function produces identical loss."""
    rng = np.random.default_rng(42)
    x = tf.Variable(rng.standard_normal((8, 4), dtype=np.float32))
    y = tf.constant([0, 0, 1, 1, 0, 1, 0, 1], dtype=tf.int32)

    with tf.GradientTape() as tape:
        loss = clad_loss(x, y, margin=1.0, squared=True)

    grads = tape.gradient(loss, x)
    assert grads is not None
    assert tf.reduce_all(tf.math.is_finite(grads))
    assert float(tf.reduce_sum(tf.abs(grads))) > 0.0

    # tf.function compilation check
    compiled_clad = tf.function(clad_loss)
    eager_val = clad_loss(x, y).numpy()
    graph_val = compiled_clad(x, y).numpy()
    np.testing.assert_allclose(eager_val, graph_val, atol=1e-7)


def test_edge_cases_and_non_finite_inputs():
    """Verify robustness to batch size 1, single-class batches, missing classes, and NaNs."""
    # Batch size 1
    x_one = tf.constant([[1.0, 2.0]], dtype=tf.float32)
    y_one = tf.constant([0], dtype=tf.int32)
    assert np.isclose(clad_loss(x_one, y_one).numpy(), 0.0)

    # Single-class batch
    x_single = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    y_single = tf.constant([0, 0], dtype=tf.int32)
    loss_single = clad_loss(x_single, y_single).numpy()
    assert np.isfinite(loss_single) and loss_single >= 0.0

    # CLOSR missing class (batch has labels [0, 1], but model has 4 classes)
    x_closr = tf.random.normal([4, 4, 8])
    y_missing = tf.constant([0, 0, 1, 1], dtype=tf.int32)
    loss_closr = closr_loss(x_closr, y_missing).numpy()
    assert np.isfinite(loss_closr) and loss_closr >= 0.0

    # Non-finite inputs must raise ValueError in eager mode
    x_nan = tf.constant([[np.nan, 1.0]], dtype=tf.float32)
    with pytest.raises(ValueError, match="non-finite"):
        clad_loss(x_nan, y_one)

    x_inf = tf.constant([[np.inf, 1.0]], dtype=tf.float32)
    with pytest.raises(ValueError, match="non-finite"):
        clad_loss(x_inf, y_one)


def test_identical_batch_finite_loss_and_no_self_pairs():
    """Verify a batch of identical vectors yields finite loss and 0 self-pairs counted (DV16)."""
    bsz = 10
    # All vectors identical and benign
    x_clad = tf.ones((bsz, 8), dtype=tf.float32)
    y_clad = tf.zeros((bsz,), dtype=tf.int32)

    l_clad = clad_loss(x_clad, y_clad).numpy()
    f_clad = clad_fraction_pos(x_clad, y_clad).numpy()
    assert np.isfinite(l_clad) and l_clad >= 0.0
    assert np.isclose(f_clad, 0.0)

    # In CLOSR with identical embeddings
    x_closr = tf.ones((bsz, 4, 16), dtype=tf.float32)
    y_closr = tf.zeros((bsz,), dtype=tf.int32)

    l_closr = closr_loss(x_closr, y_closr).numpy()
    f_closr = closr_fraction_pos(x_closr, y_closr).numpy()
    assert np.isfinite(l_closr) and l_closr >= 0.0
    assert np.isclose(f_closr, 0.0)

    # NumPy reference float64 equivalence without self-pair counting
    l_np = np_clad_loss(x_clad.numpy(), y_clad.numpy())
    f_np = np_clad_fraction_pos(x_clad.numpy(), y_clad.numpy())
    assert np.isfinite(l_np) and l_np >= 0.0
    assert np.isclose(f_np, 0.0)


def test_self_pair_float32_roundoff_exceeds_eps_matches_reference():
    """Verify batch with float32 self-distance > eps matches reference without distortion."""
    rng = np.random.default_rng(0)
    # Seed 0 generates float32 unit vectors where 5 diagonal elements evaluate to ~5.96e-8 > 1e-8
    v = rng.standard_normal((16, 8)).astype(np.float32)
    v_norm = v / np.linalg.norm(v, axis=-1, keepdims=True)
    y = np.zeros((16,), dtype=np.int32)

    diag_f32 = np.diag((1.0 - v_norm @ v_norm.T) / 2.0)
    assert np.any(diag_f32 > 1e-8), "Precondition failed: float32 roundoff must exceed eps=1e-8"

    tf_loss = float(clad_loss(tf.constant(v_norm), tf.constant(y)))
    tf_frac = float(clad_fraction_pos(tf.constant(v_norm), tf.constant(y)))
    np_loss = float(np_clad_loss(v_norm, y))
    np_frac = float(np_clad_fraction_pos(v_norm, y))

    np.testing.assert_allclose(tf_loss, np_loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(tf_frac, np_frac, rtol=1e-5, atol=1e-6)


def test_duplicate_rows_parity_with_numpy_reference():
    """Verify duplicate row edge cases (a, b, c) match independent NumPy float64 reference."""
    rng = np.random.default_rng(42)

    # (a) Duplicated rows within the same class
    x_a = rng.standard_normal((20, 12, 64), dtype=np.float32)
    y_a = rng.integers(0, 12, size=20, dtype=np.int64)
    x_a[1] = x_a[0]
    y_a[1] = y_a[0]

    # (b) Duplicated rows across different classes
    x_b = rng.standard_normal((20, 12, 64), dtype=np.float32)
    y_b = rng.integers(0, 12, size=20, dtype=np.int64)
    x_b[1] = x_b[0]
    y_b[1] = (y_b[0] + 1) % 12

    # (c) A batch where every row of a class is identical
    x_c = rng.standard_normal((20, 12, 64), dtype=np.float32)
    y_c = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    for i in range(1, 10):
        x_c[i] = x_c[0]

    cases = [
        ("same_class_dupes", x_a, y_a),
        ("cross_class_dupes", x_b, y_b),
        ("identical_class_rows", x_c, y_c),
    ]

    for _, x, y in cases:
        # CLOSR loss & fraction pos
        l_tf = float(closr_loss(tf.constant(x), tf.constant(y)))
        l_np = float(np_closr_loss(x, y))
        f_tf = float(closr_fraction_pos(tf.constant(x), tf.constant(y)))
        f_np = float(np_closr_fraction_pos(x, y))
        np.testing.assert_allclose(l_tf, l_np, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(f_tf, f_np, rtol=1e-5, atol=1e-6)

        # CLAD loss & fraction pos (use class-0 slice)
        x_clad = x[:, 0, :]
        l_tf_c = float(clad_loss(tf.constant(x_clad), tf.constant(y)))
        l_np_c = float(np_clad_loss(x_clad, y))
        f_tf_c = float(clad_fraction_pos(tf.constant(x_clad), tf.constant(y)))
        f_np_c = float(np_clad_fraction_pos(x_clad, y))
        np.testing.assert_allclose(l_tf_c, l_np_c, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(f_tf_c, f_np_c, rtol=1e-5, atol=1e-6)


def test_active_vs_inactive_hinge_pairs_analytical():
    """Verify loss and fraction_pos on batch with both active and inactive dissimilar pairs."""
    # Class 0: samples 0 and 1 (sample 1 duplicate of 0 -> d=0 <= eps)
    # Class 1: samples 2 and 3 (sample 2 d=0.5 < 0.8 -> active; sample 3 d=1.0 > 0.8 -> inactive)
    x_c0 = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32
    )
    x_c1 = np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
    )
    x_closr = np.stack([x_c0, x_c1], axis=1)
    y = np.array([0, 0, 1, 1], dtype=np.int32)

    # CLAD analytical: n_sim=0, 2 active dissimilar (diff=0.3), 2 inactive (diff=0).
    # dissim_loss = (0.3^2 * 2) / 2 = 0.09; loss = 0.5 * 0.09 = 0.045; frac_pos = 2/4 = 0.5
    l_tf_c, f_tf_c = clad_loss_and_frac(
        tf.constant(x_c0), tf.constant(y), target_class=0, margin=0.8, squared=True
    )
    l_np_c = np_clad_loss(x_c0, y, target_class=0, margin=0.8, squared=True)
    f_np_c = np_clad_fraction_pos(x_c0, y, target_class=0, margin=0.8, squared=True)

    np.testing.assert_allclose(float(l_tf_c), 0.045, atol=1e-5)
    np.testing.assert_allclose(float(f_tf_c), 0.5, atol=1e-5)
    np.testing.assert_allclose(float(l_tf_c), l_np_c, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(float(f_tf_c), f_np_c, rtol=1e-5, atol=1e-6)

    # CLOSR analytical: symmetrical across both classes -> mean loss = 0.045, frac = 0.5
    l_tf, f_tf = closr_loss_and_frac(tf.constant(x_closr), tf.constant(y), margin=0.8, squared=True)
    l_np = np_closr_loss(x_closr, y, margin=0.8, squared=True)
    f_np = np_closr_fraction_pos(x_closr, y, margin=0.8, squared=True)

    np.testing.assert_allclose(float(l_tf), 0.045, atol=1e-5)
    np.testing.assert_allclose(float(f_tf), 0.5, atol=1e-5)
    np.testing.assert_allclose(float(l_tf), l_np, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(float(f_tf), f_np, rtol=1e-5, atol=1e-6)
