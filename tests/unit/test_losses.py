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
    closr_fraction_pos,
    closr_loss,
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
