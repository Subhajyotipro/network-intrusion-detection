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
"""Edge-case unit tests for CLAD and CLOSR losses: duplicates, active pairs, and imbalance."""

from __future__ import annotations

import numpy as np
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


def test_identical_batch_finite_loss_and_no_self_pairs():
    """Verify a batch of identical vectors yields finite loss and 0 self-pairs counted (DV16)."""
    bsz = 10
    x_clad = tf.ones((bsz, 8), dtype=tf.float32)
    y_clad = tf.zeros((bsz,), dtype=tf.int32)

    l_clad = clad_loss(x_clad, y_clad).numpy()
    f_clad = clad_fraction_pos(x_clad, y_clad).numpy()
    assert np.isfinite(l_clad) and l_clad >= 0.0
    assert np.isclose(f_clad, 0.0)

    x_closr = tf.ones((bsz, 4, 16), dtype=tf.float32)
    y_closr = tf.zeros((bsz,), dtype=tf.int32)
    l_closr = closr_loss(x_closr, y_closr).numpy()
    f_closr = closr_fraction_pos(x_closr, y_closr).numpy()
    assert np.isfinite(l_closr) and l_closr >= 0.0
    assert np.isclose(f_closr, 0.0)

    l_np = np_clad_loss(x_clad.numpy(), y_clad.numpy())
    f_np = np_clad_fraction_pos(x_clad.numpy(), y_clad.numpy())
    assert np.isfinite(l_np) and l_np >= 0.0
    assert np.isclose(f_np, 0.0)


def test_self_pair_float32_roundoff_exceeds_eps_matches_reference():
    """Verify batch with float32 self-distance > eps matches reference without distortion."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal((16, 8)).astype(np.float32)
    v_norm = v / np.linalg.norm(v, axis=-1, keepdims=True)
    y = np.zeros((16,), dtype=np.int32)

    diag_f32 = np.diag((1.0 - v_norm @ v_norm.T) / 2.0)
    assert np.any(diag_f32 > 1e-8), "Precondition: float32 roundoff exceeds eps=1e-8"

    tf_loss = float(clad_loss(tf.constant(v_norm), tf.constant(y)))
    tf_frac = float(clad_fraction_pos(tf.constant(v_norm), tf.constant(y)))
    np_loss = float(np_clad_loss(v_norm, y))
    np_frac = float(np_clad_fraction_pos(v_norm, y))

    np.testing.assert_allclose(tf_loss, np_loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(tf_frac, np_frac, rtol=1e-5, atol=1e-6)


def test_duplicate_rows_parity_with_numpy_reference():
    """Verify duplicate row edge cases match independent NumPy float64 reference."""
    rng = np.random.default_rng(42)
    x_a = rng.standard_normal((20, 12, 64), dtype=np.float32)
    y_a = rng.integers(0, 12, size=20, dtype=np.int64)
    x_a[1] = x_a[0]
    y_a[1] = y_a[0]

    x_b = rng.standard_normal((20, 12, 64), dtype=np.float32)
    y_b = rng.integers(0, 12, size=20, dtype=np.int64)
    x_b[1] = x_b[0]
    y_b[1] = (y_b[0] + 1) % 12

    for x, y in [(x_a, y_a), (x_b, y_b)]:
        l_tf = float(closr_loss(tf.constant(x), tf.constant(y)))
        l_np = float(np_closr_loss(x, y))
        f_tf = float(closr_fraction_pos(tf.constant(x), tf.constant(y)))
        f_np = float(np_closr_fraction_pos(x, y))
        np.testing.assert_allclose(l_tf, l_np, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(f_tf, f_np, rtol=1e-5, atol=1e-6)

        x_clad = x[:, 0, :]
        l_tf_c = float(clad_loss(tf.constant(x_clad), tf.constant(y)))
        l_np_c = float(np_clad_loss(x_clad, y))
        f_tf_c = float(clad_fraction_pos(tf.constant(x_clad), tf.constant(y)))
        f_np_c = float(np_clad_fraction_pos(x_clad, y))
        np.testing.assert_allclose(l_tf_c, l_np_c, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(f_tf_c, f_np_c, rtol=1e-5, atol=1e-6)


def test_active_vs_inactive_hinge_pairs_analytical():
    """Verify loss and fraction_pos on batch with both active and inactive dissimilar pairs."""
    x_c0 = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32
    )
    x_c1 = np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
    )
    x_closr = np.stack([x_c0, x_c1], axis=1)
    y = np.array([0, 0, 1, 1], dtype=np.int32)

    l_tf_c, f_tf_c = clad_loss_and_frac(
        tf.constant(x_c0), tf.constant(y), target_class=0, margin=0.8, squared=True
    )
    l_np_c = np_clad_loss(x_c0, y, target_class=0, margin=0.8, squared=True)
    f_np_c = np_clad_fraction_pos(x_c0, y, target_class=0, margin=0.8, squared=True)

    np.testing.assert_allclose(float(l_tf_c), 0.045, atol=1e-5)
    np.testing.assert_allclose(float(f_tf_c), 0.5, atol=1e-5)
    np.testing.assert_allclose(float(l_tf_c), l_np_c, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(float(f_tf_c), f_np_c, rtol=1e-5, atol=1e-6)

    l_tf, f_tf = closr_loss_and_frac(tf.constant(x_closr), tf.constant(y), margin=0.8, squared=True)
    l_np = np_closr_loss(x_closr, y, margin=0.8, squared=True)
    f_np = np_closr_fraction_pos(x_closr, y, margin=0.8, squared=True)

    np.testing.assert_allclose(float(l_tf), 0.045, atol=1e-5)
    np.testing.assert_allclose(float(f_tf), 0.5, atol=1e-5)
    np.testing.assert_allclose(float(l_tf), l_np, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(float(f_tf), f_np, rtol=1e-5, atol=1e-6)


def test_imbalanced_class_counts_closr():
    """Verify CLOSR loss on imbalanced batch counts [5, 1, 12, 30] matches NumPy reference."""
    rng = np.random.default_rng(123)
    counts = [5, 1, 12, 30]
    n_classes = len(counts)
    y_list = []
    for c, cnt in enumerate(counts):
        y_list.extend([c] * cnt)
    y_np = np.array(y_list, dtype=np.int32)
    bsz = len(y_np)
    x_np = rng.standard_normal((bsz, n_classes, 16)).astype(np.float32)

    l_tf = float(closr_loss(tf.constant(x_np), tf.constant(y_np)))
    l_np = np_closr_loss(x_np, y_np)
    f_tf = float(closr_fraction_pos(tf.constant(x_np), tf.constant(y_np)))
    f_np = np_closr_fraction_pos(x_np, y_np)

    np.testing.assert_allclose(l_tf, l_np, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(f_tf, f_np, rtol=1e-5, atol=1e-6)


def test_absent_class_in_batch_closr():
    """Verify CLOSR loss with absent classes in batch matches NumPy reference."""
    rng = np.random.default_rng(456)
    # Model has 5 classes, batch only has samples for classes 0, 2, 4 (1 and 3 are absent)
    y_np = np.array([0, 0, 2, 2, 4, 4, 4], dtype=np.int32)
    x_np = rng.standard_normal((len(y_np), 5, 16)).astype(np.float32)

    l_tf = float(closr_loss(tf.constant(x_np), tf.constant(y_np)))
    l_np = np_closr_loss(x_np, y_np)
    f_tf = float(closr_fraction_pos(tf.constant(x_np), tf.constant(y_np)))
    f_np = np_closr_fraction_pos(x_np, y_np)

    np.testing.assert_allclose(l_tf, l_np, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(f_tf, f_np, rtol=1e-5, atol=1e-6)
