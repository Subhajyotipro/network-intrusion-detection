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
"""Unit tests for neural network architectures (CLAD ContrastiveMLP and CLOSRMLP)."""

import numpy as np
import pytest
import tensorflow as tf

from nids.config import get_default_config, load_config
from nids.losses import clad_loss, closr_loss
from nids.models import CLOSRMLP, ContrastiveMLP, get_model


@pytest.mark.parametrize("d_in", [8, 72])
@pytest.mark.parametrize("d_out", [8, 16])
def test_clad_output_shapes_and_unit_norm(d_in: int, d_out: int):
    """Verify CLAD model produces [batch, d_out] and unit-norm embeddings for multiple d_in."""
    model = ContrastiveMLP(d_in=d_in, hidden=[32, 32], d_out=d_out, dropout=0.0)
    x = tf.random.normal([5, d_in])
    out = model(x, training=False)

    assert out.shape == (5, d_out)
    norms = tf.norm(out, axis=-1).numpy()
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@pytest.mark.parametrize("d_in", [8, 72])
@pytest.mark.parametrize("n_classes", [3, 12])
def test_closr_output_shapes_and_unit_norm(d_in: int, n_classes: int):
    """Verify CLOSR model produces [batch, n_classes, d_out] with unit-norm per class."""
    d_out = 16
    model = CLOSRMLP(d_in=d_in, hidden=[32, 32], d_out=d_out, n_classes=n_classes, dropout=0.1)
    x = tf.random.normal([4, d_in])
    out = model(x, training=False)

    assert out.shape == (4, n_classes, d_out)
    norms = tf.norm(out, axis=-1).numpy()
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_parameter_counts_match_analytic_formulas():
    """Verify total parameter count matches exact analytic formula for Dense and Residual layers."""
    d_in, hidden, d_out = 8, [16, 32], 4
    model_plain = ContrastiveMLP(d_in=d_in, hidden=hidden, d_out=d_out, residual=False)
    # Build variables with dummy forward pass
    model_plain(tf.zeros([1, d_in]))

    # Layer 0: 8*16 + 16 = 144
    # Layer 1: 16*32 + 32 = 544
    # Head: 32*4 + 4 = 132
    # Total = 820
    expected_plain = (8 * 16 + 16) + (16 * 32 + 32) + (32 * 4 + 4)
    actual_plain = sum(np.prod(v.shape) for v in model_plain.trainable_variables)
    assert actual_plain == expected_plain

    # With residual connections:
    # Layer 0: in 8, out 16 (resized): 144 + (8*16 + 16) = 288
    # Layer 1: in 16, out 32 (resized): 544 + (16*32 + 32) = 1088
    # Head: 132
    # Total = 1508
    model_res = ContrastiveMLP(d_in=d_in, hidden=hidden, d_out=d_out, residual=True)
    model_res(tf.zeros([1, d_in]))
    expected_res = 288 + 1088 + 132
    actual_res = sum(np.prod(v.shape) for v in model_res.trainable_variables)
    assert actual_res == expected_res


def test_training_flag_determinism_and_stochasticity():
    """Verify training=False is deterministic and training=True with dropout>0 is stochastic."""
    model = CLOSRMLP(d_in=8, hidden=[64, 64], d_out=16, n_classes=4, dropout=0.5)
    x = tf.random.normal([10, 8])

    # Deterministic in evaluation mode
    eval1 = model(x, training=False).numpy()
    eval2 = model(x, training=False).numpy()
    np.testing.assert_array_equal(eval1, eval2)

    # Stochastic in training mode
    train1 = model(x, training=True).numpy()
    train2 = model(x, training=True).numpy()
    assert not np.allclose(train1, train2)


def test_gradient_flow_to_all_trainable_parameters():
    """Verify GradientTape on model + loss delivers non-None finite gradients to all params."""
    d_in, n_classes, d_out = 8, 3, 4
    model = CLOSRMLP(d_in=d_in, hidden=[16, 16], d_out=d_out, n_classes=n_classes, dropout=0.0)
    x = tf.random.normal([6, d_in])
    # Batch exercises all classes [0, 1, 2, 0, 1, 2]
    y = tf.constant([0, 1, 2, 0, 1, 2], dtype=tf.int32)

    with tf.GradientTape() as tape:
        embeddings = model(x, training=True)
        loss = closr_loss(embeddings, y)

    grads = tape.gradient(loss, model.trainable_variables)
    assert len(grads) == len(model.trainable_variables)
    for g, v in zip(grads, model.trainable_variables):
        assert g is not None, f"Gradient for {v.name} was None!"
        assert tf.reduce_all(tf.math.is_finite(g)), f"Gradient for {v.name} had NaN/Inf!"
    assert any(float(tf.reduce_sum(tf.abs(g))) > 0.0 for g in grads)


def test_loss_decreases_over_training_steps():
    """Verify that CLAD and CLOSR models decrease loss over 20 AdamW steps on separable toy data."""
    tf.keras.utils.set_random_seed(42)

    # 1. CLAD training test
    clad = ContrastiveMLP(d_in=4, hidden=[16, 16], d_out=4, dropout=0.0)
    # Well-separated clusters: benign around +2, malicious around -2
    x_clad = tf.concat([tf.random.normal([10, 4]) + 2.0, tf.random.normal([10, 4]) - 2.0], axis=0)
    y_clad = tf.constant([0] * 10 + [1] * 10, dtype=tf.int32)
    opt_clad = tf.keras.optimizers.AdamW(learning_rate=0.05)

    initial_loss_clad = clad_loss(clad(x_clad, training=False), y_clad).numpy()
    for _ in range(20):
        with tf.GradientTape() as tape:
            loss = clad_loss(clad(x_clad, training=True), y_clad)
        grads = tape.gradient(loss, clad.trainable_variables)
        opt_clad.apply_gradients(zip(grads, clad.trainable_variables))
    final_loss_clad = clad_loss(clad(x_clad, training=False), y_clad).numpy()
    msg_clad = f"CLAD loss did not decrease: {initial_loss_clad} -> {final_loss_clad}"
    assert final_loss_clad < initial_loss_clad, msg_clad

    # 2. CLOSR training test
    closr = CLOSRMLP(d_in=4, hidden=[16, 16], d_out=4, n_classes=2, dropout=0.0)
    opt_closr = tf.keras.optimizers.AdamW(learning_rate=0.05)
    initial_loss_closr = closr_loss(closr(x_clad, training=False), y_clad).numpy()
    for _ in range(20):
        with tf.GradientTape() as tape:
            loss = closr_loss(closr(x_clad, training=True), y_clad)
        grads = tape.gradient(loss, closr.trainable_variables)
        opt_closr.apply_gradients(zip(grads, closr.trainable_variables))
    final_loss_closr = closr_loss(closr(x_clad, training=False), y_clad).numpy()
    msg_closr = f"CLOSR loss did not decrease: {initial_loss_closr} -> {final_loss_closr}"
    assert final_loss_closr < initial_loss_closr, msg_closr


def test_batch_size_one_and_config_get_model():
    """Verify batch size 1 pass and get_model factory with CLI override behavior."""
    cfg = get_default_config("closr")
    model = get_model(cfg, d_in=8)
    out_one = model(tf.random.normal([1, 8]), training=False)
    assert out_one.shape == (1, cfg.n_classes, cfg.d_out)
    np.testing.assert_allclose(tf.norm(out_one, axis=-1).numpy(), 1.0, atol=1e-5)

    # CLI override --set residual=false works cleanly
    cfg_override = load_config(overrides=["residual=false"])
    assert cfg_override.residual is False
