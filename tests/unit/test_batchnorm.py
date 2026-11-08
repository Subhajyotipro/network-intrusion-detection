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
"""Unit tests verifying BatchNorm settings, moving stats, determinism, and persistence."""

import numpy as np
import tensorflow as tf

from nids.config import Config
from nids.models import ResidualBlock, get_model


def test_batchnorm_settings_match_upstream_pytorch_equivalent():
    """Verify BN momentum/epsilon match PyTorch defaults (momentum 0.1 -> Keras 0.9, eps 1e-5)."""
    cfg = Config(batch_norm=True, bn_momentum=0.9, bn_epsilon=1e-5, d_in=16)
    model = get_model(cfg)
    # Check that each BN layer inside the backbone has momentum=0.9 and epsilon=1e-5
    bn_layers = [layer.bn for layer in model.backbone.layers if getattr(layer, "bn", None)]
    assert len(bn_layers) > 0
    for bn in bn_layers:
        assert np.isclose(bn.momentum, 0.9)
        assert np.isclose(bn.epsilon, 1e-5)

    # Also verify direct block initialization
    block = ResidualBlock(in_dim=16, out_dim=16, batch_norm=True, bn_momentum=0.9, bn_epsilon=1e-5)
    _ = block(tf.zeros([2, 16]))
    assert np.isclose(block.bn.momentum, 0.9)
    assert np.isclose(block.bn.epsilon, 1e-5)


def test_batchnorm_training_updates_moving_stats_eval_does_not():
    """Verify training=True updates moving mean/variance and training=False leaves them frozen."""
    block = ResidualBlock(in_dim=8, out_dim=8, batch_norm=True, bn_momentum=0.9, bn_epsilon=1e-5)
    _ = block(tf.zeros([2, 8]))  # build layer

    initial_mean = block.bn.moving_mean.numpy().copy()
    initial_var = block.bn.moving_variance.numpy().copy()
    assert np.allclose(initial_mean, 0.0)
    assert np.allclose(initial_var, 1.0)

    # Pass non-zero input with training=False -> moving stats must not change
    x_eval = tf.constant(np.ones((16, 8), dtype=np.float32) * 5.0)
    _ = block(x_eval, training=False)
    assert np.allclose(block.bn.moving_mean.numpy(), initial_mean)
    assert np.allclose(block.bn.moving_variance.numpy(), initial_var)

    # Pass input with training=True -> moving stats must be updated
    _ = block(x_eval, training=True)
    updated_mean = block.bn.moving_mean.numpy()
    updated_var = block.bn.moving_variance.numpy()
    assert not np.allclose(updated_mean, initial_mean)
    assert not np.allclose(updated_var, initial_var)


def test_batchnorm_eval_mode_is_row_independent_and_deterministic():
    """Verify training=False output for a flow is deterministic and independent of batch context."""
    block = ResidualBlock(in_dim=4, out_dim=4, batch_norm=True, bn_momentum=0.9, bn_epsilon=1e-5)
    # Train for a few steps to get non-trivial moving stats
    rng = np.random.default_rng(42)
    for _ in range(5):
        x_tr = tf.convert_to_tensor(rng.standard_normal((16, 4), dtype=np.float32))
        _ = block(x_tr, training=True)

    flow_1 = tf.constant([[1.0, 2.0, 3.0, 4.0]], dtype=tf.float32)
    flow_2 = tf.constant([[5.0, 6.0, 7.0, 8.0]], dtype=tf.float32)
    batch = tf.concat([flow_1, flow_2], axis=0)

    # Output of flow_1 evaluated alone
    out_single = block(flow_1, training=False).numpy()
    # Output of flow_1 evaluated inside batch
    out_batch = block(batch, training=False).numpy()

    # The first row of batch output must match out_single exactly
    np.testing.assert_allclose(out_batch[0:1], out_single, rtol=1e-6, atol=1e-6)


def test_batchnorm_weights_persistence_roundtrip():
    """Verify moving mean and variance are saved in weights and reload restores identical eval."""
    cfg = Config(
        model_kind="closr",
        batch_norm=True,
        bn_momentum=0.9,
        bn_epsilon=1e-5,
        d_in=8,
        hidden=[16, 16],
        d_out=8,
        n_classes=3,
    )
    m1 = get_model(cfg)
    # Run training steps to populate moving statistics
    rng = np.random.default_rng(123)
    for _ in range(3):
        x = tf.convert_to_tensor(rng.standard_normal((8, 8), dtype=np.float32))
        _ = m1(x, training=True)

    # Save weights (including non-trainable moving mean and variance)
    weights = m1.get_weights()

    # Build fresh second model instance
    m2 = get_model(cfg)
    _ = m2(tf.zeros([1, 8]), training=False)
    m2.set_weights(weights)

    # Evaluate both on a test flow in inference mode
    test_x = tf.convert_to_tensor(rng.standard_normal((4, 8), dtype=np.float32))
    out1 = m1(test_x, training=False).numpy()
    out2 = m2(test_x, training=False).numpy()

    np.testing.assert_allclose(out1, out2, rtol=1e-7, atol=1e-7)
