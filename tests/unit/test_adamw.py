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
"""Unit tests comparing Keras AdamW against independent PyTorch AdamW NumPy reference."""

import numpy as np
import tensorflow as tf

from tests.unit.np_reference import np_adamw_step


def test_adamw_single_step_matches_numpy_reference():
    """Verify single Keras AdamW step matches PyTorch AdamW within float32 / DV17 tolerance.

    Tolerance justification (DV17):
    - Float32 precision has machine epsilon ~1.19e-7.
    - PyTorch divides by (sqrt(v_hat) + eps), whereas Keras divides by (sqrt(v) + eps) with
      bias-correction in the numerator. This induces an epsilon placement discrepancy on
      the order of eps * (1 - sqrt(1 - beta_2)) ~ 1e-8 * 0.968 ~ 9.7e-9.
    - Together, max discrepancy is bounded within 5e-6.
    """
    lr = 0.01
    wd = 0.04
    eps = 1e-8
    w_init = np.array([[1.0, -0.5, 2.0, -1.2]], dtype=np.float32)

    w_tf = tf.Variable(w_init)
    opt = tf.keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=wd,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=eps,
    )

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(0.5 * tf.square(w_tf - 1.5))
    grads = tape.gradient(loss, [w_tf])
    opt.apply_gradients(zip(grads, [w_tf]))

    # Reference step in float64
    g_np = w_init.astype(np.float64) - 1.5
    w_expected, _, _ = np_adamw_step(
        param=w_init.astype(np.float64),
        grad=g_np,
        m=np.zeros_like(w_init, dtype=np.float64),
        v=np.zeros_like(w_init, dtype=np.float64),
        step=1,
        lr=lr,
        weight_decay=wd,
        beta1=0.9,
        beta2=0.999,
        eps=eps,
    )

    np.testing.assert_allclose(w_tf.numpy(), w_expected, rtol=1e-5, atol=2e-6)


def test_adamw_multi_step_trajectory_matches_reference():
    """Verify multiple AdamW optimization steps match reference trajectory."""
    lr = 0.005
    wd = 0.02
    eps = 1e-8
    w_init = np.array([[0.8, -1.0]], dtype=np.float32)

    w_tf = tf.Variable(w_init)
    opt = tf.keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=wd,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=eps,
    )

    w_np = w_init.astype(np.float64)
    m_np = np.zeros_like(w_np)
    v_np = np.zeros_like(w_np)

    for step in range(1, 10):
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(0.5 * tf.square(w_tf - 2.0))
        grads = tape.gradient(loss, [w_tf])
        opt.apply_gradients(zip(grads, [w_tf]))

        g_np = w_np - 2.0
        w_np, m_np, v_np = np_adamw_step(
            param=w_np,
            grad=g_np,
            m=m_np,
            v=v_np,
            step=step,
            lr=lr,
            weight_decay=wd,
            beta1=0.9,
            beta2=0.999,
            eps=eps,
        )
        np.testing.assert_allclose(w_tf.numpy(), w_np, rtol=1e-5, atol=5e-6)
