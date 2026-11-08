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
"""Unit tests verifying that initializers reproduce PyTorch Linear distribution bounds."""

import numpy as np
import pytest
import tensorflow as tf

from nids.models import _get_initializers


@pytest.mark.parametrize("fan_in", [64, 256, 1024])
def test_torch_default_initialization_bounds_and_variance(fan_in: int):
    """Verify torch_default init matches PyTorch Linear Kaiming uniform bound 1/sqrt(fan_in)."""
    k_init, b_init = _get_initializers(fan_in, init="torch_default")
    bound = 1.0 / np.sqrt(fan_in)
    expected_std = 1.0 / np.sqrt(3.0 * fan_in)

    # Large sample draw for empirical checks
    sample_size = 100_000
    w = k_init(shape=[fan_in, sample_size // fan_in]).numpy().flatten()
    b = b_init(shape=[sample_size]).numpy()

    # 1. Kernel bounds: max |w| <= 1/sqrt(fan_in)
    assert np.max(np.abs(w)) <= bound + 1e-6
    # Empirical std within 3% of theoretical uniform std
    empirical_w_std = float(np.std(w))
    rel_error_w = abs(empirical_w_std - expected_std) / expected_std
    msg_w = f"Kernel std {empirical_w_std} deviated by {rel_error_w:.3%} (expected {expected_std})"
    assert rel_error_w < 0.03, msg_w

    # 2. Bias bounds: max |b| <= 1/sqrt(fan_in)
    assert np.max(np.abs(b)) <= bound + 1e-6
    empirical_b_std = float(np.std(b))
    rel_error_b = abs(empirical_b_std - expected_std) / expected_std
    msg_b = f"Bias std {empirical_b_std} deviated by {rel_error_b:.3%} (expected {expected_std})"
    assert rel_error_b < 0.03, msg_b


def test_keras_default_and_invalid_init():
    """Verify keras_default uses GlorotUniform and Zeros, and unknown init raises clear error."""
    k_init, b_init = _get_initializers(64, init="keras_default")
    assert isinstance(k_init, tf.keras.initializers.GlorotUniform)
    assert isinstance(b_init, tf.keras.initializers.Zeros)

    zero_biases = b_init(shape=[64]).numpy()
    assert np.all(zero_biases == 0.0)

    with pytest.raises(ValueError, match="Unknown init scheme"):
        _get_initializers(64, init="invalid_init")
