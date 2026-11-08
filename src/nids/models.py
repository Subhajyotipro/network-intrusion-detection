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
"""Neural network architectures for CLAD and CLOSR representation learning.

Adapted from upstream CLOSR (Apache-2.0, DOI 10.1109/TNSM.2026.3652529)
by Jack Wilkie et al. Re-implemented in TensorFlow / Keras 3 with configurable
input dimensions and weight initializations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf

from nids.config import Config


def _get_initializers(
    in_dim: int, init: str
) -> tuple[tf.keras.initializers.Initializer, tf.keras.initializers.Initializer]:
    """Return kernel and bias initializers for a layer with the given fan_in."""
    if init == "torch_default":
        b = float(1.0 / np.sqrt(in_dim))
        kernel_init = tf.keras.initializers.VarianceScaling(
            scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
        )
        bias_init = tf.keras.initializers.RandomUniform(minval=-b, maxval=b)
        return kernel_init, bias_init
    if init == "keras_default":
        return tf.keras.initializers.GlorotUniform(), tf.keras.initializers.Zeros()
    raise ValueError(f"Unknown init scheme '{init}'. Expected 'torch_default' or 'keras_default'.")


class ResidualBlock(tf.keras.layers.Layer):
    """Dense block with linear transformation, ReLU activation, dropout, and residual skip."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        residual: bool = False,
        init: str = "torch_default",
        apply_dropout: bool = True,
        apply_activation: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = residual
        k_init, b_init = _get_initializers(in_dim, init)
        self.dense = tf.keras.layers.Dense(
            out_dim, kernel_initializer=k_init, bias_initializer=b_init
        )
        self.act = tf.keras.layers.Activation("relu") if apply_activation else None
        self.drop = tf.keras.layers.Dropout(dropout) if (apply_dropout and dropout > 0.0) else None

        if residual:
            if in_dim == out_dim:
                self.resize = tf.keras.layers.Identity()
            else:
                r_k, r_b = _get_initializers(in_dim, init)
                self.resize = tf.keras.layers.Dense(
                    out_dim, kernel_initializer=r_k, bias_initializer=r_b
                )
        else:
            self.resize = None

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through linear layer, activation, dropout, and residual sum."""
        x = self.dense(inputs)
        if self.act is not None:
            x = self.act(x)
        if self.drop is not None:
            x = self.drop(x, training=training)
        if self.residual and self.resize is not None:
            x = x + self.resize(inputs)
        return x


def mlp(
    d_in: int,
    hidden: list[int],
    d_out: Optional[int] = None,
    dropout: float = 0.0,
    residual: bool = False,
    init: str = "torch_default",
    final_activation: bool = False,
) -> tf.keras.Sequential:
    """Build a Sequential MLP backbone adhering faithfully to upstream CLOSR block structure."""
    dims = [d_in] + list(hidden)
    if d_out is not None:
        dims.append(d_out)
    layers: list[tf.keras.layers.Layer] = []
    for i in range(len(dims) - 1):
        is_last = i == len(dims) - 2
        apply_act = True if not is_last else final_activation
        apply_drop = not is_last
        layers.append(
            ResidualBlock(
                in_dim=dims[i],
                out_dim=dims[i + 1],
                dropout=dropout,
                residual=residual,
                init=init,
                apply_dropout=apply_drop,
                apply_activation=apply_act,
            )
        )
    return tf.keras.Sequential(layers)


class ContrastiveMLP(tf.keras.Model):
    """CLAD model mapping flow features to an L2-normalized hyperspherical embedding vector."""

    def __init__(
        self,
        d_in: int,
        hidden: list[int],
        d_out: int = 8,
        dropout: float = 0.0,
        residual: bool = False,
        init: str = "torch_default",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_in = d_in
        self.d_out = d_out
        self.hidden = hidden
        self.dropout_rate = dropout
        self.residual = residual
        self.init = init

        self.backbone = mlp(
            d_in=d_in,
            hidden=hidden,
            dropout=dropout,
            residual=residual,
            init=init,
            final_activation=False,
        )
        head_in = hidden[-1] if hidden else d_in
        k_init, b_init = _get_initializers(head_in, init)
        self.proj = tf.keras.layers.Dense(d_out, kernel_initializer=k_init, bias_initializer=b_init)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Compute L2-normalized embedding representation of shape [batch, d_out]."""
        x = self.backbone(inputs, training=training)
        x = self.proj(x)
        return tf.linalg.l2_normalize(x, axis=-1)


class CLOSRMLP(tf.keras.Model):
    """CLOSR multiclass model producing per-class hyperspherical embedding representations."""

    def __init__(
        self,
        d_in: int,
        hidden: list[int],
        d_out: int = 64,
        n_classes: int = 12,
        dropout: float = 0.1,
        residual: bool = False,
        init: str = "torch_default",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_in = d_in
        self.d_out = d_out
        self.n_classes = n_classes
        self.hidden = hidden
        self.dropout_rate = dropout
        self.residual = residual
        self.init = init

        self.backbone = mlp(
            d_in=d_in,
            hidden=hidden,
            dropout=dropout,
            residual=residual,
            init=init,
            final_activation=True,
        )
        head_in = hidden[-1] if hidden else d_in
        k_init, b_init = _get_initializers(head_in, init)
        self.proj_head = tf.keras.layers.Dense(
            d_out * n_classes, kernel_initializer=k_init, bias_initializer=b_init
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Compute per-class normalized embeddings of shape [batch, n_classes, d_out]."""
        x = self.backbone(inputs, training=training)
        x = self.proj_head(x)
        batch_size = tf.shape(inputs)[0]
        x = tf.reshape(x, [batch_size, self.n_classes, self.d_out])
        return tf.linalg.l2_normalize(x, axis=-1)


def get_model(cfg: Config, d_in: Optional[int] = None) -> tf.keras.Model:
    """Instantiate and return the configured model kind with exact architecture specifications."""
    input_dim = d_in if d_in is not None else cfg.d_in
    if input_dim is None:
        raise ValueError("d_in must be provided directly or present in cfg.")

    kind = cfg.model_kind.lower()
    if kind == "clad":
        return ContrastiveMLP(
            d_in=input_dim,
            hidden=cfg.hidden,
            d_out=cfg.d_out,
            dropout=cfg.dropout,
            residual=cfg.residual,
            init=cfg.init,
        )
    if kind == "closr":
        return CLOSRMLP(
            d_in=input_dim,
            hidden=cfg.hidden,
            d_out=cfg.d_out,
            n_classes=cfg.n_classes,
            dropout=cfg.dropout,
            residual=cfg.residual,
            init=cfg.init,
        )
    raise ValueError(f"Unknown model_kind: '{cfg.model_kind}'. Expected 'clad' or 'closr'.")
