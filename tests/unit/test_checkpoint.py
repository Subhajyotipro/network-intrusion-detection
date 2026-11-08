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
"""Unit tests verifying checkpoint save, reload, sidecar metadata, and training resume."""

from pathlib import Path

import numpy as np
import tensorflow as tf

from nids.checkpoint import load_checkpoint, save_checkpoint
from nids.config import Config
from nids.data import generate_synthetic_data
from nids.models import get_model
from nids.train import train


def test_checkpoint_reload_restores_identical_outputs_and_optimizer_vars(tmp_path: Path):
    """Verify loading checkpoint restores identical eval outputs and optimizer variables."""
    cfg = Config(model_kind="closr", d_in=8, hidden=[16, 16], d_out=8, n_classes=3)
    model = get_model(cfg)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

    # Perform one step to build and initialize optimizer variables
    x = tf.constant(np.ones((4, 8), dtype=np.float32))
    with tf.GradientTape() as tape:
        y = tf.reduce_sum(model(x, training=True))
    grads = tape.gradient(y, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    ckpt_path = tmp_path / "test_ckpt"
    class_map = {"benign": 0, "attack1": 1, "attack2": 2}
    extra_data = {"centroids": [0.1, 0.2, 0.3]}

    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        step=42,
        cfg=cfg,
        class_map=class_map,
        seed=1,
        extra=extra_data,
    )

    # Reconstruct fresh model and optimizer
    model_loaded = get_model(cfg)
    _ = model_loaded(tf.zeros([1, 8]), training=False)
    opt_loaded = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

    meta = load_checkpoint(ckpt_path, model_loaded, opt_loaded)

    # 1. Assert metadata sidecar keys
    assert meta["step"] == 42
    assert meta["seed"] == 1
    assert meta["class_map"] == class_map
    assert meta["extra"] == extra_data
    assert meta["config"]["model_kind"] == "closr"

    # 2. Assert eval outputs identical
    test_x = tf.constant(np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32))
    out_orig = model(test_x, training=False).numpy()
    out_loaded = model_loaded(test_x, training=False).numpy()
    np.testing.assert_allclose(out_orig, out_loaded, rtol=1e-7, atol=1e-7)

    # 3. Assert optimizer variables identical
    for v1, v2 in zip(optimizer.variables, opt_loaded.variables):
        np.testing.assert_allclose(v1.numpy(), v2.numpy(), rtol=1e-7, atol=1e-7)


def test_training_resume_reproduces_identical_loss(tmp_path: Path):
    """Verify resume reproduces identical loss to continuous training within 1e-6 (dropout=0)."""
    cfg_base = Config(
        model_kind="closr",
        d_in=8,
        hidden=[16, 16],
        d_out=8,
        n_classes=3,
        dropout=0.0,  # Exact bitwise resume requires dropout=0 per Keras 3 Dropout RNG analysis
        batch_size=8,
        epochs=3,
        smoke=False,
        max_batches=2,
        seed=42,
    )
    data = generate_synthetic_data(cfg_base, num_samples=32)

    # 1. Continuous 3 epochs
    cfg_cont = Config(**{**cfg_base.__dict__, "epochs": 3})
    res_cont = train(cfg_cont, kind="closr", data=data)
    loss_cont_ep3 = res_cont["epoch_losses"][2]

    # 2. Train 2 epochs of the 3-epoch schedule, save checkpoint
    res_part1 = train(cfg_cont, kind="closr", data=data, max_epochs=2)
    ckpt_path = tmp_path / "resume_ckpt"
    save_checkpoint(
        path=ckpt_path,
        model=res_part1["model"],
        optimizer=res_part1["optimizer"],
        step=res_part1["step"],
        epoch=res_part1["epoch"],
        cfg=cfg_cont,
        class_map=data.class_map,
        seed=cfg_cont.seed,
        extra={"total_steps": res_cont["total_steps"], "total_epochs": 3},
    )

    # 3. Resume from checkpoint to finish the remaining epoch 3
    res_part2 = train(cfg_cont, kind="closr", data=data, resume_checkpoint=str(ckpt_path))
    loss_resumed_ep3 = res_part2["epoch_losses"][0]

    # Assert that epoch 3 loss from continuous matches epoch 3 loss from resume within 1e-6
    np.testing.assert_allclose(loss_resumed_ep3, loss_cont_ep3, rtol=1e-6, atol=1e-6)
