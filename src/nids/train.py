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
"""Unified training engine for CLAD and CLOSR in TensorFlow / Keras."""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Optional

import numpy as np
import tensorflow as tf

from nids.checkpoint import load_checkpoint
from nids.config import Config
from nids.data import DatasetSplits, class_balanced_sampler
from nids.losses import clad_loss_and_frac, closr_loss_and_frac
from nids.models import get_model

logger = logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores running average, sum, count, and current value."""

    def __init__(self) -> None:
        self.reset()

    def update(self, val: float, n: int = 1) -> None:
        """Update meter with a new scalar observation."""
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def reset(self) -> None:
        """Reset running metrics."""
        self.val, self.avg, self.sum, self.count = 0.0, 0.0, 0.0, 0


def make_schedule(cfg: Config, total_steps: int) -> Callable[[int], float]:
    """Return a pure WarmupCosineSchedule mapping step index to learning rate."""
    warmup_steps = int(cfg.warmup_frac * total_steps)

    def schedule(step: int) -> float:
        if step <= 0:
            return float(cfg.lr_init)
        if step > total_steps:
            return float(cfg.lr_final)
        if step < warmup_steps:
            progress = float(step) / float(max(1, warmup_steps))
            return float(cfg.lr_init + progress * (cfg.lr_peak - cfg.lr_init))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        val = cfg.lr_final + (cfg.lr_peak - cfg.lr_final) * 0.5 * (
            1.0 + math.cos(progress * math.pi)
        )
        return float(val)

    return schedule


def _generate_epoch_batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
    epoch: int,
    max_batches: Optional[int] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate reproducible class-balanced batches for a specific training epoch."""
    n_samples = len(y)
    effective_bsz = min(batch_size, n_samples)
    drop_last = n_samples >= batch_size
    epoch_seed = (seed + epoch * 10007) & 0x7FFFFFFF
    sampler = class_balanced_sampler(
        x,
        y,
        batch_size=effective_bsz,
        seed=epoch_seed,
        drop_last=drop_last,
        max_batches=max_batches,
    )
    return list(sampler)


def _get_loss_fn(kind: str, cfg: Config) -> Callable:
    """Instantiate unified loss callable returning (loss, frac_pos)."""
    if kind == "clad":
        return lambda z, y: clad_loss_and_frac(z, y, margin=cfg.margin, squared=cfg.squared)
    return lambda z, y: closr_loss_and_frac(z, y, margin=cfg.margin, squared=cfg.squared)


def _train_step(
    net: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: Callable[[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    x_b: tf.Tensor,
    y_b: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Execute single gradient update step under GradientTape."""
    with tf.GradientTape() as tape:
        z = net(x_b, training=True)
        loss, frac_pos = loss_fn(z, y_b)
    grads = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss, frac_pos


def _run_epoch(
    net: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: Callable[[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    step_fn: Callable,
    batches: list[tuple[np.ndarray, np.ndarray]],
    schedule_fn: Callable[[int], float],
    step: int,
    total_steps: int,
    epoch: int,
    log_every: int,
) -> tuple[int, float, float]:
    """Execute all batches in a training epoch, updating meters and learning rate."""
    loss_m, frac_m = AverageMeter(), AverageMeter()
    last_f_pos = 0.0
    for x_np, y_np in batches:
        step += 1
        lr = schedule_fn(step)
        optimizer.learning_rate.assign(lr)
        loss_val, frac_pos = step_fn(net, optimizer, loss_fn, tf.constant(x_np), tf.constant(y_np))
        if not tf.math.is_finite(loss_val):
            raise ValueError(f"Non-finite loss {float(loss_val)} at step {step}")
        last_f_pos = float(frac_pos)
        loss_m.update(float(loss_val), len(y_np))
        frac_m.update(last_f_pos, len(y_np))
        if step % log_every == 0:
            logger.info(
                "Epoch %d Step %d/%d LR %.3e Loss %.4f (%.4f)",
                epoch,
                step,
                total_steps,
                lr,
                loss_m.val,
                loss_m.avg,
            )
    return step, loss_m.avg, frac_m.avg if frac_m.count > 0 else last_f_pos


def _compute_steps_and_epochs(
    cfg: Config, n_samples: int, saved_meta: dict[str, Any], max_epochs: Optional[int]
) -> tuple[int, int, int, int]:
    """Calculate start/end epochs and step count for scheduling."""
    start_step = int(saved_meta.get("step", 0))
    start_epoch = int(saved_meta.get("epoch", 0))
    epochs = 2 if cfg.smoke else cfg.epochs
    target_epochs = saved_meta.get("extra", {}).get("total_epochs", epochs)
    end_epoch = start_epoch + epochs if epochs <= start_epoch else epochs
    if max_epochs is not None:
        cap = start_epoch + max_epochs if epochs <= start_epoch else max_epochs
        end_epoch = min(end_epoch, cap)

    eff_bsz = min(cfg.batch_size, n_samples)
    drop_last = n_samples >= cfg.batch_size
    b_count = n_samples // eff_bsz if drop_last else int(np.ceil(n_samples / eff_bsz))
    max_b = (cfg.max_batches or 3) if cfg.smoke else cfg.max_batches
    if max_b is not None:
        b_count = min(b_count, max_b)
    total_steps = int(saved_meta.get("extra", {}).get("total_steps", target_epochs * b_count))
    return start_step, start_epoch, end_epoch, total_steps


def _make_optimizer(cfg: Config) -> tf.keras.optimizers.Optimizer:
    """Create configured AdamW optimizer."""
    return tf.keras.optimizers.AdamW(
        learning_rate=cfg.lr_init,
        weight_decay=cfg.weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=cfg.adam_eps,
    )


def _train_epochs(
    net: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: Callable,
    step_fn: Callable,
    data: DatasetSplits,
    cfg: Config,
    schedule_fn: Callable[[int], float],
    bounds: tuple[int, int],
    start_step: int,
    total_steps: int,
) -> tuple[int, int, list[float], float]:
    """Loop through epochs, recording losses and final positive fraction."""
    start_ep, end_ep = bounds
    step, last_ep, f_pos = start_step, start_ep, 0.0
    epoch_losses, max_b = [], (cfg.max_batches or 3) if cfg.smoke else cfg.max_batches
    for epoch in range(start_ep + 1, end_ep + 1):
        last_ep = epoch
        batches = _generate_epoch_batches(
            data.x_train, data.y_train, cfg.batch_size, cfg.seed, epoch, max_b
        )
        step, ep_loss, f_pos = _run_epoch(
            net,
            optimizer,
            loss_fn,
            step_fn,
            batches,
            schedule_fn,
            step,
            total_steps,
            epoch,
            cfg.log_every,
        )
        epoch_losses.append(ep_loss)
    return step, last_ep, epoch_losses, f_pos


def _format_train_result(
    net: tf.keras.Model,
    opt: tf.keras.optimizers.Optimizer,
    step: int,
    epoch: int,
    total_steps: int,
    losses: list[float],
    f_pos: float,
) -> dict[str, Any]:
    """Format training return dictionary."""
    return {
        "model": net,
        "optimizer": opt,
        "step": step,
        "epoch": epoch,
        "total_steps": total_steps,
        "final_loss": losses[-1] if losses else 0.0,
        "epoch_losses": losses,
        "final_frac_pos": f_pos,
    }


def train(
    cfg: Config,
    kind: str,
    data: DatasetSplits,
    model: Optional[tf.keras.Model] = None,
    resume_checkpoint: Optional[str] = None,
    max_epochs: Optional[int] = None,
) -> dict[str, Any]:
    """Train CLAD or CLOSR model end-to-end with WarmupCosineSchedule and balanced sampler."""
    if kind not in ("clad", "closr"):
        raise ValueError(f"Unsupported model kind '{kind}'. Expected 'clad' or 'closr'.")

    tf.keras.utils.set_random_seed(cfg.seed)
    loss_fn = _get_loss_fn(kind, cfg)
    step_fn = tf.function(_train_step)
    net = model or get_model(cfg, d_in=data.x_train.shape[-1])
    optimizer = _make_optimizer(cfg)
    saved_meta = load_checkpoint(resume_checkpoint, net, optimizer) if resume_checkpoint else {}
    step, start_ep, end_ep, total_steps = _compute_steps_and_epochs(
        cfg, len(data.y_train), saved_meta, max_epochs
    )
    schedule_fn = make_schedule(cfg, total_steps)
    step, last_ep, ep_losses, f_pos = _train_epochs(
        net,
        optimizer,
        loss_fn,
        step_fn,
        data,
        cfg,
        schedule_fn,
        (start_ep, end_ep),
        step,
        total_steps,
    )
    return _format_train_result(net, optimizer, step, last_ep, total_steps, ep_losses, f_pos)
