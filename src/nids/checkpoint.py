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
"""Model and optimizer checkpointing utilities with JSON sidecar metadata."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tensorflow as tf


def _resolve_base_path(path: str | Path) -> Path:
    """Resolve file path by stripping archive or weight extensions for unified naming."""
    p = Path(path)
    name = p.name
    for ext in (".pt.tar", ".tar", ".weights.h5", ".h5"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return p.with_name(name)


def _to_serializable(obj: Any) -> Any:
    """Convert dataclasses or dict-like objects into JSON-serializable dictionaries."""
    if obj is None:
        return None
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def save_checkpoint(
    path: str | Path,
    model: tf.keras.Model,
    optimizer: Optional[tf.keras.optimizers.Optimizer],
    step: int,
    cfg: Any,
    class_map: Any,
    seed: int,
    epoch: int = 0,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """Save model weights, optimizer variables, and sidecar metadata to disk."""
    base = _resolve_base_path(path)
    base.parent.mkdir(parents=True, exist_ok=True)

    weights_path = str(base) + ".weights.h5"
    model.save_weights(weights_path)
    saved_files = {"weights": weights_path}

    if optimizer is not None:
        opt_path = str(base) + "_opt.npz"
        opt_vars = [v.numpy() for v in optimizer.variables]
        np.savez(opt_path, *opt_vars)
        saved_files["optimizer"] = opt_path

    meta_path = str(base) + "_meta.json"
    metadata = {
        "epoch": int(epoch),
        "step": int(step),
        "seed": int(seed),
        "config": _to_serializable(cfg),
        "class_map": _to_serializable(class_map),
        "extra": extra or {},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    saved_files["meta"] = meta_path

    return saved_files


def load_checkpoint(
    path: str | Path,
    model: tf.keras.Model,
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
) -> dict[str, Any]:
    """Load model weights, optimizer variables, and sidecar metadata from disk."""
    base = _resolve_base_path(path)
    meta_path = str(base) + "_meta.json"
    metadata: dict[str, Any] = {}
    if Path(meta_path).is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if not getattr(model, "built", False):
        d_in = getattr(model, "d_in", metadata.get("config", {}).get("d_in"))
        if d_in is not None:
            model(tf.zeros((1, int(d_in))))

    weights_path = str(base) + ".weights.h5"
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"Checkpoint weights not found: {weights_path}")
    model.load_weights(weights_path)

    if optimizer is not None:
        opt_path = str(base) + "_opt.npz"
        if Path(opt_path).is_file():
            if not getattr(optimizer, "built", False):
                optimizer.build(model.trainable_variables)
            data = np.load(opt_path)
            for v, key in zip(optimizer.variables, data.files):
                v.assign(data[key])

    return metadata
