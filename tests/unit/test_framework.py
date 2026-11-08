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
"""Tests for TensorFlow framework policy, torch-free runtime enforcement, and determinism."""

import ast
import importlib
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from nids.config import get_default_config, get_run_metadata, resolve_device, set_seed


def test_import_nids_does_not_import_torch():
    """Verify that importing all nids package modules does not load torch into sys.modules."""
    modules = [
        "nids",
        "nids.config",
        "nids.data",
        "nids.models",
        "nids.losses",
        "nids.train",
        "nids.infer",
        "nids.metrics",
        "nids.calibrate",
        "nids.cascade",
        "nids.bench",
        "nids.viz",
    ]
    for mod in modules:
        importlib.import_module(mod)

    assert "torch" not in sys.modules, f"torch found in sys.modules: {sys.modules.get('torch')}"


def test_static_scan_src_and_scripts_for_torch():
    """Verify via AST parsing that no Python file in repo imports torch (except third_party)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    excluded_dirs = {repo_root / "third_party", repo_root / ".venv", repo_root / ".git"}

    violations = []
    for py_file in repo_root.rglob("*.py"):
        if any(exc in py_file.parents for exc in excluded_dirs):
            continue
        with open(py_file, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "torch" or alias.name.startswith("torch."):
                        violations.append((str(py_file), node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == "torch" or node.module.startswith("torch.")):
                    violations.append((str(py_file), node.lineno, node.module))

    assert not violations, f"Torch import violations detected in repository: {violations}"


def test_set_seed_reproducibility():
    """Verify set_seed makes python random, numpy, and tensorflow uniform draws reproducible."""
    set_seed(1234, deterministic=True)
    r1 = random.random()
    n1 = np.random.rand(5)
    t1 = tf.random.uniform([5]).numpy()

    set_seed(1234, deterministic=True)
    r2 = random.random()
    n2 = np.random.rand(5)
    t2 = tf.random.uniform([5]).numpy()

    assert r1 == r2
    assert np.allclose(n1, n2)
    assert np.allclose(t1, t2)


def test_resolve_device_force_cpu():
    """Verify resolve_device('cpu') returns 'cpu'."""
    assert resolve_device("cpu") == "cpu"


def test_get_run_metadata_tensorflow_and_no_torch():
    """Verify runtime metadata reports tensorflow and keras, and contains no torch entry."""
    cfg = get_default_config("closr")
    meta = get_run_metadata(cfg)

    assert "versions" in meta
    assert "tensorflow" in meta["versions"]
    assert "keras" in meta["versions"]
    assert "torch" not in meta["versions"]
    assert meta["device"] in {"cpu", "gpu"}
    assert "timestamp" in meta


def test_tf_tiny_forward_pass_on_cpu():
    """Verify TensorFlow executes a minimal dense forward pass on CPU without errors."""
    with tf.device("/CPU:0"):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(4),
            ]
        )
        inputs = tf.ones((2, 10))
        outputs = model(inputs, training=False)
        assert outputs.shape == (2, 4)
