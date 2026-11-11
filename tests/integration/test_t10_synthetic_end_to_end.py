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
"""PR1-T10 tiny synthetic train, checkpoint, evaluation, and figure integration."""

import json
from pathlib import Path

import numpy as np
import pytest

from nids.checkpoint import load_checkpoint, save_checkpoint
from nids.config import load_config
from nids.data import generate_synthetic_data
from nids.eval_pipeline import run_eval_pipeline
from nids.infer import get_centroids, get_clad_centroid
from nids.models import get_model
from nids.train import train
from nids.viz import create_tsne_figure


@pytest.mark.timeout(30)
@pytest.mark.parametrize("kind", ["clad", "closr"])
def test_pr1_t10_synthetic_train_reload_eval_tsne(kind: str, tmp_path: Path) -> None:
    """Run the full small synthetic workflow and persist all expected artifacts."""
    config_path = "configs/clad.yaml" if kind == "clad" else "configs/default.yaml"
    weights_dir, results_dir = tmp_path / "weights", tmp_path / "results"
    cfg = load_config(
        config_path,
        overrides=[
            "d_in=8",
            "hidden=[16]",
            "d_out=4",
            "dropout=0.0",
            "smoke=true",
            "max_batches=2",
            "batch_size=16",
            "tsne_max_points=24",
            "tsne_perplexity=5.0",
            "tsne_max_iter=250",
            "tsne_tiny_group_threshold=3",
            f"weights_dir={weights_dir}",
            f"results_dir={results_dir}",
            "seed=17",
        ],
    )
    data = generate_synthetic_data(cfg, num_samples=30)
    trained = train(cfg, kind=kind, data=data)

    centroids = (
        get_clad_centroid(trained["model"], data.x_train, data.y_train)
        if kind == "clad"
        else get_centroids(trained["model"], data.x_train, data.y_train, data.class_map.n_known)
    )
    checkpoint = weights_dir / f"{kind}_seed{cfg.seed}"
    save_checkpoint(
        checkpoint,
        trained["model"],
        trained["optimizer"],
        trained["step"],
        cfg,
        data.class_map,
        cfg.seed,
        epoch=trained["epoch"],
        extra={"centroids": centroids.tolist()},
    )

    reloaded = get_model(cfg, d_in=data.d_in)
    metadata = load_checkpoint(checkpoint, reloaded)
    np.testing.assert_allclose(
        trained["model"](data.x_test[:2], training=False).numpy(),
        reloaded(data.x_test[:2], training=False).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
    assert metadata["config"]["model_kind"] == kind
    assert metadata["class_map"]["names"] == data.class_map.names
    assert metadata["seed"] == cfg.seed
    assert "centroids" in metadata["extra"]

    paths = run_eval_pipeline(cfg, data, "synthetic")
    assert Path(paths["eval_json"]).is_file()
    assert Path(paths["per_class_csv"]).is_file()
    arrays = np.load(paths["arrays_npz"])
    record = json.loads(Path(paths["eval_json"]).read_text(encoding="utf-8"))
    assert record["dataset"] == "synthetic"
    assert len(arrays["tsne_labels"]) == len(arrays["tsne_embeddings"])
    assert np.count_nonzero(arrays["tsne_labels"] < 0) == len(data.y_zd)

    _, figure, sidecar = create_tsne_figure(
        arrays["tsne_embeddings"],
        arrays["tsne_labels"],
        cfg,
        kind,
        "synthetic",
        True,
        record["git_commit"],
        record["versions"],
    )
    assert figure.is_file() and figure.stat().st_size > 0
    assert json.loads(sidecar.read_text(encoding="utf-8"))["n_points_per_group"]["zero_day"] == len(
        data.y_zd
    )
