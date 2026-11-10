"""PR1-T09 tests for deterministic, zero-day-preserving t-SNE figures."""

import json

import numpy as np
import pytest

from nids.config import Config
from nids.infer import stratified_tsne_indices
from nids.viz import _prepare_tsne_input, create_tsne_figure


def test_tsne_subsample_keeps_all_zero_day_and_stratifies_classes(tiny_synthetic_df):
    """Preserve all zero-day rows while sampling every known class within the cap."""
    known = ["benign", "bot", "ddos", "portscan"]
    label_ids = {name: index for index, name in enumerate(known)}
    labels = tiny_synthetic_df["label"].map(label_ids).fillna(-1).to_numpy(dtype=np.int64)

    indices = stratified_tsne_indices(labels, max_points=60, seed=7)

    assert len(indices) == 60
    assert np.all(np.isin(np.flatnonzero(labels < 0), indices))
    assert set(labels[indices]) == {-1, 0, 1, 2, 3}
    np.testing.assert_array_equal(indices, stratified_tsne_indices(labels, 60, 7))
    assert not np.array_equal(indices, stratified_tsne_indices(labels, 60, 8))


def _figure_inputs(seed: int, output_dir):
    """Build small deterministic known and zero-day embeddings and a fast t-SNE config."""
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(4), 12).tolist() + [-1, -1, -1]
    embeddings = rng.normal(size=(len(labels), 8)).astype(np.float32)
    embeddings[-3:] += 3.0
    cfg = Config(
        model_kind="clad",
        d_out=8,
        seed=seed,
        results_dir=str(output_dir),
        tsne_max_points=30,
        tsne_perplexity=5.0,
        tsne_max_iter=250,
        tsne_tiny_group_threshold=3,
    )
    return embeddings, np.asarray(labels, dtype=np.int64), cfg


@pytest.mark.parametrize("kind,shape", [("clad", (51, 8)), ("closr", (51, 4, 8))])
def test_tsne_saves_figure_and_metadata_for_both_models(tmp_path, kind, shape):
    """Write non-empty PNG/JSON artifacts with counts and dataset metadata."""
    embeddings, labels, cfg = _figure_inputs(1, tmp_path)
    vectors = embeddings if kind == "clad" else np.repeat(embeddings[:, None, :], shape[1], axis=1)

    _, figure, sidecar = create_tsne_figure(
        vectors, labels, cfg, kind, "synthetic", True, "abc123", {"sklearn": "test"}
    )

    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert figure.is_file() and figure.stat().st_size > 0
    assert sidecar.is_file() and sidecar.stat().st_size > 0
    assert metadata["n_points_per_group"] == {"benign": 7, "malicious": 20, "zero_day": 3}
    assert metadata["dataset"] == "synthetic"
    assert metadata["reduced_run"] is True
    assert metadata["zero_day_tiny"] is True


def test_tsne_coordinates_and_png_are_seed_deterministic(tmp_path):
    """Repeat runs with one seed byte-match and a different seed changes coordinates."""
    embeddings, labels, cfg = _figure_inputs(1, tmp_path)
    coords_a, figure, _ = create_tsne_figure(
        embeddings, labels, cfg, "clad", "synthetic", False, "abc123", {}
    )
    png_a = figure.read_bytes()
    coords_b, _, _ = create_tsne_figure(
        embeddings, labels, cfg, "clad", "synthetic", False, "abc123", {}
    )
    assert png_a == figure.read_bytes()
    np.testing.assert_array_equal(coords_a, coords_b)

    cfg.seed = 2
    coords_c, _, _ = create_tsne_figure(
        embeddings, labels, cfg, "clad", "synthetic", False, "abc123", {}
    )
    assert not np.array_equal(coords_a, coords_c)


@pytest.mark.parametrize(
    "embeddings,labels,message",
    [
        (None, np.array([0, -1]), "missing"),
        (np.array([[0.0, np.nan], [1.0, 0.0]]), np.array([0, -1]), "finite"),
        (np.ones((2, 2)), np.array([0, 1]), "no zero-day"),
    ],
)
def test_prepare_tsne_rejects_invalid_input(tmp_path, embeddings, labels, message):
    """Fail clearly for absent embeddings, non-finite vectors, or missing zero-day data."""
    cfg = Config(results_dir=str(tmp_path), tsne_perplexity=0.1, tsne_max_points=10)
    with pytest.raises(ValueError, match=message):
        _prepare_tsne_input(embeddings, labels, cfg, "clad")
