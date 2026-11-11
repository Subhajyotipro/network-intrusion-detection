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
"""CLI entry point for training, evaluation, calibration, benchmarking, and demo."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from nids.baselines import run_baseline
from nids.checkpoint import save_checkpoint
from nids.compare import build_comparison_tables
from nids.config import Config, get_run_metadata, load_config, setup_logging
from nids.data import generate_synthetic_data, load_data
from nids.eval_pipeline import run_eval_pipeline
from nids.infer import get_centroids, get_clad_centroid
from nids.train import train
from nids.viz import create_tsne_figure

logger = logging.getLogger("nids.cli")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add --config and --set arguments common to all subcommands."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config value in key=value format (repeatable).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build root ArgumentParser with all task subcommands."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="CLOSR -> Siamese cascade NIDS command-line interface.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    for cmd in [
        "train",
        "eval",
        "tsne",
        "calibrate",
        "cascade",
        "bench",
        "demo",
        "baseline",
        "compare",
    ]:
        sub = subparsers.add_parser(cmd, help=f"Execute {cmd} pipeline.")
        add_common_args(sub)
        if cmd == "train":
            sub.add_argument(
                "--resume",
                type=str,
                default=None,
                help="Path to saved model checkpoint to resume training from.",
            )
        if cmd == "eval":
            sub.add_argument(
                "--checkpoint",
                type=str,
                default=None,
                help="Path to saved model checkpoint weights.",
            )
        if cmd == "tsne":
            sub.add_argument(
                "--checkpoint",
                type=str,
                default=None,
                help="Saved model checkpoint to evaluate if matching arrays are absent.",
            )
        if cmd == "baseline":
            sub.add_argument(
                "--model",
                type=str,
                required=True,
                choices=["mlp", "autoencoder", "iforest"],
                help="Baseline model architecture to run.",
            )
        if cmd in ("baseline", "compare"):
            sub.add_argument(
                "--output-dir",
                type=str,
                default=None,
                help="Directory to save baseline outputs or comparison tables.",
            )

    return parser


def _load_real(cfg: Config) -> Any:
    """Load real dataset from configured data_path."""
    return load_data(
        cfg.data_path,
        val_ratio=cfg.val_ratio,
        sample_thres=cfg.sample_thres,
        split_seed=cfg.seed,
    )


def _load_train_dataset(cfg: Config) -> tuple[Any, str]:
    """Load real or synthetic dataset based on cfg.data_source."""
    if cfg.data_source == "synthetic":
        logger.info("Dataset mode: synthetic (forced via data_source='synthetic')")
        return generate_synthetic_data(cfg), "synthetic"

    data_path = Path(cfg.data_path)
    if cfg.data_source == "real":
        if not data_path.is_file():
            raise FileNotFoundError(
                f"data_source='real' requested, but file not found at: {data_path.resolve()}"
            )
        logger.info("Dataset mode: real (loading from %s)", data_path)
        return _load_real(cfg), "real"

    if data_path.is_file():
        logger.info("Dataset mode: real (auto-detected at %s)", data_path)
        return _load_real(cfg), "real"

    logger.warning(
        "PROMINENT WARNING: data_source='auto' but real dataset was not found at %s. "
        "Falling back to synthetic data generator!",
        data_path,
    )
    return generate_synthetic_data(cfg), "synthetic"


def _write_train_record(
    cfg: Config, train_res: dict[str, Any], dataset_type: str, exp_dir: Path
) -> Path:
    """Write D4.7 compliant training result record JSON to disk."""
    meta = get_run_metadata(cfg)
    exp_dir.mkdir(parents=True, exist_ok=True)
    res_path = exp_dir / f"{cfg.model_kind}_seed{cfg.seed}.json"
    epochs_run = len(train_res["epoch_losses"])
    result_record = {
        "metrics": {
            "final_loss": train_res["final_loss"],
            "epoch_losses": train_res["epoch_losses"],
            "final_frac_pos": train_res["final_frac_pos"],
        },
        "config": asdict(cfg),
        "seed": cfg.seed,
        "git_commit": meta["git_commit"],
        "versions": meta["versions"],
        "device": meta["device"],
        "dataset": dataset_type,
        "epochs": epochs_run,
        "reduced_run": bool(cfg.smoke or epochs_run < 200),
    }
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=2)
    return res_path


def run_train(cfg: Config, resume_checkpoint: str | None = None) -> dict[str, str]:
    """Execute training pipeline on real or synthetic data and write D4.7 results."""
    data, dataset_type = _load_train_dataset(cfg)
    train_res = train(cfg, kind=cfg.model_kind, data=data, resume_checkpoint=resume_checkpoint)
    centroids = (
        get_clad_centroid(
            train_res["model"], data.x_train, data.y_train, batch_size=cfg.eval_batch_size
        ).tolist()
        if cfg.model_kind == "clad"
        else get_centroids(
            train_res["model"],
            data.x_train,
            data.y_train,
            data.class_map.n_known,
            batch_size=cfg.eval_batch_size,
        ).tolist()
    )

    ckpt_dir = Path(cfg.weights_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_base = ckpt_dir / f"{cfg.model_kind}_seed{cfg.seed}"
    saved_ckpts = save_checkpoint(
        ckpt_base,
        train_res["model"],
        train_res["optimizer"],
        train_res["step"],
        cfg,
        data.class_map,
        cfg.seed,
        epoch=train_res["epoch"],
        extra={
            "centroids": centroids,
            "epoch_losses": train_res["epoch_losses"],
            "total_steps": train_res["total_steps"],
            "total_epochs": cfg.epochs,
        },
    )

    res_path = _write_train_record(cfg, train_res, dataset_type, Path(cfg.results_dir) / "train")
    logger.info("Training complete. Results written to %s", res_path)
    return {"results_json": str(res_path), "checkpoint": saved_ckpts["weights"]}


def run_tsne(cfg: Config, checkpoint: str | None = None) -> dict[str, str]:
    """Generate a t-SNE figure from saved eval arrays or an existing checkpoint."""
    arrays_path = Path(cfg.results_dir) / "eval" / f"{cfg.model_kind}_seed{cfg.seed}_arrays.npz"
    record_path = arrays_path.with_name(f"{cfg.model_kind}_seed{cfg.seed}_eval.json")
    if not arrays_path.is_file():
        data, dataset_type = _load_train_dataset(cfg)
        eval_artifacts = run_eval_pipeline(cfg, data, dataset_type, checkpoint_override=checkpoint)
        arrays_path = Path(eval_artifacts["arrays_npz"])
        record_path = Path(eval_artifacts["eval_json"])
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if "tsne_embeddings" not in arrays or "tsne_labels" not in arrays:
            raise ValueError(f"Saved evaluation arrays lack t-SNE embeddings/labels: {arrays_path}")
        embeddings, labels = arrays["tsne_embeddings"], arrays["tsne_labels"]
    with open(record_path, encoding="utf-8") as file:
        record = json.load(file)
    metadata = get_run_metadata(cfg)
    _, figure_path, sidecar_path = create_tsne_figure(
        embeddings=embeddings,
        labels=labels,
        cfg=cfg,
        model_kind=cfg.model_kind,
        dataset=record["dataset"],
        reduced_run=record["reduced_run"],
        git_commit=metadata["git_commit"],
        versions=metadata["versions"],
    )
    return {"figure": str(figure_path), "metadata": str(sidecar_path)}


def main(args_list: list[str] | None = None) -> int:
    """Main entrypoint parsing subcommands and executing dispatch."""
    parser = build_parser()
    args = parser.parse_args(args_list)

    setup_logging()
    cfg = load_config(path=args.config, overrides=args.overrides)

    log_model = (
        getattr(args, "model", cfg.model_kind)
        if args.subcommand == "baseline"
        else ("all" if args.subcommand == "compare" else cfg.model_kind)
    )
    logger.info("Subcommand: %s, model: %s, seed: %d", args.subcommand, log_model, cfg.seed)
    if args.subcommand == "train":
        artifacts = run_train(cfg, resume_checkpoint=getattr(args, "resume", None))
        logger.info("Training completed successfully: %s", artifacts)
        return 0
    if args.subcommand == "eval":
        data, dataset_type = _load_train_dataset(cfg)
        ckpt_arg = getattr(args, "checkpoint", None)
        artifacts = run_eval_pipeline(cfg, data, dataset_type, checkpoint_override=ckpt_arg)
        logger.info("Evaluation completed successfully: %s", artifacts)
        return 0
    if args.subcommand == "tsne":
        artifacts = run_tsne(cfg, checkpoint=getattr(args, "checkpoint", None))
        logger.info("t-SNE artifacts generated: %s", artifacts)
        return 0
    if args.subcommand == "baseline":
        data, dataset_type = _load_train_dataset(cfg)
        out_dir = getattr(args, "output_dir", None)
        artifacts = run_baseline(args.model, cfg, data, output_dir=out_dir)
        logger.info("Baseline run completed successfully: %s", artifacts)
        return 0
    if args.subcommand == "compare":
        out_dir = getattr(args, "output_dir", None)
        is_syn = cfg.data_source != "real"
        tables = build_comparison_tables(
            results_dir=cfg.results_dir, output_dir=out_dir, synthetic_mode=is_syn
        )
        logger.info("Comparison tables generated successfully: %s", tables)
        return 0

    logger.info(
        "Subcommand '%s' resolved config successfully. Execution is not implemented yet (PR2+).",
        args.subcommand,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
