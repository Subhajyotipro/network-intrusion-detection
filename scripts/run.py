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

from nids.checkpoint import save_checkpoint
from nids.config import Config, get_run_metadata, load_config, setup_logging
from nids.data import generate_synthetic_data, load_data
from nids.train import train

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

    for cmd in ["train", "eval", "calibrate", "cascade", "bench", "demo"]:
        sub = subparsers.add_parser(cmd, help=f"Execute {cmd} pipeline.")
        add_common_args(sub)

    return parser


def _load_train_dataset(cfg: Config) -> tuple[Any, str]:
    """Load real dataset if present, otherwise fall back to synthetic."""
    data_path = Path(cfg.data_path)
    if data_path.is_file():
        logger.info("Loading real dataset from %s", data_path)
        data = load_data(
            cfg.data_path,
            val_ratio=cfg.val_ratio,
            sample_thres=cfg.sample_thres,
            split_seed=cfg.seed,
        )
        return data, "real"
    logger.info("Dataset %s not found; falling back to synthetic generator", data_path)
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


def run_train(cfg: Config) -> dict[str, str]:
    """Execute training pipeline on real or synthetic data and write D4.7 results."""
    data, dataset_type = _load_train_dataset(cfg)
    train_res = train(cfg, kind=cfg.model_kind, data=data)

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
        extra={"total_steps": train_res["total_steps"], "total_epochs": cfg.epochs},
    )

    res_path = _write_train_record(cfg, train_res, dataset_type, Path(cfg.results_dir) / "train")
    logger.info("Training complete. Results written to %s", res_path)
    return {"results_json": str(res_path), "checkpoint": saved_ckpts["weights"]}


def main(args_list: list[str] | None = None) -> int:
    """Main entrypoint parsing subcommands and executing dispatch."""
    parser = build_parser()
    args = parser.parse_args(args_list)

    setup_logging()
    cfg = load_config(path=args.config, overrides=args.overrides)

    logger.info("Subcommand: %s, model: %s, seed: %d", args.subcommand, cfg.model_kind, cfg.seed)
    if args.subcommand == "train":
        artifacts = run_train(cfg)
        print(f"Training completed successfully: {artifacts}")
        return 0

    print(
        f"Subcommand '{args.subcommand}' resolved config successfully. "
        f"Execution is not implemented yet (PR1-T06+)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
