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
import logging
import sys

from nids.config import load_config, setup_logging

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


def main(args_list: list[str] | None = None) -> int:
    """Main entrypoint parsing subcommands and resolving Config."""
    parser = build_parser()
    args = parser.parse_args(args_list)

    setup_logging()
    cfg = load_config(path=args.config, overrides=args.overrides)

    logger.info("Resolved configuration for subcommand '%s':", args.subcommand)
    logger.info(
        "model_kind=%s, d_out=%d, hidden=%s, epochs=%d, squared=%s, residual=%s",
        cfg.model_kind,
        cfg.d_out,
        cfg.hidden,
        cfg.epochs,
        cfg.squared,
        cfg.residual,
    )
    print(
        f"Subcommand '{args.subcommand}' resolved config successfully. "
        f"Execution is not implemented yet (PR1-T05+)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
