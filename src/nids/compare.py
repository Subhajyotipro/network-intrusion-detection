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
"""Comparison table generator aggregating re-run and upstream-reported baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

UPSTREAM_TARGETS: list[dict[str, Any]] = [
    {
        "model": "CLOSR (target)",
        "closed_set_acc": 0.995276,
        "open_set_auc": 0.974022,
        "open_auc": 0.969420,
        "source": "upstream-reported",
        "reduced_run": False,
        "seed": "n/a",
        "data_source": "real",
    },
    {
        "model": "MultiStage",
        "closed_set_acc": 0.996612,
        "open_set_auc": 0.801251,
        "open_auc": 0.798537,
        "source": "upstream-reported",
        "reduced_run": False,
        "seed": "n/a",
        "data_source": "real",
    },
    {
        "model": "DOC",
        "closed_set_acc": 0.995536,
        "open_set_auc": 0.570263,
        "open_auc": 0.567717,
        "source": "upstream-reported",
        "reduced_run": False,
        "seed": "n/a",
        "data_source": "real",
    },
    {
        "model": "OpenMax",
        "closed_set_acc": 0.995615,
        "open_set_auc": 0.720174,
        "open_auc": 0.717016,
        "source": "upstream-reported",
        "reduced_run": False,
        "seed": "n/a",
        "data_source": "real",
    },
    {
        "model": "CRSOR",
        "closed_set_acc": 0.994940,
        "open_set_auc": 0.748295,
        "open_auc": 0.744509,
        "source": "upstream-reported",
        "reduced_run": False,
        "seed": "n/a",
        "data_source": "real",
    },
    {
        "model": "Siamese Network",
        "closed_set_acc": 0.997722,
        "open_set_auc": 0.720811,
        "open_auc": 0.719167,
        "source": "upstream-reported",
        "reduced_run": False,
        "seed": "n/a",
        "data_source": "real",
    },
    {
        "model": "AE (upstream)",
        "closed_set_acc": "n/a",
        "open_set_auc": 0.840659,
        "open_auc": "n/a",
        "source": "upstream-reported",
        "reduced_run": False,
        "seed": "n/a",
        "data_source": "real",
    },
    {
        "model": "IF (upstream)",
        "closed_set_acc": "n/a",
        "open_set_auc": 0.838422,
        "open_auc": "n/a",
        "source": "upstream-reported",
        "reduced_run": False,
        "seed": "n/a",
        "data_source": "real",
    },
]

COLUMNS = [
    "model",
    "closed_set_acc",
    "open_set_auc",
    "open_auc",
    "source",
    "reduced_run",
    "seed",
    "data_source",
]


def _format_cell(val: Any) -> str:
    """Format float or string value for Markdown and CSV display."""
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


def _load_rerun_records(results_dir: Path, is_synthetic: bool) -> list[dict[str, Any]]:
    """Load re-run baseline and evaluation records from results directory."""
    records = []
    base_dir = results_dir / "baselines"
    eval_dir = results_dir / "eval"

    for d in [base_dir, eval_dir]:
        if not d.is_dir():
            continue
        for p in d.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                ds = rec.get("data_source", rec.get("dataset", "real"))
                if (ds == "synthetic") != is_synthetic:
                    continue
                m = rec.get("metrics", {})
                records.append(
                    {
                        "model": rec.get("model", p.stem),
                        "closed_set_acc": m.get("closed_set_acc", "n/a"),
                        "open_set_auc": m.get("open_set_auc", "n/a"),
                        "open_auc": m.get("open_auc", "n/a"),
                        "source": "re-run",
                        "reduced_run": rec.get("reduced_run", False),
                        "seed": rec.get("seed", 1),
                        "data_source": ds,
                    }
                )
            except Exception:
                continue
    return records


def _write_csv_table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write comparison rows to CSV format."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _format_cell(r.get(k, "n/a")) for k in COLUMNS})


def _write_markdown_table(
    path: Path, rows: list[dict[str, Any]], banner: Optional[str] = None
) -> None:
    """Write comparison rows to GitHub-flavored Markdown table format."""
    lines = []
    if banner:
        lines.append(f"> **Warning**: {banner}\n")
    lines.append("| " + " | ".join(COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(COLUMNS)) + " |")
    for r in rows:
        lines.append("| " + " | ".join([_format_cell(r.get(k, "n/a")) for k in COLUMNS]) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_comparison_tables(
    results_dir: str | Path = "results",
    output_dir: Optional[str | Path] = None,
    synthetic_mode: bool = False,
) -> dict[str, str]:
    """Generate comparison.csv and comparison.md tables aggregating all results."""
    r_dir = Path(results_dir)
    out_dir = Path(output_dir or r_dir / "baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    rerun_rows = _load_rerun_records(r_dir, is_synthetic=synthetic_mode)
    base_rows = [] if synthetic_mode else list(UPSTREAM_TARGETS)
    all_rows = rerun_rows + base_rows

    prefix = "comparison_synthetic" if synthetic_mode else "comparison"
    csv_path = out_dir / f"{prefix}.csv"
    md_path = out_dir / f"{prefix}.md"

    banner = (
        "SYNTHETIC DATA RESULTS ONLY — Not validated on real Lycos2017 dataset"
        if synthetic_mode
        else None
    )
    _write_csv_table(csv_path, all_rows)
    _write_markdown_table(md_path, all_rows, banner=banner)

    return {"csv": str(csv_path), "md": str(md_path)}
