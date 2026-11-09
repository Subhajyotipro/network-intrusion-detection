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
import re
from pathlib import Path
from typing import Any, Optional

PROTOCOL_TABLE_1 = "binary AUROC vs benign (Table 1)"
PROTOCOL_TABLE_2 = "closed+open-set (Table 2)"
ALLOWED_PROTOCOLS = {PROTOCOL_TABLE_1, PROTOCOL_TABLE_2}

COLUMNS = [
    "model",
    "protocol",
    "closed_set_mean",
    "open_set_mean",
    "closed_set_acc",
    "open_set_auc",
    "open_auc",
    "source",
    "reduced_run",
    "seed",
    "data_source",
]


def _clean_markdown_cell(s: str) -> str:
    """Strip markdown formatting asterisks, underscores, and backticks."""
    return re.sub(r"[*_`]", "", s).strip()


def parse_upstream_readme(
    readme_path: str | Path = "third_party/CLOSR/README.md",
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Parse Table 1 and Table 2 numbers directly from upstream README markdown."""
    path = Path(readme_path)
    if not path.is_file():
        return {}, {}
    lines = path.read_text(encoding="utf-8").splitlines()

    t1_lines = lines[55:72]
    h1 = [_clean_markdown_cell(c) for c in t1_lines[0].split("|")[1:-1]]
    t1_dict: dict[str, dict[str, float]] = {}
    for line in t1_lines[2:]:
        parts = [_clean_markdown_cell(c) for c in line.split("|")[1:-1]]
        if parts:
            t1_dict[parts[0]] = {h1[i]: float(parts[i]) for i in range(1, len(parts))}

    t2_lines = lines[76:84]
    t2_dict: dict[str, dict[str, float]] = {}
    for line in t2_lines[2:]:
        parts = [_clean_markdown_cell(c) for c in line.split("|")[1:-1]]
        if parts:
            t2_dict[parts[0]] = {
                "closed_set_acc": float(parts[1]),
                "open_set_auc": float(parts[2]),
                "open_auc": float(parts[3]),
            }
    return t1_dict, t2_dict


def _build_upstream_rows(readme_path: str | Path) -> list[dict[str, Any]]:
    """Build upstream-reported baseline comparison rows for Table 1 and Table 2."""
    t1, t2 = parse_upstream_readme(readme_path)
    rows: list[dict[str, Any]] = []
    if "Closed Set Mean" in t1 and "Open Set Mean" in t1:
        for model in ["CLAD", "AE", "IF", "MLP"]:
            rows.append(
                {
                    "model": f"{model} (upstream)",
                    "protocol": PROTOCOL_TABLE_1,
                    "closed_set_mean": t1["Closed Set Mean"].get(model, "n/a"),
                    "open_set_mean": t1["Open Set Mean"].get(model, "n/a"),
                    "closed_set_acc": "n/a",
                    "open_set_auc": "n/a",
                    "open_auc": "n/a",
                    "source": "upstream-reported",
                    "reduced_run": False,
                    "seed": "n/a",
                    "data_source": "real",
                }
            )
    t2_models = ["CLOSR", "MultiStage", "DOC", "OPENMAX", "CRSOR", "Siamese Network"]
    for m in t2_models:
        if m in t2:
            rows.append(
                {
                    "model": f"{m} (upstream)",
                    "protocol": PROTOCOL_TABLE_2,
                    "closed_set_mean": "n/a",
                    "open_set_mean": "n/a",
                    "closed_set_acc": t2[m].get("closed_set_acc", "n/a"),
                    "open_set_auc": t2[m].get("open_set_auc", "n/a"),
                    "open_auc": t2[m].get("open_auc", "n/a"),
                    "source": "upstream-reported",
                    "reduced_run": False,
                    "seed": "n/a",
                    "data_source": "real",
                }
            )
    return rows


def _format_cell(val: Any) -> str:
    """Format float or string value for Markdown and CSV display."""
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


def _load_rerun_records(results_dir: Path, is_synthetic: bool) -> list[dict[str, Any]]:
    """Load re-run baseline and evaluation records from results directory."""
    records: list[dict[str, Any]] = []
    b_dir, e_dir = results_dir / "baselines", results_dir / "eval"
    for p in list(b_dir.glob("*.json")) + list(e_dir.glob("*_eval.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                rec = json.load(f)
            ds = rec.get("data_source", rec.get("dataset", "real"))
            if (ds == "synthetic") != is_synthetic:
                continue
            m = rec.get("metrics", {})
            name = rec.get("model", p.stem.replace("_eval", ""))
            is_t1 = "closed_set_auc_vs_benign" in m or name in ("clad", "autoencoder", "iforest")
            proto = PROTOCOL_TABLE_1 if is_t1 else PROTOCOL_TABLE_2
            records.append(
                {
                    "model": f"{name.upper()} (re-run)",
                    "protocol": proto,
                    "closed_set_mean": m.get(
                        "closed_set_auc_vs_benign", m.get("known_mean_auroc", "n/a")
                    ),
                    "open_set_mean": m.get("open_set_auc_vs_benign", m.get("zd_mean_auroc", "n/a")),
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


T1_COLS = [
    "model",
    "protocol",
    "closed_set_mean",
    "open_set_mean",
    "source",
    "reduced_run",
    "seed",
    "data_source",
]
T2_COLS = [
    "model",
    "protocol",
    "closed_set_acc",
    "open_set_auc",
    "open_auc",
    "source",
    "reduced_run",
    "seed",
    "data_source",
]


def _format_markdown_section(title: str, cols: list[str], rows: list[dict[str, Any]]) -> list[str]:
    """Format a single Markdown table section."""
    lines = [
        f"### {title}",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join([_format_cell(r.get(k, "n/a")) for k in cols]) + " |")
    return lines


def _write_markdown_table(
    path: Path, rows: list[dict[str, Any]], banner: Optional[str] = None
) -> None:
    """Write comparison rows to GitHub-flavored Markdown table format."""
    lines = []
    if banner:
        lines.append(f"> **Warning**: {banner}\n")

    t1_rows = [r for r in rows if r.get("protocol") == PROTOCOL_TABLE_1]
    t2_rows = [r for r in rows if r.get("protocol") == PROTOCOL_TABLE_2]

    lines.extend(
        _format_markdown_section(
            "Table 1: Binary AUROC vs Benign (One-vs-Benign)", T1_COLS, t1_rows
        )
    )
    lines.append("")
    lines.extend(
        _format_markdown_section("Table 2: Closed + Open-Set Recognition", T2_COLS, t2_rows)
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_comparison_tables(
    results_dir: str | Path = "results",
    output_dir: Optional[str | Path] = None,
    synthetic_mode: bool = False,
    readme_path: str | Path = "third_party/CLOSR/README.md",
) -> dict[str, str]:
    """Generate comparison.csv and comparison.md tables aggregating all results."""
    r_dir = Path(results_dir)
    out_dir = Path(output_dir or r_dir / "baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    rerun_rows = _load_rerun_records(r_dir, is_synthetic=synthetic_mode)
    base_rows = [] if synthetic_mode else _build_upstream_rows(readme_path)
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
