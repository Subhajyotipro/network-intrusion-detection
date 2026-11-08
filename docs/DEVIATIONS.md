# DEVIATIONS.md — Pre-registered and Recorded Deviations

This file tracks all pre-registered and runtime deviations from `AGENTS.md` specifications.

## Pre-Registered Deviations (from decisions.md Appendix B)

| ID | AGENTS.md says | We do instead | Why |
|---|---|---|---|
| DV1 | §4 deep `src/nids/<subpackage>/` tree, separate scripts (§4 now explicitly allows the flat form) | Flat modules, one `run.py` with subcommands | Fewer files and imports; same functionality |
| DV2 | PR2-T02: pair or triplet sampling | In-batch pair mask with class-balanced batches (only if no upstream Siamese exists) | Same loss family, far less code; label-correctness still tested |
| DV3 | PR2-T01: temporarily treat a known attack as unseen | Drop that class's centroid / prototype at calibration time, no retraining | Avoids training K extra models; limitation documented in reports |
| DV4 | §12.5 label layout | Zero-day label = `-1`, class names kept separately | Removes upstream's ordering assumption |
| DV5 | §6.5 mixed precision off/on | Mixed precision not implemented, "off" only | Complexity; no accuracy benefit needed |
| DV6 | PR1-T03 validation split | Validation carved from train **after** the fixed upstream test split; `val_ratio=0` for exact reproduction | Keeps the test set identical to upstream |

## Further Deviations

| ID | AGENTS.md / decisions.md says | We do instead | Why |
|---|---|---|---|
| DV7 | Upstream pins: `numpy==1.24.2`, `pandas==2.0.0`, `scikit-learn==1.2.2`, `torch==2.0.1` | Use nearest installable versions compatible with Python 3.11 on Apple Silicon macOS | Older binary wheels for numpy 1.24.2 / pandas 2.0.0 are not natively precompiled for modern macOS arm64 Python 3.11 |
