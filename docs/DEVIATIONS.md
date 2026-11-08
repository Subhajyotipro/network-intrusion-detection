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
| DV7 | Upstream pins: `numpy==1.24.2`, `pandas==2.0.0`, `scikit-learn==1.2.2`, `torch==2.0.1` | Pinned modern versions (`numpy==2.4.6`, `tensorflow==2.21.0`, etc.) | Compatibility with Python 3.11 and Apple Silicon macOS; torch moved to reference-only |
| DV8 | AGENTS §2.3 global model defaults (out 64, hidden 1024-1024-1024) | Per-kind defaults: CLOSR uses out 64, 3 hidden layers; CLAD uses out 8, 4 hidden layers per `train_clad.py` | Mirrors upstream scripts exactly per model kind |
| DV9 | AGENTS §5.2 / decisions.md D4.3 upstream default training seed check | Upstream scripts set no training seed (only split seed `39058032`); training seeds are `[1, 2, 3]` | Ensures deterministic, multi-seed training runs |
| DV10 | Upstream implemented in PyTorch | All runtime code (`src/nids/`, `scripts/`) implemented in TensorFlow (`tf.keras`) | Binding framework policy requirement |
| DV11 | PyTorch everywhere | PyTorch strictly isolated to `requirements-reference.txt` for `@pytest.mark.reference` tests | Guarantees torch-free runtime while enabling numerical validation |
| DV12 | PyTorch baselines and Siamese | Baselines and Siamese re-implemented in TensorFlow or marked upstream-reported | Clean framework consistency; no PyTorch runtime coupling |

