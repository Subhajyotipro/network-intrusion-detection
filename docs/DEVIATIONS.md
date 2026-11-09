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
| DV7 | Upstream pins: `numpy==1.24.2`, `pandas==2.0.0`, `scikit-learn==1.2.2`, `torch==2.0.1` | Pinned modern versions (`numpy==2.4.6`, `tensorflow==2.21.0`, etc.) | Compatibility with Python 3.11 and Apple Silicon macOS; zero PyTorch dependency |
| DV8 | AGENTS §2.3 global model defaults (out 64, hidden 1024-1024-1024) | Per-kind defaults: CLOSR uses out 64, 3 hidden layers; CLAD uses out 8, 4 hidden layers per `train_clad.py` | Mirrors upstream scripts exactly per model kind |
| DV9 | AGENTS §5.2 / decisions.md D4.3 upstream default training seed check | Upstream scripts set no training seed (only split seed `39058032`); training seeds are `[1, 2, 3]` | Ensures deterministic, multi-seed training runs |
| DV10 | Upstream implemented in PyTorch | All runtime code (`src/nids/`, `scripts/`) implemented in TensorFlow (`tf.keras`) | Binding framework policy requirement |
| DV11 | Upstream numerical-equivalence tests executing PyTorch code | Replaced by independent NumPy reference implementations transcribed from upstream source | Why: zero PyTorch dependency in the project; limitation: cannot detect a misreading of upstream source, mitigated by hand-computed cases and the real-data reproduction check in PR1-T08 |
| DV12 | PyTorch baselines and Siamese | Baselines and Siamese re-implemented in TensorFlow or marked upstream-reported | Clean framework consistency; no PyTorch runtime coupling |
| DV13 | Upstream drops zero-columns across entire dataset prior to splitting | All-zero column mask computed strictly on train split (`x_train`) and applied to val/test/zero-day | Prevents test and zero-day data leakage into feature selection (§5.1, §12.11) |
| DV14 | Upstream `load_data.py` uses `df.dropna()` dropping rows with missing or infinite values | Replace ±inf with NaN and impute using training column medians (`_clean_and_scale`) | Preserves full sample flow count across splits while adhering to train-only statistics |
| DV15 | Upstream regenerates splits in memory without on-disk persistence | Save split indices (`train`, `val`, `test`) to compressed NPZ files via `save_split_indices` | Guarantees exact split persistence, reproducibility, and verifiable zero-leakage partitions (§5.1) |
| DV16 | Upstream masks self-pairs with `dist > eps` | We zero the diagonal explicitly (`tf.linalg.set_diag`) before masking | Float32 roundoff on unit vectors can leave self-distance above eps, erroneously counting self-pairs in `n_sim` |
| DV17 | PyTorch AdamW epsilon placement $\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$ | Keras AdamW folds bias-correction into numerator $\frac{\alpha \cdot m}{\sqrt{v} + \epsilon}$ | Built-in Keras optimizer formulation; epsilon set to upstream `1e-8` via Config; expected difference negligible |
| DV18 | AGENTS §2.3 human-readable class names with spaces/caps | Raw Lycos2017 labels are lowercase snake_case (`webattack_sql_injection`, `heartbleed`, etc.) | Direct alignment with raw dataset CSV without synthetic renaming overhead |
| DV19 | Silent fallback to synthetic data when CSV absent | Tri-state `data_source: "auto" \| "real" \| "synthetic"`; prominent warning on fallback | Prevents accidental execution on synthetic data when real data is intended |

