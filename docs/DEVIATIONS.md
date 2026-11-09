# DEVIATIONS.md — Pre-registered and Recorded Deviations

This file tracks all pre-registered and runtime deviations from `AGENTS.md` specifications.

## Canonical Deviations Register (Aligned with decisions.md Appendix B)

| ID | AGENTS.md / Upstream says | We do instead | Why |
|---|---|---|---|
| DV1 | §4 deep `src/nids/<subpackage>/` tree, separate scripts (§4 now explicitly allows flat form) | Flat modules, one `run.py` with subcommands | Fewer files and imports; same functionality (decisions.md D2) |
| DV2 | PR2-T02: pair or triplet sampling | In-batch pair mask with class-balanced batches (only if no upstream Siamese exists) | Same loss family, far less code; label-correctness still tested |
| DV3 | PR2-T01: temporarily treat a known attack as unseen | Drop that class's centroid / prototype at calibration time, no retraining | Avoids training K extra models; limitation documented in reports |
| DV4 | §12.5 label layout | Zero-day label = `-1`, class names kept separately | Removes upstream's ordering assumption |
| DV5 | §6.5 mixed precision off/on | Mixed precision not implemented, "off" only | Complexity; no accuracy benefit needed |
| DV6 | PR1-T03 validation split | Validation carved from train remainder after fixed upstream test split (`val_ratio` fraction of train remainder); `val_ratio=0.0` reproduces upstream exactly | Keeps test set identical to upstream while providing validation set for calibration (D4.3) |
| DV7 | Upstream pinned versions (`numpy==1.24.2`, `torch==2.0.1`, etc.) | Pinned modern versions (`numpy==2.4.6`, `tensorflow==2.21.0`, etc.) | Compatibility with Python 3.11 and Apple Silicon; zero PyTorch dependency |
| DV8 | AGENTS §2.3 global model defaults | Per-kind defaults (`default.yaml` for CLOSR: 3 layers, out 64; `clad.yaml` for CLAD: 4 layers, out 8) | Mirrors upstream training scripts per model kind |
| DV9 | Upstream training seed check | Upstream sets no training seed (only split seed `39058032`); training seeds are `[1, 2, 3]` | Deterministic reproducibility across multiple seeds (D4.3) |
| DV10 | Upstream implemented in PyTorch | All runtime code (`src/nids/`, `scripts/`) and tests strictly implemented in TensorFlow (`tf.keras`) and NumPy | Binding framework policy requirement |
| DV11 | Upstream numerical-equivalence tests executing PyTorch code | Replaced by independent NumPy reference implementations transcribed from upstream source | Why: zero PyTorch dependency in the project; limitation: cannot detect a misreading of upstream source, mitigated by hand-computed cases and the real-data reproduction check in PR1-T08 |
| DV12 | PyTorch baselines and Siamese | Baselines and Siamese re-implemented in TensorFlow or marked upstream-reported | Clean framework consistency; no PyTorch runtime coupling |
| DV13 | Upstream drops zero-columns across entire dataset prior to splitting | All-zero column mask computed strictly on train split (`x_train`) and applied to val/test/zero-day | Prevents test and zero-day data leakage into feature selection (§5.1, §12.11) |
| DV14 | Upstream `load_data.py` uses `df.dropna()` dropping rows with missing or infinite values | Replace ±inf with NaN and impute using training column medians (`_clean_and_scale`) | Preserves full sample flow count across splits while adhering to train-only statistics |
| DV15 | Normalization degrees of freedom: upstream `ZNormaliser` uses Bessel's correction (`ddof=1`) | Scikit-learn `StandardScaler` uses population standard deviation (`ddof=0`) | Built-in scikit-learn standard scaling; numerical difference on $N \approx 800\text{k}$ samples is $< 10^{-6}$ |
| DV16 | Upstream regenerates splits in memory without on-disk persistence | Save split indices (`train`, `val`, `test`) to compressed NPZ files via `save_split_indices` | Guarantees exact split persistence, reproducibility, and verifiable zero-leakage partitions (§5.1) |
| DV17 | Upstream masks self-pairs with `dist > eps` | We zero the diagonal explicitly (`tf.linalg.set_diag`) before masking | Float32 roundoff on unit vectors can leave self-distance above eps, erroneously counting self-pairs in `n_sim` |
| DV18 | PyTorch AdamW epsilon placement $\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$ | Keras AdamW folds bias-correction into numerator $\frac{\alpha \cdot m}{\sqrt{v} + \epsilon}$ | Built-in Keras optimizer formulation; epsilon set to upstream `1e-8` via Config; expected difference negligible |
| DV19 | AGENTS §2.3 human-readable class names with spaces/caps | Raw Lycos2017 labels are lowercase snake_case (`webattack_sql_injection`, `heartbleed`, etc.) | Direct alignment with raw dataset CSV without synthetic renaming overhead |
| DV20 | Silent fallback to synthetic data when CSV absent | Tri-state `data_source: "auto" \| "real" \| "synthetic"`; prominent warning on fallback | Prevents accidental execution on synthetic data when real data is intended |
