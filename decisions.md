# decisions.md — Binding implementation decisions for AGENTS.md

> Companion to `AGENTS.md`. AGENTS.md says **what** to deliver; this file says **how** to build it with the least code, least complexity, and fewest syntax/runtime errors.
>
> **Precedence rule (matches AGENTS.md §0 item 8):**
> 1. Scientific protocol, deliverables, acceptance criteria, and the working protocol in AGENTS.md (§5.1, §5.5, §6, §10, §12, all PR task tables) are **never weakened** by this file.
> 2. Where AGENTS.md leaves a choice open, this file makes it.
> 3. The only places this file *departs* from AGENTS.md are the **pre-registered deviations in Appendix B**. AGENTS.md already points to each of them (§4, §6.5, PR1-T03, PR2-T01, PR2-T02, §13). Copy them into `docs/DEVIATIONS.md` during PR1-T01 (a PR1-T01 acceptance criterion). Any other departure is a bug in this file — follow AGENTS.md and log it.
> 4. **Sync rule:** if either file changes, update Appendix A and B of this file in the same commit, and re-check that no sentence here contradicts AGENTS.md.
>
> Appendix A maps every AGENTS.md requirement to the section here that covers it.

---

## D0. Prime directive

1. **Simplest thing that passes the acceptance criteria.** Two designs that both pass → pick the one with fewer lines and fewer concepts.
2. **Copy before you write.** If `third_party/CLOSR/` has working code for it, copy it with the Apache-2.0 header and change only what the task requires (`d_in`, device, class map, config).
3. **Never write code you cannot run.** Every file is executed/imported and tested before moving on.
4. **Honesty beats polish.** Real numbers only; gaps are reported, not hidden.
5. **Never use** upstream `weights/` binaries as results, upstream `.gitignore` rules, or upstream data not licensed for redistribution (AGENTS §3).

---

## D1. Rules to keep code small and correct

| # | Rule |
|---|---|
| 1 | Plain functions over classes. Classes only for `tf.keras.Model` / `tf.keras.layers.Layer` (when needed), dataclasses (config, `ClassMap`), `Gatekeeper`/`Cascade`. |
| 2 | No abstract base classes, registries, decorators, plugin systems, or factories. Pluggable things (scorers, threshold strategies) are a `dict[str, Callable]`. |
| 3 | No new frameworks: no Lightning, Hydra, MLflow, W&B, Optuna, Ray, Docker. |
| 4 | Functions ≤ 40 lines, files ≤ 300 lines. |
| 5 | Type hints and a one-line docstring on every public function. |
| 6 | Allowed libraries: `pathlib`, `logging`, `json`, `numpy`, `pandas`, `tensorflow`, `sklearn`, `matplotlib`, `pyyaml`, `pytest`, `pytest-cov`, `pytest-timeout`, `ruff`. Nothing else unless upstream's `requirements.txt` already needs it. PyTorch is strictly excluded from the entire repository (no runtime, no test, no reference environment). `pytest-timeout` is configured in `pyproject.toml` (120-second timeout per test) so a hanging test cannot freeze the loop; marker `real` is registered in `pyproject.toml`. |
| 7 | Logging uses the standard `logging` module (AGENTS §5.4). **Only** the `demo` subcommand may `print`, because its output is for a human reader. |
| 8 | **No hard-coded paths or magic numbers** (AGENTS §5.4): every number, path, grid size, and default lives in `Config` (D4.5). |
| 9 | No dead code, commented-out blocks, or "future use" parameters. |
| 10 | Fail loudly: `assert`/`raise ValueError` with a clear message on shape, label, or class-map problems. No silent coercion. |
| 11 | No multiprocessing, `tf.distribute`, or distributed code. Mixed precision is **not implemented** (so AGENTS §6.5's "mixed precision off/on" check reduces to "off"; recorded in Appendix B). |
| 12 | **Framework policy:** All runtime code (`src/nids`, `scripts`) and all tests strictly use TensorFlow with `tf.keras` and NumPy. No file in the repository outside `third_party/` may import `torch`, directly or indirectly (enforced by AST scan across all repository python files, unit tests, and ruff banned-api rule). Upstream is read-only source text for inspection only (never imported, executed, or on sys.path). NumPy reference tests replace upstream execution. |

---

## D2. Package layout (flattened version of AGENTS §4; registered as DV1)

```
pyproject.toml  requirements.txt  README.md  NOTICE  LICENSE  .gitignore
src/nids/
  __init__.py
  config.py      # Config dataclass + YAML load + CLI override
  data.py        # load, split, scale, sampler, synthetic generator, ClassMap, constants
  models.py      # mlp helper, CLAD/CLOSR model, Siamese encoder
  losses.py      # CLAD, CLOSR, Siamese loss
  train.py       # one trainer for all model kinds + checkpoint save/load
  infer.py       # embeddings, centroids/prototypes, scorers, Gatekeeper, SiameseClassifier
  metrics.py     # all metrics + per-class tables
  calibrate.py   # threshold strategies, pseudo-zero-day, joint calibration
  cascade.py     # Cascade + error-propagation analysis
  bench.py       # latency/throughput
  viz.py         # all plots
scripts/         # run.py (argparse subcommands: train, eval, calibrate, cascade, bench, demo) and make_report.py
configs/default.yaml
tests/unit/  tests/integration/  tests/regression/  (+ tests/conftest.py)
results/  docs/  third_party/CLOSR/
docs/            # UPSTREAM_INVENTORY.md DEVIATIONS.md BUGLOG.md DATA_CARD.md reports/
```

- `pyproject.toml` makes `pip install -e .` work so `import nids` works everywhere (required by the D3 check loop).
- AGENTS §4 requires a separate train/evaluate/calibrate/benchmark/run-demo/make-report CLI; `run.py` subcommands + `make_report.py` satisfy it.
- The **tests folder structure is kept exactly as AGENTS §4/PR1-T10 specify** (unit/integration/regression), only the source package is flattened.
- README must contain the "Quick start" and a one-page architecture diagram (CLOSR gate → Siamese) as ASCII or Mermaid text (PR3-T08).

---

## D3. Environment and the "correct on first run" protocol

1. Python 3.10+. `requirements.txt` pins `tensorflow` (no `torch`). Do not install `tensorflow-metal` or GPU plugins unless requested. The default test command is `pytest -x -q -m "not real"`. There is no reference test command or PyTorch environment.
2. Before using a library function you are not certain about, check the **installed** version's signature (`help(...)` or a one-line `python -c`). Do not rely on memory for sklearn/tensorflow argument names.
3. After **every file created or edited**, run in order and fix before moving on:
   1. `python -m py_compile <file>`
   2. `ruff check <file>` and `ruff format <file>`
   3. `python -c "import nids.<module>"`
   4. `pytest -x -q <matching test file>`
   After finishing each task (and again before writing each PR summary report), run the automatic test-and-repair loop: `python -m py_compile` on changed files, `ruff check`, then `pytest -x -q -m "not real"`. If any test fails, diagnose the root cause, fix the code, and rerun the whole suite until green (cap: 5 fix-iterations per distinct failing test).
4. Run `pytest -x -q -m "not real"` (full suite) before every commit, repeating the test-and-repair loop until green. If real data exists, also run `pytest -m real` once at the end of each PR. Suite target: **< 2 minutes on CPU**.
5. Device is selected in exactly one place (`config.py`): using TensorFlow's physical-device listing (`GPU` if available else `CPU`). Passing `--set device=cpu` explicitly hides GPUs from TensorFlow via `tf.config.set_visible_devices([], "GPU")`. All code receives the device from config. Every script must run on CPU.
6. One commit per task: `PRn-Txx: short title`. Do not commit failing tests, except documented OPEN bugs marked `xfail` with a reason.
7. If the real dataset is missing: build and use the synthetic generator, mark real-data tasks `BLOCKED: dataset`, and tell the human which file goes where (AGENTS §5.3).
8. **CPU-only real-data training:** 200 epochs at batch 4096 may be impractical. Run with a reduced `epochs` (via `--set epochs=N`), store `epochs` and `reduced_run=true` in the result JSON, and mark PR1-T08 `PARTIAL` with the real numbers. Never present reduced-epoch numbers as full reproduction.

---

## D4. Decisions on conventions that otherwise cause bugs

### D4.1 Labels and class map
- `ClassMap` dataclass in `data.py`: `names` (known classes, benign first) and `zero_day_names`.
- Constants defined **once** in `data.py`: `BENIGN_ID = 0`, `UNKNOWN_ID = -1`.
- **Zero-day flows carry label `UNKNOWN_ID` (-1) in all evaluation arrays**; their original class names are kept in a parallel `zd_names` array for per-class AUROC. This removes upstream's "zero-day labels come after known labels" assumption (AGENTS §12.5).
- `n_known` comes from `ClassMap`, never from unique test labels. Assert every known class is present in train.
- **Upstream-equivalence tests** convert `-1` labels to upstream's layout through a small adapter inside the test, so numbers can be compared directly.

### D4.2 Score direction and scorers
- **Every scorer returns a known-ness score: higher = more likely known.** Reject if `score < threshold`. For ROC with zero-day as the positive class, pass `-score` — done in exactly one helper in `metrics.py` (matches AGENTS §12.7).
- Scorers dict in `infer.py`: `"upstream"` (Σ sim² · softmax(sim), copied from upstream), `"max_cos"`, `"max_softmax"`.
- Default scorer is `"upstream"`. PR2-T05 requires recording the decision: compare the three on **validation (pseudo-zero-day, D5)** and on test, write the choice and the evidence into the PR2 report. Switch the default only if another scorer is clearly better on validation.
- FPR@95 uses zero-day as the positive class, copying upstream's `fpr_at_recall` direction; state this in every report (AGENTS §13).

### D4.3 Splits and seeds
- **Split seed is fixed at `39058032`** (upstream) for all experiments; every model sees identical data.
- **Training seeds: `[1, 2, 3]`** for every multi-seed table (AGENTS §5.2, mean ± std). Upstream sets no training seed; training seeds are 1, 2, 3.
  Note: because the split is fixed, std reflects training randomness only — state this in reports.
- `test_ratio=0.5` always, so **the test set is identical to upstream's**. `val_ratio` is carved out of the *train* remainder **after** the test split (default `0.1` for PR2/PR3; `0.0` for PR1-T08 to reproduce upstream exactly). Consequence: PR2/PR3 models train on 90 % of upstream's train set, so their numbers are **not directly comparable to §2.4**; say so in the reports.
- Splits are stratified (`sklearn.model_selection.train_test_split`) and saved to `results/splits/<hash>.npz` (indices).
- Classes with < `sample_thres` (100) samples → zero-day. Log the detected names and compare with Heartbleed / SQL Injection; any mismatch goes to DEVIATIONS and the run continues with what was detected.

### D4.4 Preprocessing
- Drop identifier columns (flow_id, src_addr, src_port, dst_addr, dst_port, ip_prot, timestamp) — always.
- Replace ±inf with NaN; fill NaN with the **train-set** median; `StandardScaler` fit on **train only** (AGENTS §12.11). First read upstream's `load_data.py`: if it scales differently or fits statistics on all data, copy the transform but fit on train only and log it in BUGLOG/DEVIATIONS.
- `d_in` = number of feature columns after dropping, written into config at load time.

### D4.5 Config
- One flat `Config` dataclass + `configs/default.yaml`. Defaults are per model kind (closr, clad), taken from the upstream training script for that kind; `configs/default.yaml` holds the CLOSR defaults and `configs/clad.yaml` holds only the values that differ. Siamese and MLP defaults get defined when those tasks start.
- CLI override: repeated `--set key=value`; the value is parsed with `yaml.safe_load`, so `false`, `1e-5`, `[1,2]` parse correctly. This fixes the boolean-flag bug (AGENTS §12.1) in a few lines. **`residual` stays in the config** (default `false`, copied from upstream's model if upstream implements it), and a test checks `--set residual=false` and `--set squared=false` yield `False`.
- Required fields (so no magic numbers live in code):
  - paths: `data_path`, `results_dir`, `weights_dir`, `upstream_dir`
  - data: `split_seed`, `test_ratio`, `val_ratio`, `sample_thres`
  - model/loss: `hidden`, `d_out`, `dropout`, `residual`, `margin`, `squared`
  - optim: `batch_size`, `weight_decay`, `lr_init`, `lr_peak`, `lr_final`, `warmup_frac`, `epochs`, `seed`, `seeds`, `smoke`, `max_batches`
  - inference/calibration: `scorer`, `grid_size` (200), `target_acceptance` ([0.95, 0.99]), `f_beta` (1.0), `max_known_rejection` (X, 0.05), `max_benign_fp` (Y, 0.01), `siamese_conf_grid` (10), `skip_if_confident_benign` (false)
  - Siamese: `support_mode` ("prototype" | "exemplar"), `k_exemplars`
  - bench: `warmup_runs` (20), `timed_runs` (200), `bench_batch_sizes` ([1, 32, 1024])
  - viz: `tsne_max_points` (5000)
- `lr_init`, `lr_peak`, `lr_final`, `warmup_frac`: read upstream's `train_closr.py` and use its values (AGENTS §12.3 lists the init as 1e-6, peak 1e-5).
- Metadata versions recorded: `tensorflow`, `keras`, `numpy`, `pandas`, `sklearn`, `python` (no `torch`).

### D4.6 Checkpoints
One checkpoint bundle per run: model weights (Keras weights file, e.g. `.weights.h5` or variable state), optimizer variables, schedule step, and a JSON/NPZ sidecar with `config`, `class_map`, `seed`, and `extra` (centroids or Siamese support set). `load_checkpoint` restores weights, optimizer variables, schedule step, and sidecar; reload restores identical outputs, and resume reproduces identical next-step loss.

### D4.7 Result files
- Every run writes `results/<exp>/<model>_seed<k>.json` with: `metrics`, `config`, `seed`, `git_commit`, `versions`, `device`, `dataset` ("real" | "synthetic"), `epochs`, `reduced_run`.
- The `eval` subcommand also writes the per-class table as CSV (one `DataFrame.to_csv` call) — AGENTS PR1-T06 requires JSON + CSV.
- Every run also saves `<model>_seed<k>_arrays.npz` (scores, predictions, labels, `zd_names`, and a seeded embedding subsample of size `tsne_max_points`). `make_report.py` regenerates **all** tables and figures from the JSONs and these arrays, with no retraining (PR3-T07).
- Aggregated CSV tables (mean ± std across seeds) come only from `make_report.py`.

---

## D5. Per-component decisions

### Models and losses (PR1-T04)
- One `mlp(d_in, hidden, d_out, dropout)` helper built with Keras layers (`keras.Sequential`, `keras.layers.Dense`, `keras.layers.Dropout`). Defaults: hidden `[1024,1024,1024]`, out 64, dropout 0.1.
- Initialisation: upstream PyTorch uses Kaiming uniform ($a=\sqrt{5}$) for linear weights and uniform for bias, which bounds weights to $\pm 1/\sqrt{\text{fan\_in}}$. In Keras, `keras.initializers.VarianceScaling(scale=1/3, mode="fan_in", distribution="uniform")` with uniform bias $[-1/\sqrt{\text{fan\_in}}, 1/\sqrt{\text{fan\_in}}]$ yields identical parameter distribution bounds (`torch_default` initialization helper).
- CLOSR model: re-implement in TensorFlow (`CLOSRMLP`), outputting L2-normalised class embeddings `[batch, n_classes, d_out]`; make `d_in` configurable.
- Losses: re-implement `CLADLoss` and `CLOSRLoss` with TensorFlow ops (`tf.linalg.l2_normalize`, `tf.square`, `tf.reduce_sum`, `tf.maximum`, `tf.reduce_mean`).
- Numerical equivalence verification without PyTorch: PR1-T04 verifies numerical equivalence without any PyTorch execution using:
  1. An independent pure NumPy reference implementation transcribed directly from upstream source text (stored in the test tier, e.g. `tests/unit/np_reference.py`, never in `src/`);
  2. Hand-computed tiny cases with analytical closed-form mathematical results;
  3. Initializer bounds tests confirming Keras uniform init matches PyTorch Linear weight and bias bounds;
  4. Exact loss formulas from the published IEEE TNSM paper.
- Unit tests (AGENTS PR1-T04 list): shapes, unit norm, finite and non-negative, gradients reach all parameters, loss drops over ~20 steps on a toy separable set, `margin` and `squared` change the value, benign/malicious asymmetry.

### Trainer (PR1-T05)
- One `train(cfg, kind)` with a plain loop using `tf.GradientTape`; `kind ∈ {"clad","closr","siamese","mlp"}`; the only per-kind difference is `(model, loss_fn)`. Pass `training=True` during training steps.
- AdamW: Keras `keras.optimizers.AdamW` (decoupled weight decay matching PyTorch AdamW semantics), upstream warm-up + cosine schedule (per-step schedule), small running-mean meter, periodic logging through `logging`.
- Step the schedule at the same granularity as upstream (per step, verified in `train_closr.py`).
- `smoke=true` → 2 epochs, `max_batches=3`.
- Resume: load weights, optimizer state, and step; continue from saved step. Note on dropout RNG: Keras 3 `Dropout` layer's `seed_generator_state` is not serialized in `model.save_weights` (`.weights.h5`). With `dropout > 0`, training resume is statistically equivalent; for bitwise-identical next-step loss across resume, `dropout=0` is used. Test: train 2 epochs + resume 1 vs 3 straight with `dropout=0` → equal loss within 1e-6.
- Sampler / Data loader: a simple NumPy class-balanced batch generator (optionally wrapped in `tf.data`). Clamp `batch_size = min(batch_size, len(train))` and use `drop_last=False` when the dataset is smaller than one batch (AGENTS §12.8). Test with a class smaller than the batch.

### Inference and metrics (PR1-T06)
- `infer.py`: `get_embeddings` (runs with `training=False`, no `tf.GradientTape`, chunked), `get_centroids`, `class_sims`, scorers dict.
- `metrics.py`: `sklearn.metrics` for AUROC, average precision, PR-AUC, F1, confusion matrix; hand-write only `fpr_at_recall` (upstream direction), `open_auc = closed_acc × open_auc_score`, per-class tables. Cover every metric listed in AGENTS §2.3.
- `safe_auroc(y, s)` returns `nan` with a logged warning when only one class is present; division by zero in recall → `nan` (AGENTS §12.12).
- CLAD evaluation: per-attack-class AUROC split into closed-set and open-set parts with means.

### Baselines (PR1-T07)
- **Must run:** MLP (closed-set, Keras), Isolation Forest (`sklearn`), Autoencoder (re-implemented in Keras with reconstruction MSE as the anomaly score).
- Upstream PyTorch baselines (MultiStage, DOC, OpenMax, CRSOR) are reported from upstream paper unless a faithful TensorFlow port is small. Every comparison-table row carries a column `source` = `re-run` or `upstream-reported`.
- MLP open-set score = max softmax probability.

### Siamese Stage 2 (PR2-T02; registered as DV2)
- Re-implement Siamese classifier in TensorFlow based on the upstream Siamese design (`siamese_network.py`):
  - Shared Keras encoder, 64-d, L2-normalised.
  - In-batch pair mask or pair sampling with class-balanced batches (DV2). Loss = margin contrastive loss on cosine distance, margin from config.
  - Support set: `support_mode="prototype"` (mean normalised embedding per class from train — deterministic) or `"exemplar"` (`k_exemplars` per class using a seeded `np.random.default_rng`). Stored in checkpoint sidecar.
  - Prediction = argmax cosine to the support set; **confidence = max cosine**.
- Pair-correctness test (PR2-T08): the positive mask equals `labels[:,None]==labels[None,:]` on a toy batch, so no pair mixes labels incorrectly. Support-set determinism test: build twice → identical.
- Accuracy target ~0.9977 (upstream). If we fall short, report the real number and the gap — do not tune on test.
- Pair-correctness test (PR2-T08): the positive mask equals `labels[:,None]==labels[None,:]` on a toy batch, so no pair mixes labels incorrectly. Support-set determinism test: build twice → identical.
- Accuracy target ~0.9977 (upstream). If we fall short, report the real number and the gap — do not tune on test.

### Gatekeeper and thresholds (PR2-T05, PR2-T06)
- `Gatekeeper(model, centroids, scorer, threshold)` with `__call__(X)` → `(score, pred_class, accept)`. A single flow is a batch of one (`X[None]`), same code path.
- Threshold helpers in `calibrate.py`, all taking validation score arrays and returning one float, on a `grid_size`-point quantile grid:
  - `thr_at_acceptance(known_scores, rate)` = `np.quantile(known_scores, 1 - rate)` (uses full-gate **known** validation scores only; run for each rate in `target_acceptance`);
  - `thr_eer(known, unknown)` via `sklearn.metrics.roc_curve`;
  - `thr_max_f(known, unknown, beta)`.
- Produce the ROC (known vs unknown) with the chosen operating points marked; save chosen thresholds to a JSON file.
- Test: realised known-acceptance on validation is within 0.01 of each target.

### Pseudo-zero-day (PR2-T01; registered as DV3)
Real zero-day classes **never** enter calibration. To simulate unseen attacks without retraining:
- For each known **attack** class `k` (id ≠ `BENIGN_ID`): rebuild the gate's similarities with centroid `k` **removed** (`keep` mask over classes), score **validation** samples of the remaining classes (→ pseudo-known) and validation samples of class `k` (→ pseudo-unknown). For Stage 2, remove prototype `k` the same way and map predictions back through `keep`.
- Pool all `(known_k, unknown_k)` pairs into `val_known` and `val_unknown`. Thresholds that need unknown scores (EER, F-score, joint, constrained) use these pools.
- **Honest limitation to state in reports:** the embedding network was still trained on class `k`, so pseudo-unknowns are probably easier to reject than real zero-days. Hence every final number is reported on the real zero-day set, and the threshold is never touched using it. Retraining the model per held-out class is a stretch goal (D9).
- Leakage test: pass calibration functions a data object whose zero-day arrays raise on access; calibration must still run. Name every calibration input `val_*`; `calibrate.py` must not import or reference the zero-day split.

### Cascade (PR3-T01)
- `Cascade(gate, siamese, thr)`. `predict(X)` returns labels in `{0..n_known-1, UNKNOWN_ID}` plus a dict of intermediates: gate score, accept mask, Stage-2 class, Stage-2 confidence.
- Stage 2 runs **only on accepted rows**; empty accepted/rejected sets return early. Tests: all rejected, all accepted, batch size 1, deterministic output, and batch result == per-flow result on 50 rows.
- Variant A (default): all accepted flows go to Siamese. Variant B (`skip_if_confident_benign=false` by default): ≤ 5 lines, used only in ablation PR3-T06(d).

### Joint calibration (PR3-T02)
- Compute Stage-2 predictions **once**; threshold changes only alter masks, so grid search is vectorised NumPy. Search: `grid_size` gate thresholds, optionally × `siamese_conf_grid` Stage-2 confidence thresholds.
- Three strategies, each a small function returning thresholds:
  1. **gate-only** = PR2-T06 choice;
  2. **joint** = maximise end-to-end macro-F1 over known ∪ {UNKNOWN} on validation (known val + pseudo-zero-day as UNKNOWN);
  3. **constrained** = maximise pseudo-zero-day recall subject to known-rejection ≤ `max_known_rejection` and benign FP ≤ `max_benign_fp`.
- Trade-off plot: (known false-rejection rate, zero-day recall) over the grid with the three chosen points marked — from validation/pseudo-zero-day for the choice, and separately from test + real zero-day for reporting only.
- Test: the threshold chosen is unchanged when the test/zero-day arrays are altered (proves it uses validation only).

### Cascade metric definitions (resolves an ambiguity in PR3-T04; mirrored in AGENTS §13)
- **Open-set AUC for the cascade** = AUROC of the gate score (identical to CLOSR-only by construction; state this explicitly).
- **Closed-set accuracy for the cascade** = reported in **two** forms: (i) Stage-2 accuracy on *all* known test flows (pure closed-set capability), used for OpenAUC = (i) × gate AUC; (ii) **end-to-end** accuracy where rejected known flows count as errors. Also report accuracy **conditioned on accepted** known flows. Always label which one a column refers to.
- Macro-F1 over known ∪ {UNKNOWN}: zero-day true label = `UNKNOWN_ID`.

### Error propagation (PR3-T03)
- `decompose(y_true, y_pred, accept, y_stage2)` builds the five exclusive categories E1–E5 from boolean masks; output = counts, rates (overall and per class). Assert `E1+E2+E3+E4+E5 == total` flows.
- Also report Stage-2 accuracy conditioned on accepted vs on all known flows, and **per-class rejection rate across the threshold grid** (per-class sensitivity to the threshold).
- Leakage matrix = `pd.crosstab(zd_names[leaked], stage2_class[leaked])`, including benign.
- Oracle gate = the same Stage-2 predictions with `accept = (y_true != UNKNOWN_ID)`; no new models.
- Prose discussion is written by the agent from the real numbers, never pre-written.

### End-to-end evaluation (PR3-T04)
- One `evaluate_system(name, preds, scores, y_true, zd_names)` used for all six systems: CLOSR-only, Siamese-only (own confidence rejection, threshold from `thr_at_acceptance` on validation), CLAD binary, Cascade gate-only, Cascade joint, Cascade oracle.
- Table columns include a `calibration_source` field. Metrics that do not apply to a system (e.g. closed-set accuracy for binary CLAD) are `nan` with a note, never faked.
- The verdict on "jointly achieves strong open-set and closed-set performance" is written from the mean ± std table and stated plainly, even if negative (AGENTS §5.5).

### Latency (PR3-T05)
- `time.perf_counter()`. Warm-up must include TensorFlow tracing; use fixed input signatures to avoid retracing; force device sync by fetching results to NumPy before stopping the timer. `warmup_runs` discarded, `timed_runs` measured, for each size in `bench_batch_sizes` (includes 1). Percentiles via `np.percentile`; throughput = flows/s.
- Stage-wise breakdown: preprocessing, Stage 1, Stage 2, decision logic. Cost saved by skipping Stage 2 for rejected flows is measured at the actual rejection fraction. Cascade overhead vs CLOSR-only in ms and %.
- Run on **every available device** (CPU always; GPU too if present). Record hardware (`platform.processor()` / TensorFlow device name) in the JSON.

### Ablations (PR3-T06)
(a) scorer, (b) threshold strategy, (c) support-set size (`k_exemplars`), (d) variant A vs B. Optional (e) number of known classes. One small table + one paragraph each, from saved results.

### Visualisation
`viz.py`, matplotlib only, one function per figure type (t-SNE, ROC, confusion, trade-off, latency). t-SNE uses the seeded subsample from the saved arrays. Figures go to `results/figures/` and are deterministic given the seed.

### Reporting
- `make_report.py` reads results + arrays, writes CSV tables and figures, and fills the numeric tables of the AGENTS §10 template.
- Prose sections (deviations, risks, discussion, FINAL_REPORT status labels) are written by the agent from real results.
- A task is DONE only if its report row names a test, result file, or figure as evidence.

---

## D6. "Done means" cheat sheet

| Task | Simplest deliverable | Quick check |
|---|---|---|
| PR1-T01 | Layout (D2), upstream clone, `NOTICE`, `UPSTREAM_INVENTORY.md`, `DEVIATIONS.md` pre-filled from Appendix B | `pip install -e . && pytest -q` |
| PR1-T02 | `config.py` + `default.yaml` | test: `--set squared=false`, `--set residual=false` → `False` |
| PR1-T03 | `data.py` real loader + synthetic generator + splits + `DATA_CARD.md` | test: no index overlap; zero-day absent from train/val |
| PR1-T04 | `models.py`, `losses.py`, unit + upstream-match tests | `pytest tests/unit` |
| PR1-T05 | `train.py`, smoke/resume/reload tests | `python scripts/run.py train --set smoke=true` |
| PR1-T06 | `infer.py`, `metrics.py`, equivalence test | `run.py eval` writes JSON + CSV |
| PR1-T07 | MLP, IF, AE, available open-set baselines; labelled table | `make_report.py` |
| PR1-T08 | Train real data, compare to §2.4; else BLOCKED/PARTIAL + synthetic | result JSONs |
| PR1-T09 | t-SNE figures | files exist; identical on rerun |
| PR1-T10 | Edge-case tests, BUGLOG, coverage for loss/data/metrics/checkpoint | `pytest -x -q --cov=nids` |
| PR1-T11 | `PR1_summary.md` | every task ID present |
| PR2-T01 | Pseudo-zero-day + leakage test | test passes |
| PR2-T02 | Siamese (D5) | smoke-train OK; accuracy reported |
| PR2-T03 | Closed-set table (CLOSR, Siamese, **MLP**), 3 seeds, confusion matrices | CSV |
| PR2-T04 | Open-set table (CLOSR, CLAD, Siamese) | CSV |
| PR2-T05 | Gatekeeper + 3 scorers, scorer decision recorded | monotonicity test |
| PR2-T06 | 3 threshold helpers + ROC plot + thresholds file | acceptance within 0.01 |
| PR2-T07 | t-SNE, ROC, confusion figures | files exist |
| PR2-T08/T09 | Bug-hunt + report | suite green |
| PR3-T01 | `Cascade` | batch == per-flow test |
| PR3-T02 | 3 strategies + trade-off plot | thresholds independent of test/zero-day |
| PR3-T03 | `decompose`, leakage matrix, oracle | E1..E5 sum to total |
| PR3-T04 | Six-system table, 3 seeds | one CSV, mean ± std |
| PR3-T05 | `bench.py`, table, plot | JSON with p50/p95/p99 |
| PR3-T06 | Ablations (a)–(d) | CSV |
| PR3-T07 | `make_report.py` regenerates all artifacts | one command on clean checkout |
| PR3-T08 | `run.py demo`, README quick start, architecture diagram | runs in minutes on CPU |
| PR3-T09 | Full regression, fuzz cases, leakage audit | suite green |
| PR3-T10 | `PR3_summary.md` + `FINAL_REPORT.md` | every abstract sentence has a status |

---

## D7. Testing decisions

- `tests/conftest.py` builds a **tiny synthetic dataset** (e.g. 8 features, 4 known + 1 zero-day class, ~200 rows/class). Tests use small models (hidden `[32,32]`, out 8) via config overrides so training tests take seconds.
- The default test command is `pytest -x -q -m "not real"`.
- Real-data tests are marked `@pytest.mark.real` and skipped if the data file is absent.
- Numerical reference equivalence tests use independent NumPy reference implementations transcribed from upstream source text and hand-computed mathematical cases; no PyTorch is installed or executed.
- Register marker `real` in `pyproject.toml`. Configure `pytest-timeout` in `pyproject.toml` (`timeout = 120`, a 120-second timeout per test) so a hanging test cannot freeze the loop.
- Required edge cases (parametrise where possible): batch size 1, single-class batch, class missing from a batch, empty accepted set, all rejected, all accepted, NaN input (pick one documented behaviour — raise a clear error — and test it).
- `third_party/CLOSR/` is read-only source text for inspection only; it is never imported, never executed, and never inserted into `sys.path`.
- A torch-free guard test in `tests/unit/test_framework.py` asserts that importing all `nids` modules does not load `torch` into `sys.modules`, and that a static AST scan finds no `torch` import across all `.py` files in the repository outside `third_party/`, `.venv`, and `.git`.
- Coverage: report `pytest --cov` for loss, data, metrics, checkpoint modules (PR1-T10); do not chase other numbers.

---

## D8. Bug-fix loop (follows AGENTS §6 exactly)

1. **Automatic test-and-repair loop:**
   - After finishing each task, and again before writing each PR summary report, run: `python -m py_compile` on changed files, `ruff check`, then `pytest -x -q -m "not real"`.
   - On any failure: read the traceback, find the root cause, fix the code, then rerun the WHOLE suite (not only the failing test). Repeat until green.
   - **Cap: 5 fix-iterations per distinct failing test** (AGENTS §6.7). If the same error appears twice with the same fix attempt, stop repeating it and re-diagnose with a different approach (check shapes, dtypes, devices, label alignment, score direction, data leakage).
   - After the cap: mark the test `xfail` with a reason, log it `OPEN` in `docs/BUGLOG.md` with your analysis, and continue with other tasks. Do not block the PR on it.
   - **Forbidden shortcuts:** never delete, skip, or weaken a test, loosen a tolerance, or hard-code expected values just to make it pass. A test may be changed only if you can show the test itself is wrong; log that in `docs/BUGLOG.md` with the evidence.
   - A task may be marked DONE only after the suite is green (or the remaining failures are documented OPEN).
   - If real data exists, also run `pytest -m real` once at the end of each PR.
2. Reproduce the bug as a **failing test first**, then fix the root cause.
3. `BUGLOG.md` row format: `ID | task | symptom | root cause | fix | guarding test | iterations`.
4. Pitfall coverage (AGENTS §12):

| §12 | Resolved by |
|---|---|
| 1 boolean flags | D4.5 `yaml.safe_load` override + test |
| 2 hard-coded 72 | `d_in` from data (D4.4) |
| 3 LR constants | config fields (D4.5) |
| 4 no validation | `val_ratio` default 0.1 (D4.3) |
| 5 label layout | `ClassMap`, zero-day = `-1`, `n_known` from map (D4.1) |
| 6 eval mode / device | one device source (D3.5); `training=False` + no GradientTape in `get_embeddings` |
| 7 score direction | known-ness convention (D4.2) + monotonicity test |
| 8 sampler / tiny splits | batch-size clamp, `drop_last=False` (D5 Trainer) |
| 9 benign = 0 | `BENIGN_ID` constant + test (D4.1) |
| 10 checkpoints | single bundle format (weights + sidecar) (D4.6) |
| 11 leakage | train-only scaler, identifiers dropped, `val_*` inputs (D4.4, D5) |
| 12 metric edge cases | `safe_auroc`, `nan` on undefined (D5 Metrics) |

---

## D9. Priority order if time or compute is limited

**Must:** D2 skeleton → config → data (+ synthetic) → losses/models → trainer → infer/metrics → CLOSR + CLAD results → baselines (MLP, IF, AE, available open-set) → Siamese → gatekeeper + thresholds + pseudo-zero-day → cascade → joint calibration → error propagation → six-system table → latency → ablations (a)–(d) → reports.

**Should:** 3 seeds everywhere, all figures, demo, GPU + CPU benchmark when a GPU exists.

**Stretch (record as "Not done" if skipped):** retraining per held-out class for pseudo-zero-day, ablation (e), baselines that fail after the fix cap, full 200-epoch runs when only CPU is available.

---

## D10. Stop-and-ask triggers (identical to AGENTS §6.8)

1. Dataset unobtainable **and** the synthetic fallback cannot satisfy the task.
2. A licence problem appears.
3. Results contradict the abstract's core claim in a way that needs a scope decision.

Everything else: decide using this file, log it, continue. A failing component after the 5-iteration cap is logged OPEN, not escalated.

---

## D11. After each PR

1. Run pytest (`pytest -x -q -m "not real"`, plus `pytest -m real` if real data exists) and `ruff check .`, executing the automatic repair loop and fixing until green (or remaining failures documented OPEN).
2. `python scripts/make_report.py`
3. Write `docs/reports/PRn_summary.md` with the AGENTS §10 template and real evidence only.
4. Commit, then **stop** and wait for the human's review.

---

## Appendix A — Traceability: AGENTS.md → decisions.md

| AGENTS.md requirement | Covered in | Status |
|---|---|---|
| §0 read order, task IDs, honesty, upstream wins on conflict | D0, D3.6, Preamble | Aligned |
| §0 item 8 (read `decisions.md`, precedence) | Preamble | Aligned (same precedence text) |
| §2.1–2.3 upstream facts and defaults | D4.5, D4.3, D5 | Aligned |
| §2.4 targets and ±0.01 tolerance | D3.8, D6 (PR1-T08) | Aligned; reduced-epoch runs labelled PARTIAL |
| §3 reuse map / DO NOT TAKE | D0.2, D0.5, D5 | Aligned |
| §4 repository layout | D2 | **DV1** — AGENTS §4 carries the flattening rule; tests/docs layout kept and mandatory |
| §5.1 data protocol, zero-day never used for calibration | D4.3, D5 Pseudo-zero-day | Aligned; leakage tests |
| §5.2 seeds, mean ± std, metadata in results | D4.3, D4.7 | Aligned |
| §5.3 CPU, synthetic fallback | D3.5, D3.7, D3.8 | Aligned |
| §5.4 type hints, no magic numbers, logging, lint | D1.5–D1.8, D4.5 | Aligned |
| §5.5 honesty rule | D0.4, D5 End-to-end | Aligned |
| §6 build-test-debug loop, 5-iteration cap, stop triggers | D3, D8, D10 | Aligned |
| §6 automatic test-and-repair loop | D3.3, D3.4, D8, D11 | Aligned |
| §6.5 mixed precision off/on | D1.11 | **DV5** — AGENTS §6.5 carries the note |
| §10 report template | D5 Reporting, D11 | Aligned |
| §12 pitfalls 1–12 | D8 table | Aligned |
| §13 metric definitions | D4.2, D5 Cascade metric definitions | Aligned; cascade OpenAUC and closed-set accuracy forms now also in AGENTS §13 |
| PR1-T01–T11 | D2, D4, D5, D6 | Aligned |
| PR1-T01 deviations file | Appendix B | Aligned; PR1-T01 now requires DV1–DV6 in `docs/DEVIATIONS.md` |
| PR1-T03 validation split, zero-day label | D4.1, D4.3 | **DV4**, **DV6** — pointers in PR1-T03 |
| PR1-T08 reduced-epoch CPU runs | D3.8 | Aligned; pointer in PR1-T08 |
| PR2-T01 pseudo-zero-day (retrain-style) | D5 Pseudo-zero-day | **DV3** — pointer in PR2-T01 |
| PR2-T02 pair/triplet sampling | D5 Siamese | **DV2** (only if upstream has no Siamese) — pointer in PR2-T02 |
| PR2-T03–T09 | D5, D6 | Aligned |
| PR3-T01–T10 | D5, D6 | Aligned |
| §11 final checklist | D6, D11 | Aligned |

## Appendix B — Pre-registered deviations (copy into `docs/DEVIATIONS.md` at PR1-T01)

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
| DV15 | Upstream regenerates splits in memory without on-disk persistence | Save split indices (`train`, `val`, `test`) to compressed NPZ files via `save_split_indices` | Guarantees exact split persistence, reproducibility, and verifiable zero-leakage partitions (§5.1) |
| DV16 | Upstream masks self-pairs with `dist > eps` | We zero the diagonal explicitly (`tf.linalg.set_diag`) before masking | Float32 roundoff on unit vectors can leave self-distance above eps, erroneously counting self-pairs in `n_sim` |
| DV17 | PyTorch AdamW epsilon placement $\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$ | Keras AdamW folds bias-correction into numerator $\frac{\alpha \cdot m}{\sqrt{v} + \epsilon}$ | Built-in Keras optimizer formulation; epsilon set to upstream `1e-8` via Config; expected difference negligible |
| DV18 | AGENTS §2.3 human-readable class names with spaces/caps | Raw Lycos2017 labels are lowercase snake_case (`webattack_sql_injection`, `heartbleed`, etc.) | Direct alignment with raw dataset CSV without synthetic renaming overhead |
| DV19 | Silent fallback to synthetic data when CSV absent | Tri-state `data_source: "auto" \| "real" \| "synthetic"`; prominent warning on fallback | Prevents accidental execution on synthetic data when real data is intended |
| DV20 | Normalization degrees of freedom: upstream `ZNormaliser` uses Bessel's correction (`ddof=1`) | Scikit-learn `StandardScaler` uses population standard deviation (`ddof=0`) | Built-in scikit-learn standard scaling; numerical difference on $N \approx 800\text{k}$ samples is $< 10^{-6}$ |
| DV21 | Contrastive loss execution structure: upstream loops over classes in Python and stacks losses | Vectorized quadratic-form einsum / matmul contrastive loss with identical mathematics | Speed-up of ~5.65x on real-data training step (from ~28s down to ~4.3-5.0s on Apple Silicon CPU) while preserving exact loss parity |
| DV22 | Full test embedding storage: upstream calls `get_features` which executes `T.cat(features, dim=0)` storing all test embeddings in RAM simultaneously (`util/features.py:38`, `eval_closr.py:170-176`) | Streaming chunked evaluation computing similarities and metric statistics on-the-fly without accumulating all embeddings in RAM | For CLOSR with 894,966 test samples, 12 classes, and 64-dim float32 embeddings, upstream's concatenated tensor alone requires $894,966 \times 12 \times 64 \times 4\text{ B} \approx 2.75\text{ GB}$ RAM; streaming bounds memory usage on memory-constrained systems |
| DV23 | t-SNE visualization: upstream repository includes static PNG images in `images/` (`README.md:143,148`) but contains no t-SNE or manifold code scripts | Seeded reservoir subsample of test embeddings capped at `tsne_max_points` (default 5000) projected via scikit-learn TSNE | Provides a reproducible, programmatic pipeline to recreate the README t-SNE figures without excessive memory consumption |
| DV24 | Periodic checkpoint cadence: upstream `train_closr.py:258` and `train_clad.py:255` only call `make_checkpoint` once after the full epoch loop completes | Configurable `checkpoint_every` saving full bundle (weights, optimizer variables, sidecar JSON metadata) during training | Enables mid-run progress inspection, failure recovery, and exact training resumption |
| DV25 | Default validation split ratio: upstream has no validation split (`val_ratio=0.0`) | `default.yaml` sets `val_ratio: 0.0` for exact reproduction of upstream paper results, while PR2/PR3 use `val_ratio: 0.1` carved from train remainder for cascade calibration | Avoids test data leakage during threshold calibration while keeping PR1 baseline reproduction faithful |
| DV26 | Upstream `eval_closr.py:216` malicious recall slicing (`test_sims[mal_mask][:, 1:]`) | WITHDRAWN (NOT A DEVIATION): upstream subtracts 1 from malicious labels (`mal_labels = test_labels[mal_mask] - 1` at line 215) and takes argmax over sliced similarities `[:, 1:]`, so labels and predictions are identically zero-indexed $\{0, \dots, \text{num\_known}-2\}$. Our code matches upstream exactly; ID retained for numbering stability | Reconciled in C1; verified with hand-computed test in `test_metrics.py` |
| DV27 | Upstream `CLOSRLoss.forward` (`closr_loss.py:103-106`) calls `loss_calc` without `squared`, `alpha`, or `eps`, hardcoding execution to defaults (`True`, `0.5`, `1e-6`) | Our `closr_loss_and_frac` explicitly honours configured `squared`, `alpha`, and `eps` arguments | Ensures configurable loss hyperparameters take effect as intended; defaults match upstream so reproduction is unaffected |
| DV28 | Upstream `CLOSRLoss.forward` (`closr_loss.py:99`) sets `self.fraction_positive_pairs = T.tensor(0.0)` as a constant dummy | Our implementation computes and reports the real mean fraction of positive pairs across all classes | Provides accurate diagnostic telemetry during training |

## Appendix C — Where AGENTS.md points to this file

AGENTS.md references `decisions.md` in: §0 item 8, §4 (flattening rule), §5.2 (seed protocol), §6 (automatic test-and-repair loop), §6.5 (mixed precision), PR1-T01 (copy DV1–DV6), PR1-T03 (DV4, DV6), PR1-T08 (reduced runs), PR2-T01 (DV3), PR2-T02 (DV2), PR3-T02 (X/Y defaults), PR3-T04 (metric definitions), §10 item 7 (report deviations), and §11 (final checklist, pytest repair loop). If a section number in AGENTS.md changes, update the references in this file too.

*End of decisions.md.*