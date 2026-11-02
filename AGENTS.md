# AGENTS.md — CLAD / CLOSR Zero-Day NIDS with a CLOSR→Siamese Cascade

> This file is the single source of truth for the coding agent. Read it fully before touching any file.
> Work is split into **PR1, PR2, PR3**. After each PR the agent MUST produce a **summary report** (template in Section 10) so the human can tick every item below against what was actually delivered.

---

## 0. How the agent must use this file

1. Read Sections 1–6 first. They apply to all three PRs.
2. Implement PRs strictly in order: PR1 → PR2 → PR3. Do not start a PR until the previous PR's report is written and its acceptance criteria are met.
3. Every task has an ID (e.g. `PR1-T04`). Use these IDs in commit messages, test names, and in the summary report so the human can match them one-to-one.
4. Follow the **build → test → debug → fix → retest** loop in Section 6 for every task. A task is not done until its tests pass and its acceptance criteria are verified.
5. Never fabricate results. If a number cannot be reproduced, report the real number and the gap.
6. If something in this file conflicts with what the upstream repo actually contains, trust the upstream repo, record the discrepancy in `docs/DEVIATIONS.md`, and continue.
7. The human asked for this file to contain no code. The agent writes all code itself, guided by the specifications here.

---

## 1. Project summary (from the project abstract)

**Problem.** Supervised ML-based Network Intrusion Detection Systems (NIDS) are strong on known attack classes but degrade sharply on zero-day attacks. Anomaly detectors generalise to unseen attacks but have high false-positive rates.

**Methods reproduced (PR1).**
- **CLAD** — Contrastive Learning for Anomaly Detection. Models *benign* traffic as a von Mises–Fisher (vMF) distribution in embedding space using an *asymmetric* contrastive loss. Trained on both benign and malicious samples without the closed-world assumption. Zero-day attacks appear as embeddings anomalously distant from the benign centroid.
- **CLOSR** — Contrastive Learning for Open-Set Recognition. Multiclass extension: each known class (benign + each known attack) is modelled as its own vMF distribution in its own embedded space. Zero-day attacks appear as embeddings far from, or orthogonal to, all known-class centroids.
- Evaluate with **AUROC** (known-attack and zero-day detection) and **OpenAUC**, against anomaly-detection, supervised, and open-set-recognition baselines.

**Limitation identified.** CLOSR gives state-of-the-art open-set (zero-day) detection but its closed-set accuracy on known attack types is slightly below dedicated closed-set classifiers such as Siamese networks.

**Contribution of this project (PR2 + PR3).** A two-stage **CLOSR + Siamese cascade**:
1. **Stage 1 — CLOSR as zero-day gatekeeper:** rejects traffic estimated to come from unseen classes.
2. **Stage 2 — Siamese network:** fine-grained classification of the traffic that Stage 1 accepts as known.

This combination is suggested in the literature but has **not been empirically validated as an integrated system**. We must therefore:
- implement the cascade explicitly,
- evaluate **error propagation** across the CLOSR–Siamese boundary,
- **jointly calibrate** the rejection threshold,
- measure **end-to-end latency** and detection performance.

---

## 2. Reference repository (STRICT reference)

- **Upstream:** https://github.com/jackwilkie/CLOSR (Apache-2.0; paper: "A Novel Contrastive Loss for Zero-Day Network Intrusion Detection", IEEE TNSM 2026, DOI 10.1109/TNSM.2026.3652529).
- Clone it into `third_party/CLOSR/` and treat it as **read-only reference**. Our own code lives in our own package and is *adapted* from it, never edited in place there.
- Preserve the Apache-2.0 license header/attribution on every adapted file and add a top-level `NOTICE` file crediting the upstream authors.

### 2.1 What is verified to exist upstream (from the README and source files)

| Upstream path | What it is |
|---|---|
| `losses/clad_loss.py` | `CLADLoss` — takes L2-normalised features `[bsz, f_dim]` and labels `[bsz]`, with a margin `m`. |
| `losses/closr_loss.py` | `CLOSRLoss` — takes L2-normalised features `[bsz, n_classes, f_dim]` and labels `[bsz]`, with margin `m`, `n_classes`, `squared`. Exposes a `get_fraction_pos` diagnostic. |
| `model/model.py` | `CLOSRMLP` (MLP backbone producing one embedding per class); a CLAD model class also lives here (verify exact name). |
| `data/load_data.py` | `get_data(...)` — loads CSV, drops identifier columns, treats classes with fewer than `sample_thres` samples as zero-day, returns train / val / test / zero-day splits. |
| `data/loaders.py` | `tabular_dl(...)` — dataloader with optional class-balanced sampling. |
| `util/schedules.py` | `WarmupCosineSchedule`, `LRSchedule`. |
| `util/meter.py` | `AverageMeter`. |
| `util/checkpoint.py` | `make_checkpoint`, `load_checkpoint`. |
| `util/features.py` | `get_features` — chunked embedding extraction. |
| `util/distance.py` | `get_centroids`, `calc_class_sims_chunked`. |
| `util/metrics.py` | `mean_auroc`, `fpr_at_recall`, `classwise_fpr_at_recall`, `compute_mean_ap_and_pr_auc`. |
| `train_clad.py`, `train_closr.py` | Training entry points; save to `weights/clad.pt.tar` and `weights/closr.pt.tar`. |
| `eval_clad.py`, `eval_closr.py` | Evaluation entry points; print metrics. |
| `baseline_implementations/` | Baseline models used in the paper's comparison tables (the README tables list: DUAD, DAE-LR, Deep SVDD, AE, SVM, AutoSVM, IF, RENOIR, MLP, Siamese; and for open-set: MultiStage, DOC, OpenMax, CRSOR, Siamese Network). |
| `data/`, `weights/`, `images/` | Data location (`data/lycos.csv`), saved weights, figures (t-SNE, architecture). |

### 2.2 What is NOT verified (agent must check by listing the cloned repo)

The agent could not open sub-folder listings when this file was written. Before PR1 coding starts, the agent must list `third_party/CLOSR/baseline_implementations/`, `model/`, `util/`, and `data/` and record the actual file inventory in `docs/UPSTREAM_INVENTORY.md`. In particular, confirm:
- exact class names in `model/model.py` (CLAD model),
- whether a **Siamese** implementation exists inside `baseline_implementations/` (the README's results tables include a Siamese network, so very likely yes),
- which baselines are present as runnable code versus only reported numbers.

### 2.3 Facts taken from the upstream training/eval scripts (use as reference defaults)

- Dataset: **Lycos2017**, expected at `data/lycos.csv`. Original CSVs from https://lycos-ids.univ-lemans.fr, or the preprocessed copy linked in the upstream README.
- Dropped identifier columns: flow_id, src_addr, src_port, dst_addr, dst_port, ip_prot, timestamp. Target column: `label`. Benign class name: `benign` (class index 0).
- Input dimension: **72** (hard-coded upstream). Number of classes: **12** (benign + 11 known attacks).
- Zero-day definition: classes with fewer than **100** samples (`sample_thres`) are held out as zero-day. Per the README's result tables, the zero-day classes are **Heartbleed** and **Web Attack (SQL Injection)**. Verify against the data.
- Split: seed `39058032`, `test_ratio=0.5`, `val_ratio=0.0` (upstream has **no validation split** — PR2 adds one).
- Model defaults: output dim 64, hidden layers 1024-1024-1024, dropout 0.1, no residual connections.
- Loss defaults: margin 1.0, squared distance enabled.
- Optimiser defaults: AdamW, batch size 4096, weight decay ≈ 0.0403709, peak LR 1e-5 with warm-up (epochs/10) then cosine decay, 200 epochs, class-balanced sampling, drop_last.
- CLOSR inference: build class centroids from training embeddings (L2-normalised), compute cosine similarity of each test embedding to each centroid; closed-set prediction = argmax; **open-set score = sum over classes of (similarity² × softmax(similarity))**; lower score ⇒ more likely zero-day.
- Metrics produced by `eval_closr.py`: closed-set accuracy, open-set AUC, **OpenAUC = closed-set accuracy × open-set AUC**, FPR@95%-recall (overall and class-wise mean), mean recall, malicious-only recall, macro-F1 (overall and malicious-only), closed-set AUROC variants, benign false-positive rate, mean average precision, mean PR-AUC.

### 2.4 Reproduction targets (from upstream README)

| Model | Closed-set Acc | Open-set AUC | OpenAUC |
|---|---|---|---|
| CLOSR | 0.995276 | 0.974022 | 0.969420 |
| Siamese Network | 0.997722 | 0.720811 | 0.719167 |
| MultiStage | 0.996612 | 0.801251 | 0.798537 |
| DOC | 0.995536 | 0.570263 | 0.567717 |
| OpenMax | 0.995615 | 0.720174 | 0.717016 |
| CRSOR | 0.994940 | 0.748295 | 0.744509 |

CLAD (AUROC): closed-set mean **0.999855**, open-set mean **0.996627** (Heartbleed 0.995557, SQL Injection 0.997696).
Acceptable reproduction tolerance: within **±0.01** on accuracy/AUROC-type metrics (hardware and seed variation). Anything outside must be explained in the PR1 report.

---

## 3. Reuse map — what to take from the existing project

**KEEP / ADAPT** means: copy into our package with attribution, then refactor for our structure.

| Our need (abstract item) | Take from upstream | Action |
|---|---|---|
| CLAD binary objective (asymmetric contrastive loss, vMF benign model) | `losses/clad_loss.py` | KEEP; add unit tests and docstrings. |
| CLOSR multiclass/open-set objective | `losses/closr_loss.py` | KEEP; add unit tests. |
| Backbone / embedding model | `model/model.py` (`CLOSRMLP`, CLAD model) | ADAPT: make `d_in` configurable (upstream hard-codes 72). |
| Data pipeline + zero-day hold-out | `data/load_data.py`, `data/loaders.py` | ADAPT: add a proper validation split (needed for threshold calibration), return named structures instead of 8-tuples. |
| LR schedule, meters, checkpoints | `util/schedules.py`, `util/meter.py`, `util/checkpoint.py` | KEEP; extend checkpoints to store config, class map, seed, and centroids. |
| Embedding extraction, centroids, similarity | `util/features.py`, `util/distance.py` | KEEP; these are the core of the Stage-1 gatekeeper. |
| Metrics (AUROC, OpenAUC, FPR@recall, etc.) | `util/metrics.py` + logic inside `eval_clad.py`/`eval_closr.py` | ADAPT: move metric logic out of the eval scripts into a reusable `metrics` module. |
| Training scripts | `train_clad.py`, `train_closr.py` | ADAPT: unify into one trainer driven by config; fix argparse issues (Section 12). |
| Evaluation scripts | `eval_clad.py`, `eval_closr.py` | ADAPT: unify; write results to JSON/CSV. |
| **Siamese classifier (Stage 2)** | `baseline_implementations/` (Siamese) — verify | ADAPT if present; otherwise implement new (PR2-T02). |
| Comparison baselines (MLP, AE, IF, SVM, DUAD, …) | `baseline_implementations/` | ADAPT the subset that exists as code; others are reported as "numbers from upstream paper", clearly labelled. |
| t-SNE figures | `images/` + any plotting util | Reference only; regenerate our own figures. |

**DO NOT TAKE:** upstream `weights/` binaries as evidence of our results (we must train ourselves), `.gitignore` rules that would hide our results folder, anything from the upstream `data/` that is not licensed for redistribution.

**NEW (not upstream — we build these):** validation protocol and pseudo-zero-day calibration, Siamese Stage-2 integration, the cascade pipeline, joint threshold calibration, error-propagation analysis, latency benchmark, end-to-end report generator, test suite, summary-report automation.

---

## 4. Target repository layout

Create this layout (names may be adjusted only if recorded in `docs/DEVIATIONS.md`):

- `AGENTS.md` — this file
- `NOTICE`, `LICENSE` (Apache-2.0), `README.md`
- `third_party/CLOSR/` — upstream clone, read-only
- `src/nids/` — our package
  - `config/` — dataclass/YAML configs; one default config per experiment
  - `data/` — loading, preprocessing, splits, dataloaders, synthetic data generator
  - `models/` — CLAD model, CLOSR model, Siamese network, shared MLP blocks
  - `losses/` — CLAD loss, CLOSR loss, Siamese loss
  - `training/` — trainer, schedules, meters, checkpointing
  - `inference/` — embedding extraction, centroids, scorers, gatekeeper, Siamese classifier, cascade
  - `evaluation/` — metrics, per-class tables, error-propagation analysis, calibration, latency benchmark
  - `viz/` — t-SNE, ROC, Pareto, confusion-matrix plots
- `scripts/` — thin CLI entry points: train, evaluate, calibrate, benchmark, run-demo, make-report
- `configs/` — YAML experiment configs
- `tests/` — `unit/`, `integration/`, `regression/`
- `results/` — metrics JSON/CSV, figures, per-seed outputs (committed in small form; large artifacts ignored)
- `docs/` — `UPSTREAM_INVENTORY.md`, `DEVIATIONS.md`, `BUGLOG.md`, `DATA_CARD.md`, `reports/PR1_summary.md`, `reports/PR2_summary.md`, `reports/PR3_summary.md`, `reports/FINAL_REPORT.md`

---

## 5. Global conventions (apply to every PR)

**5.1 Data protocol (critical for scientific validity)**
- Three populations: **train** (known classes), **validation** (known classes only), **test** (known classes) and **zero-day** (held-out classes, never seen in training or calibration).
- The zero-day classes must **never** influence training, centroids, hyperparameters, or threshold selection. Thresholds are chosen on validation (known-only, or pseudo-zero-day built by leaving one known attack out) and only *reported* on test + zero-day.
- Any change to the split must keep the upstream default seed reproducible, and every split must be saved to disk (indices or hash) so runs are comparable.

**5.2 Reproducibility**
- Seed Python, NumPy, and PyTorch everywhere; record seeds in every result file.
- Main comparisons use **at least 3 seeds**, reporting mean ± std. Single-seed numbers must be labelled as such.
- Record library versions, device, and git commit hash in every result file.

**5.3 Environment**
- Python 3 + PyTorch (+ scikit-learn, NumPy, pandas, matplotlib). Pin versions in `requirements.txt` starting from upstream's file.
- Device auto-selection: use GPU if available, otherwise CPU. Every script must run on CPU (slowly) so tests work anywhere.
- **If the Lycos2017 dataset cannot be downloaded in the agent's environment**, do NOT stop: build the synthetic data generator (72 features, 12 classes + 2 zero-day classes, matching the schema) and complete all code and tests on it; mark real-data tasks as "BLOCKED: dataset" in the report and tell the human exactly which file to place where.

**5.4 Quality**
- Type hints and docstrings on public functions; no hard-coded paths or magic numbers (use configs).
- Configs are the only place for hyperparameters. Defaults mirror Section 2.3.
- Logging via a standard logger (not bare prints); metrics saved as machine-readable files.
- Lint/format with a standard tool (ruff/black or equivalent).

**5.5 Honesty rule**
- Report results exactly as measured. If the cascade does **not** beat CLOSR-only or Siamese-only on some metric, say so plainly — that is a valid finding. No tuning on test data, no cherry-picked seeds.

---

## 6. Agent working protocol: build → test → find bugs → fix

For **every task**:

1. **Plan** — restate the task ID, inputs, outputs, and acceptance criteria in one short paragraph (kept in the PR report).
2. **Implement** the smallest working version.
3. **Write tests first or alongside**: unit tests for logic, a smoke test on synthetic data, and where relevant a regression test on real data.
4. **Run the full relevant test suite**. On failure: diagnose the root cause (do not patch symptoms), fix, rerun.
5. **Hunt bugs deliberately** (do this even when tests pass):
   - run with tiny inputs, batch size 1, empty class, single-class batch, NaN/Inf inputs, CPU-only, and mixed precision off/on;
   - check shapes, dtype, device placement, label-index alignment, and train/eval mode;
   - compare against the upstream implementation numerically on a fixed small input (loss values, centroid values, scores must match within a small tolerance);
   - check for data leakage between splits.
6. **Log every bug** in `docs/BUGLOG.md`: ID, symptom, root cause, fix, test that now guards it, task ID.
7. **Iterate** until all tests pass and acceptance criteria hold. Cap at 5 fix-iterations per bug; if still failing, document it as "OPEN" in the report with the analysis, and continue with other tasks.
8. **Stop and ask the human** only if: the dataset is unobtainable AND no synthetic fallback can satisfy the task, a licence problem appears, or results contradict the abstract's core claim in a way that needs a scope decision.

**Definition of Done for a PR:** all tasks implemented, all tests green (or documented OPEN with justification), reproducibility seeds recorded, and the PR summary report written.

---

# PR1 — Foundation + Reproduction of CLAD and CLOSR (+ baselines)

**Goal:** a clean, tested, reproducible implementation of CLAD and CLOSR with an evaluation harness that reproduces the upstream numbers within tolerance. This covers the first paragraph of the abstract.

| ID | Task | Details | Acceptance criteria |
|---|---|---|---|
| PR1-T01 | Scaffolding & attribution | Create repo layout (Section 4), clone upstream into `third_party/`, pin `requirements.txt`, add `NOTICE`, license headers, `docs/UPSTREAM_INVENTORY.md` (Section 2.2). | Fresh clone + install works; inventory file lists every upstream file inspected. |
| PR1-T02 | Config system | One config object per experiment replacing the duplicated argparse blocks in upstream train/eval scripts; CLI overrides allowed. | Same hyperparameter defaults as Section 2.3; boolean flags (residual, squared) parse correctly from CLI (see Section 12, item 1). |
| PR1-T03 | Data pipeline | Adapt `get_data` + `tabular_dl`; load Lycos2017; drop identifier columns; encode labels with benign = 0; hold out classes with < 100 samples as zero-day; class-balanced training sampler; **add validation split** support; save split indices; build the synthetic data generator. Write `docs/DATA_CARD.md` (class counts per split, zero-day classes, feature count, preprocessing/scaling). | Class counts reported; zero-day classes absent from train/val; no index overlap between splits (tested); synthetic generator reproduces schema. |
| PR1-T04 | Models & losses | Port CLAD model/loss and CLOSR model/loss. Make `d_in` configurable. | Unit tests: output shapes; embeddings unit-norm after normalisation; loss finite and non-negative; gradients flow to all parameters; loss decreases on a toy separable problem over a few steps; margin and `squared` flag change the loss as expected; asymmetric behaviour (benign vs malicious treated differently) is demonstrated by a test; numerical match with upstream loss on fixed inputs. |
| PR1-T05 | Unified trainer | One trainer supporting CLAD and CLOSR: AdamW, warm-up + cosine schedule, balanced sampler, meters, periodic logging, checkpointing (weights + optimiser + schedule + config + class map + seed). Support resume and a `--smoke` mode (few batches). | Smoke training runs end-to-end on synthetic data on CPU; resume reproduces identical next-step loss; checkpoint reload restores identical outputs. |
| PR1-T06 | Evaluation harness | Port embedding extraction, centroid computation, similarity, open-set score, and every metric listed in Section 2.3 into a reusable `metrics` module. CLAD evaluation: per-attack-class AUROC table split into closed-set and open-set parts, with means. CLOSR evaluation: closed-set accuracy, open-set AUC, OpenAUC, FPR@95, F1, recall, benign FPR, mAP, PR-AUC. Output JSON + CSV. | Metrics match upstream eval scripts on identical inputs (tested); per-class tables generated; results file includes seed, commit hash, config. |
| PR1-T07 | Baselines | Inventory `baseline_implementations/`. Port and run every baseline that exists as code. **Minimum:** MLP (closed-set supervised), Isolation Forest, Autoencoder, and the open-set baselines available upstream. All remaining baselines from the README tables are listed as "reported upstream numbers, not re-run", clearly labelled. | A comparison table CLAD/CLOSR vs. baselines produced with the same splits and metrics; each row marked "re-run" or "upstream-reported". |
| PR1-T08 | Reproduction check | Train CLAD and CLOSR with default config on real data (≥1 seed; 3 seeds preferred) and compare with Section 2.4. | Metrics within ±0.01 of targets, or a written explanation of every gap in the report. If dataset blocked, mark BLOCKED and show synthetic-data results instead. |
| PR1-T09 | Visualisation | t-SNE of CLAD embeddings and CLOSR embeddings (benign vs. malicious vs. zero-day), saved to `results/figures/`. | Figures generated by a script, deterministic given seed. |
| PR1-T10 | Test & bug-hunt pass | Run the Section 6 loop across the whole PR1 code; build `tests/unit`, `tests/integration`, and a numeric regression test against upstream. | All tests green; `docs/BUGLOG.md` populated; coverage of loss, data, metrics, checkpoint modules reported. |
| PR1-T11 | **PR1 summary report** | `docs/reports/PR1_summary.md` using Section 10 template. | Every PR1 task ID has a status and evidence. |

---

# PR2 — Closed-set vs. Open-set analysis, Siamese classifier, gatekeeper tooling

**Goal:** quantify the identified limitation (CLOSR's weaker closed-set accuracy vs Siamese) and prepare both halves of the cascade as independent, calibrated, well-tested components.

| ID | Task | Details | Acceptance criteria |
|---|---|---|---|
| PR2-T01 | Validation & calibration protocol | Finalise train/val/test/zero-day protocol (Section 5.1). Implement **leave-one-attack-out pseudo-zero-day** validation: for calibration, temporarily treat one known attack class as unseen to simulate zero-day behaviour without touching real zero-day classes. | Test proving real zero-day samples are never used in any calibration function; pseudo-zero-day option documented. |
| PR2-T02 | Siamese classifier (Stage 2) | Reuse the upstream Siamese baseline if it exists; otherwise implement: shared encoder, pair or triplet sampling with class balancing, a contrastive/pair loss, training loop, checkpointing. Inference by comparing a flow's embedding with a **support set** (class prototypes or exemplars from training data) and returning class + confidence. Trains on known classes only (benign + known attacks). | Smoke-trains on synthetic data; reaches closed-set accuracy near the upstream Siamese figure (~0.9977) on real data; support-set construction is deterministic and saved with the checkpoint. |
| PR2-T03 | Closed-set comparison CLOSR vs Siamese | Same splits, ≥3 seeds. Report accuracy, macro-F1, per-class recall/precision/F1, malicious-only metrics, confusion matrices, mean ± std. Also include the MLP baseline. | Table + confusion matrices showing the size of CLOSR's closed-set gap (or lack of one) with statistical spread. A short written conclusion on whether the abstract's claim holds in our runs. |
| PR2-T04 | Open-set comparison | Same protocol for zero-day detection: open-set AUC, OpenAUC, FPR@95, per-zero-day-class AUROC for CLOSR, CLAD, and Siamese (using a confidence-based score for Siamese). | Table showing CLOSR's open-set advantage; Siamese open-set numbers close to the upstream ~0.72 (document if not). |
| PR2-T05 | Gatekeeper API | A Stage-1 component wrapping a trained CLOSR: returns the open-set score, a closed-set prediction, and an accept/reject decision for a given threshold; supports single-flow and batch input. Pluggable scoring functions: (a) upstream score (similarity² weighted by softmax), (b) max cosine similarity, (c) max softmax. | All three scorers evaluated and compared on validation + test; default = upstream score unless another is clearly better on validation (decision recorded). |
| PR2-T06 | Threshold tooling | Utilities to pick the rejection threshold on **validation**: by target known-class acceptance rate (e.g. 95 %, 99 %), by equal error rate, and by maximising an F-score with pseudo-zero-day. Produce ROC (known vs. unknown) with the chosen operating points marked. | Chosen thresholds saved to file; test checks that the realised known-acceptance on validation is close to the target. |
| PR2-T07 | Embedding comparison figures | t-SNE for CLOSR vs Siamese embeddings; ROC curves for zero-day detection; confusion matrices. | Figures saved; scripts reproducible. |
| PR2-T08 | Test & bug-hunt pass | Section 6 loop for all PR2 code. Include tests for: Siamese pair sampling (no pair mixes labels incorrectly), support-set determinism, scorer monotonicity, threshold tool correctness. | All green; BUGLOG updated. |
| PR2-T09 | **PR2 summary report** | `docs/reports/PR2_summary.md` (Section 10 template). | Every PR2 task ID has status and evidence; includes the closed-set and open-set comparison tables. |

---

# PR3 — CLOSR→Siamese cascade, joint calibration, error propagation, latency, final deliverables

**Goal:** implement and evaluate the integrated system described as future work in the abstract.

| ID | Task | Details | Acceptance criteria |
|---|---|---|---|
| PR3-T01 | Cascade pipeline | `Cascade` object: Stage 1 (CLOSR gatekeeper) scores each flow → if accepted, Stage 2 (Siamese) assigns one of the known classes (benign + known attacks); if rejected, output **UNKNOWN / zero-day suspected**. Final label space = known classes ∪ {UNKNOWN}. Batch and single-flow APIs. Configurable variants: (A) default per abstract — accepted flows all go to the Siamese; (B) optional — if CLOSR itself is confident the flow is benign, skip Siamese (documented as an extension, off by default). | Cascade runs end-to-end on synthetic and real data; variant A is the default; unit tests cover accept path, reject path, batch consistency (batch result equals per-flow result), and deterministic output. |
| PR3-T02 | Joint threshold calibration | Calibrate the CLOSR rejection threshold **jointly** with end-to-end performance (not just the gate in isolation). Search over threshold values (and optionally a Siamese-confidence secondary threshold) on validation (+ pseudo-zero-day) to optimise end-to-end objectives; compare strategies: (1) gate-only calibration from PR2-T06, (2) joint end-to-end calibration, (3) constrained version (max zero-day recall subject to ≤ X % known-traffic rejection and ≤ Y % benign false positives). Plot the trade-off (Pareto/operating curve). | Chosen thresholds come only from validation data (tested); report shows joint calibration vs. gate-only calibration on test + zero-day; trade-off plot produced. |
| PR3-T03 | Error-propagation analysis | Decompose end-to-end outcomes into exclusive categories: (E1) known flow wrongly **rejected** by Stage 1; (E2) zero-day flow wrongly **accepted** by Stage 1; (E3) known flow accepted but **misclassified** by Stage 2; (E4) known flow accepted and classified correctly; (E5) zero-day correctly rejected. Report: counts and rates overall and per class; Stage-2 accuracy **conditioned on accepted flows** versus on all flows; **where leaked zero-day flows land** (which known class the Siamese assigns them to, including benign — a security-critical case); per-class sensitivity to threshold. Compare against **oracle gate** (perfect Stage 1) to isolate how much end-to-end loss is caused by Stage 1 mistakes. | Complete decomposition table; leakage matrix (zero-day class × assigned known class); oracle-gate comparison; a written discussion of how errors propagate across the CLOSR–Siamese boundary. |
| PR3-T04 | End-to-end evaluation | Systems compared on identical splits and ≥3 seeds: **CLOSR-only**, **Siamese-only (with its own confidence rejection)**, **CLAD (binary)**, **Cascade (gate-only calibrated)**, **Cascade (jointly calibrated)**, and **Cascade with oracle gate**. Metrics: closed-set accuracy on known flows, accuracy on accepted known flows, open-set AUC, **OpenAUC**, per-class AUROC, FPR@95, macro-F1 over known ∪ {UNKNOWN}, zero-day detection rate, benign false-positive rate, known-flow false-rejection rate. | One consolidated results table (mean ± std) and plots; each row's calibration data source stated; the abstract's claim "jointly achieves strong open-set and closed-set performance" is explicitly supported or refuted by the numbers. |
| PR3-T05 | Latency & throughput benchmark | Measure per-flow and batched latency for: CLOSR-only, Siamese-only, cascade (stage-wise breakdown: preprocessing, Stage 1, Stage 2, decision logic). Report p50/p95/p99 latency, throughput (flows/s), GPU and CPU if available, with warm-up runs excluded, proper device synchronisation, several batch sizes (including 1), and repeated runs. Also report the cost of Stage 2 being skipped for rejected flows. | Benchmark script + results table + plot; includes hardware description; overhead of the cascade over CLOSR-only quantified in absolute (ms) and relative (%) terms. |
| PR3-T06 | Ablations | (a) Scorer choice (PR2-T05), (b) threshold strategy, (c) support-set size for the Siamese, (d) variant A vs. B, (e) effect of number of known classes at the gate (optional stretch). | Small ablation table; findings summarised in a paragraph each. |
| PR3-T07 | Results aggregation & figures | Script that regenerates **every** table and figure for the final report from saved result files in one command. | Running it on a clean checkout with saved results reproduces all report artifacts. |
| PR3-T08 | Demo for the mentor | A `run-demo` entry point: takes a sample of flows (known + zero-day), prints for each flow the Stage-1 score, accept/reject decision, Stage-2 class (if any), and ground truth; prints a final confusion summary and latency. Plus a concise README "Quick start" and a one-page architecture diagram (CLOSR gate → Siamese). | Demo runs in a few minutes on CPU with saved weights or the synthetic data; output is readable by a non-expert. |
| PR3-T09 | Final test & bug-bash | Full regression: re-run PR1 and PR2 regression tests; end-to-end integration test (train tiny models on synthetic data → calibrate → cascade → evaluate → benchmark); fuzz/edge cases (batch size 1, all flows rejected, all accepted, NaN input, class absent from a batch); leakage audit (re-verify no zero-day data in calibration). | All tests green; `docs/BUGLOG.md` final; leakage audit documented. |
| PR3-T10 | **PR3 summary report + final report** | `docs/reports/PR3_summary.md` (Section 10 template) **and** `docs/reports/FINAL_REPORT.md` mapping each sentence of the abstract to the evidence (file/table/figure) that supports it, plus limitations and future work. | Every abstract claim is marked Supported / Partially supported / Not supported / Not tested with a pointer to evidence. |

---

## 10. MANDATORY summary report template (generate after each PR)

Each report (`docs/reports/PRn_summary.md`) must contain these sections, in this order, so the human can compare it against this file:

1. **Header** — PR number, date, git commit hash, branch, environment (OS, Python, PyTorch, device), dataset used (real / synthetic / mixed), seeds.
2. **Task completion matrix** — a table with one row per task ID from this file: `Task ID | Title | Status (DONE / PARTIAL / BLOCKED / NOT DONE) | Files created or changed | Evidence (test name, result file, figure) | Notes`.
3. **Acceptance-criteria checklist** — each acceptance criterion from this file, ticked or unticked, with the verification method.
4. **Results** — the key numbers/tables produced in this PR, with mean ± std where applicable, and the comparison with the Section 2.4 targets (PR1) or with previous PRs (PR2, PR3).
5. **Tests** — number of tests, pass/fail counts, coverage summary, and the command used to run them.
6. **Bugs found and fixed** — summary of `docs/BUGLOG.md` entries from this PR (ID, symptom, root cause, fix, guarding test) and any **OPEN** bugs.
7. **Deviations from AGENTS.md** — anything done differently from this file and why (also logged in `docs/DEVIATIONS.md`).
8. **Reuse log** — which upstream files/functions were reused, adapted, or skipped, and why.
9. **Risks / limitations / open questions** for the human.
10. **How to reproduce** — the exact ordered list of commands (install → data → train → evaluate → report).
11. **Next PR readiness** — one paragraph stating whether the next PR's prerequisites are met.

The report must be factual: no task may be marked DONE without evidence listed in section 2.

---

## 11. Final acceptance checklist (mentor-ready)

- [ ] CLAD and CLOSR reproduced within tolerance of upstream numbers (or gaps explained).
- [ ] Comparison against anomaly-detection, supervised, and open-set baselines, each row labelled re-run vs upstream-reported.
- [ ] CLOSR vs Siamese closed-set gap quantified with ≥3 seeds.
- [ ] CLOSR → Siamese cascade implemented, with UNKNOWN as an explicit output.
- [ ] Rejection threshold calibrated jointly, using only validation / pseudo-zero-day data.
- [ ] Error-propagation decomposition, leakage matrix, and oracle-gate comparison delivered.
- [ ] End-to-end latency and throughput measured with stage-wise breakdown.
- [ ] One command regenerates all tables and figures.
- [ ] Demo script runs and is explainable in a few minutes.
- [ ] Three PR summary reports plus the final report exist and map to the abstract.
- [ ] Test suite green; BUGLOG complete; leakage audit passed.
- [ ] Attribution to upstream authors and Apache-2.0 compliance in place.

---

## 12. Known pitfalls and suspected issues in the upstream code (check and fix in our port)

The agent must verify each of these, and if confirmed, fix it in our code and log it in `docs/BUGLOG.md`:

1. **Boolean CLI flags:** upstream declares `--residual` and `--squared` with a plain boolean type in argparse. In Python this treats any non-empty string (including "False") as true, so these flags cannot be switched off from the command line. Use proper flag handling.
2. **Hard-coded input dimension (72)** in both train and eval scripts — must come from the data/config.
3. **Learning-rate constants:** the training script hard-codes the optimiser's initial LR (1e-6) and the schedule's start/end values while exposing only the peak LR — expose them in the config and document.
4. **No validation split** (`val_ratio=0.0`) upstream — thresholds and early decisions need one; do not tune on test.
5. **Label-layout assumptions in `eval_closr`:** the number of known classes is inferred from the unique labels in the test set, and zero-day labels are assumed to come after all known labels. If a known class is missing from the test split, metrics silently break — add explicit class-map handling and assertions.
6. **Eval/train mode and device handling:** confirm the model is in eval mode with gradients disabled during inference, and that CPU-only runs work (upstream defaults to CUDA).
7. **Open-set score direction:** upstream treats a *lower* score as more likely zero-day (negated before ROC computation). Keep the direction consistent across all scorers in our gatekeeper and test monotonicity.
8. **Class-balanced sampler + `drop_last`:** confirm that rare known classes are actually sampled and that tiny splits (tests) do not crash when a class is smaller than the batch.
9. **Benign = class 0 assumption** is used in several metrics (benign FPR, malicious-only recall). Centralise this assumption and test it.
10. **Checkpoints** upstream store model/optimiser/schedule only — centroids, class map, config, and seed are missing; the cascade needs them to be reproducible.
11. **Data leakage risks:** identifiers dropped (flow ID, addresses, ports, timestamp) must stay dropped; scaling statistics must be fit on train only.
12. **Metric edge cases:** AUROC with a single class present, empty zero-day set, division by zero in per-class recall — handle explicitly.

---

## 13. Metric glossary (so reports use consistent definitions)

- **Closed-set accuracy:** accuracy on known-class test flows only.
- **Open-set AUC:** AUROC of known-vs-zero-day separation using the open-set score.
- **OpenAUC (as used upstream):** closed-set accuracy × open-set AUC.
- **FPR@95:** false-positive rate when 95 % of the positive (zero-day) class is detected (match upstream direction convention and state it).
- **Benign false-positive rate:** fraction of benign flows not labelled benign.
- **Known false-rejection rate (cascade):** fraction of known flows rejected by Stage 1.
- **Zero-day detection rate (cascade):** fraction of zero-day flows labelled UNKNOWN.
- **Zero-day leakage (cascade):** fraction of zero-day flows accepted by Stage 1 and assigned a known label (especially benign).
- **Macro-F1 over known ∪ {UNKNOWN}:** end-to-end F1 treating UNKNOWN as an extra predicted class, with zero-day flows' true label = UNKNOWN.
- **Oracle-gate cascade:** a cascade in which Stage 1 is replaced by ground-truth known/unknown knowledge, giving the upper bound for Stage 2 performance.
- **Latency percentiles:** p50 / p95 / p99 of per-flow wall-clock time, excluding warm-up.

---

*End of AGENTS.md. When finished with a PR, write its summary report, then stop and wait for the human to review before beginning the next PR.*
