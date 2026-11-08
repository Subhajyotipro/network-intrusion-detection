# Upstream Inventory: jackwilkie/CLOSR

- **Upstream Repository:** https://github.com/jackwilkie/CLOSR
- **Paper Reference:** Jack Wilkie et al., *"A Novel Contrastive Loss for Zero-Day Network Intrusion Detection"*, IEEE Transactions on Network and Service Management (TNSM), 2026. DOI: [10.1109/TNSM.2026.3652529](https://doi.org/10.1109/TNSM.2026.3652529)
- **Commit Hash:** `79da4d1f2a6f4e5ecf92c4588195aaf163df40f8`
- **License:** Apache License 2.0 (verified in `third_party/CLOSR/LICENSE`)
- **Inspection Date:** 2026-11-08

---

## 1. File-by-File Inventory

Every file in `third_party/CLOSR/` was inspected:

| Path | What It Is | Relevance to Tasks |
|---|---|---|
| `LICENSE` | Apache License Version 2.0 text | PR1-T01: Source for project `LICENSE` and attribution |
| `README.md` | Paper abstract, usage examples, and comparison tables | PR1-T01, PR1-T07, PR1-T08: Ground truth reproduction targets |
| `requirements.txt` | Pinned dependency versions (`numpy==1.24.2`, `pandas==2.0.0`, `scikit-learn==1.2.2`, `torch==2.0.1`, `libmr`) | PR1-T01: Base for our pinned `requirements.txt` |
| `.gitignore` | Upstream ignore rules | Reference only (not copied directly per AGENTS §3) |
| `train_clad.py` | CLI training entry point for CLAD | PR1-T04, PR1-T05: Training loop, optimizer, schedule reference |
| `train_closr.py` | CLI training entry point for CLOSR | PR1-T04, PR1-T05: Training loop, optimizer, schedule reference |
| `eval_clad.py` | CLI evaluation entry point for CLAD | PR1-T06: Centroid, similarity, and AUROC evaluation reference |
| `eval_closr.py` | CLI evaluation entry point for CLOSR | PR1-T06: Open-set scoring, OpenAUC, FPR@95, and metric logic |
| `losses/clad_loss.py` | `CLADLoss` asymmetric contrastive loss module | PR1-T04: Port to `src/nids/losses.py` |
| `losses/closr_loss.py` | `CLOSRLoss` multiclass contrastive loss module | PR1-T04: Port to `src/nids/losses.py` |
| `model/model.py` | Architecture definitions (`ContrastiveMLP`, `CLOSRMLP`, `DenseBlock`, `Residual`, `Norm`, `make_mlp`) | PR1-T04: Port to `src/nids/models.py` |
| `data/load_data.py` | `get_data(...)` loading Lycos CSV, zero-day holdout, train/val/test splits, ZNormaliser | PR1-T03: Port to `src/nids/data.py` |
| `data/loaders.py` | `tabular_dl(...)` PyTorch DataLoader with balanced sampling | PR1-T03: Port to `src/nids/data.py` |
| `data/utils.py` | `ZNormaliser`, `calc_stats`, `encode_labels` | PR1-T03: Port normalization and label encoding |
| `util/checkpoint.py` | `make_checkpoint`, `load_checkpoint` functions | PR1-T05: Port and extend to include config, class map, seed, centroids |
| `util/distance.py` | `get_centroids`, `calc_class_sims_chunked` | PR1-T06: Port to `src/nids/infer.py` |
| `util/features.py` | `get_features` chunked embedding extractor | PR1-T06: Port to `src/nids/infer.py` |
| `util/meter.py` | `AverageMeter` diagnostic tracker | PR1-T05: Port to `src/nids/train.py` |
| `util/metrics.py` | `mean_auroc`, `fpr_at_recall`, `classwise_fpr_at_recall`, `compute_mean_ap_and_pr_auc` | PR1-T06: Port to `src/nids/metrics.py` |
| `util/schedules.py` | `WarmupCosineSchedule`, `LRSchedule` | PR1-T05: Port to `src/nids/train.py` |
| `weights/clad.pt.tar` | Pretrained weights for CLAD | Reference inspection only (not used as evidence per AGENTS §3) |
| `weights/closr.pt.tar` | Pretrained weights for CLOSR | Reference inspection only (not used as evidence per AGENTS §3) |
| `images/architecture.png` | Architecture diagram | Reference only |
| `images/clad_tsne.png` | CLAD embedding t-SNE plot | Reference only |
| `images/contrastive_tsne.png` | CLOSR embedding t-SNE plot | Reference only |
| `baseline_implementations/README.md` | Guide to baseline code structure and components | PR1-T07: Reference for baseline inventory |
| `baseline_implementations/binary_classification/autoencoder.py` | Autoencoder anomaly detection baseline | PR1-T07: Runnable binary baseline |
| `baseline_implementations/binary_classification/daelr.py` | DAE-LR anomaly detection baseline | PR1-T07: Runnable binary baseline |
| `baseline_implementations/binary_classification/duad_train.py` | DUAD anomaly detection baseline | PR1-T07: Runnable binary baseline |
| `baseline_implementations/binary_classification/isolation_forrest.py` | Isolation Forest baseline (note spelling in filename) | PR1-T07: Runnable binary baseline |
| `baseline_implementations/binary_classification/renoir.py` | RENOIR baseline | PR1-T07: Runnable binary baseline |
| `baseline_implementations/binary_classification/svdd.py` | Deep SVDD anomaly detection baseline | PR1-T07: Runnable binary baseline |
| `baseline_implementations/binary_classification/svm.py` | One-Class SVM baseline | PR1-T07: Runnable binary baseline |
| `baseline_implementations/openset_recognition/crosr.py` | CROSR open-set baseline | PR1-T07: Runnable open-set baseline |
| `baseline_implementations/openset_recognition/doc.py` | DOC open-set baseline | PR1-T07: Runnable open-set baseline |
| `baseline_implementations/openset_recognition/multistage_nids.py` | MultiStage NIDS open-set baseline | PR1-T07: Runnable open-set baseline |
| `baseline_implementations/openset_recognition/openmax.py` | OpenMax open-set baseline (uses `libmr`) | PR1-T07: Runnable open-set baseline |
| `baseline_implementations/openset_recognition/siamese_network.py` | Siamese Network metric learning baseline | PR1-T07, PR2-T02: Primary reference for Stage-2 Siamese |
| `baseline_implementations/common/feed_forward.py` | Baseline MLP definitions | PR1-T07: Helper utilities |
| `baseline_implementations/common/losses.py` | Baseline loss definitions | PR1-T07: Helper utilities |
| `baseline_implementations/common/metrics.py` | Baseline metrics utilities | PR1-T07: Helper utilities |
| `baseline_implementations/common/model_eval.py` | Baseline evaluation utilities | PR1-T07: Helper utilities |
| `baseline_implementations/common/open_auc.py` | Baseline OpenAUC utility | PR1-T07: Helper utilities |
| `baseline_implementations/common/process_batch.py` | Baseline batch processor | PR1-T07: Helper utilities |
| `baseline_implementations/common/training_loops.py` | Baseline training loops | PR1-T07: Helper utilities |

---

## 2. Answers to Section 2.2 Questions

### Question 1: Exact class names in `model/model.py` (including CLAD model)
- **CLAD Model:** [`ContrastiveMLP`](https://github.com/jackwilkie/CLOSR/blob/main/model/model.py#L177) (inherits from `nn.Module`). It is imported and instantiated as the model in `train_clad.py` (line 5).
- **CLOSR Model:** [`CLOSRMLP`](https://github.com/jackwilkie/CLOSR/blob/main/model/model.py#L246) (inherits from `ContrastiveMLP`). Instantiated in `train_closr.py` (line 5, 127).
- **Supporting layers:** `Residual` (line 11), `Norm` (line 53), `DenseBlock` (line 60), and helper function `make_mlp` (line 101).

### Question 2: Whether a Siamese implementation exists in `baseline_implementations/`, with its path
- **YES.** A full, dedicated Siamese Network implementation exists at:
  `baseline_implementations/openset_recognition/siamese_network.py` (474 lines).
- It contains `OnlineContrastiveLoss` (line 122), pairwise distance matrices (`distance_matrix`, line 88), kNN distance weighting (`apply_knn_weight`, line 105), and open-set kNN inference (`knn_predict`, `knn_eval`, `knn_min_distance_ood_eval`).
- **Impact on DV2:** Because upstream code exists, PR2-T02 will adapt this upstream implementation directly, rather than building from scratch.

### Question 3: Which baselines are runnable code vs. reported numbers only
- **Runnable code present in `baseline_implementations/`:**
  - *Binary anomaly baselines:* Autoencoder (`autoencoder.py`), DAE-LR (`daelr.py`), DUAD (`duad_train.py`), Isolation Forest (`isolation_forrest.py`), RENOIR (`renoir.py`), Deep SVDD (`svdd.py`), One-class SVM (`svm.py`).
  - *Open-set baselines:* CROSR (`crosr.py`), DOC (`doc.py`), MultiStage (`multistage_nids.py`), OpenMax (`openmax.py`), Siamese Network (`siamese_network.py`).
- **Reported numbers only (or needing parameter variants/wrappers):**
  - `AutoSVM`: Reported in README Table 1; shares implementation with `svm.py` but with automated tuning parameters not wrapped as a separate script.
  - `MLP` (closed-set): Reported in README Table 1; uses standard feed-forward networks from `common/feed_forward.py` or `model/model.py`, but has no dedicated top-level `mlp_train.py` script.

---

## 3. Facts to Reuse in Later Tasks (with exact file and line references)

1. **Default Training Seed:**
   - In `train_closr.py` (line 36) and `train_clad.py` (line 36), the only seed exposed via CLI is `--split_seed 39058032`, passed directly to `get_data(...)`.
   - **Crucial finding:** Neither `train_closr.py` nor `train_clad.py` calls `torch.manual_seed(...)` or `np.random.seed(...)` in `main()`. Our port explicitly introduces multi-seed training (seeds 1, 2, 3) as mandated by `AGENTS.md` §5.2 and `decisions.md` D4.3.

2. **Learning Rate, Schedule, and Stepping:**
   - `train_closr.py` lines 147–167:
     - Optimizer: `AdamW` with initial `lr=1e-6`, `betas=(0.9, 0.999)`, `weight_decay=0.0403709` (CLI default line 74).
     - Schedule: `WarmupCosineSchedule(start_val=1e-6, end_val=1e-6, ref_val=opt.lr, T_max=opt.epochs * len(train_dl), warmup_steps=int((opt.epochs // 10) * len(train_dl)))`. Peak LR (`opt.lr`) defaults to `1e-5` (line 76).
     - Stepping frequency: `schedule.step()` is called **per batch / per step** (line 209 in `train()`), **not** per epoch!

3. **Feature Scaling in `data/load_data.py`:**
   - Lines 100–104:
     ```python
     norm = ZNormaliser(x_train)
     x_train = norm.normalise(x_train)
     x_val = norm.normalise(x_val) if x_val is not None else None
     x_test = norm.normalise(x_test) if x_test is not None else None
     x_zd = norm.normalise(x_zd) if x_zd is not None else None
     ```
   - **Verification:** The `ZNormaliser` is fit **strictly on `x_train` only**, preventing test or zero-day leakage into training normalization statistics. `ZNormaliser` uses Bessel's correction (`ddof=1` for sample standard deviation).

4. **Validation Split Handling:**
   - In `train_closr.py` (line 109) and `train_clad.py` (line 109), `val_ratio` is hardcoded to `0.0`.
   - In `data/load_data.py` lines 86–95, if `val_ratio > 0.0`, the validation split is carved from `x_train` after the test split:
     `test_size = val_ratio / (1 - test_ratio)`, `random_state=split_seeds[1]`, `stratify=y_train`.
   - This matches pre-registered deviation DV6.

5. **Argparse Plain Boolean Flags (Pitfall §12.1):**
   - In `train_closr.py`:
     - Line 54: `parser.add_argument("--residual", type=bool, default=False, ...)`
     - Line 65: `parser.add_argument("--squared", type=bool, default=True, ...)`
   - Passing `--residual False` or `--squared False` in standard Python argparse results in `True` because any non-empty string evaluates to `True`. Handled in our port via YAML config (`yaml.safe_load`).

---

## 4. Discrepancies with AGENTS.md

1. **CLAD Dimensionality Defaults:**
   - In `train_clad.py` line 41, `--d_out` defaults to `8`, and `--neurons` defaults to `"1024,1024,1024,1024"` (4 layers).
   - In `train_closr.py` line 41, `--d_out` defaults to `64`, and `--neurons` defaults to `"1024,1024,1024"` (3 layers).
   - AGENTS.md §2.3 lists `output dim 64, hidden layers 1024-1024-1024` as global model defaults. In our config system (`configs/default.yaml`), we will configure `d_out` and `hidden` per model kind.
2. **Siamese Network Availability:**
   - AGENTS.md PR2-T02 noted "Reuse the upstream Siamese baseline if it exists; otherwise implement...". We have confirmed that `baseline_implementations/openset_recognition/siamese_network.py` exists upstream.
3. **Dependency Versions:**
   - Upstream pinned `numpy==1.24.2`, `pandas==2.0.0`, `scikit-learn==1.2.2`, `torch==2.0.1`. On modern Python 3.11+ (especially macOS arm64), older C-extension wheels may require updated release pins (recorded as DV7 in `docs/DEVIATIONS.md`).
