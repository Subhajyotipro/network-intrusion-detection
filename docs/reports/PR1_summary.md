# PR1 Summary Report

## 1. Header

- **PR:** 1 (PR1-T01 through PR1-T11)
- **Report date:** 2026-11-11
- **Report base HEAD:** `889e3b266e22dde404866365919ba97bb0758413` on branch `main` (HEAD before the PR1-T11 summary-report commit)
- **Environment:** Windows 10 build 10.0.26300; Python 3.11.9; TensorFlow 2.21.0; Keras 3.15.1; CPU (`/physical_device:CPU:0`). Native Windows TensorFlow reports that GPU execution is unavailable.
- **Datasets:** PR1-T03 and T08 used real Lycos2017; T10 unit and end-to-end tests used synthetic data; PR1-T09 figures use real test/zero-day embeddings from the 10-epoch checkpoints. The synthetic Lycos-schema generator is used by tests and smoke runs.
- **Seeds:** split seed `39058032`; T08 model results are one training seed (`1`). Unit/regression tests use fixed seeds. No verifiable three-seed CLAD/CLOSR result set is present.
- **T08 result-record provenance:** evaluation JSONs record run-time commit `4266c6f1a6094cd93ae3fd9852c708b576d298ff`, Python 3.11.9, TensorFlow 2.21.0, Keras 3.15.1 and CPU. The top-level `epochs=10` and `reduced_run=true` describe the trained checkpoints; the nested config records the default 200-epoch value loaded by evaluation.

## 2. Task Completion Matrix

| Task ID | Title | Status | Files created or changed | Evidence | Notes |
|---|---|---|---|---|---|
| PR1-T01 | Scaffolding and attribution | PARTIAL | `LICENSE`, `NOTICE`, `pyproject.toml`, `requirements.txt`, `third_party/CLOSR/`, `docs/UPSTREAM_INVENTORY.md`, `docs/DEVIATIONS.md`, package/test skeleton | `git submodule status` shows initialized CLOSR; `python -m pip install -e . --no-deps` succeeds; `test_import_nids_modules`; `test_static_scan_src_and_scripts_for_torch` | Inventory, attribution, and deviations are present. An editable install was verified in this checkout, not a separately-created fresh clone. |
| PR1-T02 | Config system | DONE | `src/nids/config.py`, `src/nids/config_io.py`, `configs/default.yaml`, `configs/clad.yaml`, `scripts/run.py`, config tests | `test_closr_defaults_match_inventory`; `test_clad_defaults_match_inventory`; `test_cli_boolean_set_options_parse_false`; `test_yaml_round_trip` | Defaults and boolean CLI overrides are guarded. |
| PR1-T03 | Data pipeline | PARTIAL | `src/nids/data.py`, `docs/DATA_CARD.md`, `configs/default.yaml`, `scripts/run.py`, `tests/unit/test_data.py`, `tests/regression/test_real_data.py`, `results/splits/.gitkeep` | `test_data_splits_no_index_overlap_and_zero_day_absence`; `test_split_indices_save_and_reload`; `test_real_data_splits_and_zero_day_detection`; `test_synthetic_generator_schema_and_determinism` | Loading, zero-day isolation, and saving indices when `results_dir` is supplied are tested. The normal CLI real-data loader does not pass `results_dir`, and no real split NPZ is in this checkout; persistence for the recorded T08 run is therefore not evidenced. |
| PR1-T04 | Models and losses | DONE | `src/nids/models.py`, `src/nids/losses.py`, `tests/unit/test_models.py`, `tests/unit/test_losses.py`, `tests/unit/test_loss_edge_cases.py`, `tests/unit/np_reference.py`, `docs/BUGLOG.md` | `test_clad_output_shapes_and_unit_norm`; `test_closr_output_shapes_and_unit_norm`; `test_gradient_flow_to_all_trainable_parameters`; `test_loss_decreases_over_training_steps`; `test_clad_loss_matches_numpy_reference`; `test_closr_loss_matches_numpy_reference`; `test_keras_torch_linear_initializer_bounds` | TensorFlow implementations and independent NumPy regression coverage. |
| PR1-T05 | Unified trainer | DONE | `src/nids/train.py`, `src/nids/checkpoint.py`, `src/nids/config.py`, `src/nids/data.py`, `scripts/run.py`, training/checkpoint tests | `test_train_smoke_clad_end_to_end`; `test_train_smoke_closr_end_to_end`; `test_training_resume_reproduces_identical_loss`; `test_checkpoint_reload_restores_identical_outputs_and_optimizer_vars`; `test_train_cli_smoke_closr`; `test_train_cli_smoke_clad` | T10 fixed the compiled-path NaN guard. T10 also fixed normal CLI sidecars to include centroids. |
| PR1-T06 | Evaluation harness | DONE | `src/nids/infer.py`, `src/nids/metrics.py`, `src/nids/metrics_eval.py`, `src/nids/tables.py`, `src/nids/eval_pipeline.py`, `scripts/run.py`, `tests/integration/test_eval_smoke.py`, tracked files under `results/eval/` | `test_closr_eval_smoke_subprocess`; `test_clad_eval_smoke_subprocess`; `test_closr_metrics_vs_numpy_reference`; `test_clad_metrics_vs_numpy_reference`; tracked CLAD/CLOSR eval JSON and per-class CSV | T11 tracked the four small real-data metric/table files; arrays remain local-only. |
| PR1-T07 | Baselines | PARTIAL | `src/nids/baselines.py`, `src/nids/compare.py`, `scripts/run.py`, `tests/unit/test_baselines.py`, `tests/integration/test_baseline_smoke.py`, `results/baselines/comparison.csv`, `results/baselines/comparison.md` | `test_baseline_cli_smoke_all_models`; `test_comparison_table_structure_and_protocol`; tracked comparison CSV/Markdown | The comparison table contains measured CLAD/CLOSR and upstream-reported rows. Historical T08 AE/IF/MLP rerun values lack their JSON artifacts in this checkout and are not independently verified here. |
| PR1-T08 | Reproduction check | PARTIAL | `results/T08_report.md`, tracked `results/eval/clad_seed1_eval.json`, `results/eval/clad_seed1_per_class.csv`, `results/eval/closr_seed1_eval.json`, `results/eval/closr_seed1_per_class.csv`; local-only arrays and weights | Real-data seed-1 JSONs; `test_real_data_splits_and_zero_day_detection`; comparison table | The measured runs used 10 epochs rather than 200. CLAD means are within tolerance; all three CLOSR metrics are outside tolerance. The likely role of reduced training is a hypothesis, not a demonstrated cause. |
| PR1-T09 | Visualisation | DONE | `src/nids/viz.py`, `src/nids/infer.py`, `scripts/run.py`, `tests/unit/test_viz.py`, `tests/unit/test_infer.py`, four tracked files under `results/figures/`, `docs/DEVIATIONS.md` | `test_tsne_coordinates_and_png_are_seed_deterministic`; `test_tsne_subsample_keeps_all_zero_day_and_stratifies_classes`; both JSON sidecars and PNG figures | Both sidecars say `reduced_run=true`, seed 1, dataset real, and include all 23 zero-day rows. These figures come from reduced-epoch checkpoints, not full 200-epoch runs. |
| PR1-T10 | Test and bug-hunt pass | DONE | `src/nids/train.py`, `scripts/run.py`, `tests/conftest.py`, unit/integration/regression tests, `docs/BUGLOG.md` | Final non-real suite: 177 passed; real-marked suite: 3 passed; coverage below; Section 12 audit in `docs/BUGLOG.md` | BUG-005, BUG-006, BUG-007, and BUG-008 are closed. No remaining OPEN BUGLOG entries. |
| PR1-T11 | PR1 summary report | DONE | `docs/reports/PR1_summary.md`, `docs/BUGLOG.md`, `tests/conftest.py`, corrected `results/T08_report.md`, small tracked result artifacts | This report; BUG-007 real test outcome; tracked result files; validations in Section 5 | Summary and supporting T11 commits remain local and unpushed pending human review. |

## 3. Acceptance-Criteria Checklist

- **PR1-T01 scaffolding:** [x] Apache license and NOTICE present; upstream submodule initialized; inventory and DV register present. [ ] Fresh-clone installation was not tested; editable install in the current checkout passed.
- **PR1-T02 config:** [x] Per-model default tests pass. [x] `--set residual=false` and `--set squared=false` parse as false.
- **PR1-T03 data:** [x] Real schema/classes/zero-day split validated. [x] Synthetic schema validated. [x] Train/validation/test overlap and zero-day isolation are tested. [x] Split-index save/reload works when given an output directory. [ ] The actual T08 split archive is absent, and the normal real CLI loader does not pass a results directory; end-to-end run persistence remains incomplete/unverified.
- **PR1-T04 models/losses:** [x] CLAD/CLOSR output shape and unit norms. [x] Finite/nonnegative and gradient checks. [x] Toy optimization reduces loss. [x] Margin/squared/asymmetric behavior tests. [x] NumPy loss parity and initializer-bound regressions.
- **PR1-T05 trainer:** [x] CPU synthetic smoke training for CLAD/CLOSR. [x] Resume parity with dropout disabled. [x] Reload output parity and metadata tests. [x] Standard CLI checkpoint sidecars include centroids after BUG-008.
- **PR1-T06 evaluation:** [x] Evaluation JSON and per-class CSV generated for both models. [x] Metric comparison tests against independent NumPy references. [x] Result records have seed/config/version/device metadata. [x] Small JSON/CSV evidence is now tracked; large array archives remain local-only.
- **PR1-T07 baselines:** [x] Keras MLP/autoencoder and sklearn Isolation Forest implementations have smoke tests. [x] Comparison CSV labels available rows `re-run` or `upstream-reported`. [ ] Local real-data AE/IF/MLP result JSONs are absent, so their historical T08 rerun values cannot be audited from this checkout; task remains PARTIAL.
- **PR1-T08 reproduction:** [x] Real Lycos split seed and `val_ratio=0.0`; [x] measured metrics and gaps documented. [ ] Full 200-epoch training and three CLAD/CLOSR seeds were not run. CLAD meets the ±0.01 target; CLOSR does not; status remains PARTIAL.
- **PR1-T09 figures:** [x] CLAD/CLOSR PNGs and sidecars are tracked. [x] Sidecars identify real dataset, seed, and `reduced_run=true`; all zero-day rows are retained. [x] Determinism is tested.
- **PR1-T10 test/bug hunt:** [x] Unit, integration, and NumPy regression tiers exist. [x] Non-real suite and real-marked suite pass. [x] BUGLOG and Section 12 audit are populated. [x] Coverage reported below.
- **PR1-T11 report:** [x] All 11 required sections included; task statuses and evidence are stated.

## 4. Results

### CLAD and CLOSR vs. Section 2.4

Values below are read from the tracked T08 evaluation JSONs. Gaps are measured value minus upstream target. “Within ±0.01” is judged on the absolute gap.

| Model / metric | Upstream target | Measured, seed 1 | Gap | Within ±0.01 |
|---|---:|---:|---:|---|
| CLAD closed-set mean AUROC | 0.999855 | 0.999278 | -0.000577 | Yes |
| CLAD open-set mean AUROC | 0.996627 | 0.992735 | -0.003892 | Yes |
| CLAD Heartbleed AUROC | 0.995557 | 0.990057 | -0.005500 | Yes |
| CLAD SQL Injection AUROC | 0.997696 | 0.995413 | -0.002283 | Yes |
| CLOSR closed-set accuracy | 0.995276 | 0.979121 | -0.016155 | No |
| CLOSR open-set AUC | 0.974022 | 0.952900 | -0.021122 | No |
| CLOSR OpenAUC | 0.969420 | 0.933005 | -0.036415 | No |

These are one-seed, 10-epoch runs (`reduced_run=true`) compared with the upstream 200-epoch configuration. They are not called reproductions. Reduced training is a plausible contributor to the CLOSR gaps, but the available measurements do not establish the cause. The evaluation JSON’s nested `config.epochs=200` reflects the evaluation config; its top-level `epochs=10` and `reduced_run=true` are the recorded training-run values.

The fixed split seed is `39058032`. If standard deviations are reported for runs on this fixed split, they describe training randomness only, not split variation. No verifiable three-seed CLAD/CLOSR mean ± standard deviation is available.

FPR@95 follows the upstream zero-day-positive convention: project scores are known-ness scores (higher means known), negated once for ROC/FPR calculations so higher values mean more zero-day-like. FPR@95 is the known-class false-positive rate at 95% recall of zero-day samples.

### Baseline Comparison

The tracked `results/baselines/comparison.csv` is the source for this table. Rows are explicitly classified by provenance. “n/a” means the metric does not apply or a local rerun artifact is unavailable; it is not an imputed result.

| Model | Protocol | Closed metric | Open metric | OpenAUC | Source |
|---|---|---:|---:|---:|---|
| CLAD | Binary AUROC vs benign | 0.999278 mean AUROC | 0.992735 mean AUROC | n/a | re-run, seed 1, reduced |
| CLOSR | Closed/open-set recognition | 0.979121 accuracy | 0.952900 AUC | 0.933005 | re-run, seed 1, reduced |
| CLAD | Binary AUROC vs benign | 0.999855 mean AUROC | 0.996627 mean AUROC | n/a | upstream-reported |
| AE | Binary AUROC vs benign | 0.857221 mean AUROC | 0.840659 mean AUROC | n/a | upstream-reported |
| IF | Binary AUROC vs benign | 0.767531 mean AUROC | 0.838422 mean AUROC | n/a | upstream-reported |
| MLP | Binary AUROC vs benign | 0.999557 mean AUROC | 0.533859 mean AUROC | n/a | upstream-reported |
| CLOSR | Closed/open-set recognition | 0.995276 accuracy | 0.974022 AUC | 0.969420 | upstream-reported |
| MultiStage | Closed/open-set recognition | 0.996612 accuracy | 0.801251 AUC | 0.798537 | upstream-reported |
| DOC | Closed/open-set recognition | 0.995536 accuracy | 0.570263 AUC | 0.567717 | upstream-reported |
| OpenMax | Closed/open-set recognition | 0.995615 accuracy | 0.720174 AUC | 0.717016 | upstream-reported |
| CRSOR | Closed/open-set recognition | 0.994940 accuracy | 0.748295 AUC | 0.744509 | upstream-reported |
| Siamese Network | Closed/open-set recognition | 0.997722 accuracy | 0.720811 AUC | 0.719167 | upstream-reported |
| AE local rerun | Binary AUROC vs benign | n/a | n/a | n/a | re-run claimed in earlier T08 staging text; result JSON absent, not verified |
| IF local rerun | Binary AUROC vs benign | n/a | n/a | n/a | re-run claimed in earlier T08 staging text; result JSON absent, not verified |
| MLP local rerun | Binary AUROC vs benign | n/a | n/a | n/a | re-run claimed in earlier T08 staging text; result JSON absent, not verified |

The comparison is incomplete for local AE/IF/MLP results. The historical numbers in the earlier T08 staging table are not copied here because the corresponding baseline JSONs are missing from this checkout.

## 5. Tests

- **Collected:** 180 test cases in the non-real invocation (including parametrized cases); 177 passed, 3 deselected, 0 failed.
- **Non-real baseline before PR1-T11 edits:** `pytest -x -q -m "not real"` with `TF_NUM_INTRAOP_THREADS=1` and `TF_NUM_INTEROP_THREADS=1`: 177 passed, 3 deselected, 117.33s.
- **Final coverage command:** `pytest -q -m "not real" --cov=nids --cov-report=term-missing` with the same TensorFlow thread limits: 177 passed, 3 deselected in 238.07s. Coverage: `losses.py` 100%; `data.py` 99%; `metrics.py` 96% (`metrics_eval.py` 96%); `checkpoint.py` 92%; checkpoint orchestration in `train.py` 95%.
- **Final non-real command:** `pytest -x -q -m "not real"` with the same thread limits: 177 passed, 3 deselected in 193.54s. This final run exceeds the stated two-minute target; the pre-edit baseline was 117.33s. All tests passed, and no timeout setting for non-real tests was changed. Runtime variance is recorded as a limitation rather than attributing it to the real-only timeout change.
- **Real-marked command after BUG-007 fix:** `pytest -m real -q` with the same thread limits: 3 passed, 177 deselected, 10 warnings in 241.52s. This includes the real evaluation smoke test; the real-only 600-second timeout did not trigger.
- **Lint/install/format:** final `ruff check .` passed; `ruff format .` reported all 56 Python files unchanged; `py_compile` passed for `tests/conftest.py`; editable install `python -m pip install -e . --no-deps` succeeded.
- **Repair-loop iterations by task:** T01 0 recorded; T02 0; T03 2 (BUG-002, BUG-003); T04 2 (BUG-001); T05 0 separately logged; T06 0; T07 0; T08 0; T09 1 (BUG-004); T10 3 (BUG-005, BUG-006, BUG-008); T11 1 (BUG-007). Counts are from `docs/BUGLOG.md`; they count fix iterations, not test executions.

## 6. Bugs Found and Fixed

| ID | Symptom | Root cause | Fix | Guarding test | Iterations |
|---|---|---|---|---|---:|
| BUG-001 | TensorFlow CLAD/CLOSR loss differed from the NumPy reference | Float32 roundoff made diagonal self-distance pass the pair threshold | Explicitly zero self-pair diagonals | `test_clad_loss_matches_numpy_reference`; `test_closr_loss_matches_numpy_reference`; `test_identical_batch_finite_loss_and_no_self_pairs` | 2 |
| BUG-002 | Synthetic split test expected lowercase `heartbleed` but fixture had title-case labels | Fixture labels predated canonical label normalization | Standardized fixture labels | `test_data_splits_no_index_overlap_and_zero_day_absence` | 1 |
| BUG-003 | CLI smoke test loaded and trained on the large real CSV | Test used automatic data selection without forcing synthetic data | Force synthetic mode for the test | `test_cli_subcommands` | 1 |
| BUG-004 | Uniform t-SNE subsampling could omit rare zero-day rows | Uniform sampling ignored label frequency | Seeded stratification retains all zero-day rows | `test_tsne_subsample_keeps_all_zero_day_and_stratifies_classes` | 1 |
| BUG-005 | NaN features did not raise from the compiled train path | Eager-only finite check did not run inside `tf.function` | Validate feature batch before compiled update | `test_train_nan_input_raises_value_error`; `test_train_inf_input_raises_value_error` | 1 |
| BUG-006 | Non-demo CLI paths wrote status messages with `print` | CLI dispatch violated logging convention | Replace status prints with logger calls | `test_cli_subcommands` | 1 |
| BUG-007 | Real evaluation smoke exceeded 120 seconds during centroid extraction | Global synthetic-suite timeout was too short for the full real split on CPU | `tests/conftest.py` sets a 600-second timeout only for `real`-marked tests; real run then passed in 241.52 seconds | `test_real_data_eval_smoke_pipeline`; `pytest -m real -q` | 1 |
| BUG-008 | Standard CLI checkpoint sidecars omitted centroids | `run_train` saved losses/schedule metadata only | Save CLAD benign or CLOSR per-class centroids in sidecar `extra` | `test_train_cli_smoke_closr`; `test_train_cli_smoke_clad` | 1 |

**Open bugs:** None remain in `docs/BUGLOG.md` after the real-marked run. T03 split persistence and T07 missing baseline outputs remain task limitations, not bugs misrepresented as closed issues.

### Section 12 Audit

| Pitfall | Verdict | Evidence |
|---|---|---|
| 1. Boolean CLI flags | NOT PRESENT | `test_cli_boolean_set_options_parse_false`; `test_boolean_flag_overrides` |
| 2. Hard-coded input dimension | NOT PRESENT | `test_clad_output_shapes_and_unit_norm`; `test_closr_output_shapes_and_unit_norm`; train and T10 end-to-end tests |
| 3. Learning-rate constants | NOT PRESENT | `test_closr_defaults_match_inventory`; `test_clad_defaults_match_inventory`; `test_schedule_matches_numpy_reference_at_every_step` |
| 4. Missing validation split | NOT PRESENT | `test_data_splits_no_index_overlap_and_zero_day_absence`; `test_val_ratio_zero_reproduces_upstream_split` |
| 5. Label-layout assumptions | NOT PRESENT | `test_known_class_absent_from_test_split`; regression label adapter in `test_malicious_recall_and_open_auc_match_upstream_after_unknown_adapter` |
| 6. Eval/train mode and device handling | NOT PRESENT | `test_training_flag_determinism_and_stochasticity`; `test_get_embeddings_deterministic_eval`; `test_get_embeddings_does_not_open_gradient_tape`; `test_tf_tiny_forward_pass_on_cpu` |
| 7. Open-set score direction | NOT PRESENT | `test_score_direction`; `test_every_registered_scorer_orders_known_above_unknown`; upstream score NumPy regression; `to_zero_day_scores` is the single ROC sign-conversion helper |
| 8. Balanced sampler and tiny batches | NOT PRESENT | `test_balanced_sampler_properties`; `test_train_robustness_small_dataset_and_class` |
| 9. Benign ID | NOT PRESENT | `test_benign_id_constant`; `test_mal_recall_and_macro_f1_hand_computed` |
| 10. Checkpoints | NOT PRESENT | Reload/resume tests and `test_train_cli_smoke_closr`/`test_train_cli_smoke_clad` verify production sidecars including centroids |
| 11. Data leakage | NOT PRESENT | Identifier drop, train-only mask/imputation/scaler, split disjointness and split save/reload tests |
| 12. Metric edge cases | NOT PRESENT | Single-class AUROC, empty zero-day/known cases, undefined recall, absent known class, and FPR regression tests |

## 7. Deviations

The following records are in `docs/DEVIATIONS.md`. “Applied” means used by PR1 code or protocol; DV2 and DV3 are PR2 design decisions and were not applied in PR1.

| ID | PR1 disposition |
|---|---|
| DV1 | Applied: flat `src/nids` modules and CLI subcommands. |
| DV2 | Not applied: PR2-only conditional pair-mask decision. Upstream Siamese code exists, so the “only if no upstream Siamese exists” condition is false. |
| DV3 | Not applied: PR2-only pseudo-zero-day calibration choice. |
| DV4 | Applied: zero-day evaluation label is `UNKNOWN_ID=-1`, with names retained separately. |
| DV5 | Applied: mixed precision is not implemented; tests run with it off. |
| DV6 | Applied: validation is carved from the fixed-test remainder when requested; T08 uses `val_ratio=0.0`. |
| DV7 | Applied: modern dependency pins and no PyTorch dependency. |
| DV8 | Applied: CLAD/CLOSR per-kind defaults follow upstream training scripts. |
| DV9 | Applied in configuration; T08 evidence is seed 1 only, not the full three-seed protocol. |
| DV10 | Applied: runtime and tests use TensorFlow/Keras and NumPy. |
| DV11 | Applied: numeric references are independent NumPy transcriptions, not upstream execution. |
| DV12 | Applied in PR1 scope: minimum baselines are local implementations; unavailable comparison results are explicitly upstream-reported or unavailable. Siamese work remains PR2. |
| DV13 | Applied: all-zero feature mask is fit on train only. |
| DV14 | Applied: infinities converted to missing values; train medians impute. |
| DV15 | Implemented and unit-tested when `results_dir` is supplied; actual run persistence is incomplete because the normal CLI loader does not pass that directory. |
| DV16 | Applied: self-pair distance diagonal explicitly zeroed. |
| DV17 | Applied: built-in Keras AdamW formulation with configured epsilon. |
| DV18 | Applied: raw lowercase snake_case Lycos labels. |
| DV19 | Applied: explicit `auto`/`real`/`synthetic` data source and warning on fallback. |
| DV20 | Applied: sklearn `StandardScaler` population standard deviation. |
| DV21 | Applied: vectorized contrastive loss implementation. |
| DV22 | Applied: streaming inference/evaluation avoids retaining all embeddings. |
| DV23 | Applied as seeded class-stratified sampling retaining every zero-day row. **Stale wording remains in the table:** it still says “reservoir”; the file’s leading update note correctly supersedes that phrase. |
| DV24 | Applied: configurable periodic checkpoint cadence. |
| DV25 | Applied: default config uses `val_ratio=0.0`; PR2/PR3 are specified to use 0.1. |
| DV26 | Withdrawn/not a deviation; retained only for numbering stability. |
| DV27 | Applied: CLOSR configured loss arguments are honored. |
| DV28 | Applied: actual positive-pair fraction diagnostic is reported. |
| DV29 | Applied: local AE architecture is an explicitly chosen design because upstream layer widths/training protocol were not specified. |
| DV30 | Applied: configurable local baseline hyperparameters and `reduced_run` marking; local real baseline outputs are not present for verification here. |
| DV31 | Applied in T09: CLOSR per-class vectors are flattened for one point per flow. It is documented in DEVIATIONS.md but is not in decisions.md Appendix B. |

No new scientific-method deviation was introduced in PR1-T11. The real-only pytest timeout override changes test runtime, not model/data/results behavior.

## 8. Reuse Log

- Reimplemented upstream `losses/clad_loss.py` and `losses/closr_loss.py` mathematics in TensorFlow in `src/nids/losses.py`; numerical checks use `tests/unit/np_reference.py` and `tests/regression/test_upstream_numeric.py`.
- Adapted upstream `model/model.py` (`ContrastiveMLP`, `CLOSRMLP`) to configurable input dimensions and TensorFlow/Keras in `src/nids/models.py`.
- Adapted the upstream `data/load_data.py`, `data/loaders.py`, and `data/utils.py` concepts in `src/nids/data.py`, adding named split structures, train-only cleaning/scaling and synthetic data generation.
- Adapted scheduling, meter, and checkpoint concepts from `util/schedules.py`, `util/meter.py`, and `util/checkpoint.py` into `src/nids/train.py` and `src/nids/checkpoint.py`; checkpoint metadata extends the reference with config, class map, seed, and centroids.
- Adapted `util/features.py` and `util/distance.py` concepts into chunked/streamed `src/nids/infer.py`; adapted `util/metrics.py` and eval-script metrics into `src/nids/metrics.py`, `metrics_eval.py`, and `tables.py`.
- The upstream training and evaluation scripts informed defaults and protocol but are not executed. Upstream weights are not used.
- Upstream baseline implementations are inventoried in `docs/UPSTREAM_INVENTORY.md`. The local PR1 baseline path implements Keras MLP/AE and sklearn Isolation Forest; no PyTorch baseline is imported or run. The upstream Siamese implementation exists but is deferred to PR2.
- Upstream images are references only; PR1-T09 generates its own t-SNE outputs. No upstream source files were edited.

## 9. Risks, Limitations, and Open Questions

- CLOSR is outside tolerance on closed accuracy (-0.016155), open AUC (-0.021122), and OpenAUC (-0.036415). Ten epochs versus 200 may contribute, but that cause remains unverified; do not treat these results as reproduced performance.
- CLAD and CLOSR T08 values are each one training seed. The fixed split seed is `39058032`; a multi-seed spread is not available for these runs.
- Local AE/IF/MLP baseline JSONs and the earlier baseline result table are absent. Their T08 staging rerun figures are not independently auditable. The tracked comparison table contains verified CLAD/CLOSR re-runs and upstream-reported comparison values.
- The real-marked test suite passes with a 600-second per-real-test timeout but takes 241.52 seconds on this CPU. The global 120-second timeout remains for synthetic/non-real tests. The final non-real suite took 193.54 seconds, exceeding the two-minute target; the observed run-to-run runtime varies substantially.
- The real T08 eval JSONs do not include OS/hardware metadata. The staging T08 text identifies an Apple Silicon CPU, which is consistent with `docs/PERF_INVESTIGATION.md`, but that machine detail is not independently stored in the JSON record. T11 was executed on Windows CPU.
- Normal CLI real-data loading does not persist split archives, despite the data loader’s optional save facility and its save/reload unit test. T03 is PARTIAL until each experiment’s split indices are persisted by the run path.
- `scripts/make_report.py` exists but contains only a module docstring and performs no aggregation. It was invoked and exited without producing artifacts. The existing `run.py compare` command generated the tracked baseline comparison table; full report regeneration is PR3-T07 scope.
- T09 figures and T08 results come from reduced 10-epoch checkpoints. Both figure sidecars have `reduced_run=true`.
- The T11 timeout and artifact-evidence commits, plus this report commit, are unpushed and left for human review.
- PR2/PR3 are specified to use `val_ratio=0.1`, reducing known training data relative to T08 (`val_ratio=0.0`). Their results will not be directly comparable to Section 2.4 without stating this protocol difference.
- The human should decide whether to run longer CLOSR training before PR2; no tuning or rerun was performed for this report.

## 10. How to Reproduce

Run these PowerShell commands from the repository root. Place the licensed/preprocessed Lycos CSV at `data/lycos.csv`; do not commit the dataset. The measured T08 run used 10 epochs, seed 1, fixed split seed 39058032, and `val_ratio=0.0`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
& 'C:\Program Files\Git\cmd\git.exe' submodule update --init --recursive

# Place Lycos2017 at data\lycos.csv before selecting data_source=real.

.\.venv\Scripts\python.exe scripts\run.py train --config configs\clad.yaml --set data_source=real --set epochs=10 --set val_ratio=0.0 --set seed=1
.\.venv\Scripts\python.exe scripts\run.py train --config configs\default.yaml --set data_source=real --set epochs=10 --set val_ratio=0.0 --set seed=1

.\.venv\Scripts\python.exe scripts\run.py eval --config configs\clad.yaml --set data_source=real --set val_ratio=0.0 --set seed=1
.\.venv\Scripts\python.exe scripts\run.py eval --config configs\default.yaml --set data_source=real --set val_ratio=0.0 --set seed=1

.\.venv\Scripts\python.exe scripts\run.py baseline --model mlp --config configs\default.yaml --set data_source=real --set baseline_epochs=10 --set seed=1
.\.venv\Scripts\python.exe scripts\run.py baseline --model autoencoder --config configs\default.yaml --set data_source=real --set baseline_epochs=10 --set seed=1
.\.venv\Scripts\python.exe scripts\run.py baseline --model iforest --config configs\default.yaml --set data_source=real --set seed=1
.\.venv\Scripts\python.exe scripts\run.py compare --config configs\default.yaml --set data_source=real
.\.venv\Scripts\python.exe scripts\run.py tsne --config configs\clad.yaml --set data_source=real --set seed=1
.\.venv\Scripts\python.exe scripts\run.py tsne --config configs\default.yaml --set data_source=real --set seed=1

$env:TF_NUM_INTRAOP_THREADS = '1'
$env:TF_NUM_INTEROP_THREADS = '1'
.\.venv\Scripts\python.exe -m pytest -x -q -m 'not real'
.\.venv\Scripts\python.exe -m pytest -m real -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format .
.\.venv\Scripts\python.exe -m pytest -q -m 'not real' --cov=nids --cov-report=term-missing
.\.venv\Scripts\python.exe scripts\make_report.py
```

The last command currently exits successfully but is a no-op; use `scripts/run.py compare` for the implemented comparison-table generation. The 10-epoch commands document the measured reduced run; omit `--set epochs=10` only for a separate, full 200-epoch training run.

## 11. Next PR Readiness

PR2 prerequisites are partially met: validation splitting is covered by synthetic and real tests; the metrics, CLOSR inference, and checkpoint APIs are present; standard CLI checkpoints now include centroids. However, the normal real-data load path does not yet persist split indices, no Siamese classifier has been implemented, local AE/IF/MLP real-result artifacts are unavailable, and the reduced CLOSR result misses all three target metrics. Before PR2, the human should decide whether to allocate a longer CLOSR run and whether to require T03 split persistence in the CLI path; PR2’s `val_ratio=0.1` results must be reported as a different training-data protocol from T08.
