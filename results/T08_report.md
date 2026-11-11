# PR1-T08 Reproduction Report (Status: PARTIAL)

> **Execution Protocol:** Real Lycos2017 dataset (`data/lycos.csv`), fixed split seed `39058032`, `test_ratio=0.5`, `val_ratio=0.0`.
> **CPU Training Budget:** Reduced training (10 epochs, `reduced_run=true`) on Apple Silicon CPU.
> **Staging Notice:** This is a staging document for PR1-T08; its findings will be formally integrated into `docs/reports/PR1_summary.md` at PR1-T11.

---

## 1. Task Definition & Acceptance Criteria (Verbatim)

### AGENTS.md PR1-T08
```text
| PR1-T08 | Reproduction check | Train CLAD and CLOSR with default config on real data (≥1 seed; 3 seeds preferred) and compare with Section 2.4. Use `val_ratio=0.0` here so the setup matches upstream. If only a CPU is available and epochs must be reduced, record `reduced_run=true` and mark the task PARTIAL (`decisions.md` D3.8). | Metrics within ±0.01 of targets, or a written explanation of every gap in the report. If dataset blocked, mark BLOCKED and show synthetic-data results instead. |
```

### decisions.md Section D3.8
```text
8. **CPU-only real-data training:** 200 epochs at batch 4096 may be impractical. Run with a reduced `epochs` (via `--set epochs=N`), store `epochs` and `reduced_run=true` in the result JSON, and mark PR1-T08 `PARTIAL` with the real numbers. Never present reduced-epoch numbers as full reproduction.
```

---

## 2. Reproduction Results vs. Targets

Reproduction target tolerance per AGENTS.md Section 2.4 is within **$\pm 0.01$**.
The CLAD and CLOSR entries are verified against the evaluation JSONs. The baseline rerun values in the following table were recorded in the original staging analysis, but their baseline JSON files are absent from this checkout; they are therefore not independently verified for PR1-T11 (see Sections 5–6).

### Table 1: Binary AUROC vs. Benign Traffic (One-vs-Benign)
| Model | Target Closed Mean | Re-Run Closed Mean | Closed Gap | Target Open Mean | Re-Run Open Mean | Open Gap | Within $\pm 0.01$? | Source / Seed |
|---|---|---|---|---|---|---|---|---|
| **CLAD** | 0.999855 | 0.999278 | -0.000577 | 0.996627 | 0.992735 | -0.003892 | **YES** | re-run (seed 1) |
| **Autoencoder** | 0.857221 | 0.879493 | +0.022272 | 0.840659 | 0.873558 | +0.032899 | **NO** (+0.022 / +0.033) | re-run (seed 1) |
| **Isolation Forest** | 0.767531 | 0.803184 | +0.035653 | 0.838422 | 0.861697 | +0.023275 | **NO** (+0.036 / +0.023) | re-run (seed 1) |
| **MLP Classifier** | 0.999557 | 0.999396 | -0.000161 | 0.533859 | 0.715434 | +0.181575 | **NO** (closed YES, open NO) | re-run (seed 1) |

*Per-class Zero-Day breakdown for CLAD:*
- **Heartbleed:** Target = `0.995557`, Re-run = `0.990057` (Gap = `-0.005500`, within $\pm 0.01$: **YES**)
- **Web Attack (SQL Injection):** Target = `0.997696`, Re-run = `0.995413` (Gap = `-0.002283`, within $\pm 0.01$: **YES**)

### Table 2: Closed + Open-Set Recognition
| Model | Target Closed Acc | Re-Run Closed Acc | Closed Gap | Target Open AUC | Re-Run Open AUC | Open Gap | Target OpenAUC | Re-Run OpenAUC | OpenAUC Gap | Within $\pm 0.01$? |
|---|---|---|---|---|---|---|---|---|---|---|
| **CLOSR** | 0.995276 | 0.979121 | -0.016155 | 0.974022 | 0.952900 | -0.021122 | 0.969420 | 0.933005 | -0.036415 | **NO** (-0.016 / -0.021 / -0.036) |

---

## 3. Multi-Seed Sensitivity Analysis: Isolation Forest

To determine whether the Isolation Forest gap is due to seed noise or systematic differences, seeds 1, 2, and 3 were evaluated using the exact same split and protocol:

| Metric | Target | Seed 1 | Seed 2 | Seed 3 | 3-Seed Mean $\pm$ Std | Gap vs. Target |
|---|---|---|---|---|---|---|
| **Closed Set Mean** | 0.767531 | 0.803184 | 0.810748 | 0.819874 | $0.811269 \pm 0.008364$ | **+0.043738** |
| **Open Set Mean** | 0.838422 | 0.861697 | 0.870601 | 0.864649 | $0.865649 \pm 0.004535$ | **+0.027227** |

**Finding:** All three seeds exceed the target. The closed gap ranges from +0.036 to +0.052, a spread larger than the $\pm 0.01$ reproduction tolerance; the open gap ranges from +0.023 to +0.032. The upstream seed is unknown in the reference repository. A consistent direction across 3 seeds suggests but does not prove a systematic offset.

---

## 4. Written Explanation of Gaps Outside $\pm 0.01$

1. **CLOSR (Closed Acc: -0.016, Open AUC: -0.021, OpenAUC: -0.036):**
   - *Directly demonstrated:* The model was trained on CPU for only **10 epochs** (`reduced_run=true`) instead of upstream's **200 epochs** (D3.8). The evaluation used the full $N=894,966$ test samples with `val_ratio=0.0`, matching upstream data split protocol exactly.
   - *Unverified hypothesis:* Contrastive margin loss optimization across the remaining 190 epochs would close the remaining gap to the asymptotic Section 2.4 targets.
2. **Autoencoder (Closed: +0.022, Open: +0.033):**
   - *Directly demonstrated:* Re-run AUROC exceeds upstream reported numbers. Upstream defines `SymetricAutoEncoder` with a generic `bottleneck_ratio` in `third_party/CLOSR/baseline_implementations/binary_classification/autoencoder.py:56-98`, but specifies no layer dimensions or runnable training script anywhere in the upstream repository (DV29). Our implementation uses a symmetric `[64, 32] -> 16 -> [32, 64]` bottleneck with AdamW (`lr=1e-3`, `baseline_epochs=10`).
   - *Unverified hypothesis:* That our chosen `[64, 32] -> 16` architecture provides stronger regularization against reconstruction of anomalies than the unrecorded upstream configuration.
3. **Isolation Forest (Closed: +0.044, Open: +0.027):**
   - *Directly demonstrated:* All three seeds exceed the target (closed gap ranges +0.036 to +0.052; open gap ranges +0.023 to +0.032). The upstream seed is not recorded in the upstream files.
   - *Unverified hypothesis:* That the offset is caused by tree-partitioning heuristics, floating-point precision differences, or differences across scikit-learn releases.
4. **MLP Classifier Open Set Mean (+0.182):**
   - *Directly demonstrated:* In upstream Table 1 (`README.md:70`), the MLP Heartbleed AUROC is reported as `0.071957` (raw, no flip), yielding an Open Set Mean of `0.533859`. In our 10-epoch re-run, Heartbleed AUROC is `0.435773` while SQL Injection is `0.995096` (matching upstream target `0.995762`). Upstream provides no training script or hyperparameter specification for MLP (DV30).
   - *Unverified hypothesis:* That our 10-epoch AdamW training schedule did not saturate known attack predictions on Heartbleed as heavily as upstream's model did, or that upstream trained for a different number of epochs.

---

## 5. Artifacts and File Manifest

The CLAD and CLOSR evaluation metrics below are verified from the local evaluation JSONs and per-class CSVs. PR1-T11 stages the small evaluation JSONs and CSVs, together with the generated upstream/re-run comparison CSV and Markdown table, so this evidence is present in a clean checkout after that commit.

The baseline rerun values listed earlier in this staging report were reported during PR1-T08, but their baseline JSON files are not present in the current checkout. They cannot be independently revalidated here and are not used as verified numeric evidence in `docs/reports/PR1_summary.md`. The comparison table generated for PR1-T11 contains the current CLAD/CLOSR evaluation records and upstream-reported rows; it does not claim locally rerun AE, IF, or MLP values.

- **CLAD:**
  - Metrics: `results/eval/clad_seed1_eval.json`
  - Per-class table: `results/eval/clad_seed1_per_class.csv`
  - Raw prediction arrays: `results/eval/clad_seed1_arrays.npz`
  - Training log: `results/train/clad_seed1.json`
- **CLOSR:**
  - Metrics: `results/eval/closr_seed1_eval.json`
  - Per-class table: `results/eval/closr_seed1_per_class.csv`
  - Raw prediction arrays: `results/eval/closr_seed1_arrays.npz`
  - Training log: `results/train/closr_seed1.json`
- **Baselines:**
  - MLP: `results/baselines/mlp_seed1.json`
  - Autoencoder: `results/baselines/autoencoder_seed1.json`
  - Isolation Forest Seed 1: `results/baselines/iforest_seed1.json`
  - Isolation Forest Seed 2: `results/baselines/iforest_seed2.json` / `isolation_forest_seed2.json`
  - Isolation Forest Seed 3: `results/baselines/iforest_seed3.json` / `isolation_forest_seed3.json`
- **Comparison Tables:**
  - `results/baselines/comparison.csv`
  - `results/baselines/comparison.md`

## 6. Tracking Status

The repository `.gitignore` rules still ignore generated evaluation, baseline, training, split, and weight outputs by default. PR1-T11 explicitly tracks only these small result artifacts: `results/eval/clad_seed1_eval.json`, `results/eval/clad_seed1_per_class.csv`, `results/eval/closr_seed1_eval.json`, `results/eval/closr_seed1_per_class.csv`, `results/baselines/comparison.csv`, and `results/baselines/comparison.md`. The evaluation array archives (`*_arrays.npz`), training JSONs, split archives, weight files, and optimizer archives remain local-only and are not required as tracked report evidence. The baseline JSON files previously cited in this staging report are absent from this checkout.

---

## 7. Methodological Protocol Notes

1. **Config Epochs vs. Top-Level Epochs:**
   In the evaluation records (`closr_seed1_eval.json` and `clad_seed1_eval.json`), the inner `config.epochs` displays `200` because the evaluation CLI command loaded the default configuration without passing an explicit `--set epochs=10` override. The top-level `epochs: 10` and `reduced_run: true` metadata represent the authoritative execution parameters of the trained checkpoints.
2. **Zero-Day Sample Counts and Metric Noise:**
   The Lycos2017 dataset holds out 2 zero-day attack classes based on `sample_thres=100`:
   - `Heartbleed`: 11 total rows, all included in zero-day evaluation.
   - `Web Attack (SQL Injection)`: 12 total rows, all included in zero-day evaluation.
   The combined zero-day evaluation set has 23 rows, alongside 894,966 known test rows (894,989 evaluation rows total). Single-row changes can therefore affect per-zero-day AUROCs noticeably.
3. **Stochastic Variation vs. Split Variation:**
   Only seed 1 was run for CLAD, CLOSR, AE, and MLP due to CPU training time limits (~5.6 minutes per MLP run, ~5.0 minutes per CLAD run). Because the data split seed is held strictly fixed at `39058032` (DV6, D4.3), any standard deviation across training seeds reflects initialization and batch-order stochasticity only, not dataset variation.
4. **Staging Status:**
   This file is a staging document for the PR1-T08 reproduction milestone. Its text and tables will be merged into `docs/reports/PR1_summary.md` during PR1-T11.

