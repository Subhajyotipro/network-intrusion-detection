# DATA_CARD.md — Lycos2017 Dataset Card & Synthetic Fallback Specification

---

## 1. Dataset Status

> **Status:** **VERIFIED (Real Lycos2017 Dataset Present & Validated)**  
> The physical dataset file `lycos.csv` (823 MB, 1,789,954 rows) is located at the workspace root and linked via relative symlink at `data/lycos.csv`.  
> Real-data validation has been executed, confirming schema consistency, zero-day class isolation, and split persistence.  
> The deterministic **synthetic data generator** (`nids.data.generate_synthetic_dataset`) remains available and is used by unit/smoke tests to ensure fast, offline CI execution.

---

## 2. Schema Specification

- **Raw Rows:** **1,789,954**
- **Raw Columns:** **83** (7 identifier columns + 75 features + 1 label)
- **Input Feature Dimension ($d_{in}$):** **72** numeric traffic features.
  - 3 all-zero columns across the dataset are dropped on training split: `flag_urg`, `fwd_flag_urg`, `bwd_flag_urg`.
  - Feature count after zero-variance filtering: $75 - 3 = 72$.
- **Identifier Columns Dropped (7 columns, always removed):**
  - `flow_id`
  - `src_addr`
  - `src_port`
  - `dst_addr`
  - `dst_port`
  - `ip_prot`
  - `timestamp`
- **Target Column:** `label` (categorical string).
- **Benign Class Convention:** `benign` is assigned class index **0** (`BENIGN_ID = 0`).
- **Data Quality:** 0 NaNs and 0 Infs across all feature columns; 319,637 duplicate flow rows exist in the raw capture.

---

## 3. Class Taxonomy & Zero-Day Isolation

Classes with fewer than `sample_thres = 100` instances are held out entirely as **zero-day attacks**. They are completely absent from training and validation splits.

### Real Dataset Class Taxonomy (14 classes: 12 known + 2 zero-day)

| Class Label (String) | Category | Real Total Count | Classification |
|---|---|---|---|
| `benign` | Normal | 1,341,169 | Known (Index 0) |
| `portscan` | Attack | 160,106 | Known (Index 8) |
| `dos_hulk` | Attack | 158,988 | Known (Index 4) |
| `ddos` | Attack | 95,683 | Known (Index 2) |
| `ftp_patator` | Attack | 8,006 | Known (Index 7) |
| `dos_goldeneye` | Attack | 6,765 | Known (Index 3) |
| `ssh_patator` | Attack | 5,918 | Known (Index 9) |
| `dos_slowloris` | Attack | 5,674 | Known (Index 6) |
| `dos_slowhttptest` | Attack | 4,866 | Known (Index 5) |
| `webattack_bruteforce` | Attack | 1,360 | Known (Index 10) |
| `bot` | Attack | 735 | Known (Index 1) |
| `webattack_xss` | Attack | 661 | Known (Index 11) |
| `webattack_sql_injection` | Attack | 12 | **Zero-Day** (< 100 samples) |
| `heartbleed` | Attack | 11 | **Zero-Day** (< 100 samples) |
| **Total Flows** | | **1,789,954** | **12 Known (1,789,931), 2 Zero-Day (23)** |

- **Zero-Day Evaluation Label:** Assigned label `-1` (`UNKNOWN_ID = -1`) in all evaluation arrays. True class names are retained in a parallel `zd_names` array.

---

## 4. Preprocessing & Normalization Protocol

1. **Infinite & Missing Values (DV14):**
   - Any positive or negative infinite values ($\pm\infty$) are converted to `NaN`.
   - Feature medians are computed **strictly on the training split**.
   - `NaN` values across all splits (train, validation, test, zero-day) are imputed using the **training medians** only.
2. **Feature Scaling (DV13):**
   - `StandardScaler` (z-score standardization: $(x - \mu)/\sigma$) is fit **strictly on the training split**.
   - The fitted scaler transforms validation, test, and zero-day sets without updating mean or standard deviation.

---

## 5. Splitting Protocols & Real vs. Synthetic Distributions

- **Split Seed:** `39058032` (fixed upstream seed).
- **Test Ratio:** `0.5` (stratified across known classes).
- **Validation Ratio:** `0.1` (carved from the train remainder after the test split, DV6).
- **Upstream Reproduction Mode:** When `val_ratio = 0.0`, no validation set is carved, producing identical train and test partitions to upstream.

### A. Real Dataset: Upstream Protocol (`test_ratio = 0.5`, `val_ratio = 0.0`)
- **Split Hash:** `d8861d25a38d` (`results/splits/d8861d25a38d.npz`)

| Class | Total Flows | Train ($N=894,965$) | Val ($N=0$) | Test ($N=894,966$) | Zero-Day ($N=23$) |
|---|---|---|---|---|---|
| `benign` | 1,341,169 | 670,584 | 0 | 670,585 | — |
| `bot` | 735 | 368 | 0 | 367 | — |
| `ddos` | 95,683 | 47,841 | 0 | 47,842 | — |
| `dos_goldeneye` | 6,765 | 3,382 | 0 | 3,383 | — |
| `dos_hulk` | 158,988 | 79,494 | 0 | 79,494 | — |
| `dos_slowhttptest` | 4,866 | 2,433 | 0 | 2,433 | — |
| `dos_slowloris` | 5,674 | 2,837 | 0 | 2,837 | — |
| `ftp_patator` | 8,006 | 4,003 | 0 | 4,003 | — |
| `portscan` | 160,106 | 80,053 | 0 | 80,053 | — |
| `ssh_patator` | 5,918 | 2,959 | 0 | 2,959 | — |
| `webattack_bruteforce` | 1,360 | 680 | 0 | 680 | — |
| `webattack_xss` | 661 | 331 | 0 | 330 | — |
| `heartbleed` | 11 | — | — | — | 11 |
| `webattack_sql_injection` | 12 | — | — | — | 12 |
| **Total** | **1,789,954** | **894,965** | **0** | **894,966** | **23** |

### B. Real Dataset: Calibrated Protocol (`test_ratio = 0.5`, `val_ratio = 0.1`)
- **Split Hash:** `c35f2e63a188` (`results/splits/c35f2e63a188.npz`)

| Split | Fraction of Known | Known Samples | Zero-Day Samples | Purpose |
|---|---|---|---|---|
| **Train** | 45% (90% of remainder) | 805,468 | 0 | Model training & representation learning |
| **Validation** | 5% (10% of remainder) | 89,497 | 0 | Open-set threshold & cascade calibration (PR2) |
| **Test** | 50% | 894,966 | 0 | Closed-set & known-class evaluation |
| **Zero-Day** | Held out | 0 | 23 | Unseen zero-day detection evaluation |
| **Total** | 100% | **1,789,931** | **23** | |

### C. Synthetic Dataset Protocol (`num_samples = 200`, `n_features = 72`)

| Split | Samples | Known Classes | Zero-Day Classes | Purpose |
|---|---|---|---|---|
| **Train** | 960 | 12 (80 per class) | 0 | Unit tests & fast offline CI |
| **Validation** | 240 | 12 (20 per class) | 0 | Unit tests for calibration logic |
| **Test** | 1,200 | 12 (100 per class) | 0 | Unit tests for metrics |
| **Zero-Day** | 20 | 0 | 2 (10 per class) | Unit tests for open-set detection |
| **Total** | **2,420** | **12** | **2** | |
