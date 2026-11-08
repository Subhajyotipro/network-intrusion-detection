# DATA_CARD.md — Lycos2017 Dataset Card & Synthetic Fallback Specification

---

## 1. Dataset Status

> **Status:** **BLOCKED (Real Dataset Absent)**  
> The physical dataset file `data/lycos.csv` was not present in the workspace environment when this task was executed.  
> Per **AGENTS.md §5.3**, execution proceeded using the deterministic **synthetic data generator** (`nids.data.generate_synthetic_dataset`), which mirrors the exact feature dimensionality, column schema, and label taxonomy of Lycos2017.
>
> **Human Action Required:** To train or evaluate on real network traffic, place the preprocessed Lycos2017 CSV file at:
> ```
> data/lycos.csv
> ```

---

## 2. Schema Specification

- **Input Feature Dimension ($d_{in}$):** **72** numeric traffic features.
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

---

## 3. Class Taxonomy & Zero-Day Isolation

Classes with fewer than `sample_thres = 100` instances are held out entirely as **zero-day attacks**. They are completely absent from training and validation splits:

### Known Classes (12 classes: benign + 11 attacks)
1. `benign` (index 0)
2. `Botnet`
3. `DDoS`
4. `DoS (Golden Eye)`
5. `DoS (Hulk)`
6. `DoS (Slow HTTP Test)`
7. `DoS (Slow Loris)`
8. `FTP Patator`
9. `Portscan`
10. `SSH Patator`
11. `Web Attack (Brute Force)`
12. `Web Attack (XSS)`

### Zero-Day Classes (2 classes held out)
- `Heartbleed`
- `Web Attack (SQL Injection)`
- **Evaluation Label:** Assigned label `-1` (`UNKNOWN_ID = -1`) in all evaluation arrays. True class names are retained in a parallel `zd_names` array.

---

## 4. Preprocessing & Normalization Protocol

1. **Infinite & Missing Values:**
   - Any positive or negative infinite values ($\pm\infty$) are converted to `NaN`.
   - Feature medians are computed **strictly on the training split**.
   - `NaN` values across all splits (train, validation, test, zero-day) are imputed using the **training medians** only.
2. **Feature Scaling:**
   - `StandardScaler` (z-score standardization: $(x - \mu)/\sigma$) is fit **strictly on the training split**.
   - The fitted scaler transforms validation, test, and zero-day sets without updating mean or standard deviation.

---

## 5. Splitting Protocol & Class Counts

- **Split Seed:** `39058032` (fixed upstream seed).
- **Test Ratio:** `0.5` (stratified across known classes).
- **Validation Ratio:** `0.1` (carved from the train remainder after the test split, DV6).
- **Upstream Reproduction Mode:** When `val_ratio = 0.0`, no validation set is carved, producing identical train and test partitions to upstream.

### Synthetic Dataset Class Counts per Split (`val_ratio = 0.1`)

| Class | Total Samples | Train ($N=960$) | Val ($N=240$) | Test ($N=1200$) | Zero-Day ($N=60$) |
|---|---|---|---|---|---|
| `benign` | 200 | 80 | 20 | 100 | — |
| `Botnet` | 200 | 80 | 20 | 100 | — |
| `DDoS` | 200 | 80 | 20 | 100 | — |
| `DoS (Golden Eye)` | 200 | 80 | 20 | 100 | — |
| `DoS (Hulk)` | 200 | 80 | 20 | 100 | — |
| `DoS (Slow HTTP Test)` | 200 | 80 | 20 | 100 | — |
| `DoS (Slow Loris)` | 200 | 80 | 20 | 100 | — |
| `FTP Patator` | 200 | 80 | 20 | 100 | — |
| `Portscan` | 200 | 80 | 20 | 100 | — |
| `SSH Patator` | 200 | 80 | 20 | 100 | — |
| `Web Attack (Brute Force)` | 200 | 80 | 20 | 100 | — |
| `Web Attack (XSS)` | 200 | 80 | 20 | 100 | — |
| `Heartbleed` | 30 | — | — | — | 30 |
| `Web Attack (SQL Injection)` | 30 | — | — | — | 30 |
| **Total** | **2,460** | **960** | **240** | **1,200** | **60** |

*(When real dataset `data/lycos.csv` is provided, this table will be updated with empirical counts from the raw Lycos2017 capture).*
