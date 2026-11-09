# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data loading, preprocessing, splits, samplers, and synthetic data generation."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

BENIGN_ID: int = 0
UNKNOWN_ID: int = -1
IDENTIFIER_COLS: list[str] = (
    "flow_id,src_addr,src_port,dst_addr,dst_port,ip_prot,timestamp"
).split(",")

KNOWN_CLASSES: list[str] = (
    "benign,bot,ddos,dos_goldeneye,dos_hulk,dos_slowhttptest,"
    "dos_slowloris,ftp_patator,portscan,ssh_patator,webattack_bruteforce,webattack_xss"
).split(",")
ZERO_DAY_CLASSES: list[str] = ["heartbleed", "webattack_sql_injection"]


@dataclass
class ClassMap:
    """Class label mapping container with known classes and held-out zero-day classes."""

    names: list[str]
    zero_day_names: list[str]

    @property
    def n_known(self) -> int:
        return len(self.names)

    @property
    def n_zero_day(self) -> int:
        return len(self.zero_day_names)


@dataclass
class DatasetSplits:
    """Container for processed dataset splits, class mapping, and feature metadata."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: Optional[np.ndarray]
    y_val: Optional[np.ndarray]
    x_test: Optional[np.ndarray]
    y_test: Optional[np.ndarray]
    x_zd: Optional[np.ndarray]
    y_zd: Optional[np.ndarray]
    zd_names: Optional[np.ndarray]
    class_map: ClassMap
    split_indices: dict[str, np.ndarray]
    d_in: int


def _clean_and_scale(x_train: np.ndarray, *others: Optional[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Fit zero-mask, imputer, and scaler on train only; transform all passed arrays (DV13)."""
    mask = ~np.all(x_train == 0, axis=0)

    def _prep(arr: np.ndarray, med: np.ndarray) -> np.ndarray:
        c = np.where(np.isneginf(arr[:, mask]) | np.isposinf(arr[:, mask]), np.nan, arr[:, mask])
        inds = np.where(np.isnan(c))
        c[inds] = np.take(med, inds[1])
        return c

    x_raw = np.where(
        np.isneginf(x_train[:, mask]) | np.isposinf(x_train[:, mask]), np.nan, x_train[:, mask]
    )
    medians = np.nan_to_num(np.nanmedian(x_raw, axis=0), nan=0.0)
    scaler = StandardScaler()
    scaled_tr = scaler.fit_transform(_prep(x_train, medians))
    scaled_others = [None if x is None else scaler.transform(_prep(x, medians)) for x in others]
    return (scaled_tr, *scaled_others)


def save_split_indices(indices: dict[str, np.ndarray], out_path: str | Path) -> Path:
    """Save split indices dictionary to an NPZ archive."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **indices)
    return p


def load_split_indices(archive_path: str | Path) -> dict[str, np.ndarray]:
    """Load split indices from an NPZ archive."""
    data = np.load(archive_path)
    return {k: data[k] for k in data.files}


def _split_train_val_test(
    n_samples: int, y: np.ndarray, test_ratio: float, val_ratio: float, seed: int
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute stratified indices for test and validation carved from train remainder."""
    all_indices = np.arange(n_samples)
    if test_ratio <= 0.0:
        return all_indices, None, None
    tr_val, te = train_test_split(
        all_indices, test_size=test_ratio, random_state=seed, shuffle=True, stratify=y
    )
    if val_ratio <= 0.0:
        return tr_val, None, te
    eff_val = val_ratio / (1.0 - test_ratio)
    tr, val = train_test_split(
        tr_val, test_size=eff_val, random_state=seed + 1, shuffle=True, stratify=y[tr_val]
    )
    return tr, val, te


def _read_features_and_labels(
    data_source: str | Path | pd.DataFrame, drop_cols: list[str], target_col: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Read CSV or DataFrame, drop identifier columns, and extract features and labels."""
    df = data_source if isinstance(data_source, pd.DataFrame) else pd.read_csv(data_source)
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    y_raw = df[target_col].astype(str).to_numpy()
    x_raw = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
    return x_raw, y_raw, len(df)


def _split_known_and_zero_day(
    x_raw: np.ndarray, y_raw: np.ndarray, sample_thres: int
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], ClassMap]:
    """Separate classes below sample_thres into zero-day and encode known labels."""
    classes, counts = np.unique(y_raw, return_counts=True)
    zd_mask = np.isin(y_raw, classes[counts < sample_thres])
    x_k, y_k_raw = x_raw[~zd_mask], y_raw[~zd_mask]
    x_zd = x_raw[zd_mask] if np.any(zd_mask) else None
    zd_names = y_raw[zd_mask] if np.any(zd_mask) else None

    class_names = ["benign"] + sorted([c for c in np.unique(y_k_raw) if c != "benign"])
    lmap = {name: idx for idx, name in enumerate(class_names)}
    y_k = np.array([lmap[c] for c in y_k_raw], dtype=np.int64)
    zd_names_list = sorted(list(set(classes[counts < sample_thres])))
    return x_k, y_k, x_zd, zd_names, ClassMap(class_names, zd_names_list)


def load_data(
    data_path: str | Path | pd.DataFrame,
    drop_cols: Optional[list[str]] = None,
    target_col: str = "label",
    sample_thres: int = 100,
    split_seed: int = 39058032,
    test_ratio: float = 0.5,
    val_ratio: float = 0.0,
    results_dir: Optional[str | Path] = None,
) -> DatasetSplits:
    """Load dataset, hold out zero-day classes, stratify splits, and scale features."""
    drop = IDENTIFIER_COLS if drop_cols is None else drop_cols
    x_raw, y_raw, n_rows = _read_features_and_labels(data_path, drop, target_col)
    x_k, y_k, x_zd, zd_names, cmap = _split_known_and_zero_day(x_raw, y_raw, sample_thres)
    y_zd = np.full(len(zd_names), UNKNOWN_ID, dtype=np.int64) if zd_names is not None else None

    tr_idx, val_idx, te_idx = _split_train_val_test(
        len(y_k), y_k, test_ratio, val_ratio, split_seed
    )
    x_tr, y_tr = x_k[tr_idx], y_k[tr_idx]
    x_v, y_v = (x_k[val_idx], y_k[val_idx]) if val_idx is not None else (None, None)
    x_te, y_te = (x_k[te_idx], y_k[te_idx]) if te_idx is not None else (None, None)

    x_tr, x_v, x_te, x_zd = _clean_and_scale(x_tr, x_v, x_te, x_zd)
    splits = {"train": tr_idx, **({"val": val_idx} if val_idx is not None else {})}
    if te_idx is not None:
        splits["test"] = te_idx

    if results_dir is not None:
        h = hashlib.sha256(
            f"{n_rows}_{split_seed}_{test_ratio}_{val_ratio}_{sample_thres}".encode()
        ).hexdigest()[:12]
        save_split_indices(splits, Path(results_dir) / "splits" / f"{h}.npz")

    return DatasetSplits(
        x_tr, y_tr, x_v, y_v, x_te, y_te, x_zd, y_zd, zd_names, cmap, splits, x_tr.shape[1]
    )


def generate_synthetic_dataset(
    n_samples_known: int = 200,
    n_samples_zd: int = 30,
    n_features: int = 72,
    seed: int = 39058032,
    known_classes: Optional[list[str]] = None,
    zero_day_classes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Generate a synthetic DataFrame conforming to the Lycos2017 schema and label space."""
    rng = np.random.default_rng(seed)
    known = KNOWN_CLASSES if known_classes is None else known_classes
    zd = ZERO_DAY_CLASSES if zero_day_classes is None else zero_day_classes

    dfs: list[pd.DataFrame] = []
    flow_counter = 1

    for label in known + zd:
        count = n_samples_zd if label in zd else n_samples_known
        feats = rng.standard_normal((count, n_features), dtype=np.float32)
        # Class-specific mean shift to simulate cluster structure
        class_hash = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16) % 1000
        feats += class_hash / 100.0

        df_class = pd.DataFrame(feats, columns=[f"feature_{i}" for i in range(n_features)])
        df_class["flow_id"] = [f"flow_{flow_counter + i}" for i in range(count)]
        df_class["src_addr"], df_class["dst_addr"] = "192.168.1.10", "10.0.0.1"
        df_class["src_port"] = rng.integers(1024, 65535, size=count)
        df_class["dst_port"] = 80
        df_class["ip_prot"], df_class["timestamp"] = 6, "2017-07-05 09:00:00"
        df_class["label"] = label
        flow_counter += count
        dfs.append(df_class)

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_synthetic_data(
    cfg: Optional[Any] = None,
    num_samples: int = 200,
    seed: int = 39058032,
    d_in: int = 72,
    val_ratio: float = 0.0,
) -> DatasetSplits:
    """Generate synthetic DatasetSplits for testing, smoke runs, and synthetic fallback."""
    s = getattr(cfg, "seed", seed)
    feat_dim = getattr(cfg, "d_in", None) or d_in or 72
    vr = getattr(cfg, "val_ratio", val_ratio)
    st = getattr(cfg, "sample_thres", 100)
    st = max(2, num_samples // 2) if num_samples <= st else st
    df = generate_synthetic_dataset(
        n_samples_known=num_samples,
        n_samples_zd=max(1, num_samples // 10),
        n_features=feat_dim,
        seed=s,
    )
    return load_data(df, sample_thres=st, val_ratio=vr, split_seed=s)


def class_balanced_sampler(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int = 42,
    drop_last: bool = False,
    max_batches: Optional[int] = None,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yield class-balanced mini-batches with replacement, safe for small or skewed classes."""
    n_samples = len(x)
    if n_samples == 0:
        return

    eff_batch_size = min(batch_size, n_samples)
    if eff_batch_size <= 0:
        return

    classes, counts = np.unique(y, return_counts=True)
    class_weight_map = {c: 1.0 / count for c, count in zip(classes, counts)}
    sample_weights = np.array([class_weight_map[label] for label in y], dtype=np.float64)
    probs = sample_weights / sample_weights.sum()

    rng = np.random.default_rng(seed)
    batches_yielded = 0
    step_ceil = int(np.ceil(n_samples / eff_batch_size))
    total_steps = max(1, n_samples // eff_batch_size if drop_last else step_ceil)

    while max_batches is None or batches_yielded < max_batches:
        idx = rng.choice(n_samples, size=eff_batch_size, replace=True, p=probs)
        yield x[idx], y[idx]
        batches_yielded += 1
        if max_batches is None and batches_yielded >= total_steps:
            break
