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
"""Independent pure-NumPy float64 reference implementation of CLAD and CLOSR loss formulas.

Transcribed directly from upstream CLOSR PyTorch source text:
- third_party/CLOSR/losses/clad_loss.py
- third_party/CLOSR/losses/closr_loss.py
- third_party/CLOSR/util/distance.py
Used strictly in the test tier for numerical equivalence verification without PyTorch execution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def np_cossim(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    """NumPy transcription of upstream cossim/cosdist in util/distance.py."""
    a = a.astype(np.float64)
    a_norm = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    if b is not None:
        b = b.astype(np.float64)
        b_norm = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    else:
        b_norm = a_norm.copy()
    similarity = a_norm @ b_norm.T
    dists = (1.0 - similarity) / 2.0
    if b is None:
        np.fill_diagonal(dists, 0.0)
    return dists


def np_loss_calc(
    dists: np.ndarray,
    y: np.ndarray,
    target_class: int,
    margin: float = 1.0,
    eps: float = 1e-8,
    squared: bool = True,
    alpha: float = 0.5,
) -> tuple[float, float]:
    """NumPy transcription of upstream clad_calc / loss_calc in losses/."""
    dists = dists.astype(np.float64)
    y = np.asarray(y, dtype=np.int64)

    # In upstream:
    # eq_pairs = T.eq(T.unsqueeze(y, 0), T.unsqueeze(y, 1))
    # sim_mask = eq_pairs & (unsqueeze(y, 0) == target_class)
    # dissim_mask = (~eq_pairs) & (unsqueeze(y, 1) == target_class)
    y_row = y[None, :]  # shape [1, B]
    y_col = y[:, None]  # shape [B, 1]
    eq_pairs = y_row == y_col

    sim_mask = eq_pairs & (y_row == target_class)
    dissim_mask = (~eq_pairs) & (y_col == target_class)

    sim_dists = dists * sim_mask
    n_sim = np.sum((sim_dists > eps).astype(np.float64))

    sim_dists_calc = (sim_dists**2) if squared else sim_dists
    sim_loss = np.sum(sim_dists_calc) / (n_sim + eps)

    margin_tensor = np.full(dists.shape, margin, dtype=np.float64)
    dissim_dists = np.where(dissim_mask, dists, margin_tensor)

    dissim_diff = np.maximum(0.0, margin - dissim_dists)
    dissim_diff_calc = (dissim_diff**2) if squared else dissim_diff
    n_dissim = np.sum((dissim_diff_calc > eps).astype(np.float64))
    dissim_loss = np.sum(dissim_diff_calc) / (n_dissim + eps)

    total_dissim_pairs = np.sum(dissim_mask.astype(np.float64))
    frac_pos = n_dissim / (total_dissim_pairs + eps)

    loss = (alpha * sim_loss) + ((1.0 - alpha) * dissim_loss)
    return float(loss), float(frac_pos)


def np_clad_loss(
    x: np.ndarray,
    y: np.ndarray,
    target_class: int = 0,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> float:
    """Compute CLAD loss using pure NumPy float64 reference implementation."""
    dists = np_cossim(x)
    loss, _ = np_loss_calc(
        dists=dists,
        y=y,
        target_class=target_class,
        margin=margin,
        eps=eps,
        squared=squared,
        alpha=alpha,
    )
    return loss


def np_clad_fraction_pos(
    x: np.ndarray,
    y: np.ndarray,
    target_class: int = 0,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> float:
    """Compute fraction of positive dissimilar pairs using pure NumPy float64 reference."""
    dists = np_cossim(x)
    _, frac = np_loss_calc(
        dists=dists,
        y=y,
        target_class=target_class,
        margin=margin,
        eps=eps,
        squared=squared,
        alpha=alpha,
    )
    return frac


def np_closr_loss(
    x: np.ndarray,
    y: np.ndarray,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Compute CLOSR loss across all classes using pure NumPy float64 reference."""
    x = x.astype(np.float64)
    # x shape: [B, C, D]
    n_classes = x.shape[1]
    losses = []
    for c in range(n_classes):
        xc = x[:, c, :]
        dists_c = np_cossim(xc)
        l_c, _ = np_loss_calc(
            dists=dists_c,
            y=y,
            target_class=c,
            margin=margin,
            eps=eps,
            squared=squared,
            alpha=alpha,
        )
        losses.append(l_c)
    return float(np.mean(losses))


def np_closr_fraction_pos(
    x: np.ndarray,
    y: np.ndarray,
    margin: float = 1.0,
    squared: bool = True,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Compute CLOSR mean fraction of positive pairs across all classes using NumPy reference."""
    x = x.astype(np.float64)
    n_classes = x.shape[1]
    fracs = []
    for c in range(n_classes):
        xc = x[:, c, :]
        dists_c = np_cossim(xc)
        _, f_c = np_loss_calc(
            dists=dists_c,
            y=y,
            target_class=c,
            margin=margin,
            eps=eps,
            squared=squared,
            alpha=alpha,
        )
        fracs.append(f_c)
    return float(np.mean(fracs))
