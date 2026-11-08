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
"""Unit tests for WarmupCosineSchedule against independent NumPy reference."""

import numpy as np
import pytest

from nids.config import Config
from nids.train import make_schedule
from tests.unit.np_reference import np_warmup_cosine_schedule


@pytest.mark.parametrize("total_steps", [50, 100, 200])
@pytest.mark.parametrize("warmup_frac", [0.05, 0.1, 0.2])
def test_schedule_matches_numpy_reference_at_every_step(total_steps: int, warmup_frac: float):
    """Verify make_schedule matches independent NumPy reference at every step."""
    cfg = Config(lr_init=1e-6, lr_peak=1e-5, lr_final=1e-6, warmup_frac=warmup_frac)
    schedule = make_schedule(cfg, total_steps)
    warmup_steps = int(warmup_frac * total_steps)

    for step in range(total_steps + 10):
        actual = schedule(step)
        expected = np_warmup_cosine_schedule(
            step=step,
            start_val=cfg.lr_init,
            end_val=cfg.lr_final,
            ref_val=cfg.lr_peak,
            T_max=total_steps,
            warmup_steps=warmup_steps,
            plateau=True,
        )
        assert np.isclose(actual, expected, rtol=1e-6, atol=1e-12)


def test_schedule_boundary_values_and_monotonicity():
    """Verify schedule at step 0, end of warmup, last step, and monotonicity trends."""
    total_steps = 100
    warmup_frac = 0.1
    cfg = Config(lr_init=1e-6, lr_peak=1e-5, lr_final=1e-6, warmup_frac=warmup_frac)
    schedule = make_schedule(cfg, total_steps)
    warmup_steps = int(warmup_frac * total_steps)  # 10

    # Step 0: lr_init
    assert np.isclose(schedule(0), cfg.lr_init)

    # End of warmup: lr_peak
    assert np.isclose(schedule(warmup_steps), cfg.lr_peak)

    # Last step: lr_final
    assert np.isclose(schedule(total_steps), cfg.lr_final)

    # Warmup phase is monotonically non-decreasing
    warmup_vals = [schedule(s) for s in range(1, warmup_steps + 1)]
    assert all(warmup_vals[i] <= warmup_vals[i + 1] for i in range(len(warmup_vals) - 1))

    # Annealing phase is monotonically non-increasing
    decay_vals = [schedule(s) for s in range(warmup_steps, total_steps + 1)]
    assert all(decay_vals[i] >= decay_vals[i + 1] for i in range(len(decay_vals) - 1))

    # Past total_steps plateaus at lr_final
    assert np.isclose(schedule(total_steps + 5), cfg.lr_final)


def test_schedule_edge_cases_small_steps():
    """Verify schedule behaves stably with tiny total steps."""
    cfg = Config(lr_init=1e-6, lr_peak=1e-5, lr_final=1e-6, warmup_frac=0.1)
    schedule_tiny = make_schedule(cfg, total_steps=2)
    assert np.isfinite(schedule_tiny(0))
    assert np.isfinite(schedule_tiny(1))
    assert np.isfinite(schedule_tiny(2))
    assert np.isfinite(schedule_tiny(3))
