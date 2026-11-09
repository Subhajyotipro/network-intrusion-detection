# Performance Investigation: CLOSR Training Speed & Contrastive Loss Optimization

## 1. Machine Baseline (Apple M1, 8 Cores, 8 GB RAM)
- **CPU Brand:** Apple M1 (8 cores: 4 performance + 4 efficiency)
- **RAM:** 8.0 GB LPDDR4X (unified memory)
- **TensorFlow Threading:** `intra_op_parallelism_threads = 0`, `inter_op_parallelism_threads = 0` (uses all 8 logical cores)
- **4096 × 4096 Float32 Matmul Benchmark (`tf.matmul` in `@tf.function`):**
  - FLOPs: $2 \times 4096^3 = 137.44\text{ GFLOP}$ per matmul
  - Mean latency: **0.8497 seconds**
  - Mean throughput: **161.7 GFLOPS** (peak: 190.5 GFLOPS)
  - Thermal drift across 10 continuous runs: Run 1 (1.14s) -> Runs 6-10 (0.72s - 0.76s); fanless M1 sustains ~180-190 GFLOPS once CPU clock frequency ramps.

## 2. Memory & System Swap Pressure
- **Initial Swap State:** `vm.swapusage`: total = 6144.00 MB, **used = 5399.94 MB**, free = 744.06 MB.
- **Memory Pressure:** With 8 GB physical RAM and 5.4 GB swap already in use, allocating large intermediate tensors directly triggers macOS memory compressor (`Pages stored in compressor: 1,141,159` ~18.7 GB) and page faults.
- **Intermediate Tensor Sizes (Batch 4096):**
  - Pairwise similarity matrix $[12, 4096, 4096]$ float32: **805.3 MB**
  - Distance matrix $[12, 4096, 4096]$ float32: **805.3 MB**
  - GradientTape activation cache (72 classwise intermediate tensors in unvectorized code): **>4.8 GB**!

## 3. Step Breakdown at Batch 4096 (Inside `tf.function` Before Optimization)
- **Model Forward Pass:** 0.1696s (1.4%)
- **Loss Forward Pass Only:** 1.5518s (13.2%)
- **Loss Forward + Backward Pass:** 8.4863s (72.0%)
- **Full Train Step (Compiled):** 11.7826s (100.0%)
- **Full Train Step (Eager execution + redundant diagnostic pass):** **28.36s**

## 4. Root Causes of Initial 28 s Step Latency
1. **Uncompiled Python Eager Execution:** Training loop in `train.py` executed without graph compilation, incurring Python runtime dispatch and allocation overhead per operation on 201 million matrix elements.
2. **Redundant Diagnostic Forward Pass:** On every batch, `_run_epoch` called `closr_fraction_pos`, recomputing all 12 class distance matrices and margins from scratch.
3. **Classwise Dynamic Masking & Allocations:** For each of the 12 classes, `_contrastive_loss_calc` allocated `[4096, 4096]` boolean equality matrices, `tf.where` selections, `tf.fill` margin tensors, and individual `tf.nn.relu` calls. Autodiff held all 12 intermediate matrices on the tape (>4.8 GB), causing heavy swap activity.
4. **Low Arithmetic Intensity & Memory Bandwidth Bound:** While the batched matmul $(12, 4096, 64) \times (12, 64, 4096)$ requires 25.8 GFLOPs (~0.16s), the subsequent elementwise operations and tape adjoint transfers stream 805 MB multiple times. At M1 memory bandwidth (~68 GB/s), streaming this data takes 1.5–2.5 seconds.

## 5. Algorithmic & Implementation Optimizations
1. **Quadratic Matrix-Vector Formulation (BLAS GEMV):**
   - Recognized that similar loss $\sum_{i, j} d_{ij}^2 S_{c, ij} = v_c^T D^2 v_c$ where $v_c = \text{one\_hot}(y)[:, c]$.
   - Evaluated across all $C=12$ classes in 4 batched matrix-vector multiplications ($[12, 4096, 4096] \times [12, 4096, 1]$) with zero $[B, B]$ boolean mask allocations.
   - Eliminated `tf.fill`, `tf.where`, and 12 separate `relu` calls; elementwise ReLU and square are evaluated once on the shared distance matrix.
2. **Single-Pass Loss and Diagnostic Fraction (`closr_loss_and_frac`):**
   - Returns both `loss` and `frac_pos` in the same forward pass without recomputing distance matrices.
3. **Compiled `@tf.function` Training Step:**
   - Compiled `step_fn = tf.function(_train_step)` once per training run, eliminating eager dispatch.

## 6. Before / After Comparison on Real Lycos2017 Data (Batch 4096)
| Metric | Before Optimization | After Optimization | Improvement |
|---|---|---|---|
| **Per-Step Train Time** | **28.36 s / step** | **5.02 s / step** (steady-state: 3.7 - 4.2 s) | **5.65x speedup** |
| **5-Batch Train Time** | 141.79 s | 25.08 s | **5.65x speedup** |
| **Real Smoke Run Time** | 197.0 s | 42.0 s | **4.69x speedup** |
| **Peak RSS Memory** | 3,386.3 MB | 3,198.9 MB | **-187 MB** |
| **Estimated 1 Full Epoch** | 6,181.9 s (103.0 min / 1.72 hrs) | **1,093.4 s (18.2 min / 0.30 hrs)** | **1.42 hours faster** |
| **Estimated 10 Epochs** | 17.17 hours | **3.04 hours** | **14.13 hours faster** |
| **Estimated 200 Epochs** | 343.44 hours (14.3 days) | **60.74 hours (2.53 days)** | **11.8 days faster** |

## 7. Numerical Equivalence & Verification
- Unit test suite: `pytest -x -q -m "not real"` -> **112 passed**
- Numerical match with independent NumPy float64 reference: difference $< 3 \times 10^{-8}$ (`assert np.isclose(..., rtol=1e-5, atol=1e-6)`).
- Real data regression suite: `pytest -m real -q` -> **2 passed**.
