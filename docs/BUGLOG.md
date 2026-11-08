# Bug Log

| ID | task | symptom | root cause | fix | guarding test | iterations |
|---|---|---|---|---|---|---|
| BUG-001 | PR1-T04 | TF loss and NumPy reference differed by ~0.008 on CLAD / CLOSR | Float32 roundoff in cosine self-distance `(1 - u·u)/2` on unit vectors evaluates to ~3e-8, exceeding `eps=1e-8` in float32 and erroneously counting diagonal self-pairs in `n_sim` | Explicitly zero diagonal of self-pairwise distance matrices via `tf.linalg.set_diag`; independent float64 NumPy reference naturally satisfies `D_ii < eps` (~1e-16) | `test_clad_loss_matches_numpy_reference`, `test_closr_loss_matches_numpy_reference`, `test_identical_batch_finite_loss_and_no_self_pairs` | 2 |

