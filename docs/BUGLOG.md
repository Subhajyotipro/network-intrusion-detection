# Bug Log

| ID | task | symptom | root cause | fix | guarding test | iterations |
|---|---|---|---|---|---|---|
| BUG-001 | PR1-T04 | TF loss and NumPy reference differed by ~0.008 on CLAD / CLOSR | Float32 roundoff in cosine self-distance `(1 - u·u)/2` caused diagonal elements to be ~3e-8, exceeding `eps=1e-8` and erroneously counting self-pairs in `n_sim` | Explicitly zero diagonal of self-pairwise distance matrices via `tf.linalg.set_diag` | `test_clad_loss_matches_numpy_reference`, `test_closr_loss_matches_numpy_reference` | 1 |

