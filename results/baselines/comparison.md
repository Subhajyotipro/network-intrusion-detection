### Table 1: Binary AUROC vs Benign (One-vs-Benign)
| model | protocol | closed_set_mean | open_set_mean | source | reduced_run | seed | data_source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AUTOENCODER (re-run) | binary AUROC vs benign (Table 1) | 0.874429 | 0.857935 | re-run | True | 1 | real |
| IFOREST (re-run) | binary AUROC vs benign (Table 1) | 0.803184 | 0.861697 | re-run | True | 1 | real |
| MLP (re-run) | binary AUROC vs benign (Table 1) | 0.999396 | 0.715434 | re-run | True | 1 | real |
| CLAD (re-run) | binary AUROC vs benign (Table 1) | 0.999278 | 0.992735 | re-run | True | 1 | real |
| CLAD (upstream) | binary AUROC vs benign (Table 1) | 0.999855 | 0.996627 | upstream-reported | False | n/a | n/a |
| AE (upstream) | binary AUROC vs benign (Table 1) | 0.857221 | 0.840659 | upstream-reported | False | n/a | n/a |
| IF (upstream) | binary AUROC vs benign (Table 1) | 0.767531 | 0.838422 | upstream-reported | False | n/a | n/a |
| MLP (upstream) | binary AUROC vs benign (Table 1) | 0.999557 | 0.533859 | upstream-reported | False | n/a | n/a |

### Table 2: Closed + Open-Set Recognition
| model | protocol | closed_set_acc | open_set_auc | open_auc | source | reduced_run | seed | data_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLOSR (re-run) | closed+open-set (Table 2) | 0.979121 | 0.952900 | 0.933005 | re-run | True | 1 | real |
| CLOSR (upstream) | closed+open-set (Table 2) | 0.995276 | 0.974022 | 0.969420 | upstream-reported | False | n/a | n/a |
| MultiStage (upstream) | closed+open-set (Table 2) | 0.996612 | 0.801251 | 0.798537 | upstream-reported | False | n/a | n/a |
| DOC (upstream) | closed+open-set (Table 2) | 0.995536 | 0.570263 | 0.567717 | upstream-reported | False | n/a | n/a |
| OPENMAX (upstream) | closed+open-set (Table 2) | 0.995615 | 0.720174 | 0.717016 | upstream-reported | False | n/a | n/a |
| CRSOR (upstream) | closed+open-set (Table 2) | 0.994940 | 0.748295 | 0.744509 | upstream-reported | False | n/a | n/a |
| Siamese Network (upstream) | closed+open-set (Table 2) | 0.997722 | 0.720811 | 0.719167 | upstream-reported | False | n/a | n/a |
