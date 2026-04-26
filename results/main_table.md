# Main Results

## appliances_energy (regression, n_train=11841, d=27)
| model         | test_rmse   | test_r2   | train_time_s   | inf_s_per_sample   |
|:--------------|:------------|:----------|:---------------|:-------------------|
| xrfm          | 65.2940     | 0.5740    | 11.5           | 26.0 μs            |
| xgboost       | 65.7484     | 0.5680    | 41.7           | 94.8 μs            |
| random_forest | 72.6140     | 0.4731    | 1.7            | 41.4 μs            |
| tabpfn        | -           | -         | -              | -                  |
| catboost      | 66.9635     | 0.5519    | 11.1           | 3.5 μs             |

## crop_mapping (multiclass, n_train=30000, d=174)
| model         | test_accuracy   | test_auc_roc   | train_time_s   | inf_s_per_sample   |
|:--------------|:----------------|:---------------|:---------------|:-------------------|
| xrfm          | 0.1202          | -              | 13.0           | 2.6 μs             |
| random_forest | 0.9892          | 0.9998         | 12.2           | 23.6 μs            |
| tabpfn        | -               | -              | -              | -                  |

## hcc_survival (binary, n_train=99, d=49)
| model         | test_accuracy   | test_auc_roc   | train_time_s   | inf_s_per_sample   |
|:--------------|:----------------|:---------------|:---------------|:-------------------|
| xrfm          | 0.7576          | 0.8000         | 0.1            | 50.5 μs            |
| xgboost       | 0.8182          | 0.8346         | 0.0            | 18.7 μs            |
| random_forest | 0.8485          | 0.8500         | 0.9            | 3666.9 μs          |
| tabpfn        | -               | -              | -              | -                  |
| catboost      | 0.7879          | 0.8192         | 0.6            | 46.6 μs            |

## ida2016 (binary, n_train=36000, d=170)
| model         | test_accuracy   | test_auc_roc   | train_time_s   | inf_s_per_sample   |
|:--------------|:----------------|:---------------|:---------------|:-------------------|
| xgboost       | 0.9929          | 0.9928         | 52.1           | 30.2 μs            |
| random_forest | 0.9918          | 0.9893         | 5.8            | 15.1 μs            |
| tabpfn        | -               | -              | -              | -                  |
| catboost      | 0.9937          | 0.9918         | 3.1            | 0.5 μs             |

## seoul_bike (regression, n_train=5256, d=12)
| model         | test_rmse   | test_r2   | train_time_s   | inf_s_per_sample   |
|:--------------|:------------|:----------|:---------------|:-------------------|
| xrfm          | 238.4426    | 0.8635    | 2.1            | 15.9 μs            |
| xgboost       | 232.3219    | 0.8705    | 5.4            | 37.2 μs            |
| random_forest | 245.6418    | 0.8552    | 0.5            | 57.4 μs            |
| tabpfn        | -           | -         | -              | -                  |
| catboost      | 225.6604    | 0.8778    | 5.2            | 1.9 μs             |
