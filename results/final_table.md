### Regression results
| Dataset           | Model    |   RMSE |    R² |   Train (s) |   Inf. (μs/sample) |
|:------------------|:---------|-------:|------:|------------:|-------------------:|
| Appliances Energy | CatBoost |  66.96 | 0.552 |        11.1 |                3.5 |
| Appliances Energy | RF       |  72.61 | 0.473 |         1.7 |               41.4 |
| Appliances Energy | XGBoost  |  65.75 | 0.568 |        41.7 |               94.8 |
| Appliances Energy | xRFM     |  65.29 | 0.574 |        11.5 |               26   |
| Seoul Bike        | CatBoost | 225.66 | 0.878 |         5.2 |                1.9 |
| Seoul Bike        | RF       | 245.64 | 0.855 |         0.5 |               57.4 |
| Seoul Bike        | XGBoost  | 232.32 | 0.87  |         5.4 |               37.2 |
| Seoul Bike        | xRFM     | 238.44 | 0.864 |         2.1 |               15.9 |

### Classification results
| Dataset      | Model    |   Accuracy |   AUC-ROC |   Train (s) |   Inf. (μs/sample) |
|:-------------|:---------|-----------:|----------:|------------:|-------------------:|
| Crop Mapping | RF       |     0.9892 |    0.9998 |        12.2 |               23.6 |
| Crop Mapping | XGBoost  |     0.991  |    0.9999 |        40   |               19.6 |
| Crop Mapping | xRFM     |     0.9945 |    0.9999 |        12.8 |                2.6 |
| HCC Survival | CatBoost |     0.7879 |    0.8192 |         0.6 |               46.6 |
| HCC Survival | RF       |     0.8485 |    0.85   |         0.9 |             3666.9 |
| HCC Survival | XGBoost  |     0.8182 |    0.8346 |         0   |               18.7 |
| HCC Survival | xRFM     |     0.7576 |    0.8    |         0.1 |               50.5 |
| IDA2016      | CatBoost |     0.9937 |    0.9918 |         3.1 |                0.5 |
| IDA2016      | RF       |     0.9918 |    0.9893 |         5.8 |               15.1 |
| IDA2016      | XGBoost  |     0.9929 |    0.9928 |        52.1 |               30.2 |
| IDA2016      | xRFM     |     0.9897 |    0.9898 |        16.9 |                5.9 |
