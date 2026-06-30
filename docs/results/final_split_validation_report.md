# Final Split Validation Report

Lock status: `FINAL_LOCKED_BEFORE_MODEL_TRAINING`

Split file: `data/processed/revision/series_split.csv`

| Split | Count | Proportion |
|---|---:|---:|
| train | 700 | 0.700 |
| validation | 150 | 0.150 |
| test | 150 | 0.150 |

Train/validation overlap: 0
Train/test overlap: 0
Validation/test overlap: 0
Static parameter combinations spanning multiple splits: 0

Preprocessing policy: revision benchmark code fits scalers on `split == train` only; validation is used for early stopping/model selection; test is reserved for final evaluation.
