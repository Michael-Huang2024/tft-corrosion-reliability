# Pretraining Lock Check

Overall status: `PASS`

| Check | Passed | Detail |
|---|---|---|
| `PARAMETER_LOCK_STATUS` | True | `FINAL_LOCKED_BEFORE_MODEL_TRAINING` |
| `SELECTED_PARAMETER_CANDIDATE` | True | `C` |
| `final_dataset_exists` | True | `D:\论文1-Pf(t)\tft-corrosion-reliability-main\data\processed\revision\final_chloride_labeled.parquet` |
| `split_exists` | True | `D:\论文1-Pf(t)\tft-corrosion-reliability-main\data\processed\revision\series_split.csv` |
| `split_counts` | True | `{'train': 700, 'validation': 150, 'test': 150}` |
| `target_column` | True | `onset_flag` |
| `C_th_in_point_inputs` | True | `['Cs', 'D28', 'm_aging', 'cover_mm', 'C_th', 'time_idx', 't_year']` |
| `C_th_in_tft_static` | True | `['Cs', 'D28', 'm_aging', 'cover_mm', 'C_th']` |
| `forbidden_point_predictors_absent` | True | `[]` |
| `chloride_rebar_excluded` | True | `None` |
| `instantaneous_labels_excluded` | True | `['Cs', 'D28', 'm_aging', 'cover_mm', 'C_th', 'time_idx', 't_year']` |

No data regeneration was performed by this check.
No parameter distribution was changed by this check.
