# Representative TFT Population-Level Error Report

## Scope

- **Table 5** uses the **representative TFT seed 20250111** (selected by lowest validation loss, not test MAE).
- **Table 10** reports **three-seed mean ± standard deviation** across seeds 20250111, 20250112, and 20250113; it is unchanged by this update.
- Predictions source: `outputs/revision/predictions/final_pf_tft_seed20250111.csv`
- Test set: 150 independent held-out series; 731 common population evaluation time points (~3.99–59.95 years).
- No model retraining was performed.

## Population trajectories

Reference and predicted cumulative initiation probabilities:

- `Pf_true(t)`: test-set mean of `onset_flag` at each evaluation year
- `Pf_pred(t)`: TFT-predicted population mean at each evaluation year

## Table 5 metrics (representative seed 20250111)

  table model  representative_seed    selection_criterion                                                  checkpoint  test_series  evaluation_time_points  evaluation_start_year  evaluation_end_year      MAE     RMSE  max_abs_error  year_of_max_error  final_year_abs_error
Table 5   TFT             20250111 lowest_validation_loss outputs/revision/checkpoints/tft/20250111_10epoch/best.ckpt          150                     731               3.986311            59.947981 0.004543 0.006382       0.024123          36.796715              0.001347

## Time-dependent absolute error

- Figure: `outputs/revision/figures/final_tft_time_dependent_error_seed20250111.png`
- Maximum absolute error: 0.024123 at year 36.80
- Final-year absolute error (~60 y): 0.001347

## Note on superseded values

Earlier draft values (MAE = 0.002188; RMSE = 0.002827) are **not** used here. The updated representative-seed metrics match the locked final benchmark evaluation on the common 731-point test horizon.
