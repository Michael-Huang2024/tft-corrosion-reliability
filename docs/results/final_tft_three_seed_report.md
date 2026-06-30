# Final TFT Three-Seed Report

    seed      MAE     RMSE  max_abs_error  year_of_max_error  final_year_abs_error  best_epoch  best_validation_loss  epochs_completed  early_stopping_triggered  training_time_seconds  avg_seconds_per_epoch  pure_inference_time_seconds  parameter_count  evaluation_time_points                                                                                     checkpoint_dir                                                                                   prediction_file
20250111 0.004543 0.006382       0.024123          36.796715              0.001347           8              0.007161                10                     False                17020.5            1702.050000                    90.899866            75992                     731      outputs/revision/checkpoints/tft/20250111_10epoch outputs/revision/predictions/tft_20250111_10epoch.csv
20250112 0.004861 0.007164       0.028865          36.490075              0.003533           3              0.007931                 7                      True                14573.4            2081.914286                    86.133044            75992                     731      outputs/revision/checkpoints/tft/20250112_10epoch outputs/revision/predictions/tft_20250112_10epoch.csv
20250113 0.004222 0.005575       0.018222          36.490075              0.003117           4              0.007803                 8                      True                14425.2            1803.150000                    92.541295            75992                     731 outputs/revision/checkpoints/tft/20250113_10epoch_test outputs/revision/predictions/tft_20250113_10epoch.csv

## TFT mean ± SD
- MAE: 0.004542 ± 0.000261
- RMSE: 0.006373 ± 0.000649
- max error: 0.023737 ± 0.004354
- final-year error: 0.002666 ± 0.000948
- training time (s): 15339.7 ± 1190.0
- inference time (s): 89.9 ± 2.7
- best epoch distribution: 8,3,4
