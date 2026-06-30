# MC Dropout Uncertainty Quantification Report

## Method

MC Dropout-based approximate Bayesian inference on the locked TFT benchmark.

The trained checkpoint is loaded once with fixed weights. The model is set globally
to evaluation mode (`model.eval()`), while dropout layers remain active during
stochastic forward passes. Fifty independent passes approximate the approximate
posterior predictive distribution and provide an epistemic uncertainty estimate.

This analysis does **not** claim exact Bayesian inference, a full Bayesian posterior,
or a rigorous Bayesian TFT.

## Model selection

- Selected seed: **20250111** (by best validation loss, not test MAE)
- Best validation loss: **0.007161**
- Checkpoint: `D:/论文1-Pf(t)/tft-corrosion-reliability-main/outputs/revision/checkpoints/tft/20250111_10epoch/best.ckpt`

## Evaluation scope

- Split: independent test only
- Test series: 150
- Population evaluation time points: 731
- Time range: 3.99–59.95 years
- Target: cumulative corrosion-initiation `onset_flag`
- Stochastic forward passes: 50

## Predictive summaries

Per time point:
- `predictive_mean`: mean across stochastic passes
- `predictive_std`: standard deviation across stochastic passes (epistemic uncertainty estimate)
- `q025`, `q975`: 95% predictive interval bounds

Saved predictions: `outputs/revision/predictions/mc_dropout_population_predictions.csv`

## Metrics

 selected_seed                                                                                              checkpoint  selection_criterion  best_validation_loss  stochastic_passes  test_series_count  evaluation_time_points  evaluation_start_year  evaluation_end_year  MAE_predictive_mean  RMSE_predictive_mean  mean_predictive_std  max_predictive_std  PICP_95  MPIW_95  total_mc_seconds  seconds_per_pass
      20250111 D:\论文1-Pf(t)\tft-corrosion-reliability-main\outputs\revision\checkpoints\tft\20250111_10epoch\best.ckpt best_validation_loss              0.007161                 50                150                     731               3.986311            59.947981             0.004608              0.006499             0.000461            0.001682 0.114911 0.001668            5964.6            119.29

## Convergence

comparison  mean_abs_predictive_mean_change  mean_abs_predictive_std_change  mean_predictive_std  evaluation_time_points
  20_vs_50                         0.000063                        0.000052                  NaN                     NaN
 20_passes                              NaN                             NaN             0.000455                   731.0
 50_passes                              NaN                             NaN             0.000461                   731.0

Total MC Dropout runtime: 5964.6 s (119.29 s per pass).
