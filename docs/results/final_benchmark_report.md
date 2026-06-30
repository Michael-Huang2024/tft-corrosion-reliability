# Final Benchmark Report

All models were evaluated on the independent test split using cumulative `onset_flag`, shared time range, shared population aggregation, and identical physical predictors with `C_th` included. `chloride_rebar` and target-derived fields were not used as predictors. MLP, GRU, and TFT share a 52-step encoder and 13-step prediction horizon with unit stride; Logistic Regression is a pointwise tabular baseline.

## Model Ranking by Test MAE

              model  MAE_mean  MAE_std  RMSE_mean  RMSE_std  training_time_seconds_mean  pure_inference_time_seconds_mean  parameter_count
                GRU  0.001934 0.000169   0.002931  0.000207                         NaN                               NaN              NaN
                TFT  0.004542 0.000261   0.006373  0.000649                         NaN                               NaN              NaN
                MLP  0.006975 0.000843   0.009964  0.001328                  247.149469                          2.658343         135437.0
Logistic Regression  0.020652 0.000000   0.024271  0.000000                         NaN                               NaN              8.0

## Explicit Comparison Answers

Does TFT outperform Logistic Regression? yes.
Does TFT outperform MLP? yes.
Does TFT outperform GRU? no.
Is any advantage consistent across seeds? See `outputs/revision/tables/final_model_comparison_by_seed.csv`; neural-model seed dispersion is reported in the final comparison table.
Is any accuracy advantage large enough to justify greater complexity? This should be judged from the held-out MAE/RMSE differences together with parameter count and inference time in the table above.
Best accuracy-efficiency tradeoff by MAE and deterministic inference time: GRU has the lowest MAE; compare inference time before making a complexity claim.

No TFT superiority claim should be made unless supported by the final held-out metrics above.
