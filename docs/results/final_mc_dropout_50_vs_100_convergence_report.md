# Final MC Dropout 50 vs 100 Convergence Report

## A. Model selection

- Selected seed: **20250111**
- Selection criterion: **best validation loss** (not test MAE)
- Best validation loss: **0.007161**
- Checkpoint: `outputs/revision/checkpoints/tft/20250111_10epoch/best.ckpt`

## B. Evaluation setup

- Split: independent test only
- Test series: 150
- Population evaluation time points: 731
- Time range: ~3.99–59.95 years
- Target: cumulative corrosion initiation `onset_flag`
- Method: MC Dropout-based approximate Bayesian inference
- Dropout activation: `model.eval()` globally; dropout modules set to train mode only
- Active dropout modules (24): static_variable_selection.flattened_grn.gate_norm.glu.dropout, static_variable_selection.single_variable_grns.Cs.gate_norm.glu.dropout, static_variable_selection.single_variable_grns.D28.gate_norm.glu.dropout, static_variable_selection.single_variable_grns.m_aging.gate_norm.glu.dropout, static_variable_selection.single_variable_grns.cover_mm.gate_norm.glu.dropout, static_variable_selection.single_variable_grns.C_th.gate_norm.glu.dropout, static_variable_selection.single_variable_grns.encoder_length.gate_norm.glu.dropout, encoder_variable_selection.flattened_grn.gate_norm.glu.dropout, encoder_variable_selection.single_variable_grns.time_idx.gate_norm.glu.dropout, encoder_variable_selection.single_variable_grns.t_year.gate_norm.glu.dropout, encoder_variable_selection.single_variable_grns.relative_time_idx.gate_norm.glu.dropout, decoder_variable_selection.flattened_grn.gate_norm.glu.dropout, decoder_variable_selection.single_variable_grns.time_idx.gate_norm.glu.dropout, decoder_variable_selection.single_variable_grns.t_year.gate_norm.glu.dropout, decoder_variable_selection.single_variable_grns.relative_time_idx.gate_norm.glu.dropout, static_context_variable_selection.gate_norm.glu.dropout, static_context_initial_hidden_lstm.gate_norm.glu.dropout, static_context_initial_cell_lstm.gate_norm.glu.dropout, static_context_enrichment.gate_norm.glu.dropout, post_lstm_gate_encoder.dropout, static_enrichment.gate_norm.glu.dropout, multihead_attn.dropout, post_attn_gate_norm.glu.dropout, pos_wise_ff.gate_norm.glu.dropout
- Stochastic forward passes: 100 (base seed 20250626 + pass_id)
- Inference batch size: 64; num_workers: 0; precision: 32-true

### Pass-level reuse

Prior MC Dropout output (`mc_dropout_population_predictions.csv`) contained only aggregated 731-point summaries without pass-level trajectories, and passes 1–50 were not saved with reproducible per-pass seeds (`base_seed=20250626`). Therefore all 100 stochastic forward passes were re-run for fair nested 20/50/100 comparison.

## C. Results for 20, 50, and 100 passes

 n_passes      MAE     RMSE  mean_predictive_std  max_predictive_std  PICP_95  MPIW_95  final_time_abs_error  max_abs_error  year_of_max_error  monotonicity_violations  evaluation_time_points
       20 0.004610 0.006497             0.000435            0.001226 0.094391 0.001452              0.001288       0.024932          36.490075                       14                     731
       50 0.004602 0.006489             0.000451            0.001562 0.105335 0.001643              0.001071       0.024791          36.490075                       12                     731
      100 0.004616 0.006503             0.000455            0.001803 0.113543 0.001715              0.000824       0.024758          36.490075                       10                     731

## D. 20 vs 50 convergence

comparison  mean_abs_predictive_mean_change  max_abs_predictive_mean_change  mean_abs_predictive_std_change  max_abs_predictive_std_change  mean_abs_q025_change  mean_abs_q975_change  MAE_change  RMSE_change  PICP_change  MPIW_change  relative_change_in_mean_predictive_std  relative_change_in_MPIW
  20_vs_50                         0.000071                        0.000475                        0.000048                       0.000336              0.000145              0.000152   -0.000008    -0.000008     0.010944     0.000191                                0.036332                 0.131734

## E. 50 vs 100 convergence

comparison  mean_abs_predictive_mean_change  max_abs_predictive_mean_change  mean_abs_predictive_std_change  max_abs_predictive_std_change  mean_abs_q025_change  mean_abs_q975_change  MAE_change  RMSE_change  PICP_change  MPIW_change  relative_change_in_mean_predictive_std  relative_change_in_MPIW
 50_vs_100                         0.000038                        0.000247                        0.000027                        0.00024              0.000087              0.000088    0.000014     0.000013     0.008208     0.000072                                0.009463                 0.043689

### Descriptive stability assessment (50 vs 100)

- mean_abs_predictive_mean_change: 3.770999e-05 (< 1e-4)
- mean_abs_predictive_std_change: 2.731967e-05 (< 1e-4)
- relative_change_in_MPIW: 4.3689% (< 5%)
- Assessment: predictive mean highly stable; predictive std highly stable; MPIW basically stable (<5%)

## F. Main conclusion

1. **Is 50 passes sufficient for predictive mean?** Yes — mean absolute change 50→100 is 3.770999e-05.
2. **Is 50 passes sufficient for predictive std?** Yes — mean absolute std change is 2.731967e-05 (relative 0.9463%).
3. **Does 100 passes significantly change 95% interval, PICP, or MPIW?** PICP change +0.0082, MPIW change +7.179271e-05 (relative MPIW 4.3689%).
4. **Is 50 passes a reasonable cost–stability trade-off?** Yes — 50 passes yield highly stable predictive mean and std (<1e-4 mean absolute change vs 100).

## G. Important interpretation

This analysis uses **MC Dropout-based approximate Bayesian inference** to approximate the **approximate posterior predictive distribution** and an **epistemic uncertainty estimate** via **stochastic forward passes** over a single trained TFT checkpoint.

Each dropout mask corresponds to one random effective sub-network of the same trained weights; this is **not** exact Bayesian inference, **not** an exact posterior, **not** a fully Bayesian TFT, and **not** 100 independently trained TFT models.

## Runtime and hardware

- Inference time (100 passes): 12237.8 s (122.38 s/pass)
- Additional passes completed this run: 100
- Aggregation / metrics time: 0.1 s
- Figure generation time: 1.5 s
- Hardware: {"torch_version": "2.6.0+cu124", "cuda_available": true, "cuda_version": "12.4", "cudnn_version": 90100, "gpu_count": 1, "gpus": ["NVIDIA GeForce RTX 4060"]}
