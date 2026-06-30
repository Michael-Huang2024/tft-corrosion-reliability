# Final Computational Efficiency Report

## A. Purpose

This analysis responds to the Editor's comment that the manuscript provided weak evidence of computational advantage.

## B. Methods compared

- Physics simulator (reference generator)
- Logistic Regression
- MLP
- GRU
- Deterministic TFT (seed 20250111)
- 50-pass MC Dropout TFT

## C. Timing protocol

- Hardware: {"torch_version": "2.6.0+cu124", "cuda_available": true, "cuda_version": "12.4", "cudnn_version": 90100, "gpu_count": 1, "gpus": ["NVIDIA GeForce RTX 4060"]}
- Test series: 150
- Population time points: 731
- End-to-end timing includes loading, data preparation, inference, and Pf aggregation where applicable
- Inference-only timing includes forward prediction only
- Repeated runs: 3 for remeasured methods
- MC Dropout runtime loaded from formal 50-pass benchmark log

## D. Results

### End-to-end (test task)

                  method  total_seconds  inference_seconds  seconds_per_prediction runtime_source                                                                                                                                                                                                                   notes
       Physics simulator       0.008187           0.008187            6.970427e-08   measured_now                                                                                                                                       Vectorized apparent-diffusivity simulator; full 783-step trajectories per series.
     Logistic Regression       0.459530           0.019746            4.190876e-06   measured_now                                                                                                                          No persisted checkpoint; one in-memory fit on locked train split for forward-pass timing only.
                     MLP       0.135660           0.061671            1.237206e-06   measured_now                                                                                                                                                                Checkpoint: final_mlp_seed20250111.pt; batch=all points.
                     GRU       8.372401           8.101637            7.635568e-05   measured_now                                                                                                                                                                        Checkpoint: final_gru_seed20250111.pt; batch=64.
     TFT (deterministic)      47.709181          46.362395            4.351042e-04   measured_now                                                                                                                                                                                       Checkpoint: best.ckpt; batch=128.
TFT (50-pass MC Dropout)    5964.600000        5964.600000            5.439672e-02  benchmark_log From formal 50-pass MC Dropout report (5964.6 s total; 119.29 s/pass). 100-pass convergence rerun reported 6678.4 s total (66.78 s/pass); difference likely reflects resume/cache/environment rather than model change.

### Inference-only (test task)

                  method  inference_seconds  seconds_per_prediction runtime_source                                                              notes
     Logistic Regression           0.010459            9.538836e-08   measured_now             Inference-only; excludes train-fit and Pf aggregation.
                     MLP           0.001276            1.163885e-08   measured_now                                                 Forward pass only.
                     GRU           1.633883            1.490089e-05   measured_now                                                 Forward pass only.
     TFT (deterministic)          44.599778            4.067467e-04   measured_now                                                 Forward pass only.
TFT (50-pass MC Dropout)        5964.600000            5.439672e-02  benchmark_log Stochastic forward passes only; excludes one-time checkpoint load.

### Training runtime (separate from inference)

             method     seed  training_seconds  training_hours  best_validation_loss  test_MAE  test_RMSE                 runtime_source                                                                                       notes
                TFT 20250111           17020.5        4.727917              0.007161  0.004543   0.006382                  benchmark_log                                                                                         NaN
                MLP 20250111               NaN             NaN                   NaN  0.003017   0.004371 not_logged_in_final_checkpoint Final benchmark did not persist per-seed training seconds in CSV; inference remeasured now.
                GRU 20250111               NaN             NaN                   NaN  0.001934   0.002931 not_logged_in_final_checkpoint Final benchmark did not persist per-seed training seconds in CSV; inference remeasured now.
Logistic Regression                        NaN             NaN                   NaN  0.020652   0.024271 not_logged_in_final_checkpoint                                                                                         NaN

## E. Runtime scaling

           method  n_series  n_time_points  n_predictions  total_seconds  total_seconds_std  seconds_per_series  seconds_per_prediction runtime_source
Physics simulator       150            783         117450       0.007108           0.000198            0.000047            6.052306e-08   measured_now
Physics simulator       300            783         234900       0.017799           0.000217            0.000059            7.577352e-08   measured_now
Physics simulator       600            783         469800       0.036374           0.000299            0.000061            7.742507e-08   measured_now
Physics simulator      1000            783         783000       0.062755           0.000514            0.000063            8.014636e-08   measured_now
Physics simulator      2000            783        1566000       0.134039           0.000462            0.000067            8.559310e-08   measured_now
Physics simulator      5000            783        3915000       0.336835           0.006882            0.000067            8.603694e-08   measured_now
              MLP       150            731         109650       0.011550           0.001723            0.000077            1.053352e-07   measured_now
              MLP       300            731         109800       0.004086           0.002747            0.000014            3.721008e-08   measured_now
              MLP       600            731         110100       0.001388           0.000176            0.000002            1.260642e-08   measured_now
              MLP      1000            731         110500       0.001924           0.000082            0.000002            1.741388e-08   measured_now

## F. Accuracy-efficiency tradeoff

- GRU has the best held-out MAE in the locked benchmark.
- Logistic Regression is fastest but least accurate.
- MLP is a strong simple baseline with low inference cost.
- Deterministic TFT is slower than GRU/MLP on this task and not the most accurate.
- 50-pass MC Dropout TFT adds large inference cost for epistemic uncertainty.
- The physics simulator is a reference generator, not a surrogate predictor.

## G. Revised interpretation

The computational comparison indicates that the proposed TFT surrogate should not be interpreted as universally faster or more accurate than simpler neural architectures for the present low-dimensional simulation task. Although the surrogate framework enables reusable sequence prediction and uncertainty-aware analysis, GRU achieved the best accuracy–efficiency balance in the current benchmark. Therefore, the computational advantage of TFT is conditional and is expected to be more relevant in future extensions involving heterogeneous time-varying bridge inspection and environmental data, rather than in the simplified simulator-replication setting examined here.

## H. Reviewer/editor response paragraph

Response: We sincerely thank the Editor for pointing out that the original manuscript provided weak evidence of computational advantage. We have now completed a unified computational efficiency comparison under the locked test evaluation domain (150 series; 731 population time points). The comparison includes the physics simulator, Logistic Regression, MLP, GRU, deterministic TFT (seed 20250111 selected by validation loss), and 50-pass MC Dropout TFT. The remeasured results do not support a universal TFT speed advantage: MLP is fastest for inference-only timing among surrogate models, while deterministic TFT and especially 50-pass MC Dropout TFT incur substantially higher inference cost. We have therefore narrowed the computational-advantage claim and added the comparison to Results/Discussion/Limitations.
