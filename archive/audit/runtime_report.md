# Runtime Report

Runtime values are measured with `time.perf_counter()`; missing checkpoints are reported as skipped.

                        method  scenario_count               stage  seconds_mean  seconds_std  peak_memory_bytes_max                        status  parameter_count                                                                                                             checkpoint
vectorized_diffusion_simulator              20 simulator_execution      0.001812          0.0               587901.0                      measured              NaN                                                                                                                    NaN
                           MLP              20      pure_inference      0.112270          0.0                 2362.0                      measured           4737.0                           D:\论文1-Pf(t)\tft-corrosion-reliability-main\outputs\revision\checkpoints\mlp_seed20250111.pt
                           GRU              20      pure_inference      0.018108          0.0                 2197.0                      measured           4365.0                           D:\论文1-Pf(t)\tft-corrosion-reliability-main\outputs\revision\checkpoints\gru_seed20250111.pt
                           TFT              20      pure_inference           NaN          NaN                    NaN not_measured_in_point_harness              NaN D:\论文1-Pf(t)\tft-corrosion-reliability-main\outputs\revision\checkpoints\tft_seed20250113-epoch=0-val_loss=0.3935.ckpt
