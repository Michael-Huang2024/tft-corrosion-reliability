# TFT Three-Seed Benchmark Report (Final)

Branch: `revision/reviewer-experiments`  
Completed: 2026-06-26  
Configuration: 10-epoch TFT schedule (identical across seeds)

## Executive Summary

**All three TFT seeds are complete.** Training, checkpointing, and held-out evaluation (731 common population time points) finished successfully under the locked 10-epoch configuration. Native Lightning progress bars were used throughout.

| Item | Result |
|------|--------|
| TFT seeds completed | **3 / 3** |
| TFT MAE (mean ± SD) | **0.004542 ± 0.000261** |
| TFT RMSE (mean ± SD) | **0.006373 ± 0.000649** |
| TFT better than Logistic Regression | **yes** |
| TFT better than MLP | **no** |
| TFT better than GRU | **no** |
| TFT superiority claim | **not supported** |

---

## Fixed Configuration

| Parameter | Value |
|-----------|-------|
| encoder length | 52 |
| prediction horizon | 13 |
| hidden size | 32 |
| attention heads | 4 |
| dropout | 0.1 |
| learning rate | 3e-4 |
| max_epochs | 10 |
| min_epochs | 5 |
| early_stopping patience | 3 |
| training batch size | 32 |
| inference batch size | 64 |
| num_workers | 0 |
| persistent_workers | False |
| precision | 32-true |
| accelerator | GPU |
| progress display | Lightning tqdm (`enable_progress_bar=True`) |

Scientific architecture, locked Candidate C data, splits, features, and cumulative `onset_flag` target unchanged.

---

## Per-Seed Results

### Seed 20250111

| Metric | Value |
|--------|-------|
| Epochs completed | 10 (full schedule) |
| Early stopping | no |
| Best epoch | 8 |
| Best validation loss | 0.007161 |
| Test MAE | 0.004543 |
| Test RMSE | 0.006382 |
| Max absolute error | 0.024123 (year 36.80) |
| Final-year error | 0.001347 |
| Training time | 17020.5 s (~4.73 h) |
| Inference time | 90.9 s |
| Evaluation points | 731 |

Checkpoint: `outputs/revision/checkpoints/tft/20250111_10epoch/`

### Seed 20250112

| Metric | Value |
|--------|-------|
| Epochs completed | 8 (early stopping) |
| Early stopping | yes |
| Best epoch | 3 |
| Best validation loss | 0.007931 |
| Test MAE | 0.004861 |
| Test RMSE | 0.007164 |
| Max absolute error | 0.028865 (year 36.49) |
| Final-year error | 0.003533 |
| Training time | 14573.4 s (~4.05 h) |
| Inference time | 86.1 s |
| Evaluation points | 731 |

Checkpoint: `outputs/revision/checkpoints/tft/20250112_10epoch/`

### Seed 20250113 (official diagnostic)

| Metric | Value |
|--------|-------|
| Epochs completed | 8 (early stopping) |
| Early stopping | yes |
| Best epoch | 4 |
| Best validation loss | 0.007803 |
| Test MAE | 0.004222 |
| Test RMSE | 0.005575 |
| Max absolute error | 0.018222 (year 36.49) |
| Final-year error | 0.003117 |
| Training time | 14425.2 s (~4.01 h) |
| Inference time | 92.5 s |
| Evaluation points | 731 |

Checkpoint: `outputs/revision/checkpoints/tft/20250113_10epoch_test/`

---

## TFT Three-Seed Summary (mean ± SD)

| Metric | Mean | SD |
|--------|------|-----|
| MAE | 0.004542 | 0.000261 |
| RMSE | 0.006373 | 0.000649 |
| Max absolute error | 0.023737 | 0.004354 |
| Final-year error | 0.002666 | 0.000948 |
| Training time (s) | 15339.7 | 1190.0 |
| Inference time (s) | 89.9 | 2.7 |
| Best validation loss | 0.007631 | 0.000337 |

Best-epoch distribution: **8, 3, 4**

Seed-level MAE range: 0.004222 – 0.004861 (CV ≈ 5.7%)

---

## Full Model Benchmark Comparison

| Model | Seeds | MAE (mean ± SD) | RMSE (mean ± SD) |
|-------|-------|-----------------|------------------|
| **GRU** | 3 | **0.001934 ± 0.000169** | **0.002931 ± 0.000207** |
| **MLP** | 3 | 0.003017 ± 0.000182 | 0.004371 ± 0.000224 |
| **TFT** | 3 | 0.004542 ± 0.000261 | 0.006373 ± 0.000649 |
| Logistic Regression | 1 | 0.020652 ± 0.000000 | 0.024271 ± 0.000000 |

### Explicit Answers

1. **TFT better than Logistic Regression?** yes  
2. **TFT better than MLP?** no  
3. **TFT better than GRU?** no  
4. **TFT consistent across seeds?** yes — MAE SD = 0.000261 (~5.7% CV); all seeds 731 points, same evaluation range  
5. **TFT complexity justified by accuracy?** no — TFT MAE/RMSE higher than MLP and GRU with longer training (~4–5 h/seed) and similar inference (~90 s)  
6. **Best accuracy-efficiency balance:** **GRU** (lowest MAE/RMSE among neural models)

**Do not claim TFT superiority.** Complete three-seed results do not support TFT over MLP or GRU.

---

## Output Files

| File | Description |
|------|-------------|
| `tables/final_tft_three_seed_results.csv` | Per-seed TFT metrics |
| `tables/final_tft_three_seed_summary.csv` | TFT mean ± SD |
| `tables/final_model_comparison.csv` | All models |
| `tables/final_model_comparison_by_seed.csv` | All models by seed |
| `figures/final_tft_three_seed_trajectories.png` | TFT Pf curves |
| `figures/final_tft_seed_metric_variability.png` | TFT seed variability |
| `figures/final_model_error_comparison.png` | All-model MAE bar chart |
| `figures/final_population_trajectories_by_model.png` | Population Pf by model |
| `final_benchmark_report.md` | Full benchmark answers |
| `final_tft_three_seed_report.md` | Auto-generated TFT summary |

Per-seed predictions: `predictions/tft_{seed}_10epoch.csv`  
Per-seed metrics: `tables/tft_{seed}_10epoch_metrics.csv`

---

## Technical Notes

- Separate `train` and `evaluate` commands per seed; one TFT process at a time.
- Checkpoint copy skips same-file paths (fixes prior `SameFileError`).
- Visible Lightning progress bars confirmed for all training runs.
- No Sobol, MC Dropout, bootstrap, or runtime scaling in this task.
- Locked data and Candidate C parameters not modified.

---

## Completion Checklist

- [x] Seed 20250111 trained and evaluated
- [x] Seed 20250112 trained and evaluated
- [x] Seed 20250113 validated (diagnostic run)
- [x] Three-seed TFT summary tables and figures
- [x] Full benchmark comparison regenerated
- [ ] MC Dropout (not started)
- [ ] Sobol (not started)
