# TFT 10-Epoch Diagnostic Report (seed 20250113)

Branch: `revision/reviewer-experiments`  
Run tag: `20250113_10epoch_test`  
Generated: 2026-06-25

## Executive Summary

Training **completed 8 of 10 epochs** with visible Lightning progress bars, then the wrapper exited with a **post-training checkpoint copy error** (`shutil.SameFileError`). Model weights and epoch metrics were saved successfully. **Evaluation was completed separately** from `best.ckpt` on the held-out test set (731 common time points).

| Stage | Status |
|-------|--------|
| GPU preflight | PASS |
| Training (8 epochs) | PASS (weights saved) |
| Post-train wrapper | FAIL (`SameFileError` on `best.ckpt` copy — fixed in script) |
| Test evaluation | PASS |
| Visible progress bar | PASS |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Seed | 20250113 |
| max_epochs | 10 |
| min_epochs | 5 |
| early_stopping patience | 3 |
| training batch size | 32 |
| inference batch size | 64 |
| num_workers | 0 |
| persistent_workers | False |
| learning rate | 3e-4 |
| encoder length | 52 |
| prediction horizon | 13 |
| hidden size | 32 |
| attention heads | 4 |
| dropout | 0.1 |
| precision | 32-true (full) |
| accelerator | GPU (CUDA) |

Scientific TFT architecture unchanged. Locked dataset and Candidate C parameters unchanged.

## GPU Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4060 |
| Driver | 610.62 |
| PyTorch | 2.6.0+cu124 |
| CUDA runtime | 12.4 |
| CUDA tensor test | PASS |

## Training Results

| Metric | Value |
|--------|-------|
| Training completed | **yes** (8 epochs; stopped before max 10) |
| Epochs completed | **8 / 10** |
| Early stopping triggered | **yes** (val_loss plateau after epoch 4) |
| Best epoch | **4** (0-indexed) |
| Best validation loss | **0.007803** |
| Final training loss (epoch 7) | **0.009770** |
| Final validation loss (epoch 7) | **0.008656** |
| Total training time | **14425 s (~4.01 h)** |
| Average time per epoch | **~1803 s (~30.1 min)** |
| Peak GPU memory (reserved) | **0.0645 GB** (PyTorch metric) |
| Process exit code (original run) | **1** (post-train copy error) |
| Visible progress bar | **yes** (Lightning tqdm, train/val batch bars) |

### Epoch History

| Epoch | train_loss | val_loss | best_val_loss | epoch_time (s) |
|-------|------------|----------|---------------|----------------|
| 0 | 0.01038 | 0.01206 | 0.01206 | 1693 |
| 1 | 0.03002 | 0.007995 | 0.007995 | 1800 |
| 2 | 0.01677 | 0.008314 | 0.007995 | 1789 |
| 3 | 0.01420 | 0.01011 | 0.007995 | 1712 |
| 4 | 0.01262 | **0.007803** | **0.007803** | 1784 |
| 5 | 0.01155 | 0.01054 | 0.007803 | 1752 |
| 6 | 0.01069 | 0.008473 | 0.007803 | 1711 |
| 7 | 0.009770 | 0.008656 | 0.007803 | 2184 |

## Checkpoints

Directory: `outputs/revision/checkpoints/tft/20250113_10epoch_test/`

| File | Status |
|------|--------|
| `best.ckpt` | saved (epoch 4, val_loss=0.007803) |
| `last.ckpt` | saved (epoch 7) |
| `epoch_metrics.csv` | saved |

Earlier checkpoints under `outputs/revision/checkpoints/tft/20250113/` were **not overwritten**.

## Test Evaluation (held-out, 731 time points)

Evaluated from `best.ckpt` on independent test split.

| Metric | Value |
|--------|-------|
| Test MAE | **0.004222** |
| Test RMSE | **0.005575** |
| Maximum absolute error | **0.018222** (year ≈ 36.49) |
| Final-year absolute error | **0.003117** |
| Evaluation time points | **731** |
| Inference time | **92.5 s** |
| Evaluation range (years) | 3.99 – 59.95 |

### Comparison with prior partial TFT result (seed 20250113, incomplete run)

| Metric | Prior (1 seed, partial) | This diagnostic |
|--------|-------------------------|-----------------|
| MAE | 0.003544 | 0.004222 |
| RMSE | 0.004746 | 0.005575 |

This diagnostic used a different checkpoint (10-epoch schedule, best epoch 4) and a completed stable training path; direct superiority claims should not be made until three-seed full benchmark is complete.

## Output Files

| File | Path |
|------|------|
| Population Pf predictions | `outputs/revision/predictions/tft_20250113_10epoch.csv` |
| Point predictions | `outputs/revision/predictions/tft_20250113_10epoch_points.csv` |
| Test metrics | `outputs/revision/tables/tft_20250113_10epoch_metrics.csv` |
| Training log | `outputs/revision/logs/tft_20250113_10epoch_training.log` |
| Evaluation log | `outputs/revision/logs/tft_20250113_10epoch_evaluation.log` |
| This report | `outputs/revision/tft_20250113_10epoch_report.md` |

## Technical Notes

1. **Original wrapper failure**: After epoch 8, `shutil.copy2` attempted to copy `best.ckpt` onto itself. Training weights were already correct; evaluation was run with `--skip-train` after fixing the script.
2. **Progress display**: Native Lightning progress bars showed epoch, batch fraction, and `train_loss_step` in the active terminal as required.
3. **Next steps (not executed)**: Full three-seed TFT benchmark (20250111, 20250112, 20250113) and final benchmark table regeneration remain pending explicit approval.

## Completion Checklist

- [x] Single diagnostic seed 20250113 trained
- [x] Visible Lightning progress bar
- [x] best.ckpt and last.ckpt saved
- [x] epoch_metrics.csv saved
- [x] Training log saved
- [x] Test evaluation on 731 points
- [x] Predictions and metrics CSV saved
- [x] Evaluation log saved
- [ ] Three-seed TFT benchmark (not started)
- [ ] MC Dropout (not started)
