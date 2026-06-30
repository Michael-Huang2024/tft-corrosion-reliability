# TFT Failure Diagnosis

Date: 2026-06-25  
Branch: `revision/reviewer-experiments`

## Summary

Prior TFT runs on Windows repeatedly entered the Lightning training loop, occasionally wrote partial checkpoints, then exited **without Python traceback**. The wrapper shell reported completion while no active Python process remained. This pattern is **not consistent with a normal Python exception** and likely reflects process-level termination during or immediately after GPU/DataLoader/training activity.

The upgraded GPU driver (610.62) and successful CUDA tensor test suggest the previous environment was unstable, but driver alone is **not assumed** to be the sole cause.

## Evidence reviewed

### Checkpoint inventory (legacy flat directory)

| File | Size | Notes |
|------|------|-------|
| `final_tft_seed20250111-epoch=2-val_loss=0.0082.ckpt` | 1.34 MB | Best partial run for seed 11 |
| `final_tft_seed20250112-epoch=7-val_loss=0.0064.ckpt` | 1.34 MB | Deepest partial run for seed 12 |
| `final_tft_seed20250113-epoch=0-val_loss=0.0173.ckpt` | 1.34 MB | Only epoch 0 for seed 13 |
| Smoke `tft_seed*_epoch=0-val_loss=~0.39-0.45.ckpt` | 1.29 MB | Smoke-test artifacts |

Checkpoints prove training reached **at least one validation epoch** before wrapper exit. No checkpoint was paired with a completed prediction CSV for seeds 12/13.

### Prediction outputs

- `final_pf_tft_seed20250111.csv` exists (731 time points, MAE 0.003544)
- No `final_pf_tft_seed20250112.csv` or `final_pf_tft_seed20250113.csv`

### Prior remediation attempts (07_train_benchmarks.py)

All failed to produce stable three-seed completion:

1. Disabled Lightning progress bar and model summary
2. Set `num_sanity_val_steps=0`
3. Tested CPU trainer path
4. Tested training-window stride subsampling
5. Combined train + evaluation in one process

Observed failure signature in each case:

- Log ends shortly after `Trainer.fit` / dataloader initialization
- No `Traceback`, `RuntimeError`, or CUDA OOM message in terminal capture
- Background wrapper remains while Python child disappears
- Partial `.ckpt` files appear, but post-training evaluation/CSV merge often never runs

### Windows Event Viewer

Not queried in this session (requires elevated/local access). No application crash dump was found in project logs.

## Stage attribution

| Stage | Evidence | Likely? |
|-------|----------|---------|
| Dataset construction | Completed; same data loads for MLP/GRU | Unlikely primary cause |
| DataLoader initialization | Log often stops near dataloader hint | Possible trigger |
| Training loop | Partial epoch checkpoints exist | **Most likely exit during/after training** |
| Validation | val_loss embedded in checkpoint names | Reached at least once |
| Checkpoint saving | `.ckpt` files present | Succeeded intermittently |
| Prediction | Missing for seeds 12/13 | Failed when combined with training wrapper |
| Aggregation / CSV writing | Never reached in failed runs | Downstream of training exit |

## Suspected causes (ranked)

1. **GPU driver / CUDA runtime instability on Windows** — silent process termination without Python exception; partial checkpoints suggest GPU work started. Driver now upgraded to 610.62; audit passes.
2. **Combined train+eval monolithic wrapper** — even when training wrote checkpoints, evaluation never ran; separating commands removes this failure mode.
3. **Lightning + PyTorch Forecasting Windows DataLoader edge case** — `num_workers=0` already used; persistent workers disabled in new stable path.
4. **Terminal background wrapper losing child process** — wrapper exit code not always propagated; new workflow records explicit Python exit codes and completion markers.
5. **OOM** — no explicit OOM logged; RTX 4060 8 GB with batch 32 should fit TFT (~76k params), but unlogged native OOM remains possible.

## Mitigations applied (Phase 3)

- Separate `train` and `evaluate` commands (`scripts/14_tft_stable.py`)
- Per-seed checkpoint directory: `outputs/revision/checkpoints/tft/{seed}/`
- Training batch 32, inference batch 64, `num_workers=0`, `persistent_workers=False`
- Save best + last checkpoint each epoch
- Epoch JSON history + flushed file logs + GPU memory after each epoch
- Full precision (`32-true`) for first diagnostic run
- Explicit completion marker and exit code checking

## Additional finding (2026-06-25 diagnostic session)

Reproduced a native **Windows access violation (exit code 0xC0000005)** when `torch` is imported before `pandas`/`pyarrow`. Import order must be pandas/pyarrow first, then torch. The stable script `14_tft_stable.py` already follows this order.

Full TFT training uses **503,300 windows / 15,728 batches per epoch** at batch size 32. Timed benchmark: ~0.19 s/batch → **~50 minutes per training epoch** before validation and checkpointing. Prior background wrapper runs terminated around 15 minutes, **before the first epoch could finish**, which explains missing checkpoints without Python traceback.

## Conclusion

Original silent exits most likely occurred **during or immediately after Lightning training/validation**, with **checkpoint saving succeeding intermittently** and **downstream evaluation never completing** in the combined wrapper. GPU driver instability is a plausible contributing factor but not the only one. The new separated, logged workflow is required before claiming any TFT benchmark completion.
