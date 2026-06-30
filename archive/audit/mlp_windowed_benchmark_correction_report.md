# MLP Windowed Benchmark Correction Report — Reviewer Comment 5

**Date:** 2026-06-28  
**Status:** Completed (exit code 0, total runtime ~12.7 min)  
**Script:** `scripts/07_train_benchmarks.py --models mlp --seeds 20250111 20250112 20250113 --device cuda`

---

## Executive Summary

The minimal fair-benchmark correction has been **completed successfully**. MLP was retrained as a **windowed 52→13** model using the same `WindowDataset`, split, inference overlap-averaging, and Pf(t) evaluation pipeline as GRU. All three seeds finished; Table 10 artifacts were regenerated.

**Key outcome:** After correction, **TFT now outperforms MLP** on held-out Pf(t) MAE, and the model ranking changed from *GRU > MLP > TFT* to *GRU > TFT > MLP*.

---

## 1. Is MLP now windowed 52→13?

**Yes.**

| Item | Implementation |
|---|---|
| Input window | 52 time steps |
| Features per step | `Cs, D28, m_aging, cover_mm, C_th, time_idx, t_year` (7 features) |
| Flattened input | 52 × 7 = **364** dimensions |
| Output | **13 logits** (one per future `onset_flag` step) |
| Loss | `BCEWithLogitsLoss` over all 13 outputs |
| Stride | 1 (via existing `WindowDataset`) |
| Architecture | `WindowedMLPClassifier`: 364 → 256 → 128 → 64 → 13 (dropout 0.1) |
| Parameters | **135,437** (all three seeds) |

Checkpoint evidence: all `final_mlp_seed*.pt` files are now **0.52 MB** (was **0.01 MB** for the old pointwise MLP) and include `model_type: "windowed_mlp"`.

---

## 2. Are MLP, GRU, and TFT now aligned on task definition?

| Criterion | MLP (corrected) | GRU | TFT | Aligned? |
|---|---|---|---|---|
| 52-step encoder | Yes | Yes | Yes | **Yes** |
| 13-step horizon | Yes | Yes | Yes | **Yes** |
| Stride = 1 | Yes | Yes | Yes | **Yes** |
| Target `onset_flag` | Yes | Yes | Yes | **Yes** |
| Same covariates (no leakage) | Yes | Yes | Yes | **Yes** |
| Series-level 70/15/15 split | Yes | Yes | Yes | **Yes** |
| Overlap-averaged test inference | Yes | Yes | Yes | **Yes** |
| Population Pf(t) aggregation | Yes | Yes | Yes | **Yes** |
| MAE/RMSE on Pf curve | Yes | Yes | Yes | **Yes** |
| Evaluation time points | 731 | 731 | 731 | **Yes** |

**Logistic Regression** remains a **pointwise** simple linear baseline (by design).

**Remaining minor asymmetry:** TFT final Table 10 runs used a **10-epoch** training cap (`17_tft_three_seed_benchmark.py`), while MLP/GRU use up to **30 epochs** with early stopping. This does not affect the primary window-alignment fix.

---

## 3. Updated Table 10 Values

Source: `outputs/revision/tables/final_model_comparison.csv`

### Summary (mean ± std over seeds)

| Model | MAE | RMSE | Training time (s) | Inference time (s) | Params |
|---|---|---|---|---|---|
| **GRU** | 0.001934 ± 0.000169 | 0.002931 ± 0.000207 | — | — | — |
| **TFT** | 0.004542 ± 0.000261 | 0.006373 ± 0.000649 | — | — | — |
| **MLP** (windowed) | **0.006975 ± 0.000843** | **0.009964 ± 0.001328** | 247.1 ± 42.5 | 2.66 ± 0.17 | 135,437 |
| **Logistic Regression** | 0.020652 ± 0.000000 | 0.024271 ± 0.000000 | — | — | 8 |

### Per-seed MLP results (windowed)

Source: `outputs/revision/tables/final_model_comparison_by_seed.csv`

| Seed | MAE | RMSE | Epochs | Best epoch | Best val loss | Training (s) | Inference (s) |
|---|---|---|---|---|---|---|---|
| 20250111 | 0.007104 | 0.009440 | 11 | 4 | 0.01890 | 233.6 | 2.90 |
| 20250112 | 0.007935 | 0.011787 | 10 | 3 | 0.01834 | 203.1 | 2.53 |
| 20250113 | 0.005884 | 0.008663 | 15 | 8 | 0.01785 | 304.7 | 2.54 |

### Before vs after correction (MLP only)

| Metric | Old pointwise MLP | New windowed MLP | Change |
|---|---|---|---|
| MAE mean | 0.003017 | 0.006975 | +131% (worse) |
| RMSE mean | 0.004371 | 0.009964 | +128% (worse) |
| MAE std | 0.000182 | 0.000843 | — |
| Checkpoint size | ~0.01 MB | 0.52 MB | — |

The windowed MLP performs **worse** than the old pointwise MLP because the old model had an easier (and unfair) task: it predicted from a single feature row without using 52-step history.

---

## 4. Did the ranking change?

**Yes.**

### MAE ranking

| Rank | Before correction | After correction |
|---|---|---|
| 1 | GRU (0.001934) | GRU (0.001934) — unchanged |
| 2 | **MLP (0.003017)** | **TFT (0.004542)** |
| 3 | **TFT (0.004542)** | **MLP (0.006975)** |
| 4 | Logistic (0.020652) | Logistic (0.020652) — unchanged |

### Key comparison shifts

| Comparison | Before | After |
|---|---|---|
| TFT vs MLP | TFT worse (MAE 0.00454 vs 0.00302) | **TFT better** (0.00454 vs 0.00697) |
| TFT vs GRU | TFT worse | TFT worse (unchanged) |
| TFT vs Logistic | TFT better | TFT better (unchanged) |
| MLP vs GRU | MLP worse | MLP worse (unchanged) |

---

## 5. Manuscript and Response to Reviewers — Suggested Text Updates

### Methods (benchmark paragraph)

> Sequence baselines (MLP, GRU, and TFT) were trained with an identical 52-week encoder and 13-week prediction horizon (stride = 1) on the cumulative initiation label `onset_flag`. The MLP baseline flattens the 52 × 7 covariate window into a 364-dimensional vector and outputs 13 logits trained with binary cross-entropy. GRU and TFT consume the same per-step covariates in sequential form. Test-set predictions from overlapping windows were averaged at each `(series_id, time_idx)` before population-level Pf(t) aggregation. Logistic regression is reported separately as a pointwise linear baseline using the same covariates and target at each evaluation time. All models were evaluated on 150 held-out series over 731 common time points (time index ≥ 52). Neural models used seeds 20250111, 20250112, and 20250113; Table 10 reports mean ± standard deviation.

### Table 10

Replace MLP row with:

| Model | MAE | RMSE |
|---|---|---|
| GRU | 0.00193 ± 0.00017 | 0.00293 ± 0.00021 |
| TFT | 0.00454 ± 0.00026 | 0.00637 ± 0.00065 |
| MLP | 0.00697 ± 0.00084 | 0.00996 ± 0.00133 |
| Logistic Regression | 0.02065 | 0.02427 |

### Response to Reviewers (Comment 5)

> We revised the MLP baseline to match the GRU and TFT experimental setup. The corrected MLP uses the same 52-step input window and 13-step prediction horizon (stride = 1), the same covariates and cumulative target, the same series-level train/validation/test split, and the same overlap-averaged inference and population Pf(t) evaluation. Logistic regression is retained as a simpler pointwise linear reference. After this correction, GRU remains the most accurate model, TFT outperforms the windowed MLP, and all sequence models substantially outperform logistic regression. Gaussian process regression was not included due to scalability constraints on the full scenario–time grid; we instead report linear, feedforward (windowed MLP), recurrent (GRU), and attention-based (TFT) surrogates.

### Claims to revise

- **Remove** any statement that MLP outperforms TFT on Pf(t) MAE.
- **Add** explicit note that MLP is now a windowed 52→13 feedforward baseline, not pointwise.
- **Retain** that GRU achieves the lowest Pf(t) error among tested models.

---

## Regenerated Artifacts

| File | Status |
|---|---|
| `outputs/revision/checkpoints/final_mlp_seed20250111.pt` | Updated (2026-06-28 14:33) |
| `outputs/revision/checkpoints/final_mlp_seed20250112.pt` | Updated (2026-06-28 14:36) |
| `outputs/revision/checkpoints/final_mlp_seed20250113.pt` | Updated (2026-06-28 14:41) |
| `outputs/revision/predictions/final_pf_mlp_seed*.csv` | Regenerated (3 files) |
| `outputs/revision/tables/final_model_comparison.csv` | Regenerated |
| `outputs/revision/tables/final_model_comparison_by_seed.csv` | Regenerated |
| `outputs/revision/tables/final_training_summary.csv` | Updated (MLP rows) |
| `outputs/revision/final_benchmark_report.md` | Regenerated |
| `outputs/revision/figures/final_model_error_comparison.png` | Regenerated |
| `outputs/revision/figures/final_population_trajectories_by_model.png` | Regenerated |

GRU, TFT, and Logistic Regression artifacts were **not retrained**; their values were merged from existing saved predictions.

---

## Code Changes

**File modified:** `scripts/07_train_benchmarks.py`

- Removed pointwise `MLPClassifier` / `train_torch_point_model`
- Added `WindowedMLPClassifier` (364 → 256 → 128 → 64 → 13)
- Added `train_windowed_mlp` mirroring GRU training and inference logic
- Updated benchmark report text to document windowed MLP vs pointwise Logistic

---

## Training Run Log

| Item | Value |
|---|---|
| Start | 2026-06-28 14:29:18 (local) |
| End | 2026-06-28 14:42:01 (local) |
| Total duration | ~12.7 minutes |
| Device | CUDA (RTX 4060) |
| Exit code | 0 |

---

*End of report.*
