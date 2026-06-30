# Reproducibility Guide

**Manuscript:** *Transformer-Based Sequence Learning for Multi-Horizon Corrosion Initiation Probability Prediction in Reinforced Concrete Bridges*

This document describes how to reproduce the **authoritative revision/paper pipeline**. The demo pipeline (`scripts/01–05`) uses different parameter distributions and does not reproduce the reported benchmark metrics.

---

## 1. Python environment

| Item | Value |
|---|---|
| Python | 3.11.9 (3.10+ supported) |
| OS (reference) | Windows 10 (10.0.26200) |
| CPU | Intel 64-bit, 16 logical cores |
| RAM | 16 GB |
| GPU | NVIDIA GeForce RTX 4060 (8 GB), CUDA 12.4 |

Create and activate a virtual environment, then:

```bash
pip install -r requirements.txt
```

Install PyTorch with CUDA separately if needed: https://pytorch.org/get-started/locally/

Reference environment snapshot: `outputs/revision/environment.json`.

---

## 2. Package versions

Pinned in `requirements.txt` (repository checkout). Reference run (`environment.json`):

| Package | Reference version |
|---|---|
| Python | 3.11.9 |
| numpy | 2.4.6 |
| pandas | 3.0.2 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| torch | 2.6.0+cu124 |
| lightning | 2.6.1 |
| pytorch-forecasting | 1.7.0 |
| pyarrow | 24.0.0 |
| SALib | 1.5.2 |

Minor version drift within the same major release is usually acceptable; re-verify metrics after upgrading.

---

## 3. Random seeds

| Seed | Role |
|---|---|
| **20250111** | Manuscript / split seed; representative TFT seed (lowest validation loss) |
| **20250112** | TFT benchmark replicate |
| **20250113** | TFT benchmark replicate |
| **20250627** | Sobol Saltelli sampling seed (`scripts/19_final_sobol_sensitivity.py`) |

Train/validation/test split: 700 / 150 / 150 series (`data/processed/revision/series_split.csv`), drawn with seed 20250111.

---

## 4. Authoritative pipeline order

```bash
python run_revision_pipeline.py
```

Equivalent manual sequence:

| Step | Script | Output |
|---|---|---|
| 1 | `12_generate_final_revision_data.py` | `data/processed/revision/final_chloride_labeled.parquet` |
| 2 | `07_train_benchmarks.py --models logistic mlp gru` | `final_pf_*`, `final_model_comparison.csv` |
| 3 | `23_windowed_logistic_baseline.py` | `final_pf_windowed_logistic_regression.csv` |
| 4 | `17_tft_three_seed_benchmark.py run-remaining` | TFT checkpoints, `final_pf_tft_seed*.csv` |
| 5 | `09_mc_dropout_uq.py` | `mc_dropout_population_predictions.csv` |
| 6 | `18_mc_dropout_convergence.py` | `final_mc_dropout_metrics_20_50_100.csv` |
| 7 | `19_final_sobol_sensitivity.py` | `final_sobol_indices_margin.csv` |
| 8 | `20_final_computational_efficiency.py` | `final_computational_efficiency_summary.csv` |
| 9 | `21_representative_tft_table5.py` | Table 5 CSV + representative error figure |
| 10 | `22_plot_mc_dropout_revised_figure.py` | MC Dropout uncertainty figure |
| 11 | `generate_fig3_revision.py` | Figure 3 (requires point-level predictions; see §7) |

Inference-only (no training, no data regeneration):

```bash
python run_revision_pipeline.py --skip-training --skip-data-generation
```

---

## 5. Expected final metrics (held-out test split)

From `outputs/paper/tables/final_model_comparison.csv` (population-level MAE / RMSE on cumulative `onset_flag`):

| Model | MAE | RMSE |
|---|---:|---:|
| GRU | 0.001934 | 0.002931 |
| TFT | 0.004542 | 0.006373 |
| Windowed MLP | 0.006975 | 0.009964 |
| Windowed multi-output linear/sigmoid regression | 0.020212 | 0.023823 |
| Pointwise Logistic Regression | 0.020652 | 0.024271 |

Population reference at 60 years (locked data): **Pf ≈ 0.310** (`docs/results/final_locked_parameter_configuration.md`).

Allow small numerical tolerance when re-running on different hardware (typically MAE ±0.0005).

---

## 6. Reproducing Tables 5–10

### Table 5 — Representative TFT accuracy

```bash
python scripts/21_representative_tft_table5.py
```

- **Output:** `outputs/revision/tables/final_representative_tft_accuracy_table5.csv` → synced to `outputs/paper/tables/`
- **Requires:** `outputs/revision/predictions/final_pf_tft_seed20250111.csv`
- **Figure:** `final_tft_time_dependent_error_seed20250111.png`

### Table 6 — Cover-depth groups at final evaluation year

Derived from the same point-level predictions as Figure 3 (broad groups 40–60, 60–80, 80–110 mm at `t_year ≈ 59.95`). See `archive/audit/Table6_verification_report.md` for aggregation logic.

- **Requires:** point predictions (`tft_*_10epoch_points.csv` in `archive/diagnostics/predictions/` for the reference run) + `final_onset_summary.csv` + `series_split.csv`
- **Regenerate:** restore points file or re-run TFT evaluation with point-level export; then run `scripts/generate_fig3_revision.py` and compute Table 6 per audit script logic

### Table 7 — TFT three-seed summary

```bash
python scripts/17_tft_three_seed_benchmark.py build-tft-summary
```

- **Output:** `final_tft_three_seed_summary.csv`, `final_tft_three_seed_results.csv`

### Table 8 — Computational efficiency

```bash
python scripts/20_final_computational_efficiency.py
```

- **Output:** `final_computational_efficiency_summary.csv`, `final_computational_efficiency_scaling.csv`
- **Figures:** `computational_efficiency_*.png`

### Table 9 — Sobol sensitivity indices

```bash
python scripts/19_final_sobol_sensitivity.py
```

- **Output:** `final_sobol_indices_margin.csv`, `final_sobol_indices_binary.csv`, `final_sobol_rank_summary.csv`
- **Note:** Full run evaluates 14,336 simulator samples (~tens of minutes)

### Table 10 — MC Dropout uncertainty metrics

```bash
python scripts/09_mc_dropout_uq.py
python scripts/18_mc_dropout_convergence.py
```

- **Output:** `final_mc_dropout_metrics_20_50_100.csv`, `final_mc_dropout_convergence_20_50_100.csv`
- **Figure:** `mc_dropout_uncertainty_band_revised.png` via `scripts/22_plot_mc_dropout_revised_figure.py`

### Table 4 — Full model comparison

Produced by `scripts/07_train_benchmarks.py` and rebuilt by `scripts/17_tft_three_seed_benchmark.py rebuild-benchmark`:

- **Output:** `final_model_comparison.csv`, `final_model_comparison_by_seed.csv`

---

## 7. Reproducing Figures 1–4

| Figure | Curated file | Primary script | Notes |
|---|---|---|---|
| **Figure 1** | `outputs/paper/figures/final_population_trajectories_by_model.png` | `07_train_benchmarks.py` / `17` rebuild | Population Pf(t) by model |
| **Figure 2** | `outputs/paper/figures/final_model_error_comparison.png` | `07_train_benchmarks.py` | Absolute error vs time |
| **Figure 3** | `outputs/paper/figures/Fig3_pf_by_cover_depth.pdf` | `generate_fig3_revision.py` | Needs point-level TFT predictions (see below) |
| **Figure 4** | `outputs/paper/figures/computational_efficiency_accuracy_vs_runtime.png` | `20_final_computational_efficiency.py` | Accuracy–runtime tradeoff |

**Figure 3 caveat:** `generate_fig3_revision.py` reads `outputs/revision/predictions/tft_20250111_10epoch_points.csv`. The reference file is archived at `archive/diagnostics/predictions/tft_20250111_10epoch_points.csv`. Copy it back before regenerating, or re-export point predictions from TFT evaluation.

Plot-only (no inference):

```bash
python scripts/22_plot_mc_dropout_revised_figure.py   # MC Dropout band from saved CSV
python scripts/21_representative_tft_table5.py        # Table 5 + error curve
```

---

## 8. Data and checkpoints

**Do not delete for reproduction:**

- `data/processed/revision/final_chloride_labeled.parquet`
- `data/processed/revision/series_split.csv`
- `data/processed/revision/final_onset_summary.csv`
- `outputs/revision/checkpoints/final_*`
- `outputs/revision/predictions/final_pf_*.csv`

Checkpoints are gitignored. Download from the GitHub release / Zenodo deposit if not training locally.

---

## 9. Runtime estimates (reference hardware)

| Stage | Approximate time |
|---|---|
| Data generation (1000 series) | < 5 min |
| MLP + GRU × 3 seeds | ~1–2 h |
| TFT × 3 seeds | ~12–15 h each |
| MC Dropout 50 passes | ~1.5 h |
| MC Dropout 100-pass convergence | ~2 h |
| Sobol N=2048 | ~30–60 min |
| Efficiency remeasurement | ~15 min |

Use `--skip-training` to validate plotting and tables in minutes.

---

## 10. Validation

Dry-run file check (no training):

```bash
python scripts/validate_release.py
```

See `docs/release_validation_report.md` for the latest results.
