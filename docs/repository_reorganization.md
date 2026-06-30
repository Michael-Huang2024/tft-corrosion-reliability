# Repository Reorganization Report

**Date:** 2026-06-30  
**Scope:** Phase 1–3 cleanup per `docs/repository_audit.md`  
**Paper pipeline:** Revision scripts (`12–23` + `revision_*`) are authoritative.

---

## Summary

| Action | Count |
|---|---:|
| Deleted | 9 |
| Archived | 125 |
| Moved (to `docs/results/` or `outputs/paper/`) | 55 |
| Kept for reproduction | 23 (+ core source/data) |

**Previous issue:** An earlier PowerShell command hung during recursive `__pycache__` deletion (~100+ minutes). Cleanup was completed in **~4 seconds** using a targeted Python script.

**Not modified:** `README.md`, model logic, no training runs, no `LICENSE`/`CITATION.cff`.

---

## Final Repository Tree

```text
tft-corrosion-reliability-main/
├── .gitignore                          [updated]
├── README.md                           [unchanged]
├── requirements.txt
├── run_pipeline.py                     [manuscript demo; revision is authoritative]
├── archive/
│   ├── legacy/                         [19 files — former src/legacy/]
│   ├── scripts/                        [7 superseded revision scripts]
│   ├── audit/                          [28 internal audit/report markdown files]
│   ├── data/
│   │   └── chloride_labeled_revision.parquet
│   └── diagnostics/
│       ├── checkpoints/                [33 non-final / duplicate checkpoints]
│       ├── predictions/                [17 smoke/diagnostic CSVs]
│       ├── tables/                     [10 superseded tables]
│       └── figures/                    [10 superseded figures]
├── data/
│   ├── raw/.gitkeep
│   ├── sim/.gitkeep
│   └── processed/
│       ├── chloride_labeled.parquet    [manuscript demo data]
│       ├── onset_summary.csv
│       └── revision/
│           ├── final_chloride_labeled.parquet   [paper dataset]
│           ├── final_onset_summary.csv
│           └── series_split.csv
├── docs/
│   ├── repository_audit.md
│   ├── repository_reorganization.md    [this file]
│   └── results/                        [12 final scientific reports]
├── outputs/
│   ├── paper/
│   │   ├── figures/                    [19 curated paper figures]
│   │   └── tables/                     [24 curated paper tables]
│   └── revision/
│       ├── checkpoints/                [10 final model checkpoints]
│       ├── predictions/                [13 final prediction CSVs]
│       └── environment.json
└── scripts/
    ├── 01_generate_data.py … 05_make_figures.py   [manuscript demo]
    ├── 07_train_benchmarks.py
    ├── 09_mc_dropout_uq.py
    ├── 12_generate_final_revision_data.py
    ├── 14_tft_stable.py
    ├── 17_tft_three_seed_benchmark.py
    ├── 18_mc_dropout_convergence.py
    ├── 19_final_sobol_sensitivity.py
    ├── 20_final_computational_efficiency.py
    ├── 21_representative_tft_table5.py
    ├── 22_plot_mc_dropout_revised_figure.py
    ├── 23_windowed_logistic_baseline.py
    ├── generate_fig3_revision.py
    ├── revision_config.py
    ├── revision_data.py
    ├── revision_metrics.py
    └── scripts_compat.py
```

**Removed directories:** `src/legacy/`, `src/` (empty), `.idea/`, `checkpoints/` (root), `outputs/revision/logs/`, empty `outputs/revision/figures/`, `outputs/revision/tables/`, `outputs/figures/`.

---

## Phase 1 — Hygiene (Deleted)

| Path | Reason |
|---|---|
| `.idea/` | JetBrains IDE config |
| `checkpoints/` (repo root) | Stray PyTorch Lightning output |
| `outputs/revision/logs/` | Training/eval logs (14 files) |
| `data/processed/revision/series_split_smoke.csv` | Smoke-test split |
| `outputs/revision/predictions/mc_dropout_seed20250111_100pass_state.json` | MC Dropout resume state |
| `data/sim/chloride_long.csv` | Regenerable simulation (~117 MB) |
| `data/sim/chloride_long.parquet` | Regenerable simulation |
| `scripts/__pycache__/` | Python bytecode cache |

---

## Phase 2 — Diagnostic Duplicates (Archived, Not Deleted)

### Checkpoints → `archive/diagnostics/checkpoints/`

| Category | Files |
|---|---|
| Duplicate TFT epoch saves | `final_tft_seed20250111-epoch=2-val_loss=0.0082-v{1,2}.ckpt`, `*-0.0158*.ckpt`, `epoch=3` |
| Smoke TFT (epoch 0, high val_loss) | `tft_seed202501{11,12,13}-epoch=0-*.ckpt` (8 files) |
| Non-final MLP/GRU | `mlp_seed*.pt`, `gru_seed*.pt` (6 files) |
| 10-epoch diagnostic runs | `tft/20250111_10epoch/`, `20250112_10epoch/`, `20250113/`, `20250113_10epoch_test/` |

### Predictions → `archive/diagnostics/predictions/`

- `pf_*.csv` (10 smoke benchmark files)
- `tft_*_10epoch.csv` and `tft_*_10epoch_points.csv` (6 files)
- `mc_dropout_seed20250111_100pass_level_population.csv` (4.1 MB pass-level storage)

### Tables → `archive/diagnostics/tables/`

- `model_comparison.csv`, `model_comparison_by_seed.csv`
- `sobol_indices.csv`, `runtime_scaling.csv`
- `mc_dropout_convergence.csv`, `mc_dropout_metrics.csv`
- `tft_*_10epoch_metrics.csv` (3 files)
- `benchmark_checkpoints.csv` (contained local absolute paths)

### Figures → `archive/diagnostics/figures/`

- `model_comparison.png`, `runtime_scaling.png`
- `mc_dropout_uncertainty_band.png`, `mc_dropout_uncertainty_band_100passes.png`
- `mc_dropout_uncertainty_band_revised_600dpi.png`
- `computational_efficiency_runtime_scaling.png`
- `sobol_binary_supplemental.png`, `sobol_s1_st_margin_comparison.png`, `sobol_total_effect_20_40_60.png`
- `Fig3_pf_by_cover_depth_600dpi.png`

---

## Phase 3 — Restructure

### 3.1 `archive/legacy/` (19 files)

All files from `src/legacy/`:

`00_init_project.py`, `00_sanity_check.py`, `01_generate_chloride_data.py`, `02_label_corrosion_onset.py`, `03_plot_pf_curve.py`, `03_train_TFT_onset.py`, `04_train_TFT_continuous.py`, `04_train_tft_time_to_onset.py`, `04b_train_tft_onset_flag.py`, `05_cross_validation_vs_paper.py`, `05_infer_time_to_onset.py`, `05b_infer_onset_flag_and_pf.py`, `05c_rolling_pf_fullcurve_stream.py`, `06_extract_predicted_onset.py`, `06_fig3_pf_by_cover_depth.py`, `07_plot_pf_compare.py`, `07b_plot_pf_fullcurve.py`, `08_make_paper_figures.py`, `utils.py`

### 3.2 `archive/scripts/` (7 files)

| Script | Superseded by |
|---|---|
| `06_baseline_comparison.py` | `07_train_benchmarks.py` + `final_model_comparison.csv` |
| `08_sobol_sensitivity.py` | `19_final_sobol_sensitivity.py` |
| `10_bootstrap_reference_uq.py` | Optional supplemental |
| `11_runtime_scaling.py` | `20_final_computational_efficiency.py` |
| `13_screen_parameter_candidates.py` | Completed; tables in `outputs/paper/tables/` |
| `15_tft_progress.py` | Dev log monitor |
| `16_tft_10epoch_diagnostic.py` | `17_tft_three_seed_benchmark.py` |

### 3.3 `archive/audit/` (28 files)

Internal reviewer-response and development audits, including:

`revision_experiment_audit_v2.md`, `PARAMETER_CONFIRMATION_REQUIRED.md`, `phase_1_4_status.md`, `physical_simulator_audit.md`, `tft_failure_diagnosis.md`, `Figure3_verification_report.md`, `Table6_verification_report.md`, and 21 others.

### 3.4 `archive/data/`

| File | Reason |
|---|---|
| `chloride_labeled_revision.parquet` | Intermediate relabeled manuscript data; superseded by `final_chloride_labeled.parquet` |

### 3.5 `docs/results/` (12 final reports)

| Report |
|---|
| `final_benchmark_report.md` |
| `final_sobol_sensitivity_report.md` |
| `final_computational_efficiency_report.md` |
| `final_tft_three_seed_report.md` |
| `final_locked_parameter_configuration.md` |
| `final_target_validation_report.md` |
| `final_split_validation_report.md` |
| `final_data_generation_report.md` |
| `final_data_sufficiency_report.md` |
| `final_locked_data_sufficiency_report.md` |
| `final_representative_tft_error_report.md` |
| `final_mc_dropout_50_vs_100_convergence_report.md` |

### 3.6 `outputs/paper/figures/` (19 files)

| Figure | Paper role |
|---|---|
| `Fig3_pf_by_cover_depth.pdf`, `.png` | Figure 3 |
| `final_model_error_comparison.png` | Model error |
| `final_population_trajectories_by_model.png` | Population Pf(t) |
| `final_tft_three_seed_trajectories.png` | TFT seed variability |
| `final_tft_time_dependent_error_seed20250111.png` | Time-dependent error |
| `final_tft_seed_metric_variability.png` | Seed metrics |
| `mc_dropout_uncertainty_band_revised.png` | MC Dropout UQ |
| `mc_dropout_mean_convergence_20_50_100.png` | MC convergence |
| `mc_dropout_std_convergence_20_50_100.png` | MC convergence |
| `mc_dropout_convergence_difference_50_vs_100.png` | MC convergence |
| `computational_efficiency_accuracy_vs_runtime.png` | Table 8 / efficiency |
| `computational_efficiency_total_runtime.png` | Table 8 |
| `computational_efficiency_inference_only.png` | Table 8 |
| `sobol_s1_margin_20_40_60.png` | Sobol sensitivity |
| `sobol_st_margin_20_40_60.png` | Sobol sensitivity |
| `sobol_time_evolution.png` | Sobol time evolution |
| `parameter_candidate_pf_curves.png` | Parameter screening |
| `parameter_candidate_cover_groups.png` | Parameter screening |

### 3.7 `outputs/paper/tables/` (24 files)

All `final_*.csv`, `final_sobol_*.parquet`, `final_sobol_state.json`, and parameter-candidate screening tables:

- `final_model_comparison.csv`, `final_model_comparison_by_seed.csv`
- `final_computational_efficiency_summary.csv`, `final_computational_efficiency_scaling.csv`
- `final_representative_tft_accuracy_table5.csv` (Table 5)
- `final_mc_dropout_convergence_20_50_100.csv`, `final_mc_dropout_metrics_20_50_100.csv`
- `final_sobol_indices_binary.csv`, `final_sobol_indices_margin.csv`, `final_sobol_rank_summary.csv`
- `final_sobol_responses.parquet`, `final_sobol_samples.parquet`, `final_sobol_sampling_diagnostics.csv`, `final_sobol_state.json`
- `final_tft_three_seed_results.csv`, `final_tft_three_seed_summary.csv`
- `final_training_summary.csv`, `final_training_runtime_summary.csv`, `final_input_distribution_summary.csv`
- `parameter_candidate_screening.csv`, `parameter_candidate_{A,B,C}_distribution_summary.csv`, `parameter_candidate_cover_groups.csv`

---

## Files Kept for Reproduction

### Paper data (`data/processed/revision/`)

| File | Role |
|---|---|
| `final_chloride_labeled.parquet` | Locked Candidate C dataset (783k rows) |
| `final_onset_summary.csv` | Onset summary |
| `series_split.csv` | 700/150/150 train/val/test split |

### Final checkpoints (`outputs/revision/checkpoints/`)

| File | Model |
|---|---|
| `final_tft_seed20250111-epoch=2-val_loss=0.0082.ckpt` | TFT seed 20250111 |
| `final_tft_seed20250112-epoch=7-val_loss=0.0064.ckpt` | TFT seed 20250112 |
| `final_tft_seed20250113-epoch=0-val_loss=0.0173.ckpt` | TFT seed 20250113 |
| `final_mlp_seed20250111.pt` … `final_mlp_seed20250113.pt` | MLP × 3 |
| `final_gru_seed20250111.pt` … `final_gru_seed20250113.pt` | GRU × 3 |
| `final_windowed_logistic_regression.joblib` | Windowed LR |

### Final predictions (`outputs/revision/predictions/`)

- `final_pf_*.csv` (11 files — all models/seeds)
- `mc_dropout_population_predictions.csv`
- `mc_dropout_seed20250111_summary_20_50_100.csv`

### Active revision scripts (`scripts/`)

`07`, `09`, `12`, `14`, `17`, `18`, `19`, `20`, `21`, `22`, `23`, `generate_fig3_revision.py`, `revision_config.py`, `revision_data.py`, `revision_metrics.py`, `scripts_compat.py`

### Manuscript demo (secondary)

`run_pipeline.py`, `scripts/01–05`, `data/processed/chloride_labeled.parquet`, `onset_summary.csv`

### Environment record

`outputs/revision/environment.json` — hardware/package snapshot from experiments

---

## `.gitignore` Updates

Added per audit:

- `/checkpoints/` (root stray output)
- `data/sim/*.csv`, `data/sim/*.parquet`
- `outputs/revision/checkpoints/`, `*.ckpt`, `*.pt`, `*.joblib`
- Large intermediate predictions and `archive/diagnostics/`
- `outputs/revision/logs/`

---

## Uncertain Files — Human Approval Recommended

| File / topic | Question | Current disposition |
|---|---|---|
| `data/processed/chloride_labeled.parquet` + `onset_summary.csv` | Keep for manuscript demo, or archive if release is revision-only? | **Kept** in place |
| `outputs/revision/environment.json` | Commit to git or move to `docs/`? | **Kept** in `outputs/revision/` |
| `archive/diagnostics/predictions/mc_dropout_seed20250111_100pass_level_population.csv` | Needed for exact MC Dropout figure reproduction? | **Archived** (summary CSV kept) |
| `archive/audit/phase_1_4_status.md` | Useful methods record vs internal note? | **Archived** |
| `archive/scripts/13_screen_parameter_candidates.py` | Re-run parameter screening, or tables-only? | **Archived** (tables in `outputs/paper/`) |
| Entire `archive/` directory | Ship in public repo or exclude via `.gitignore`? | **Present**; `archive/diagnostics/` gitignored |
| `*.ckpt` / `*.pt` in `.gitignore` | Blocks committing final checkpoints; use Git LFS or Zenodo? | **Gitignored** — needs release strategy |
| `scripts/01–05` vs revision `12–23` | README still points to demo pipeline only | **No change** (per instructions) |

---

## Script Path Compatibility Note

Revision scripts (`revision_config.py`) still write to `outputs/revision/figures/` and `outputs/revision/tables/` on re-run. Curated paper artifacts now live in `outputs/paper/`. After re-running experiments, copy or symlink new `final_*` outputs to `outputs/paper/` for release.

---

## Next Steps (Not Done in This Cleanup)

1. Rewrite `README.md` — revision pipeline as authoritative
2. Add `LICENSE`, `CITATION.cff`, `docs/REPRODUCIBILITY.md`
3. Add `run_revision_pipeline.py`
4. Decide checkpoint/data hosting (Git LFS vs Zenodo)
5. Phase 4–5 validation from `docs/repository_audit.md`

---

*Generated after Phase 1–3 cleanup. Temporary helper `scripts/_reorganize_repo.py` was removed after execution.*
