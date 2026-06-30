# Repository Audit for Public GitHub Research Release

**Paper:** *Transformer-Based Sequence Learning for Multi-Horizon Corrosion Initiation Probability Prediction in Reinforced Concrete Bridges*

**Audit date:** 2026-06-30  
**Repository:** `tft-corrosion-reliability-main`  
**Git branch:** `main`  
**Scope:** Read-only structural audit. No files were modified except creation of this report.

---

## Executive Summary

This repository contains **two overlapping scientific workflows** that evolved during manuscript preparation and reviewer revision:

| Workflow | Entry point | Data source | Output root | Status |
|---|---|---|---|---|
| **Manuscript demo pipeline** | `run_pipeline.py` → `scripts/01–05` | `scripts/01_generate_data.py` (legacy parameter ranges) | `outputs/` | Tracked in git; README describes this path |
| **Revision / paper pipeline** | `scripts/12–23` + shared `revision_*` modules | `scripts/12_generate_final_revision_data.py` (locked Candidate C distributions) | `outputs/revision/` | Mostly **untracked** locally; scientifically complete |

**Current state:** 255 files (~206 MB on disk). Only **34 files are git-tracked** (~7.6 MB). The revision experiment layer (scripts 07–23, `data/processed/revision/`, `outputs/revision/`) exists locally but is **not committed**.

**Release readiness:** **Not ready** for public release without cleanup and documentation consolidation. Core issues:

1. **Dual pipelines** with different parameter distributions are not explained in the README.
2. **~190 MB of generated artifacts** (data, checkpoints, predictions) are mixed with source code; many are duplicates or diagnostics.
3. **`src/legacy/`** (19 scripts) largely duplicates the active `scripts/01–05` workflow plus abandoned experiments.
4. **Stray artifacts** at repo root (`checkpoints/`, `.idea/`, `__pycache__/`) should not ship.
5. **No `LICENSE`**, **`CITATION.cff`**, or **`docs/REPRODUCIBILITY.md`** for a research release.
6. **`outputs/revision_experiment_audit_v2.md`** describes blockers that have since been addressed in code (e.g. `onset_flag` is now used in scripts 03–06); it is **stale** and will confuse readers.

**Recommended strategy:** Ship a **curated reproducibility release** with (a) source + locked configuration, (b) one representative pretrained checkpoint set, (c) final paper tables/figures, and (d) instructions to regenerate everything. Archive development history; delete diagnostic duplicates.

---

## 1. Current Repository Tree

```text
tft-corrosion-reliability-main/                    [~206 MB, 255 files]
├── .gitignore
├── .idea/                                         [IDE config — should not release]
├── README.md                                      [tracked]
├── requirements.txt                               [tracked, fully pinned]
├── run_pipeline.py                                [tracked, manuscript pipeline only]
├── checkpoints/                                   [UNTRACKED, stray Lightning output — 2.6 MB]
│   ├── epoch=0-step=2.ckpt
│   └── epoch=0-step=50.ckpt
├── data/                                          [~143 MB]
│   ├── raw/.gitkeep
│   ├── sim/                                       [gitignored except .gitkeep]
│   │   ├── chloride_long.parquet                  [7.1 MB, untracked]
│   │   └── chloride_long.csv                      [117 MB, untracked]
│   └── processed/
│       ├── chloride_labeled.parquet               [7.4 MB, tracked]
│       ├── onset_summary.csv                      [tracked]
│       └── revision/                              [UNTRACKED]
│           ├── chloride_labeled_revision.parquet  [7.5 MB, relabeled manuscript data]
│           ├── final_chloride_labeled.parquet     [7.0 MB, locked Candidate C data]
│           ├── final_onset_summary.csv
│           ├── series_split.csv
│           └── series_split_smoke.csv             [smoke-test split]
├── docs/
│   └── repository_audit.md                        [this file]
├── outputs/                                       [~28 MB excluding revision sub-tree counts overlap]
│   ├── figures/
│   │   ├── Fig3_pf_by_cover_depth.pdf/png/_600dpi.png   [partial manuscript figures only]
│   └── revision/                                  [~25 MB artifacts + 33 MB checkpoints]
│       ├── checkpoints/                           [33 models, many duplicates — see §5]
│       ├── figures/                               [27 PNG figures]
│       ├── logs/                                  [14 training/eval logs]
│       ├── predictions/                           [36 CSV files, ~21 MB]
│       ├── tables/                                [37 CSV/parquet/json files]
│       └── *.md                                   [45 audit/report markdown files]
│   └── revision_experiment_audit_v2.md          [stale pre-revision audit]
├── scripts/                                       [28 Python files]
│   ├── 01_generate_data.py … 05_make_figures.py   [tracked, manuscript pipeline]
│   ├── 06_baseline_comparison.py                [tracked, early reviewer baseline]
│   ├── 07_train_benchmarks.py … 23_windowed_logistic_baseline.py  [UNTRACKED revision suite]
│   ├── revision_config.py, revision_data.py, revision_metrics.py
│   ├── scripts_compat.py
│   └── generate_fig3_revision.py
└── src/legacy/                                    [19 exploratory scripts, all tracked]
    ├── 00_init_project.py                         [misnamed duplicate of 01_generate_chloride_data]
    ├── 00_sanity_check.py
    ├── 01_generate_chloride_data.py … 08_make_paper_figures.py
    └── utils.py
```

**Git tracking summary**

| Category | Tracked | Untracked |
|---|---:|---:|
| Root config | 4 | 0 |
| `scripts/` | 6 | 22 |
| `src/legacy/` | 19 | 0 |
| `data/` | 4 | 8 |
| `outputs/` | 0 | ~178 |
| Other (`.idea`, `checkpoints/`, `__pycache__`) | 0 | ~23 |

---

## 2. File Purposes

### 2.1 Root & configuration

| File | Purpose | Recommendation |
|---|---|---|
| `README.md` | Setup and manuscript `01–05` pipeline | **Keep** — must be rewritten for dual-pipeline clarity |
| `requirements.txt` | Pinned Python dependencies (torch, lightning, pytorch-forecasting, SALib, …) | **Keep** — consider splitting `requirements-dev.txt` |
| `run_pipeline.py` | Orchestrates `01→02→03→04→05` | **Keep** as quick demo; add `run_revision_pipeline.py` for paper |
| `.gitignore` | Ignores `.venv`, `data/sim/`, `outputs/checkpoints/`, etc. | **Keep** — extend for revision artifacts |
| `.idea/` | JetBrains IDE project files | **Delete** |
| `checkpoints/` (root) | Accidental PyTorch Lightning default directory | **Delete** |

### 2.2 Manuscript pipeline (`scripts/01–05`)

| Script | Step | Inputs | Outputs | Notes |
|---|---|---|---|---|
| `01_generate_data.py` | Chloride diffusion simulation | CLI `--n-series` | `data/sim/chloride_long.{parquet,csv}` | Uses **legacy uniform parameter ranges** (not locked Candidate C) |
| `02_label_onset.py` | Onset labeling | `data/sim/*` | `data/processed/chloride_labeled.parquet`, `onset_summary.csv` | Computes cumulative `onset_flag` |
| `03_train_model.py` | TFT training | labeled parquet | `outputs/checkpoints/tft_onset_flag-*.ckpt` | Target: `onset_flag`; checkpoints gitignored |
| `04_infer.py` | Rolling TFT inference → Pf(t) | checkpoint + labeled data | `outputs/predictions/pf_full_true_vs_pred.csv`, etc. | Streaming Pf reconstruction |
| `05_make_figures.py` | Manuscript figures 1–4 | predictions | `outputs/figures/Fig*.png/pdf`, `outputs/tables/*` | Only Fig3 currently present in `outputs/figures/` |

### 2.3 Revision shared modules

| Module | Purpose | Recommendation |
|---|---|---|
| `revision_config.py` | Locked parameter distributions (Candidate C), paths, TFT hyperparameters, split fractions | **Keep** — canonical config for paper |
| `revision_data.py` | Series-level splits, leakage checks, cumulative target validation | **Keep** |
| `revision_metrics.py` | Fair MAE/RMSE/Pf curve evaluation across models | **Keep** |
| `scripts_compat.py` | Dynamic import helper for `07_train_benchmarks` classes | **Keep** |

### 2.4 Revision experiment scripts (`scripts/06–23`)

| Script | Purpose | Phase | Recommendation |
|---|---|---|---|
| `06_baseline_comparison.py` | Pointwise logistic regression baseline → `outputs/tables/` | Early revision | **Archive** — superseded by `07` + `final_*` tables |
| `07_train_benchmarks.py` | Train LR, MLP, GRU, TFT on cumulative target | Core benchmark | **Keep** |
| `08_sobol_sensitivity.py` | Early/blocked Sobol (smoke) | Diagnostic | **Archive** — superseded by `19` |
| `09_mc_dropout_uq.py` | MC Dropout uncertainty for TFT | UQ | **Keep** |
| `10_bootstrap_reference_uq.py` | Bootstrap bands on reference Pf(t) | Optional supplemental | **Archive** (optional reproduce) |
| `11_runtime_scaling.py` | Runtime scaling benchmark | Diagnostic | **Archive** — superseded by `20` |
| `12_generate_final_revision_data.py` | Generate locked Candidate C dataset | **Data lock** | **Keep** — required for reproduction |
| `13_screen_parameter_candidates.py` | Simulator-only parameter screening A/B/C | Pre-lock audit | **Archive** script; **keep** screening tables/figures |
| `14_tft_stable.py` | Stable TFT trainer (Windows DLL fix, separate train/eval) | Core benchmark | **Keep** — preferred TFT entry point |
| `15_tft_progress.py` | Parse logs → progress display | Dev utility | **Delete** |
| `16_tft_10epoch_diagnostic.py` | 10-epoch diagnostic for seed 20250113 | Diagnostic | **Archive** |
| `17_tft_three_seed_benchmark.py` | Three-seed TFT benchmark orchestration | Core benchmark | **Keep** |
| `18_mc_dropout_convergence.py` | 20/50/100-pass MC Dropout convergence | UQ validation | **Keep** |
| `19_final_sobol_sensitivity.py` | Final Saltelli Sobol on locked parameters | Sensitivity | **Keep** |
| `20_final_computational_efficiency.py` | Unified efficiency comparison | Efficiency | **Keep** |
| `21_representative_tft_table5.py` | Table 5 + representative error figure | Paper tables | **Keep** |
| `22_plot_mc_dropout_revised_figure.py` | Replot MC Dropout figure from saved CSV | Figure utility | **Keep** |
| `23_windowed_logistic_baseline.py` | Windowed logistic regression baseline | Benchmark | **Keep** |
| `generate_fig3_revision.py` | Fig 3 from revision TFT predictions | Figure utility | **Keep** |

### 2.5 Legacy code (`src/legacy/`)

| File | Relationship to active code | Recommendation |
|---|---|---|
| `01_generate_chloride_data.py` | Ancestor of `scripts/01_generate_data.py` | **Archive** |
| `00_init_project.py` | **Misnamed duplicate** of `01_generate_chloride_data.py` | **Archive** |
| `02_label_corrosion_onset.py` | Ancestor of `scripts/02_label_onset.py` | **Archive** |
| `04b_train_tft_onset_flag.py` | Ancestor of `scripts/03_train_model.py` | **Archive** |
| `05b_infer_onset_flag_and_pf.py` | Ancestor of `scripts/04_infer.py` | **Archive** |
| `05c_rolling_pf_fullcurve_stream.py` | Streaming inference prototype | **Archive** |
| `08_make_paper_figures.py` | Ancestor of `scripts/05_make_figures.py` | **Archive** |
| `06_fig3_pf_by_cover_depth.py` | Ancestor of `generate_fig3_revision.py` | **Archive** |
| `04_train_tft_time_to_onset.py` | Abandoned **regression** target | **Archive** |
| `05_infer_time_to_onset.py` | Inference for regression model | **Archive** |
| `06_extract_predicted_onset.py` | Post-process regression outputs | **Archive** |
| `03_train_TFT_onset.py`, `04_train_TFT_continuous.py`, `05_cross_validation_vs_paper.py` | **TODO stubs** (`print("TODO")`) | **Delete** |
| `03_plot_pf_curve.py`, `07_plot_pf_compare.py`, `07b_plot_pf_fullcurve.py` | Exploratory plotting | **Archive** |
| `00_sanity_check.py` | One-off sanity check | **Archive** |
| `utils.py` | Empty/minimal placeholder | **Delete** |

### 2.6 Data artifacts

| File | Purpose | Recommendation |
|---|---|---|
| `data/processed/chloride_labeled.parquet` | Manuscript-pipeline labeled data (legacy params) | **Keep** (demo) or **archive** if revision-only release |
| `data/processed/onset_summary.csv` | Manuscript onset summary | **Keep** (demo) |
| `data/sim/chloride_long.*` | Raw simulation output (regenerable) | **Delete from repo** — gitignore; regenerate via `01` or `12` |
| `data/processed/revision/chloride_labeled_revision.parquet` | Manuscript data relabeled with corrected `onset_flag` | **Archive** — intermediate; superseded by `final_*` |
| `data/processed/revision/final_chloride_labeled.parquet` | **Locked Candidate C** training data (783k rows) | **Keep** — primary paper dataset |
| `data/processed/revision/final_onset_summary.csv` | Onset summary for final data | **Keep** |
| `data/processed/revision/series_split.csv` | 700/150/150 series split | **Keep** |
| `data/processed/revision/series_split_smoke.csv` | Smoke-test subset split | **Delete** |

### 2.7 Model checkpoints (`outputs/revision/checkpoints/`)

**Canonical final checkpoints (validation-selected, paper results):**

| File | Model / seed | Keep? |
|---|---|---|
| `final_tft_seed20250111-epoch=2-val_loss=0.0082.ckpt` | TFT seed 20250111 (best) | **Keep** |
| `final_tft_seed20250112-epoch=7-val_loss=0.0064.ckpt` | TFT seed 20250112 | **Keep** |
| `final_tft_seed20250113-epoch=0-val_loss=0.0173.ckpt` | TFT seed 20250113 | **Keep** |
| `final_mlp_seed20250111.pt` … `final_mlp_seed20250113.pt` | MLP × 3 seeds | **Keep** |
| `final_gru_seed20250111.pt` … `final_gru_seed20250113.pt` | GRU × 3 seeds | **Keep** |
| `final_windowed_logistic_regression.joblib` | Windowed LR | **Keep** |

**Duplicates / diagnostics to remove (~18 MB recoverable):**

| Pattern | Count | Reason |
|---|---:|---|
| `tft_seed*-epoch=0-val_loss=0.3–0.45*.ckpt` | 8 | Failed/smoke epoch-0 runs; superseded by `final_tft_*` |
| `final_tft_seed20250111-epoch=2-val_loss=0.0082-v{1,2}.ckpt` | 2 | Duplicate saves |
| `final_tft_seed20250111-epoch=2-val_loss=0.0158*.ckpt` | 2 | Worse epoch for same seed |
| `final_tft_seed20250111-epoch=3-val_loss=0.0169.ckpt` | 1 | Worse epoch |
| `tft/20250111_10epoch/`, `20250112_10epoch/`, `20250113_10epoch_test/` | 9 files | Diagnostic 10-epoch runs |
| `tft/20250113/best-epoch=00*.ckpt`, `last*.ckpt` | 2 | Superseded |
| `mlp_seed*.pt`, `gru_seed*.pt` (non-`final_`) | 6 | Smoke/non-final weights |
| `checkpoints/epoch=0-step=*.ckpt` (repo root) | 2 | Stray Lightning output |

### 2.8 Predictions (`outputs/revision/predictions/`)

| Pattern | Purpose | Recommendation |
|---|---|---|
| `final_pf_*.csv` (11 files) | **Final** population Pf(t) per model/seed | **Keep** |
| `pf_*.csv` (10 files, no `final_` prefix) | Smoke / intermediate benchmark outputs | **Delete** |
| `tft_*_10epoch.csv` | Diagnostic population curves | **Delete** (keep summary tables if needed) |
| `tft_*_10epoch_points.csv` (3 files, ~16 MB) | Point-level diagnostic predictions | **Delete** — largest prediction bloat |
| `mc_dropout_seed20250111_100pass_level_population.csv` (4.1 MB) | Pass-level MC Dropout storage | **Archive** or Zenodo; keep summary CSV |
| `mc_dropout_population_predictions.csv` | 50-pass aggregated MC Dropout | **Keep** |
| `mc_dropout_seed20250111_summary_20_50_100.csv` | Convergence summary | **Keep** |
| `mc_dropout_seed20250111_100pass_state.json` | Resume state | **Delete** after analysis complete |

### 2.9 Figures (`outputs/revision/figures/` + `outputs/figures/`)

**Keep for paper release (curate into `outputs/paper/figures/`):**

- `final_model_error_comparison.png`
- `final_population_trajectories_by_model.png`
- `final_tft_three_seed_trajectories.png`
- `final_tft_time_dependent_error_seed20250111.png`
- `mc_dropout_uncertainty_band_revised.png` (drop `_600dpi` duplicate or keep one high-res)
- `computational_efficiency_accuracy_vs_runtime.png`
- `computational_efficiency_total_runtime.png`
- `sobol_s1_margin_20_40_60.png`, `sobol_st_margin_20_40_60.png`, `sobol_time_evolution.png`
- `parameter_candidate_pf_curves.png`, `parameter_candidate_cover_groups.png` (methods validation)
- `outputs/figures/Fig3_pf_by_cover_depth.pdf` (+ one PNG)

**Archive or delete:**

- `model_comparison.png` — smoke-era figure
- `runtime_scaling.png` — superseded by `computational_efficiency_*`
- `mc_dropout_uncertainty_band.png`, `mc_dropout_uncertainty_band_100passes.png` — superseded by `_revised`
- `mc_dropout_*_convergence_*.png` — keep if cited in supplement; else archive
- `sobol_binary_supplemental.png`, `sobol_s1_st_margin_comparison.png`, `sobol_total_effect_20_40_60.png` — supplemental only

### 2.10 Reports & audit markdown (`outputs/revision/*.md`)

| Report | Role | Recommendation |
|---|---|---|
| `final_benchmark_report.md` | **Final model comparison summary** | **Keep** → move to `docs/results/` |
| `final_sobol_sensitivity_report.md` | Final Sobol results | **Keep** |
| `final_computational_efficiency_report.md` | Runtime/efficiency | **Keep** |
| `final_tft_three_seed_report.md` | TFT seed variability | **Keep** |
| `final_locked_parameter_configuration.md` | Locked parameters | **Keep** |
| `final_target_validation_report.md`, `final_split_validation_report.md` | Validation gates | **Keep** |
| `phase_1_4_status.md` | Phase completion record | **Archive** |
| `physical_simulator_audit.md`, `parameter_candidate_screening_report.md` | Methods audit trail | **Archive** in `docs/audit/` |
| `PARAMETER_CONFIRMATION_REQUIRED.md` | **Resolved** blocker notice | **Delete** (superseded by locked config) |
| `revision_experiment_report.md` | Early incomplete status | **Delete** (stale) |
| `outputs/revision_experiment_audit_v2.md` | Pre-fix audit (incorrect for current code) | **Archive** with staleness warning |
| `tft_failure_diagnosis.md`, `tft_gpu_environment_audit.md`, `smoke_test_report.md` | Dev diagnostics | **Archive** |
| `*_audit*.md`, `*_verification_report.md` (15+ files) | Internal reviewer-response working notes | **Archive** — not for public root |

---

## 3. Redundancy Analysis

### 3.1 Duplicated script lineages

```text
src/legacy/01_generate_chloride_data.py  ──►  scripts/01_generate_data.py
src/legacy/02_label_corrosion_onset.py  ──►  scripts/02_label_onset.py
src/legacy/04b_train_tft_onset_flag.py    ──►  scripts/03_train_model.py
src/legacy/05b_infer_onset_flag_and_pf.py ──►  scripts/04_infer.py
src/legacy/08_make_paper_figures.py       ──►  scripts/05_make_figures.py
src/legacy/06_fig3_pf_by_cover_depth.py   ──►  scripts/generate_fig3_revision.py
```

The legacy folder preserves **four generations** of experimentation (regression → instantaneous → cumulative flag → revision benchmarks). None of it is required to reproduce paper results if `scripts/01–05` and the revision suite are documented.

### 3.2 Duplicated scientific pipelines

| Concern | Manuscript `01` | Revision `12` |
|---|---|---|
| Parameter sampling | Uniform ranges in `SimConfig` | Locked truncated lognormal/normal (Candidate C) |
| Output path | `data/sim/` → `data/processed/` | `data/processed/revision/final_*` |
| Pf(60) reference | Lower (legacy params) | **0.310** (locked params) |
| Paper alignment | Demo / concept | **Authoritative for revised manuscript** |

**Risk:** Running `run_pipeline.py` does **not** reproduce revision benchmark tables. Public docs must state this explicitly.

### 3.3 Duplicated experiment outputs

| Superseded | Superseded by |
|---|---|
| `tables/model_comparison.csv` | `tables/final_model_comparison.csv` |
| `tables/sobol_indices.csv` | `tables/final_sobol_indices_{margin,binary}.csv` |
| `tables/runtime_scaling.csv` | `tables/final_computational_efficiency_*.csv` |
| `predictions/pf_*.csv` | `predictions/final_pf_*.csv` |
| `checkpoints/tft_seed*_epoch=0_*` | `checkpoints/final_tft_*` |
| `08_sobol_sensitivity.py` | `19_final_sobol_sensitivity.py` |
| `11_runtime_scaling.py` | `20_final_computational_efficiency.py` |
| `07_train_benchmarks.py` (TFT path) | `14_tft_stable.py` + `17_tft_three_seed_benchmark.py` |

### 3.4 README vs reality gaps

| README claims | Actual state |
|---|---|
| `outputs/figures/Fig1–Fig4` exist after pipeline | Only **Fig3** present; Figs 1, 2, 4 missing |
| `outputs/checkpoints/` used by `--skip-training` | Directory empty/gitignored; no bundled checkpoint |
| Pipeline "fully reproduces reported results" | Revision results require **untracked** scripts 07–23 |
| `src/legacy/` only for audit | Correct, but entire revision layer also uncommitted |

---

## 4. File Categorization

Legend: **K** = Keep in public release · **A** = Archive (move to `archive/` or supplemental Zenodo) · **D** = Delete

### 4.1 Root & config

| File | Action | Rationale |
|---|---|---|
| `README.md` | K | Essential; needs rewrite |
| `requirements.txt` | K | Reproducibility |
| `run_pipeline.py` | K | Demo entry point |
| `.gitignore` | K | Extend before release |
| `.idea/` | D | IDE-specific |
| `checkpoints/` (root) | D | Stray artifacts |

### 4.2 Scripts — keep (K)

`01_generate_data.py`, `02_label_onset.py`, `03_train_model.py`, `04_infer.py`, `05_make_figures.py`, `revision_config.py`, `revision_data.py`, `revision_metrics.py`, `scripts_compat.py`, `07_train_benchmarks.py`, `09_mc_dropout_uq.py`, `12_generate_final_revision_data.py`, `14_tft_stable.py`, `17_tft_three_seed_benchmark.py`, `18_mc_dropout_convergence.py`, `19_final_sobol_sensitivity.py`, `20_final_computational_efficiency.py`, `21_representative_tft_table5.py`, `22_plot_mc_dropout_revised_figure.py`, `23_windowed_logistic_baseline.py`, `generate_fig3_revision.py`

### 4.3 Scripts — archive (A)

`06_baseline_comparison.py`, `08_sobol_sensitivity.py`, `10_bootstrap_reference_uq.py`, `11_runtime_scaling.py`, `13_screen_parameter_candidates.py`, `16_tft_10epoch_diagnostic.py`, `15_tft_progress.py`

### 4.4 Scripts — delete (D)

None at script level if archiving is preferred; `15_tft_progress.py` is the only strong **D** candidate (pure dev monitor).

### 4.5 `src/legacy/` — archive entire directory (A)

All 19 files → `archive/legacy/`. Optionally **D** the three TODO stubs and `utils.py` instead of archiving.

### 4.6 Data

| File | Action |
|---|---|
| `data/raw/.gitkeep`, `data/sim/.gitkeep`, `data/processed/.gitkeep` | K |
| `data/processed/chloride_labeled.parquet`, `onset_summary.csv` | K (demo) or A (revision-only release) |
| `data/sim/chloride_long.*` | D from repo (regenerate) |
| `data/processed/revision/final_chloride_labeled.parquet` | K |
| `data/processed/revision/final_onset_summary.csv` | K |
| `data/processed/revision/series_split.csv` | K |
| `data/processed/revision/chloride_labeled_revision.parquet` | A |
| `data/processed/revision/series_split_smoke.csv` | D |

### 4.7 Checkpoints — keep (K)

`final_tft_seed202501{11,12,13}-*.ckpt` (best epoch only per seed), `final_mlp_seed*.pt` ×3, `final_gru_seed*.pt` ×3, `final_windowed_logistic_regression.joblib`

### 4.8 Checkpoints — delete (D)

All other files under `outputs/revision/checkpoints/` (~18 MB) and root `checkpoints/`.

### 4.9 Predictions — keep (K)

All `final_pf_*.csv`, `mc_dropout_population_predictions.csv`, `mc_dropout_seed20250111_summary_20_50_100.csv`

### 4.10 Predictions — delete (D)

All `pf_*.csv`, `tft_*_10epoch*.csv`, `mc_dropout_seed20250111_100pass_level_population.csv`, `mc_dropout_seed20250111_100pass_state.json`

### 4.11 Tables — keep (K)

All `final_*.csv`, `final_sobol_*.parquet`, `parameter_candidate_screening.csv`, `parameter_candidate_{A,B,C}_distribution_summary.csv`

### 4.12 Tables — delete (D)

`model_comparison.csv`, `model_comparison_by_seed.csv`, `sobol_indices.csv`, `runtime_scaling.csv`, `mc_dropout_convergence.csv`, `mc_dropout_metrics.csv`, `tft_*_10epoch_metrics.csv`, `benchmark_checkpoints.csv` (contains local absolute paths)

### 4.13 Figures — keep (K)

Listed in §2.9 first group; prefer PDF + one PNG per figure.

### 4.14 Figures — archive or delete (A/D)

Smoke, duplicate-resolution, and superseded figures per §2.9 second group.

### 4.15 Logs — delete (D)

Entire `outputs/revision/logs/` (14 files) — regenerate on retraining; not needed for inference-only reproduction.

### 4.16 Reports — keep in `docs/` (K)

`final_benchmark_report.md`, `final_sobol_sensitivity_report.md`, `final_computational_efficiency_report.md`, `final_tft_three_seed_report.md`, `final_locked_parameter_configuration.md`, `final_target_validation_report.md`, `final_split_validation_report.md`, `final_data_generation_report.md`, `final_representative_tft_error_report.md`, `final_mc_dropout_50_vs_100_convergence_report.md`

### 4.17 Reports — archive (A)

All other `outputs/revision/*.md` and `outputs/revision_experiment_audit_v2.md`

### 4.18 New files to create (not present)

| File | Action |
|---|---|
| `LICENSE` (MIT or Apache-2.0 recommended) | K — **create** |
| `CITATION.cff` | K — **create** |
| `docs/REPRODUCIBILITY.md` | K — **create** |
| `run_revision_pipeline.py` | K — **create** (orchestrate 12→17→19→20→21) |

---

## 5. Recommended Final Structure

Target layout for a clean public research release:

```text
tft-corrosion-reliability/
├── README.md                      # Overview, citation, quickstart, hardware notes
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── run_pipeline.py                # Optional: manuscript demo (legacy parameters)
├── run_revision_pipeline.py       # Primary: full paper reproduction
├── scripts/
│   ├── 01_generate_data.py … 05_make_figures.py
│   ├── revision_config.py
│   ├── revision_data.py
│   ├── revision_metrics.py
│   ├── scripts_compat.py
│   ├── 07_train_benchmarks.py
│   ├── 09_mc_dropout_uq.py
│   ├── 12_generate_final_revision_data.py
│   ├── 14_tft_stable.py
│   ├── 17_tft_three_seed_benchmark.py
│   ├── 18_mc_dropout_convergence.py
│   ├── 19_final_sobol_sensitivity.py
│   ├── 20_final_computational_efficiency.py
│   ├── 21_representative_tft_table5.py
│   ├── 22_plot_mc_dropout_revised_figure.py
│   ├── 23_windowed_logistic_baseline.py
│   └── generate_fig3_revision.py
├── data/
│   ├── raw/.gitkeep
│   ├── sim/.gitkeep               # generated; gitignored
│   └── processed/
│       ├── .gitkeep
│       └── revision/
│           ├── final_chloride_labeled.parquet    # ship OR document download
│           ├── final_onset_summary.csv
│           └── series_split.csv
├── outputs/
│   ├── checkpoints/               # gitignored; one release bundle documented
│   ├── paper/                     # curated manuscript artifacts
│   │   ├── figures/
│   │   └── tables/
│   └── revision/                  # optional: full experiment outputs (gitignored)
├── docs/
│   ├── repository_audit.md        # this document
│   ├── REPRODUCIBILITY.md         # step-by-step, seeds, hardware, expected metrics
│   ├── parameter_configuration.md # from final_locked_parameter_configuration
│   └── results/                   # final_* reports
└── archive/                       # not required for reproduction
    ├── legacy/                    # former src/legacy/
    ├── scripts/                   # superseded 06, 08, 10, 11, 13, 15, 16
    └── audit/                     # internal reviewer working notes
```

**Large artifact strategy (choose one):**

1. **Git LFS** for `final_chloride_labeled.parquet` (~7 MB) + 9 checkpoints (~25 MB) — fits GitHub LFS budget.
2. **Zenodo / OSF deposit** for full bundle (~80 MB curated) with download script in README.
3. **Regenerate-only** — ship code + `series_split.csv` + small validation subset; users run `12` and training locally (~hours on GPU).

Recommended: **(2) Zenodo for checkpoints + final data**, git for code and small CSV tables.

---

## 6. Cleanup Plan

### Phase 0 — Safety (before any deletion)

1. Create full backup or tag: `git tag pre-release-audit-2026-06-30`.
2. Verify `final_model_comparison.csv` metrics match paper tables.
3. Export `outputs/revision/environment.json` into `docs/REPRODUCIBILITY.md`.

### Phase 1 — Hygiene (immediate, low risk)

| Step | Action | Est. savings |
|---|---|---|
| 1.1 | Delete `.idea/`, all `__pycache__/`, root `checkpoints/` | ~3 MB |
| 1.2 | Delete `outputs/revision/logs/` | negligible |
| 1.3 | Delete `data/processed/revision/series_split_smoke.csv` | negligible |
| 1.4 | Remove absolute paths from `benchmark_checkpoints.csv` or delete file | — |

### Phase 2 — Deduplicate model artifacts

| Step | Action | Est. savings |
|---|---|---|
| 2.1 | Delete non-`final_` checkpoints (§2.7) | ~18 MB |
| 2.2 | Delete `tft/*_10epoch*` checkpoint folders | ~10 MB |
| 2.3 | Delete duplicate `final_tft` epoch variants (keep best per seed) | ~5 MB |
| 2.4 | Delete `pf_*.csv` and `tft_*_10epoch_points.csv` | ~17 MB |
| 2.5 | Delete `data/sim/chloride_long.csv` (keep parquet or regenerate) | ~117 MB |

### Phase 3 — Restructure source

| Step | Action |
|---|---|
| 3.1 | Move `src/legacy/` → `archive/legacy/` |
| 3.2 | Move superseded scripts → `archive/scripts/` |
| 3.3 | Move internal audit markdown → `archive/audit/` |
| 3.4 | Promote final reports → `docs/results/` |
| 3.5 | Curate figures/tables → `outputs/paper/` |

### Phase 4 — Git & release configuration

| Step | Action |
|---|---|
| 4.1 | Extend `.gitignore` (see §7) |
| 4.2 | Commit revision scripts 07–23 + shared modules |
| 4.3 | Decide tracked vs LFS vs Zenodo for data/checkpoints |
| 4.4 | Add `LICENSE`, `CITATION.cff` |
| 4.5 | Rewrite `README.md` with dual-pipeline clarity |
| 4.6 | Add `docs/REPRODUCIBILITY.md` with expected metrics from `final_model_comparison.csv` |

### Phase 5 — Validation

| Step | Command / check |
|---|---|
| 5.1 | `python scripts/12_generate_final_revision_data.py --n-series 10` (smoke) |
| 5.2 | Load `final_tft_seed20250111` checkpoint → run `09_mc_dropout_uq.py` inference-only |
| 5.3 | `python scripts/22_plot_mc_dropout_revised_figure.py` regenerates figure from CSV |
| 5.4 | Compare output metrics to `final_model_comparison.csv` tolerances |
| 5.5 | Fresh clone + install test on clean machine |

---

## 7. Recommended `.gitignore` Additions

```gitignore
# IDE & caches (ensure present)
.idea/
**/__pycache__/
*.pyc

# Stray training output
/checkpoints/

# Generated simulation (already partial)
data/sim/*.csv
data/sim/*.parquet

# All checkpoints by default
outputs/checkpoints/
outputs/revision/checkpoints/
*.ckpt
*.pt
*.joblib

# Large/intermediate predictions
outputs/revision/predictions/tft_*_10epoch_points.csv
outputs/revision/predictions/mc_dropout_seed*_100pass_level_population.csv
outputs/revision/logs/

# Optional: ignore full revision tree (ship via Zenodo)
# outputs/revision/
```

**Ship via release asset or LFS:** `final_chloride_labeled.parquet`, `final_*` checkpoints, `outputs/paper/`.

---

## 8. Reproducibility Checklist for Public Release

### 8.1 Minimum viable release (code + tables)

- [ ] All **K** scripts committed
- [ ] `revision_config.py` documents locked parameters
- [ ] `data/processed/revision/series_split.csv` committed
- [ ] `outputs/paper/tables/final_*.csv` committed
- [ ] `docs/REPRODUCIBILITY.md` with seeds `20250111/12/13`, hardware, expected MAE
- [ ] `LICENSE` + `CITATION.cff`

### 8.2 Full reproduction release

- [ ] Above + `final_chloride_labeled.parquet`
- [ ] Above + 9 final checkpoints (3 TFT + 3 MLP + 3 GRU + windowed LR)
- [ ] `run_revision_pipeline.py` with `--skip-training` support
- [ ] Zenodo DOI linked in README

### 8.3 Expected key results (verification targets)

From `outputs/revision/tables/final_model_comparison.csv` (test split, cumulative `onset_flag`):

| Model | MAE (mean) | Notes |
|---|---:|---|
| GRU | 0.00193 | Best pointwise accuracy |
| TFT | 0.00454 | Primary method |
| MLP | 0.00697 | Windowed sequence baseline |
| Logistic Regression | 0.02065 | Pointwise tabular baseline |

Population reference Pf at 60 years (locked data): **0.310** (`phase_1_4_status.md`).

---

## 9. Complete File Inventory (255 files)

### 9.1 Tracked files (34) — all **Keep** unless noted

| Path | Size | Action |
|---|---:|---|
| `.gitignore` | 0.2 KB | K |
| `README.md` | 3.4 KB | K (rewrite) |
| `requirements.txt` | 0.8 KB | K |
| `run_pipeline.py` | 1.7 KB | K |
| `scripts/01_generate_data.py` | 4.6 KB | K |
| `scripts/02_label_onset.py` | 4.6 KB | K |
| `scripts/03_train_model.py` | 5.4 KB | K |
| `scripts/04_infer.py` | 7.2 KB | K |
| `scripts/05_make_figures.py` | 7.9 KB | K |
| `scripts/06_baseline_comparison.py` | 9.0 KB | A |
| `data/processed/.gitkeep` | 0 | K |
| `data/processed/chloride_labeled.parquet` | 7393 KB | K/A |
| `data/processed/onset_summary.csv` | 133 KB | K/A |
| `data/raw/.gitkeep` | 0 | K |
| `src/legacy/*` (19 files) | ~90 KB total | A (entire dir) |

### 9.2 Untracked scripts (22) — see §4.2–4.3

### 9.3 Untracked data (8 files)

| Path | Size | Action |
|---|---:|---|
| `data/sim/.gitkeep` | 0 | K |
| `data/sim/chloride_long.parquet` | 7249 KB | D |
| `data/sim/chloride_long.csv` | 117041 KB | D |
| `data/processed/revision/chloride_labeled_revision.parquet` | 7670 KB | A |
| `data/processed/revision/final_chloride_labeled.parquet` | 7049 KB | K |
| `data/processed/revision/final_onset_summary.csv` | 114 KB | K |
| `data/processed/revision/series_split.csv` | 11 KB | K |
| `data/processed/revision/series_split_smoke.csv` | 0.4 KB | D |

### 9.4 Untracked checkpoints (43 files, ~36 MB)

See §2.7 and §4.7–4.8. **Keep 10 files; delete 33.**

### 9.5 Untracked predictions (36 files, ~21 MB)

See §2.8 and §4.9–4.10. **Keep 13; delete 23.**

### 9.6 Untracked tables (37 files)

See §4.11–4.12. **Keep 20; delete 9.**

### 9.7 Untracked figures (27 + 3 manuscript)

See §2.9 and §4.13–4.14. **Keep ~12; archive/delete ~18.**

### 9.8 Untracked reports (46 markdown files)

See §2.10 and §4.16–4.17. **Keep 10; archive ~36.**

### 9.9 Untracked logs (14 files)

All **Delete** (§4.15).

### 9.10 Other untracked

| Path | Action |
|---|---|
| `checkpoints/epoch=0-step=*.ckpt` | D |
| `outputs/revision/environment.json` | K → merge into `docs/REPRODUCIBILITY.md` |
| `outputs/revision_experiment_audit_v2.md` | A (stale) |
| `docs/repository_audit.md` | K |

---

## 10. Summary Statistics After Recommended Cleanup

| Metric | Current | After cleanup (est.) |
|---|---:|---:|
| Total files | 255 | ~90–110 |
| Disk size | ~206 MB | ~45–65 MB (without Zenodo offload) |
| With `data/sim` CSV removed | — | **~89 MB** |
| Git-tracked files | 34 | ~60–80 (code + tables + docs) |
| Checkpoint files | 45 | 10 |
| Prediction CSVs | 36 | 13 |

---

## 11. Priority Actions (Ordered)

1. **Document authoritative pipeline** — revision `12→17` is paper source of truth; update README.
2. **Delete hygiene artifacts** — `.idea/`, root `checkpoints/`, `__pycache__/`, logs.
3. **Deduplicate checkpoints and predictions** — keep only `final_*` canonical set.
4. **Archive `src/legacy/` and superseded scripts** — reduce contributor confusion.
5. **Curate `outputs/paper/`** — figures/tables that match manuscript numbering.
6. **Add LICENSE, CITATION.cff, REPRODUCIBILITY.md**.
7. **Commit revision scripts** — currently invisible to git clone users.
8. **Publish large artifacts on Zenodo** — link DOI in README.
9. **Run Phase 5 validation** before tagging `v1.0.0`.

---

*End of audit. No repository files were modified during this analysis except creation of `docs/repository_audit.md`.*
