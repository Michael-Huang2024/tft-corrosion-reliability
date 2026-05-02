# TFT Corrosion Reliability

This repository provides the reproducibility code for a physics-guided, simulation-trained surrogate modeling workflow for estimating population-level corrosion initiation probability Pf(t) in reinforced concrete. The implementation reproduces the manuscript pipeline for concept demonstration and does not add models, datasets, or evaluation metrics beyond the reported workflow.

## Key Modeling Principle

The underlying chloride concentration field C(x,t) is used only for label generation and is not provided as an input to the learning model, ensuring no information leakage.

## Environment Setup

Use Python 3.11 or a compatible Python 3.10+ environment.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If installing a GPU-specific PyTorch build, follow the official PyTorch installation command for the target CUDA version, then install the remaining requirements.

## Reproducibility Pipeline

Run all steps from the repository root:

```bash
python run_pipeline.py
```

The canonical manuscript workflow is:

```bash
python scripts/01_generate_data.py
python scripts/02_label_onset.py
python scripts/03_train_model.py
python scripts/04_infer.py
python scripts/05_make_figures.py
```

To use an existing checkpoint instead of retraining:

```bash
python run_pipeline.py --skip-training --checkpoint outputs/checkpoints/<checkpoint-file>.ckpt
```

## Pipeline Description

1. `scripts/01_generate_data.py` generates chloride diffusion simulation data.
2. `scripts/02_label_onset.py` labels corrosion initiation onset and produces processed training data.
3. `scripts/03_train_model.py` trains the TFT onset model used for Pf(t).
4. `scripts/04_infer.py` reconstructs population-level corrosion initiation probability Pf(t).
5. `scripts/05_make_figures.py` regenerates manuscript figures and summary tables.

## Expected Outputs

Generated data:

- `data/sim/chloride_long.parquet`
- `data/sim/chloride_long.csv`
- `data/processed/chloride_labeled.parquet`
- `data/processed/onset_summary.csv`

Model and prediction outputs:

- `outputs/checkpoints/tft_onset_flag-*.ckpt`
- `outputs/checkpoints/best_checkpoint.txt`
- `outputs/predictions/pf_full_true_vs_pred.csv`
- `outputs/predictions/onset_flag_pred_point.parquet`
- `outputs/predictions/series_static.csv`

Manuscript outputs:

- `outputs/figures/Fig1_pf_true_vs_pred.png`
- `outputs/figures/Fig2_pf_abs_error_vs_time.png`
- `outputs/figures/Fig3_pf_by_cover_depth.png`
- `outputs/figures/Fig4_efficiency_comparison.png`
- `outputs/tables/Fig2_pf_error_table.csv`
- `outputs/tables/Fig4_efficiency_timing.csv`

## Repository Structure

```text
tft-corrosion-reliability/
  README.md
  requirements.txt
  run_pipeline.py
  scripts/
    01_generate_data.py
    02_label_onset.py
    03_train_model.py
    04_infer.py
    05_make_figures.py
  data/
    raw/
    sim/
    processed/
  outputs/
    figures/
    predictions/
    tables/
  src/
    legacy/
```

`src/legacy/` contains earlier exploratory scripts retained only for auditability. The publication workflow uses only `run_pipeline.py` and the five scripts in `scripts/`.

## Notes on Reproducibility

- Data are generated via simulation.
- The pipeline fully reproduces the reported results.
- Results are valid within the sampled parameter space.
- This repository provides a concept demonstration, not field validation.
