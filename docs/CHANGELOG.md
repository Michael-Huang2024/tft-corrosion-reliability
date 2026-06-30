# Changelog

All notable changes to this research repository are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.0.0] — 2026-06-30

### Added

- **Public research release** for *Transformer-Based Sequence Learning for Multi-Horizon Corrosion Initiation Probability Prediction in Reinforced Concrete Bridges*.
- `run_revision_pipeline.py` — orchestrates the authoritative revision/paper workflow with `--skip-training`, `--skip-data-generation`, `--skip-uq`, `--skip-sobol`, and `--skip-efficiency` flags.
- `docs/REPRODUCIBILITY.md` — environment, seeds, expected metrics, and step-by-step reproduction for Tables 5–10 and Figures 1–4.
- `docs/PAPER_ARTIFACTS.md` — manuscript Figure/Table → file → script mapping.
- `docs/GITHUB_RELEASE_CHECKLIST.md` — commit, release, and validation checklist.
- `docs/release_validation_report.md` — dry-run validation results.
- `docs/results/` — twelve final scientific reports from the locked revision experiments.
- `outputs/paper/figures/` and `outputs/paper/tables/` — curated paper artifacts for GitHub release.
- `LICENSE` (MIT).
- `CITATION.cff` (v1.0.0, authors Yuzhong Huang and Wei Zheng).

### Changed

- **Revision pipeline cleanup** — repository reorganized per `docs/repository_audit.md` and `docs/repository_reorganization.md`.
- **README rewritten** — revision/paper pipeline documented as authoritative; `scripts/01–05` demoted to lightweight demo only.
- **`.gitignore` updated** — keeps `outputs/paper/` trackable; ignores checkpoints, logs, and `archive/diagnostics/`.

### Research content (reference release)

- **Final paper artifacts** — locked Candidate C dataset, benchmark tables, figures, and ten final model checkpoints.
- **Benchmark expansion** — pointwise logistic regression, windowed MLP, GRU, TFT (three seeds), and windowed multi-output logistic regression on cumulative `onset_flag`.
- **Sensitivity analysis** — final Saltelli Sobol study on five locked physical parameters (`scripts/19_final_sobol_sensitivity.py`).
- **MC Dropout uncertainty** — 50-pass formal UQ and 20/50/100-pass convergence analysis (`scripts/09`, `18`, `22`).
- **Computational efficiency reinterpretation** — unified runtime comparison; TFT not claimed as universally fastest (`scripts/20_final_computational_efficiency.py`, `docs/results/final_computational_efficiency_report.md`).

### Archived

- `src/legacy/` → `archive/legacy/` (19 exploratory scripts).
- Superseded revision scripts → `archive/scripts/` (`06`, `08`, `10`, `11`, `13`, `15`, `16`).
- Internal audit markdown → `archive/audit/`.
- Smoke/diagnostic outputs → `archive/diagnostics/`.

### Removed from active tree

- IDE config (`.idea/`), stray root `checkpoints/`, training logs, smoke-test split, regenerable `data/sim/*` copies.

---

## [Pre-release] — 2025-01 – 2026-06

### Development history (summary)

- Initial TFT corrosion Pf(t) pipeline with physics-guided simulation.
- Reviewer revision: corrected cumulative target `onset_flag`, locked parameter distributions, series-level splits, benchmark fairness audit.
- Phase 1–4 parameter screening and data lock (`Candidate C`).
- Full-scale benchmark training, three-seed TFT evaluation, Sobol and MC Dropout studies.
- Repository audit and Phase 1–3 cleanup for public release.

---

[1.0.0]: https://github.com/Michael-Huang2024/tft-corrosion-reliability/releases/tag/v1.0.0
