"""Dry-run validation for v1.0.0 public release. No training or data generation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
  "data/processed/revision/final_chloride_labeled.parquet",
  "data/processed/revision/series_split.csv",
  "data/processed/revision/final_onset_summary.csv",
  "outputs/paper/tables/final_model_comparison.csv",
  "outputs/paper/tables/final_sobol_indices_margin.csv",
  "outputs/paper/tables/final_computational_efficiency_summary.csv",
  "outputs/paper/tables/final_mc_dropout_metrics_20_50_100.csv",
  "outputs/paper/figures/Fig3_pf_by_cover_depth.pdf",
  "outputs/paper/figures/Fig3_pf_by_cover_depth.png",
]

REQUIRED_CHECKPOINTS = [
  "outputs/revision/checkpoints/final_tft_seed20250111-epoch=2-val_loss=0.0082.ckpt",
  "outputs/revision/checkpoints/final_tft_seed20250112-epoch=7-val_loss=0.0064.ckpt",
  "outputs/revision/checkpoints/final_tft_seed20250113-epoch=0-val_loss=0.0173.ckpt",
  "outputs/revision/checkpoints/final_mlp_seed20250111.pt",
  "outputs/revision/checkpoints/final_mlp_seed20250112.pt",
  "outputs/revision/checkpoints/final_mlp_seed20250113.pt",
  "outputs/revision/checkpoints/final_gru_seed20250111.pt",
  "outputs/revision/checkpoints/final_gru_seed20250112.pt",
  "outputs/revision/checkpoints/final_gru_seed20250113.pt",
  "outputs/revision/checkpoints/final_windowed_logistic_regression.joblib",
]

RELEASE_DOCS = [
  "README.md",
  "LICENSE",
  "CITATION.cff",
  "run_revision_pipeline.py",
  "docs/REPRODUCIBILITY.md",
  "docs/PAPER_ARTIFACTS.md",
  "docs/CHANGELOG.md",
  "docs/GITHUB_RELEASE_CHECKLIST.md",
]

WARNINGS = [
  (
    "archive/diagnostics/predictions/tft_20250111_10epoch_points.csv",
    "Figure 3 regeneration needs point-level predictions (archived, not in active tree)",
  ),
  (
    "outputs/revision/predictions/tft_20250111_10epoch_points.csv",
    "generate_fig3_revision.py expects this path; copy from archive before regenerating Fig 3",
  ),
]

EXPECTED_METRICS = {
  "GRU": (0.001934, 0.002931),
  "TFT": (0.004542, 0.006373),
  "MLP": (0.006975, 0.009964),
  "Windowed Logistic Regression": (0.020212, 0.023823),
  "Logistic Regression": (0.020652, 0.024271),
}


def check_files(paths: list[str]) -> tuple[list[str], list[str]]:
  passed, missing = [], []
  for rel in paths:
    if (ROOT / rel).exists():
      passed.append(rel)
    else:
      missing.append(rel)
  return passed, missing


def check_metrics() -> tuple[bool, str]:
  path = ROOT / "outputs/paper/tables/final_model_comparison.csv"
  if not path.exists():
    return False, "final_model_comparison.csv missing"
  import pandas as pd

  df = pd.read_csv(path)
  tol = 1e-4
  issues = []
  for model, (mae, rmse) in EXPECTED_METRICS.items():
    row = df[df["model"] == model]
    if row.empty:
      issues.append(f"{model}: not in table")
      continue
    mae_v = float(row["MAE_mean"].iloc[0])
    rmse_v = float(row["RMSE_mean"].iloc[0])
    if abs(mae_v - mae) > tol or abs(rmse_v - rmse) > tol:
      issues.append(f"{model}: MAE={mae_v:.6f} RMSE={rmse_v:.6f} (expected {mae}/{rmse})")
  if issues:
    return False, "; ".join(issues)
  return True, "All expected metrics match within tolerance"


def main() -> int:
  results = {"passed": [], "missing": [], "warnings": [], "metrics_ok": False, "metrics_msg": ""}

  for group_name, group in [
    ("release_docs", RELEASE_DOCS),
    ("required_data_and_paper_artifacts", REQUIRED_FILES),
    ("final_checkpoints", REQUIRED_CHECKPOINTS),
  ]:
    passed, missing = check_files(group)
    results["passed"].extend([f"[{group_name}] {p}" for p in passed])
    results["missing"].extend([f"[{group_name}] {m}" for m in missing])

  for path, msg in WARNINGS:
    if not (ROOT / path).exists():
      results["warnings"].append(f"{msg} (`{path}`)")

  results["metrics_ok"], results["metrics_msg"] = check_metrics()

  out = ROOT / "docs" / "release_validation_report.md"
  lines = [
    "# Release Validation Report",
    "",
    "**Mode:** dry-run (no training, no data regeneration)",
    "**Root:** repository root (relative paths only)",
    "",
    "## Summary",
    "",
    f"- Passed checks: **{len(results['passed'])}**",
    f"- Missing files: **{len(results['missing'])}**",
    f"- Warnings: **{len(results['warnings'])}**",
    f"- Metrics validation: **{'PASS' if results['metrics_ok'] else 'FAIL'}** — {results['metrics_msg']}",
    "",
  ]
  if results["missing"]:
    lines += ["## Missing files", ""] + [f"- `{m}`" for m in results["missing"]] + [""]
  else:
    lines += ["## Missing files", "", "None.", ""]

  lines += ["## Passed checks", ""] + [f"- `{p}`" for p in results["passed"]] + [""]

  if results["warnings"]:
    lines += ["## Warnings", ""] + [f"- {w}" for w in results["warnings"]] + [""]
  else:
    lines += ["## Warnings", "", "None.", ""]

  lines += [
    "## Exact next git commands",
    "",
    "```bash",
    "git status",
    "git add README.md LICENSE CITATION.cff requirements.txt .gitignore",
    "git add run_pipeline.py run_revision_pipeline.py scripts/",
    "git add docs/ outputs/paper/ data/processed/revision/",
    "git add outputs/revision/predictions/final_pf_*.csv outputs/revision/predictions/mc_dropout_*.csv",
    "git add archive/legacy/ archive/scripts/ archive/audit/",
    'git commit -m "Release v1.0.0: public research reproducibility package."',
    'git tag -a v1.0.0 -m "v1.0.0 public research release"',
    "# git push origin main && git push origin v1.0.0  # when ready",
    "```",
    "",
    "Upload `outputs/revision/checkpoints/final_*` to Zenodo or enable Git LFS before users need --skip-training.",
    "",
  ]
  out.write_text("\n".join(lines), encoding="utf-8")
  print(
      "Wrote docs/release_validation_report.md",
      f"passed={len(results['passed'])}",
      f"missing={len(results['missing'])}",
      f"warnings={len(results['warnings'])}",
  )

  ok = not results["missing"] and results["metrics_ok"]
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
