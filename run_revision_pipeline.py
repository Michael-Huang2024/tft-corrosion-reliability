"""
Run the authoritative revision/paper reproducibility pipeline.

This orchestrates scripts/12–23 for the locked Candidate C dataset, benchmark
comparison, TFT three-seed training, uncertainty quantification, Sobol
sensitivity, computational efficiency, and curated paper artifacts.

Use scripts/01–05 and run_pipeline.py only for the lightweight demo workflow.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAPER_FIGURES = ROOT / "outputs" / "paper" / "figures"
PAPER_TABLES = ROOT / "outputs" / "paper" / "tables"
REVISION_FIGURES = ROOT / "outputs" / "revision" / "figures"
REVISION_TABLES = ROOT / "outputs" / "revision" / "tables"

PAPER_FIGURE_NAMES = [
    "final_model_error_comparison.png",
    "final_population_trajectories_by_model.png",
    "final_tft_three_seed_trajectories.png",
    "final_tft_time_dependent_error_seed20250111.png",
    "final_tft_seed_metric_variability.png",
    "mc_dropout_uncertainty_band_revised.png",
    "mc_dropout_mean_convergence_20_50_100.png",
    "mc_dropout_std_convergence_20_50_100.png",
    "mc_dropout_convergence_difference_50_vs_100.png",
    "computational_efficiency_accuracy_vs_runtime.png",
    "computational_efficiency_total_runtime.png",
    "computational_efficiency_inference_only.png",
    "sobol_s1_margin_20_40_60.png",
    "sobol_st_margin_20_40_60.png",
    "sobol_time_evolution.png",
    "parameter_candidate_pf_curves.png",
    "parameter_candidate_cover_groups.png",
    "Fig3_pf_by_cover_depth.png",
    "Fig3_pf_by_cover_depth.pdf",
]

PAPER_TABLE_GLOBS = ["final_*", "parameter_candidate_*"]


class PipelineError(RuntimeError):
    pass


def run_step(name: str, script: str, extra_args: list[str] | None = None, required: bool = True) -> None:
    extra_args = extra_args or []
    command = [sys.executable, str(ROOT / "scripts" / script), *extra_args]
    print(f"\n{'=' * 72}\nSTEP: {name}\nRunning: {' '.join(command)}\n{'=' * 72}")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        message = f"Step failed ({result.returncode}): {name} [{script}]"
        if required:
            raise PipelineError(message)
        print(f"WARNING: {message} — continuing.")


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def sync_paper_artifacts() -> None:
    """Copy regenerated revision outputs into outputs/paper/ for release."""
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in PAPER_FIGURE_NAMES:
        if copy_if_exists(REVISION_FIGURES / name, PAPER_FIGURES / name):
            copied += 1
    fig3_src = ROOT / "outputs" / "figures"
    for name in ("Fig3_pf_by_cover_depth.png", "Fig3_pf_by_cover_depth.pdf"):
        if copy_if_exists(fig3_src / name, PAPER_FIGURES / name):
            copied += 1
    if REVISION_TABLES.exists():
        for path in REVISION_TABLES.iterdir():
            if not path.is_file():
                continue
            if path.name.startswith("final_") or path.name.startswith("parameter_candidate_"):
                if copy_if_exists(path, PAPER_TABLES / path.name):
                    copied += 1
    print(f"\nSynced {copied} artifact(s) to outputs/paper/.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the authoritative revision/paper pipeline.")
    parser.add_argument("--skip-data-generation", action="store_true", help="Use existing locked revision data.")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training; use existing checkpoints.")
    parser.add_argument("--skip-uq", action="store_true", help="Skip MC Dropout UQ and convergence scripts.")
    parser.add_argument("--skip-sobol", action="store_true", help="Skip Sobol sensitivity analysis.")
    parser.add_argument("--skip-efficiency", action="store_true", help="Skip computational efficiency analysis.")
    parser.add_argument("--n-series", type=int, default=1000, help="Series count for data generation.")
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available).")
    parser.add_argument("--no-sync-paper", action="store_true", help="Do not copy outputs to outputs/paper/.")
    return parser.parse_args()


def device_args(args: argparse.Namespace) -> list[str]:
    if args.device:
        return ["--device", args.device]
    return []


def main() -> None:
    args = parse_args()
    print("Revision/paper pipeline (authoritative)")
    print(f"  skip-data-generation: {args.skip_data_generation}")
    print(f"  skip-training:        {args.skip_training}")
    print(f"  skip-uq:              {args.skip_uq}")
    print(f"  skip-sobol:           {args.skip_sobol}")
    print(f"  skip-efficiency:      {args.skip_efficiency}")

    try:
        if not args.skip_data_generation:
            run_step(
                "Generate locked revision dataset",
                "12_generate_final_revision_data.py",
                ["--n-series", str(args.n_series)],
            )
        else:
            print("\nSKIP: data generation (using existing data/processed/revision/final_*)")

        if not args.skip_training:
            run_step(
                "Train logistic / MLP / GRU benchmarks",
                "07_train_benchmarks.py",
                ["--models", "logistic", "mlp", "gru", *device_args(args)],
            )
            run_step("Train windowed logistic baseline", "23_windowed_logistic_baseline.py", device_args(args))
            run_step(
                "TFT three-seed train/evaluate and rebuild comparison",
                "17_tft_three_seed_benchmark.py",
                ["run-remaining"],
            )
        else:
            print("\nSKIP: training — rebuilding benchmark tables from saved predictions/checkpoints")
            run_step(
                "Rebuild full benchmark comparison",
                "17_tft_three_seed_benchmark.py",
                ["rebuild-benchmark"],
                required=False,
            )
            run_step(
                "Build TFT three-seed summary",
                "17_tft_three_seed_benchmark.py",
                ["build-tft-summary"],
                required=False,
            )

        if not args.skip_uq:
            run_step("MC Dropout UQ (50 passes)", "09_mc_dropout_uq.py", device_args(args))
            run_step("MC Dropout convergence (50 vs 100)", "18_mc_dropout_convergence.py", device_args(args))
        else:
            print("\nSKIP: MC Dropout UQ and convergence")

        if not args.skip_sobol:
            run_step("Final Sobol sensitivity", "19_final_sobol_sensitivity.py")
        else:
            print("\nSKIP: Sobol sensitivity")

        if not args.skip_efficiency:
            run_step(
                "Computational efficiency analysis",
                "20_final_computational_efficiency.py",
                device_args(args),
            )
        else:
            print("\nSKIP: computational efficiency")

        run_step("Table 5 and representative TFT error figure", "21_representative_tft_table5.py", required=False)
        run_step("MC Dropout revised figure", "22_plot_mc_dropout_revised_figure.py", required=False)
        run_step("Figure 3 (cover depth)", "generate_fig3_revision.py", required=False)

        if not args.no_sync_paper:
            sync_paper_artifacts()

        print("\n" + "=" * 72)
        print("Revision pipeline completed successfully.")
        print("Curated artifacts: outputs/paper/figures/, outputs/paper/tables/")
        print("Full outputs:      outputs/revision/")
        print("=" * 72)
    except PipelineError as exc:
        print(f"\nPIPELINE ABORTED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print("\nPIPELINE INTERRUPTED by user.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
