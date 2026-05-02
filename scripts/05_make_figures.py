"""
Generate manuscript figures and summary tables from regenerated outputs.

This script implements Step 5: paper figure generation for population-level
corrosion initiation probability Pf(t).

Generated files:
- outputs/figures/Fig1_pf_true_vs_pred.png/.pdf
- outputs/figures/Fig2_pf_abs_error_vs_time.png/.pdf
- outputs/figures/Fig3_pf_by_cover_depth.png/.pdf
- outputs/figures/Fig4_efficiency_comparison.png/.pdf
- outputs/tables/Fig2_pf_error_table.csv
- outputs/tables/Fig4_efficiency_timing.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def set_journal_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_pf_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Pf(t) table: {path}. Run scripts/04_infer.py first.")
    df = pd.read_csv(path)
    required = {"t_year", "Pf_true", "Pf_pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Pf(t) table missing columns {missing}. Found: {list(df.columns)}")
    return df.sort_values("t_year").reset_index(drop=True)


def load_point_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing point prediction table: {path}. Run scripts/04_infer.py first.")
    df = pd.read_parquet(path)
    required = ["series_id", "time_idx", "t_year", "p_onset1_pred", "target_onset"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Point prediction table missing columns {missing}. Found: {list(df.columns)}")
    return df[required].dropna().sort_values(["series_id", "time_idx"]).reset_index(drop=True)


def make_fig1(pf: pd.DataFrame, fig_dir: Path) -> None:
    set_journal_style()
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(pf["t_year"], pf["Pf_true"], linewidth=2.0, label="Ground-truth Pf(t)")
    plt.plot(pf["t_year"], pf["Pf_pred"], linewidth=2.0, linestyle="--", label="TFT-predicted Pf(t)")
    plt.xlabel("Time (years)")
    plt.ylabel("Corrosion initiation probability, Pf(t)")
    plt.title("Population-level Pf(t): ground truth vs TFT prediction")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "Fig1_pf_true_vs_pred.png")
    plt.savefig(fig_dir / "Fig1_pf_true_vs_pred.pdf")
    plt.close()


def make_fig2(pf: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    df = pf.copy()
    df["abs_error"] = (df["Pf_pred"] - df["Pf_true"]).abs()
    mae = float(df["abs_error"].mean())
    rmse = float(np.sqrt(((df["Pf_pred"] - df["Pf_true"]) ** 2).mean()))
    df[["t_year", "Pf_true", "Pf_pred", "abs_error"]].to_csv(
        table_dir / "Fig2_pf_error_table.csv",
        index=False,
    )

    set_journal_style()
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(df["t_year"], df["abs_error"], linewidth=2.0)
    plt.xlabel("Time (years)")
    plt.ylabel("Absolute error |Pf_pred - Pf_true|")
    plt.title(f"Absolute error over time (MAE={mae:.6f}, RMSE={rmse:.6f})")
    plt.tight_layout()
    plt.savefig(fig_dir / "Fig2_pf_abs_error_vs_time.png")
    plt.savefig(fig_dir / "Fig2_pf_abs_error_vs_time.pdf")
    plt.close()


def make_fig3(point: pd.DataFrame, static_path: Path, fig_dir: Path) -> None:
    if not static_path.exists():
        raise FileNotFoundError(f"Missing series static table: {static_path}. Run scripts/04_infer.py first.")
    cover_map = pd.read_csv(static_path)
    if "cover_mm" not in cover_map.columns:
        raise ValueError(f"cover_mm not found in {static_path}")

    df = point.merge(cover_map[["series_id", "cover_mm"]].drop_duplicates(), on="series_id", how="left")
    df = df.dropna(subset=["cover_mm"])
    cover_bins_mm = [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 110)]

    curves = []
    for lo, hi in cover_bins_mm:
        group = df[(df["cover_mm"] >= lo) & (df["cover_mm"] < hi)]
        if group.empty:
            print(f"Fig3: no rows for cover bin {lo}-{hi} mm; skipped.")
            continue
        curve = (
            group.groupby("t_year", as_index=False)
            .agg(Pf_true=("target_onset", "mean"), Pf_pred=("p_onset1_pred", "mean"))
            .sort_values("t_year")
        )
        curve["bin"] = f"{lo}-{hi} mm"
        curves.append(curve)

    if not curves:
        raise RuntimeError("Fig3: no cover-depth curves generated. Check simulated cover range.")

    set_journal_style()
    plt.figure(figsize=(6.6, 4.4))
    for curve in curves:
        label = curve["bin"].iloc[0]
        plt.plot(curve["t_year"], curve["Pf_true"], linewidth=2.0, label=f"{label} (true)")
        plt.plot(curve["t_year"], curve["Pf_pred"], linewidth=2.0, linestyle="--", label=f"{label} (TFT)")
    plt.xlabel("Time (years)")
    plt.ylabel("Corrosion initiation probability, Pf(t)")
    plt.title("Population-level Pf(t) stratified by concrete cover depth")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / "Fig3_pf_by_cover_depth.png")
    plt.savefig(fig_dir / "Fig3_pf_by_cover_depth.pdf")
    plt.close()


def make_fig4(point: pd.DataFrame, fig_dir: Path, table_dir: Path) -> None:
    start = perf_counter()
    _ = point.groupby("t_year", as_index=False)["target_onset"].mean()
    traditional_seconds = perf_counter() - start

    start = perf_counter()
    _ = point.groupby("t_year", as_index=False)["p_onset1_pred"].mean()
    tft_seconds = perf_counter() - start

    timing = pd.DataFrame(
        {
            "method": ["Traditional Pf(t) post-processing", "TFT Pf(t) post-processing"],
            "seconds": [traditional_seconds, tft_seconds],
        }
    )
    timing.to_csv(table_dir / "Fig4_efficiency_timing.csv", index=False)

    set_journal_style()
    plt.figure(figsize=(6.2, 4.2))
    plt.bar(timing["method"], timing["seconds"])
    plt.yscale("log")
    plt.ylabel("Computation time (seconds, log scale)")
    plt.title("Computational efficiency comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "Fig4_efficiency_comparison.png")
    plt.savefig(fig_dir / "Fig4_efficiency_comparison.pdf")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript figures and tables.")
    parser.add_argument("--pf-table", type=Path, default=ROOT / "outputs" / "predictions" / "pf_full_true_vs_pred.csv")
    parser.add_argument(
        "--point-table",
        type=Path,
        default=ROOT / "outputs" / "predictions" / "onset_flag_pred_point.parquet",
    )
    parser.add_argument("--series-static", type=Path, default=ROOT / "outputs" / "predictions" / "series_static.csv")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "outputs" / "figures")
    parser.add_argument("--table-dir", type=Path, default=ROOT / "outputs" / "tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    pf = load_pf_table(args.pf_table)
    point = load_point_table(args.point_table)

    make_fig1(pf, args.figure_dir)
    make_fig2(pf, args.figure_dir, args.table_dir)
    make_fig3(point, args.series_static, args.figure_dir)
    make_fig4(point, args.figure_dir, args.table_dir)

    print(f"Saved manuscript figures under: {args.figure_dir}")
    print(f"Saved manuscript tables under: {args.table_dir}")


if __name__ == "__main__":
    main()
