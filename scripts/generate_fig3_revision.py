"""Generate manuscript Figure 3 from revision TFT point predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT / "outputs/revision/predictions/tft_20250111_10epoch_points.csv"
COVER = ROOT / "data/processed/revision/final_onset_summary.csv"
OUT_DIR = ROOT / "outputs/figures"

COVER_BINS_MM = [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 110)]


def set_journal_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_point_data() -> pd.DataFrame:
    points = pd.read_csv(POINTS)
    cover = pd.read_csv(COVER)[["series_id", "cover_mm"]].drop_duplicates()
    df = points.merge(cover, on="series_id", how="left")
    df = df.rename(columns={"p_onset_pred": "p_onset1_pred"})
    return df.dropna(subset=["cover_mm", "t_year", "p_onset1_pred", "onset_flag"])


def make_fig3(df: pd.DataFrame, fig_dir: Path) -> list[str]:
    curves = []
    for lo, hi in COVER_BINS_MM:
        group = df[(df["cover_mm"] >= lo) & (df["cover_mm"] < hi)]
        if group.empty:
            print(f"Fig3: no rows for cover bin {lo}-{hi} mm; skipped.")
            continue
        curve = (
            group.groupby("t_year", as_index=False)
            .agg(Pf_true=("onset_flag", "mean"), Pf_pred=("p_onset1_pred", "mean"))
            .sort_values("t_year")
        )
        curve["bin"] = f"{lo}-{hi} mm"
        curves.append(curve)
        print(
            f"Fig3 bin {lo}-{hi} mm: "
            f"{group['series_id'].nunique()} series, {len(curve)} time points"
        )

    if not curves:
        raise RuntimeError("Fig3: no cover-depth curves generated.")

    set_journal_style()
    plt.figure(figsize=(6.6, 4.4))
    for curve in curves:
        label = curve["bin"].iloc[0]
        plt.plot(curve["t_year"], curve["Pf_true"], linewidth=2.0, label=f"{label} (simulator)")
        plt.plot(curve["t_year"], curve["Pf_pred"], linewidth=2.0, linestyle="--", label=f"{label} (TFT)")
    plt.xlabel("Time (years)")
    plt.ylabel("Corrosion initiation probability, P_f(t)")
    plt.title("Population-level P_f(t) stratified by concrete cover depth")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        fig_dir / "Fig3_pf_by_cover_depth.png",
        fig_dir / "Fig3_pf_by_cover_depth.pdf",
        fig_dir / "Fig3_pf_by_cover_depth_600dpi.png",
    ]
    plt.savefig(outputs[0], dpi=300)
    plt.savefig(outputs[1])
    plt.savefig(outputs[2], dpi=600)
    plt.close()
    return [str(p) for p in outputs]


def main() -> None:
    df = load_point_data()
    paths = make_fig3(df, OUT_DIR)
    for path in paths:
        print(f"Saved: {Path(path).name}")


if __name__ == "__main__":
    main()
