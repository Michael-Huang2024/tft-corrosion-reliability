"""Regenerate MC Dropout uncertainty figure from saved 50-pass CSV (no inference rerun)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from revision_config import REVISION_FIGURE_DIR, REVISION_OUTPUT_DIR, ensure_revision_dirs

INPUT_CSV = Path("outputs/revision/predictions/mc_dropout_population_predictions.csv")
OUT_300 = REVISION_FIGURE_DIR / "mc_dropout_uncertainty_band_revised.png"
OUT_600 = REVISION_FIGURE_DIR / "mc_dropout_uncertainty_band_revised_600dpi.png"
REPORT = REVISION_OUTPUT_DIR / "mc_dropout_figure_revision_report.md"

CAPTION_NOTE = (
    "The shaded interval represents an approximate MC Dropout-based epistemic uncertainty band "
    "and should not be interpreted as a calibrated field confidence interval."
)


def build_figure(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t = df["t_year"]
    ax.plot(t, df["Pf_true"], color="#222222", linewidth=2.0, label="Reference cumulative Pf(t)")
    ax.plot(t, df["predictive_mean"], color="#2c5282", linewidth=2.0, label="MC Dropout predictive mean")
    ax.fill_between(
        t,
        df["q025"],
        df["q975"],
        color="#6b8cae",
        alpha=0.32,
        linewidth=0,
        label="Approximate 95% predictive interval",
    )
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Cumulative corrosion initiation probability")
    ax.set_title("MC Dropout uncertainty estimate for corrosion initiation probability")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=9)
    fig.tight_layout()
    return fig


def main() -> None:
    ensure_revision_dirs()
    df = pd.read_csv(INPUT_CSV)
    required = {"t_year", "Pf_true", "predictive_mean", "q025", "q975"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing columns in {INPUT_CSV}: {missing}")

    df = df.sort_values("t_year").reset_index(drop=True)
    df["interval_width"] = df["q975"] - df["q025"]
    max_row = df.loc[df["interval_width"].idxmax()]

    fig = build_figure(df)
    fig.savefig(OUT_300, dpi=300)
    fig.savefig(OUT_600, dpi=600)
    plt.close(fig)

    report_lines = [
        "# MC Dropout Figure Revision Report",
        "",
        "## Output files",
        "",
        f"- `{OUT_300.as_posix()}` (300 dpi)",
        f"- `{OUT_600.as_posix()}` (600 dpi)",
        "",
        "Original figure `mc_dropout_uncertainty_band.png` was not overwritten.",
        "",
        "## Columns used",
        "",
        "- Reference: `Pf_true` vs `t_year`",
        "- Predictive mean: `predictive_mean`",
        "- Interval bounds: `q025`, `q975`",
        "",
        "## Caption note (for manuscript, not embedded in plot)",
        "",
        CAPTION_NOTE,
        "",
        "## Interval width check",
        "",
        f"- Year of maximum interval width: **{max_row['t_year']:.2f} years**",
        f"- Maximum interval width: **{max_row['interval_width']:.6f}**",
        f"- Predictive mean at that year: {max_row['predictive_mean']:.6f}",
        f"- Lower bound (q025): {max_row['q025']:.6f}",
        f"- Upper bound (q975): {max_row['q975']:.6f}",
        "",
    ]
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Saved: {OUT_300.as_posix()}")
    print(f"Saved: {OUT_600.as_posix()}")
    print(f"Report: {REPORT.as_posix()}")
    print(f"Max interval width: {max_row['interval_width']:.6f} at t={max_row['t_year']:.2f} y")


if __name__ == "__main__":
    main()
