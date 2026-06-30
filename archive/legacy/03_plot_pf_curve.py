"""
03_plot_pf_curve.py

Plot corrosion initiation probability curve:
    Pf(t) = P(t_init <= t)

This corresponds directly to Fig.4-type results in the literature.

Input:
- data/processed/onset_summary.csv

Output:
- outputs/reports/pf_curve.png
- outputs/reports/pf_curve.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_pf_curve(t_init: pd.Series, time_grid: np.ndarray) -> pd.DataFrame:
    """
    Empirical Pf(t):
    Pf(t) = fraction of series with t_init <= t

    NaN t_init means "not initiated yet" (right-censored),
    so they are simply counted as not failed.
    """
    pf = []
    total = len(t_init)

    for t in time_grid:
        pf_t = np.sum(t_init <= t) / total
        pf.append(pf_t)

    return pd.DataFrame({
        "t_year": time_grid,
        "Pf": pf
    })


def main():
    in_path = Path("data/processed/onset_summary.csv")
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # t_init_year: NaN means "no initiation within observation window"
    t_init = df["t_init_year"]

    # Define plotting time grid (0 to max simulated year)
    t_max = np.nanmax(df["t_init_year"].values)
    if np.isnan(t_max):
        t_max = 60.0  # fallback
    time_grid = np.linspace(0.0, t_max, 300)

    pf_df = compute_pf_curve(t_init, time_grid)

    # Save Pf curve data
    pf_csv = out_dir / "pf_curve.csv"
    pf_df.to_csv(pf_csv, index=False)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(pf_df["t_year"], pf_df["Pf"], lw=2)
    plt.xlabel("Time (years)")
    plt.ylabel("Probability of corrosion initiation, $P_f(t)$")
    plt.title("Corrosion initiation probability curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    pf_png = out_dir / "pf_curve.png"
    plt.savefig(pf_png, dpi=300)
    plt.close()

    print("✅ Pf(t) curve saved:")
    print(" -", pf_png)
    print(" -", pf_csv)

    # Key checkpoints (useful for paper discussion)
    for yr in [5, 10, 20, 30, 40, 60]:
        pf_val = np.sum(t_init <= yr) / len(t_init)
        print(f"Pf({yr:>2} years) = {pf_val*100:.2f}%")

    # Final value should match onset rate
    final_pf = pf_df["Pf"].iloc[-1]
    print(f"\nFinal Pf({t_max:.1f} years) = {final_pf*100:.2f}% (should match onset rate)")


if __name__ == "__main__":
    main()
