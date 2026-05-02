from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    in_csv = Path("outputs/predictions/pf_full_true_vs_pred.csv")
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing: {in_csv}")

    df = pd.read_csv(in_csv).sort_values("t_year")

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_png = out_dir / "pf_full_true_vs_pred.png"
    fig_pdf = out_dir / "pf_full_true_vs_pred.pdf"

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(df["t_year"], df["Pf_true"] * 100.0, label="Pf (True, from simulation)")
    plt.plot(df["t_year"], df["Pf_pred"] * 100.0, label="Pf (Pred, TFT classifier)")
    plt.xlabel("Time (years)")
    plt.ylabel("Probability of corrosion initiation, Pf (%)")
    plt.title("Pf(t): True vs TFT Predicted (0–60 years)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()

    print("✅ Saved:", fig_png)
    print("✅ Saved:", fig_pdf)


if __name__ == "__main__":
    main()
