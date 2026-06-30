from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    in_path = Path("outputs/predictions/pf_true_vs_pred.csv")
    if not in_path.exists():
        raise FileNotFoundError(f"Missing: {in_path}")

    df = pd.read_csv(in_path).sort_values("t_year")

    # Convert to percentage for plotting
    df["Pf_true_pct"] = df["Pf_true"] * 100.0
    df["Pf_pred_pct"] = df["Pf_pred"] * 100.0

    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(df["t_year"], df["Pf_true_pct"], label="Pf (True, from simulation)")
    plt.plot(df["t_year"], df["Pf_pred_pct"], label="Pf (Pred, TFT classifier)")

    plt.xlabel("Time (years)")
    plt.ylabel("Probability of corrosion initiation, Pf (%)")
    plt.title("Pf(t): True vs TFT Predicted")
    plt.grid(True, alpha=0.3)
    plt.legend()

    png_path = out_dir / "pf_true_vs_pred.png"
    pdf_path = out_dir / "pf_true_vs_pred.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print("✅ Saved:", png_path)
    print("✅ Saved:", pdf_path)


if __name__ == "__main__":
    main()
