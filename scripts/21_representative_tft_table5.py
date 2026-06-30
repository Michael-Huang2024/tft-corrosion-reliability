"""Recompute Table 5 TFT accuracy and Fig. 4 error curve for representative seed 20250111."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from revision_config import REVISION_FIGURE_DIR, REVISION_OUTPUT_DIR, REVISION_TABLE_DIR, TIME_COLUMN, ensure_revision_dirs
from revision_metrics import evaluate_pf_curve

PREDICTION_FILE = Path("outputs/revision/predictions/final_pf_tft_seed20250111.csv")
TABLE5_FILE = REVISION_TABLE_DIR / "final_representative_tft_accuracy_table5.csv"
FIGURE_FILE = REVISION_FIGURE_DIR / "final_tft_time_dependent_error_seed20250111.png"
REPORT_FILE = REVISION_OUTPUT_DIR / "final_representative_tft_error_report.md"
CHECKPOINT = "outputs/revision/checkpoints/tft/20250111_10epoch/best.ckpt"


def main() -> None:
    ensure_revision_dirs()
    pf = pd.read_csv(PREDICTION_FILE)
    pf[TIME_COLUMN] = pf["t_year"]
    pf["abs_error"] = (pf["Pf_pred"] - pf["Pf_true"]).abs()

    metrics = evaluate_pf_curve(
        pf,
        model_name="TFT",
        seed=20250111,
        parameter_count_value=75992,
        training_time_seconds=None,
        inference_time_seconds=None,
    )

    table5 = pd.DataFrame(
        [
            {
                "table": "Table 5",
                "model": "TFT",
                "representative_seed": 20250111,
                "selection_criterion": "lowest_validation_loss",
                "checkpoint": CHECKPOINT,
                "test_series": 150,
                "evaluation_time_points": metrics["evaluation_time_points"],
                "evaluation_start_year": metrics["evaluation_start_year"],
                "evaluation_end_year": metrics["evaluation_end_year"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "max_abs_error": metrics["max_abs_error"],
                "year_of_max_error": metrics["year_of_max_error"],
                "final_year_abs_error": metrics["final_year_abs_error"],
            }
        ]
    )
    table5.to_csv(TABLE5_FILE, index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(pf["t_year"], pf["abs_error"], color="C0", linewidth=1.8)
    plt.xlabel("Time (years)")
    plt.ylabel(r"Absolute error  $e(t)=|P_f^{\mathrm{pred}}(t)-P_f^{\mathrm{true}}(t)|$")
    plt.title("TFT population-level absolute error (representative seed 20250111)")
    plt.xlim(0, 60)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300)
    plt.close()

    lines = [
        "# Representative TFT Population-Level Error Report",
        "",
        "## Scope",
        "",
        "- **Table 5** uses the **representative TFT seed 20250111** (selected by lowest validation loss, not test MAE).",
        "- **Table 10** reports **three-seed mean ± standard deviation** across seeds 20250111, 20250112, and 20250113; it is unchanged by this update.",
        f"- Predictions source: `{PREDICTION_FILE.as_posix()}`",
        "- Test set: 150 independent held-out series; 731 common population evaluation time points (~3.99–59.95 years).",
        "- No model retraining was performed.",
        "",
        "## Population trajectories",
        "",
        "Reference and predicted cumulative initiation probabilities:",
        "",
        "- `Pf_true(t)`: test-set mean of `onset_flag` at each evaluation year",
        "- `Pf_pred(t)`: TFT-predicted population mean at each evaluation year",
        "",
        "## Table 5 metrics (representative seed 20250111)",
        "",
        table5.to_string(index=False),
        "",
        "## Time-dependent absolute error",
        "",
        f"- Figure: `{FIGURE_FILE.as_posix()}`",
        f"- Maximum absolute error: {metrics['max_abs_error']:.6f} at year {metrics['year_of_max_error']:.2f}",
        f"- Final-year absolute error (~60 y): {metrics['final_year_abs_error']:.6f}",
        "",
        "## Note on superseded values",
        "",
        "Earlier draft values (MAE = 0.002188; RMSE = 0.002827) are **not** used here. "
        "The updated representative-seed metrics match the locked final benchmark evaluation on the common 731-point test horizon.",
        "",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(table5.to_string(index=False))
    print(f"\nSaved: {TABLE5_FILE.name}")
    print(f"Saved: {FIGURE_FILE.name}")
    print(f"Saved: {REPORT_FILE.name}")


if __name__ == "__main__":
    main()
