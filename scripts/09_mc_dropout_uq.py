"""
MC Dropout-based approximate Bayesian inference for the locked TFT benchmark.

Loads the validation-selected TFT checkpoint once, keeps weights fixed, sets the
model globally to evaluation mode, reactivates dropout layers during stochastic
forward passes, and aggregates population-level cumulative Pf(t).
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from revision_config import (
    FINAL_LABELED_DATA,
    GROUP_COLUMN,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    REVISION_FIGURE_DIR,
    REVISION_OUTPUT_DIR,
    REVISION_PREDICTION_DIR,
    REVISION_TABLE_DIR,
    TARGET_COLUMN,
    TFT_STATIC_REALS,
    TFT_TIME_VARYING_KNOWN_REALS,
    TFT_TIME_VARYING_UNKNOWN_REALS,
    TIME_COLUMN,
    TIME_INDEX_COLUMN,
    ensure_revision_dirs,
)
from revision_data import attach_split, load_or_create_series_split, validate_series_split

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "revision"
    / "checkpoints"
    / "tft"
    / "20250111_10epoch"
    / "best.ckpt"
)
SELECTED_SEED = 20250111
SELECTED_BEST_VAL_LOSS = 0.007161
EXPECTED_EVAL_POINTS = 731
INFER_BATCH = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal MC Dropout UQ for validation-selected TFT.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--passes", type=int, default=50)
    parser.add_argument("--convergence-passes", nargs="+", type=int, default=[20, 50])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-series", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path.resolve()


def enable_dropout_only(model: nn.Module) -> None:
    """Keep normalization in eval mode; retain dropout during stochastic prediction."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def prepare_test_loader(smoke_series: int | None = None):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import NaNLabelEncoder

    df = pd.read_parquet(FINAL_LABELED_DATA)
    split = load_or_create_series_split(df)
    validate_series_split(df, split)
    df = attach_split(df, split)
    if smoke_series is not None:
        train_series = sorted(df.loc[df["split"] == "train", GROUP_COLUMN].unique())[: max(smoke_series, 2)]
        test_series = sorted(df.loc[df["split"] == "test", GROUP_COLUMN].unique())[: max(smoke_series, 2)]
        keep = set(train_series) | set(test_series)
        df = df[df[GROUP_COLUMN].isin(keep)].copy()
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    n_test_series = int(test_df[GROUP_COLUMN].nunique())
    test_df_sorted = test_df.sort_values([GROUP_COLUMN, TIME_INDEX_COLUMN]).reset_index(drop=True)
    categorical_encoders = {f"__group_id__{GROUP_COLUMN}": NaNLabelEncoder(add_nan=True)}
    training = TimeSeriesDataSet(
        train_df,
        time_idx=TIME_INDEX_COLUMN,
        target=TARGET_COLUMN,
        group_ids=[GROUP_COLUMN],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_known_reals=TFT_TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TFT_TIME_VARYING_UNKNOWN_REALS,
        static_reals=TFT_STATIC_REALS,
        categorical_encoders=categorical_encoders,
        add_relative_time_idx=True,
        add_encoder_length=True,
    )
    testing = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)
    loader = testing.to_dataloader(train=False, batch_size=INFER_BATCH, num_workers=0)
    target_map = test_df[[GROUP_COLUMN, TIME_INDEX_COLUMN, TIME_COLUMN, TARGET_COLUMN]].drop_duplicates()
    return testing, loader, target_map, test_df_sorted, n_test_series


def one_population_pass(
    model,
    dataset,
    loader,
    target_map: pd.DataFrame,
    test_df_sorted: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    sums: dict[tuple[int, int], float] = defaultdict(float)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    sample_index = dataset.index.reset_index(drop=True)
    sample_offset = 0
    with torch.no_grad():
        for batch in loader:
            x, _ = batch
            x_dev = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in x.items()}
            logits = model(x_dev)["prediction"]
            probs = torch.softmax(logits, dim=-1)[..., 1].detach().cpu().numpy()
            if not np.isfinite(probs).all():
                raise ValueError("Non-finite MC Dropout predictions")
            batch_size = probs.shape[0]
            row_starts = sample_index.iloc[sample_offset : sample_offset + batch_size]["index_start"].to_numpy(dtype=int)
            series_ids = test_df_sorted.iloc[row_starts][GROUP_COLUMN].to_numpy(dtype=int)
            sample_offset += batch_size
            decoder_time_idxs = x["decoder_time_idx"].detach().cpu().numpy().astype(int)
            for b, sid in enumerate(series_ids):
                for j in range(probs.shape[1]):
                    key = (int(sid), int(decoder_time_idxs[b, j]))
                    sums[key] += float(probs[b, j])
                    counts[key] += 1
    point = pd.DataFrame(
        {
            GROUP_COLUMN: [key[0] for key in sums],
            TIME_INDEX_COLUMN: [key[1] for key in sums],
            "p_onset_pred": [sums[key] / counts[key] for key in sums],
        }
    )
    point = point.merge(target_map, on=[GROUP_COLUMN, TIME_INDEX_COLUMN], how="left").dropna()
    return (
        point.groupby(TIME_COLUMN, as_index=False)
        .agg(Pf_true=(TARGET_COLUMN, "mean"), Pf_pred=("p_onset_pred", "mean"))
        .sort_values(TIME_COLUMN)
        .reset_index(drop=True)
    )


def summarize_passes(passes: list[pd.DataFrame]) -> pd.DataFrame:
    base = passes[0][["t_year", "Pf_true"]].copy()
    matrix = np.vstack([pf["Pf_pred"].to_numpy(dtype=float) for pf in passes])
    base["predictive_mean"] = matrix.mean(axis=0)
    base["predictive_std"] = matrix.std(axis=0, ddof=1) if len(passes) > 1 else 0.0
    base["q025"] = np.quantile(matrix, 0.025, axis=0)
    base["q975"] = np.quantile(matrix, 0.975, axis=0)
    if not np.isfinite(base.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Non-finite MC Dropout summary statistics")
    return base


def write_outputs(
    summary: pd.DataFrame,
    convergence: pd.DataFrame,
    *,
    checkpoint: Path,
    n_passes: int,
    n_test_series: int,
    total_seconds: float,
    seconds_per_pass: float,
) -> None:
    summary.to_csv(REVISION_PREDICTION_DIR / "mc_dropout_population_predictions.csv", index=False)
    error = summary["predictive_mean"] - summary["Pf_true"]
    in_interval = (summary["Pf_true"] >= summary["q025"]) & (summary["Pf_true"] <= summary["q975"])
    metrics = pd.DataFrame(
        [
            {
                "selected_seed": SELECTED_SEED,
                "checkpoint": str(checkpoint),
                "selection_criterion": "best_validation_loss",
                "best_validation_loss": SELECTED_BEST_VAL_LOSS,
                "stochastic_passes": n_passes,
                "test_series_count": n_test_series,
                "evaluation_time_points": int(len(summary)),
                "evaluation_start_year": float(summary["t_year"].iloc[0]),
                "evaluation_end_year": float(summary["t_year"].iloc[-1]),
                "MAE_predictive_mean": float(error.abs().mean()),
                "RMSE_predictive_mean": float(np.sqrt((error**2).mean())),
                "mean_predictive_std": float(summary["predictive_std"].mean()),
                "max_predictive_std": float(summary["predictive_std"].max()),
                "PICP_95": float(in_interval.mean()),
                "MPIW_95": float((summary["q975"] - summary["q025"]).mean()),
                "total_mc_seconds": round(total_seconds, 1),
                "seconds_per_pass": round(seconds_per_pass, 2),
            }
        ]
    )
    metrics.to_csv(REVISION_TABLE_DIR / "mc_dropout_metrics.csv", index=False)
    convergence.to_csv(REVISION_TABLE_DIR / "mc_dropout_convergence.csv", index=False)

    deterministic_path = REVISION_PREDICTION_DIR / "tft_20250111_10epoch.csv"
    deterministic_note = ""
    if deterministic_path.exists():
        det = pd.read_csv(deterministic_path).sort_values("t_year").reset_index(drop=True)
        merged = summary.merge(det[["t_year", "Pf_pred"]], on="t_year", how="inner", suffixes=("", "_det"))
        if len(merged) == len(summary):
            mean_diff = float((merged["predictive_mean"] - merged["Pf_pred"]).abs().mean())
            deterministic_note = (
                f"Mean absolute difference between MC Dropout predictive mean and deterministic "
                f"TFT Pf_pred on the same 731 points: {mean_diff:.6f}."
            )

    plt.figure(figsize=(7, 4))
    plt.plot(summary["t_year"], summary["Pf_true"], label="Reference cumulative Pf(t)")
    plt.plot(summary["t_year"], summary["predictive_mean"], label="MC Dropout predictive mean")
    plt.fill_between(summary["t_year"], summary["q025"], summary["q975"], alpha=0.25, label="95% predictive interval")
    plt.xlabel("Time (years)")
    plt.ylabel("Cumulative corrosion initiation probability")
    plt.title("MC Dropout-based approximate posterior predictive distribution")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "mc_dropout_uncertainty_band.png", dpi=300)
    plt.close()

    lines = [
        "# MC Dropout Uncertainty Quantification Report",
        "",
        "## Method",
        "",
        "MC Dropout-based approximate Bayesian inference on the locked TFT benchmark.",
        "",
        "The trained checkpoint is loaded once with fixed weights. The model is set globally",
        "to evaluation mode (`model.eval()`), while dropout layers remain active during",
        "stochastic forward passes. Fifty independent passes approximate the approximate",
        "posterior predictive distribution and provide an epistemic uncertainty estimate.",
        "",
        "This analysis does **not** claim exact Bayesian inference, a full Bayesian posterior,",
        "or a rigorous Bayesian TFT.",
        "",
        "## Model selection",
        "",
        f"- Selected seed: **{SELECTED_SEED}** (by best validation loss, not test MAE)",
        f"- Best validation loss: **{SELECTED_BEST_VAL_LOSS:.6f}**",
        f"- Checkpoint: `{checkpoint.as_posix()}`",
        "",
        "## Evaluation scope",
        "",
        f"- Split: independent test only",
        f"- Test series: {n_test_series}",
        f"- Population evaluation time points: {len(summary)}",
        f"- Time range: {summary['t_year'].iloc[0]:.2f}–{summary['t_year'].iloc[-1]:.2f} years",
        f"- Target: cumulative corrosion-initiation `onset_flag`",
        f"- Stochastic forward passes: {n_passes}",
        "",
        "## Predictive summaries",
        "",
        "Per time point:",
        "- `predictive_mean`: mean across stochastic passes",
        "- `predictive_std`: standard deviation across stochastic passes (epistemic uncertainty estimate)",
        "- `q025`, `q975`: 95% predictive interval bounds",
        "",
        "Saved predictions: `outputs/revision/predictions/mc_dropout_population_predictions.csv`",
        "",
        "## Metrics",
        "",
        metrics.to_string(index=False),
        "",
    ]
    if deterministic_note:
        lines.extend(["## Deterministic comparison", "", deterministic_note, ""])
    lines.extend(
        [
            "## Convergence",
            "",
            convergence.to_string(index=False),
            "",
            f"Total MC Dropout runtime: {total_seconds:.1f} s ({seconds_per_pass:.2f} s per pass).",
        ]
    )
    (REVISION_OUTPUT_DIR / "mc_dropout_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from pytorch_forecasting import TemporalFusionTransformer

    args = parse_args()
    ensure_revision_dirs()
    checkpoint = resolve_checkpoint(args.checkpoint)
    smoke_n = args.smoke_series if args.smoke_test else None
    dataset, loader, target_map, test_df_sorted, n_test_series = prepare_test_loader(smoke_n)
    device = torch.device(args.device)

    model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint), weights_only=False)
    model.to(device)
    enable_dropout_only(model)

    passes = args.passes
    convergence_levels = sorted(set(args.convergence_passes + [passes]))
    if args.smoke_test:
        passes = min(passes, 5)
        convergence_levels = [2, passes]

    max_passes = max(convergence_levels)
    collected: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    for i in range(1, max_passes + 1):
        collected.append(one_population_pass(model, dataset, loader, target_map, test_df_sorted, device))
        if i in convergence_levels or i % 10 == 0 or i == max_passes:
            elapsed = time.perf_counter() - t0
            print(f"MC Dropout pass {i}/{max_passes} ({elapsed:.1f}s elapsed)", flush=True)

    total_seconds = time.perf_counter() - t0
    seconds_per_pass = total_seconds / max_passes

    summaries = {n: summarize_passes(collected[:n]) for n in convergence_levels}
    final_summary = summaries[passes]
    if len(final_summary) != EXPECTED_EVAL_POINTS and not args.smoke_test:
        raise ValueError(f"Expected {EXPECTED_EVAL_POINTS} evaluation points, got {len(final_summary)}")

    convergence_rows = []
    if 20 in summaries and 50 in summaries:
        diff_mean = np.abs(summaries[50]["predictive_mean"] - summaries[20]["predictive_mean"]).mean()
        diff_std = np.abs(summaries[50]["predictive_std"] - summaries[20]["predictive_std"]).mean()
        convergence_rows.append(
            {
                "comparison": "20_vs_50",
                "mean_abs_predictive_mean_change": diff_mean,
                "mean_abs_predictive_std_change": diff_std,
            }
        )
    for n, summary in summaries.items():
        convergence_rows.append(
            {
                "comparison": f"{n}_passes",
                "mean_predictive_std": float(summary["predictive_std"].mean()),
                "evaluation_time_points": int(len(summary)),
            }
        )
    convergence = pd.DataFrame(convergence_rows)

    write_outputs(
        final_summary,
        convergence,
        checkpoint=checkpoint,
        n_passes=passes,
        n_test_series=n_test_series,
        total_seconds=total_seconds,
        seconds_per_pass=seconds_per_pass,
    )
    print(f"Wrote outputs/revision/mc_dropout_report.md ({passes} passes, {len(final_summary)} time points)")


if __name__ == "__main__":
    main()
