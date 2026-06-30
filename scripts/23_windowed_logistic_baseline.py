"""
Train and evaluate the Windowed Logistic Regression baseline.

Reuses the same WindowDataset, splits, overlap-averaged inference, and
population-level Pf(t) evaluation as the corrected Windowed MLP benchmark.
Distinct from the pointwise Logistic Regression baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from revision_config import (  # noqa: E402
    FINAL_LABELED_DATA,
    GROUP_COLUMN,
    INFERENCE_BATCH_SIZE,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    POINT_FEATURES,
    REVISION_CHECKPOINT_DIR,
    REVISION_PREDICTION_DIR,
    REVISION_TABLE_DIR,
    TARGET_COLUMN,
    TIME_INDEX_COLUMN,
    ensure_revision_dirs,
)
from revision_metrics import evaluate_point_predictions  # noqa: E402


MODEL_NAME = "Windowed Logistic Regression"
CHECKPOINT_NAME = "final_windowed_logistic_regression.joblib"
PREDICTION_NAME = "final_pf_windowed_logistic_regression.csv"
REPORT_NAME = "windowed_logistic_baseline_report.md"


def _load_benchmarks():
    path = SCRIPT_DIR / "07_train_benchmarks.py"
    spec = importlib.util.spec_from_file_location("train_benchmarks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bench = _load_benchmarks()
WindowDataset = bench.WindowDataset
fit_point_scaler = bench.fit_point_scaler
merge_existing_by_seed = bench.merge_existing_by_seed
prepare_data = bench.prepare_data
summarize_results = bench.summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Windowed Logistic Regression baseline.")
    parser.add_argument("--smoke-test", action="store_true", help="Use a small subset for a quick sanity check.")
    parser.add_argument("--smoke-series", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=INFERENCE_BATCH_SIZE)
    return parser.parse_args()


def windows_to_arrays(dataset: WindowDataset) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for idx in range(len(dataset)):
        enc, dec_target, *_meta = dataset[idx]
        xs.append(enc.numpy().reshape(-1))
        ys.append(dec_target.numpy())
    return np.vstack(xs).astype(np.float32), np.vstack(ys).astype(np.int8)


def count_windowed_logistic_params(model: Pipeline) -> int:
    multi = model.named_steps["classifier"]
    total = 0
    for estimator in multi.estimators_:
        total += int(np.size(estimator.coef_) + np.size(estimator.intercept_))
    return total


def predict_horizon_probabilities(model: Pipeline, x_flat: np.ndarray) -> np.ndarray:
    proba_list = model.predict_proba(x_flat)
    return np.column_stack([proba[:, 1] for proba in proba_list])


def overlap_average_test_predictions(
    model: Pipeline,
    test_ds: WindowDataset,
    df: pd.DataFrame,
    batch_size: int,
) -> tuple[pd.DataFrame, float]:
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    sums: dict[tuple[int, int], float] = defaultdict(float)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    meta_year: dict[tuple[int, int], float] = {}

    start_infer = time.perf_counter()
    for xb, _yb, series_id, dec_time_idx, dec_t_year in test_loader:
        x_flat = xb.numpy().reshape(xb.shape[0], -1)
        probs = predict_horizon_probabilities(model, x_flat)
        sid_np = np.asarray(series_id)
        time_np = dec_time_idx.numpy()
        year_np = dec_t_year.numpy()
        for b, sid in enumerate(sid_np):
            for j in range(probs.shape[1]):
                key = (int(sid), int(time_np[b, j]))
                sums[key] += float(probs[b, j])
                counts[key] += 1
                meta_year[key] = float(year_np[b, j])
    infer_seconds = time.perf_counter() - start_infer

    pred = pd.DataFrame(
        {
            GROUP_COLUMN: [key[0] for key in sums],
            TIME_INDEX_COLUMN: [key[1] for key in sums],
            "t_year": [meta_year[key] for key in sums],
            "p_onset_pred": [sums[key] / counts[key] for key in sums],
        }
    )
    target_map = df[[GROUP_COLUMN, TIME_INDEX_COLUMN, TARGET_COLUMN]].drop_duplicates()
    pred = pred.merge(target_map, on=[GROUP_COLUMN, TIME_INDEX_COLUMN], how="left").dropna()
    return pred, infer_seconds


def train_windowed_logistic(
    df: pd.DataFrame,
    args: argparse.Namespace,
    *,
    save_checkpoint: bool = True,
) -> tuple[dict[str, object], pd.DataFrame, Path | None]:
    scaler = fit_point_scaler(df)
    scaled = df.copy()
    scaled[POINT_FEATURES] = scaler.transform(df[POINT_FEATURES])
    scaled["_eval_time_idx"] = df[TIME_INDEX_COLUMN].to_numpy()
    scaled["_eval_t_year"] = df["t_year"].to_numpy()

    train_ds = WindowDataset(
        scaled[scaled["split"] == "train"],
        POINT_FEATURES,
        MAX_ENCODER_LENGTH,
        MAX_PREDICTION_LENGTH,
    )
    test_ds = WindowDataset(
        scaled[scaled["split"] == "test"],
        POINT_FEATURES,
        MAX_ENCODER_LENGTH,
        MAX_PREDICTION_LENGTH,
    )

    x_train, y_train = windows_to_arrays(train_ds)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                MultiOutputClassifier(
                    LogisticRegression(max_iter=1000, solver="lbfgs"),
                    n_jobs=1,
                ),
            ),
        ]
    )

    start_train = time.perf_counter()
    model.fit(x_train, y_train.astype(int))
    train_seconds = time.perf_counter() - start_train

    pred, infer_seconds = overlap_average_test_predictions(model, test_ds, df, args.batch_size)
    result = evaluate_point_predictions(
        pred,
        model_name=MODEL_NAME,
        seed=None,
        parameter_count_value=count_windowed_logistic_params(model),
        training_time_seconds=train_seconds,
        inference_time_seconds=infer_seconds,
    )
    result.metrics["epochs"] = None
    result.metrics["best_epoch"] = None
    result.metrics["best_validation_loss"] = None

    checkpoint_path = REVISION_CHECKPOINT_DIR / CHECKPOINT_NAME
    if save_checkpoint:
        joblib.dump(
            {
                "pipeline": model,
                "features": POINT_FEATURES,
                "encoder_length": MAX_ENCODER_LENGTH,
                "prediction_length": MAX_PREDICTION_LENGTH,
                "point_scaler_mean": scaler.mean_,
                "point_scaler_scale": scaler.scale_,
                "model_name": MODEL_NAME,
            },
            checkpoint_path,
        )
        result.metrics["checkpoint_path"] = str(checkpoint_path)
    else:
        result.metrics["checkpoint_path"] = None
    return result.metrics, result.pf_curve, checkpoint_path if save_checkpoint else None


def write_report(
    metrics: dict[str, object],
    comparison_summary: pd.DataFrame,
    smoke_test: bool,
) -> Path:
    report_path = Path(__file__).resolve().parents[1] / "outputs" / "revision" / REPORT_NAME
    pointwise = comparison_summary.loc[comparison_summary["model"] == "Logistic Regression"]
    mlp = comparison_summary.loc[comparison_summary["model"] == "MLP"]
    gru = comparison_summary.loc[comparison_summary["model"] == "GRU"]
    tft = comparison_summary.loc[comparison_summary["model"] == "TFT"]
    windowed = comparison_summary.loc[comparison_summary["model"] == MODEL_NAME]

    def _metric(row: pd.Series, name: str) -> float | None:
        if row.empty:
            return None
        return float(row.iloc[0][name])

    mae = float(metrics["MAE"])
    rmse = float(metrics["RMSE"])
    max_abs = float(metrics["max_abs_error"])
    final_year = float(metrics["final_year_abs_error"])
    train_s = float(metrics["training_time_seconds"])
    infer_s = float(metrics["pure_inference_time_seconds"])
    params = int(metrics["parameter_count"])

    point_mae = _metric(pointwise, "MAE_mean")
    mlp_mae = _metric(mlp, "MAE_mean")
    gru_mae = _metric(gru, "MAE_mean")
    tft_mae = _metric(tft, "MAE_mean")

    improves_pointwise = point_mae is not None and mae < point_mae
    worse_than_mlp = mlp_mae is not None and mae > mlp_mae
    worse_than_gru = gru_mae is not None and mae > gru_mae
    worse_than_tft = tft_mae is not None and mae > tft_mae

    lines = [
        "# Windowed Logistic Regression Baseline Report",
        "",
        f"**Run mode:** {'smoke test' if smoke_test else 'full benchmark'}",
        "",
        "## Configuration",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- Input: 52-step encoder × 7 covariates → flattened 364 features",
        f"- Output: 13-step horizon via `MultiOutputClassifier(LogisticRegression)`",
        f"- Preprocessing: `StandardScaler` on flattened windows",
        f"- Inference: overlap-averaged window predictions (stride = 1)",
        f"- Target: cumulative onset probability Pf(t)",
        "",
        "## Results",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| MAE | {mae:.6f} |",
        f"| RMSE | {rmse:.6f} |",
        f"| Max absolute error | {max_abs:.6f} |",
        f"| Final-year absolute error | {final_year:.6f} |",
        f"| Training time (s) | {train_s:.2f} |",
        f"| Inference time (s) | {infer_s:.2f} |",
        f"| Parameter count | {params:,} |",
        "",
        "## Comparison",
        "",
        f"- Improves over pointwise Logistic Regression (MAE {point_mae:.6f}): **{'Yes' if improves_pointwise else 'No'}**",
        f"- Remains worse than Windowed MLP (MAE {mlp_mae:.6f}): **{'Yes' if worse_than_mlp else 'No'}**",
        f"- Remains worse than GRU (MAE {gru_mae:.6f}): **{'Yes' if worse_than_gru else 'No'}**",
        f"- Remains worse than TFT (MAE {tft_mae:.6f}): **{'Yes' if worse_than_tft else 'No'}**",
        "",
        "## Manuscript Table Row",
        "",
        "| Model | MAE (mean ± std) | RMSE (mean ± std) |",
        "|---|---|---|",
    ]
    if not windowed.empty:
        row = windowed.iloc[0]
        lines.append(
            f"| {MODEL_NAME} | {row['MAE_mean']:.6f} ± {row['MAE_std']:.6f} | "
            f"{row['RMSE_mean']:.6f} ± {row['RMSE_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Predictions: `outputs/revision/predictions/{PREDICTION_NAME}`",
            f"- Checkpoint: `outputs/revision/checkpoints/{CHECKPOINT_NAME}`",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    df = prepare_data(args)
    metrics, pf_curve, checkpoint_path = train_windowed_logistic(
        df,
        args,
        save_checkpoint=not args.smoke_test,
    )

    print(f"Model: {MODEL_NAME}")
    print(f"MAE: {metrics['MAE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"Max absolute error: {metrics['max_abs_error']:.6f}")
    print(f"Final-year absolute error: {metrics['final_year_abs_error']:.6f}")
    print(f"Training time (s): {metrics['training_time_seconds']:.2f}")
    print(f"Inference time (s): {metrics['pure_inference_time_seconds']:.2f}")
    if checkpoint_path is not None:
        print(f"Checkpoint: {Path(checkpoint_path).name}")

    if args.smoke_test:
        print("Smoke test completed; final benchmark artifacts were not written.")
        return

    pred_path = REVISION_PREDICTION_DIR / PREDICTION_NAME
    pf_curve.to_csv(pred_path, index=False)
    print(f"Predictions: {pred_path.name}")

    by_seed = merge_existing_by_seed([metrics])
    by_seed_path = REVISION_TABLE_DIR / "final_model_comparison_by_seed.csv"
    by_seed.to_csv(by_seed_path, index=False)

    summary = summarize_results(by_seed.to_dict("records"))
    summary_path = REVISION_TABLE_DIR / "final_model_comparison.csv"
    summary.to_csv(summary_path, index=False)

    report_path = write_report(metrics, summary, smoke_test=False)
    print(f"Report: {report_path.name}")


if __name__ == "__main__":
    main()
