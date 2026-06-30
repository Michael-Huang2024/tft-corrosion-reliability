"""
Shared evaluation utilities for fair reviewer-revision model comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from revision_config import GROUP_COLUMN, TARGET_COLUMN, TIME_COLUMN, TIME_INDEX_COLUMN


@dataclass(frozen=True)
class EvaluationResult:
    model: str
    seed: int | None
    metrics: dict[str, float | int | str | None]
    pf_curve: pd.DataFrame


def parameter_count(model: object | None) -> int | None:
    if model is None:
        return None
    if hasattr(model, "parameters"):
        return int(sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", False)))
    if hasattr(model, "coef_"):
        coef = getattr(model, "coef_")
        intercept = getattr(model, "intercept_", [])
        return int(np.size(coef) + np.size(intercept))
    if hasattr(model, "named_steps"):
        for step in reversed(model.named_steps.values()):
            count = parameter_count(step)
            if count is not None:
                return count
    return None


def aggregate_population_pf(
    point_predictions: pd.DataFrame,
    probability_column: str = "p_onset_pred",
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    required = {TIME_COLUMN, probability_column, target_column}
    if missing := sorted(required - set(point_predictions.columns)):
        raise KeyError(f"Missing columns for Pf aggregation: {missing}")

    pf = (
        point_predictions.groupby(TIME_COLUMN, as_index=False)
        .agg(Pf_true=(target_column, "mean"), Pf_pred=(probability_column, "mean"))
        .sort_values(TIME_COLUMN)
        .reset_index(drop=True)
    )
    return pf


def evaluate_pf_curve(
    pf: pd.DataFrame,
    model_name: str,
    seed: int | None,
    parameter_count_value: int | None,
    training_time_seconds: float | None,
    inference_time_seconds: float | None,
) -> dict[str, float | int | str | None]:
    required = {"Pf_true", "Pf_pred", TIME_COLUMN}
    if missing := sorted(required - set(pf.columns)):
        raise KeyError(f"Missing columns for metric calculation: {missing}")

    error = pf["Pf_pred"].astype(float) - pf["Pf_true"].astype(float)
    abs_error = error.abs()
    max_idx = int(abs_error.idxmax())
    return {
        "model": model_name,
        "seed": seed,
        "MAE": float(abs_error.mean()),
        "RMSE": float(np.sqrt((error**2).mean())),
        "max_abs_error": float(abs_error.loc[max_idx]),
        "year_of_max_error": float(pf.loc[max_idx, TIME_COLUMN]),
        "final_year_abs_error": float(abs_error.iloc[-1]),
        "parameter_count": parameter_count_value,
        "training_time_seconds": training_time_seconds,
        "pure_inference_time_seconds": inference_time_seconds,
        "evaluation_start_year": float(pf[TIME_COLUMN].iloc[0]),
        "evaluation_end_year": float(pf[TIME_COLUMN].iloc[-1]),
        "evaluation_time_points": int(len(pf)),
    }


def evaluate_point_predictions(
    point_predictions: pd.DataFrame,
    model_name: str,
    seed: int | None,
    parameter_count_value: int | None,
    training_time_seconds: float | None,
    inference_time_seconds: float | None,
    probability_column: str = "p_onset_pred",
) -> EvaluationResult:
    pf = aggregate_population_pf(point_predictions, probability_column=probability_column)
    metrics = evaluate_pf_curve(
        pf,
        model_name=model_name,
        seed=seed,
        parameter_count_value=parameter_count_value,
        training_time_seconds=training_time_seconds,
        inference_time_seconds=inference_time_seconds,
    )
    return EvaluationResult(model=model_name, seed=seed, metrics=metrics, pf_curve=pf)


def restrict_common_evaluation_range(
    df: pd.DataFrame,
    min_time_idx: int,
    split_name: str = "test",
) -> pd.DataFrame:
    required = {"split", GROUP_COLUMN, TIME_INDEX_COLUMN, TIME_COLUMN, TARGET_COLUMN}
    if missing := sorted(required - set(df.columns)):
        raise KeyError(f"Missing required evaluation columns: {missing}")
    return df[(df["split"] == split_name) & (df[TIME_INDEX_COLUMN] >= min_time_idx)].copy()


def cover_depth_group(cover_mm: pd.Series) -> pd.Series:
    bins = [40, 50, 60, 70, 80, 90, 100, 110]
    labels = [f"{lo}-{hi} mm" for lo, hi in zip(bins[:-1], bins[1:])]
    return pd.cut(cover_mm, bins=bins, labels=labels, include_lowest=True, right=False)
