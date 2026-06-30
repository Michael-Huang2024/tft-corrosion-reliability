"""
Shared data utilities for reviewer-revision experiments.

The helpers in this file enforce the corrected cumulative target, forbid
predictor leakage, and create series-level train/validation/test splits used by
all revision models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from revision_config import (
    FORBIDDEN_PREDICTORS,
    GROUP_COLUMN,
    INSTANTANEOUS_TARGET_COLUMN,
    PHYSICAL_FEATURES,
    POINT_FEATURES,
    RAW_ONSET_COLUMN,
    REVISION_DATA_DIR,
    REVISION_LABELED_DATA,
    SERIES_SPLIT_PATH,
    SOURCE_LABELED_DATA,
    SPLIT_FRACTIONS,
    SPLIT_SEED,
    TARGET_COLUMN,
    TIME_COLUMN,
    TIME_INDEX_COLUMN,
    ensure_revision_dirs,
)


def ensure_cover_mm(df: pd.DataFrame) -> pd.DataFrame:
    if "cover_mm" in df.columns:
        return df
    if "cover_m" in df.columns:
        df = df.copy()
        df["cover_mm"] = df["cover_m"].astype(float) * 1000.0
        return df
    raise KeyError("Neither 'cover_mm' nor 'cover_m' exists in the dataset.")


def load_source_labeled(path: Path = SOURCE_LABELED_DATA) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing labeled data: {path}")
    df = pd.read_parquet(path)
    df = ensure_cover_mm(df)
    return ensure_cumulative_target(df)


def ensure_cumulative_target(df: pd.DataFrame) -> pd.DataFrame:
    required = [GROUP_COLUMN, TIME_INDEX_COLUMN, "chloride_rebar", "C_th"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for target construction: {missing}")

    df = df.sort_values([GROUP_COLUMN, TIME_INDEX_COLUMN]).reset_index(drop=True).copy()
    df[RAW_ONSET_COLUMN] = (df["chloride_rebar"].values >= df["C_th"].values).astype(int)
    df[TARGET_COLUMN] = df.groupby(GROUP_COLUMN)[RAW_ONSET_COLUMN].cummax().astype(int)
    return df


def validate_target_definition(df: pd.DataFrame, tolerance: float = 1e-12) -> dict[str, object]:
    df = df.sort_values([GROUP_COLUMN, TIME_INDEX_COLUMN]).copy()
    target_diff = df.groupby(GROUP_COLUMN)[TARGET_COLUMN].diff()
    one_to_zero = int((target_diff < 0).sum())

    recomputed_flag = df.groupby(GROUP_COLUMN)[RAW_ONSET_COLUMN].cummax().astype(int)
    final_matches_cummax = bool((df[TARGET_COLUMN].astype(int).values == recomputed_flag.values).all())

    pf = (
        df.groupby(TIME_COLUMN, as_index=False)
        .agg(Pf_corrected=(TARGET_COLUMN, "mean"), Pf_old=(INSTANTANEOUS_TARGET_COLUMN, "mean"))
        .sort_values(TIME_COLUMN)
        .reset_index(drop=True)
    )
    pf_diff = pf["Pf_corrected"].diff()
    decreasing_steps = int((pf_diff < -tolerance).sum())

    comparison_years = []
    for requested_year in [20.0, 40.0, 60.0]:
        idx = (pf[TIME_COLUMN] - requested_year).abs().idxmin()
        row = pf.loc[idx]
        comparison_years.append(
            {
                "requested_year": requested_year,
                "nearest_t_year": float(row[TIME_COLUMN]),
                "Pf_old_instantaneous": float(row["Pf_old"]),
                "Pf_corrected_cumulative": float(row["Pf_corrected"]),
                "difference": float(row["Pf_corrected"] - row["Pf_old"]),
            }
        )

    return {
        "series_count": int(df[GROUP_COLUMN].nunique()),
        "row_count": int(len(df)),
        "one_to_zero_transitions_in_onset_flag": one_to_zero,
        "reference_pf_decreasing_steps": decreasing_steps,
        "final_label_equals_cummax_raw_threshold": final_matches_cummax,
        "final_corrected_pf": float(pf["Pf_corrected"].iloc[-1]),
        "final_old_pf": float(pf["Pf_old"].iloc[-1]),
        "audit_years": comparison_years,
    }


def write_revision_labeled_data(df: pd.DataFrame, path: Path = REVISION_LABELED_DATA) -> Path:
    ensure_revision_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def assert_no_forbidden_predictors(features: list[str]) -> None:
    forbidden = sorted(set(features) & FORBIDDEN_PREDICTORS)
    if forbidden:
        raise ValueError(f"Forbidden leakage-prone predictors configured: {forbidden}")


def validate_model_features(features: list[str]) -> None:
    assert_no_forbidden_predictors(features)
    missing_physical = sorted(set(PHYSICAL_FEATURES) - set(features))
    if missing_physical:
        raise ValueError(f"Model features are missing required physical inputs: {missing_physical}")


def get_point_feature_columns() -> list[str]:
    validate_model_features(POINT_FEATURES)
    return list(POINT_FEATURES)


def create_series_split(
    df: pd.DataFrame,
    split_path: Path = SERIES_SPLIT_PATH,
    seed: int = SPLIT_SEED,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> pd.DataFrame:
    ensure_revision_dirs()
    series = np.array(sorted(df[GROUP_COLUMN].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(series)

    n_total = len(series)
    n_train = int(round(fractions["train"] * n_total))
    n_val = int(round(fractions["validation"] * n_total))
    n_test = n_total - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"Invalid split sizes for {n_total} series: {n_train}, {n_val}, {n_test}")

    split = pd.DataFrame(
        {
            GROUP_COLUMN: np.concatenate(
                [series[:n_train], series[n_train : n_train + n_val], series[n_train + n_val :]]
            ),
            "split": ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test,
        }
    ).sort_values(GROUP_COLUMN)
    split.to_csv(split_path, index=False)
    return split


def load_or_create_series_split(df: pd.DataFrame, split_path: Path = SERIES_SPLIT_PATH) -> pd.DataFrame:
    if split_path.exists():
        split = pd.read_csv(split_path)
    else:
        split = create_series_split(df, split_path)
    validate_series_split(df, split)
    return split


def attach_split(df: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(split, on=GROUP_COLUMN, how="left")
    if merged["split"].isna().any():
        missing = merged.loc[merged["split"].isna(), GROUP_COLUMN].drop_duplicates().head(10).tolist()
        raise ValueError(f"Some series are missing split assignments: {missing}")
    return merged


def validate_series_split(df: pd.DataFrame, split: pd.DataFrame) -> dict[str, object]:
    required = {GROUP_COLUMN, "split"}
    if missing := sorted(required - set(split.columns)):
        raise KeyError(f"Split file missing columns: {missing}")

    duplicated = split[GROUP_COLUMN].duplicated()
    if duplicated.any():
        raise ValueError(f"Duplicate series_id entries in split file: {split.loc[duplicated, GROUP_COLUMN].head().tolist()}")

    split_sets = {
        name: set(split.loc[split["split"] == name, GROUP_COLUMN].astype(int))
        for name in ["train", "validation", "test"]
    }
    overlaps = {
        "train_validation": len(split_sets["train"] & split_sets["validation"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "validation_test": len(split_sets["validation"] & split_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Series split overlap detected: {overlaps}")

    all_series = set(df[GROUP_COLUMN].astype(int).unique())
    assigned = set(split[GROUP_COLUMN].astype(int))
    if all_series != assigned:
        raise ValueError(
            f"Split assignments do not match data series. Missing={len(all_series - assigned)}, extra={len(assigned - all_series)}"
        )

    static_cols = ["Cs", "D28", "m_aging", "cover_mm", "C_th"]
    combos = df[[GROUP_COLUMN, *static_cols]].drop_duplicates()
    combo_split = combos.merge(split, on=GROUP_COLUMN, how="left")
    combo_overlap = int(combo_split.groupby(static_cols)["split"].nunique().gt(1).sum())
    if combo_overlap:
        raise ValueError(f"Static parameter combinations appear in multiple splits: {combo_overlap}")

    counts = split["split"].value_counts().to_dict()
    proportions = {name: counts.get(name, 0) / len(split) for name in ["train", "validation", "test"]}
    expected = SPLIT_FRACTIONS
    near_exact = {
        name: abs(proportions[name] - expected[name]) <= max(0.01, 1.0 / len(split))
        for name in expected
    }
    if not all(near_exact.values()):
        raise ValueError(f"Split proportions outside tolerance: observed={proportions}, expected={expected}")

    return {
        "counts": {name: int(counts.get(name, 0)) for name in ["train", "validation", "test"]},
        "proportions": proportions,
        "overlaps": overlaps,
        "static_parameter_combo_overlap_count": combo_overlap,
        "series_count": int(len(split)),
    }


def save_json_audit(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def nearest_common_times(df: pd.DataFrame, min_time_idx: int) -> pd.DataFrame:
    return df[df[TIME_INDEX_COLUMN] >= min_time_idx].copy()
