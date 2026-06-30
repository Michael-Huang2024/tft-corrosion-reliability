"""
50 vs 100 MC Dropout convergence check for validation-selected TFT (seed 20250111).

Runs 100 stochastic forward passes with fixed weights and dropout-only activation,
saves pass-level population trajectories, and compares nested subsets at 20, 50, and
100 passes.
"""

from __future__ import annotations

import argparse
import json
import random
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
BASE_SEED = 20250626
TOTAL_PASSES = 100
CONVERGENCE_LEVELS = (20, 50, 100)

PASS_LEVEL_FILE = REVISION_PREDICTION_DIR / "mc_dropout_seed20250111_100pass_level_population.csv"
STATE_FILE = REVISION_PREDICTION_DIR / "mc_dropout_seed20250111_100pass_state.json"
SUMMARY_FILE = REVISION_PREDICTION_DIR / "mc_dropout_seed20250111_summary_20_50_100.csv"
METRICS_FILE = REVISION_TABLE_DIR / "final_mc_dropout_metrics_20_50_100.csv"
CONVERGENCE_FILE = REVISION_TABLE_DIR / "final_mc_dropout_convergence_20_50_100.csv"
REPORT_FILE = REVISION_OUTPUT_DIR / "final_mc_dropout_50_vs_100_convergence_report.md"

DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MC Dropout 50 vs 100 convergence check.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--passes", type=int, default=TOTAL_PASSES)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from saved pass-level checkpoints when available (default: true).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete saved pass-level progress and restart from pass 1.",
    )
    return parser.parse_args()


def set_pass_seed(pass_id: int) -> None:
    seed = BASE_SEED + pass_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_dropout_modules(model: nn.Module) -> list[str]:
    names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, DROPOUT_TYPES):
            names.append(name if name else type(module).__name__)
    return names


def enable_dropout_only(model: nn.Module) -> list[str]:
    model.eval()
    active: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, DROPOUT_TYPES):
            module.train()
            active.append(name if name else type(module).__name__)
    return active


def prepare_test_loader():
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import NaNLabelEncoder

    df = pd.read_parquet(FINAL_LABELED_DATA)
    split = load_or_create_series_split(df)
    validate_series_split(df, split)
    df = attach_split(df, split)
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    n_test_series = int(test_df[GROUP_COLUMN].nunique())
    test_df_sorted = test_df.sort_values([GROUP_COLUMN, TIME_INDEX_COLUMN]).reset_index(drop=True)
    enc = {f"__group_id__{GROUP_COLUMN}": NaNLabelEncoder(add_nan=True)}
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
        categorical_encoders=enc,
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
    offset = 0
    with torch.no_grad():
        for batch in loader:
            x, _ = batch
            x_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
            probs = torch.softmax(model(x_dev)["prediction"], dim=-1)[..., 1].detach().cpu().numpy()
            if not np.isfinite(probs).all():
                raise ValueError("Non-finite MC Dropout predictions")
            bs = probs.shape[0]
            row_starts = sample_index.iloc[offset : offset + bs]["index_start"].to_numpy(dtype=int)
            series_ids = test_df_sorted.iloc[row_starts][GROUP_COLUMN].to_numpy(dtype=int)
            offset += bs
            dec_t = x["decoder_time_idx"].detach().cpu().numpy().astype(int)
            for b, sid in enumerate(series_ids):
                for j in range(probs.shape[1]):
                    key = (int(sid), int(dec_t[b, j]))
                    sums[key] += float(probs[b, j])
                    counts[key] += 1
    point = pd.DataFrame(
        {
            GROUP_COLUMN: [k[0] for k in sums],
            TIME_INDEX_COLUMN: [k[1] for k in sums],
            "p_onset_pred": [sums[k] / counts[k] for k in sums],
        }
    )
    point = point.merge(target_map, on=[GROUP_COLUMN, TIME_INDEX_COLUMN], how="left").dropna()
    return (
        point.groupby([TIME_INDEX_COLUMN, TIME_COLUMN], as_index=False)
        .agg(Pf_true=(TARGET_COLUMN, "mean"), Pf_pred=("p_onset_pred", "mean"))
        .sort_values(TIME_INDEX_COLUMN)
        .reset_index(drop=True)
    )


def summarize_subset(trajectories: list[pd.DataFrame]) -> pd.DataFrame:
    base = trajectories[0][[TIME_INDEX_COLUMN, TIME_COLUMN, "Pf_true"]].copy()
    matrix = np.vstack([t["Pf_pred"].to_numpy(dtype=float) for t in trajectories])
    out = base.rename(columns={TIME_COLUMN: "time_years", "Pf_true": "reference_pf"})
    out["predictive_mean"] = matrix.mean(axis=0)
    out["predictive_median"] = np.median(matrix, axis=0)
    out["predictive_std"] = matrix.std(axis=0, ddof=1) if len(trajectories) > 1 else 0.0
    out["q025"] = np.quantile(matrix, 0.025, axis=0)
    out["q975"] = np.quantile(matrix, 0.975, axis=0)
    out["interval_width"] = out["q975"] - out["q025"]
    if not np.isfinite(out.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Non-finite summary statistics")
    return out


def compute_global_metrics(summary: pd.DataFrame) -> dict[str, float | int]:
    error = summary["predictive_mean"] - summary["reference_pf"]
    abs_error = error.abs()
    in_interval = (summary["reference_pf"] >= summary["q025"]) & (summary["reference_pf"] <= summary["q975"])
    diffs = summary["predictive_mean"].diff()
    mono_violations = int((diffs < -1e-12).sum())
    max_idx = int(abs_error.idxmax())
    return {
        "MAE": float(abs_error.mean()),
        "RMSE": float(np.sqrt((error**2).mean())),
        "mean_predictive_std": float(summary["predictive_std"].mean()),
        "max_predictive_std": float(summary["predictive_std"].max()),
        "PICP_95": float(in_interval.mean()),
        "MPIW_95": float(summary["interval_width"].mean()),
        "final_time_abs_error": float(abs_error.iloc[-1]),
        "max_abs_error": float(abs_error.loc[max_idx]),
        "year_of_max_error": float(summary["time_years"].loc[max_idx]),
        "monotonicity_violations": mono_violations,
        "evaluation_time_points": int(len(summary)),
    }


def compare_summaries(a: pd.DataFrame, b: pd.DataFrame, ma: dict, mb: dict, label: str) -> dict:
    mean_diff = (b["predictive_mean"] - a["predictive_mean"]).abs()
    std_diff = (b["predictive_std"] - a["predictive_std"]).abs()
    q025_diff = (b["q025"] - a["q025"]).abs()
    q975_diff = (b["q975"] - a["q975"]).abs()
    rel_std = abs(mb["mean_predictive_std"] - ma["mean_predictive_std"]) / max(ma["mean_predictive_std"], 1e-12)
    rel_mpiw = abs(mb["MPIW_95"] - ma["MPIW_95"]) / max(ma["MPIW_95"], 1e-12)
    return {
        "comparison": label,
        "mean_abs_predictive_mean_change": float(mean_diff.mean()),
        "max_abs_predictive_mean_change": float(mean_diff.max()),
        "mean_abs_predictive_std_change": float(std_diff.mean()),
        "max_abs_predictive_std_change": float(std_diff.max()),
        "mean_abs_q025_change": float(q025_diff.mean()),
        "mean_abs_q975_change": float(q975_diff.mean()),
        "MAE_change": float(mb["MAE"] - ma["MAE"]),
        "RMSE_change": float(mb["RMSE"] - ma["RMSE"]),
        "PICP_change": float(mb["PICP_95"] - ma["PICP_95"]),
        "MPIW_change": float(mb["MPIW_95"] - ma["MPIW_95"]),
        "relative_change_in_mean_predictive_std": float(rel_std),
        "relative_change_in_MPIW": float(rel_mpiw),
    }


def stability_label(mean_change: float, std_change: float, rel_mpiw: float) -> str:
    parts = []
    parts.append("predictive mean highly stable" if mean_change < 1e-4 else "predictive mean not at 1e-4 threshold")
    parts.append("predictive std highly stable" if std_change < 1e-4 else "predictive std not at 1e-4 threshold")
    parts.append("MPIW basically stable (<5%)" if rel_mpiw < 0.05 else "MPIW change >=5%")
    return "; ".join(parts)


def load_existing_pass_level() -> pd.DataFrame | None:
    if not PASS_LEVEL_FILE.exists():
        return None
    df = pd.read_csv(PASS_LEVEL_FILE)
    required = {"pass_id", "time_idx", "time_years", "reference_pf", "population_prediction"}
    if not required.issubset(df.columns):
        return None
    return df


def load_state_file() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def validate_completed_passes(df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """Return the highest contiguous completed pass id starting from 1."""
    if df.empty:
        return 0, df.iloc[0:0].copy()
    valid_parts: list[pd.DataFrame] = []
    last_ok = 0
    for pass_id in range(1, int(df["pass_id"].max()) + 1):
        sub = df[df["pass_id"] == pass_id].sort_values("time_idx").reset_index(drop=True)
        if len(sub) != EXPECTED_EVAL_POINTS:
            break
        if sub["time_idx"].nunique() != EXPECTED_EVAL_POINTS:
            break
        valid_parts.append(sub)
        last_ok = pass_id
    if not valid_parts:
        return 0, df.iloc[0:0].copy()
    return last_ok, pd.concat(valid_parts, ignore_index=True)


def atomic_write_pass_level(df: pd.DataFrame) -> None:
    tmp = PASS_LEVEL_FILE.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(PASS_LEVEL_FILE)


def write_state_file(
    *,
    pass_id_completed: int,
    total_passes: int,
    tft_checkpoint: Path,
    device: str,
) -> None:
    payload = {
        "pass_id_completed": pass_id_completed,
        "total_passes": total_passes,
        "tft_checkpoint": tft_checkpoint.as_posix(),
        "base_seed": BASE_SEED,
        "evaluation_time_points": EXPECTED_EVAL_POINTS,
        "device": device,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def clear_resume_files() -> None:
    for path in (PASS_LEVEL_FILE, STATE_FILE):
        if path.exists():
            path.unlink()


def resolve_resume_plan(
    *,
    n_passes: int,
    tft_checkpoint: Path,
    resume: bool,
    fresh: bool,
) -> tuple[int, pd.DataFrame | None, list[pd.DataFrame], str, int]:
    if fresh:
        clear_resume_files()
        return (
            1,
            None,
            [],
            f"Fresh start requested; removed any prior `{PASS_LEVEL_FILE.name}` / `{STATE_FILE.name}`.",
            n_passes,
        )

    if not resume:
        return (
            1,
            None,
            [],
            "Resume disabled (`--no-resume`); running all passes from scratch.",
            n_passes,
        )

    existing = load_existing_pass_level()
    if existing is None:
        return (
            1,
            None,
            [],
            "No pass-level checkpoint found; starting from pass 1.",
            n_passes,
        )

    completed, valid_df = validate_completed_passes(existing)
    state = load_state_file()
    if state and state.get("tft_checkpoint") and state["tft_checkpoint"] != tft_checkpoint.as_posix():
        raise ValueError(
            f"Saved state checkpoint ({state['tft_checkpoint']}) does not match "
            f"requested checkpoint ({tft_checkpoint.as_posix()}). Use --fresh to restart."
        )

    if completed >= n_passes:
        trajectories = trajectories_from_pass_level(valid_df)[:n_passes]
        return (
            n_passes + 1,
            valid_df,
            trajectories,
            f"Reused all {n_passes} pass-level trajectories from `{PASS_LEVEL_FILE.name}`; "
            "no inference re-run required.",
            0,
        )

    if completed >= 1:
        trajectories = trajectories_from_pass_level(valid_df)
        start = completed + 1
        return (
            start,
            valid_df,
            trajectories,
            f"Resumed from pass {start}: reused passes 1–{completed} from `{PASS_LEVEL_FILE.name}` "
            f"({completed * EXPECTED_EVAL_POINTS:,} saved rows).",
            n_passes - completed,
        )

    return (
        1,
        None,
        [],
        "Pass-level file existed but contained no valid complete passes; restarting from pass 1.",
        n_passes,
    )


def trajectories_from_pass_level(df: pd.DataFrame) -> list[pd.DataFrame]:
    trajectories: list[pd.DataFrame] = []
    for pass_id in sorted(df["pass_id"].unique()):
        sub = df[df["pass_id"] == pass_id].sort_values("time_idx").reset_index(drop=True)
        trajectories.append(
            pd.DataFrame(
                {
                    TIME_INDEX_COLUMN: sub["time_idx"].astype(int),
                    TIME_COLUMN: sub["time_years"].astype(float),
                    "Pf_true": sub["reference_pf"].astype(float),
                    "Pf_pred": sub["population_prediction"].astype(float),
                }
            )
        )
    return trajectories


def hardware_info(device: str) -> dict:
    env_path = REVISION_OUTPUT_DIR / "environment.json"
    if env_path.exists():
        return json.loads(env_path.read_text(encoding="utf-8"))
    return {
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_figures(summaries: dict[int, pd.DataFrame]) -> float:
    t0 = time.perf_counter()
    ref = summaries[100]
    t = ref["time_years"]

    plt.figure(figsize=(7, 4))
    plt.plot(t, ref["reference_pf"], label="Reference cumulative Pf(t)")
    plt.plot(t, ref["predictive_mean"], label="MC Dropout predictive mean (100 passes)")
    plt.fill_between(t, ref["q025"], ref["q975"], alpha=0.25, label="95% approximate posterior predictive interval")
    plt.xlabel("Time (years)")
    plt.ylabel("Cumulative corrosion initiation probability")
    plt.title("MC Dropout-based approximate posterior predictive distribution (100 passes)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "mc_dropout_uncertainty_band_100passes.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(t, ref["reference_pf"], label="Reference Pf(t)", color="black", linewidth=1.5)
    for n, color in [(20, "C0"), (50, "C1"), (100, "C2")]:
        plt.plot(t, summaries[n]["predictive_mean"], label=f"{n}-pass predictive mean", color=color)
    plt.xlabel("Time (years)")
    plt.ylabel("Predictive mean")
    plt.title("Predictive mean convergence (20 / 50 / 100 passes)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "mc_dropout_mean_convergence_20_50_100.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    for n, color in [(20, "C0"), (50, "C1"), (100, "C2")]:
        plt.plot(t, summaries[n]["predictive_std"], label=f"{n}-pass predictive std", color=color)
    plt.xlabel("Time (years)")
    plt.ylabel("Predictive std (epistemic uncertainty estimate)")
    plt.title("Predictive std convergence (20 / 50 / 100 passes)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "mc_dropout_std_convergence_20_50_100.png", dpi=300)
    plt.close()

    s50, s100 = summaries[50], summaries[100]
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(t, (s100["predictive_mean"] - s50["predictive_mean"]).abs(), color="C0")
    axes[0].set_ylabel("|mean_100 - mean_50|")
    axes[0].set_title("50 vs 100 pass absolute differences")
    axes[1].plot(t, (s100["predictive_std"] - s50["predictive_std"]).abs(), color="C1")
    axes[1].set_ylabel("|std_100 - std_50|")
    axes[1].set_xlabel("Time (years)")
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "mc_dropout_convergence_difference_50_vs_100.png", dpi=300)
    plt.close()
    return time.perf_counter() - t0


def write_report(
    *,
    checkpoint: Path,
    n_test_series: int,
    dropout_modules: list[str],
    reuse_note: str,
    metrics_rows: list[dict],
    conv_rows: list[dict],
    cmp_50_100: dict,
    hardware: dict,
    inference_seconds: float,
    aggregate_seconds: float,
    figure_seconds: float,
    passes_completed: int,
    additional_passes: int,
) -> None:
    m = {r["n_passes"]: r for r in metrics_rows}
    c50 = cmp_50_100 if cmp_50_100 else {}
    mean_stable = c50.get("mean_abs_predictive_mean_change", float("nan")) < 1e-4 if c50 else False
    std_stable = c50.get("mean_abs_predictive_std_change", float("nan")) < 1e-4 if c50 else False
    mpiw_stable = c50.get("relative_change_in_MPIW", float("nan")) < 0.05 if c50 else False

    if not c50:
        adequacy = "Not evaluated — 50 vs 100 comparison requires at least 100 passes."
    elif mean_stable and std_stable:
        adequacy = "Yes — 50 passes yield highly stable predictive mean and std (<1e-4 mean absolute change vs 100)."
    elif mean_stable:
        adequacy = "Partial — predictive mean is stable at 50 passes; predictive std shows larger 50→100 drift."
    else:
        adequacy = "No — 50→100 changes exceed the descriptive stability thresholds; report values honestly."

    lines = [
        "# Final MC Dropout 50 vs 100 Convergence Report",
        "",
        "## A. Model selection",
        "",
        f"- Selected seed: **{SELECTED_SEED}**",
        f"- Selection criterion: **best validation loss** (not test MAE)",
        f"- Best validation loss: **{SELECTED_BEST_VAL_LOSS:.6f}**",
        f"- Checkpoint: `{checkpoint.as_posix()}`",
        "",
        "## B. Evaluation setup",
        "",
        f"- Split: independent test only",
        f"- Test series: {n_test_series}",
        f"- Population evaluation time points: {EXPECTED_EVAL_POINTS}",
        f"- Time range: ~3.99–59.95 years",
        f"- Target: cumulative corrosion initiation `onset_flag`",
        f"- Method: MC Dropout-based approximate Bayesian inference",
        f"- Dropout activation: `model.eval()` globally; dropout modules set to train mode only",
        f"- Active dropout modules ({len(dropout_modules)}): {', '.join(dropout_modules)}",
        f"- Stochastic forward passes: {passes_completed} (base seed {BASE_SEED} + pass_id)",
        f"- Inference batch size: {INFER_BATCH}; num_workers: 0; precision: 32-true",
        "",
        "### Pass-level reuse",
        "",
        reuse_note,
        f"- Resume state file: `{STATE_FILE.name}`",
        "",
        "## C. Results for 20, 50, and 100 passes",
        "",
        pd.DataFrame(metrics_rows).to_string(index=False),
        "",
        "## D. 20 vs 50 convergence",
        "",
        pd.DataFrame([r for r in conv_rows if r["comparison"] == "20_vs_50"]).to_string(index=False),
        "",
        "## E. 50 vs 100 convergence",
        "",
        pd.DataFrame([r for r in conv_rows if r["comparison"] == "50_vs_100"]).to_string(index=False)
        if any(r["comparison"] == "50_vs_100" for r in conv_rows)
        else "_Not available (requires 100 passes)._",
        "",
    ]
    if c50:
        lines.extend(
            [
                "### Descriptive stability assessment (50 vs 100)",
                "",
                f"- mean_abs_predictive_mean_change: {c50['mean_abs_predictive_mean_change']:.6e} "
                f"({'<' if mean_stable else '>='} 1e-4)",
                f"- mean_abs_predictive_std_change: {c50['mean_abs_predictive_std_change']:.6e} "
                f"({'<' if std_stable else '>='} 1e-4)",
                f"- relative_change_in_MPIW: {c50['relative_change_in_MPIW']:.4%} "
                f"({'<' if mpiw_stable else '>='} 5%)",
                f"- Assessment: {stability_label(c50['mean_abs_predictive_mean_change'], c50['mean_abs_predictive_std_change'], c50['relative_change_in_MPIW'])}",
                "",
            ]
        )
    lines.extend(
        [
        "## F. Main conclusion",
        "",
    ])
    if c50:
        lines.extend(
            [
                f"1. **Is 50 passes sufficient for predictive mean?** "
                f"{'Yes' if mean_stable else 'Not fully'} — mean absolute change 50→100 is "
                f"{c50['mean_abs_predictive_mean_change']:.6e}.",
                f"2. **Is 50 passes sufficient for predictive std?** "
                f"{'Yes' if std_stable else 'Not fully'} — mean absolute std change is "
                f"{c50['mean_abs_predictive_std_change']:.6e} (relative {c50['relative_change_in_mean_predictive_std']:.4%}).",
                f"3. **Does 100 passes significantly change 95% interval, PICP, or MPIW?** "
                f"PICP change {c50['PICP_change']:+.4f}, MPIW change {c50['MPIW_change']:+.6e} "
                f"(relative MPIW {c50['relative_change_in_MPIW']:.4%}).",
                f"4. **Is 50 passes a reasonable cost–stability trade-off?** {adequacy}",
                "",
            ]
        )
    else:
        lines.append("_50 vs 100 conclusions require the full 100-pass run._\n")
    lines.extend(
        [
            "## G. Important interpretation",
            "",
            "This analysis uses **MC Dropout-based approximate Bayesian inference** to approximate the "
            "**approximate posterior predictive distribution** and an **epistemic uncertainty estimate** "
            "via **stochastic forward passes** over a single trained TFT checkpoint.",
            "",
            "Each dropout mask corresponds to one random effective sub-network of the same trained weights; "
            "this is **not** exact Bayesian inference, **not** an exact posterior, **not** a fully Bayesian TFT, "
            "and **not** 100 independently trained TFT models.",
            "",
            "## Runtime and hardware",
            "",
            f"- Inference time ({passes_completed} passes): {inference_seconds:.1f} s "
            f"({inference_seconds / passes_completed:.2f} s/pass)",
            f"- Additional passes completed this run: {additional_passes}",
            f"- Aggregation / metrics time: {aggregate_seconds:.1f} s",
            f"- Figure generation time: {figure_seconds:.1f} s",
            f"- Hardware: {json.dumps(hardware.get('torch_cuda', hardware), ensure_ascii=False)}",
            "",
        ]
    )
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    from pytorch_forecasting import TemporalFusionTransformer

    args = parse_args()
    ensure_revision_dirs()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    n_passes = min(args.passes, 5) if args.smoke_test else args.passes
    levels = tuple(n for n in CONVERGENCE_LEVELS if n <= n_passes) or (n_passes,)

    start, existing, trajectories, reuse_note, additional_passes = resolve_resume_plan(
        n_passes=n_passes,
        tft_checkpoint=checkpoint,
        resume=args.resume,
        fresh=args.fresh,
    )
    inference_seconds = 0.0

    if start > n_passes:
        print(reuse_note, flush=True)
    elif start > 1:
        print(reuse_note, flush=True)
        print(f"Continuing MC Dropout passes {start}–{n_passes} ({additional_passes} remaining).", flush=True)
    else:
        print(reuse_note, flush=True)

    dataset, loader, target_map, test_df_sorted, n_test_series = prepare_test_loader()
    device = torch.device(args.device)
    model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint), weights_only=False)
    model.to(device)
    dropout_modules = enable_dropout_only(model)

    if start <= n_passes and additional_passes > 0:
        t0 = time.perf_counter()
        checkpoint_rows = existing.copy() if existing is not None and start > 1 else None
        for pass_id in range(start, n_passes + 1):
            set_pass_seed(pass_id)
            traj = one_population_pass(model, dataset, loader, target_map, test_df_sorted, device)
            if len(traj) != EXPECTED_EVAL_POINTS and not args.smoke_test:
                raise ValueError(f"Pass {pass_id}: expected {EXPECTED_EVAL_POINTS} points, got {len(traj)}")
            trajectories.append(traj)
            pass_rows = [
                {
                    "pass_id": pass_id,
                    "time_idx": int(row[TIME_INDEX_COLUMN]),
                    "time_years": float(row[TIME_COLUMN]),
                    "reference_pf": float(row["Pf_true"]),
                    "population_prediction": float(row["Pf_pred"]),
                }
                for _, row in traj.iterrows()
            ]
            pass_df = pd.DataFrame(pass_rows)
            if checkpoint_rows is None:
                checkpoint_rows = pass_df
            else:
                checkpoint_rows = pd.concat([checkpoint_rows, pass_df], ignore_index=True)
            atomic_write_pass_level(checkpoint_rows)
            write_state_file(
                pass_id_completed=pass_id,
                total_passes=n_passes,
                tft_checkpoint=checkpoint,
                device=args.device,
            )
            elapsed = time.perf_counter() - t0
            if pass_id % 10 == 0 or pass_id == n_passes:
                print(f"MC Dropout pass {pass_id}/{n_passes} ({elapsed:.1f}s elapsed, checkpoint saved)", flush=True)
            else:
                print(f"MC Dropout pass {pass_id}/{n_passes} checkpoint saved", flush=True)

        inference_seconds = time.perf_counter() - t0

    if len(trajectories) < 2:
        raise RuntimeError("Need at least two passes for stochastic variability check")
    p0 = trajectories[0]["Pf_pred"].to_numpy()
    p1 = trajectories[1]["Pf_pred"].to_numpy()
    if np.allclose(p0, p1):
        raise RuntimeError("Passes 1 and 2 produced identical population predictions — dropout may be inactive")

    t_agg = time.perf_counter()
    summaries = {n: summarize_subset(trajectories[:n]) for n in levels}
    metrics_rows = []
    for n in levels:
        m = compute_global_metrics(summaries[n])
        metrics_rows.append({"n_passes": n, **m})

    conv_rows = []
    metrics_by_n = {r["n_passes"]: r for r in metrics_rows}
    if 20 in summaries and 50 in summaries:
        conv_rows.append(compare_summaries(summaries[20], summaries[50], metrics_by_n[20], metrics_by_n[50], "20_vs_50"))
    if 50 in summaries and 100 in summaries:
        conv_rows.append(compare_summaries(summaries[50], summaries[100], metrics_by_n[50], metrics_by_n[100], "50_vs_100"))
    cmp_50_100 = conv_rows[-1] if conv_rows and conv_rows[-1]["comparison"] == "50_vs_100" else {}

    summary_parts = []
    for n in levels:
        part = summaries[n].copy()
        part.insert(0, "n_passes", n)
        part = part.rename(columns={TIME_INDEX_COLUMN: "time_idx"})
        summary_parts.append(part)
    pd.concat(summary_parts, ignore_index=True).to_csv(SUMMARY_FILE, index=False)
    pd.DataFrame(metrics_rows).to_csv(METRICS_FILE, index=False)
    pd.DataFrame(conv_rows).to_csv(CONVERGENCE_FILE, index=False)
    aggregate_seconds = time.perf_counter() - t_agg

    figure_seconds = write_figures(summaries) if 100 in summaries else 0.0
    hw = hardware_info(args.device)

    write_report(
        checkpoint=checkpoint,
        n_test_series=n_test_series,
        dropout_modules=dropout_modules,
        reuse_note=reuse_note,
        metrics_rows=metrics_rows,
        conv_rows=conv_rows,
        cmp_50_100=cmp_50_100,
        hardware=hw,
        inference_seconds=inference_seconds if inference_seconds else float("nan"),
        aggregate_seconds=aggregate_seconds,
        figure_seconds=figure_seconds,
        passes_completed=n_passes,
        additional_passes=additional_passes,
    )

    m20 = next((r for r in metrics_rows if r["n_passes"] == 20), metrics_rows[0])
    m50 = next((r for r in metrics_rows if r["n_passes"] == 50), metrics_rows[-1])
    m100 = next((r for r in metrics_rows if r["n_passes"] == 100), metrics_rows[-1])

    print("\nMC DROPOUT 50 VS 100 CONVERGENCE CHECK COMPLETE\n")
    print(f"Selected checkpoint: {checkpoint}")
    print(f"Additional passes completed: {additional_passes}")
    print(f"Total passes: {n_passes}")
    print(f"20-pass MAE: {m20['MAE']:.6f}")
    print(f"50-pass MAE: {m50['MAE']:.6f}")
    print(f"100-pass MAE: {m100['MAE']:.6f}")
    print(f"20-pass mean predictive std: {m20['mean_predictive_std']:.6f}")
    print(f"50-pass mean predictive std: {m50['mean_predictive_std']:.6f}")
    print(f"100-pass mean predictive std: {m100['mean_predictive_std']:.6f}")
    if cmp_50_100:
        print(f"50 vs 100 mean absolute predictive mean change: {cmp_50_100['mean_abs_predictive_mean_change']:.6e}")
        print(f"50 vs 100 mean absolute predictive std change: {cmp_50_100['mean_abs_predictive_std_change']:.6e}")
        print(f"50 vs 100 MPIW relative change: {cmp_50_100['relative_change_in_MPIW']:.4%}")
        print(f"50 vs 100 PICP change: {cmp_50_100['PICP_change']:+.4f}")
    print(f"Conclusion on adequacy of 50 passes: {stability_label(cmp_50_100.get('mean_abs_predictive_mean_change', float('nan')), cmp_50_100.get('mean_abs_predictive_std_change', float('nan')), cmp_50_100.get('relative_change_in_MPIW', float('nan')))}")
    print(f"Total additional runtime: {inference_seconds:.1f} s")
    print(f"Output report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
