"""
06_extract_predicted_onset.py

Convert raw TFT predictions (time_to_onset) into predicted corrosion initiation time (t_onset_pred)
and compare with true initiation time (t_onset_true).

Inputs:
- data/processed/chloride_labeled.parquet
- outputs/predictions/time_to_onset_preds_raw.npy

Outputs:
- outputs/predictions/onset_comparison.csv
- outputs/predictions/onset_pred_long.parquet (optional, large)
"""

from pathlib import Path
import numpy as np
import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer


def main():
    data_path = Path("data/processed/chloride_labeled.parquet")
    pred_path = Path("outputs/predictions/time_to_onset_preds_raw.npy")

    if not data_path.exists():
        raise FileNotFoundError(f"Missing: {data_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing: {pred_path}")

    df = pd.read_parquet(data_path)

    # keep only observed onset series (same filter as training/inference)
    df = df[df["time_to_onset"].notna()].copy()

    max_encoder_length = 52
    max_prediction_length = 13
    training_cutoff = df["time_idx"].max() - max_prediction_length

    # rebuild dataset EXACTLY like inference (must match!)
    dataset = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="time_to_onset",
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx", "t_year"],
        time_varying_unknown_reals=["chloride_rebar", "time_to_onset"],
        static_reals=["Cs", "cover_m", "D28", "m_aging", "C_th"],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # raw predictions: shape (n_samples, prediction_length)
    preds = np.load(pred_path)
    n_samples, pred_len = preds.shape
    if pred_len != max_prediction_length:
        raise ValueError(f"Prediction length mismatch: preds has {pred_len}, expected {max_prediction_length}")

    # Build mapping from sample index -> (series_id, decoder_time_idx)
    # We use dataset.to_dataloader(train=False) which yields batches in a deterministic order.
    dl = dataset.to_dataloader(train=False, batch_size=256, num_workers=0)

    series_ids = []
    decoder_time_idxs = []

    # We only need x (inputs) to extract decoder time index and group ids
    for x, y in dl:
        # group_ids are stored in x["groups"] (tensor), first column is series_id (because group_ids=["series_id"])
        g = x["groups"][:, 0].detach().cpu().numpy().astype(int)
        # decoder time_idx: x["decoder_time_idx"] shape (batch, pred_len)
        dti = x["decoder_time_idx"].detach().cpu().numpy().astype(int)

        series_ids.append(g)
        decoder_time_idxs.append(dti)

    series_ids = np.concatenate(series_ids, axis=0)
    decoder_time_idxs = np.concatenate(decoder_time_idxs, axis=0)

    if len(series_ids) != n_samples:
        raise ValueError(f"Sample count mismatch: mapping has {len(series_ids)}, preds has {n_samples}")

    # Expand predictions to long format (can be big but manageable here)
    # Each row: sample_idx, step, series_id, time_idx_pred, time_to_onset_pred
    rows = []
    for i in range(n_samples):
        sid = series_ids[i]
        for k in range(pred_len):
            rows.append((sid, int(decoder_time_idxs[i, k]), float(preds[i, k])))

    pred_long = pd.DataFrame(rows, columns=["series_id", "time_idx", "time_to_onset_pred"])
    pred_long.sort_values(["series_id", "time_idx"], inplace=True)

    # True initiation time per series (constant)
    true_onset = (
        df.groupby("series_id", as_index=False)
          .agg(t_onset_true=("t_init_year", "first"))
    )

    # Convert predicted onset: first time_idx where predicted time_to_onset <= 0
    # If never <=0 in predicted windows, leave as NaN
    hit = pred_long["time_to_onset_pred"] <= 0.0
    pred_long["hit"] = hit.astype(int)

    first_hit = (
        pred_long[pred_long["hit"] == 1]
        .groupby("series_id", as_index=False)
        .agg(time_idx_pred_onset=("time_idx", "min"))
    )

    # Map time_idx -> t_year (using df lookup table)
    # create a mapping from (series_id, time_idx) to t_year (from df)
    # use drop_duplicates for efficiency
    time_map = df[["series_id", "time_idx", "t_year"]].drop_duplicates()

    first_hit = first_hit.merge(time_map, left_on=["series_id", "time_idx_pred_onset"],
                                right_on=["series_id", "time_idx"], how="left")
    first_hit.rename(columns={"t_year": "t_onset_pred"}, inplace=True)
    first_hit.drop(columns=["time_idx"], inplace=True)

    # merge all
    comp = true_onset.merge(first_hit[["series_id", "t_onset_pred"]], on="series_id", how="left")

    # metrics (only where prediction exists)
    valid = comp["t_onset_pred"].notna() & comp["t_onset_true"].notna()
    mae = np.mean(np.abs(comp.loc[valid, "t_onset_pred"] - comp.loc[valid, "t_onset_true"])) if valid.any() else np.nan
    rmse = np.sqrt(np.mean((comp.loc[valid, "t_onset_pred"] - comp.loc[valid, "t_onset_true"]) ** 2)) if valid.any() else np.nan

    out_dir = Path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    comp_path = out_dir / "onset_comparison.csv"
    comp.to_csv(comp_path, index=False)

    # optional save long predictions (can be large)
    long_path = out_dir / "onset_pred_long.parquet"
    pred_long.drop(columns=["hit"]).to_parquet(long_path, index=False)

    print("✅ Saved:", comp_path)
    print("✅ Saved:", long_path)
    print(f"MAE (years) = {mae:.4f}")
    print(f"RMSE (years) = {rmse:.4f}")
    print("Prediction coverage:", float(comp["t_onset_pred"].notna().mean()) * 100, "%")


if __name__ == "__main__":
    main()
