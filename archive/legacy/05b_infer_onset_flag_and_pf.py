"""
05b_infer_onset_flag_and_pf.py

Infer onset-flag probabilities with trained TFT classifier and compute Pf(t).

Inputs:
- data/processed/chloride_labeled.parquet
- outputs/checkpoints/tft_onset_flag-epoch=9-val_loss=0.0223.ckpt

Outputs:
- outputs/predictions/pf_true_vs_pred.csv
- outputs/predictions/onset_flag_pred_long.parquet (optional, large)
- terminal: key Pf values at selected years + simple metrics
"""

from pathlib import Path
import numpy as np
import pandas as pd

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer  # not used but safe to keep


def main():
    ckpt_path = Path(r"outputs/checkpoints/tft_onset_flag-epoch=9-val_loss=0.0223.ckpt")
    data_path = Path("data/processed/chloride_labeled.parquet")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_parquet(data_path).copy()
    df["target_onset"] = df["target_onset"].astype(int)

    max_encoder_length = 52
    max_prediction_length = 13
    training_cutoff = df["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target_onset",
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx", "t_year"],
        time_varying_unknown_reals=["chloride_rebar"],
        static_reals=["Cs", "cover_m", "D28", "m_aging", "C_th"],
        add_relative_time_idx=True,
        add_encoder_length=True,
    )

    # predict on a safe window (same approach as before)
    pred_df = df[df.time_idx > training_cutoff - max_encoder_length - max_prediction_length]
    predict_ds = TimeSeriesDataSet.from_dataset(training, pred_df, predict=True, stop_randomization=True)
    dl = predict_ds.to_dataloader(train=False, batch_size=256, num_workers=0)

    # ✅ PyTorch 2.6+ safe-load fix: you trust your own ckpt
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path), weights_only=False)

    # Get raw network output (logits) for 2 classes -> then softmax for P(class=1)
    # output shape: (n_samples, pred_len, 2)
    logits = model.predict(dl, mode="raw", return_x=False)["prediction"].detach().cpu().numpy()

    # softmax to probabilities
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    p_onset1 = probs[..., 1]  # (n_samples, pred_len)

    n_samples, pred_len = p_onset1.shape
    assert pred_len == max_prediction_length

    # Build mapping sample -> (series_id, decoder_time_idx, decoder_t_year)
    series_ids = []
    decoder_time_idxs = []
    decoder_t_years = []

    for x, y in dl:
        g = x["groups"][:, 0].detach().cpu().numpy().astype(int)
        dti = x["decoder_time_idx"].detach().cpu().numpy().astype(int)
        # decoder time is in idx; map to t_year via provided known real (decoder_cont includes t_year)
        # easiest robust way: later merge with df mapping, so here only keep time_idx
        series_ids.append(g)
        decoder_time_idxs.append(dti)

    series_ids = np.concatenate(series_ids, axis=0)
    decoder_time_idxs = np.concatenate(decoder_time_idxs, axis=0)

    if len(series_ids) != n_samples:
        raise ValueError(f"Sample mapping mismatch: {len(series_ids)} vs {n_samples}")

    # Expand predicted probabilities to long table
    rows = []
    for i in range(n_samples):
        sid = series_ids[i]
        for k in range(pred_len):
            rows.append((sid, int(decoder_time_idxs[i, k]), float(p_onset1[i, k])))

    pred_long = pd.DataFrame(rows, columns=["series_id", "time_idx", "p_onset1_pred"])
    pred_long.sort_values(["series_id", "time_idx"], inplace=True)

    # Map time_idx -> t_year
    time_map = df[["series_id", "time_idx", "t_year", "target_onset"]].drop_duplicates()
    pred_long = pred_long.merge(time_map, on=["series_id", "time_idx"], how="left")

    # Compute Pf(t) as mean across series at each t_year
    # Predicted Pf: mean of p_onset1_pred
    pf_pred = pred_long.groupby("t_year", as_index=False)["p_onset1_pred"].mean()
    pf_pred.rename(columns={"p_onset1_pred": "Pf_pred"}, inplace=True)

    # True Pf: mean of target_onset (0/1)
    pf_true = pred_long.groupby("t_year", as_index=False)["target_onset"].mean()
    pf_true.rename(columns={"target_onset": "Pf_true"}, inplace=True)

    pf = pf_true.merge(pf_pred, on="t_year", how="inner").sort_values("t_year")

    out_dir = Path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    pf_path = out_dir / "pf_true_vs_pred.csv"
    pf.to_csv(pf_path, index=False)

    # optional (large)
    long_path = out_dir / "onset_flag_pred_long.parquet"
    pred_long.to_parquet(long_path, index=False)

    print("✅ Saved Pf table:", pf_path)
    print("✅ Saved long predictions:", long_path)
    print("Pf rows:", len(pf))

    # Print key checkpoints (paper-friendly)
    for yr in [5, 10, 15, 20, 25, 30, 40, 60]:
        # nearest year in pf (t_year is continuous depending on your step)
        idx = (pf["t_year"] - yr).abs().idxmin()
        t = pf.loc[idx, "t_year"]
        print(f"t={t:6.2f} yr | Pf_true={pf.loc[idx,'Pf_true']*100:6.2f}% | Pf_pred={pf.loc[idx,'Pf_pred']*100:6.2f}%")

    # Simple overall error on Pf curve
    mae_pf = float(np.mean(np.abs(pf["Pf_pred"] - pf["Pf_true"])))
    print(f"\nMAE on Pf curve (absolute prob) = {mae_pf:.4f}  (~{mae_pf*100:.2f}%)")


if __name__ == "__main__":
    main()
