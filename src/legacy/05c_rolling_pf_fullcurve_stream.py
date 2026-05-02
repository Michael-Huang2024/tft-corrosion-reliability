"""
05c_rolling_pf_fullcurve_stream.py

Memory-safe rolling Pf(t) reconstruction (0–60 yrs) using streaming inference.
Avoids model.predict(...) that stores all raw outputs in RAM.

Outputs:
- outputs/predictions/pf_full_true_vs_pred.csv
- outputs/predictions/onset_flag_pred_point.parquet   (optional, can disable)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer


def softmax_lastdim_torch(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def main():
    # ===== paths =====
    ckpt_path = Path(r"outputs/checkpoints/tft_onset_flag-epoch=9-val_loss=0.0223.ckpt")
    data_path = Path("data/processed/chloride_labeled.parquet")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    # ===== load data =====
    df = pd.read_parquet(data_path).copy()
    df["target_onset"] = df["target_onset"].astype(int)

    # ===== dataset config (must match training) =====
    max_encoder_length = 52
    max_prediction_length = 13

    base_ds = TimeSeriesDataSet(
        df,
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

    rolling_ds = TimeSeriesDataSet.from_dataset(base_ds, df, stop_randomization=True)
    dl = rolling_ds.to_dataloader(train=False, batch_size=128, num_workers=0)  # 128更稳

    # ===== load model =====
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path), weights_only=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ===== streaming accumulators =====
    # key = (series_id, time_idx) -> sum_p, count
    # Using python dict keeps memory manageable because unique points are limited.
    sum_map = {}
    cnt_map = {}

    # ===== run streaming inference =====
    with torch.no_grad():
        for i, batch in enumerate(dl):
            x, y = batch

            # move x tensors to device
            x_dev = {}
            for k, v in x.items():
                if torch.is_tensor(v):
                    x_dev[k] = v.to(device)
                else:
                    x_dev[k] = v

            # forward pass: model returns dict; "prediction" are logits
            out = model(x_dev)
            logits = out["prediction"]  # (B, pred_len, n_classes=2)

            probs = softmax_lastdim_torch(logits)
            p1 = probs[..., 1].detach().cpu().numpy()  # (B, pred_len)

            sid = x["groups"][:, 0].detach().cpu().numpy().astype(int)      # (B,)
            dti = x["decoder_time_idx"].detach().cpu().numpy().astype(int)  # (B, pred_len)

            B, L = p1.shape

            # accumulate per point
            # NOTE: using loops but still ok; batch=128 keeps it fast enough.
            for b in range(B):
                s = int(sid[b])
                for j in range(L):
                    t = int(dti[b, j])
                    key = (s, t)
                    sum_map[key] = sum_map.get(key, 0.0) + float(p1[b, j])
                    cnt_map[key] = cnt_map.get(key, 0) + 1

            if (i + 1) % 50 == 0:
                print(f"... processed batches: {i+1}")

    # ===== build point-level table (series_id, time_idx) =====
    keys = list(sum_map.keys())
    series_ids = np.fromiter((k[0] for k in keys), dtype=np.int32)
    time_idxs = np.fromiter((k[1] for k in keys), dtype=np.int32)
    sums = np.fromiter((sum_map[k] for k in keys), dtype=np.float64)
    cnts = np.fromiter((cnt_map[k] for k in keys), dtype=np.int32)

    pred_point = pd.DataFrame({
        "series_id": series_ids,
        "time_idx": time_idxs,
        "p_onset1_pred": sums / np.maximum(cnts, 1),
        "n_votes": cnts,
    })

    # attach t_year and true label
    time_map = df[["series_id", "time_idx", "t_year", "target_onset"]].drop_duplicates()
    pred_point = pred_point.merge(time_map, on=["series_id", "time_idx"], how="left")

    # Pf_pred(t)
    pf_pred = pred_point.groupby("t_year", as_index=False)["p_onset1_pred"].mean()
    pf_pred.rename(columns={"p_onset1_pred": "Pf_pred"}, inplace=True)

    # Pf_true(t)
    pf_true = df.groupby("t_year", as_index=False)["target_onset"].mean()
    pf_true.rename(columns={"target_onset": "Pf_true"}, inplace=True)

    pf = pf_true.merge(pf_pred, on="t_year", how="inner").sort_values("t_year").reset_index(drop=True)

    # ===== save =====
    out_pred = Path("outputs/predictions")
    out_pred.mkdir(parents=True, exist_ok=True)

    pf_path = out_pred / "pf_full_true_vs_pred.csv"
    point_path = out_pred / "onset_flag_pred_point.parquet"

    pf.to_csv(pf_path, index=False)

    # 如果你机器内存紧张，下一行可以先注释掉（点级表可能较大）
    pred_point.to_parquet(point_path, index=False)

    mae_pf = float(np.mean(np.abs(pf["Pf_pred"] - pf["Pf_true"])))
    print("✅ Saved full Pf table:", pf_path)
    print("✅ Saved point predictions:", point_path)
    print("Pf rows (full):", len(pf))
    print(f"MAE on full Pf curve (absolute prob) = {mae_pf:.4f} (~{mae_pf*100:.2f}%)")

    for yr in [0, 1, 5, 10, 20, 30, 40, 50, 60]:
        idx = (pf["t_year"] - yr).abs().idxmin()
        t = pf.loc[idx, "t_year"]
        print(
            f"t={t:6.2f} yr | Pf_true={pf.loc[idx,'Pf_true']*100:6.2f}% | "
            f"Pf_pred={pf.loc[idx,'Pf_pred']*100:6.2f}%"
        )


if __name__ == "__main__":
    main()
