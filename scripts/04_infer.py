"""
Run TFT inference and reconstruct population-level corrosion initiation Pf(t).

This script implements Step 4: rolling inference for the trained TFT onset
model. It streams predictions over the labeled scenario space and aggregates
them into the manuscript quantity Pf(t).

Generated files:
- outputs/predictions/pf_full_true_vs_pred.csv
- outputs/predictions/onset_flag_pred_point.parquet
- outputs/predictions/series_static.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet


ROOT = Path(__file__).resolve().parents[1]
MODEL_STATIC_REALS = ["Cs", "cover_mm", "D28", "m_aging", "C_th"]
MODEL_TIME_VARYING_KNOWN_REALS = ["time_idx", "t_year"]
MODEL_TIME_VARYING_UNKNOWN_REALS: list[str] = []
TARGET_COLUMN = "onset_flag"
FORBIDDEN_INPUTS = {"chloride_rebar", "target_onset", "onset_flag", "binary_label", "onset_raw", "time_to_onset", "t_init_year"}


def resolve_checkpoint(checkpoint: Path | None, checkpoint_dir: Path) -> Path:
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    pointer = checkpoint_dir / "best_checkpoint.txt"
    if pointer.exists():
        path = Path(pointer.read_text(encoding="utf-8").strip())
        if path.exists():
            return path

    candidates = sorted(checkpoint_dir.glob("tft_onset_flag-*.ckpt"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        "No checkpoint found. Run scripts/03_train_model.py first or pass --checkpoint."
    )


def ensure_cover_mm(df: pd.DataFrame) -> pd.DataFrame:
    if "cover_mm" in df.columns:
        return df
    if "cover_m" in df.columns:
        df = df.copy()
        df["cover_mm"] = df["cover_m"].astype(float) * 1000.0
        return df
    raise KeyError("Neither 'cover_mm' nor 'cover_m' exists in the dataset.")


def validate_model_inputs() -> None:
    model_inputs = (
        set(MODEL_STATIC_REALS)
        | set(MODEL_TIME_VARYING_KNOWN_REALS)
        | set(MODEL_TIME_VARYING_UNKNOWN_REALS)
    )
    leaked_inputs = sorted(model_inputs & FORBIDDEN_INPUTS)
    if leaked_inputs:
        raise ValueError(f"Forbidden leakage-prone model inputs configured: {leaked_inputs}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer TFT onset probabilities and compute Pf(t).")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "chloride_labeled.parquet")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "outputs" / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "predictions")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-encoder-length", type=int, default=52)
    parser.add_argument("--max-prediction-length", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    validate_model_inputs()
    checkpoint = resolve_checkpoint(args.checkpoint, args.checkpoint_dir)

    df = pd.read_parquet(args.data).copy()
    df = ensure_cover_mm(df)
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Corrected cumulative target column missing: {TARGET_COLUMN}")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    base_ds = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=TARGET_COLUMN,
        group_ids=["series_id"],
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
        time_varying_known_reals=MODEL_TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=MODEL_TIME_VARYING_UNKNOWN_REALS,
        static_reals=MODEL_STATIC_REALS,
        add_relative_time_idx=True,
        add_encoder_length=True,
    )
    rolling_ds = TimeSeriesDataSet.from_dataset(base_ds, df, stop_randomization=True)
    dl = rolling_ds.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)

    model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint), weights_only=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sum_map: dict[tuple[int, int], float] = {}
    cnt_map: dict[tuple[int, int], int] = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            x, _ = batch
            x_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
            logits = model(x_dev)["prediction"]
            p1 = torch.softmax(logits, dim=-1)[..., 1].detach().cpu().numpy()

            series_ids = x["groups"][:, 0].detach().cpu().numpy().astype(int)
            decoder_time_idxs = x["decoder_time_idx"].detach().cpu().numpy().astype(int)

            for b in range(p1.shape[0]):
                sid = int(series_ids[b])
                for j in range(p1.shape[1]):
                    key = (sid, int(decoder_time_idxs[b, j]))
                    sum_map[key] = sum_map.get(key, 0.0) + float(p1[b, j])
                    cnt_map[key] = cnt_map.get(key, 0) + 1

            if (batch_idx + 1) % 50 == 0:
                print(f"Processed batches: {batch_idx + 1}")

    keys = list(sum_map.keys())
    pred_point = pd.DataFrame(
        {
            "series_id": np.fromiter((k[0] for k in keys), dtype=np.int32),
            "time_idx": np.fromiter((k[1] for k in keys), dtype=np.int32),
            "p_onset1_pred": np.fromiter((sum_map[k] / max(cnt_map[k], 1) for k in keys), dtype=np.float64),
            "n_votes": np.fromiter((cnt_map[k] for k in keys), dtype=np.int32),
        }
    )

    time_map = df[["series_id", "time_idx", "t_year", TARGET_COLUMN]].drop_duplicates()
    pred_point = pred_point.merge(time_map, on=["series_id", "time_idx"], how="left")

    pf_pred = pred_point.groupby("t_year", as_index=False)["p_onset1_pred"].mean()
    pf_pred.rename(columns={"p_onset1_pred": "Pf_pred"}, inplace=True)
    pf_true = df.groupby("t_year", as_index=False)[TARGET_COLUMN].mean()
    pf_true.rename(columns={TARGET_COLUMN: "Pf_true"}, inplace=True)
    pf = pf_true.merge(pf_pred, on="t_year", how="inner").sort_values("t_year").reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pf_path = args.output_dir / "pf_full_true_vs_pred.csv"
    point_path = args.output_dir / "onset_flag_pred_point.parquet"
    static_path = args.output_dir / "series_static.csv"

    pf.to_csv(pf_path, index=False)
    df[["series_id", "cover_mm"]].drop_duplicates().to_csv(static_path, index=False)
    pred_point.to_parquet(point_path, index=False)

    mae_pf = float(np.mean(np.abs(pf["Pf_pred"] - pf["Pf_true"])))
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Saved Pf(t) table: {pf_path}")
    print(f"Saved series static map: {static_path}")
    print(f"Saved point predictions: {point_path}")
    print(f"MAE on full Pf(t) curve = {mae_pf:.4f} ({mae_pf * 100:.2f} percentage points)")
    print(f"[Timing] Inference: {time.perf_counter() - start_time:.4f} s")


if __name__ == "__main__":
    main()
