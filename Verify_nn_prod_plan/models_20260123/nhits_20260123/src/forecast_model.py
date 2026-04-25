# src/forecast_model.py
"""
Unified forecaster training/loading with persistence for:
  - NHITS
  - NBEATS

Project invariant:
  - Demand is clipped to >= 0 before training (demand can't be negative)
  - Scaling into the Marabou box ([0,1] for I, [0,2] for f) is NOT done here.
    It is done in scripts/train_policy_two_onnx_both.py and recorded in models/scaling.json.

Saves per workspace:
  - NeuralForecast bundle to <workspace>/
  - series.parquet to <workspace>/series.parquet
"""

from pathlib import Path
import pandas as pd
import numpy as np

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE

from .data_prep import load_single_series

FORCED_ITEM_ID = "85123A"


def _to_nf_df(series_df: pd.DataFrame, uid: str) -> pd.DataFrame:
    """
    Accepts either columns (date, qty) or (ds, y) or (ds, qty) etc.
    Returns NeuralForecast format: unique_id, ds, y
    """
    df = series_df.copy()

    if "ds" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "ds"})
        else:
            raise RuntimeError(f"Missing 'ds'/'date' in series_df columns: {list(df.columns)}")

    if "y" not in df.columns:
        if "qty" in df.columns:
            df = df.rename(columns={"qty": "y"})
        else:
            raise RuntimeError(f"Missing 'y'/'qty' in series_df columns: {list(df.columns)}")

    # Clip demand to nonnegative
    y = df["y"].to_numpy(dtype=np.float32)
    df["y"] = np.clip(y, 0.0, None)

    df["unique_id"] = uid
    df = df[["unique_id", "ds", "y"]]
    return df


def train_nhits(
    csv_path: str,
    quantity_col: str = "Quantity",
    horizon: int = 14,
    freq: str = "D",
    max_steps: int = 2000,
    workspace: str = "models/nhits",
):
    from neuralforecast.models import NHITS

    series_df = load_single_series(
        csv_path=csv_path,
        item_col="StockCode",
        item_id=FORCED_ITEM_ID,
        qty_col=quantity_col,
        date_col="InvoiceDate",
    )

    nf_df = _to_nf_df(series_df, uid=FORCED_ITEM_ID)

    input_size = 4 * horizon
    model = NHITS(
        h=horizon,
        input_size=input_size,
        loss=MAE(),
        stack_types=["identity", "identity", "identity"],
        n_blocks=[1, 1, 1],
        mlp_units=[[256, 256], [256, 256], [256, 256]],
        n_pool_kernel_size=[4, 4, 1],
        n_freq_downsample=[4, 2, 1],
        pooling_mode="MaxPool1d",
        interpolation_mode="linear",
        dropout_prob_theta=0.1,
        learning_rate=1e-3,
        max_steps=max_steps,
        batch_size=64,
        windows_batch_size=256,
        step_size=1,
    )

    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=nf_df)

    w = Path(workspace)
    w.mkdir(parents=True, exist_ok=True)
    nf.save(path=str(w), overwrite=True, save_dataset=False)
    nf_df.to_parquet(w / "series.parquet", index=False)

    fcst_df = nf.predict(df=nf_df).reset_index(drop=True)
    return nf, fcst_df, nf_df


def train_nbeats(
    csv_path: str,
    quantity_col: str = "Quantity",
    horizon: int = 14,
    freq: str = "D",
    max_steps: int = 2000,
    workspace: str = "models/nbeats",
):
    from neuralforecast.models import NBEATS

    series_df = load_single_series(
        csv_path=csv_path,
        item_col="StockCode",
        item_id=FORCED_ITEM_ID,
        qty_col=quantity_col,
        date_col="InvoiceDate",
    )

    nf_df = _to_nf_df(series_df, uid=FORCED_ITEM_ID)

    input_size = 4 * horizon
    model = NBEATS(
        h=horizon,
        input_size=input_size,
        loss=MAE(),
        max_steps=max_steps,
        learning_rate=1e-3,
        batch_size=64,
        windows_batch_size=256,
        stack_types=["trend", "seasonality"],
        n_blocks=[3, 3],
        mlp_units=[[256, 256], [256, 256]],
    )

    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=nf_df)

    w = Path(workspace)
    w.mkdir(parents=True, exist_ok=True)
    nf.save(path=str(w), overwrite=True, save_dataset=False)
    nf_df.to_parquet(w / "series.parquet", index=False)

    fcst_df = nf.predict(df=nf_df).reset_index(drop=True)
    return nf, fcst_df, nf_df


def train_tft(
    csv_path: str,
    quantity_col: str = "Quantity",
    horizon: int = 14,
    freq: str = "D",
    max_steps: int = 2000,
    workspace: str = "models/tft",
):
    from neuralforecast.models import TFT

    series_df = load_single_series(
        csv_path=csv_path,
        item_col="StockCode",
        item_id=FORCED_ITEM_ID,
        qty_col=quantity_col,
        date_col="InvoiceDate",
    )

    nf_df = _to_nf_df(series_df, uid=FORCED_ITEM_ID)

    input_size = 4 * horizon
    model = TFT(
        h=horizon,
        input_size=input_size,
        loss=MAE(),
        max_steps=max_steps,
        learning_rate=1e-3,
        batch_size=64,
        windows_batch_size=256,
        hidden_size=64,
        n_head=4,
        dropout=0.1,
    )

    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(df=nf_df)

    w = Path(workspace)
    w.mkdir(parents=True, exist_ok=True)
    nf.save(path=str(w), overwrite=True, save_dataset=False)
    nf_df.to_parquet(w / "series.parquet", index=False)

    fcst_df = nf.predict(df=nf_df).reset_index(drop=True)
    return nf, fcst_df, nf_df


def load_trained_nhits(workspace: str = "models/nhits"):
    w = Path(workspace)
    nf = NeuralForecast.load(path=str(w))
    series_df = pd.read_parquet(w / "series.parquet")
    return nf, series_df


def load_trained_nbeats(workspace: str = "models/nbeats"):
    w = Path(workspace)
    nf = NeuralForecast.load(path=str(w))
    series_df = pd.read_parquet(w / "series.parquet")
    return nf, series_df


def load_trained_tft(workspace: str = "models/tft"):
    w = Path(workspace)
    nf = NeuralForecast.load(path=str(w))
    series_df = pd.read_parquet(w / "series.parquet")
    return nf, series_df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["nhits", "nbeats", "tft"], required=True)
    ap.add_argument("--csv", default="data/uci_online_retail_clean.csv")
    ap.add_argument("--quantity-col", default="Quantity")
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--freq", default="D")
    ap.add_argument("--workspace", default=None)
    args = ap.parse_args()

    if args.workspace is None:
        args.workspace = f"models/{args.kind}"

    if args.kind == "nhits":
        nf, fcst_df, nf_df = train_nhits(args.csv, args.quantity_col, args.horizon, args.freq, args.max_steps, args.workspace)
    elif args.kind == "nbeats":
        nf, fcst_df, nf_df = train_nbeats(args.csv, args.quantity_col, args.horizon, args.freq, args.max_steps, args.workspace)
    else:
        nf, fcst_df, nf_df = train_tft(args.csv, args.quantity_col, args.horizon, args.freq, args.max_steps, args.workspace)

    print(f"[ok] saved {args.kind} to {args.workspace}")
    print(f"[info] FORCED_ITEM_ID={FORCED_ITEM_ID}  series length={len(nf_df)}")
    print(fcst_df.head())

