#!/usr/bin/env python3
"""
scripts/train_policy_two_onnx_both.py

Train BOTH:
  (1) baseline policy (no SN, no robust loss)  -> models/policy_baseline/
  (2) robust policy   (SN + robust loss)       -> models/policy_robust/

Key invariant (matches verification box):
  - Inventory input to policy: I_scaled in [0,1]
  - Forecast input to policy:  f_scaled in [0,2]
  - Forecast negatives clipped to 0

This script writes a single source of truth:
  models/scaling.json
with the same schema across NHITS/NBEATS.

Assumes your local src/forecast_model.py provides:
  load_trained_nhits(workspace)
  load_trained_nbeats(workspace)
and that series.parquet has columns: unique_id, ds, y
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only (keep deterministic)

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from src.policy_model_two_onnx import (
    OrderingPolicy, RobustConfig, train_policy,
    export_policy_onnx, export_two_copy_onnx_noslice,
    make_plain_from_sn, estimate_lipschitz_upper_bound_plain,
    write_lipschitz_metadata,
)

FORCED_UID = "85123A"


# -------------------------
# Forecast loading helpers
# -------------------------
def load_forecaster_and_series(kind: str, workspace: str):
    """
    Returns (nf, series_df) where nf is NeuralForecast and series_df is the saved parquet.
    """
    from src import forecast_model as fm

    if kind == "nhits":
        return fm.load_trained_nhits(workspace)
    if kind == "nbeats":
        return fm.load_trained_nbeats(workspace)
    if kind == "tft":
        return fm.load_trained_tft(workspace)

    raise ValueError(f"Unknown forecast kind: {kind}")


def _safe_float(x: float, default: float = 1.0) -> float:
    try:
        x = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(x) or x <= 0.0:
        return float(default)
    return float(x)


def _infer_forecast_col(pred_df, kind: str):
    """
    NeuralForecast predict() typically returns a column named like the model class,
    e.g. 'NHITS', 'NBEATS'.
    """
    preferred = {"nhits": "NHITS", "nbeats": "NBEATS", "tft": "TFT"}[kind]
    if preferred in pred_df.columns:
        return preferred
    # fallback: choose the last non-index column that isn't unique_id/ds
    banned = {"unique_id", "ds"}
    candidates = [c for c in pred_df.columns if c not in banned]
    if not candidates:
        raise RuntimeError(f"Could not infer forecast column from predict() output: {list(pred_df.columns)}")
    return candidates[-1]


# -------------------------
# Scaling computation
# -------------------------
def compute_scales(series_df, nf, k: int, kind: str, uid: str, seed: int):
    """
    Compute scaling constants so that:
      - I_scaled = I_raw / i_scale_raw  clipped to [0,1]
      - f_scaled = f_raw / f_scale_raw  clipped to [0,2]

    We choose:
      - f_scale_raw = max(last_forecast_raw_clipped_nonneg) / 2
        so the max of the last k-step forecast maps to ~2.0 before clipping.
      - i_scale_raw from robust demand envelope (p99 + margin), conservative for I0.
    """
    # Demand series (raw, nonnegative)
    y = series_df["y"].to_numpy(dtype=np.float32)
    y = np.clip(y, 0.0, None)

    # inventory scale: conservative envelope
    i_scale_raw = float(np.percentile(y, 99)) + 10.0
    i_scale_raw = _safe_float(i_scale_raw, default=max(float(y.max()) if y.size else 1.0, 1.0))

    # model forecast scale: based on model prediction on saved series
    pred_df = nf.predict(df=series_df)
    col = _infer_forecast_col(pred_df, kind)

    uid_df = pred_df[pred_df["unique_id"] == uid]
    last_forecast = uid_df.tail(k)[col].to_numpy(dtype=np.float32)
    last_forecast = np.clip(last_forecast, 0.0, None)

    f_max = float(last_forecast.max()) if last_forecast.size else 1.0
    f_max = _safe_float(f_max, default=1.0)

    # map max forecast -> 2.0
    f_scale_raw = f_max / 2.0
    f_scale_raw = _safe_float(f_scale_raw, default=1.0)

    return float(i_scale_raw), float(f_scale_raw), last_forecast


def build_forecast_features(series_df, nf, k: int, kind: str, uid: str, f_scale_raw: float):
    """
    Build a forecast feature matrix [T,k] used for policy training episodes.
    We keep it simple/deterministic: tile the last k-step forecast across time.

    Transform:
      f_raw -> clip>=0 -> divide by f_scale_raw -> clip to [0,2]
    """
    pred_df = nf.predict(df=series_df)
    col = _infer_forecast_col(pred_df, kind)

    uid_df = pred_df[pred_df["unique_id"] == uid]
    last_forecast = uid_df.tail(k)[col].to_numpy(dtype=np.float32)
    last_forecast = np.clip(last_forecast, 0.0, None)

    last_forecast_scaled = last_forecast / float(f_scale_raw)
    last_forecast_scaled = np.clip(last_forecast_scaled, 0.0, 2.0)

    T = len(series_df)
    return np.tile(last_forecast_scaled.reshape(1, -1), (T, 1))


def build_random_episodes(series_df, fore_matrix, episode_len: int, num_episodes: int,
                         seed: int, i_scale_raw: float):
    """
    Episodes with inputs scaled to match the verification box:
      - I_scaled in [0,1]
      - f_scaled in [0,2]  (already)
    """
    rng = np.random.default_rng(seed)

    y = series_df["y"].to_numpy(dtype=np.float32)
    y = np.clip(y, 0.0, None)

    T = len(y)
    episodes = []

    for _ in range(num_episodes):
        start = 0 if T <= episode_len else int(rng.integers(0, T - episode_len))
        end = start + episode_len

        demand_seg = y[start:end]
        fore_seg = fore_matrix[start:end, :]

        recent_hist = y[max(0, start - 7):start]
        I0_raw = float(np.mean(recent_hist) + 5.0) if len(recent_hist) else float(np.mean(y[:7]) + 5.0)

        I0_scaled = float(I0_raw) / float(i_scale_raw)
        I0_scaled = float(np.clip(I0_scaled, 0.0, 1.0))

        episodes.append((I0_scaled, demand_seg, fore_seg))

    return episodes


# -------------------------
# Train + export
# -------------------------
def train_and_export(out_dir: Path, episodes, k: int, hidden_dim: int,
                     use_sn: bool, output_scale: float,
                     robust_cfg: RobustConfig, epochs: int, seed: int,
                     tag: str, forecast_kind: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    input_dim = 1 + k

    policy = OrderingPolicy(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        use_spectral_norm=use_sn,
        output_scale=output_scale,
    ).to(device)

    train_policy(
        policy,
        episodes,
        lr=1e-3,
        epochs=epochs,
        log_interval=10,
        Q_min=0.0,
        Q_max=100.0,
        robust_cfg=robust_cfg,
        seed=seed,
    )

    export_policy_onnx(policy, input_dim=input_dim, path=str(out_dir / "policy.onnx"))
    export_two_copy_onnx_noslice(policy, k=k, path=str(out_dir / "policy_two_copy.onnx"))

    # Lipschitz metadata
    plain = make_plain_from_sn(policy) if use_sn else policy
    L_hat = estimate_lipschitz_upper_bound_plain(plain)

    write_lipschitz_metadata(
        out_dir=out_dir,
        L_hat=L_hat,
        k=k,
        extra={
            "forecast_model": forecast_kind,
            "tag": tag,
            "seed": seed,
            "hidden_dim": hidden_dim,
            "use_spectral_norm_train": bool(use_sn),
            "output_scale": float(output_scale),
            "pert_radius": float(robust_cfg.pert_radius),
            "eps_q": float(robust_cfg.eps_q),
            "robust_weight": float(robust_cfg.robust_weight),
            "robust_num_samples": int(robust_cfg.robust_num_samples),
            "robust_use_fgsm": bool(robust_cfg.robust_use_fgsm),
            "perturb_forecast_only": bool(robust_cfg.perturb_forecast_only),
            "epochs": int(epochs),
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", choices=["nhits", "nbeats", "tft"], required=True)
    ap.add_argument("--forecast-workspace", default=None)

    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-dim", type=int, default=20)

    ap.add_argument("--episode-len", type=int, default=28)
    ap.add_argument("--num-episodes", type=int, default=200)

    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--output-scale", type=float, default=0.5)

    # robust training knobs
    ap.add_argument("--robust-weight", type=float, default=50.0)
    ap.add_argument("--robust-num-samples", type=int, default=10)
    ap.add_argument("--pert-radius", type=float, default=1.0)   # r / eps_f
    ap.add_argument("--eps-q", type=float, default=0.1)
    ap.add_argument("--robust-use-fgsm", action="store_true")
    ap.add_argument("--perturb-forecast-only", action="store_true", default=True)

    args = ap.parse_args()

    root = Path(".")
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load forecaster + saved series
    forecast_workspace = args.forecast_workspace
    if forecast_workspace is None:
        forecast_workspace = str(models_dir / args.forecast)

    nf, series_df = load_forecaster_and_series(args.forecast, forecast_workspace)

    # Force UID selection
    uid = FORCED_UID
    if "unique_id" in series_df.columns:
        if (series_df["unique_id"] == uid).any() is False:
            # fallback to first UID but still record it in scaling.json
            uid = str(series_df["unique_id"].iloc[0])

    # Compute deterministic scales and write scaling.json (SINGLE SOURCE OF TRUTH)
    i_scale_raw, f_scale_raw, last_forecast_raw = compute_scales(
        series_df=series_df, nf=nf, k=args.k, kind=args.forecast, uid=uid, seed=args.seed
    )

    scaling = {
        "uid": uid,
        "I_MIN": 0.0,
        "I_MAX": 1.0,
        "F_MIN": 0.0,
        "F_MAX": 2.0,
        "i_scale_raw": float(i_scale_raw),
        "f_scale_raw": float(f_scale_raw),
        "last_forecast_raw_clipped_nonneg": [float(x) for x in last_forecast_raw.tolist()],
        "forecast_kind": args.forecast,
        "k": int(args.k),
        "seed": int(args.seed),
    }
    (models_dir / "scaling.json").write_text(json.dumps(scaling, indent=2))
    print(f"[ok] wrote scaling.json -> {models_dir / 'scaling.json'}")

    # Prepare forecast features (scaled to [0,2])
    fore_matrix = build_forecast_features(
        series_df=series_df, nf=nf, k=args.k, kind=args.forecast, uid=uid, f_scale_raw=f_scale_raw
    )

    # Build episodes (inventory scaled to [0,1])
    episodes = build_random_episodes(
        series_df=series_df,
        fore_matrix=fore_matrix,
        episode_len=args.episode_len,
        num_episodes=args.num_episodes,
        seed=args.seed,
        i_scale_raw=i_scale_raw,
    )

    # Save episodes for reproducibility
    (models_dir / "episodes_seed.pkl").write_bytes(pickle.dumps({
        "seed": args.seed,
        "episodes": episodes,
        "k": args.k,
        "uid": uid,
        "forecast": args.forecast,
    }))
    print(f"[ok] wrote episodes -> {models_dir / 'episodes_seed.pkl'}")

    # Train baseline
    train_and_export(
        out_dir=models_dir / "policy_baseline",
        episodes=episodes,
        k=args.k,
        hidden_dim=args.hidden_dim,
        use_sn=False,
        output_scale=args.output_scale,
        robust_cfg=RobustConfig(
            robust_weight=0.0,
            robust_num_samples=0,
            pert_radius=args.pert_radius,
            eps_q=args.eps_q,
            robust_use_fgsm=False,
            perturb_forecast_only=True,
        ),
        epochs=args.epochs,
        seed=args.seed,
        tag="baseline",
        forecast_kind=args.forecast,
    )

    # Train robust (SN + robust loss)
    train_and_export(
        out_dir=models_dir / "policy_robust",
        episodes=episodes,
        k=args.k,
        hidden_dim=args.hidden_dim,
        use_sn=True,
        output_scale=args.output_scale,
        robust_cfg=RobustConfig(
            robust_weight=args.robust_weight,
            robust_num_samples=args.robust_num_samples,
            pert_radius=args.pert_radius,
            eps_q=args.eps_q,
            robust_use_fgsm=args.robust_use_fgsm,
            perturb_forecast_only=args.perturb_forecast_only,
        ),
        epochs=args.epochs,
        seed=args.seed,
        tag="robust",
        forecast_kind=args.forecast,
    )

    print("[done] trained baseline + robust policies.")


if __name__ == "__main__":
    main()

