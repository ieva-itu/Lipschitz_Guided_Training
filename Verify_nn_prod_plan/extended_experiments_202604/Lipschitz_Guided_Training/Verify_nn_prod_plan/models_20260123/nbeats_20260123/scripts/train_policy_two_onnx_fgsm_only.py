#!/usr/bin/env python3
"""
scripts/train_policy_two_onnx_fgsm_only.py

Train a single FGSM-only policy:
  - no spectral normalization
  - robustness loss enabled
  - FGSM enabled
  - random robustness samples disabled

Output:
  models/policy_fgsm_only/
    policy.onnx
    policy_two_copy.onnx
    lipschitz.json

Also writes/refreshes:
  models/scaling.json
  models/episodes_seed.pkl

Designed to match the existing project layout and Marabou workflow.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only

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


def load_forecaster_and_series(kind: str, workspace: str):
    from src import forecast_model as fm
    if kind == "nhits":
        return fm.load_trained_nhits(workspace)
    if kind == "nbeats":
        return fm.load_trained_nbeats(workspace)
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
    preferred = {"nhits": "NHITS", "nbeats": "NBEATS"}[kind]
    if preferred in pred_df.columns:
        return preferred
    banned = {"unique_id", "ds"}
    candidates = [c for c in pred_df.columns if c not in banned]
    if not candidates:
        raise RuntimeError(f"Could not infer forecast column from predict() output: {list(pred_df.columns)}")
    return candidates[-1]


def compute_scales(series_df, nf, k: int, kind: str, uid: str):
    y = series_df["y"].to_numpy(dtype=np.float32)
    y = np.clip(y, 0.0, None)

    i_scale_raw = float(np.percentile(y, 99)) + 10.0
    i_scale_raw = _safe_float(i_scale_raw, default=max(float(y.max()) if y.size else 1.0, 1.0))

    pred_df = nf.predict(df=series_df)
    col = _infer_forecast_col(pred_df, kind)

    uid_df = pred_df[pred_df["unique_id"] == uid]
    last_forecast = uid_df.tail(k)[col].to_numpy(dtype=np.float32)
    last_forecast = np.clip(last_forecast, 0.0, None)

    f_max = float(last_forecast.max()) if last_forecast.size else 1.0
    f_max = _safe_float(f_max, default=1.0)

    f_scale_raw = f_max / 2.0
    f_scale_raw = _safe_float(f_scale_raw, default=1.0)

    return float(i_scale_raw), float(f_scale_raw), last_forecast


def build_forecast_features(series_df, nf, k: int, kind: str, uid: str, f_scale_raw: float):
    pred_df = nf.predict(df=series_df)
    col = _infer_forecast_col(pred_df, kind)

    uid_df = pred_df[pred_df["unique_id"] == uid]
    last_forecast = uid_df.tail(k)[col].to_numpy(dtype=np.float32)
    last_forecast = np.clip(last_forecast, 0.0, None)

    last_forecast_scaled = last_forecast / float(f_scale_raw)
    last_forecast_scaled = np.clip(last_forecast_scaled, 0.0, 2.0)

    T = len(series_df)
    return np.tile(last_forecast_scaled.reshape(1, -1), (T, 1))


def build_random_episodes(series_df, fore_matrix, episode_len: int, num_episodes: int, seed: int, i_scale_raw: float):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", choices=["nhits", "nbeats"], required=True)
    ap.add_argument("--forecast-workspace", default=None)

    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-dim", type=int, default=20)

    ap.add_argument("--episode-len", type=int, default=28)
    ap.add_argument("--num-episodes", type=int, default=200)

    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--output-scale", type=float, default=0.5)

    ap.add_argument("--robust-weight", type=float, default=50.0)
    ap.add_argument("--pert-radius", type=float, default=1.0)
    ap.add_argument("--eps-q", type=float, default=0.1)
    ap.add_argument("--perturb-forecast-only", action="store_true", default=True)
    args = ap.parse_args()

    root = Path(".")
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    forecast_workspace = args.forecast_workspace
    if forecast_workspace is None:
        forecast_workspace = str(models_dir / args.forecast)

    nf, series_df = load_forecaster_and_series(args.forecast, forecast_workspace)

    uid = FORCED_UID
    if "unique_id" in series_df.columns and not (series_df["unique_id"] == uid).any():
        uid = str(series_df["unique_id"].iloc[0])

    i_scale_raw, f_scale_raw, last_forecast_raw = compute_scales(
        series_df=series_df, nf=nf, k=args.k, kind=args.forecast, uid=uid
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

    fore_matrix = build_forecast_features(
        series_df=series_df, nf=nf, k=args.k, kind=args.forecast, uid=uid, f_scale_raw=f_scale_raw
    )

    episodes = build_random_episodes(
        series_df=series_df,
        fore_matrix=fore_matrix,
        episode_len=args.episode_len,
        num_episodes=args.num_episodes,
        seed=args.seed,
        i_scale_raw=i_scale_raw,
    )

    (models_dir / "episodes_seed.pkl").write_bytes(pickle.dumps({
        "seed": args.seed,
        "episodes": episodes,
        "k": args.k,
        "uid": uid,
        "forecast": args.forecast,
    }))
    print(f"[ok] wrote episodes -> {models_dir / 'episodes_seed.pkl'}")

    out_dir = models_dir / "policy_fgsm_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    input_dim = 1 + args.k

    policy = OrderingPolicy(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        use_spectral_norm=False,     # FGSM-only: no SN
        output_scale=args.output_scale,
    ).to(device)

    robust_cfg = RobustConfig(
        robust_weight=args.robust_weight,
        robust_num_samples=0,        # FGSM-only: disable random perturbations
        pert_radius=args.pert_radius,
        eps_q=args.eps_q,
        robust_use_fgsm=True,        # FGSM-only: enable FGSM
        perturb_forecast_only=args.perturb_forecast_only,
    )

    train_policy(
        policy,
        episodes,
        lr=1e-3,
        epochs=args.epochs,
        log_interval=10,
        Q_min=0.0,
        Q_max=100.0,
        robust_cfg=robust_cfg,
        seed=args.seed,
    )

    export_policy_onnx(policy, input_dim=input_dim, path=str(out_dir / "policy.onnx"))
    export_two_copy_onnx_noslice(policy, k=args.k, path=str(out_dir / "policy_two_copy.onnx"))

    plain = policy  # no SN used, so plain == policy
    L_hat = estimate_lipschitz_upper_bound_plain(plain)

    write_lipschitz_metadata(
        out_dir=out_dir,
        L_hat=L_hat,
        k=args.k,
        extra={
            "forecast_model": args.forecast,
            "tag": "fgsm_only",
            "seed": args.seed,
            "hidden_dim": args.hidden_dim,
            "use_spectral_norm_train": False,
            "output_scale": float(args.output_scale),
            "pert_radius": float(args.pert_radius),
            "eps_q": float(args.eps_q),
            "robust_weight": float(args.robust_weight),
            "robust_num_samples": 0,
            "robust_use_fgsm": True,
            "perturb_forecast_only": bool(args.perturb_forecast_only),
            "epochs": int(args.epochs),
        },
    )

    print(f"[done] trained FGSM-only policy -> {out_dir}")


if __name__ == "__main__":
    main()
