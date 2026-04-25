#!/usr/bin/env python3
"""
proof_aligned_empirical_compare.py

Same proof-aligned comparison as before, but with path auto-resolution for the
user's actual nested project layout under:

  extended_experiments_202604/
    Lipschitz_Guided_Training/
      Verify_nn_prod_plan/
        models_20260123/
          nhits_20260123/
            models/scaling.json
          nbeats_20260123/
            models/scaling.json

You can pass either:
- absolute paths, or
- shorthand like:
    --models-dir nhits_20260123/models
    --results-root results_nhits
  as long as you run from extended_experiments_202604.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class Scaling:
    H: int
    I_MAX_SCALED: float
    F_MAX_SCALED: float
    f_scale: float

    @staticmethod
    def load(models_dir: Path) -> "Scaling":
        p = models_dir / "scaling.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing scaling.json at {p}")
        d = json.loads(p.read_text())

        H = int(d.get("HORIZON", d.get("horizon", d.get("k", 7))))
        I_MAX_SCALED = float(d.get("I_MAX_SCALED", d.get("I_MAX", 1.0)))
        F_MAX_SCALED = float(d.get("F_MAX_SCALED", d.get("F_MAX", 2.0)))

        if "f_scale" in d:
            f_scale = float(d["f_scale"])
        elif "f_scale_raw" in d:
            f_scale = 1.0 / float(d["f_scale_raw"])
        else:
            f_scale = 1.0

        return Scaling(
            H=H,
            I_MAX_SCALED=I_MAX_SCALED,
            F_MAX_SCALED=F_MAX_SCALED,
            f_scale=f_scale,
        )


class TwoCopyONNX:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        ins = self.sess.get_inputs()
        outs = self.sess.get_outputs()

        if len(ins) != 3 or len(outs) != 2:
            raise RuntimeError(f"{model_path}: expected 3 inputs and 2 outputs, got {len(ins)} and {len(outs)}")

        dims = []
        for info in ins:
            shp = list(info.shape)
            if len(shp) != 2 or shp[1] is None:
                raise RuntimeError(f"{model_path}: cannot infer input shape from {info.name}: {shp}")
            dims.append((info.name, int(shp[1])))

        dims_sorted = sorted(dims, key=lambda x: x[1])
        if dims_sorted[0][1] != 1 or dims_sorted[1][1] != dims_sorted[2][1]:
            raise RuntimeError(f"{model_path}: unexpected input dimensions {dims}")

        self.name_I = dims_sorted[0][0]
        self.name_f1 = dims_sorted[1][0]
        self.name_f2 = dims_sorted[2][0]
        self.H = dims_sorted[1][1]
        self.name_q1 = outs[0].name
        self.name_q2 = outs[1].name

    def eval(self, I: float, f1: np.ndarray, f2: np.ndarray) -> Tuple[float, float]:
        feed = {
            self.name_I: np.array([[I]], dtype=np.float32),
            self.name_f1: f1.reshape(1, -1).astype(np.float32),
            self.name_f2: f2.reshape(1, -1).astype(np.float32),
        }
        q1, q2 = self.sess.run([self.name_q1, self.name_q2], feed)
        return float(np.array(q1).reshape(-1)[0]), float(np.array(q2).reshape(-1)[0])


def sample_pair(rng: np.random.Generator, H: int, I_max: float, F_max: float, eps_f: float):
    I = float(rng.uniform(0.0, I_max))
    f1 = rng.uniform(0.0, F_max, size=(H,)).astype(np.float32)
    delta = rng.uniform(-eps_f, eps_f, size=(H,)).astype(np.float32)
    f2 = np.clip(f1 + delta, 0.0, F_max).astype(np.float32)
    return I, f1, f2


def summarize(values: np.ndarray) -> Dict[str, float]:
    q = np.quantile(values, [0.5, 0.9, 0.95, 0.99])
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "median": float(q[0]),
        "q90": float(q[1]),
        "q95": float(q[2]),
        "q99": float(q[3]),
    }


def load_lhat(results_root: Path, method: str):
    p = results_root / "models" / method / "lipschitz.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return float(d.get("L_hat"))
    except Exception:
        return None


def method_model_path(results_root: Path, method: str) -> Path:
    p = results_root / "models" / method / "policy_two_copy.onnx"
    if not p.exists():
        raise FileNotFoundError(f"Missing ONNX for method '{method}' at {p}")
    return p


def infer_forecast_from_results(results_root: Path) -> str | None:
    name = results_root.name.lower()
    if "nhits" in name:
        return "nhits"
    if "nbeats" in name:
        return "nbeats"
    return None


def resolve_models_dir(models_dir_arg: str, top_root: Path, results_root: Path) -> Path:
    raw = Path(models_dir_arg).expanduser()

    # 1) direct / relative to cwd
    if raw.exists():
        return raw.resolve()

    # 2) relative to top_root
    cand = (top_root / raw)
    if cand.exists():
        return cand.resolve()

    # 3) nested shorthand resolution: nhits_20260123/models or nbeats_20260123/models
    parts = raw.parts
    if len(parts) >= 2 and parts[-1] == "models":
        forecast_dir = parts[-2]
        nested = top_root / "Lipschitz_Guided_Training" / "Verify_nn_prod_plan" / "models_20260123" / forecast_dir / "models"
        if nested.exists():
            return nested.resolve()

    # 4) infer from results root
    forecast = infer_forecast_from_results(results_root)
    if forecast is not None:
        nested = top_root / "Lipschitz_Guided_Training" / "Verify_nn_prod_plan" / "models_20260123" / f"{forecast}_20260123" / "models"
        if nested.exists():
            return nested.resolve()

    raise FileNotFoundError(
        f"Could not resolve models-dir from '{models_dir_arg}'. "
        f"Tried direct, top-root-relative, and nested forecast layout."
    )


def resolve_results_root(results_root_arg: str, top_root: Path) -> Path:
    raw = Path(results_root_arg).expanduser()
    if raw.exists():
        return raw.resolve()
    cand = top_root / raw
    if cand.exists():
        return cand.resolve()
    raise FileNotFoundError(f"Could not resolve results-root from '{results_root_arg}'")


def run_one_method(
    method: str,
    model_path: Path,
    scaling: Scaling,
    eps_q: float,
    eps_f: float,
    I_max: float,
    F_max: float,
    num_samples: int,
    seed: int,
    save_samples_path: Path | None = None,
):
    model = TwoCopyONNX(model_path)
    if model.H != scaling.H:
        raise RuntimeError(f"{method}: H mismatch: ONNX has H={model.H}, scaling.json says H={scaling.H}")

    rng = np.random.default_rng(seed)
    abs_dq = np.empty((num_samples,), dtype=np.float64)
    plus_margin = np.empty((num_samples,), dtype=np.float64)
    minus_margin = np.empty((num_samples,), dtype=np.float64)
    max_df = np.empty((num_samples,), dtype=np.float64)
    rows = []

    for i in range(num_samples):
        I, f1, f2 = sample_pair(rng, model.H, I_max, F_max, eps_f)
        q1, q2 = model.eval(I, f1, f2)

        dq = abs(q2 - q1)
        plus = (q2 - q1) - eps_q
        minus = (q1 - q2) - eps_q
        df = float(np.max(np.abs(f2 - f1)))

        abs_dq[i] = dq
        plus_margin[i] = plus
        minus_margin[i] = minus
        max_df[i] = df

        rows.append({
            "method": method,
            "sample_id": i,
            "I": I,
            "q1": q1,
            "q2": q2,
            "abs_dq": dq,
            "plus_margin": plus,
            "minus_margin": minus,
            "max_df": df,
            "violates_abs": int(dq >= eps_q),
            "violates_plus": int(plus >= 0.0),
            "violates_minus": int(minus >= 0.0),
        })

    abs_stats = summarize(abs_dq)
    df_stats = summarize(max_df)

    summary = {
        "method": method,
        "num_samples": int(num_samples),
        "seed": int(seed),
        "eps_q": float(eps_q),
        "eps_f_scaled": float(eps_f),
        "I_max_scaled": float(I_max),
        "F_max_scaled": float(F_max),
        "mean_abs_dq": abs_stats["mean"],
        "max_abs_dq": abs_stats["max"],
        "median_abs_dq": abs_stats["median"],
        "q90_abs_dq": abs_stats["q90"],
        "q95_abs_dq": abs_stats["q95"],
        "q99_abs_dq": abs_stats["q99"],
        "viol_rate_abs": float(np.mean(abs_dq >= eps_q)),
        "viol_rate_plus": float(np.mean(plus_margin >= 0.0)),
        "viol_rate_minus": float(np.mean(minus_margin >= 0.0)),
        "max_observed_df": df_stats["max"],
        "mean_observed_df": df_stats["mean"],
    }

    if save_samples_path is not None:
        with save_samples_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-root", default=".", help="Run from extended_experiments_202604 or pass it explicitly")
    ap.add_argument("--models-dir", required=True, help="Original forecast-family models dir containing scaling.json")
    ap.add_argument("--results-root", required=True, help="Results dir containing models/<method>/policy_two_copy.onnx")
    ap.add_argument("--eps-q", type=float, required=True)

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--epsf-scaled", type=float)
    g.add_argument("--epsf-raw", type=float)

    ap.add_argument("--i-max-scaled", type=float, default=None)
    ap.add_argument("--f-max-scaled", type=float, default=None)
    ap.add_argument("--num-samples", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["vanilla", "sn_only", "fgsm_only", "random_only", "full"])
    ap.add_argument("--save-per-method-samples", action="store_true")
    ap.add_argument("--outdir", default="proof_aligned_compare_out")
    args = ap.parse_args()

    top_root = Path(args.top_root).expanduser().resolve()
    results_root = resolve_results_root(args.results_root, top_root)
    models_dir = resolve_models_dir(args.models_dir, top_root, results_root)

    outdir = (top_root / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    scaling = Scaling.load(models_dir)
    eps_f = float(args.epsf_scaled) if args.epsf_scaled is not None else float(args.epsf_raw) * scaling.f_scale
    I_max = float(args.i_max_scaled) if args.i_max_scaled is not None else scaling.I_MAX_SCALED
    F_max = float(args.f_max_scaled) if args.f_max_scaled is not None else scaling.F_MAX_SCALED

    summaries = []
    for offset, method in enumerate(args.methods):
        model_path = method_model_path(results_root, method)
        samples_path = (outdir / f"empirical_alignment_samples_{method}.csv") if args.save_per_method_samples else None
        summary = run_one_method(
            method=method,
            model_path=model_path,
            scaling=scaling,
            eps_q=float(args.eps_q),
            eps_f=float(eps_f),
            I_max=float(I_max),
            F_max=float(F_max),
            num_samples=int(args.num_samples),
            seed=int(args.seed) + offset,
            save_samples_path=samples_path,
        )
        lhat = load_lhat(results_root, method)
        summary["L_hat"] = lhat if lhat is not None else ""
        summaries.append(summary)
        print(
            f"[{method:11s}] max|Δq|={summary['max_abs_dq']:.6f} "
            f"mean|Δq|={summary['mean_abs_dq']:.6f} "
            f"viol={summary['viol_rate_abs']:.6f}"
        )

    summary_csv = outdir / "empirical_alignment_compare_summary.csv"
    fields = [
        "method", "L_hat", "num_samples", "seed", "eps_q", "eps_f_scaled",
        "I_max_scaled", "F_max_scaled",
        "mean_abs_dq", "max_abs_dq", "median_abs_dq", "q90_abs_dq",
        "q95_abs_dq", "q99_abs_dq",
        "viol_rate_abs", "viol_rate_plus", "viol_rate_minus",
        "max_observed_df", "mean_observed_df",
    ]
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summaries:
            w.writerow(row)

    report = outdir / "empirical_alignment_compare_report.txt"
    with report.open("w") as f:
        f.write("Proof-aligned empirical comparison report\n")
        f.write("=" * 80 + "\n")
        f.write(f"top_root: {top_root}\n")
        f.write(f"models_dir: {models_dir}\n")
        f.write(f"results_root: {results_root}\n")
        f.write(f"eps_q: {args.eps_q}\n")
        f.write(f"eps_f_scaled: {eps_f}\n")
        f.write(f"I_max_scaled: {I_max}\n")
        f.write(f"F_max_scaled: {F_max}\n")
        f.write(f"num_samples: {args.num_samples}\n\n")
        for row in summaries:
            f.write(
                f"{row['method']}: "
                f"L_hat={row['L_hat']} "
                f"max|Δq|={row['max_abs_dq']:.6f} "
                f"mean|Δq|={row['mean_abs_dq']:.6f} "
                f"viol={row['viol_rate_abs']:.6f} "
                f"plus={row['viol_rate_plus']:.6f} "
                f"minus={row['viol_rate_minus']:.6f}\n"
            )

    print(f"[done] resolved models-dir -> {models_dir}")
    print(f"[done] resolved results-root -> {results_root}")
    print(f"[done] wrote {summary_csv}")
    print(f"[done] wrote {report}")


if __name__ == "__main__":
    main()
    


'''
NHITS OUTPUT:
~/Documents/Verify_nn_prod_plan/extended_experiments_202604$ python3 proof_aligned_empirical_compare.py   --models-dir nhits_20260123/models   --results-root ~/Documents/Verify_nn_prod_plan/extended_experiments_202604/results_nhits   --eps-q 0.2   --epsf-scaled 1.0   --i-max-scaled 1.0   --f-max-scaled 2.0   --num-samples 50000   --seed 0   --methods vanilla sn_only fgsm_only random_only full   --outdir proof_aligned_compare_nhits   --save-per-method-samples
[vanilla    ] max|Δq|=60.744270 mean|Δq|=16.426337 viol=0.991720
[sn_only    ] max|Δq|=0.572536 mean|Δq|=0.207768 viol=0.476520
[fgsm_only  ] max|Δq|=0.097279 mean|Δq|=0.014843 viol=0.000000
[random_only] max|Δq|=4.819214 mean|Δq|=1.001085 viol=0.567100
[full       ] max|Δq|=0.158743 mean|Δq|=0.059697 viol=0.000000
[done] resolved models-dir -> .../Verify_nn_prod_plan/extended_experiments_202604/Lipschitz_Guided_Training/Verify_nn_prod_plan/models_20260123/nhits_20260123/models
[done] resolved results-root -> .../Verify_nn_prod_plan/extended_experiments_202604/results_nhits
[done] wrote .../Verify_nn_prod_plan/extended_experiments_202604/proof_aligned_compare_nhits/empirical_alignment_compare_summary.csv
[done] wrote .../Verify_nn_prod_plan/extended_experiments_202604/proof_aligned_compare_nhits/empirical_alignment_compare_report.txt

NBEATS OUTPUT:
$ python3 proof_aligned_empirical_compare.py \
>   --models-dir nbeats_20260123/models \
>   --results-root ~/Documents/Verify_nn_prod_plan/extended_experiments_202604/results_nbeats \
>   --eps-q 0.2 \
>   --epsf-scaled 1.0 \
>   --i-max-scaled 1.0 \
>   --f-max-scaled 2.0 \
>   --num-samples 50000 \
>   --seed 0 \
>   --methods vanilla sn_only fgsm_only random_only full \
>   --outdir proof_aligned_compare_nbeats \
>   --save-per-method-samples
[vanilla    ] max|Δq|=121.525528 mean|Δq|=25.667402 viol=0.995020
[sn_only    ] max|Δq|=0.542909 mean|Δq|=0.208219 viol=0.478580
[fgsm_only  ] max|Δq|=0.097279 mean|Δq|=0.014843 viol=0.000000
[random_only] max|Δq|=5.897568 mean|Δq|=1.112878 viol=0.566560
[full       ] max|Δq|=0.158650 mean|Δq|=0.058961 viol=0.000000
[done] resolved models-dir -> .../Verify_nn_prod_plan/extended_experiments_202604/Lipschitz_Guided_Training/Verify_nn_prod_plan/models_20260123/nbeats_20260123/models
[done] resolved results-root -> .../Verify_nn_prod_plan/extended_experiments_202604/results_nbeats
[done] wrote .../Verify_nn_prod_plan/extended_experiments_202604/proof_aligned_compare_nbeats/empirical_alignment_compare_summary.csv
[done] wrote .../Verify_nn_prod_plan/extended_experiments_202604/proof_aligned_compare_nbeats/empirical_alignment_compare_report.txt

'''
