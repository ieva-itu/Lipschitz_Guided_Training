#!/usr/bin/env python3
"""
domain_coverage_analysis.py — Experiment (Domain Coverage Analysis)

Objective:
  Demonstrate robustness holds across diverse operating regions (domain families),
  not just verification corner cases.

Domain families (as in your enclosed CSVs):
  - baseline
  - data-driven
  - stress

For each (controller, epsq, domain):
  - Sample (I, f) uniformly in the domain box:
        I_scaled ~ U[0, I_cap_scaled]
        f        ~ U[0, F_cap_scaled]^H
  - Sample perturbation:
        δ ~ U[-epsf_scaled, epsf_scaled]^H
        f' = clip(f + δ, 0, F_cap_scaled)
  - Compute:
        q  = π(I, f)
        q' = π(I, f')
        Δq = |q' - q|
  - Report: max|Δq|, mean|Δq|, violation_rate = Pr(Δq > epsq)
  - Aggregate by family (pooling all domains in that family)

Inputs:
  - ONNX policies (single-input or multi-input two-copy; for multi we evaluate π(I,f,f))
  - Domain table CSV: columns must include
        domain_index, family, I_cap_scaled, F_cap_scaled
 
Outputs:
  - CSV with per-domain stats
  - CSV with per-family aggregated stats
  - Expected results table

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import onnxruntime as ort
except ImportError as e:
    raise SystemExit("onnxruntime is required: pip install onnxruntime") from e


# -----------------------------
# ONNX policy wrapper 
# -----------------------------
class OnnxPolicy:
    def __init__(self, onnx_path: Path):
        self.onnx_path = onnx_path
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.infos = self.sess.get_inputs()
        self.out_name = self.sess.get_outputs()[0].name

        self.in_names = [i.name for i in self.infos]
        self.mode = "single" if len(self.in_names) == 1 else "multi"

        def _shape(info):
            return list(info.shape)

        if self.mode == "single":
            shp = _shape(self.infos[0])
            if len(shp) != 2 or shp[1] is None:
                raise RuntimeError(f"Cannot infer H from single-input shape: {shp} in {onnx_path}")
            dim = int(shp[1])
            if dim < 2:
                raise RuntimeError(f"Single-input dim too small: {dim} in {onnx_path}")
            self.H = dim - 1
            self.role_I = self.in_names[0]
            self.role_f1 = None
            self.role_f2 = None
        else:
            shapes = [(n, _shape(info)) for n, info in zip(self.in_names, self.infos)]
            dims2 = []
            for n, shp in shapes:
                if len(shp) == 2 and shp[1] is not None:
                    dims2.append((n, int(shp[1])))

            if len(dims2) < 3:
                raise RuntimeError(f"Cannot infer multi-input dims from shapes={shapes} in {onnx_path}")

            dims2_sorted = sorted(dims2, key=lambda x: x[1])
            self.role_I = dims2_sorted[0][0]
            self.role_f1 = dims2_sorted[-2][0]
            self.role_f2 = dims2_sorted[-1][0]

            H1 = dims2_sorted[-2][1]
            H2 = dims2_sorted[-1][1]
            if H1 != H2:
                raise RuntimeError(f"f1/f2 dims mismatch: {H1} vs {H2} in {onnx_path}")
            self.H = H1

    def q_of(self, I_scaled: float, f: np.ndarray) -> float:
        f = np.asarray(f, dtype=np.float32).reshape(-1)
        if f.shape[0] != self.H:
            raise RuntimeError(
                f"{self.onnx_path.name}: forecast dim mismatch (got {f.shape[0]}, expected {self.H})"
            )

        if self.mode == "single":
            x = np.concatenate([[float(I_scaled)], f]).reshape(1, -1).astype(np.float32)
            feed = {self.role_I: x}
        else:
            # nominal evaluation is π(I,f,f)
            feed = {
                self.role_I: np.array([[float(I_scaled)]], dtype=np.float32),
                self.role_f1: f.reshape(1, -1),
                self.role_f2: f.reshape(1, -1),
            }

        out = self.sess.run([self.out_name], feed)[0]
        return float(np.array(out).reshape(-1)[0])


# -----------------------------
# Domain table loader
# -----------------------------
@dataclass(frozen=True)
class Domain:
    domain_index: int
    family: str
    I_cap_scaled: float
    F_cap_scaled: float


def load_domain_table(path: Path) -> List[Domain]:
    df = pd.read_csv(path)
    need = {"domain_index", "family", "I_cap_scaled", "F_cap_scaled"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing required columns: {sorted(missing)}")

    doms: List[Domain] = []
    for _, r in df.iterrows():
        doms.append(
            Domain(
                domain_index=int(r["domain_index"]),
                family=str(r["family"]),
                I_cap_scaled=float(r["I_cap_scaled"]),
                F_cap_scaled=float(r["F_cap_scaled"]),
            )
        )
    if len(doms) == 0:
        raise SystemExit(f"{path} has no rows")
    return doms


# -----------------------------
# Sampling + metrics
# -----------------------------
def sample_domain(
    rng: np.random.Generator,
    H: int,
    dom: Domain,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (I_scaled, f) with shapes:
      I_scaled: (n,)
      f: (n,H)
    """
    I = rng.uniform(0.0, dom.I_cap_scaled, size=(n,)).astype(np.float32)
    f = rng.uniform(0.0, dom.F_cap_scaled, size=(n, H)).astype(np.float32)
    return I, f


def perturb_linf(
    rng: np.random.Generator,
    f: np.ndarray,
    epsf_scaled: float,
    F_cap_scaled: float,
) -> np.ndarray:
    """
    f: (n,H) or (H,)
    δ ~ U[-epsf, epsf], and clip to [0, F_cap_scaled]
    """
    delta = rng.uniform(-epsf_scaled, epsf_scaled, size=f.shape).astype(np.float32)
    fp = f + delta
    fp = np.clip(fp, 0.0, F_cap_scaled).astype(np.float32)
    return fp


def compute_stats(dq: np.ndarray, epsq: float) -> Dict[str, float]:
    dq = np.asarray(dq, dtype=np.float64).reshape(-1)
    return {
        "max_dq": float(np.max(dq)) if dq.size else float("nan"),
        "mean_dq": float(np.mean(dq)) if dq.size else float("nan"),
        "violation_rate": float(np.mean(dq > epsq)) if dq.size else float("nan"),
    }


def fmt_cell(max_dq: float, mean_dq: float, viol: float) -> str:
    # compact + paper-friendly
    return f"max={max_dq:.3g}, mean={mean_dq:.3g}, viol={viol:.1%}"


# -----------------------------
# LaTeX table writer
# -----------------------------
def write_overleaf_table(
    outpath: Path,
    controllers: List[str],
    epsq: float,
    fam_stats: Dict[str, Dict[str, Dict[str, float]]],
    families_order: List[str] = ["baseline", "data-driven", "stress"],
):
    """
    fam_stats[controller][family] = {max_dq, mean_dq, violation_rate}
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(r"\begin{tabular}{l c p{3.7cm} p{3.7cm} p{3.7cm}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Controller} & $\boldsymbol{\epsilon_q}$ & \textbf{Baseline Domains} & \textbf{Data Domains} & \textbf{Stress Domains} \\")
    lines.append(r"\midrule")

    for ctrl in controllers:
        row = [ctrl, f"{epsq:g}"]
        for fam in families_order:
            s = fam_stats.get(ctrl, {}).get(fam, None)
            if s is None:
                row.append("--")
            else:
                row.append(fmt_cell(s["max_dq"], s["mean_dq"], s["violation_rate"]))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Domain coverage analysis (empirical). For each controller and each domain family, we sample $(I,f)$ uniformly within the family-specific box constraints and apply time-varying perturbations $\|\delta\|_\infty \le \epsilon_f$. We report the maximum and mean deviation $|\Delta q|$ and the violation rate $\Pr(|\Delta q| > \epsilon_q)$ aggregated over all domains in the family.}"
    )
    lines.append(r"\label{tab:domain_coverage_empirical}")
    lines.append(r"\end{table}")

    outpath.write_text("\n".join(lines))


# -----------------------------
# CLI utils
# -----------------------------
def parse_map(items: List[str]) -> Dict[str, Path]:
    m: Dict[str, Path] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"Expected entries like key=path, got: {it}")
        k, p = it.split("=", 1)
        m[k.strip()] = Path(p).expanduser()
    return m


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--domain-table-csv", type=str, required=True, help="CSV with domain families + caps (scaled units)")
    ap.add_argument("--controllers", nargs="+", required=True)
    ap.add_argument("--policy-onnx", nargs="+", required=True, help="Mapping tag=onnx_path")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--epsf-scaled", type=float, help="ε_f in scaled forecast units")
    g.add_argument("--epsf-raw", type=float, help="ε_f in raw units (ONLY if you also pass --f-scale)")
    ap.add_argument("--f-scale", type=float, default=None, help="If using --epsf-raw: epsf_scaled = epsf_raw * f_scale")

    ap.add_argument("--epsq", type=float, required=True, help="ε_q threshold (policy output units)")
    ap.add_argument("--num-samples", type=int, default=5000, help="Samples per domain")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--outroot", type=str, default="models_20260123")

    args = ap.parse_args()

    domain_path = Path(args.domain_table_csv).expanduser()
    domains = load_domain_table(domain_path)

    policy_map = parse_map(args.policy_onnx)
    policies: Dict[str, OnnxPolicy] = {}
    for tag in args.controllers:
        if tag not in policy_map:
            raise SystemExit(f"No --policy-onnx entry provided for controller '{tag}'")
        policies[tag] = OnnxPolicy(policy_map[tag])

    # sanity: all policies must have same H for shared domains
    Hs = {tag: pol.H for tag, pol in policies.items()}
    H_unique = sorted(set(Hs.values()))
    if len(H_unique) != 1:
        raise SystemExit(f"Controllers have different horizons H: {Hs}")
    H = H_unique[0]

    # epsf_scaled
    if args.epsf_scaled is not None:
        epsf_scaled = float(args.epsf_scaled)
    else:
        if args.f_scale is None:
            raise SystemExit("--epsf-raw requires --f-scale (so epsf_scaled = epsf_raw * f_scale)")
        epsf_scaled = float(args.epsf_raw) * float(args.f_scale)

    epsq = float(args.epsq)
    n = int(args.num_samples)
    rng = np.random.default_rng(int(args.seed))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outroot).expanduser() / f"empirical_out_{stamp}" / "domain_coverage"
    outdir.mkdir(parents=True, exist_ok=True)

    # per-domain stats records
    rows = []

    # also collect dq per family (pool across domains) for aggregation
    fam_pool: Dict[str, Dict[str, List[float]]] = {ctrl: {} for ctrl in args.controllers}

    for dom in domains:
        I_s, f = sample_domain(rng, H, dom, n)
        f_p = perturb_linf(rng, f, epsf_scaled, dom.F_cap_scaled)

        # for each controller compute dq samples
        for ctrl, pol in policies.items():
            # run policy per sample (vectorization is limited by onnxruntime API; keep it simple & safe)
            dq = np.empty((n,), dtype=np.float32)
            for i in range(n):
                q = pol.q_of(float(I_s[i]), f[i])
                qp = pol.q_of(float(I_s[i]), f_p[i])
                dq[i] = abs(qp - q)

            s = compute_stats(dq, epsq)

            rows.append(
                {
                    "controller": ctrl,
                    "epsq": epsq,
                    "domain_index": dom.domain_index,
                    "family": dom.family,
                    "I_cap_scaled": dom.I_cap_scaled,
                    "F_cap_scaled": dom.F_cap_scaled,
                    "epsf_scaled": epsf_scaled,
                    "num_samples": n,
                    **s,
                }
            )

            fam_pool.setdefault(ctrl, {}).setdefault(dom.family, []).append(dq)

    df_domain = pd.DataFrame(rows)
    df_domain.to_csv(outdir / "domain_coverage_per_domain.csv", index=False)

    # aggregate by family (pool dq across all domains in that family)
    fam_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    fam_rows = []
    for ctrl in args.controllers:
        fam_stats[ctrl] = {}
        for fam, chunks in fam_pool.get(ctrl, {}).items():
            dq_all = np.concatenate(chunks, axis=0)
            s = compute_stats(dq_all, epsq)
            fam_stats[ctrl][fam] = s
            fam_rows.append(
                {
                    "controller": ctrl,
                    "epsq": epsq,
                    "family": fam,
                    "epsf_scaled": epsf_scaled,
                    "num_samples_total": int(dq_all.size),
                    **s,
                }
            )

    df_family = pd.DataFrame(fam_rows).sort_values(["controller", "family"])
    df_family.to_csv(outdir / "domain_coverage_by_family.csv", index=False)

    # Overleaf table
    write_overleaf_table(
        outpath=outdir / "domain_coverage_table.tex",
        controllers=args.controllers,
        epsq=epsq,
        fam_stats=fam_stats,
        families_order=["baseline", "data-driven", "stress"],
    )

    # print a compact summary
    print("\n=== Domain Coverage Analysis (aggregated by family) ===")
    for ctrl in args.controllers:
        print(f"\n[{ctrl}]")
        for fam in ["baseline", "data-driven", "stress"]:
            s = fam_stats.get(ctrl, {}).get(fam, None)
            if s is None:
                continue
            print(
                f"  {fam:11s}  max|Δq|={s['max_dq']:.6g}  mean|Δq|={s['mean_dq']:.6g}  viol={s['violation_rate']:.3f}"
            )

    print(f"\n[ok] wrote outputs to: {outdir}")
    print(f"     - {outdir/'domain_coverage_per_domain.csv'}")
    print(f"     - {outdir/'domain_coverage_by_family.csv'}")
    print(f"     - {outdir/'domain_coverage_table.tex'}")


if __name__ == "__main__":
    main()

