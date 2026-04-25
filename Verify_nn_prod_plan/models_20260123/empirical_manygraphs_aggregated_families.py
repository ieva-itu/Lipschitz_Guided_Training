#!/usr/bin/env python3
"""
empirical_manygraphs_aggregated_families.py

Empirical robustness + figure (NHITS/NBEATS baseline/robust).

Main features:
- Works with ONNX policies with either:
    (A) single input tensor: x = [I, f]          shape (1, 1+H)
    (B) three inputs: I, f1, f2                  shapes (1,1), (1,H), (1,H)
- Option 3 semantic "domain index": baseline → data-driven → stress (in 3 blocks)
- Emits:
    * Figures: max|Δq| vs domain index for each eps_q
    * Overlays: baseline vs robust for each family at --paper-epsq
- Writes:
    * domain_table_<tag>.csv  and domain_table_<tag>.tex
      mapping domain index -> (family, I_cap, F_cap), with notations used in the plots.

Output location:
- If --outdir is provided: writes there
- Else: writes into models_20260123/empirical_out_<TIMESTAMP> (if models_20260123 exists), otherwise ./empirical_out_<TIMESTAMP>

"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import onnxruntime as ort
except ImportError as e:
    raise SystemExit("onnxruntime is required: pip install onnxruntime") from e


# -----------------------------
# Styling (CAV/TACAS-ish)
# -----------------------------
def set_cav_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.color": "0.85",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


# -----------------------------
# Scaling / config
# -----------------------------
@dataclass(frozen=True)
class Scaling:
    i_scale: float
    f_scale: float
    output_scale: float
    i_max_scaled: float
    f_max_scaled: float
    horizon: int

    @staticmethod
    def load(path: Path) -> "Scaling":
        d = json.loads(path.read_text())
        return Scaling(
            i_scale=float(d.get("i_scale", 1.0 / 200.0)),
            f_scale=float(d.get("f_scale", 1.0 / 50.0)),
            output_scale=float(d.get("output_scale", 1.0)),
            i_max_scaled=float(d.get("I_MAX_SCALED", d.get("i_max_scaled", 4.0))),
            f_max_scaled=float(d.get("F_MAX_SCALED", d.get("f_max_scaled", 2.0))),
            horizon=int(d.get("HORIZON", d.get("horizon", 10))),
        )


# -----------------------------
# ONNX policy wrapper (flexible inputs)
# -----------------------------
class OnnxPolicy:
    """
    Supports:
      - single input: x=[I,f] (1,1+H)
      - multi inputs: I,(f1),(f2) each (1,dim)

    Also infers H from ONNX input shapes to avoid scaling.json mismatches (e.g., 7 vs 10).
    """

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
            self.role_single = self.in_names[0]
            self.role_I = None
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
            self.role_single = None

    def __call__(self, I: float, f1: np.ndarray, f2: Optional[np.ndarray] = None) -> float:
        if self.mode == "single":
            x = np.concatenate([[I], f1.astype(np.float32).reshape(-1)]).reshape(1, -1).astype(np.float32)
            feed = {self.role_single: x}
        else:
            assert f2 is not None, "Multi-input ONNX expects f2."
            feed = {
                self.role_I: np.array([[I]], dtype=np.float32),
                self.role_f1: f1.reshape(1, -1).astype(np.float32),
                self.role_f2: f2.reshape(1, -1).astype(np.float32),
            }

        out = self.sess.run([self.out_name], feed)[0]
        return float(np.array(out).reshape(-1)[0])


# -----------------------------
# Domain model (Option 3)
# -----------------------------
def make_domain_caps(
    I_base: float,
    F_base: float,
    n_domains: int,
    data_mult: float,
    stress_mult: float,
) -> List[Tuple[float, float, str]]:
    """
    Option 3: semantic domains as 3 blocks:
      block 0: baseline
      block 1: data-driven (expanded by data_mult)
      block 2: stress (expanded by stress_mult)

    Returns list [(I_cap, F_cap, family_label)] of length n_domains.
    """
    if n_domains < 3:
        raise ValueError("n_domains must be >= 3")

    n0 = n_domains // 3
    n1 = n_domains // 3
    n2 = n_domains - n0 - n1  # remainder to stress

    caps: List[Tuple[float, float, str]] = []
    for _ in range(n0):
        caps.append((I_base, F_base, "baseline"))
    for _ in range(n1):
        caps.append((I_base * data_mult, F_base * data_mult, "data-driven"))
    for _ in range(n2):
        caps.append((I_base * stress_mult, F_base * stress_mult, "stress"))
    return caps


def domain_block_cuts(caps: List[Tuple[float, float, str]]) -> Tuple[int, int]:
    fam = [c[2] for c in caps]
    cut1 = fam.index("data-driven") if "data-driven" in fam else len(caps)
    cut2 = fam.index("stress") if "stress" in fam else len(caps)
    return cut1, cut2


def write_domain_tables(outdir: Path, tag: str, caps: List[Tuple[float, float, str]], image_name: str) -> None:
    """
    Writes:
      - domain_table_<tag>.csv
      - domain_table_<tag>.tex
    Includes the image name for traceability.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"domain_table_{tag}.csv"
    tex_path = outdir / f"domain_table_{tag}.tex"

    # CSV
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "domain_index", "family", "I_cap_scaled", "F_cap_scaled"])
        for i, (I_cap, F_cap, fam) in enumerate(caps):
            w.writerow([image_name, i, fam, f"{I_cap:.6g}", f"{F_cap:.6g}"])

    # LaTeX (compact)
    rows = []
    for i, (I_cap, F_cap, fam) in enumerate(caps):
        rows.append(rf"{i} & {fam} & {I_cap:.3g} & {F_cap:.3g} \\")
    tex = r"""
\begin{table}[t]
\centering
\small
\begin{tabular}{r l r r}
\toprule
Domain & Family & $I_{\max}$ & $f_{\max}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Domain index mapping for \textsc{""" + tag + r"""} (Option~3). Each row specifies the caps used when sampling $(I,f)$ in the empirical evaluation; units are in the \emph{scaled} verification space.}
\label{tab:domain-map-""" + tag + r"""}
\end{table}
""".lstrip()
    tex_path.write_text(tex)


# -----------------------------
# Sampling + perturbation
# -----------------------------
def sample_state(rng: np.random.Generator, I_cap: float, F_cap: float, H: int) -> Tuple[float, np.ndarray]:
    I = float(rng.uniform(0.0, I_cap))
    f = rng.uniform(0.0, F_cap, size=(H,)).astype(np.float32)
    return I, f


def perturb_forecast_linf(
    rng: np.random.Generator, f: np.ndarray, eps_f: float, F_min: float, F_max: float
) -> np.ndarray:
    delta = rng.uniform(-eps_f, eps_f, size=f.shape).astype(np.float32)
    return np.clip(f + delta, F_min, F_max).astype(np.float32)


# -----------------------------
# Core empirical evaluation
# -----------------------------
def run_empirical_eval(
    policy: OnnxPolicy,
    scaling: Scaling,
    eps_f_scaled: float,
    n_pairs: int,
    seed: int,
    caps: List[Tuple[float, float, str]],
) -> np.ndarray:
    """
    Returns max_dq_per_domain (len = len(caps)), where dq = |pi(I,f2) - pi(I,f1)|.
    All quantities are in the policy's *scaled* space.
    """
    rng = np.random.default_rng(seed)
    H = policy.H  # prefer ONNX-inferred horizon

    max_dq = np.zeros(len(caps), dtype=np.float64)
    for d, (I_cap, F_cap, _fam) in enumerate(caps):
        worst = 0.0
        for _ in range(n_pairs):
            I, f1 = sample_state(rng, I_cap, F_cap, H)
            f2 = perturb_forecast_linf(rng, f1, eps_f_scaled, F_min=0.0, F_max=F_cap)

            if policy.mode == "multi":
                q1 = policy(I, f1, f2)
                q2 = policy(I, f2, f1)
            else:
                q1 = policy(I, f1)
                q2 = policy(I, f2)

            worst = max(worst, abs(q2 - q1))
        max_dq[d] = worst
    return max_dq


# -----------------------------
# Plotting
# -----------------------------
def stamp_filename(fig, fname: str) -> None:
    fig.text(0.995, 0.01, fname, ha="right", va="bottom", fontsize=8, color="0.35")


def plot_appendix_maxdq(
    outdir: Path,
    tag: str,
    eps_q: float,
    eps_f_scaled: float,
    max_dq: np.ndarray,
    caps: List[Tuple[float, float, str]],
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(max_dq)
    x = np.arange(n)

    cut1, cut2 = domain_block_cuts(caps)

    fig = plt.figure(figsize=(7.2, 2.8))
    ax = plt.gca()

    # Single curve (black)
    ax.plot(x, max_dq, linewidth=1.6, color="k")

    # Threshold line (dark gray dashed)
    ax.axhline(eps_q, linestyle="--", linewidth=1.3, color="0.35")

    # Block separators
    if 0 < cut1 < n:
        ax.axvline(cut1 - 0.5, color="0.75", linewidth=0.9)
    if 0 < cut2 < n:
        ax.axvline(cut2 - 0.5, color="0.75", linewidth=0.9)

    # Block labels: put slightly below top to avoid collisions
    def center(a, b): return (a + b) / 2.0
    if cut1 < n and cut2 < n and cut1 > 0 and cut2 > cut1:
        ax.text(center(0, cut1 - 1), 0.94, "baseline", transform=ax.get_xaxis_transform(),
                ha="center", va="top", color="0.35")
        ax.text(center(cut1, cut2 - 1), 0.94, "data-driven", transform=ax.get_xaxis_transform(),
                ha="center", va="top", color="0.35")
        ax.text(center(cut2, n - 1), 0.94, "stress", transform=ax.get_xaxis_transform(),
                ha="center", va="top", color="0.35")

    ax.set_xlabel("domain index")
    ax.set_ylabel(r"max $|\Delta q|$")
    
    ax.set_title(f"{tag}: ||δ||∞ ≤ r={eps_f_scaled}, εq={eps_q}")


    
    fig.tight_layout()

    fname = outdir / f"maxdq_{tag}_epsq_{eps_q}.png"
    #stamp_filename(fig, fname.name)
    fig.savefig(fname, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_paper_overlay_baseline_vs_robust(
    outdir: Path,
    family: str,
    eps_q: float,
    eps_f_scaled: float,
    maxdq_baseline: np.ndarray,
    maxdq_robust: np.ndarray,
    caps: List[Tuple[float, float, str]],
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(maxdq_baseline)
    x = np.arange(n)
    cut1, cut2 = domain_block_cuts(caps)

    fig = plt.figure(figsize=(7.2, 2.8))
    ax = plt.gca()

    # Baseline: solid black
    ax.plot(x, maxdq_baseline, linewidth=1.6, color="k", label=f"{family} baseline")
    # Robust: dashed medium gray
    ax.plot(x, maxdq_robust, linewidth=1.6, color="0.45", linestyle="--", label=f"{family} robust")
    # Threshold: dotted darker gray
    ax.axhline(eps_q, linestyle=":", linewidth=1.3, color="0.25", label=r"$\epsilon_q$")

    if 0 < cut1 < n:
        ax.axvline(cut1 - 0.5, color="0.80", linewidth=0.9)
    if 0 < cut2 < n:
        ax.axvline(cut2 - 0.5, color="0.80", linewidth=0.9)

    ax.set_xlabel("domain index")
    ax.set_ylabel(r"max $|\Delta q|$")
    ax.set_title(f"{family.upper()}: baseline vs robust (r={eps_f_scaled}, εq={eps_q})")


    # ✅ Legend: lower right, slightly inset, non-intrusive
    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="0.85",
        framealpha=0.95,
        borderpad=0.25,
        handlelength=2.2,
        bbox_to_anchor=(0.985, 0.04),
    )

    fig.tight_layout()

    fname = outdir / f"paper_{family}_baseline_vs_robust_epsq_{eps_q}.png"
    #stamp_filename(fig, fname.name)
    fig.savefig(fname, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return fname


# -----------------------------
# Captions
# -----------------------------
def save_caption_tex(path: Path, caption: str) -> None:
    path.write_text(caption.strip() + "\n")


def caption_appendix_maxdq(tag: str, eps_q: float, eps_f_scaled: float, n_pairs: int, n_domains: int) -> str:
    return rf"""
\caption{{\textbf{{Empirical local robustness (appendix):}}
Maximum observed output deviation $\max|\Delta q|$ over {n_pairs} sampled state--perturbation pairs per domain
for \textsc{{{tag}}} across {n_domains} domains (Option~3: baseline/data-driven/stress),
under perturbations $\|\delta\|_\infty \le r$ with $r={eps_f_scaled}$ ($r=\varepsilon_f$).
The dashed horizontal line indicates the contract threshold $\epsilon_q={eps_q}$.}}
""".strip()


def caption_paper_baseline_vs_robust(family: str, eps_q: float, eps_f_scaled: float, n_pairs: int, n_domains: int) -> str:
    return rf"""
\caption{{\textbf{{Empirical local robustness: {family.upper()} baseline vs robust.}}
Maximum observed output deviation $\max|\Delta q|$ across {n_domains} domains (Option~3)
under perturbations $\|\delta\|_\infty \le r$ with $r={eps_f_scaled}$.
Each point is the maximum over {n_pairs} sampled state--perturbation pairs for that domain.
The dotted horizontal line marks $\epsilon_q={eps_q}$.}}
""".strip()


# -----------------------------
# CLI parsing helpers
# -----------------------------
def parse_policy_map(items: List[str]) -> Dict[str, Path]:
    m: Dict[str, Path] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--policy-onnx expects entries like key=path, got: {it}")
        k, p = it.split("=", 1)
        m[k.strip()] = Path(p).expanduser()
    return m


def default_outdir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path("models_20260123")
    if base.exists() and base.is_dir():
        return base / f"empirical_out_{ts}"
    return Path(f"empirical_out_{ts}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", required=True)
    ap.add_argument("--policy-onnx", nargs="+", required=True, help="tag=onnx_path entries")
    ap.add_argument("--scaling-json", type=str, required=True)

    ap.add_argument("--outdir", type=str, default=None, help="Output directory root.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pairs", type=int, default=2000)
    ap.add_argument("--n-domains", type=int, default=15)

    ap.add_argument("--data-mult", type=float, default=1.10,
                    help="Multiplier for caps in data-driven domains (relative to baseline).")
    ap.add_argument("--stress-mult", type=float, default=1.25,
                    help="Multiplier for caps in stress domains (relative to baseline).")

    ap.add_argument("--epsq-list", type=float, nargs="+",
                    default=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--epsf-scaled", type=float)
    g.add_argument("--epsf-raw", type=float)

    ap.add_argument("--paper-epsq", type=float, default=0.1)

    args = ap.parse_args()
    set_cav_style()

    outroot = Path(args.outdir).expanduser() if args.outdir else default_outdir()
    out_appendix = outroot / "appendix"
    out_paper = outroot / "paper"
    out_tables = outroot / "tables"
    out_appendix.mkdir(parents=True, exist_ok=True)
    out_paper.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    scaling = Scaling.load(Path(args.scaling_json).expanduser())

    eps_f_scaled = float(args.epsf_scaled) if args.epsf_scaled is not None else float(args.epsf_raw) * scaling.f_scale
    policy_map = parse_policy_map(args.policy_onnx)

    # Load policies
    policies: Dict[str, OnnxPolicy] = {}
    for tag in args.controllers:
        if tag not in policy_map:
            raise SystemExit(f"No --policy-onnx entry provided for controller '{tag}'.")
        policies[tag] = OnnxPolicy(policy_map[tag])

    # Common caps for this run (Option 3)
    # Use the (scaled) base caps from scaling.json; domain multipliers expand them.
    caps = make_domain_caps(
        I_base=scaling.i_max_scaled,
        F_base=scaling.f_max_scaled,
        n_domains=args.n_domains,
        data_mult=args.data_mult,
        stress_mult=args.stress_mult,
    )

    # Results[tag][eps_q] = maxdq array
    results: Dict[str, Dict[float, np.ndarray]] = {tag: {} for tag in args.controllers}

    for tag in args.controllers:
        print(f"\n=== Controller: {tag} ===")

        for eps_q in args.epsq_list:
            maxdq = run_empirical_eval(
                policy=policies[tag],
                scaling=scaling,
                eps_f_scaled=eps_f_scaled,
                n_pairs=args.n_pairs,
                seed=args.seed,
                caps=caps,
            )
            results[tag][eps_q] = maxdq

            fig_path = plot_appendix_maxdq(
                outdir=out_appendix / tag,
                tag=tag,
                eps_q=float(eps_q),
                eps_f_scaled=eps_f_scaled,
                max_dq=maxdq,
                caps=caps,
            )
            save_caption_tex(
                fig_path.with_suffix(".tex"),
                caption_appendix_maxdq(tag, float(eps_q), eps_f_scaled, args.n_pairs, args.n_domains),
            )

            # Domain tables (CSV + LaTeX), with the image name included
            write_domain_tables(out_tables, tag, caps, image_name=fig_path.name)

            print(f"  eps_q={eps_q:<5} saved {fig_path}")

    # Paper overlays: baseline vs robust for NHITS and NBEATS at paper_epsq
    peps = float(args.paper_epsq)

    def maybe_emit_pair(family: str):
        b = f"{family}_baseline"
        r = f"{family}_robust"
        if b in results and r in results and peps in results[b] and peps in results[r]:
            fig_path = plot_paper_overlay_baseline_vs_robust(
                outdir=out_paper,
                family=family,
                eps_q=peps,
                eps_f_scaled=eps_f_scaled,
                maxdq_baseline=results[b][peps],
                maxdq_robust=results[r][peps],
                caps=caps,
            )
            save_caption_tex(
                fig_path.with_suffix(".tex"),
                caption_paper_baseline_vs_robust(family, peps, eps_f_scaled, args.n_pairs, args.n_domains),
            )
            print(f"\n[paper] wrote {fig_path}")
        else:
            print(f"\n[paper] skipping {family}: need {b},{r} and eps_q={peps} present.")

    maybe_emit_pair("nhits")
    maybe_emit_pair("nbeats")

    print(f"\n[done] outputs in: {outroot}")


if __name__ == "__main__":
    main()

