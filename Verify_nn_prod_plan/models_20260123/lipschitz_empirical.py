#!/usr/bin/env python3
"""lipschitz_empirical.py — Lipschitz Constant Verification (Theorem 1) + empirical validation.

Goal :
  - Definition 1 (Lipschitz continuity): for fixed I, |π(I,f2)-π(I,f1)| ≤ L · ||f2-f1||_∞
  - Theorem 1 (layerwise bound for ReLU MLP): \hat L ≤ Π_ℓ ||W_ℓ||₂
  - Empirical check:
        L_emp := max_{samples} |Δq| / ||Δf||_∞
    and compare to \hat L; also compare max|Δq| to \hat L · ε_f.

Units:
  - Uses *scaled* inputs (I in [0,1], f in [0,2]^H), consistent with Marabou specs.
  - If passed --epsf-raw, converts ε_f(raw) to ε_f(scaled) using f_scale in scaling.json.

Supported ONNX formats:
  (A) single-input policy: x=[I,f] -> q
  (B) two-copy verification policy: inputs (I,f1,f2) -> outputs (q1,q2)

Outputs:
  - lipschitz_validation_table.csv
  - lipschitz_vs_empirical.png      (\hat L·ε_f vs observed max|Δq|)
  - Lemp_vs_Lhat.png                (scatter: L_emp vs \hat L)
  - lipschitz_vs_empirical.tex      (caption snippet)

"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import onnxruntime as ort
except ImportError as e:
    raise SystemExit("onnxruntime is required: pip install onnxruntime") from e


# -----------------------------
# Scaling
# -----------------------------
@dataclass(frozen=True)
class Scaling:
    f_scale: float

    @staticmethod
    def load(path: Path) -> "Scaling":
        d = json.loads(path.read_text())
        return Scaling(f_scale=float(d.get("f_scale", 1.0 / 50.0)))


# -----------------------------
# ONNX wrapper + Lipschitz bound
# -----------------------------
def _safe_int(x):
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


class PolicyONNX:
    """Wrapper for either single-input or two-copy (I,f1,f2)->(q1,q2) policies."""

    def __init__(self, onnx_path: Path):
        self.onnx_path = onnx_path
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.inputs = self.sess.get_inputs()
        self.outputs = self.sess.get_outputs()

        self.in_names = [i.name for i in self.inputs]
        self.out_names = [o.name for o in self.outputs]

        if len(self.in_names) == 1:
            self.mode = "single"
        elif len(self.in_names) == 3:
            self.mode = "twocopy"
        else:
            raise RuntimeError(f"Unsupported input arity {len(self.in_names)} for {onnx_path}")

        def shape2(info):
            shp = list(info.shape)
            if len(shp) != 2:
                return None
            return _safe_int(shp[1])

        if self.mode == "single":
            dim = shape2(self.inputs[0])
            if dim is None or dim < 2:
                raise RuntimeError(f"Cannot infer H from single input shape={self.inputs[0].shape} ({onnx_path})")
            self.H = dim - 1
            self.role_x = self.in_names[0]
            self.role_I = None
            self.role_f1 = None
            self.role_f2 = None
            if len(self.out_names) != 1:
                raise RuntimeError(f"Single-input policy should have 1 output, got {self.out_names} ({onnx_path})")
            self.out_q = self.out_names[0]
        else:
            dims = [(n, shape2(info)) for n, info in zip(self.in_names, self.inputs)]
            cand_I = [n for (n, d2) in dims if d2 == 1]
            if len(cand_I) != 1:
                raise RuntimeError(f"Expected exactly one I input with dim2=1, got {dims} ({onnx_path})")
            self.role_I = cand_I[0]

            cand_f = [(n, d2) for (n, d2) in dims if d2 not in (None, 1)]
            if len(cand_f) != 2:
                raise RuntimeError(f"Expected two forecast inputs with same dim2, got {dims} ({onnx_path})")
            if cand_f[0][1] != cand_f[1][1]:
                raise RuntimeError(f"f1/f2 dim mismatch: {cand_f} ({onnx_path})")
            self.H = int(cand_f[0][1])
            # keep their names (often f1,f2)
            self.role_f1 = cand_f[0][0]
            self.role_f2 = cand_f[1][0]
            self.role_x = None

            if len(self.out_names) != 2:
                raise RuntimeError(f"Two-copy policy should have 2 outputs, got {self.out_names} ({onnx_path})")
            self.out_q1, self.out_q2 = self.out_names[0], self.out_names[1]
            self.out_q = None

    def eval_single(self, I: float, f: np.ndarray) -> float:
        if self.mode != "single":
            raise RuntimeError("eval_single requires single-input policy")
        x = np.concatenate([[I], f.reshape(-1)]).astype(np.float32).reshape(1, -1)
        out = self.sess.run([self.out_q], {self.role_x: x})[0]
        return float(np.array(out).reshape(-1)[0])

    def eval_twocopy(self, I: float, f1: np.ndarray, f2: np.ndarray) -> Tuple[float, float]:
        if self.mode != "twocopy":
            raise RuntimeError("eval_twocopy requires two-copy policy")
        feed = {
            self.role_I: np.array([[I]], dtype=np.float32),
            self.role_f1: f1.astype(np.float32).reshape(1, -1),
            self.role_f2: f2.astype(np.float32).reshape(1, -1),
        }
        q1 = self.sess.run([self.out_q1], feed)[0]
        q2 = self.sess.run([self.out_q2], feed)[0]
        return float(np.array(q1).reshape(-1)[0]), float(np.array(q2).reshape(-1)[0])


def lipschitz_bound_from_onnx(onnx_path: Path) -> float:
    """Compute \hat L := Π ||W||_2 over Gemm/MatMul weights, incl. constant Mul scalars."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    graph = model.graph
    inits = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    def spec_norm(W: np.ndarray) -> float:
        W = np.asarray(W, dtype=np.float64)
        if W.ndim != 2:
            W = W.reshape(W.shape[0], -1)
        s = np.linalg.svd(W, compute_uv=False)
        return float(s[0]) if s.size else 0.0

    L = 1.0
    used_any = False

    for node in graph.node:
        if node.op_type in ("Gemm", "MatMul"):
            if len(node.input) >= 2 and node.input[1] in inits:
                L *= spec_norm(inits[node.input[1]])
                used_any = True

        if node.op_type == "Mul":
            scal = None
            for inp in node.input:
                if inp in inits and inits[inp].size == 1:
                    scal = float(inits[inp].reshape(-1)[0])
                    break
            if scal is not None:
                L *= abs(scal)

    if not used_any:
        raise RuntimeError(f"No Gemm/MatMul weights found in {onnx_path}; cannot compute \hat L.")
    return float(L)


# -----------------------------
# Empirical sampling
# -----------------------------
def sample_pair(
    rng: np.random.Generator,
    H: int,
    I_cap: float,
    F_cap: float,
    eps_f: float,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    I = float(rng.uniform(0.0, I_cap))
    f1 = rng.uniform(0.0, F_cap, size=(H,)).astype(np.float32)
    delta = rng.uniform(-eps_f, eps_f, size=(H,)).astype(np.float32)
    f2 = np.clip(f1 + delta, 0.0, F_cap).astype(np.float32)
    df = float(np.max(np.abs(f2 - f1)))
    return I, f1, f2, df


def empirical_lipschitz(
    pol: PolicyONNX,
    eps_f: float,
    n_pairs: int,
    seed: int,
    I_cap: float,
    F_cap: float,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    worst_ratio = 0.0
    worst_dq = 0.0

    for _ in range(n_pairs):
        I, f1, f2, df = sample_pair(rng, pol.H, I_cap, F_cap, eps_f)

        if pol.mode == "twocopy":
            q1, q2 = pol.eval_twocopy(I, f1, f2)
            dq = abs(q2 - q1)
        else:
            q1 = pol.eval_single(I, f1)
            q2 = pol.eval_single(I, f2)
            dq = abs(q2 - q1)

        worst_dq = max(worst_dq, dq)
        if df > 1e-12:
            worst_ratio = max(worst_ratio, dq / df)

    return float(worst_ratio), float(worst_dq)


# -----------------------------
# Plotting (grayscale)
# -----------------------------
@dataclass
class Row:
    controller: str
    H: int
    eps_f: float
    n_pairs: int
    L_hat: float
    L_emp: float
    max_abs_dq: float
    bound_Lhat_epsf: float
    bound_holds: bool


def plot_lipschitz_vs_empirical(rows: List[Row], out_png: Path) -> None:
    labels = [r.controller for r in rows]
    x = np.arange(len(rows))
    bound = np.array([r.bound_Lhat_epsf for r in rows], dtype=np.float64)
    obs = np.array([r.max_abs_dq for r in rows], dtype=np.float64)

    fig = plt.figure(figsize=(7.0, 2.6))
    ax = plt.gca()
    ax.plot(x, bound, linestyle="--", linewidth=1.6, color="0.35", label=r"$\hat L\cdot\varepsilon_f$")
    ax.plot(x, obs, linestyle="-", linewidth=1.8, color="k", marker="o", markersize=4.0,
            label=r"observed $\max|\Delta q|$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("value")
    ax.set_title(r"Lipschitz validation: observed $\max|\Delta q|$ vs $\hat L\cdot \varepsilon_f$")
    ax.grid(True, axis="y", linewidth=0.4, color="0.9")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="0.85", framealpha=0.95)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=360, bbox_inches="tight")
    plt.close(fig)


def plot_Lemp_vs_Lhat(rows: List[Row], out_png: Path) -> None:
    Lhat = np.array([r.L_hat for r in rows], dtype=np.float64)
    Lemp = np.array([r.L_emp for r in rows], dtype=np.float64)

    fig = plt.figure(figsize=(3.4, 3.0))
    ax = plt.gca()
    ax.scatter(Lhat, Lemp, s=25, c="k")
    lo = min(Lhat.min(), Lemp.min()) * 0.95
    hi = max(Lhat.max(), Lemp.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="0.35")
    ax.set_xlabel(r"$\hat L$")
    ax.set_ylabel(r"$L_{\mathrm{emp}}$")
    ax.set_title(r"Empirical Lipschitz vs bound")
    ax.grid(True, linewidth=0.4, color="0.9")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=360, bbox_inches="tight")
    plt.close(fig)


def parse_map(items: List[str]) -> Dict[str, Path]:
    m: Dict[str, Path] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"Expected key=path entry, got: {it}")
        k, p = it.split("=", 1)
        m[k.strip()] = Path(p).expanduser()
    return m


def write_csv(rows: List[Row], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", "H", "eps_f", "n_pairs", "L_hat", "L_emp",
                    "max_abs_dq", "Lhat_times_epsf", "bound_holds"])
        for r in rows:
            w.writerow([r.controller, r.H, r.eps_f, r.n_pairs, r.L_hat, r.L_emp,
                        r.max_abs_dq, r.bound_Lhat_epsf, int(r.bound_holds)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", required=True)
    ap.add_argument("--policy-onnx", nargs="+", required=True, help="Mappings tag=path_to_onnx")
    ap.add_argument("--scaling-json", type=str, required=True)

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--epsf-scaled", type=float)
    g.add_argument("--epsf-raw", type=float)

    ap.add_argument("--n-pairs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--I-cap", type=float, default=1.0)
    ap.add_argument("--F-cap", type=float, default=2.0)

    ap.add_argument("--outdir", type=str, default="empirical_out")
    ap.add_argument("--timestamp", action="store_true", default=True)
    args = ap.parse_args()

    scaling = Scaling.load(Path(args.scaling_json).expanduser())
    eps_f = float(args.epsf_scaled) if args.epsf_scaled is not None else float(args.epsf_raw) * scaling.f_scale

    outroot = Path(args.outdir).expanduser()
    if args.timestamp:
        outroot = outroot / f"lipschitz_out_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outroot.mkdir(parents=True, exist_ok=True)

    policy_map = parse_map(args.policy_onnx)

    rows: List[Row] = []
    for tag in args.controllers:
        if tag not in policy_map:
            raise SystemExit(f"Missing --policy-onnx entry for {tag}")
        pol = PolicyONNX(policy_map[tag])
        L_hat = lipschitz_bound_from_onnx(policy_map[tag])

        L_emp, max_dq = empirical_lipschitz(
            pol=pol,
            eps_f=eps_f,
            n_pairs=int(args.n_pairs),
            seed=int(args.seed),
            I_cap=float(args.I_cap),
            F_cap=float(args.F_cap),
        )

        bound = float(L_hat * eps_f)
        holds = bool(max_dq <= bound + 1e-9)

        rows.append(Row(
            controller=tag,
            H=int(pol.H),
            eps_f=float(eps_f),
            n_pairs=int(args.n_pairs),
            L_hat=float(L_hat),
            L_emp=float(L_emp),
            max_abs_dq=float(max_dq),
            bound_Lhat_epsf=float(bound),
            bound_holds=holds,
        ))

        print(f"[ok] {tag}: H={pol.H}  L_hat={L_hat:.6g}  L_emp={L_emp:.6g}  max|dq|={max_dq:.6g}  L_hat*eps_f={bound:.6g}  holds={holds}")

    out_csv = outroot / "lipschitz_validation_table.csv"
    write_csv(rows, out_csv)

    plot_lipschitz_vs_empirical(rows, outroot / "lipschitz_vs_empirical.png")
    plot_Lemp_vs_Lhat(rows, outroot / "Lemp_vs_Lhat.png")

    (outroot / "lipschitz_vs_empirical.tex").write_text(
        """\\caption{\\textbf{Lipschitz validation (Theorem~1).}
For each controller we compute the layerwise bound $\\hat L=\\prod_\\ell\\|W_\\ell\\|_2$ (including any constant output scaling)
and empirically estimate $L_{\\mathrm{emp}}=\\max |\\Delta q|/\\|\\Delta f\\|_\\infty$ over sampled pairs with $\\|\\Delta f\\|_\\infty\\le\\varepsilon_f$.
The plot compares the observed maximum deviation $\\max|\\Delta q|$ to the theoretical upper bound $\\hat L\\cdot\\varepsilon_f$.}
"""
    )

    print(f"[done] wrote: {out_csv}")
    print(f"[done] wrote: {outroot / 'lipschitz_vs_empirical.png'}")
    print(f"[done] wrote: {outroot / 'Lemp_vs_Lhat.png'}")


if __name__ == "__main__":
    main()
