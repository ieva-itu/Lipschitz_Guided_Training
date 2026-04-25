#!/usr/bin/env python3
"""
src/verify_robust_marabou.py  

Expected ONNX signature:
  Inputs:  I  [1], f1 [k], f2 [k]
  Outputs: q1 [1], q2 [1]

Property (negated SAT query, split into two directions):
  I in [0, I_max]
  f1,f2 in [0, F_max]^k
  ||f2 - f1||_inf <= eps_f
  and (q2 - q1 >= eps_q)  OR  (q1 - q2 >= eps_q)

We solve two queries (+ and -). Robustness holds iff UNSAT in both.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from maraboupy import Marabou, MarabouCore


# -------------------------
# Scaling.json loader (forward/backward compatible)
# -------------------------
def load_scaling(models_dir: Path) -> dict:
    p = models_dir / "scaling.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing scaling.json at {p}")
    d = json.loads(p.read_text())

    # Horizon: accept HORIZON/horizon/k
    if "HORIZON" not in d:
        if "horizon" in d:
            d["HORIZON"] = int(d["horizon"])
        elif "k" in d:
            d["HORIZON"] = int(d["k"])

    # Bounds: accept I_MAX_SCALED/F_MAX_SCALED or I_MAX/F_MAX
    if "I_MAX_SCALED" not in d:
        d["I_MAX_SCALED"] = float(d.get("I_MAX", 1.0))
    if "F_MAX_SCALED" not in d:
        d["F_MAX_SCALED"] = float(d.get("F_MAX", 2.0))

    # Forecast scaling:
    # old: f_scale (scaled = raw * f_scale)
    # new: f_scale_raw (scaled = raw / f_scale_raw) => multiplier 1/f_scale_raw
    if "f_scale" not in d:
        if "f_scale_raw" in d:
            d["f_scale"] = 1.0 / float(d["f_scale_raw"])
        else:
            d["f_scale"] = 1.0

    return d


def _as_int_var(v) -> int:
    if isinstance(v, (np.ndarray, list, tuple)):
        return int(np.array(v).reshape(-1)[0])
    return int(v)


def _add_leq(ipq, lhs_terms, rhs_scalar):
    """sum(ci*xi) <= rhs"""
    eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    for c, var in lhs_terms:
        eq.addAddend(float(c), _as_int_var(var))
    eq.setScalar(float(rhs_scalar))
    ipq.addEquation(eq)


def _add_geq(ipq, lhs_terms, rhs_scalar):
    """sum(ci*xi) >= rhs  <=>  -sum(ci*xi) <= -rhs"""
    _add_leq(ipq, [(-c, var) for (c, var) in lhs_terms], -float(rhs_scalar))


def _exitcode_to_str(exit_code) -> str:
    if isinstance(exit_code, str):
        return exit_code.lower()
    return str(exit_code).lower()


def _get_io_vars_from_network(net):
    """
    Expect 3 inputs: I, f1, f2 and 2 outputs: q1, q2.
    This matches your observed sizes: input blocks [1,7,7], output blocks [1,1].
    """
    if len(net.inputVars) != 3:
        raise RuntimeError(f"Expected 3 inputs (I,f1,f2), got {len(net.inputVars)} with sizes "
                           f"{[len(np.array(v).reshape(-1)) for v in net.inputVars]}")
    if len(net.outputVars) != 2:
        raise RuntimeError(f"Expected 2 outputs (q1,q2), got {len(net.outputVars)} with sizes "
                           f"{[len(np.array(v).reshape(-1)) for v in net.outputVars]}")

    I_vars = np.array(net.inputVars[0]).reshape(-1)
    f1_vars = np.array(net.inputVars[1]).reshape(-1)
    f2_vars = np.array(net.inputVars[2]).reshape(-1)

    q1_vars = np.array(net.outputVars[0]).reshape(-1)
    q2_vars = np.array(net.outputVars[1]).reshape(-1)

    if I_vars.size != 1:
        raise RuntimeError(f"Expected scalar I input, got {I_vars.size}")
    if q1_vars.size != 1 or q2_vars.size != 1:
        raise RuntimeError(f"Expected scalar outputs, got q1={q1_vars.size}, q2={q2_vars.size}")
    if f1_vars.size != f2_vars.size:
        raise RuntimeError(f"Expected f1/f2 same size, got {f1_vars.size} vs {f2_vars.size}")

    I = _as_int_var(I_vars[0])
    f1 = [_as_int_var(v) for v in f1_vars]
    f2 = [_as_int_var(v) for v in f2_vars]
    q1 = _as_int_var(q1_vars[0])
    q2 = _as_int_var(q2_vars[0])

    return I, f1, f2, q1, q2


def build_ipq_for_direction(
    model_path: str,
    H: int,
    I_min: float, I_max: float,
    F_min: float, F_max: float,
    eps_f: float,
    eps_q: float,
    sign: str,
    verbose: bool,
):
    """
    sign="+" encodes q2 - q1 >= eps_q
    sign="-" encodes q1 - q2 >= eps_q
    """
    assert sign in {"+", "-"}

    net = Marabou.read_onnx(model_path)
    ipq = net.getMarabouQuery()

    I_var, f1_vars, f2_vars, q1, q2 = _get_io_vars_from_network(net)
    k = len(f1_vars)
    if k != H:
        # Don't silently mismatch: it causes weird behavior
        raise RuntimeError(f"H mismatch: scaling H={H} but model has k={k}")

    if verbose:
        print(f"[info] Parsed model k={k} inputs=[I,f1,f2] outputs=[q1,q2]")
        print(f"[info] Bounds: I∈[{I_min},{I_max}] f∈[{F_min},{F_max}] eps_f={eps_f} eps_q={eps_q}")

    # Bounds
    ipq.setLowerBound(I_var, float(I_min))
    ipq.setUpperBound(I_var, float(I_max))
    for v in f1_vars + f2_vars:
        ipq.setLowerBound(v, float(F_min))
        ipq.setUpperBound(v, float(F_max))

    # ||f2 - f1||_inf <= eps_f
    for v1, v2 in zip(f1_vars, f2_vars):
        _add_leq(ipq, [(1.0, v2), (-1.0, v1)], float(eps_f))
        _add_leq(ipq, [(1.0, v1), (-1.0, v2)], float(eps_f))

    # Negated robustness direction
    if sign == "+":
        _add_geq(ipq, [(1.0, q2), (-1.0, q1)], float(eps_q))
    else:
        _add_geq(ipq, [(1.0, q1), (-1.0, q2)], float(eps_q))

    dbg = dict(I_var=I_var, f1=f1_vars, f2=f2_vars, q1=q1, q2=q2, k=k)
    return ipq, dbg


def solve_direction(
    model_path: str,
    H: int,
    I_min: float, I_max: float,
    F_min: float, F_max: float,
    eps_f: float,
    eps_q: float,
    sign: str,
    timeout: int,
    verbosity: int,
    verbose: bool,
):
    print(f"=== Robustness check (sign='{sign}') model={model_path} ===")

    ipq, dbg = build_ipq_for_direction(
        model_path=model_path,
        H=H,
        I_min=I_min, I_max=I_max,
        F_min=F_min, F_max=F_max,
        eps_f=eps_f,
        eps_q=eps_q,
        sign=sign,
        verbose=verbose,
    )

    options = Marabou.createOptions(timeoutInSeconds=int(timeout), verbosity=int(verbosity))

    t0 = time.time()
    exit_code, vals, stats = MarabouCore.solve(ipq, options, "")
    dt = time.time() - t0

    exit_s = _exitcode_to_str(exit_code)
    print(f"[result] exitCode={exit_code} elapsed={dt:.3f}s")

    # UNSAT if no assignment
    if "unsat" in exit_s or vals is None or len(vals) == 0:
        print("UNSAT: no counterexample in this direction.")
        return False

    if "sat" in exit_s:
        print("SAT: found counterexample.")
        q1 = dbg["q1"]; q2 = dbg["q2"]
        I_var = dbg["I_var"]
        f1_vars = dbg["f1"]; f2_vars = dbg["f2"]

        q1_val = float(vals[q1]); q2_val = float(vals[q2])
        print(f"  q1={q1_val} q2={q2_val} |Δq|={abs(q2_val-q1_val)}")
        I_val = float(vals[I_var])
        f1_val = [float(vals[v]) for v in f1_vars]
        f2_val = [float(vals[v]) for v in f2_vars]
        print(f"  I={I_val}")
        print(f"  f1[:5]={f1_val[:5]}")
        print(f"  f2[:5]={f2_val[:5]}")
        print(f"  max|f2-f1|={max(abs(a-b) for a,b in zip(f1_val,f2_val))}")
        return True

    print("UNKNOWN/OTHER: solver did not return SAT/UNSAT cleanly.")
    return None


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--models-dir", type=str, default="models", help="directory containing scaling.json")
    ap.add_argument("--model-path", type=str, default="models/policy/policy_two_copy.onnx")

    ap.add_argument("--eps-q", type=float, required=True, help="εq in policy output units")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--epsf-scaled", type=float, help="εf in *scaled* forecast units")
    g.add_argument("--epsf-raw", type=float, help="εf in *raw* forecast units (converted using f_scale)")

    ap.add_argument("--i-max-scaled", type=float, default=None)
    ap.add_argument("--f-max-scaled", type=float, default=None)

    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--marabou-verbosity", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    scaling = load_scaling(Path(args.models_dir))
    H = int(scaling["HORIZON"])
    f_scale = float(scaling["f_scale"])

    # epsf conversion:
    # old convention: epsf_raw * f_scale  (since scaled = raw * f_scale)
    eps_f = float(args.epsf_scaled) if args.epsf_scaled is not None else float(args.epsf_raw) * f_scale

    I_min = 0.0
    F_min = 0.0
    I_max = float(args.i_max_scaled) if args.i_max_scaled is not None else float(scaling["I_MAX_SCALED"])
    F_max = float(args.f_max_scaled) if args.f_max_scaled is not None else float(scaling["F_MAX_SCALED"])

    if args.verbose:
        print(f"[cfg] H={H} I∈[{I_min},{I_max}] F∈[{F_min},{F_max}] eps_f={eps_f} eps_q={args.eps_q}")
        print(f"[cfg] models-dir={args.models_dir}")
        print(f"[cfg] model-path={args.model_path}")

    sat_plus = solve_direction(
        model_path=args.model_path,
        H=H,
        I_min=I_min, I_max=I_max,
        F_min=F_min, F_max=F_max,
        eps_f=eps_f,
        eps_q=float(args.eps_q),
        sign="+",
        timeout=int(args.timeout),
        verbosity=int(args.marabou_verbosity),
        verbose=bool(args.verbose),
    )

    sat_minus = solve_direction(
        model_path=args.model_path,
        H=H,
        I_min=I_min, I_max=I_max,
        F_min=F_min, F_max=F_max,
        eps_f=eps_f,
        eps_q=float(args.eps_q),
        sign="-",
        timeout=int(args.timeout),
        verbosity=int(args.marabou_verbosity),
        verbose=bool(args.verbose),
    )

    if sat_plus is False and sat_minus is False:
        print("\n✅ ROBUSTNESS PROVED: UNSAT in both directions.")
        return 0
    else:
        print("\n❌ ROBUSTNESS NOT PROVED: SAT/UNKNOWN in at least one direction.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

