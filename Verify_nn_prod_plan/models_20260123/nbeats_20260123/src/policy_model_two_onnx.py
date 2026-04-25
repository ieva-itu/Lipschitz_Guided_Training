#!/usr/bin/env python3
"""
src/policy_model_two_onnx.py

Self-contained policy model + training + Marabou-friendly ONNX export.

Designed to be imported by scripts/train_policy_two_onnx_both.py:

    from src.policy_model_two_onnx import (
        OrderingPolicy, RobustConfig, train_policy,
        export_policy_onnx, export_two_copy_onnx_noslice,
        make_plain_from_sn, estimate_lipschitz_upper_bound_plain,
        write_lipschitz_metadata,
    )

Key points:
- Optional spectral_norm during training (Lipschitz-ish control).
- Robustness hinge penalty training (optional; can be disabled with robust_weight=0 and/or 0 samples).
- Export avoids SN parametrization artifacts by copying effective weights to a plain model.
- Two-copy ONNX exported without slicing packed inputs: (I, f1, f2) -> (q1, q2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.utils import spectral_norm


__all__ = [
    "RobustConfig",
    "OrderingPolicy",
    "train_policy",
    "make_plain_from_sn",
    "estimate_lipschitz_upper_bound_plain",
    "write_lipschitz_metadata",
    "export_policy_onnx",
    "export_two_copy_onnx_noslice",
]


# ============================================================
#  Robust training config
# ============================================================

@dataclass
class RobustConfig:
    pert_radius: float = 1.0
    eps_q: float = 0.1
    robust_weight: float = 10.0
    robust_num_samples: int = 6
    robust_use_fgsm: bool = True
    perturb_forecast_only: bool = True


# ============================================================
#  Policy network
# ============================================================

class OrderingPolicy(nn.Module):
    """
    Small MLP:
      input:  [I, f0, ..., f_{k-1}]  (dim = 1 + k)
      output: scalar q  (shape [B,1])

    If use_spectral_norm=True, wraps Linear layers with spectral_norm.
    For ONNX export, always export a plain (non-SN) copy.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 20,
        use_spectral_norm: bool = True,
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_spectral_norm = bool(use_spectral_norm)
        self.output_scale = float(output_scale)

        def lin(in_f: int, out_f: int) -> nn.Module:
            layer = nn.Linear(in_f, out_f)
            if self.use_spectral_norm:
                layer = spectral_norm(layer)
            return layer

        self.net = nn.Sequential(
            lin(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            lin(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            lin(self.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns [B,1]
        return self.net(x) * self.output_scale


# ============================================================
#  Proxy cost + robustness loss
# ============================================================

def _cost_one_step(
    I: torch.Tensor,
    q: torch.Tensor,
    d: torch.Tensor,
    h: float = 0.1,
    p: float = 1.0,
    c: float = 0.0,
) -> torch.Tensor:
    """
    post = I + q - d
    holding  ~ relu(post)
    stockout ~ relu(-post)
    order_cost = c*q
    """
    post = I + q - d
    holding = h * torch.relu(post)
    stockout = p * torch.relu(-post)
    order_cost = c * q
    return holding + stockout + order_cost


def _robustness_loss(
    policy: OrderingPolicy,
    x: torch.Tensor,
    q_base_raw: torch.Tensor,
    cfg: RobustConfig,
) -> torch.Tensor:
    """
    Hinge robustness penalty:
      mean max(0, |q(x+δ)-q(x)| - eps_q),  ||δ||_inf <= pert_radius

    If robust_num_samples <= 0 and robust_use_fgsm == False, returns 0.
    """
    # Fully disable robustness loss if configured off
    if int(cfg.robust_num_samples) <= 0 and (not bool(cfg.robust_use_fgsm)):
        return torch.zeros((), device=x.device)

    device = x.device
    B, D = x.shape

    # Mask for forecast-only perturbations
    if cfg.perturb_forecast_only:
        mask = torch.zeros((B, D), device=device)
        mask[:, 1:] = 1.0
    else:
        mask = torch.ones((B, D), device=device)

    losses: List[torch.Tensor] = []

    # Random perturbations
    for _ in range(int(cfg.robust_num_samples)):
        delta = (2.0 * torch.rand((B, D), device=device) - 1.0) * float(cfg.pert_radius)
        delta = delta * mask
        q_pert = policy(x + delta)
        dq = torch.abs(q_pert - q_base_raw)
        losses.append(torch.relu(dq - float(cfg.eps_q)).mean())

    # FGSM-like perturbation
    if bool(cfg.robust_use_fgsm):
        x_adv = x.detach().clone().requires_grad_(True)
        q_adv = policy(x_adv)
        obj = torch.abs(q_adv - q_base_raw.detach()).mean()
        obj.backward()
        grad = x_adv.grad.detach()
        delta = float(cfg.pert_radius) * torch.sign(grad) * mask
        q_pert = policy(x.detach() + delta)
        dq = torch.abs(q_pert - q_base_raw.detach())
        losses.append(torch.relu(dq - float(cfg.eps_q)).mean())

    return torch.stack(losses).mean()


# ============================================================
#  Training
# ============================================================

def train_policy(
    policy: OrderingPolicy,
    episodes: List[Tuple[float, np.ndarray, np.ndarray]],
    lr: float = 1e-3,
    epochs: int = 120,
    log_interval: int = 10,
    Q_min: float = 0.0,
    Q_max: float = 100.0,
    robust_cfg: Optional[RobustConfig] = None,
    seed: int = 0,
) -> OrderingPolicy:
    """
    Flatten episodes into (state, demand) samples and optimize:
        base_cost + robust_weight * robustness_hinge

    Baseline training:
        robust_cfg.robust_weight = 0 (and/or 0 samples, FGSM off)
        policy.use_spectral_norm = False (recommended)
    """
    if robust_cfg is None:
        robust_cfg = RobustConfig()

    device = next(policy.parameters()).device
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Flatten dataset
    states: List[np.ndarray] = []
    demands: List[float] = []
    for (I0, demand_seg, fore_seg) in episodes:
        I = float(I0)
        for t in range(len(demand_seg)):
            d = float(demand_seg[t])
            f = fore_seg[t].astype(np.float32)
            s = np.concatenate([[I], f], axis=0).astype(np.float32)  # [1+k]
            states.append(s)
            demands.append(d)

    X = torch.tensor(np.stack(states), dtype=torch.float32, device=device)                # [N,1+k]
    Dm = torch.tensor(np.array(demands), dtype=torch.float32, device=device).unsqueeze(1)  # [N,1]
    N = X.shape[0]
    batch_size = min(512, N)

    opt = optim.Adam(policy.parameters(), lr=lr)
    policy.train()

    for ep in range(1, int(epochs) + 1):
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        total_base = 0.0
        total_rob = 0.0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            x = X[idx]
            d = Dm[idx]

            q_raw = policy(x)  # [B,1]
            q = torch.clamp(q_raw, float(Q_min), float(Q_max))

            I = x[:, :1]
            base = _cost_one_step(I, q, d).mean()
            rob = _robustness_loss(policy, x, q_raw, robust_cfg)

            loss = base + float(robust_cfg.robust_weight) * rob

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()

            total_loss += float(loss.detach().cpu())
            total_base += float(base.detach().cpu())
            total_rob += float(rob.detach().cpu())

        if ep % int(log_interval) == 0 or ep == 1 or ep == int(epochs):
            print(
                f"[train_policy] epoch={ep:04d} "
                f"loss={total_loss:.4f} base={total_base:.4f} rob={total_rob:.4f} "
                f"(w={robust_cfg.robust_weight}, r={robust_cfg.pert_radius}, eps={robust_cfg.eps_q})"
            )

    policy.eval()
    return policy


# ============================================================
#  Plain copy builder (SN-safe)
# ============================================================

def make_plain_from_sn(policy_sn: OrderingPolicy) -> OrderingPolicy:
    """
    Create a plain (non-spectral-norm) OrderingPolicy and copy effective weights.

    Works even if policy_sn was already non-SN; still returns a plain copy.
    """
    plain = OrderingPolicy(
        input_dim=policy_sn.input_dim,
        hidden_dim=policy_sn.hidden_dim,
        use_spectral_norm=False,
        output_scale=policy_sn.output_scale,
    )

    sn_linears = [m for m in policy_sn.net.modules() if isinstance(m, nn.Linear)]
    pl_linears = [m for m in plain.net.modules() if isinstance(m, nn.Linear)]

    if len(sn_linears) != len(pl_linears):
        raise RuntimeError(f"Unexpected Linear count: sn={len(sn_linears)} plain={len(pl_linears)}")

    with torch.no_grad():
        for sn_l, pl_l in zip(sn_linears, pl_linears):
            pl_l.weight.copy_(sn_l.weight.detach().cpu())
            if sn_l.bias is not None:
                pl_l.bias.copy_(sn_l.bias.detach().cpu())

    plain.eval()
    return plain


# ============================================================
#  Lipschitz bound (plain model)
# ============================================================

@torch.no_grad()
def estimate_lipschitz_upper_bound_plain(policy_plain: OrderingPolicy) -> float:
    """
    Conservative bound:
        L <= output_scale * Π ||W_i||_2   (ReLU is 1-Lipschitz)
    """
    prod = 1.0
    for m in policy_plain.modules():
        if isinstance(m, nn.Linear):
            sigma = torch.linalg.matrix_norm(m.weight, ord=2).item()
            prod *= float(sigma)
    prod *= float(getattr(policy_plain, "output_scale", 1.0))
    return float(prod)


def write_lipschitz_metadata(
    out_dir: Path,
    L_hat: float,
    k: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    meta: Dict[str, Any] = {"L_hat": float(L_hat), "k": int(k)}
    if extra:
        meta.update(extra)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lipschitz.json").write_text(json.dumps(meta, indent=2))
    print(f"[lipschitz] wrote: {out_dir / 'lipschitz.json'} (L_hat={L_hat:.6g})")


# ============================================================
#  ONNX export: single-copy (EXPORT PLAIN)
# ============================================================

def export_policy_onnx(
    policy_sn: OrderingPolicy,
    input_dim: int,
    path: str = "models/policy/policy.onnx",
) -> None:
    """
    Export a plain (non-SN) ONNX model:

        input:  state [B, 1+k]
        output: q     [B, 1]
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    policy_plain = make_plain_from_sn(policy_sn)

    dummy = torch.zeros((1, int(input_dim)), dtype=torch.float32)
    torch.onnx.export(
        policy_plain,
        dummy,
        out_path.as_posix(),
        input_names=["state"],
        output_names=["q"],
        dynamic_axes={"state": {0: "batch"}, "q": {0: "batch"}},
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"[export_policy_onnx] wrote: {out_path}")


# ============================================================
#  Two-copy wrapper WITHOUT packed vector slicing (Marabou-friendly)
# ============================================================

class TwoCopyNoSlice(nn.Module):
    """
    Inputs:
      I  [B,1]
      f1 [B,k]
      f2 [B,k]
    Outputs:
      q1 [B,1]
      q2 [B,1]
    """

    def __init__(self, policy_plain: nn.Module):
        super().__init__()
        self.policy = policy_plain

    def forward(self, I: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor):
        if I.dim() == 1:
            I = I.unsqueeze(1)
        x1 = torch.cat([I, f1], dim=1)
        x2 = torch.cat([I, f2], dim=1)
        q1 = self.policy(x1)
        q2 = self.policy(x2)
        return q1, q2


def export_two_copy_onnx_noslice(
    policy_sn: OrderingPolicy,
    k: int,
    path: str = "models/policy/policy_two_copy.onnx",
) -> None:
    """
    Export two-copy NO-SLICE ONNX:

      Inputs:  I [B,1], f1 [B,k], f2 [B,k]
      Outputs: q1 [B,1], q2 [B,1]
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    policy_plain = make_plain_from_sn(policy_sn)
    wrapper = TwoCopyNoSlice(policy_plain).eval()

    I = torch.zeros((1, 1), dtype=torch.float32)
    f1 = torch.zeros((1, int(k)), dtype=torch.float32)
    f2 = torch.zeros((1, int(k)), dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (I, f1, f2),
        out_path.as_posix(),
        input_names=["I", "f1", "f2"],
        output_names=["q1", "q2"],
        dynamic_axes={
            "I":  {0: "batch"},
            "f1": {0: "batch"},
            "f2": {0: "batch"},
            "q1": {0: "batch"},
            "q2": {0: "batch"},
        },
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"[export_two_copy_onnx_noslice] wrote: {out_path}")

