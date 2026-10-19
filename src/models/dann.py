"""
dann.py - Domain-Adversarial Inter-Source Alignment for Load-Robust Fault Classification.

Strategy: Domain Generalization (NOT Domain Adaptation)
-------------------------------------------------------
This model implements the "inter-source adversarial alignment" DG strategy.
The GRL forces the backbone to produce representations that are invariant
across ALL N-1 seen training domains (torque/speed levels). By induction,
these representations should generalise to unseen held-out domains.

    Training fold k (e.g., hold out torque T_k):
      Source domains: T_0, T_1, ..., T_{k-1}, T_{k+1}, ..., T_{N-1}
      GRL objective:  backbone CANNOT predict which of these N-1 torques a sample is from
      DG hypothesis:  if f(x) is torque-invariant across all seen torques,
                      it is also torque-invariant at unseen T_k.

Architecture Reference:
    Ganin et al. (2016). "Domain-Adversarial Training of Neural Networks." JMLR 17(59):1-35.
    Li et al. (2018). "Domain Generalization with Adversarial Feature Statistics."
    Zhao et al. (2020). "Domain Generalization via Entropy Regularization."

Application to PdM:
    Motor fault signatures (broken rotor bar sidebands) shift in frequency as
    load/torque changes due to varying slip frequency.  A standard classifier
    learns load-conditioned patterns.  DANN forces the backbone to learn
    load-INVARIANT fault representations by adversarially preventing the
    backbone from encoding which torque level it is processing.

    Input → Backbone (registry-selectable) → feat vector
                                              │
                      ┌───────────────────────┤
                      │                       │ (GRL negates gradient)
                 Fault Head              Domain Head
                 (minimize CE)           (minimize CE for head,
                                          maximize CE for backbone)

    Net effect:
        • Backbone: learns features useful for fault detection, useless for
          torque detection → load-invariant representation.
        • Fault head: clean fault discrimination.
        • Domain head: struggles to predict torque (desired behaviour).
          Convergence indicator: domain_loss → log(n_domains),
                                 domain_acc  → 1/n_domains (random chance).
"""

import torch
import torch.nn as nn

from src.models.grl import gradient_reversal
from src.models.heads import FaultHead, DomainHead
from src.models.backbones import build_backbone


# ---------------------------------------------------------------------------
# Backbone-agnostic DANN
# ---------------------------------------------------------------------------

class DANNModel(nn.Module):
    """Backbone-agnostic DANN: backbone -> (fault_head, domain_head via GRL).

    forward (training): (fault_logits, domain_logits)
    forward (eval):     fault_logits
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, n_classes: int, n_domains: int,
                 lambda_grl: float = 0.0, domain_hidden: int = 64) -> None:
        super().__init__()
        self.lambda_grl = lambda_grl
        self.backbone = backbone                         # (B,C,T) -> (B,feat_dim)
        self.fault_head = FaultHead(feat_dim, n_classes)
        self.domain_head = DomainHead(feat_dim, n_domains, hidden=domain_hidden)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        fault_logits = self.fault_head(feat)
        if self.training:
            rev = gradient_reversal(feat, self.lambda_grl)
            return fault_logits, self.domain_head(rev)
        return fault_logits

    def verify_domain_labels(self, y_domain: torch.Tensor) -> None:
        import math
        n_domains = self.domain_head.out_features
        unique = y_domain.unique().tolist()
        invalid = [int(l) for l in unique if int(l) < 0 or int(l) >= n_domains]
        if invalid:
            raise ValueError(
                f"[ABORT] Domain labels out of range [0, {n_domains-1}]: found {invalid}. "
                f"Ensure masked labels (-1) are excluded from training split."
            )
        print(f"  [GRL-DG] Domain label check PASSED: unique={sorted(int(l) for l in unique)}, "
              f"n_domains={n_domains}, max_entropy=log({n_domains})={math.log(n_domains):.4f}")

    def sanity_check(self, c_in: int, seq_len: int) -> None:
        n_classes = self.fault_head.out_features
        n_domains = self.domain_head.out_features
        device = next(self.parameters()).device
        was_training = self.training
        self.train()
        with torch.no_grad():
            out = self(torch.zeros(2, c_in, seq_len, device=device))
        self.train(was_training)
        if not isinstance(out, tuple) or len(out) != 2:
            raise RuntimeError(
                f"[ABORT] sanity_check: expected (fault, domain) tuple, got {type(out)}."
            )
        fault_logits, domain_logits = out
        if fault_logits.shape != (2, n_classes):
            raise RuntimeError(f"[ABORT] fault head shape {tuple(fault_logits.shape)} != (2, {n_classes}).")
        if domain_logits.shape != (2, n_domains):
            raise RuntimeError(f"[ABORT] domain head shape {tuple(domain_logits.shape)} != (2, {n_domains}).")
        print(f"  [OK] DANNModel sanity check PASSED: c_in={c_in}, seq_len={seq_len} -> "
              f"fault={tuple(fault_logits.shape)}, domain={tuple(domain_logits.shape)}")


def build_dann_model(backbone_name: str, c_in: int, seq_len: int, n_classes: int, n_domains: int,
                     lambda_grl: float = 0.0, domain_hidden: int = 64) -> DANNModel:
    """Build a DANNModel with the named backbone (feat_dim discovered automatically)."""
    backbone, feat_dim = build_backbone(backbone_name, c_in, seq_len)
    return DANNModel(backbone, feat_dim, n_classes, n_domains, lambda_grl, domain_hidden)
