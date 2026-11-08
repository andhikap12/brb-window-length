"""dann.py - Domain-Adversarial Inter-Source Alignment for Load-Robust Fault Classification."""

import torch
import torch.nn as nn

from src.models.grl import gradient_reversal
from src.models.heads import FaultHead, DomainHead
from src.models.backbones import build_backbone


class DANNModel(nn.Module):
    """Backbone-agnostic DANN: backbone -> (fault_head, domain_head via GRL)."""

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
