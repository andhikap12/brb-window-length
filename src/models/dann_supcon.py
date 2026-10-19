"""dann_supcon.py — DANN + Supervised Contrastive (gabungan).

Mempertahankan DANN (GRL adversarial) sebagai METODE INTI,
lalu MENAMBAH SupCon (lengan DANN + SupCon, Tabel I makalah CENIM 2026) untuk menstabilkan: cabang adversarial mendorong backbone
tak meng-encode beban, cabang contrastive menarik same-fault lintas-beban saling
dekat. Hipotesis: SupCon meredam negative-transfer DANN (yang sendirian selalu
kalah) tanpa membuang sinyal fault.

    x → [backbone] → feat → fault_head            (CE)
                          ↘ GRL(λ) → domain_head  (CE adversarial — ciri DANN)
                          ↘ proj_head → z (L2)    (SupCon loss, pakai label fault)

Loss total (DANNSupConCallback):
    L = CE_fault + CE_domain(GRL feat) + λ_sc · SupCon(z, fault_labels)
λ_grl dijadwalkan 0→λ_max (anneal Ganin). SupCon TIDAK pakai label domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.grl import gradient_reversal
from src.models.heads import FaultHead, DomainHead, DomainRegHead
from src.models.backbones import build_backbone


class DANNSupConModel(nn.Module):
    """DANN (GRL domain head) + SupCon projection head on a shared backbone.

    forward (training): (fault_logits, domain_logits, z_embed)
    forward (eval):     fault_logits
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, n_classes: int, n_domains: int,
                 lambda_grl: float = 0.0, domain_hidden: int = 64,
                 proj_hidden: int = 128, proj_dim: int = 128,
                 domain_regression: bool = False) -> None:
        super().__init__()
        self.lambda_grl = lambda_grl
        self.backbone = backbone
        self.fault_head = FaultHead(feat_dim, n_classes)
        # UJI A: label domain yang MENGALIR di pipeline tetap indeks kategoris 0..n-1
        # (splits/dataloader tidak tersentuh). Yang berubah hanya bentuk KEPALAnya;
        # konversi indeks -> skalar u dikerjakan callback lewat tabel per-fold.
        self.n_domains = n_domains
        self.domain_regression = domain_regression
        self.domain_head = (
            DomainRegHead(feat_dim, hidden=domain_hidden) if domain_regression
            else DomainHead(feat_dim, n_domains, hidden=domain_hidden)
        )
        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, proj_hidden), nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim),
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        fault_logits = self.fault_head(feat)
        if self.training:
            rev = gradient_reversal(feat, self.lambda_grl)
            domain_logits = self.domain_head(rev)
            z = F.normalize(self.proj_head(feat), dim=1)
            return fault_logits, domain_logits, z
        return fault_logits

    def verify_domain_labels(self, y_domain: torch.Tensor) -> None:
        # Sengaja memakai self.n_domains, bukan domain_head.out_features: pada varian
        # regresi kepala hanya punya 1 keluaran, sedangkan LABELnya tetap 0..n_domains-1.
        n_domains = self.n_domains
        unique = y_domain.unique().tolist()
        invalid = [int(l) for l in unique if int(l) < 0 or int(l) >= n_domains]
        if invalid:
            raise ValueError(f"[ABORT] Domain labels out of range [0,{n_domains-1}]: {invalid}.")
        print(f"  [DANN+SupCon] Domain label check PASSED: unique={sorted(int(l) for l in unique)}, "
              f"n_domains={n_domains}")

    def sanity_check(self, c_in: int, seq_len: int) -> None:
        n_classes = self.fault_head.out_features
        n_domains = self.domain_head.out_features  # 1 pada varian regresi
        device = next(self.parameters()).device
        was_training = self.training
        self.train()
        with torch.no_grad():
            out = self(torch.zeros(2, c_in, seq_len, device=device))
        self.train(was_training)
        if not isinstance(out, tuple) or len(out) != 3:
            raise RuntimeError(f"[ABORT] DANN+SupCon sanity: expected (fault, domain, z), got {type(out)}.")
        fault_logits, domain_logits, z = out
        assert fault_logits.shape == (2, n_classes), fault_logits.shape
        assert domain_logits.shape == (2, n_domains), domain_logits.shape
        assert z.shape[0] == 2, z.shape
        norms = z.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), norms
        print(f"  [OK] DANNSupConModel sanity PASSED: c_in={c_in}, seq_len={seq_len} -> "
              f"fault={tuple(fault_logits.shape)}, domain={tuple(domain_logits.shape)}, "
              f"z={tuple(z.shape)} (unit-norm)")


def build_dann_supcon_model(backbone_name: str, c_in: int, seq_len: int, n_classes: int,
                            n_domains: int, lambda_grl: float = 0.0, domain_hidden: int = 64,
                            proj_hidden: int = 128, proj_dim: int = 128,
                            domain_regression: bool = False) -> DANNSupConModel:
    backbone, feat_dim = build_backbone(backbone_name, c_in, seq_len)
    return DANNSupConModel(backbone, feat_dim, n_classes, n_domains, lambda_grl,
                           domain_hidden, proj_hidden, proj_dim, domain_regression)
