"""supcon.py — Supervised Contrastive Learning untuk Domain Generalization.

Motivasi (akar masalah BRB): sideband fault termodulasi-beban → fault & beban
TERJALIN di fitur. DANN/GRL memaksa invariansi dengan MENGHAPUS info beban →
ikut menghapus sinyal fault → negative transfer (terbukti kalah di semua uji).

SupCon (Khosla et al. 2020) tak memakai discriminator/min-max sama sekali. Ia
membentuk RUANG METRIK berdasar identitas fault: tarik semua sampel kelas-fault
SAMA agar berdekatan — TERLEPAS dari torsi 0 atau 7 — sekaligus dorong menjauh
kelas berbeda. Karena positif span seluruh beban di dalam batch, pemisah kelas
jadi tak bergantung beban TANPA menghapus apa pun secara adversarial.

    backbone(x) -> feat
        ├─ fault_head(feat)             -> fault_logits   (CE)
        └─ proj_head(feat) -> L2-norm   -> z (embedding)  (SupCon loss, pakai label fault)

Loss total (dirakit di SupConCallback):
    L = CE_fault + lambda_sc * SupCon(z, fault_labels)
Tanpa GRL, tanpa label domain. Eval: fault_head saja.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.heads import FaultHead
from src.models.backbones import build_backbone


class SupConModel(nn.Module):
    """Backbone + fault head + projection head (untuk supervised contrastive).

    forward (training): (fault_logits, z_norm)  z_norm = L2-normalized embedding
    forward (eval):     fault_logits
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, n_classes: int,
                 proj_hidden: int = 128, proj_dim: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        self.fault_head = FaultHead(feat_dim, n_classes)
        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim),
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        fault_logits = self.fault_head(feat)
        if self.training:
            z = F.normalize(self.proj_head(feat), dim=1)   # unit-sphere embedding
            return fault_logits, z
        return fault_logits

    def sanity_check(self, c_in: int, seq_len: int) -> None:
        n_classes = self.fault_head.out_features
        device = next(self.parameters()).device
        was_training = self.training
        self.train()
        with torch.no_grad():
            out = self(torch.zeros(2, c_in, seq_len, device=device))
        self.train(was_training)
        if not isinstance(out, tuple) or len(out) != 2:
            raise RuntimeError(f"[ABORT] SupCon sanity: expected (fault, z), got {type(out)}.")
        fault_logits, z = out
        assert fault_logits.shape == (2, n_classes), fault_logits.shape
        assert z.shape[0] == 2, z.shape
        norms = z.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), norms
        print(f"  [OK] SupConModel sanity check PASSED: c_in={c_in}, seq_len={seq_len} -> "
              f"fault={tuple(fault_logits.shape)}, z={tuple(z.shape)} (unit-norm)")


def build_supcon_model(backbone_name: str, c_in: int, seq_len: int, n_classes: int,
                       proj_hidden: int = 128, proj_dim: int = 128) -> SupConModel:
    backbone, feat_dim = build_backbone(backbone_name, c_in, seq_len)
    return SupConModel(backbone, feat_dim, n_classes, proj_hidden, proj_dim)
