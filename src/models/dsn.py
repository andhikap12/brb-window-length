"""dsn.py — Orthogonal Feature Separation (DSN-inspired, NO-adversarial variant).

PENAMAAN JUJUR (audit 2026-06-27): ini BUKAN DSN kanonik (Bousmalis et al. 2016).
DSN asli punya (a) encoder shared & private TERPISAH, (b) L_similarity adversarial
yang membuat rep shared invarian-domain, (c) rekonstruksi input. Varian ini SENGAJA
membuang L_similarity adversarial karena pada data kita (BRB, fault termodulasi-beban)
erasure adversarial terbukti MERUSAK (DANN/DSRAN selalu kalah). Yang dipertahankan:
separasi via orthogonality + (BARU) rekonstruksi-fitur untuk menggerakkan cabang
private secara DG-appropriate (tak terikat label domain diskrit).

Mekanisme (akar masalah BRB: sideband fault termodulasi-beban → fault & beban terjalin):
    backbone(x) -> feat
        ├─ proj_task (linear)        -> z_task -> fault_head   (klasifikasi fault)
        └─ proj_dom  (MLP, B1)       -> z_dom  -> domain_head  (prediksi beban, TANPA GRL)
    + L_ortho  : ‖ẑ_taskᵀ ẑ_dom‖²_F  (per-sampel L2-norm, formula kanonik — fix audit #5)
    + L_recon  : MSE(decoder([z_task ⊕ z_dom]), feat.detach())  (fix audit #3/#4)

Efek: cabang domain menyerap variasi beban; ortho menjaga z_task lepas dari arah-beban;
rekonstruksi memaksa (z_task ⊕ z_dom) bersama menyimpan info `feat` → private wajib
memuat apa yang task buang, bukan kolaps. Invariansi z_task dicapai via ortho + recon,
bukan via penghapusan adversarial.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.heads import FaultHead, DomainHead
from src.models.backbones import build_backbone


class DSNModel(nn.Module):
    """Orthogonal feature separation + feature reconstruction (no adversarial).

    forward (training): (fault_logits, domain_logits); stashes z_task/z_dom and
                        recon pred/target for the ortho + reconstruction losses.
    forward (eval):     fault_logits  (via z_task only)
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, n_classes: int, n_domains: int,
                 proj_dim: int = 64, domain_hidden: int = 64, recon_hidden: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        self.proj_task = nn.Linear(feat_dim, proj_dim)                       # shared/task (linear)
        self.proj_dom = nn.Sequential(                                       # B1: private deeper (MLP)
            nn.Linear(feat_dim, proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.fault_head = FaultHead(proj_dim, n_classes)
        self.domain_head = DomainHead(proj_dim, n_domains, hidden=domain_hidden)
        self.decoder = nn.Sequential(                                        # C: feat reconstruction
            nn.Linear(proj_dim * 2, recon_hidden), nn.ReLU(inplace=True),
            nn.Linear(recon_hidden, feat_dim),
        )
        self._z_task: torch.Tensor | None = None
        self._z_dom: torch.Tensor | None = None
        self._recon_pred: torch.Tensor | None = None
        self._recon_target: torch.Tensor | None = None

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)               # (B, feat_dim)
        z_task = self.proj_task(feat)
        fault_logits = self.fault_head(z_task)
        if self.training:
            z_dom = self.proj_dom(feat)
            self._z_task = z_task
            self._z_dom = z_dom
            self._recon_target = feat.detach()                              # fixed target (no collapse)
            self._recon_pred = self.decoder(torch.cat([z_task, z_dom], dim=1))
            return fault_logits, self.domain_head(z_dom)
        return fault_logits

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
            raise RuntimeError(f"[ABORT] DSN sanity: expected (fault, domain), got {type(out)}.")
        fault_logits, domain_logits = out
        assert fault_logits.shape == (2, n_classes), fault_logits.shape
        assert domain_logits.shape == (2, n_domains), domain_logits.shape
        assert self._recon_pred.shape == self._recon_target.shape, \
            (self._recon_pred.shape, self._recon_target.shape)
        print(f"  [OK] DSNModel(OFS+recon) sanity PASSED: c_in={c_in}, seq_len={seq_len} -> "
              f"fault={tuple(fault_logits.shape)}, domain={tuple(domain_logits.shape)}, "
              f"recon={tuple(self._recon_pred.shape)}")

    def verify_domain_labels(self, y_domain: torch.Tensor) -> None:
        n_domains = self.domain_head.out_features
        unique = y_domain.unique().tolist()
        invalid = [int(l) for l in unique if int(l) < 0 or int(l) >= n_domains]
        if invalid:
            raise ValueError(f"[ABORT] Domain labels out of range [0,{n_domains-1}]: {invalid}.")
        print(f"  [DSN] Domain label check PASSED: unique={sorted(int(l) for l in unique)}, "
              f"n_domains={n_domains}")


def build_dsn_model(backbone_name: str, c_in: int, seq_len: int, n_classes: int, n_domains: int,
                    proj_dim: int = 64, domain_hidden: int = 64, recon_hidden: int = 128) -> DSNModel:
    backbone, feat_dim = build_backbone(backbone_name, c_in, seq_len)
    return DSNModel(backbone, feat_dim, n_classes, n_domains, proj_dim, domain_hidden, recon_hidden)
