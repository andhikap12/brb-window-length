"""dsran.py — DSRAN-inspired: Separation + Reconstruction + Adversarial.

PENAMAAN JUJUR (audit 2026-06-27): mendekati DSN/DSRAN kanonik LEBIH dari varian
`dsn.py` kita, karena DI SINI ada L_similarity adversarial (GRL pada z_task) =
komponen invariansi DSN asli. Bedanya dari kanonik: encoder shared & private masih
satu backbone + dua proyeksi (bukan dua encoder terpisah). Rekonstruksi (audit #3/#4)
dan formula ortho kanonik (audit #5) kini DITAMBAHKAN.

CATATAN empiris: tekanan adversarial pada data kita terbukti MERUSAK (DANN/DSRAN
kalah). Varian ini dipertahankan untuk kelengkapan/perbandingan, BUKAN kandidat utama.

    backbone(x) -> feat
        ├─ proj_task (linear) -> z_task ─┬─ fault_head              (CE fault)
        │                                └─ GRL(λ) -> domain_adv    (CE adversarial → invarian)
        └─ proj_dom  (MLP, B1) -> z_dom  ─── domain_head            (CE beban, TANPA GRL)
    + L_ortho : ‖ẑ_taskᵀ ẑ_dom‖²_F  (per-sampel L2-norm — fix audit #5)
    + L_recon : MSE(decoder([z_task ⊕ z_dom]), feat.detach())  (fix audit #3/#4)

Loss total (DSRANCallback):
    L = CE_fault + CE_adv(GRL z_task) + λ_domain·CE_dom(z_dom) + λ_ortho·L_ortho + λ_recon·L_recon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.grl import gradient_reversal
from src.models.heads import FaultHead, DomainHead
from src.models.backbones import build_backbone


class DSRANModel(nn.Module):
    """Separation + reconstruction + adversarial: (z_task→fault, GRL→adv) & (z_dom→domain).

    forward (training): (fault_logits, domain_adv_logits, domain_clean_logits);
                        stashes z_task/z_dom and recon pred/target.
    forward (eval):     fault_logits  (via z_task only)
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, n_classes: int, n_domains: int,
                 lambda_grl: float = 0.0, proj_dim: int = 64, domain_hidden: int = 64,
                 recon_hidden: int = 128) -> None:
        super().__init__()
        self.lambda_grl = lambda_grl
        self.backbone = backbone
        self.proj_task = nn.Linear(feat_dim, proj_dim)
        self.proj_dom = nn.Sequential(                                       # B1: private deeper (MLP)
            nn.Linear(feat_dim, proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.fault_head = FaultHead(proj_dim, n_classes)
        self.domain_adv_head = DomainHead(proj_dim, n_domains, hidden=domain_hidden)   # GRL → invarian
        self.domain_head = DomainHead(proj_dim, n_domains, hidden=domain_hidden)       # beban (no GRL)
        self.decoder = nn.Sequential(                                        # C: feat reconstruction
            nn.Linear(proj_dim * 2, recon_hidden), nn.ReLU(inplace=True),
            nn.Linear(recon_hidden, feat_dim),
        )
        self._z_task: torch.Tensor | None = None
        self._z_dom: torch.Tensor | None = None
        self._recon_pred: torch.Tensor | None = None
        self._recon_target: torch.Tensor | None = None

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        z_task = self.proj_task(feat)
        fault_logits = self.fault_head(z_task)
        if self.training:
            z_dom = self.proj_dom(feat)
            self._z_task = z_task
            self._z_dom = z_dom
            self._recon_target = feat.detach()
            self._recon_pred = self.decoder(torch.cat([z_task, z_dom], dim=1))
            adv_logits = self.domain_adv_head(gradient_reversal(z_task, self.lambda_grl))
            clean_logits = self.domain_head(z_dom)
            return fault_logits, adv_logits, clean_logits
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
        if not isinstance(out, tuple) or len(out) != 3:
            raise RuntimeError(f"[ABORT] DSRAN sanity: expected (fault, adv, clean), got {type(out)}.")
        fault_logits, adv_logits, clean_logits = out
        assert fault_logits.shape == (2, n_classes), fault_logits.shape
        assert adv_logits.shape == (2, n_domains), adv_logits.shape
        assert clean_logits.shape == (2, n_domains), clean_logits.shape
        assert self._recon_pred.shape == self._recon_target.shape, \
            (self._recon_pred.shape, self._recon_target.shape)
        print(f"  [OK] DSRANModel(sep+recon+adv) sanity PASSED: c_in={c_in}, seq_len={seq_len} -> "
              f"fault={tuple(fault_logits.shape)}, adv={tuple(adv_logits.shape)}, "
              f"clean={tuple(clean_logits.shape)}, recon={tuple(self._recon_pred.shape)}")

    def verify_domain_labels(self, y_domain: torch.Tensor) -> None:
        n_domains = self.domain_head.out_features
        unique = y_domain.unique().tolist()
        invalid = [int(l) for l in unique if int(l) < 0 or int(l) >= n_domains]
        if invalid:
            raise ValueError(f"[ABORT] Domain labels out of range [0,{n_domains-1}]: {invalid}.")
        print(f"  [DSRAN] Domain label check PASSED: unique={sorted(int(l) for l in unique)}, "
              f"n_domains={n_domains}")


def build_dsran_model(backbone_name: str, c_in: int, seq_len: int, n_classes: int, n_domains: int,
                      lambda_grl: float = 0.0, proj_dim: int = 64, domain_hidden: int = 64,
                      recon_hidden: int = 128) -> DSRANModel:
    backbone, feat_dim = build_backbone(backbone_name, c_in, seq_len)
    return DSRANModel(backbone, feat_dim, n_classes, n_domains, lambda_grl, proj_dim,
                      domain_hidden, recon_hidden)
