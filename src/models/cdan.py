"""cdan.py — Conditional Adversarial Domain Adaptation (CDAN) + varian CDAN-SupCon.

Motivasi (dari diagnosis empiris kita): DANN menyelaraskan distribusi MARGINAL p(f).
Pada BRB motor, informasi beban dan tanda-tangan fault TERJALIN (sideband termodulasi
slip), sehingga menggosok beban secara marginal ikut menggosok sinyal fault yang
berguna di beban ekstrem → negative transfer (DANN 86.62 < ERM 89.40).

CDAN mengganti penyelarasan marginal dengan penyelarasan GABUNGAN p(f, ŷ): domain head
tidak menerima fitur mentah, melainkan multilinear map antara fitur dan prediksi kelas,

    h = f ⊗ g,   g = softmax(fault_logits).detach()

sehingga penyelarasan terjadi DI DALAM tiap kelas, bukan lintas-kelas secara global.
Hipotesis: sinyal fault yang termodulasi beban tidak lagi ikut tergosok.

    x → [backbone] → feat ──→ fault_head → logits ──┐  (CE)
                       │                            │
                       └──→ multilinear(feat, g) ──→ GRL(λ) → domain_head  (CE)
                                                       g = softmax(logits).detach()

Referensi: Long et al. (2018), "Conditional Adversarial Domain Adaptation", NeurIPS.
    - Multilinear conditioning: T(f, g) = f ⊗ g (outer product).
    - Randomized multilinear jika dim(f)·dim(g) > 4096 (Long et al. §4.1).
    - Entropy conditioning (CDAN+E): bobot per-sampel w = 1 + exp(−H(g)) menekan
      kontribusi sampel yang prediksi kelasnya tidak pasti.

Catatan implementasi: g DI-DETACH (mengikuti implementasi resmi thuml/CDAN) agar
gradien adversarial mencapai backbone hanya lewat jalur fitur, tidak merusak kepala
klasifikasi melalui jalur pengondisian.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.grl import gradient_reversal
from src.models.heads import FaultHead, DomainHead
from src.models.backbones import build_backbone

# Ambang dimensi multilinear penuh sebelum beralih ke randomized map (Long et al. 2018).
MULTILINEAR_MAX_DIM = 4096
RANDOMIZED_DIM = 1024


class MultilinearMap(nn.Module):
    """Peta multilinear T(f, g) untuk pengondisian CDAN.

    Mode penuh (dim(f)·dim(g) <= 4096): outer product f ⊗ g → (B, feat_dim·n_classes).
    Mode randomized (selebihnya): (f R_f) ⊙ (g R_g) / sqrt(d), R tetap (buffer, tak dilatih).
    """

    def __init__(self, feat_dim: int, n_classes: int, out_dim: int = RANDOMIZED_DIM) -> None:
        super().__init__()
        full_dim = feat_dim * n_classes
        self.randomized = full_dim > MULTILINEAR_MAX_DIM
        if self.randomized:
            self.out_features = out_dim
            # Matriks acak TETAP (bukan parameter) — sesuai Long et al. §4.1.
            self.register_buffer("R_f", torch.randn(feat_dim, out_dim))
            self.register_buffer("R_g", torch.randn(n_classes, out_dim))
        else:
            self.out_features = full_dim

    def forward(self, feat: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if self.randomized:
            return (feat @ self.R_f) * (g @ self.R_g) / math.sqrt(self.out_features)
        # outer product: (B, n_classes, feat_dim) -> flatten
        return torch.bmm(g.unsqueeze(2), feat.unsqueeze(1)).view(feat.size(0), -1)


def _conditioning_input(feat: torch.Tensor, fault_logits: torch.Tensor,
                        multilinear: MultilinearMap) -> torch.Tensor:
    """Bangun masukan domain head: multilinear(feat, softmax(logits).detach())."""
    g = F.softmax(fault_logits, dim=1).detach()
    return multilinear(feat, g)


class CDANModel(nn.Module):
    """CDAN: backbone → fault_head + domain_head atas multilinear(feat, ŷ) via GRL.

    forward (training): (fault_logits, domain_logits, fault_logits_for_entropy)
    forward (eval):     fault_logits
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, n_classes: int, n_domains: int,
                 lambda_grl: float = 0.0, domain_hidden: int = 64) -> None:
        super().__init__()
        self.lambda_grl = lambda_grl
        self.backbone = backbone
        self.fault_head = FaultHead(feat_dim, n_classes)
        self.multilinear = MultilinearMap(feat_dim, n_classes)
        self.domain_head = DomainHead(self.multilinear.out_features, n_domains, hidden=domain_hidden)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        fault_logits = self.fault_head(feat)
        if self.training:
            h = _conditioning_input(feat, fault_logits, self.multilinear)
            rev = gradient_reversal(h, self.lambda_grl)
            return fault_logits, self.domain_head(rev), fault_logits
        return fault_logits

    def verify_domain_labels(self, y_domain: torch.Tensor) -> None:
        n_domains = self.domain_head.out_features
        unique = y_domain.unique().tolist()
        invalid = [int(l) for l in unique if int(l) < 0 or int(l) >= n_domains]
        if invalid:
            raise ValueError(f"[ABORT] Domain labels out of range [0,{n_domains-1}]: {invalid}.")
        print(f"  [CDAN] Domain label check PASSED: unique={sorted(int(l) for l in unique)}, "
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
        if not isinstance(out, tuple) or len(out) != 3:
            raise RuntimeError(f"[ABORT] CDAN sanity: expected 3-tuple, got {type(out)}.")
        fault_logits, domain_logits, _ = out
        assert fault_logits.shape == (2, n_classes), fault_logits.shape
        assert domain_logits.shape == (2, n_domains), domain_logits.shape
        mode = "randomized" if self.multilinear.randomized else "full outer-product"
        print(f"  [OK] CDANModel sanity PASSED: c_in={c_in}, seq_len={seq_len} -> "
              f"fault={tuple(fault_logits.shape)}, domain={tuple(domain_logits.shape)} | "
              f"multilinear={mode}, dim={self.multilinear.out_features}")


class CDANSupConModel(nn.Module):
    """CDAN + kepala proyeksi Supervised Contrastive (pengganti DANNSupCon).

    forward (training): (fault_logits, domain_logits, z_embed, fault_logits)
    forward (eval):     fault_logits
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, n_classes: int, n_domains: int,
                 lambda_grl: float = 0.0, domain_hidden: int = 64,
                 proj_hidden: int = 128, proj_dim: int = 128) -> None:
        super().__init__()
        self.lambda_grl = lambda_grl
        self.backbone = backbone
        self.fault_head = FaultHead(feat_dim, n_classes)
        self.multilinear = MultilinearMap(feat_dim, n_classes)
        self.domain_head = DomainHead(self.multilinear.out_features, n_domains, hidden=domain_hidden)
        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, proj_hidden), nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_dim),
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        fault_logits = self.fault_head(feat)
        if self.training:
            h = _conditioning_input(feat, fault_logits, self.multilinear)
            rev = gradient_reversal(h, self.lambda_grl)
            domain_logits = self.domain_head(rev)
            z = F.normalize(self.proj_head(feat), dim=1)
            return fault_logits, domain_logits, z, fault_logits
        return fault_logits

    def verify_domain_labels(self, y_domain: torch.Tensor) -> None:
        n_domains = self.domain_head.out_features
        unique = y_domain.unique().tolist()
        invalid = [int(l) for l in unique if int(l) < 0 or int(l) >= n_domains]
        if invalid:
            raise ValueError(f"[ABORT] Domain labels out of range [0,{n_domains-1}]: {invalid}.")
        print(f"  [CDAN+SupCon] Domain label check PASSED: "
              f"unique={sorted(int(l) for l in unique)}, n_domains={n_domains}")

    def sanity_check(self, c_in: int, seq_len: int) -> None:
        n_classes = self.fault_head.out_features
        n_domains = self.domain_head.out_features
        device = next(self.parameters()).device
        was_training = self.training
        self.train()
        with torch.no_grad():
            out = self(torch.zeros(2, c_in, seq_len, device=device))
        self.train(was_training)
        if not isinstance(out, tuple) or len(out) != 4:
            raise RuntimeError(f"[ABORT] CDAN+SupCon sanity: expected 4-tuple, got {type(out)}.")
        fault_logits, domain_logits, z, _ = out
        assert fault_logits.shape == (2, n_classes), fault_logits.shape
        assert domain_logits.shape == (2, n_domains), domain_logits.shape
        assert z.shape[0] == 2, z.shape
        norms = z.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), norms
        mode = "randomized" if self.multilinear.randomized else "full outer-product"
        print(f"  [OK] CDANSupConModel sanity PASSED: c_in={c_in}, seq_len={seq_len} -> "
              f"fault={tuple(fault_logits.shape)}, domain={tuple(domain_logits.shape)}, "
              f"z={tuple(z.shape)} (unit-norm) | multilinear={mode}, "
              f"dim={self.multilinear.out_features}")


def build_cdan_model(backbone_name: str, c_in: int, seq_len: int, n_classes: int, n_domains: int,
                     lambda_grl: float = 0.0, domain_hidden: int = 64) -> CDANModel:
    backbone, feat_dim = build_backbone(backbone_name, c_in, seq_len)
    return CDANModel(backbone, feat_dim, n_classes, n_domains, lambda_grl, domain_hidden)


def build_cdan_supcon_model(backbone_name: str, c_in: int, seq_len: int, n_classes: int,
                            n_domains: int, lambda_grl: float = 0.0, domain_hidden: int = 64,
                            proj_hidden: int = 128, proj_dim: int = 128) -> CDANSupConModel:
    backbone, feat_dim = build_backbone(backbone_name, c_in, seq_len)
    return CDANSupConModel(backbone, feat_dim, n_classes, n_domains, lambda_grl,
                           domain_hidden, proj_hidden, proj_dim)
