"""cdan_callback.py — Callback CDAN dan CDAN + SupCon.

Merakit ke loss_grad (BUKAN display-clone `loss` — fix fastai yang sama seperti DANN):

    CDAN         : L = CE_fault + CE_domain(GRL[multilinear(feat, ŷ)])
    CDAN+SupCon  : L = CE_fault + CE_domain(...) + λ_sc · SupCon(z, fault_labels)

Perbedaan dari DANNCallback: domain head menerima peta multilinear fitur×prediksi
(dibangun di dalam model), bukan fitur mentah. Callback di sisi ini hanya perlu
menambahkan bobot ENTROPY CONDITIONING (CDAN+E, Long et al. 2018 §4.2):

    w_i = 1 + exp(−H(g_i)),   H(g) = −Σ_c g_c log g_c

lalu dinormalisasi agar rata-ratanya 1. Sampel dengan prediksi kelas tidak pasti
(entropi tinggi) diberi bobot lebih kecil, sehingga penyelarasan kondisional tidak
dipandu oleh pseudo-label yang buruk.

DEVIASI dari implementasi resmi (thuml/CDAN): pada CDAN asli bobot entropi ikut
melewati GRL. Di sini bobot DI-DETACH karena setting kita adalah domain
generalization multi-sumber (tanpa domain target tak berlabel), sehingga tidak ada
alasan mengalirkan gradien adversarial lewat jalur bobot.
"""

import numpy as np
import torch
import torch.nn.functional as F
from fastai.callback.core import Callback

from src.training.supcon_callback import supcon_loss


def entropy_weights(fault_logits: torch.Tensor) -> torch.Tensor:
    """Bobot entropy conditioning CDAN+E: w = 1 + exp(−H(softmax(logits))), mean-1."""
    g = F.softmax(fault_logits, dim=1)
    h = -(g * torch.log(g + 1e-8)).sum(dim=1)
    w = 1.0 + torch.exp(-h)
    return (w / w.sum().clamp_min(1e-8) * w.numel()).detach()


def _conditional_domain_loss(domain_logits: torch.Tensor, domain_tgt: torch.Tensor,
                             fault_logits: torch.Tensor, use_entropy: bool) -> torch.Tensor:
    """CE domain, opsional dibobot entropi prediksi kelas (CDAN+E)."""
    if not use_entropy:
        return F.cross_entropy(domain_logits, domain_tgt)
    w = entropy_weights(fault_logits).to(domain_logits.device)
    ce = F.cross_entropy(domain_logits, domain_tgt, reduction="none")
    return (w * ce).sum() / w.sum().clamp_min(1e-8)


class _CDANBase(Callback):
    """Bagian bersama: anneal λ Ganin, split label 2-D, akumulasi metrik per-epoch."""

    order = 12

    def __init__(self, lambda_max: float = 1.0, gamma: float = 10.0,
                 use_entropy: bool = True, log_every: int = 50) -> None:
        self.lambda_max = lambda_max
        self.gamma = gamma
        self.use_entropy = use_entropy
        self.log_every = log_every
        self._domain_logits = None
        self._domain_tgt = None
        self._fault_logits = None
        self._fault_tgt = None
        self._ep_dloss: list[float] = []
        self._ep_dacc: list[float] = []
        self._batch = 0

    def _get_lambda(self) -> float:
        p = float(getattr(self.learn, "pct_train", 0.0) or 0.0)
        return self.lambda_max * (2.0 / (1.0 + np.exp(-self.gamma * p)) - 1.0)

    def _reset_batch_state(self) -> None:
        self._domain_logits = self._domain_tgt = None
        self._fault_logits = self._fault_tgt = None

    def before_epoch(self) -> None:
        self._ep_dloss, self._ep_dacc, self._batch = [], [], 0

    def before_batch(self) -> None:
        self._reset_batch_state()
        if hasattr(self.model, "lambda_grl"):
            self.model.lambda_grl = self._get_lambda()
        if not self.learn.yb:  # type: ignore
            return
        y = self.learn.yb[0]  # type: ignore
        if y.dim() == 2 and y.shape[1] == 2:
            if self.training:
                dt = y[:, 1].long()
                if (dt < 0).any():
                    raise RuntimeError("CDAN: masked domain labels in training batch.")
                self._domain_tgt = dt
            self.learn.yb = (y[:, 0].long(),) + self.learn.yb[1:]  # type: ignore
        if self.training:
            self._fault_tgt = self.learn.yb[0].long()  # type: ignore

    def _accumulate(self, domain_loss: torch.Tensor, extra_log: str = "") -> None:
        with torch.no_grad():
            dtgt = self._domain_tgt.to(self._domain_logits.device)
            dacc = float((self._domain_logits.argmax(1) == dtgt).float().mean().item())
        self._ep_dloss.append(float(domain_loss.item()))
        self._ep_dacc.append(dacc)
        self._batch += 1
        if self.log_every > 0 and self._batch % self.log_every == 0:
            n_dom = self._domain_logits.shape[1]
            print(f"    [{self._tag}] step={self._batch:4d} | lambda={self.model.lambda_grl:.3f} "
                  f"| domain_loss={domain_loss.item():.4f}/{np.log(n_dom):.3f} "
                  f"domain_acc={dacc:.3f} (chance={1.0/n_dom:.3f}){extra_log}")


class CDANCallback(_CDANBase):
    """CDAN murni: CE_fault + CE_domain terkondisi (multilinear + opsional entropi)."""

    _tag = "CDAN"

    def before_fit(self) -> None:
        print(f"\n  [CDAN] Strategy: Conditional Adversarial Alignment (joint p(f, y-hat))")
        print(f"  [CDAN] lambda_max={self.lambda_max}, gamma={self.gamma}, "
              f"entropy_conditioning={self.use_entropy}")
        print(f"  [CDAN] GRL log interval: every {self.log_every} batches\n")

    def after_pred(self) -> None:
        pred = self.learn.pred  # type: ignore
        if self.training and isinstance(pred, (tuple, list)) and len(pred) >= 3:
            self._domain_logits = pred[1]
            self._fault_logits = pred[2]
            self.learn.pred = pred[0]  # type: ignore

    def after_loss(self) -> None:
        if not (self.training and self._domain_logits is not None
                and self._domain_tgt is not None):
            self._reset_batch_state()
            return
        dtgt = self._domain_tgt.to(self._domain_logits.device)
        domain_loss = _conditional_domain_loss(
            self._domain_logits, dtgt, self._fault_logits, self.use_entropy)
        self.learn.loss_grad = self.learn.loss_grad + domain_loss
        self.learn.loss = self.learn.loss + domain_loss  # type: ignore
        self._accumulate(domain_loss)
        self._reset_batch_state()

    def after_epoch(self) -> None:
        if not self._ep_dloss:
            return
        ep = getattr(self.learn, "epoch", "?")
        print(f"\n  [CDAN-EPOCH {ep}] domain_loss={np.mean(self._ep_dloss):.4f} "
              f"domain_acc={np.mean(self._ep_dacc):.3f} (→chance = fitur invarian per-kelas)")


class CDANSupConCallback(_CDANBase):
    """CDAN + SupCon: CE_fault + CE_domain terkondisi + λ_sc · SupCon(z, label fault)."""

    _tag = "CDAN+SupCon"

    def __init__(self, lambda_max: float = 1.0, gamma: float = 10.0,
                 use_entropy: bool = True, lambda_sc: float = 1.0,
                 temperature: float = 0.07, log_every: int = 50) -> None:
        super().__init__(lambda_max=lambda_max, gamma=gamma,
                         use_entropy=use_entropy, log_every=log_every)
        self.lambda_sc = lambda_sc
        self.temperature = temperature
        self._embed = None
        self._ep_sc: list[float] = []

    def _reset_batch_state(self) -> None:
        super()._reset_batch_state()
        self._embed = None

    def before_fit(self) -> None:
        print(f"\n  [CDAN+SupCon] Conditional adversarial alignment + supervised contrastive")
        print(f"  [CDAN+SupCon] lambda_max={self.lambda_max}, gamma={self.gamma}, "
              f"entropy_conditioning={self.use_entropy}, lambda_sc={self.lambda_sc}, "
              f"temperature={self.temperature}\n")

    def before_epoch(self) -> None:
        super().before_epoch()
        self._ep_sc = []

    def after_pred(self) -> None:
        pred = self.learn.pred  # type: ignore
        if self.training and isinstance(pred, (tuple, list)) and len(pred) >= 4:
            self._domain_logits = pred[1]
            self._embed = pred[2]
            self._fault_logits = pred[3]
            self.learn.pred = pred[0]  # type: ignore

    def after_loss(self) -> None:
        if not (self.training and self._domain_logits is not None
                and self._domain_tgt is not None and self._embed is not None):
            self._reset_batch_state()
            return
        dtgt = self._domain_tgt.to(self._domain_logits.device)
        domain_loss = _conditional_domain_loss(
            self._domain_logits, dtgt, self._fault_logits, self.use_entropy)
        sc = supcon_loss(self._embed, self._fault_tgt.to(self._embed.device), self.temperature)

        extra = domain_loss + self.lambda_sc * sc
        self.learn.loss_grad = self.learn.loss_grad + extra
        self.learn.loss = self.learn.loss + extra  # type: ignore

        self._ep_sc.append(float(sc.item()))
        self._accumulate(domain_loss, extra_log=f" | supcon={sc.item():.4f}")
        self._reset_batch_state()

    def after_epoch(self) -> None:
        if not self._ep_dloss:
            return
        ep = getattr(self.learn, "epoch", "?")
        print(f"\n  [CDAN+SupCon-EPOCH {ep}] domain_loss={np.mean(self._ep_dloss):.4f} "
              f"domain_acc={np.mean(self._ep_dacc):.3f} (→chance = fitur invarian per-kelas) "
              f"| supcon={np.mean(self._ep_sc):.4f} (LOW = same-fault rapat lintas beban)")
