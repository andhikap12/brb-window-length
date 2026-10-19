"""dann_supcon_callback.py — Callback DANN + Supervised Contrastive (gabungan).

Merakit ke loss_grad (BUKAN display-clone — fix fastai):
    L = CE_fault + CE_domain(GRL feat) + λ_sc · SupCon(z, fault_labels)

- Cabang DANN: domain head di GRL(feat); λ_grl dijadwalkan 0→λ_max (anneal Ganin),
  diset ke model.lambda_grl tiap batch. domain_acc → chance = backbone invarian (baik).
- Cabang SupCon: contrastive (single-view, L_out) pakai LABEL FAULT; menarik same-fault
  lintas-beban. Tanpa label domain.
"""

import numpy as np
import torch
import torch.nn.functional as F
from fastai.callback.core import Callback

from src.training.domain_objective import DomainObjectiveMixin
from src.training.supcon_callback import supcon_loss


class DANNSupConCallback(DomainObjectiveMixin, Callback):
    order = 12

    def __init__(self, lambda_max: float = 1.0, gamma: float = 10.0,
                 lambda_sc: float = 1.0, temperature: float = 0.07,
                 log_every: int = 50, domain_u=None) -> None:
        self.lambda_max = lambda_max
        self.gamma = gamma
        self.lambda_sc = lambda_sc
        self.temperature = temperature
        self.log_every = log_every
        # UJI A: bila diberi tabel u[indeks_domain], kepala domain diperlakukan sebagai
        # REGRESI (MSE) alih-alih klasifikasi (CE). None = perilaku lama persis.
        self._init_domain_objective(domain_u)
        self._domain_logits = None
        self._embed = None
        self._domain_tgt = None
        self._fault_tgt = None
        self._ep_dloss: list[float] = []
        self._ep_dacc: list[float] = []
        self._ep_sc: list[float] = []
        self._batch = 0

    def _get_lambda(self) -> float:
        p = float(getattr(self.learn, "pct_train", 0.0) or 0.0)
        return self.lambda_max * (2.0 / (1.0 + np.exp(-self.gamma * p)) - 1.0)

    def before_fit(self) -> None:
        print(f"  [DANN+SupCon] lambda_max={self.lambda_max}, gamma={self.gamma}, "
              f"lambda_sc={self.lambda_sc}, temperature={self.temperature}")

    def before_epoch(self) -> None:
        self._ep_dloss, self._ep_dacc, self._ep_sc, self._batch = [], [], [], 0

    def before_batch(self) -> None:
        self._domain_logits = self._embed = self._domain_tgt = self._fault_tgt = None
        if hasattr(self.model, "lambda_grl"):
            self.model.lambda_grl = self._get_lambda()
        if not self.learn.yb:  # type: ignore
            return
        y = self.learn.yb[0]  # type: ignore
        if y.dim() == 2 and y.shape[1] == 2:
            if self.training:
                dt = y[:, 1].long()
                if (dt < 0).any():
                    raise RuntimeError("DANN+SupCon: masked domain labels in training batch.")
                self._domain_tgt = dt
            self.learn.yb = (y[:, 0].long(),) + self.learn.yb[1:]  # type: ignore
        if self.training:
            self._fault_tgt = self.learn.yb[0].long()  # type: ignore

    def after_pred(self) -> None:
        pred = self.learn.pred  # type: ignore
        if self.training and isinstance(pred, (tuple, list)) and len(pred) >= 3:
            self._domain_logits = pred[1]
            self._embed = pred[2]
            self.learn.pred = pred[0]  # type: ignore

    def after_loss(self) -> None:
        if not (self.training and self._domain_logits is not None
                and self._domain_tgt is not None and self._embed is not None):
            self._domain_logits = self._embed = self._domain_tgt = self._fault_tgt = None
            return

        dev = self._domain_logits.device
        dtgt = self._domain_tgt.to(dev)
        # CE kategoris, atau MSE terhadap indeks skalar u bila UJI A aktif.
        # GRL tetap yang membalik gradiennya — bentuk loss tidak mengubah itu.
        domain_loss = self._dom_loss(self._domain_logits, dtgt)
        sc = supcon_loss(self._embed, self._fault_tgt.to(self._embed.device), self.temperature)

        extra = domain_loss + self.lambda_sc * sc
        self.learn.loss_grad = self.learn.loss_grad + extra
        self.learn.loss = self.learn.loss + extra  # type: ignore

        with torch.no_grad():
            dacc = self._dom_acc(self._domain_logits, dtgt)
            corr = self._dom_corr(self._domain_logits, dtgt)
        self._ep_dloss.append(float(domain_loss.item()))
        self._ep_dacc.append(dacc)
        self._ep_sc.append(float(sc.item()))
        if np.isfinite(corr):
            self._ep_corr.append(corr)
        self._batch += 1
        if self.log_every > 0 and self._batch % self.log_every == 0:
            ekor = (f" corr={corr:+.3f}" if np.isfinite(corr)
                    else f" domain_acc={dacc:.3f}")
            print(f"    [DANN+SupCon] step={self._batch:4d} | lambda={self.model.lambda_grl:.3f} "
                  f"domain_loss={domain_loss.item():.4f}{ekor} | supcon={sc.item():.4f}")

        self._domain_logits = self._embed = self._domain_tgt = self._fault_tgt = None

    def after_epoch(self) -> None:
        if not self._ep_dloss:
            return
        ep = getattr(self.learn, "epoch", "?")
        # Pada mode regresi domain_acc tidak terdefinisi (NaN) dan plafon ln(6) tidak
        # berlaku; yang menggantikannya sebagai tanda patologi adalah korelasi.
        if self.domain_regression:
            kepala = f"domain_mse={np.mean(self._ep_dloss):.5f}{self._corr_epoch_report('')}"
        else:
            kepala = (f"domain_loss={np.mean(self._ep_dloss):.4f} "
                      f"domain_acc={np.mean(self._ep_dacc):.3f} (→chance = backbone invarian)")
        print(f"\n  [DANN+SupCon-EPOCH {ep}] {kepala} "
              f"| supcon={np.mean(self._ep_sc):.4f} (LOW = same-fault tight across loads)")
