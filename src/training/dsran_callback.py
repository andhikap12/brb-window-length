"""dsran_callback.py — Callback DSRAN-combo (separasi + adversarial).

Merakit ke loss_grad (BUKAN display-clone — sama seperti fix DANN/DSN):
    L = CE_fault
      + CE_adv(GRL z_task, dtgt)              # adversarial → z_task invarian-beban
      + lambda_domain * CE_dom(z_dom, dtgt)   # z_dom MENYERAP beban (no GRL)
      + lambda_ortho  * L_ortho               # pisahkan subruang task vs dom

λ_grl dijadwalkan 0→lambda_max (anneal Ganin) dan diset ke model.lambda_grl tiap
batch; skala adversarial terhadap backbone/proj_task terjadi di GRL backward.
domain_acc (z_dom) TINGGI = baik; adv_acc (z_task) menuju chance = baik.
"""

import numpy as np
import torch
import torch.nn.functional as F
from fastai.callback.core import Callback


class DSRANCallback(Callback):
    order = 12

    def __init__(self, lambda_max: float = 1.0, gamma: float = 10.0,
                 lambda_domain: float = 1.0, lambda_ortho: float = 1.0,
                 lambda_recon: float = 1.0, log_every: int = 50) -> None:
        self.lambda_max = lambda_max
        self.gamma = gamma
        self.lambda_domain = lambda_domain
        self.lambda_ortho = lambda_ortho
        self.lambda_recon = lambda_recon
        self.log_every = log_every
        self._adv_logits = None
        self._clean_logits = None
        self._domain_tgt = None
        self._ep_adv: list[float] = []
        self._ep_advacc: list[float] = []
        self._ep_dacc: list[float] = []
        self._ep_ortho: list[float] = []
        self._ep_recon: list[float] = []
        self._batch = 0

    def _get_lambda(self) -> float:
        p = float(getattr(self.learn, "pct_train", 0.0) or 0.0)
        return self.lambda_max * (2.0 / (1.0 + np.exp(-self.gamma * p)) - 1.0)

    def before_fit(self) -> None:
        print(f"  [DSRAN] lambda_max={self.lambda_max}, gamma={self.gamma}, "
              f"lambda_domain={self.lambda_domain}, lambda_ortho={self.lambda_ortho}")

    def before_epoch(self) -> None:
        self._ep_adv, self._ep_advacc, self._ep_dacc = [], [], []
        self._ep_ortho, self._ep_recon, self._batch = [], [], 0

    def before_batch(self) -> None:
        self._adv_logits = self._clean_logits = self._domain_tgt = None
        if hasattr(self.model, "lambda_grl"):
            self.model.lambda_grl = self._get_lambda()
        if not self.learn.yb:  # type: ignore
            return
        y = self.learn.yb[0]  # type: ignore
        if y.dim() == 2 and y.shape[1] == 2:
            if self.training:
                dt = y[:, 1].long()
                if (dt < 0).any():
                    raise RuntimeError("DSRAN: masked domain labels in training batch.")
                self._domain_tgt = dt
            self.learn.yb = (y[:, 0].long(),) + self.learn.yb[1:]  # type: ignore

    def after_pred(self) -> None:
        pred = self.learn.pred  # type: ignore
        if self.training and isinstance(pred, (tuple, list)) and len(pred) >= 3:
            self._adv_logits = pred[1]
            self._clean_logits = pred[2]
            self.learn.pred = pred[0]  # type: ignore

    def after_loss(self) -> None:
        if not (self.training and self._adv_logits is not None and self._domain_tgt is not None):
            self._adv_logits = self._clean_logits = self._domain_tgt = None
            return

        dev = self._adv_logits.device
        dtgt = self._domain_tgt.to(dev)
        adv_loss = F.cross_entropy(self._adv_logits, dtgt)        # GRL → backbone maksimisasi
        domain_loss = F.cross_entropy(self._clean_logits, dtgt)   # z_dom serap beban

        # Orthogonality — canonical ‖H_cᵀH_p‖²_F, per-sample L2-norm (audit fix #5).
        zt = self.model._z_task  # type: ignore
        zd = self.model._z_dom   # type: ignore
        bs = max(1, zt.shape[0])
        zt_n = F.normalize(zt, dim=1)
        zd_n = F.normalize(zd, dim=1)
        cross = zt_n.t() @ zd_n
        ortho_loss = cross.pow(2).sum() / float(bs * bs)

        # Reconstruction (audit fix #3/#4): task+private jointly retain feat.
        recon_loss = F.mse_loss(self.model._recon_pred, self.model._recon_target)  # type: ignore

        extra = (adv_loss + self.lambda_domain * domain_loss + self.lambda_ortho * ortho_loss
                 + self.lambda_recon * recon_loss)
        self.learn.loss_grad = self.learn.loss_grad + extra
        self.learn.loss = self.learn.loss + extra  # type: ignore

        with torch.no_grad():
            adv_acc = float((self._adv_logits.argmax(1) == dtgt).float().mean().item())
            dacc = float((self._clean_logits.argmax(1) == dtgt).float().mean().item())
        self._ep_adv.append(float(adv_loss.item()))
        self._ep_advacc.append(adv_acc)
        self._ep_dacc.append(dacc)
        self._ep_ortho.append(float(ortho_loss.item()))
        self._ep_recon.append(float(recon_loss.item()))
        self._batch += 1
        if self.log_every > 0 and self._batch % self.log_every == 0:
            print(f"    [DSRAN] step={self._batch:4d} | lambda={self.model.lambda_grl:.3f} "
                  f"adv_loss={adv_loss.item():.4f} adv_acc={adv_acc:.3f} "
                  f"| dom_acc={dacc:.3f} ortho={ortho_loss.item():.4f} recon={recon_loss.item():.4f}")

        self._adv_logits = self._clean_logits = self._domain_tgt = None

    def after_epoch(self) -> None:
        if not self._ep_adv:
            return
        ep = getattr(self.learn, "epoch", "?")
        print(f"\n  [DSRAN-EPOCH {ep}] adv_loss={np.mean(self._ep_adv):.4f} "
              f"adv_acc={np.mean(self._ep_advacc):.3f} (LOW→chance = z_task invarian) "
              f"| dom_acc={np.mean(self._ep_dacc):.3f} (HIGH = z_dom serap beban) "
              f"| ortho={np.mean(self._ep_ortho):.4f} (LOW = terpisah) "
              f"| recon={np.mean(self._ep_recon):.4f} (LOW = feat retained)")
