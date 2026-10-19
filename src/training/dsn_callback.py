"""dsn_callback.py — Callback untuk Domain Separation Network (disentanglement).

Menambah ke total loss (via loss_grad, BUKAN display-clone — sama seperti fix DANN):
    L = CE_fault  +  lambda_domain * CE_domain(z_dom)  +  lambda_ortho * L_ortho

L_ortho = || zt_c^T zd_c ||_F^2 / B^2   (zt_c, zd_c = z_task, z_dom mean-centered)
memaksa subruang task & domain ORTHOGONAL → cabang domain menyerap beban,
cabang task bebas mempertahankan sideband BRB. TIDAK ada GRL (domain head BELAJAR
beban — domain_acc TINGGI itu DIINGINKAN, kebalikan DANN).
"""

import numpy as np
import torch
import torch.nn.functional as F
from fastai.callback.core import Callback


class DSNCallback(Callback):
    order = 12

    def __init__(self, lambda_domain: float = 1.0, lambda_ortho: float = 1.0,
                 lambda_recon: float = 1.0, log_every: int = 50) -> None:
        self.lambda_domain = lambda_domain
        self.lambda_ortho = lambda_ortho
        self.lambda_recon = lambda_recon
        self.log_every = log_every
        self._domain_logits = None
        self._domain_tgt = None
        self._ep_dloss: list[float] = []
        self._ep_dacc: list[float] = []
        self._ep_ortho: list[float] = []
        self._ep_recon: list[float] = []
        self._batch = 0

    def before_epoch(self) -> None:
        self._ep_dloss, self._ep_dacc, self._ep_ortho, self._ep_recon, self._batch = [], [], [], [], 0

    def before_batch(self) -> None:
        self._domain_logits = None
        self._domain_tgt = None
        if not self.learn.yb:  # type: ignore
            return
        y = self.learn.yb[0]  # type: ignore
        if y.dim() == 2 and y.shape[1] == 2:
            if self.training:
                dt = y[:, 1].long()
                if (dt < 0).any():
                    raise RuntimeError("DSN: masked domain labels in training batch.")
                self._domain_tgt = dt
            self.learn.yb = (y[:, 0].long(),) + self.learn.yb[1:]  # type: ignore

    def after_pred(self) -> None:
        pred = self.learn.pred  # type: ignore
        if self.training and isinstance(pred, (tuple, list)) and len(pred) >= 2:
            self._domain_logits = pred[1]
            self.learn.pred = pred[0]  # type: ignore

    def after_loss(self) -> None:
        if not (self.training and self._domain_logits is not None and self._domain_tgt is not None):
            self._domain_logits = None
            self._domain_tgt = None
            return

        dev = self._domain_logits.device
        dtgt = self._domain_tgt.to(dev)
        domain_loss = F.cross_entropy(self._domain_logits, dtgt)

        # Orthogonality (difference) loss — canonical ‖H_cᵀH_p‖²_F with per-sample
        # L2-normalized rows (audit fix #5: was mean-centered cross-covariance).
        zt = self.model._z_task  # type: ignore  # (B, D)
        zd = self.model._z_dom   # type: ignore
        bs = max(1, zt.shape[0])
        zt_n = F.normalize(zt, dim=1)
        zd_n = F.normalize(zd, dim=1)
        cross = zt_n.t() @ zd_n               # (D, D)
        ortho_loss = (cross.pow(2).sum()) / float(bs * bs)

        # Reconstruction loss (audit fix #3/#4): private+task must jointly retain feat.
        recon_loss = F.mse_loss(self.model._recon_pred, self.model._recon_target)  # type: ignore

        # Add to the tensor fastai actually backpropagates (loss_grad), mirror to loss.
        extra = (self.lambda_domain * domain_loss + self.lambda_ortho * ortho_loss
                 + self.lambda_recon * recon_loss)
        self.learn.loss_grad = self.learn.loss_grad + extra
        self.learn.loss = self.learn.loss + extra  # type: ignore

        with torch.no_grad():
            dacc = float((self._domain_logits.argmax(1) == dtgt).float().mean().item())
        self._ep_dloss.append(float(domain_loss.item()))
        self._ep_dacc.append(dacc)
        self._ep_ortho.append(float(ortho_loss.item()))
        self._ep_recon.append(float(recon_loss.item()))
        self._batch += 1
        if self.log_every > 0 and self._batch % self.log_every == 0:
            print(f"    [DSN] step={self._batch:4d} | domain_loss={domain_loss.item():.4f} "
                  f"domain_acc={dacc:.3f} | ortho={ortho_loss.item():.4f} recon={recon_loss.item():.4f}")

        self._domain_logits = None
        self._domain_tgt = None

    def after_epoch(self) -> None:
        if not self._ep_dloss:
            return
        ep = getattr(self.learn, "epoch", "?")
        print(f"\n  [DSN-EPOCH {ep}] domain_loss={np.mean(self._ep_dloss):.4f} "
              f"domain_acc={np.mean(self._ep_dacc):.3f} (HIGH = good, domain captured by z_dom) "
              f"| ortho={np.mean(self._ep_ortho):.4f} (LOW = subspaces separated) "
              f"| recon={np.mean(self._ep_recon):.4f} (LOW = feat retained)")
