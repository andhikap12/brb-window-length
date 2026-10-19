"""supcon_callback.py — Callback Supervised Contrastive (Khosla et al. 2020).

Merakit ke loss_grad (BUKAN display-clone — sama seperti fix DANN/DSN):
    L = CE_fault + lambda_sc * SupCon(z, fault_labels)

SupCon (single-view, in-batch positives): untuk tiap anchor i, positif = sampel
lain berkelas-fault SAMA (lintas torsi); negatif = kelas berbeda. Tak ada GRL,
tak ada label domain. Label fault dipakai sebagai supervisi kontras.
"""

import numpy as np
import torch
import torch.nn.functional as F
from fastai.callback.core import Callback


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss (single view, in-batch positives).

    z       : (B, D) L2-normalized embeddings.
    labels  : (B,) integer class labels (fault).
    """
    device = z.device
    B = z.shape[0]
    if B < 2:
        return z.new_tensor(0.0)

    sim = torch.matmul(z, z.t()) / temperature           # (B, B)
    sim_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - sim_max.detach()                      # numerical stability

    labels = labels.view(-1, 1)
    eq = torch.eq(labels, labels.t()).float().to(device)  # same-class mask
    self_mask = 1.0 - torch.eye(B, device=device)         # remove self-pairs
    pos_mask = eq * self_mask                             # positives (self removed)

    exp_logits = torch.exp(logits) * self_mask            # denominator: all but self
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp(min=1.0)
    valid = pos_count > 0                                  # anchors with ≥1 positive
    if not valid.any():
        return z.new_tensor(0.0)
    return -mean_log_prob_pos[valid].mean()


class SupConCallback(Callback):
    order = 12

    def __init__(self, lambda_sc: float = 1.0, temperature: float = 0.07,
                 log_every: int = 50) -> None:
        self.lambda_sc = lambda_sc
        self.temperature = temperature
        self.log_every = log_every
        self._embed = None
        self._fault_tgt = None
        self._ep_sc: list[float] = []
        self._batch = 0

    def before_fit(self) -> None:
        print(f"  [SupCon] lambda_sc={self.lambda_sc}, temperature={self.temperature}")

    def before_epoch(self) -> None:
        self._ep_sc, self._batch = [], 0

    def before_batch(self) -> None:
        self._embed = None
        self._fault_tgt = None
        if not self.learn.yb:  # type: ignore
            return
        y = self.learn.yb[0]  # type: ignore
        if y.dim() == 2 and y.shape[1] == 2:
            # collapse 2-D (fault, domain) -> 1-D fault; SupCon uses fault labels only.
            self.learn.yb = (y[:, 0].long(),) + self.learn.yb[1:]  # type: ignore
        if self.training:
            self._fault_tgt = self.learn.yb[0].long()  # type: ignore

    def after_pred(self) -> None:
        pred = self.learn.pred  # type: ignore
        if self.training and isinstance(pred, (tuple, list)) and len(pred) >= 2:
            self._embed = pred[1]
            self.learn.pred = pred[0]  # type: ignore

    def after_loss(self) -> None:
        if not (self.training and self._embed is not None and self._fault_tgt is not None):
            self._embed = None
            self._fault_tgt = None
            return

        sc = supcon_loss(self._embed, self._fault_tgt.to(self._embed.device), self.temperature)
        extra = self.lambda_sc * sc
        self.learn.loss_grad = self.learn.loss_grad + extra
        self.learn.loss = self.learn.loss + extra  # type: ignore

        self._ep_sc.append(float(sc.item()))
        self._batch += 1
        if self.log_every > 0 and self._batch % self.log_every == 0:
            print(f"    [SupCon] step={self._batch:4d} | supcon_loss={sc.item():.4f}")

        self._embed = None
        self._fault_tgt = None

    def after_epoch(self) -> None:
        if not self._ep_sc:
            return
        ep = getattr(self.learn, "epoch", "?")
        print(f"\n  [SupCon-EPOCH {ep}] supcon_loss={np.mean(self._ep_sc):.4f} "
              f"(LOW = same-fault tight across loads)")
