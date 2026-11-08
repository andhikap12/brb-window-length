"""domain_objective.py — UJI A: objektif diskriminator yang bisa kategoris ATAU skalar."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class DomainObjectiveMixin:
    """Loss + telemetri domain, kategoris atau regresi."""

    domain_u: np.ndarray | None

    def _init_domain_objective(self, domain_u) -> None:
        self.domain_u = None if domain_u is None else np.asarray(domain_u, dtype=np.float32)
        self._u_tensor: torch.Tensor | None = None
        self._ep_corr: list[float] = []

    @property
    def domain_regression(self) -> bool:
        return self.domain_u is not None

    # -- target ---------------------------------------------------------------

    def _u_for(self, dtgt: torch.Tensor) -> torch.Tensor:
        """indeks domain (long) -> skalar u di [0,1], pada device yang sama."""
        if self._u_tensor is None or self._u_tensor.device != dtgt.device:
            self._u_tensor = torch.as_tensor(self.domain_u, device=dtgt.device)
        return self._u_tensor[dtgt]

    # -- loss -----------------------------------------------------------------

    def _dom_loss(self, logits: torch.Tensor, dtgt: torch.Tensor,
                  label_smoothing: float = 0.0) -> torch.Tensor:
        if not self.domain_regression:
            return F.cross_entropy(logits, dtgt, label_smoothing=label_smoothing)
        return F.mse_loss(logits.reshape(-1), self._u_for(dtgt))

    # -- telemetri ------------------------------------------------------------

    def _dom_acc(self, logits: torch.Tensor, dtgt: torch.Tensor) -> float:
        """Akurasi domain (kategoris). Pada regresi tidak terdefinisi -> NaN."""
        if self.domain_regression:
            return float("nan")
        return float((logits.argmax(dim=1) == dtgt).float().mean().item())

    def _dom_corr(self, logits: torch.Tensor, dtgt: torch.Tensor) -> float:
        """korelasi Pearson(D(z), u) — A-P2. Hanya bermakna pada regresi."""
        if not self.domain_regression:
            return float("nan")
        pred = logits.reshape(-1).detach().float()
        u = self._u_for(dtgt).detach().float()
        if pred.numel() < 2:
            return float("nan")
        pc, uc = pred - pred.mean(), u - u.mean()
        denom = float(pc.norm().item() * uc.norm().item())
        if denom < 1e-12:
            return float("nan")
        return float((pc @ uc).item() / denom)

    def _corr_epoch_report(self, tag: str) -> str:
        """Ringkasan korelasi per epoch; string kosong bila bukan mode regresi."""
        if not self.domain_regression:
            return ""
        nilai = [c for c in self._ep_corr if np.isfinite(c)]
        self._ep_corr = []
        if not nilai:
            return ""
        r = float(np.mean(nilai))
        # A-P2 gugur bila korelasi < -0,5 (jalan kabur pembalikan monoton terjadi)
        vonis = "PEMBALIKAN MONOTON" if r < -0.5 else ("positif" if r > 0 else "lemah/negatif")
        return f" | corr(D(z),u)={r:+.4f} [{vonis}]"
