"""bt_dann_callback.py — BT-DANN (*Balanced Target*): normalisasi gradien adversarial."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.training.callbacks import DANNCallback
from src.training.dann_supcon_callback import DANNSupConCallback
from src.training.supcon_callback import supcon_loss

EPS = 1e-12


class _GradBalanceMixin:
    """Pengukuran dan penjadwalan lambda berbasis keseimbangan gradien."""

    # diisi oleh __init__ kelas turunan
    target_ratio: float
    lambda_cap: float
    ema_beta: float
    anneal: bool
    gamma: float

    def _init_balance(self) -> None:
        self._feat: torch.Tensor | None = None
        self._hook = None
        self._log_ratio_ema: float = 0.0
        self._have_measurement: bool = False
        self._ep_ratio_raw: list[float] = []
        self._ep_ratio_eff: list[float] = []
        self._ep_lambda: list[float] = []


    # -- kait fitur ---------------------------------------------------------

    def _install_hook(self) -> None:
        backbone = getattr(self.model, "backbone", None)  # type: ignore[attr-defined]
        if backbone is None:
            raise RuntimeError(
                "BT-DANN memerlukan model.backbone untuk mengait vektor fitur."
            )

        def _simpan(_module, _inp, out):
            self._feat = out

        self._hook = backbone.register_forward_hook(_simpan)

    def _remove_hook(self) -> None:
        if getattr(self, "_hook", None) is not None:
            self._hook.remove()
            self._hook = None

    # -- lambda -------------------------------------------------------------

    def _anneal_factor(self) -> float:
        if not self.anneal:
            return 1.0
        p = float(getattr(self.learn, "pct_train", 0.0) or 0.0)  # type: ignore[attr-defined]
        return 2.0 / (1.0 + np.exp(-self.gamma * p)) - 1.0

    def _get_lambda(self) -> float:
        """Menggantikan anneal Ganin: lambda kini ditentukan keseimbangan gradien."""
        if not self._have_measurement:
            return 0.0
        rasio_halus = float(np.exp(self._log_ratio_ema))
        skala = self.target_ratio / max(rasio_halus, EPS)
        anneal = self._anneal_factor()
        return float(np.clip(anneal * skala, 0.0, self.lambda_cap))

    # -- pengukuran ---------------------------------------------------------

    def _measure_and_update(self, nonadv_loss: torch.Tensor, domain_tgt: torch.Tensor) -> None:
        """Ukur ||g_nonadv|| dan ||g_dom_mentah|| di `feat`, lalu perbarui EMA lambda."""
        feat = self._feat
        if feat is None or not feat.requires_grad:
            return

        try:
            (g_nonadv,) = torch.autograd.grad(
                nonadv_loss, feat, retain_graph=True, allow_unused=False
            )
        except RuntimeError:
            return

        feat_probe = feat.detach().requires_grad_(True)
        dom_logits = self.model.domain_head(feat_probe)  # type: ignore[attr-defined]
        loss_probe = self._dom_loss(dom_logits, domain_tgt)  # type: ignore[attr-defined]
        (g_dom_raw,) = torch.autograd.grad(loss_probe, feat_probe)

        n_nonadv = float(g_nonadv.norm().item())
        n_dom = float(g_dom_raw.norm().item())
        if not np.isfinite(n_nonadv) or not np.isfinite(n_dom) or n_dom < EPS:
            return

        rasio_mentah = n_dom / max(n_nonadv, EPS)
        log_rasio = float(np.log(max(rasio_mentah, EPS)))

        if self._have_measurement:
            self._log_ratio_ema = (
                self.ema_beta * self._log_ratio_ema + (1 - self.ema_beta) * log_rasio
            )
        else:
            self._log_ratio_ema = log_rasio
            self._have_measurement = True

        lam_kini = float(getattr(self.model, "lambda_grl", 0.0))  # type: ignore[attr-defined]
        rasio_efektif_kini = lam_kini * rasio_mentah
        self._ep_ratio_raw.append(rasio_mentah)
        self._ep_ratio_eff.append(rasio_efektif_kini)
        self._ep_lambda.append(lam_kini)


    # -- pelaporan ----------------------------------------------------------

    def _balance_epoch_report(self, tag: str) -> None:
        if not self._ep_ratio_eff:
            return
        ep = getattr(self.learn, "epoch", "?")  # type: ignore[attr-defined]
        print(
            f"  [{tag}-BAL ep {ep}] rasio_efektif={np.median(self._ep_ratio_eff):.3f} "
            f"(target {self.target_ratio:.2f}) | lambda_eff={np.median(self._ep_lambda):.3f} "
            f"| rasio_mentah={np.median(self._ep_ratio_raw):.3f}"
        )
        self._ep_ratio_raw, self._ep_ratio_eff, self._ep_lambda = [], [], []

class BTDANNCallback(_GradBalanceMixin, DANNCallback):
    """DANN dengan gradien adversarial dinormalisasi terhadap gradien klasifikasi."""

    def __init__(
        self,
        target_ratio: float = 0.3,
        gamma: float = 10.0,
        log_every: int = 50,
        domain_label_smoothing: float = 0.0,
        lambda_cap: float = 10.0,
        ema_beta: float = 0.7,
        anneal: bool = True,
    ) -> None:
        # lambda_max tidak lagi menjadi knob; rasio_target yang mengendalikan.
        super().__init__(
            lambda_max=1.0,
            gamma=gamma,
            log_every=log_every,
            domain_label_smoothing=domain_label_smoothing,
        )
        self.target_ratio = target_ratio
        self.lambda_cap = lambda_cap
        self.ema_beta = ema_beta
        self.anneal = anneal
        self._init_balance()

    def before_fit(self) -> None:
        self._install_hook()
        print(
            f"\n  [BT-DANN] normalisasi gradien AKTIF — rasio_target={self.target_ratio}, "
            f"cap={self.lambda_cap}, ema={self.ema_beta}, "
            f"anneal={self.anneal}"
        )
        print(
            "  [BT-DANN] lambda kini ditentukan keseimbangan gradien, bukan jadwal waktu. "
            "Rujukan tanpa normalisasi: rasio 0,48 -> 5,06 (Sec. IV-D makalah CENIM 2026)."
        )
        super().before_fit()

    def after_fit(self) -> None:
        self._remove_hook()

    def after_loss(self) -> None:
        if self.training and self._domain_logits is not None and self._domain_tgt is not None:
            # loss_grad di titik ini masih murni CE fault — belum ditambah suku domain.
            self._measure_and_update(
                self.learn.loss_grad,  # type: ignore[attr-defined]
                self._domain_tgt.to(self._domain_logits.device),
            )
        super().after_loss()

    def after_epoch(self) -> None:
        self._balance_epoch_report("BT-DANN")
        super().after_epoch()


class BTDANNSupConCallback(_GradBalanceMixin, DANNSupConCallback):
    """DANN+SupCon dengan normalisasi gradien."""

    def __init__(
        self,
        target_ratio: float = 0.3,
        gamma: float = 10.0,
        lambda_sc: float = 1.0,
        temperature: float = 0.07,
        log_every: int = 50,
        lambda_cap: float = 10.0,
        ema_beta: float = 0.7,
        anneal: bool = True,
        domain_u=None,
    ) -> None:
        super().__init__(
            lambda_max=1.0,
            gamma=gamma,
            lambda_sc=lambda_sc,
            temperature=temperature,
            log_every=log_every,
            domain_u=domain_u,   # UJI A: regresi indeks domain di atas BT
        )
        self.target_ratio = target_ratio
        self.lambda_cap = lambda_cap
        self.ema_beta = ema_beta
        self.anneal = anneal
        self._init_balance()

    def before_fit(self) -> None:
        self._install_hook()
        print(
            f"\n  [BT-DANN+SupCon] normalisasi gradien AKTIF — rasio_target={self.target_ratio}, "
            f"cap={self.lambda_cap}, ema={self.ema_beta}, "
            f"anneal={self.anneal}"
        )
        print(
            "  [BT-DANN+SupCon] penyeimbang = CE_fault + lambda_sc*SupCon (bukan CE saja)."
        )
        super().before_fit()

    def after_fit(self) -> None:
        self._remove_hook()

    def after_loss(self) -> None:
        if (
            self.training
            and self._domain_logits is not None
            and self._domain_tgt is not None
            and self._embed is not None
        ):
            sc = supcon_loss(
                self._embed, self._fault_tgt.to(self._embed.device), self.temperature
            )
            nonadv = self.learn.loss_grad + self.lambda_sc * sc  # type: ignore[attr-defined]
            self._measure_and_update(
                nonadv, self._domain_tgt.to(self._domain_logits.device)
            )
        super().after_loss()

    def after_epoch(self) -> None:
        self._balance_epoch_report("BT-DANN+SupCon")
        super().after_epoch()
