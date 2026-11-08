"""callbacks.py - FastAI Callback for Domain-Adversarial Inter-Source Alignment."""

import math
import numpy as np
import torch
import torch.nn.functional as F
from fastai.callback.core import Callback

from src.training.domain_objective import DomainObjectiveMixin


class DANNCallback(DomainObjectiveMixin, Callback):
    """FastAI Callback implementing Domain-Adversarial Inter-Source Alignment for DG."""

    order = 12

    def __init__(
        self,
        lambda_max: float = 1.0,
        gamma: float = 10.0,
        log_every: int = 50,
        domain_label_smoothing: float = 0.0,
        domain_u=None,
    ) -> None:
        self.lambda_max = lambda_max
        self.gamma = gamma
        self.log_every = log_every
        # UJI A: tabel u[indeks_domain] -> kepala domain jadi REGRESI (MSE). None = lama.
        self._init_domain_objective(domain_u)
        self.domain_label_smoothing = domain_label_smoothing
        self._domain_logits: torch.Tensor | None = None
        self._domain_tgt: torch.Tensor | None = None

        # Per-epoch accumulators for convergence monitoring
        self._epoch_domain_losses: list[float] = []
        self._epoch_domain_accs: list[float] = []
        self._epoch_lambdas: list[float] = []
        self._batch_count: int = 0


    def _get_lambda(self) -> float:
        """DANN λ annealing: 0 at start → lambda_max at end of training."""
        p = float(getattr(self.learn, "pct_train", 0.0))
        return self.lambda_max * (2.0 / (1.0 + np.exp(-self.gamma * p)) - 1.0)

    def _theoretical_chance(self, n_domains: int) -> float:
        """Theoretical domain classifier accuracy under maximum entropy (random chance)."""
        return 1.0 / max(n_domains, 1)

    def _theoretical_max_loss(self, n_domains: int) -> float:
        """Maximum entropy loss = log(n_domains), target for a fully-confused domain classifier."""
        return math.log(max(n_domains, 1))


    def before_fit(self) -> None:
        """Print alignment protocol summary at start of training."""
        n_domains = getattr(self.model, "domain_head", None)
        # Try to infer n_domains from the final linear layer of domain_head
        try:
            n_dom = list(self.model.domain_head.modules())[-1].out_features
        except Exception:
            n_dom = "?"
        chance = self._theoretical_chance(n_dom) if isinstance(n_dom, int) else "?"
        max_loss = self._theoretical_max_loss(n_dom) if isinstance(n_dom, int) else "?"
        print(f"\n  [GRL-DG] Strategy: Domain-Adversarial Inter-Source Alignment")
        print(f"  [GRL-DG] lambda_max={self.lambda_max}, gamma={self.gamma}")
        print(f"  [GRL-DG] n_source_domains={n_dom}")
        print(f"  [GRL-DG] Convergence targets: domain_loss→{max_loss:.3f} (log(n)), "
              f"domain_acc→{chance:.3f} (1/n = random chance)")
        print(f"  [GRL-DG] GRL log interval: every {self.log_every} batches\n")

    def before_epoch(self) -> None:
        """Reset per-epoch accumulators."""
        self._epoch_domain_losses = []
        self._epoch_domain_accs = []
        self._epoch_lambdas = []
        self._batch_count = 0

    def before_batch(self) -> None:
        self._domain_logits = None
        self._domain_tgt = None

        # 1. Anneal GRL strength
        if hasattr(self.model, "lambda_grl"):
            self.model.lambda_grl = self._get_lambda()

        if not self.learn.yb:  # type: ignore
            return
        y = self.learn.yb[0]  # type: ignore
        if y.dim() == 2 and y.shape[1] == 2:
            if self.training:
                domain_tgt = y[:, 1].long()
                if (domain_tgt < 0).any():
                    raise RuntimeError(
                        "Training batch contains masked domain labels. "
                        "Check per-fold domain remapping."
                    )
                self._domain_tgt = domain_tgt
            # Replace yb so loss / metrics receive 1-D fault labels
            self.learn.yb = (y[:, 0].long(),) + self.learn.yb[1:]  # type: ignore

    def after_pred(self) -> None:
        pred = self.learn.pred  # type: ignore
        if self.training and isinstance(pred, (tuple, list)) and len(pred) >= 2:
            fault_logits = pred[0]
            domain_logits = pred[1]
            self._domain_logits = domain_logits
            self.learn.pred = fault_logits  # type: ignore # downstream sees only this

    def after_loss(self) -> None:
        if (
            self.training
            and self._domain_logits is not None
            and self._domain_tgt is not None
        ):
            lam = self._get_lambda()
            domain_tgt_dev = self._domain_tgt.to(self._domain_logits.device)
            # CE kategoris (dengan ELS), atau MSE terhadap indeks skalar bila UJI A aktif.
            domain_loss = self._dom_loss(
                self._domain_logits, domain_tgt_dev,
                label_smoothing=self.domain_label_smoothing,  # ELS (ICLR 2023): 0.0 = off
            )

            self.learn.loss_grad = self.learn.loss_grad + domain_loss
            self.learn.loss = self.learn.loss + domain_loss  # type: ignore

            # ── GRL Convergence Monitoring ───────────────────────────────
            with torch.no_grad():
                domain_acc = self._dom_acc(self._domain_logits, domain_tgt_dev)
                corr = self._dom_corr(self._domain_logits, domain_tgt_dev)
                d_loss_val = float(domain_loss.item())
                n_domains = int(getattr(self.model, "n_domains",
                                        self._domain_logits.shape[-1]))
            if np.isfinite(corr):
                self._ep_corr.append(corr)

            self._epoch_domain_losses.append(d_loss_val)
            self._epoch_domain_accs.append(domain_acc)
            self._epoch_lambdas.append(lam)
            self._batch_count += 1

            # Batch-level log (if enabled)
            if self.log_every > 0 and self._batch_count % self.log_every == 0:
                if self.domain_regression:
                    print(
                        f"    [GRL-REG] step={self._batch_count:4d} | λ={lam:.3f} | "
                        f"domain_mse={d_loss_val:.5f} | corr(D(z),u)={corr:+.3f}"
                    )
                else:
                    chance = self._theoretical_chance(n_domains)
                    max_loss = self._theoretical_max_loss(n_domains)
                    convergence_pct = min(100.0, 100.0 * d_loss_val / max_loss) if max_loss > 0 else 0.0
                    print(
                        f"    [GRL] step={self._batch_count:4d} | "
                        f"λ={lam:.3f} | "
                        f"domain_loss={d_loss_val:.4f}/{max_loss:.3f} "
                        f"({convergence_pct:.0f}% of max_entropy) | "
                        f"domain_acc={domain_acc:.3f} (chance={chance:.3f})"
                    )

        # Reset for next batch
        self._domain_logits = None
        self._domain_tgt = None

    def after_epoch(self) -> None:
        """Print per-epoch GRL convergence summary."""
        if not self._epoch_domain_losses:
            return

        avg_loss = float(np.mean(self._epoch_domain_losses))
        avg_acc = float(np.mean(self._epoch_domain_accs))
        avg_lam = float(np.mean(self._epoch_lambdas))

        if self.domain_regression:
            epoch_num = getattr(self.learn, "epoch", "?")
            status = "WARMUP (λ≈0, adversarial inactive)" if avg_lam < 0.01 else "AKTIF"
            print(
                f"\n  [GRL-REG-EPOCH {epoch_num}] λ={avg_lam:.3f} | "
                f"domain_mse={avg_loss:.5f}{self._corr_epoch_report('')} | "
                f"STATUS: {status}"
            )
            return

        try:
            n_domains = list(self.model.domain_head.modules())[-1].out_features
        except Exception:
            n_domains = max(1, round(1.0 / avg_acc)) if avg_acc > 0 else 1

        chance = self._theoretical_chance(n_domains)
        max_loss = self._theoretical_max_loss(n_domains)

        if max_loss > 0:
            alignment_quality = min(100.0, 100.0 * avg_loss / max_loss)
        else:
            alignment_quality = 0.0

        # Warn if GRL is clearly not converging (λ is active but domain acc still high)
        is_grl_active = avg_lam > 0.1
        is_domain_discriminative = avg_acc > 3.0 * chance
        epoch_num = getattr(self.learn, "epoch", "?")

        status = "OK"
        if is_grl_active and is_domain_discriminative:
            status = "WARN — domain classifier still discriminative, backbone not yet domain-invariant"
        elif avg_lam < 0.01:
            status = "WARMUP (λ≈0, adversarial inactive)"
        elif alignment_quality >= 80.0:
            status = "CONVERGING"

        print(
            f"\n  [GRL-EPOCH {epoch_num}] λ={avg_lam:.3f} | "
            f"domain_loss={avg_loss:.4f}/{max_loss:.3f} "
            f"(alignment={alignment_quality:.0f}%) | "
            f"domain_acc={avg_acc:.3f} vs chance={chance:.3f} | "
            f"STATUS: {status}"
        )
