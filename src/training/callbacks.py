"""
callbacks.py - FastAI Callback for Domain-Adversarial Inter-Source Alignment.

Strategy: Domain Generalization via Inter-Source Adversarial Training
----------------------------------------------------------------------
This implements the "domain-adversarial inter-source alignment" DG strategy:
    - During training fold k, the backbone sees N-1 source domains (torque levels).
    - The GRL forces the backbone to be INVARIANT across all N-1 seen torque levels.
    - Hypothesis: if f(x) is invariant across all seen loads, it generalises to
      the unseen held-out load (fold k validation domain).
    - The validation domain is NEVER seen by the backbone — evaluation is
      purely transductive-free. This is Domain Generalization, not Domain Adaptation.

Reference: Li et al. (2018) "Domain Generalization with Adversarial Feature Statistics";
           Zhao et al. (2020) "Domain Generalization via Entropy Regularization".

How it integrates with the standard FastAI training loop
---------------------------------------------------------

FastAI training step (simplified):
    1. before_batch   ← DANNCallback extracts domain targets from 2-D yb;
                        anneals λ in the model
    2. model.forward(xb)  → (fault_logits, domain_logits)  [training mode]
    3. after_pred     ← DANNCallback hides domain_logits, exposes only
                        fault_logits so that standard metrics/loss work
    4. loss = cross_entropy(fault_logits, fault_targets)
    5. after_loss     ← DANNCallback adds  λ × CE(domain_logits, domain_targets)
                        to the total loss
    6. loss.backward()
       • domain_head  weights: standard gradient descent (domain CE minimised)
       • backbone     weights: gradient from fault CE  MINUS  λ × gradient
                               from domain CE  (because GRL negated the sign)
       → backbone is pushed to be discriminative for faults, indiscriminate
         for domains (operating conditions / torque levels).

Target format
-------------
The DataLoader must supply y as a *2-D integer tensor* of shape (N, 2):
    column 0  — fault class index    (0 … n_fault_classes-1)
    column 1  — domain (torque) index (0 … n_domains-1)

Use ``np.stack([y_fault, y_domain], axis=1)`` when building the DataLoaders.
The Callback splits this in ``before_batch`` so all downstream code (loss,
metrics, Recorder) only sees the 1-D fault targets.

λ annealing schedule (DANN paper, Ganin et al. 2016)
------------------------------------------------------
    p = training progress ∈ [0, 1]
    λ(p) = λ_max · (2 / (1 + exp(−γ · p)) − 1)

    p=0 → λ=0  (no adversarial signal at start, let fault head warm up)
    p=1 → λ≈λ_max  (full adversarial pressure at end of training)

GRL Convergence Verification (Inter-Source Alignment Monitor)
-------------------------------------------------------------
The GRL is converging correctly when the domain classifier cannot predict
the torque level better than chance. Theoretical bounds:
    - domain_loss ≈ log(n_domains)  →  maximum entropy = domain classifier confused
    - domain_loss > log(n_domains)  →  OVER-ALIGNMENT, not convergence: the discriminator
                                       is confidently WRONG. log(n) is the entropy CEILING
                                       for a uniform predictor, so exceeding it is a
                                       pathology. Measured in 32/32 variant x seed
                                       combinations (Sec. IV-D of the paper). Corrected
                                       2026-08-03: this line previously read ">= log(n) ->
                                       converged", which would report success exactly when
                                       the system is over-aligning.
    - domain_acc ≈ 1/n_domains      →  random guessing
    - domain_acc ≫ 1/n_domains      →  GRL NOT converging, backbone still domain-discriminative

CAUTION: none of the above is a valid measure of representation invariance. The online
discriminator lags a backbone that moves every step; measured discrepancy against a
frozen-backbone probe is 0.79 (Sec. IV-D of the paper). Use these for monitoring only.
Every verdict reported in the paper uses the frozen-backbone probe instead.

Logged every `log_every` batches and summarised per epoch via `after_epoch`.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from fastai.callback.core import Callback

from src.training.domain_objective import DomainObjectiveMixin


class DANNCallback(DomainObjectiveMixin, Callback):
    """
    FastAI Callback implementing Domain-Adversarial Inter-Source Alignment for DG.

    Strategy
    --------
    The GRL forces the backbone to produce representations that are invariant
    across ALL N-1 seen training domains (torque levels / speeds). By induction,
    these representations generalise to the unseen held-out domain (fold validation).
    This is a Domain Generalization strategy, NOT Domain Adaptation —
    the validation domain is never observed during training.

    GRL Convergence Monitor
    -----------------------
    Logs per-batch and per-epoch statistics for the domain classifier:
        - domain_loss  : should approach log(n_domains) when confused (max entropy)
        - domain_acc   : should approach 1/n_domains  when confused (random chance)
        - lambda (λ)   : annealed 0 → lambda_max during training

    Parameters
    ----------
    lambda_max : float
        Maximum GRL reversal strength (reached at end of training).
        Typical values: 0.5 – 1.0.
    gamma : float
        Annealing speed.  Higher values → λ rises faster.
        DANN paper used γ = 10.
    log_every : int
        Log GRL stats every N training batches (0 = disable batch-level log).
    """

    # Run after TrainEvalCallback (order=10) which sets model.training,
    # but before Recorder (order=50) which logs metrics.
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
        # Environment Label Smoothing (ELS, Zhang et al. ICLR 2023, "Free Lunch for
        # Domain Adversarial Training"): smooth the domain labels in the discriminator
        # CE so it outputs soft probabilities → less over-confident discriminator →
        # more stable adversarial training, lower variance. 0.0 = off (vanilla DANN).
        self.domain_label_smoothing = domain_label_smoothing
        self._domain_logits: torch.Tensor | None = None
        self._domain_tgt: torch.Tensor | None = None

        # Per-epoch accumulators for convergence monitoring
        self._epoch_domain_losses: list[float] = []
        self._epoch_domain_accs: list[float] = []
        self._epoch_lambdas: list[float] = []
        self._batch_count: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

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

        # 2. Split combined 2-D target into fault target + domain target.
        #    yb[0] shape: (batch, 2)  →  col 0 = fault,  col 1 = domain
        # Guard: label-less inference dataloaders (e.g. test_dl for the held-out
        # torque) carry no targets — nothing to split, skip safely.
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
        # During training the model returns (fault_logits, domain_logits).
        # Expose only fault_logits to the standard loss / metric pipeline.
        pred = self.learn.pred  # type: ignore
        if self.training and isinstance(pred, (tuple, list)) and len(pred) >= 2:
            fault_logits = pred[0]
            domain_logits = pred[1]
            self._domain_logits = domain_logits
            self.learn.pred = fault_logits  # type: ignore # downstream sees only this

    def after_loss(self) -> None:
        # Add λ-weighted domain adversarial loss to the total loss.
        # The GRL in the backbone already negates the gradient for the
        # feature extractor; here we add the *positive* domain loss for
        # the domain head's own gradient.
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

            # Standard DANN (Ganin 2016): domain loss enters with WEIGHT 1; lambda is applied
            # ONLY by the GRL (which scales the reversed gradient to the backbone by -lambda).
            # CRITICAL: fastai backpropagates `loss_grad` (Learner._backward: self.loss_grad
            # .backward()), NOT `loss` (a display-only clone). The original code added the
            # domain loss to `self.learn.loss`, so it was NEVER backpropagated → the entire
            # adversarial term (domain head + GRL) was inert → DANN == baseline in every
            # experiment. Add to loss_grad so it actually trains; mirror into loss for logging.
            self.learn.loss_grad = self.learn.loss_grad + domain_loss
            self.learn.loss = self.learn.loss + domain_loss  # type: ignore

            # ── GRL Convergence Monitoring ───────────────────────────────
            with torch.no_grad():
                domain_acc = self._dom_acc(self._domain_logits, domain_tgt_dev)
                corr = self._dom_corr(self._domain_logits, domain_tgt_dev)
                d_loss_val = float(domain_loss.item())
                # Pada regresi keluarannya 1 kolom; n_domains diambil dari model agar
                # plafon ln(n) yang dicetak tetap merujuk jumlah domain sebenarnya.
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
                    # Tidak ada plafon ln(n) pada regresi — yang dipantau MSE
                    # dan tanda korelasi, bukan persen entropi maksimum.
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
        """Print per-epoch GRL convergence summary.

        Uses the training-phase accumulators (populated in after_loss during the
        train pass). after_epoch fires once per epoch AFTER validation, so
        self.training is already False here — gating on it would suppress the
        summary entirely (the old bug: invisible domain_acc when batches/epoch <
        log_every, e.g. 43 < 50). Gate only on having accumulated data.
        """
        if not self._epoch_domain_losses:
            return

        avg_loss = float(np.mean(self._epoch_domain_losses))
        avg_acc = float(np.mean(self._epoch_domain_accs))
        avg_lam = float(np.mean(self._epoch_lambdas))

        # UJI A: pada regresi tidak ada plafon ln(n) maupun chance 1/n, sehingga
        # seluruh blok "alignment %" di bawah tidak terdefinisi. Yang menggantikannya
        # sebagai tanda patologi adalah korelasi(D(z), u) — A-P2 gugur bila < -0,5.
        if self.domain_regression:
            epoch_num = getattr(self.learn, "epoch", "?")
            status = "WARMUP (λ≈0, adversarial inactive)" if avg_lam < 0.01 else "AKTIF"
            print(
                f"\n  [GRL-REG-EPOCH {epoch_num}] λ={avg_lam:.3f} | "
                f"domain_mse={avg_loss:.5f}{self._corr_epoch_report('')} | "
                f"STATUS: {status}"
            )
            return

        # Infer n_domains from accumulated logit shape (stored in last domain_logits isn't
        # available after_epoch, so derive from acc list length and model)
        try:
            n_domains = list(self.model.domain_head.modules())[-1].out_features
        except Exception:
            n_domains = max(1, round(1.0 / avg_acc)) if avg_acc > 0 else 1

        chance = self._theoretical_chance(n_domains)
        max_loss = self._theoretical_max_loss(n_domains)

        # Alignment quality: 0% = domain classifier still good (GRL not working)
        #                   100% = domain classifier fully confused (GRL converged)
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
