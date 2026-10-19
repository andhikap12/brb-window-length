"""dann_factory.py — Factory DataLoader & Learner DANN (fastai/tsai).

build_dann_learner(model_selection=True) → early stop + best ckpt;
model_selection=False → fixed-budget, tanpa seleksi.
"""

import numpy as np
import torch
import torch.nn.functional as F
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.learner import Learner
from fastai.metrics import accuracy
from tsai.basics import (
    AccumMetric, TSMagScale, TSNan2Value, TSStandardize, TSTimeNoise,
    TSWindowWarp, get_ts_dls, set_seed,
)

from src.config import SEED
from src.models.dann import build_dann_model
from src.models.dsn import build_dsn_model
from src.models.dsran import build_dsran_model
from src.models.supcon import build_supcon_model
from src.models.dann_supcon import build_dann_supcon_model
from src.models.cdan import build_cdan_model, build_cdan_supcon_model
from src.training.callbacks import DANNCallback
from src.training.cdan_callback import CDANCallback, CDANSupConCallback
from src.training.dsn_callback import DSNCallback
from src.training.dsran_callback import DSRANCallback
from src.training.supcon_callback import SupConCallback
from src.training.dann_supcon_callback import DANNSupConCallback
from src.training.bt_dann_callback import BTDANNCallback, BTDANNSupConCallback
from src.training.metrics import win_f1_macro_metric, roc_auc_metric


def build_dann_dataloaders(
    X: np.ndarray,
    y_fault: np.ndarray,
    y_domain: np.ndarray,
    splits: tuple,
    batch_size: int = 64,
    augment: bool = True,
    num_workers: int = 0,
    seed: int = SEED,
    by_sample_standardize: bool = True,
    by_var_standardize: bool = False,
    pin_memory: bool = False,
):
    """
    Build tsai DataLoaders with 2-D combined targets.

    y_combined[:, 0] = fault class index
    y_combined[:, 1] = domain (torque level / speed) index

    The DANNCallback splits this in ``before_batch`` so all loss / metric
    computations receive 1-D fault labels.

    num_workers=0 is REQUIRED on this Windows+CUDA setup, unconditionally — tested
    empirically 28 Aug 2026: num_workers>0 crashes even
    with --fp32 ("Cannot pickle CUDA storage"), because the dataset tensors are
    already CUDA-resident before DataLoader workers are spawned (spawn-based
    multiprocessing on Windows, unlike Linux fork). This is NOT specific to
    to_fp16() as an earlier comment here claimed — do not re-attempt num_workers>0
    without first keeping the underlying data on CPU until inside each worker.
    """
    set_seed(seed)

    y_combined = np.stack([y_fault, y_domain], axis=1).astype(np.int64)

    # Default: per-sample z-norm (fixed protocol). Opt-in global (by_var) standardization
    # PRESERVES inter-window amplitude — used only for the domain-shift ablation.
    #
    # by_var_standardize (2026-07-31): per-sample AND per-channel statistics. Distinct from
    # --global_standardize, which drops per-sample normalization entirely (two changes at
    # once). Motivation is measured, not stylistic: channel std spans 25.8x across the six
    # channels (Vib_base 12.14 ... Vib_axial 0.47), so one shared mean/std suppresses
    # Vib_axial to 3.9% and the Hilbert envelope to 5.7% of the dominant channel — two of
    # six channels are effectively discarded. See Sec. IV-C of the paper.
    if by_sample_standardize:
        _std = TSStandardize(by_sample=True, by_var=by_var_standardize)
    else:
        _std = TSStandardize(by_sample=False, by_var=True)
    batch_tfms = [TSNan2Value(), _std]  # NaN→0 BEFORE normalization (BUG-2)

    if augment:
        batch_tfms += [TSWindowWarp(magnitude=0.1, ex=None),
                       TSMagScale(magnitude=0.2, ex=None),
                       TSTimeNoise(magnitude=0.01, ex=None)]

    # tfms=[None, None] — do NOT Categorize y; we manage 2-D labels ourselves.
    dls = get_ts_dls(
        X,
        y_combined,
        splits=splits,
        tfms=[None, None],
        batch_tfms=batch_tfms,
        bs=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # dls.c is inferred from y_combined and would be wrong (2). Override to the
    # true fault class count so Learner reports correct c.
    dls.c = int(np.unique(y_fault).size)  # type: ignore[attr-defined]
    return dls


def build_dann_learner(
    dls,
    n_fault_classes: int,
    n_domains: int,
    lambda_max: float,
    model_name: str,
    save_dir,
    wd: float = 1e-5,
    patience: int = 10,
    model_selection: bool = True,
    backbone: str = "tsitplus",
    grl_gamma: float = 10.0,
    domain_label_smoothing: float = 0.0,
    bt_target_ratio: float | None = None,
    bt_anneal: bool = True,
) -> Learner:
    """
    Build a fastai Learner wrapping DANNModel with the backbone chosen by the `backbone` arg (default: tsitplus).

    Fault loss is plain cross-entropy; the adversarial domain loss is added by
    DANNCallback.after_loss().

    model_selection:
      True  — add SaveModelCallback (best by win_f1_macro) + EarlyStopping
              (valid_loss). Use when a leakage-free inner validation split exists
              (motor).
      False — DANNCallback only: fixed epoch budget, no selection. Use when no
              leakage-free inner split is possible (motor, --fixed_budget).
    """
    model = build_dann_model(
        backbone,
        c_in=dls.vars,
        seq_len=dls.len,
        n_classes=n_fault_classes,
        n_domains=n_domains,
        lambda_grl=0.0,
    ).to(dls.device)

    # Verify domain labels are in range [0, n_domains-1] — catch remapping bugs early
    train_domain_labels = torch.tensor(
        [int(y[1]) for y in dls.train_ds.tls[-1] if int(y[1]) >= 0],
        dtype=torch.long,
    )
    if len(train_domain_labels) > 0:
        model.verify_domain_labels(train_domain_labels)
    model.sanity_check(c_in=dls.vars, seq_len=dls.len)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  Model: DANNModel | backbone={backbone} | params={n_params:,} | "
        f"c_in={dls.vars} | c_out={n_fault_classes} | "
        f"n_domains={n_domains} | lambda_max={lambda_max}"
    )

    def fault_loss(pred, targ):
        # During inference pred is just fault_logits; during training after_pred
        # already replaced pred with fault_logits.
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        return F.cross_entropy(pred, targ.long())

    roc_auc = AccumMetric(roc_auc_metric, flatten=False)
    win_f1_macro = AccumMetric(win_f1_macro_metric, flatten=False)
    win_f1_macro.name = "win_f1_macro"   # label shown in fastai progress
    roc_auc.name = "roc_auc"

    save_dir.mkdir(parents=True, exist_ok=True)

    # BT-DANN (Sec. IV-A dan IV-D makalah): bila bt_target_ratio diberikan, lambda tidak lagi
    # dijadwalkan menurut waktu melainkan dihitung agar ||g_dom||/||g_fault|| tetap di
    # rasio itu. Aditif dan default mati — None berarti perilaku DANN lama persis.
    if bt_target_ratio is not None:
        cbs = [BTDANNCallback(target_ratio=bt_target_ratio, gamma=grl_gamma,
                              domain_label_smoothing=domain_label_smoothing,
                              anneal=bt_anneal)]
    else:
        cbs = [DANNCallback(lambda_max=lambda_max, gamma=grl_gamma,
                            domain_label_smoothing=domain_label_smoothing)]
    if model_selection:
        cbs += [
            SaveModelCallback(
                monitor="win_f1_macro",          # save best by PRIMARY metric
                comp=np.greater,
                min_delta=1e-4,
                fname=model_name,
                reset_on_fit=True,
            ),
            EarlyStoppingCallback(
                monitor="valid_loss",
                patience=patience,
                min_delta=1e-4,
            ),
        ]

    return Learner(
        dls=dls,
        model=model,
        loss_func=fault_loss,
        metrics=[accuracy, win_f1_macro, roc_auc],
        path=save_dir,
        model_dir="",   # save directly in save_dir, not a sub-folder
        cbs=cbs,
        wd=wd,
    )


def build_cdan_learner(
    dls,
    n_fault_classes: int,
    n_domains: int,
    lambda_max: float,
    model_name: str,
    save_dir,
    wd: float = 1e-5,
    patience: int = 10,
    model_selection: bool = True,
    backbone: str = "tsitplus",
    grl_gamma: float = 10.0,
    use_entropy: bool = True,
) -> Learner:
    """CDAN learner: conditional adversarial alignment. Domain head melihat peta
    multilinear fitur x prediksi kelas (bukan fitur mentah), sehingga penyelarasan
    terjadi per-kelas dan tidak menggosok sinyal fault yang termodulasi beban."""
    model = build_cdan_model(
        backbone, c_in=dls.vars, seq_len=dls.len,
        n_classes=n_fault_classes, n_domains=n_domains, lambda_grl=0.0,
    ).to(dls.device)

    train_domain_labels = torch.tensor(
        [int(y[1]) for y in dls.train_ds.tls[-1] if int(y[1]) >= 0], dtype=torch.long,
    )
    if len(train_domain_labels) > 0:
        model.verify_domain_labels(train_domain_labels)
    model.sanity_check(c_in=dls.vars, seq_len=dls.len)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: CDANModel | backbone={backbone} | params={n_params:,} | "
          f"c_in={dls.vars} | c_out={n_fault_classes} | n_domains={n_domains} | "
          f"lambda_max={lambda_max} entropy_conditioning={use_entropy}")

    def fault_loss(pred, targ):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        return F.cross_entropy(pred, targ.long())

    roc_auc = AccumMetric(roc_auc_metric, flatten=False)
    win_f1_macro = AccumMetric(win_f1_macro_metric, flatten=False)
    win_f1_macro.name = "win_f1_macro"
    roc_auc.name = "roc_auc"
    save_dir.mkdir(parents=True, exist_ok=True)

    cbs = [CDANCallback(lambda_max=lambda_max, gamma=grl_gamma, use_entropy=use_entropy)]
    if model_selection:
        cbs += [
            SaveModelCallback(monitor="win_f1_macro", comp=np.greater, min_delta=1e-4,
                              fname=model_name, reset_on_fit=True),
            EarlyStoppingCallback(monitor="valid_loss", patience=patience, min_delta=1e-4),
        ]

    return Learner(
        dls=dls, model=model, loss_func=fault_loss,
        metrics=[accuracy, win_f1_macro, roc_auc],
        path=save_dir, model_dir="", cbs=cbs, wd=wd,
    )




def build_cdan_supcon_learner(
    dls,
    n_fault_classes: int,
    n_domains: int,
    lambda_max: float,
    model_name: str,
    save_dir,
    wd: float = 1e-5,
    patience: int = 10,
    model_selection: bool = True,
    backbone: str = "tsitplus",
    grl_gamma: float = 10.0,
    use_entropy: bool = True,
    lambda_sc: float = 1.0,
    temperature: float = 0.07,
    proj_dim: int = 128,
) -> Learner:
    """CDAN + SupCon learner: penyelarasan adversarial KONDISIONAL (pengganti GRL
    marginal pada DANNSupCon) plus kepala kontrastif yang menarik same-fault lintas beban."""
    model = build_cdan_supcon_model(
        backbone, c_in=dls.vars, seq_len=dls.len,
        n_classes=n_fault_classes, n_domains=n_domains, lambda_grl=0.0, proj_dim=proj_dim,
    ).to(dls.device)

    train_domain_labels = torch.tensor(
        [int(y[1]) for y in dls.train_ds.tls[-1] if int(y[1]) >= 0], dtype=torch.long,
    )
    if len(train_domain_labels) > 0:
        model.verify_domain_labels(train_domain_labels)
    model.sanity_check(c_in=dls.vars, seq_len=dls.len)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: CDANSupConModel | backbone={backbone} | params={n_params:,} | "
          f"c_in={dls.vars} | c_out={n_fault_classes} | n_domains={n_domains} | "
          f"lambda_max={lambda_max} entropy_conditioning={use_entropy} "
          f"lambda_sc={lambda_sc} temperature={temperature}")

    def fault_loss(pred, targ):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        return F.cross_entropy(pred, targ.long())

    roc_auc = AccumMetric(roc_auc_metric, flatten=False)
    win_f1_macro = AccumMetric(win_f1_macro_metric, flatten=False)
    win_f1_macro.name = "win_f1_macro"
    roc_auc.name = "roc_auc"
    save_dir.mkdir(parents=True, exist_ok=True)

    cbs = [CDANSupConCallback(lambda_max=lambda_max, gamma=grl_gamma, use_entropy=use_entropy,
                              lambda_sc=lambda_sc, temperature=temperature)]
    if model_selection:
        cbs += [
            SaveModelCallback(monitor="win_f1_macro", comp=np.greater, min_delta=1e-4,
                              fname=model_name, reset_on_fit=True),
            EarlyStoppingCallback(monitor="valid_loss", patience=patience, min_delta=1e-4),
        ]

    return Learner(
        dls=dls, model=model, loss_func=fault_loss,
        metrics=[accuracy, win_f1_macro, roc_auc],
        path=save_dir, model_dir="", cbs=cbs, wd=wd,
    )


def build_dsn_learner(
    dls,
    n_fault_classes: int,
    n_domains: int,
    model_name: str,
    save_dir,
    wd: float = 1e-5,
    patience: int = 10,
    model_selection: bool = True,
    backbone: str = "tsitplus",
    lambda_domain: float = 1.0,
    lambda_ortho: float = 1.0,
    lambda_recon: float = 1.0,
) -> Learner:
    """Orthogonal Feature Separation (+reconstruction, NO adversarial) learner:
    disentangle load (z_dom) from fault (z_task) via domain head + orthogonality +
    feature reconstruction, instead of adversarial erasure (GRL)."""
    model = build_dsn_model(
        backbone, c_in=dls.vars, seq_len=dls.len,
        n_classes=n_fault_classes, n_domains=n_domains,
    ).to(dls.device)

    train_domain_labels = torch.tensor(
        [int(y[1]) for y in dls.train_ds.tls[-1] if int(y[1]) >= 0], dtype=torch.long,
    )
    if len(train_domain_labels) > 0:
        model.verify_domain_labels(train_domain_labels)
    model.sanity_check(c_in=dls.vars, seq_len=dls.len)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: DSNModel(OFS+recon) | backbone={backbone} | params={n_params:,} | "
          f"c_in={dls.vars} | c_out={n_fault_classes} | n_domains={n_domains} | "
          f"lambda_domain={lambda_domain} lambda_ortho={lambda_ortho} lambda_recon={lambda_recon}")

    def fault_loss(pred, targ):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        return F.cross_entropy(pred, targ.long())

    roc_auc = AccumMetric(roc_auc_metric, flatten=False)
    win_f1_macro = AccumMetric(win_f1_macro_metric, flatten=False)
    win_f1_macro.name = "win_f1_macro"
    roc_auc.name = "roc_auc"
    save_dir.mkdir(parents=True, exist_ok=True)

    cbs = [DSNCallback(lambda_domain=lambda_domain, lambda_ortho=lambda_ortho,
                       lambda_recon=lambda_recon)]
    if model_selection:
        cbs += [
            SaveModelCallback(monitor="win_f1_macro", comp=np.greater, min_delta=1e-4,
                              fname=model_name, reset_on_fit=True),
            EarlyStoppingCallback(monitor="valid_loss", patience=patience, min_delta=1e-4),
        ]

    return Learner(
        dls=dls, model=model, loss_func=fault_loss,
        metrics=[accuracy, win_f1_macro, roc_auc],
        path=save_dir, model_dir="", cbs=cbs, wd=wd,
    )


def build_dsran_learner(
    dls,
    n_fault_classes: int,
    n_domains: int,
    lambda_max: float,
    model_name: str,
    save_dir,
    wd: float = 1e-5,
    patience: int = 10,
    model_selection: bool = True,
    backbone: str = "tsitplus",
    grl_gamma: float = 10.0,
    lambda_domain: float = 1.0,
    lambda_ortho: float = 1.0,
    lambda_recon: float = 1.0,
) -> Learner:
    """DSRAN-inspired learner: separation (z_task/z_dom + ortho) + reconstruction PLUS
    adversarial GRL on z_task. z_dom absorbs load (no GRL); z_task pushed invariant
    adversarially. NOTE: adversarial empirically hurts our data — kept for completeness."""
    model = build_dsran_model(
        backbone, c_in=dls.vars, seq_len=dls.len,
        n_classes=n_fault_classes, n_domains=n_domains, lambda_grl=0.0,
    ).to(dls.device)

    train_domain_labels = torch.tensor(
        [int(y[1]) for y in dls.train_ds.tls[-1] if int(y[1]) >= 0], dtype=torch.long,
    )
    if len(train_domain_labels) > 0:
        model.verify_domain_labels(train_domain_labels)
    model.sanity_check(c_in=dls.vars, seq_len=dls.len)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: DSRANModel(sep+recon+adv) | backbone={backbone} | params={n_params:,} | "
          f"c_in={dls.vars} | c_out={n_fault_classes} | n_domains={n_domains} | "
          f"lambda_max={lambda_max} lambda_domain={lambda_domain} lambda_ortho={lambda_ortho} "
          f"lambda_recon={lambda_recon}")

    def fault_loss(pred, targ):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        return F.cross_entropy(pred, targ.long())

    roc_auc = AccumMetric(roc_auc_metric, flatten=False)
    win_f1_macro = AccumMetric(win_f1_macro_metric, flatten=False)
    win_f1_macro.name = "win_f1_macro"
    roc_auc.name = "roc_auc"
    save_dir.mkdir(parents=True, exist_ok=True)

    cbs = [DSRANCallback(lambda_max=lambda_max, gamma=grl_gamma,
                         lambda_domain=lambda_domain, lambda_ortho=lambda_ortho,
                         lambda_recon=lambda_recon)]
    if model_selection:
        cbs += [
            SaveModelCallback(monitor="win_f1_macro", comp=np.greater, min_delta=1e-4,
                              fname=model_name, reset_on_fit=True),
            EarlyStoppingCallback(monitor="valid_loss", patience=patience, min_delta=1e-4),
        ]

    return Learner(
        dls=dls, model=model, loss_func=fault_loss,
        metrics=[accuracy, win_f1_macro, roc_auc],
        path=save_dir, model_dir="", cbs=cbs, wd=wd,
    )


def build_supcon_learner(
    dls,
    n_fault_classes: int,
    n_domains: int,
    model_name: str,
    save_dir,
    wd: float = 1e-5,
    patience: int = 10,
    model_selection: bool = True,
    backbone: str = "tsitplus",
    lambda_sc: float = 1.0,
    temperature: float = 0.07,
    proj_dim: int = 128,
) -> Learner:
    """Supervised Contrastive learner: shape the metric space by fault identity
    (pull same-fault across loads together) instead of erasing domain (no GRL).
    n_domains accepted for signature symmetry but unused (SupCon needs no domain labels)."""
    model = build_supcon_model(
        backbone, c_in=dls.vars, seq_len=dls.len,
        n_classes=n_fault_classes, proj_dim=proj_dim,
    ).to(dls.device)
    model.sanity_check(c_in=dls.vars, seq_len=dls.len)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: SupConModel | backbone={backbone} | params={n_params:,} | "
          f"c_in={dls.vars} | c_out={n_fault_classes} | "
          f"lambda_sc={lambda_sc} temperature={temperature} proj_dim={proj_dim}")

    def fault_loss(pred, targ):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        return F.cross_entropy(pred, targ.long())

    roc_auc = AccumMetric(roc_auc_metric, flatten=False)
    win_f1_macro = AccumMetric(win_f1_macro_metric, flatten=False)
    win_f1_macro.name = "win_f1_macro"
    roc_auc.name = "roc_auc"
    save_dir.mkdir(parents=True, exist_ok=True)

    cbs = [SupConCallback(lambda_sc=lambda_sc, temperature=temperature)]
    if model_selection:
        cbs += [
            SaveModelCallback(monitor="win_f1_macro", comp=np.greater, min_delta=1e-4,
                              fname=model_name, reset_on_fit=True),
            EarlyStoppingCallback(monitor="valid_loss", patience=patience, min_delta=1e-4),
        ]

    return Learner(
        dls=dls, model=model, loss_func=fault_loss,
        metrics=[accuracy, win_f1_macro, roc_auc],
        path=save_dir, model_dir="", cbs=cbs, wd=wd,
    )


def build_dann_supcon_learner(
    dls,
    n_fault_classes: int,
    n_domains: int,
    lambda_max: float,
    model_name: str,
    save_dir,
    wd: float = 1e-5,
    patience: int = 10,
    model_selection: bool = True,
    backbone: str = "tsitplus",
    grl_gamma: float = 10.0,
    lambda_sc: float = 1.0,
    temperature: float = 0.07,
    proj_dim: int = 128,
    bt_target_ratio: float | None = None,
    bt_anneal: bool = True,
    domain_u=None,
) -> Learner:
    """DANN + SupCon learner: GRL adversarial domain head (DANN core) PLUS a supervised-contrastive projection head that pulls same-fault
    samples together across loads to stabilize the adversarial training."""
    model = build_dann_supcon_model(
        backbone, c_in=dls.vars, seq_len=dls.len,
        n_classes=n_fault_classes, n_domains=n_domains, lambda_grl=0.0, proj_dim=proj_dim,
        domain_regression=domain_u is not None,   # UJI A: kepala 128->64->1, loss MSE
    ).to(dls.device)

    train_domain_labels = torch.tensor(
        [int(y[1]) for y in dls.train_ds.tls[-1] if int(y[1]) >= 0], dtype=torch.long,
    )
    if len(train_domain_labels) > 0:
        model.verify_domain_labels(train_domain_labels)
    model.sanity_check(c_in=dls.vars, seq_len=dls.len)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: DANNSupConModel | backbone={backbone} | params={n_params:,} | "
          f"c_in={dls.vars} | c_out={n_fault_classes} | n_domains={n_domains} | "
          f"lambda_max={lambda_max} lambda_sc={lambda_sc} temperature={temperature}")

    def fault_loss(pred, targ):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        return F.cross_entropy(pred, targ.long())

    roc_auc = AccumMetric(roc_auc_metric, flatten=False)
    win_f1_macro = AccumMetric(win_f1_macro_metric, flatten=False)
    win_f1_macro.name = "win_f1_macro"
    roc_auc.name = "roc_auc"
    save_dir.mkdir(parents=True, exist_ok=True)

    # BT-DANN — lihat catatan di build_dann_learner. Penyeimbangnya di sini adalah
    # CE_fault + lambda_sc*SupCon, bukan CE saja.
    if bt_target_ratio is not None:
        cbs = [BTDANNSupConCallback(target_ratio=bt_target_ratio, gamma=grl_gamma,
                                    lambda_sc=lambda_sc, temperature=temperature,
                                    anneal=bt_anneal, domain_u=domain_u)]
    else:
        cbs = [DANNSupConCallback(lambda_max=lambda_max, gamma=grl_gamma,
                                  lambda_sc=lambda_sc, temperature=temperature,
                                  domain_u=domain_u)]
    if model_selection:
        cbs += [
            SaveModelCallback(monitor="win_f1_macro", comp=np.greater, min_delta=1e-4,
                              fname=model_name, reset_on_fit=True),
            EarlyStoppingCallback(monitor="valid_loss", patience=patience, min_delta=1e-4),
        ]

    return Learner(
        dls=dls, model=model, loss_func=fault_loss,
        metrics=[accuracy, win_f1_macro, roc_auc],
        path=save_dir, model_dir="", cbs=cbs, wd=wd,
    )
