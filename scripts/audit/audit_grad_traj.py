"""audit_grad_traj.py — Lintasan rasio gradien adversarial/klasifikasi SELAMA pelatihan.

Pengukuran statis pada checkpoint hanya memberi titik akhir. Yang menentukan adalah
bagaimana keseimbangan kedua objektif BERGESER sepanjang pelatihan:

    rasio(t) = ||d L_domain / d theta_backbone||  /  ||d L_fault / d theta_backbone||

Hipotesis dari pengukuran statis: rasio ~0.6 di awal (seimbang, sesuai maksud Ganin)
lalu meledak ke ~10 di akhir, karena L_fault jatuh ke ~0 (data latih dihafal) sementara
lambda justru DINAIKKAN oleh jadwal anneal. Kalau benar, backbone di paruh akhir
pelatihan digerakkan hampir sepenuhnya oleh objektif adversarial.

Protokol identik battery 8-seed: tsitplus, EXT, fp32, no-augment, fixed-budget, 50 epoch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastai.callback.core import Callback
from tsai.basics import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, PROCESSED_DIR
from src.preprocessing.io import load_dataset
from src.training.dann_factory import build_dann_dataloaders, build_dann_learner
from src.training.splits import prepare_fold_domains
from src.training.recordings import create_recording_ids
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
EPOCHS = 50
LR = 1e-3
LAMBDA_MAX = 1.0
EVERY = 30                      # ukur tiap N batch latih
OUT = Path(__file__).parent / "grad_traj.json"


class GradRatioProbe(Callback):
    """order=11 -> before_batch berjalan SEBELUM DANNCallback(order=12) mengganti yb."""
    order = 11

    def __init__(self, every: int = 30) -> None:
        self.every = every
        self.n = 0
        self.rows: list[dict] = []
        self._raw_y = None

    def before_batch(self) -> None:
        if self.training and self.learn.yb:
            y = self.learn.yb[0]
            self._raw_y = y.detach().clone() if y.dim() == 2 and y.shape[1] == 2 else None

    def after_batch(self) -> None:
        if not self.training or self._raw_y is None:
            return
        self.n += 1
        if self.n % self.every:
            return
        xb = self.learn.xb[0].detach()
        yf = self._raw_y[:, 0].long()
        yd = self._raw_y[:, 1].long()
        model = self.learn.model
        bb = [p for p in model.backbone.parameters() if p.requires_grad]

        fl, dl_ = model(xb)
        lf = F.cross_entropy(fl, yf)
        ld = F.cross_entropy(dl_, yd)
        gf = torch.autograd.grad(lf, bb, retain_graph=True, allow_unused=True)
        gd = torch.autograd.grad(ld, bb, retain_graph=False, allow_unused=True)

        def nrm(gs):
            return float(torch.sqrt(sum((g ** 2).sum() for g in gs if g is not None)).item())

        nf, nd = nrm(gf), nrm(gd)
        with torch.no_grad():
            dacc = float((dl_.argmax(1) == yd).float().mean().item())
        self.rows.append(dict(
            epoch=int(getattr(self.learn, "epoch", -1)),
            pct=float(getattr(self.learn, "pct_train", 0.0) or 0.0),
            lam=float(getattr(model, "lambda_grl", float("nan"))),
            g_fault=nf, g_dom=nd, ratio=nd / max(nf, 1e-12),
            L_fault=float(lf.item()), L_dom=float(ld.item()), dacc=dacc,
        ))


def main() -> None:
    motor = PROCESSED_DIR / "motor"
    ds = motor / "motor_multiclass_w4096_s4096_ss_env.npz"
    X, y_fault, _ = load_dataset(ds)
    si = np.load(motor / f"{ds.stem}_split_info.npz")
    torque, reps = si["torque"], si["repetition"]
    rec_ids = create_recording_ids(y_fault, torque, reps)

    uniq = sorted(set(int(t) for t in torque.tolist()))
    test_mask = np.isin(torque, [uniq[0], uniq[-1]])
    train_idx, test_idx = np.where(~test_mask)[0], np.where(test_mask)[0]
    y_domain, n_domains, _ = prepare_fold_domains(torque, train_idx, test_idx)
    n_classes = int(np.unique(y_fault).size)

    set_seed(SEED)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    itr, iva = next(sgkf.split(train_idx, y_fault[train_idx], groups=rec_ids[train_idx]))
    inner_tr, inner_va = train_idx[itr], train_idx[iva]

    dls = build_dann_dataloaders(
        X=X, y_fault=y_fault, y_domain=y_domain,
        splits=(list(inner_tr), list(inner_va)),
        batch_size=64, augment=False, seed=SEED,
    )
    learn = build_dann_learner(
        dls=dls, n_fault_classes=n_classes, n_domains=n_domains,
        lambda_max=LAMBDA_MAX, model_name=f"GRADTRAJ_seed{SEED}",
        save_dir=MODELS_DIR / "motor" / "dann_lto", backbone="tsitplus",
        model_selection=False,           # fixed_budget, sama dengan battery
    )
    probe = GradRatioProbe(every=EVERY)
    learn.add_cb(probe)
    learn.fit_one_cycle(EPOCHS, lr_max=LR)

    OUT.write_text(json.dumps(probe.rows), encoding="utf-8")
    print(f"\ntersimpan {len(probe.rows)} pengukuran -> {OUT}")

    # ringkasan per-epoch
    import collections
    by = collections.defaultdict(list)
    for r in probe.rows:
        by[r["epoch"]].append(r)
    print(f"\n{'epoch':>6} {'lambda':>7} {'L_fault':>9} {'||g_fault||':>12} "
          f"{'||g_dom||':>11} {'RASIO':>9} {'dom_acc':>8}")
    for ep in sorted(by):
        if ep % 5 and ep != max(by):
            continue
        rs = by[ep]
        print(f"{ep:>6} {np.mean([r['lam'] for r in rs]):>7.3f} "
              f"{np.mean([r['L_fault'] for r in rs]):>9.5f} "
              f"{np.mean([r['g_fault'] for r in rs]):>12.5f} "
              f"{np.mean([r['g_dom'] for r in rs]):>11.5f} "
              f"{np.mean([r['ratio'] for r in rs]):>9.3f} "
              f"{np.mean([r['dacc'] for r in rs]):>8.3f}")


if __name__ == "__main__":
    main()
