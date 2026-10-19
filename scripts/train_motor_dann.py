"""
train_motor_dann.py - Load-Robust Motor Fault Classification via Domain-Adversarial DG.

Strategy: Domain Generalization via Inter-Source Adversarial Alignment
-----------------------------------------------------------------------
The GRL forces the backbone to produce representations that are invariant
across all seen torque levels. By induction, these features generalise to
the unseen held-out torque (Leave-Torque-Out fold). This is Domain Generalization
(DG), NOT Domain Adaptation — the validation torque is NEVER seen during training.

Compared baseline: same architecture without GRL (lambda_max=0)
                   = plain TSiTPlus with identical data/splits.
The baseline directly quantifies GRL's contribution to load robustness.

How to run
----------
    # Full DANN-DG (λ_max=1.0, recommended)
    python scripts/train_motor_dann.py --name motor_dann_lto_v1

    # Ablation: no adversarial signal (baseline)
    python scripts/train_motor_dann.py --lambda_max 0.0 --name motor_dann_baseline_v1

    # Weaker adversarial pressure (useful if fault accuracy degrades)
    python scripts/train_motor_dann.py --lambda_max 0.5 --name motor_dann_half_v1

Metrics
-------
  Window-level Accuracy : inflated (many windows per recording), secondary.
  Recording-level Accuracy: majority vote over all windows from the same
                             physical recording, UNSEEN TORQUES. This is the
                             paper's secondary metric (Sec. III-D); the primary
                             metric is window-level macro F1.

GRL Convergence Verification
------------------------------
Monitor [GRL-EPOCH] lines in training output. Healthy signs:
  - domain_loss approaches log(n_domains) = theoretical max entropy
  - domain_acc  approaches 1/n_domains   = random chance
  - STATUS: CONVERGING
If STATUS shows WARN (domain classifier still discriminative), the backbone
is not yet load-invariant — try increasing lambda_max or training epochs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from fastai.learner import Learner
from sklearn.model_selection import StratifiedGroupKFold
from tsai.basics import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, PROCESSED_DIR, SEED, TRAIN_CONFIG
from src.preprocessing.io import load_dataset
from src.training.dann_factory import (
    build_dann_dataloaders, build_dann_learner, build_dsn_learner, build_dsran_learner,
    build_supcon_learner, build_dann_supcon_learner,
    build_cdan_learner, build_cdan_supcon_learner,
)
from src.training.splits import prepare_fold_domains, verify_no_leakage
from src.training.domain_index import build_domain_u, ringkas_u
from src.training.protokol import bangun_fold
from src.training.cross_seed import run_cross_seed, print_cross_seed_summary
from src.evaluation.diagnostics import generalization_gap
from src.evaluation.metrics import recording_level_metrics, window_level_metrics
from src.training.recordings import create_recording_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inner_train_valid_split(
    train_idx: np.ndarray,
    recording_ids: np.ndarray,
    y_fault: np.ndarray,
    seed: int,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carve an inner validation set out of the TRAINING torques for model
    selection (early stopping, checkpointing). The held-out test torque is
    NEVER used for this — it stays fully unseen until final evaluation.

    The split is grouped by physical recording (so 50%-overlap windows of the
    same recording never straddle the inner train/valid boundary) and
    stratified by fault class (so every class is represented in the inner
    validation set). Returns absolute indices into the global arrays.
    """
    rec = recording_ids[train_idx]
    y = y_fault[train_idx]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    inner_tr_rel, inner_va_rel = next(sgkf.split(train_idx, y, groups=rec))
    return train_idx[inner_tr_rel], train_idx[inner_va_rel]


# ---------------------------------------------------------------------------
# Recording-level majority-vote evaluation
# ---------------------------------------------------------------------------


def evaluate_on_test(
    learn: Learner,
    X: np.ndarray,
    y_fault: np.ndarray,
    recording_ids: np.ndarray,
    test_idx: np.ndarray,
    n_fault_classes: int = 5,
    batch_size: int = 64,
    seed: int = SEED,
    by_var_standardize: bool = False,
) -> dict:
    """
    Evaluate on the held-out (UNSEEN) torque — never used for training or model
    selection. Builds a labelled eval dataloader with the held-out torque as the
    valid split (same batch transforms, augmentation OFF) and runs the trained
    model on it. Domain labels are unused at inference (the model returns only
    fault logits in eval mode), so a dummy domain is supplied.

    Recording-level prediction = majority vote (mean softmax) over all windows
    that belong to the same physical recording — the paper's secondary metric (Sec. III-D).
    """
    # Reuse the project DataLoader factory so the held-out windows get the exact
    # same per-sample standardization as training (test_dl() is incompatible with
    # the tfms=[None, None] setup, so we build an explicit valid split instead).
    y_domain_dummy = np.zeros_like(y_fault, dtype=np.int64)
    dummy_train = list(test_idx[: max(1, min(8, len(test_idx)))])
    eval_dls = build_dann_dataloaders(
        X=X,
        y_fault=y_fault,
        y_domain=y_domain_dummy,
        splits=(dummy_train, list(test_idx)),
        batch_size=batch_size,
        augment=False,
        seed=seed,
        by_var_standardize=by_var_standardize,   # MUST match training, else train/test mismatch
    )
    try:
        eval_dls.valid.fake_l.num_workers = 0
    except AttributeError:
        pass

    preds_raw, _ = learn.get_preds(dl=eval_dls.valid)
    probs = torch.softmax(preds_raw.float(), dim=-1).numpy()

    y_true = y_fault[test_idx].astype(int)

    win = window_level_metrics(probs, y_true, n_fault_classes)
    rec = recording_level_metrics(probs, y_true, recording_ids[test_idx], n_fault_classes)
    return {
        "win_accuracy": win["win_accuracy"],
        "win_f1_macro": win["win_f1_macro"],
        "rec_accuracy": rec["rec_accuracy"],
        "rec_f1_macro": rec["rec_f1_macro"],
        "n_recordings": rec["n_recordings"],
        "n_windows": win["n_windows"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Robust against Windows cp1252 console when logs contain λ, ρ, ±, etc.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="DANN Motor Fault Classifier — Load-Robust Training"
    )
    parser.add_argument("--epochs",      type=int,   default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch_size",  type=int,   default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr",          type=float, default=TRAIN_CONFIG["learning_rate"])
    parser.add_argument("--name",        type=str,   default="motor_dann_lto_v1")
    parser.add_argument(
        "--lambda_max", type=float, default=1.0,
        help=(
            "Maximum GRL reversal strength (0 = no adversarial = baseline (no GRL, chosen backbone), "
            "1.0 = full DANN recommended)."
        ),
    )
    parser.add_argument("--n_folds",     type=int,   default=8,
                        help="LTO-CV folds. Default 8 = leave one torque out per fold "
                             "(matches the 8-fold LTO-CV table in bab3).")
    parser.add_argument(
        "--no_augment", action="store_true",
        help="Disable time-series augmentation (useful for ablation studies).",
    )
    parser.add_argument(
        "--seeds", type=str, default="42,123,2025",
        help="Comma-separated random seeds. The full LTO-CV runs once per seed; "
             "results are reported as mean +/- std across seeds to control "
             "adversarial-training variance.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip folds whose checkpoint (.pth) already exists. Useful to resume interrupted runs.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader worker processes for the TRAINING dls only (eval dl always forces 0). "
             "Default 0 matches all prior runs. >0 is unvalidated with --fp16 on Windows "
             "(CUDA-storage pickling crash risk) — only use with --fp32 until validated.",
    )
    parser.add_argument(
        "--pin_memory", action="store_true",
        help="Pin host memory for faster H2D transfer. Only useful with --num_workers > 0.",
    )
    parser.add_argument(
        "--backbone", type=str, default="tsitplus",
        choices=["tsitplus", "tssequencerplus"],
        help="Feature-extractor backbone inside the DANN frame.",
    )
    parser.add_argument(
        "--npz", type=str, default=None,
        help="Explicit dataset .npz filename under data/processed/motor/ "
             "(overrides auto-selection; used for the envelope ablation).",
    )
    parser.add_argument(
        "--max_folds", type=int, default=0,
        help="If >0, run only the first N LTO folds (quick ablation; 1 = single fold).",
    )
    parser.add_argument(
        "--fold_idx", type=int, default=-1,
        help="If >=0, run ONLY this single LTO fold index (overrides --max_folds for "
             "fold selection). Use to pick an INTERIOR held-out torque for a clean "
             "interpolation probe — e.g. fold 3 holds out torque 4 (surrounded by train "
             "domains on both sides), unlike fold 0 which holds out the extreme torque 7.",
    )
    parser.add_argument(
        "--extremes_out", action="store_true",
        help="Mature DG protocol: train on MIDDLE torque levels, test on EXTREME loads "
             "(lowest+highest). Tests load EXTRAPOLATION (harder than LTO interpolation). "
             "Produces a single custom fold instead of GroupKFold LTO.",
    )
    parser.add_argument(
        "--n_extremes", type=int, default=1,
        help="With --extremes_out: how many torque levels to hold out at EACH end "
             "(1 = {lowest, highest}; 2 = {2 lowest, 2 highest}).",
    )
    parser.add_argument(
        "--by_var", action="store_true",
        help="Per-sample AND per-channel standardization: TSStandardize(by_sample=True, "
             "by_var=True). Default (off) shares one mean/std across all six channels, which "
             "suppresses Vib_axial to 3.9%% and the Hilbert envelope to 5.7%% of the dominant "
             "channel (channel std spans 25.8x — see Sec. IV-C of the paper). Distinct from "
             "--global_standardize, which also drops per-sample normalization.",
    )
    parser.add_argument(
        "--bt_ratio", type=float, default=None,
        help="BT-DANN (Sec. IV-A dan IV-D makalah): normalisasi gradien adversarial. Bila diberi "
             "nilai (mis. 0.3), lambda tidak lagi dijadwalkan menurut waktu melainkan "
             "dihitung tiap batch agar ||g_dom||/||g_fault|| tetap di rasio ini. Tanpa "
             "normalisasi rasio meledak 0.48 -> 5.06 dan 45/50 epoch dikuasai objektif "
             "adversarial (Sec. IV-D makalah). Default None = perilaku DANN lama persis. Berlaku untuk "
             "--method dann dan dannsupcon.",
    )
    parser.add_argument(
        "--bt_no_anneal", action="store_true",
        help="Dengan --bt_ratio: matikan anneal Ganin sehingga rasio target berlaku penuh "
             "sejak batch pertama. Default anneal tetap aktif (naik 0 -> rasio target).",
    )
    parser.add_argument(
        "--domain_index", type=str, default="none",
        choices=["none", "torque", "slip", "f2s"],
        help="Ganti kepala domain KATEGORIS (feat->64->n_domains, CE) dengan kepala REGRESI "
             "(feat->64->1, MSE) yang menargetkan indeks domain skalar u, dinormalisasi "
             "[0,1] dari statistik torsi LATIH saja. 'slip' dan 'f2s' menghasilkan target "
             "IDENTIK setelah normalisasi (2sf = 120*slip, penskalaan murni). Default "
             "'none' = perilaku lama persis. Berlaku untuk --method dannsupcon (dengan "
             "atau tanpa --bt_ratio).",
    )
    parser.add_argument(
        "--global_standardize", action="store_true",
        help="ABLATION ONLY: use global (by_var) standardization instead of per-sample "
             "z-norm, preserving inter-window amplitude (the dominant torque/domain cue). "
             "Use to test whether DANN beats baseline when domain shift is NOT removed.",
    )
    parser.add_argument(
        "--grl_gamma", type=float, default=10.0,
        help="GRL lambda-annealing speed in lambda(p)=lambda_max*(2/(1+exp(-gamma*p))-1). "
             "Default 10.0 (Ganin 2016 standard). Lower (e.g. 2-3) ramps slower, giving the "
             "domain head room to learn before adversarial suppression — useful when the "
             "fault task saturates within 1-2 epochs.",
    )
    parser.add_argument(
        "--domain_label_smoothing", type=float, default=0.0,
        help="Environment Label Smoothing (ELS, Zhang et al. ICLR 2023). Smooth the domain "
             "labels in the discriminator CE (e.g. 0.1) to reduce discriminator over-confidence "
             "→ more stable, lower-variance adversarial training. 0.0 = off (vanilla DANN).",
    )
    parser.add_argument(
        "--fixed_budget", action="store_true",
        help="Disable model selection (SaveModelCallback best-by-F1 + EarlyStopping) and "
             "evaluate the FINAL model after the full epoch budget. Needed "
             "for a FAIR DANN test: otherwise the best-fault-F1 checkpoint is saved at an "
             "early epoch where lambda is still ~0, so DANN == baseline by construction.",
    )
    parser.add_argument(
        "--method", type=str, default="dann",
        choices=["dann", "dsn", "dsran", "supcon", "dannsupcon", "cdan", "cdansupcon"],
        help="Which DG method: 'dann' (adversarial GRL, default), 'dsn' (Domain Separation "
             "Network — disentangle load vs fault via orthogonality, no GRL), 'dsran' "
             "(DSRAN-combo — separation + adversarial GRL on z_task), 'supcon' "
             "(Supervised Contrastive — pull same-fault across loads together, no GRL), "
             "'dannsupcon' (DANN GRL core + SupCon stabilizer), 'cdan' (Conditional "
             "adversarial — domain head sees multilinear feat x class-prediction, so "
             "alignment is per-class instead of marginal), or 'cdansupcon' (CDAN + SupCon).",
    )
    parser.add_argument("--cdan_no_entropy", action="store_true",
                        help="CDAN/CDAN-SupCon: disable entropy conditioning (CDAN+E -> plain "
                             "CDAN). Default is entropy conditioning ON (the paper's best variant).")
    parser.add_argument("--dsn_lambda_domain", type=float, default=1.0,
                        help="DSN/DSRAN: weight of domain CE on z_dom.")
    parser.add_argument("--dsn_lambda_ortho", type=float, default=1.0,
                        help="DSN/DSRAN: weight of orthogonality loss between z_task and z_dom.")
    parser.add_argument("--dsn_lambda_recon", type=float, default=1.0,
                        help="DSN/DSRAN: weight of feature-reconstruction loss (private+task retain feat).")
    parser.add_argument("--supcon_lambda", type=float, default=1.0,
                        help="SupCon: weight of supervised contrastive loss added to CE.")
    parser.add_argument("--supcon_temp", type=float, default=0.07,
                        help="SupCon: temperature for the contrastive softmax.")
    parser.add_argument("--supcon_proj_dim", type=int, default=128,
                        help="SupCon: projection-head output (embedding) dimension.")
    parser.add_argument(
        "--fp32", action="store_true",
        help="Train in full fp32 instead of fp16 mixed precision. Transformers (attention/"
             "softmax) + adversarial GRL are numerically sensitive in fp16, which can amplify "
             "seed-to-seed variance/instability. Use for TSiTPlus to test/remove that artifact.",
    )
    args = parser.parse_args()

    # --domain_index hanya terpasang di jalur dannsupcon. Gagal keras daripada diam-diam
    # mengabaikan flag.
    if args.domain_index != "none" and args.method != "dannsupcon":
        parser.error(
            f"--domain_index {args.domain_index} hanya terpasang untuk "
            f"--method dannsupcon, bukan '{args.method}'."
        )

    # ── Load dataset ──────────────────────────────────────────────────────
    motor_dir = PROCESSED_DIR / "motor"
    # Motor uses MOTOR_STRIDE=4096 (0% overlap) — see config. Prefer slip-band envelope
    # (c_in=6); fallback to steady-state only (c_in=5), then to full recording (c_in=5).
    _candidates = [
        motor_dir / "motor_multiclass_w4096_s4096_ss_env.npz",  # preferred: envelope
        motor_dir / "motor_multiclass_w4096_s4096_ss.npz",      # fallback: no envelope
        motor_dir / "motor_multiclass_w4096_s4096_ss.npz",      # (kept for parity)
    ]
    if args.npz:
        dataset_path = motor_dir / args.npz
        if not dataset_path.exists():
            print(f"[ERROR] --npz file not found: {dataset_path}")
            return
    else:
        dataset_path = next((p for p in _candidates if p.exists()), None)
    if dataset_path is None:
        print(f"[ERROR] No motor dataset found. Expected one of:")
        for p in _candidates:
            print(f"  {p}")
        print("Run: python scripts/run_preprocessing.py --motor")
        return
    if "_env" not in dataset_path.name:
        print(f"[WARN] Loading dataset WITHOUT slip-band envelope: {dataset_path.name}")
        print("       Re-run preprocessing to include envelope channel (c_in=6).")
    else:
        print(f"[OK] Loading dataset with slip-band envelope: {dataset_path.name}")
    split_info_path = motor_dir / f"{dataset_path.stem}_split_info.npz"

    print(f"\n{'='*60}")
    print(f"  DANN TRAINING: {args.name}")
    print(f"  lambda_max     = {args.lambda_max}")
    print(f"  seeds          = {args.seeds}")
    print(f"  epochs         = {args.epochs}")
    print(f"  batch_size     = {args.batch_size}")
    print(f"  lr             = {args.lr}")
    print(f"  n_folds        = {args.n_folds}")
    print(f"{'='*60}")

    X, y_fault, _ = load_dataset(dataset_path)
    # Lever 1 (RAM): hold the feature array in float16. fp16 rounding (~0.05%) is far
    # below the deliberate augmentation magnitudes (TSTimeNoise σ=0.01, TSMagScale ±20%)
    # and matches the fp16 training precision (learn.to_fp16()), so it is numerically inert.
    # With --fp32 the model stays fp32, so keep X in float32 too (else HalfTensor input
    # vs FloatTensor weight mismatch).
    if not args.fp32:
        X = X.astype(np.float16)
    split_info = np.load(split_info_path)
    torque_levels = split_info["torque"]
    reps          = split_info["repetition"]

    n_fault_classes = int(np.unique(y_fault).size)

    recording_ids = create_recording_ids(y_fault, torque_levels, reps)

    print(f"  Dataset: X={X.shape}, fault_classes={n_fault_classes}, "
          f"torque_levels={int(np.unique(torque_levels).size)}")
    print(f"  Total recordings: {len(np.unique(recording_ids))}")

    # ── Leave-Torque-Out CV, repeated across seeds ────────────────────────
    save_dir = MODELS_DIR / "motor" / "dann_lto"
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    def run_cv_for_seed(seed: int) -> list[dict]:
        """Run the full Leave-Torque-Out CV for one seed.

        For every fold the held-out torque is used ONLY for final evaluation.
        Model selection (early stopping, checkpointing) uses an inner validation
        set carved from the TRAINING torques, so the held-out torque stays fully
        unseen — no leakage into model selection.
        """
        set_seed(seed)
        protokol = "extremes" if args.extremes_out else "lto"
        domain_values = torque_levels

        folds = bangun_fold(
            protokol=protokol,
            X_len=len(X),
            y_fault=y_fault,
            torque_levels=torque_levels,
            n_folds=args.n_folds,
            n_extremes=args.n_extremes,
        )
        if protokol == "extremes":
            uniq = sorted(set(int(t) for t in torque_levels.tolist()))
            ne = max(1, args.n_extremes)
            torsi_uji = sorted(set(uniq[:ne]) | set(uniq[-ne:]))
            print(f"  [EXTREMES-OUT] train torques={sorted(set(uniq) - set(torsi_uji))}  "
                  f"test (EXTREME) torques={torsi_uji}  << load extrapolation")
        fold_results: list[dict] = []

        for fold_i, (train_idx, test_idx) in enumerate(folds):
            if args.fold_idx >= 0 and fold_i != args.fold_idx:
                continue  # single-fold selection: run only the requested interior fold
            if args.max_folds and fold_i >= args.max_folds:
                print(f"  [max_folds={args.max_folds}] stopping after {fold_i} fold(s).")
                break
            train_torques = sorted(set(torque_levels[train_idx].tolist()))
            test_torques = sorted(set(torque_levels[test_idx].tolist()))

            print(f"\n{'#'*60}")
            print(f"  SEED {seed} | FOLD {fold_i + 1}/{len(folds)}  (protokol={protokol})")
            print(f"  Train torques: {train_torques}")
            print(f"  Test torques : {test_torques}  << UNSEEN LOADS (held-out)")
            print(f"{'#'*60}")

            # Held-out torque never overlaps training torques (by construction).
            verify_no_leakage(recording_ids, train_idx, test_idx)

            # Inner validation for model selection — carved from TRAIN torques only
            inner_train_idx, inner_valid_idx = _inner_train_valid_split(
                train_idx=train_idx,
                recording_ids=recording_ids,
                y_fault=y_fault,
                seed=seed,
            )
            verify_no_leakage(recording_ids, inner_train_idx, inner_valid_idx)
            print(
                f"  Inner split: {len(set(recording_ids[inner_train_idx]))} train + "
                f"{len(set(recording_ids[inner_valid_idx]))} valid recordings "
                f"(model selection on inner valid, NOT the held-out torque)."
            )

            # Domain targets span all TRAIN torque levels; held-out ones masked (-1).
            y_domain, n_domains, train_torque_to_idx = prepare_fold_domains(
                domain_values, train_idx, test_idx)
            print(f"  Domain mapping (train): {train_torque_to_idx}")

            # Tabel indeks domain skalar, dibangun PER FOLD dari pemetaan torsi latih,
            # sehingga normalisasi [0,1] tidak pernah menyentuh torsi uji.
            domain_u = None
            if args.domain_index != "none":
                domain_u = build_domain_u(args.domain_index, train_torque_to_idx)
                print(f"  [UJI A] {ringkas_u(args.domain_index, domain_u, train_torque_to_idx)}")

            # fastai dls: train = inner_train, valid = inner_valid (both SEEN torques)
            splits = (list(inner_train_idx), list(inner_valid_idx))
            dls = build_dann_dataloaders(
                X=X,
                y_fault=y_fault,
                y_domain=y_domain,
                splits=splits,
                batch_size=args.batch_size,
                augment=not args.no_augment,
                seed=seed,
                by_sample_standardize=not args.global_standardize,
                by_var_standardize=args.by_var,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )

            fold_name = f"{args.name}_{args.backbone}_seed{seed}_fold{fold_i}"
            ckpt_path = save_dir / f"{fold_name}.pth"

            if args.method == "dsn":
                learn = build_dsn_learner(
                    dls=dls,
                    n_fault_classes=n_fault_classes,
                    n_domains=n_domains,
                    model_name=fold_name,
                    save_dir=save_dir,
                    wd=TRAIN_CONFIG.get("weight_decay", 1e-5),
                    patience=TRAIN_CONFIG.get("early_stopping_patience", 10),
                    backbone=args.backbone,
                    lambda_domain=args.dsn_lambda_domain,
                    lambda_ortho=args.dsn_lambda_ortho,
                    lambda_recon=args.dsn_lambda_recon,
                    model_selection=not args.fixed_budget,
                )
            elif args.method == "dsran":
                learn = build_dsran_learner(
                    dls=dls,
                    n_fault_classes=n_fault_classes,
                    n_domains=n_domains,
                    lambda_max=args.lambda_max,
                    model_name=fold_name,
                    save_dir=save_dir,
                    wd=TRAIN_CONFIG.get("weight_decay", 1e-5),
                    patience=TRAIN_CONFIG.get("early_stopping_patience", 10),
                    backbone=args.backbone,
                    grl_gamma=args.grl_gamma,
                    lambda_domain=args.dsn_lambda_domain,
                    lambda_ortho=args.dsn_lambda_ortho,
                    lambda_recon=args.dsn_lambda_recon,
                    model_selection=not args.fixed_budget,
                )
            elif args.method == "supcon":
                learn = build_supcon_learner(
                    dls=dls,
                    n_fault_classes=n_fault_classes,
                    n_domains=n_domains,
                    model_name=fold_name,
                    save_dir=save_dir,
                    wd=TRAIN_CONFIG.get("weight_decay", 1e-5),
                    patience=TRAIN_CONFIG.get("early_stopping_patience", 10),
                    backbone=args.backbone,
                    lambda_sc=args.supcon_lambda,
                    temperature=args.supcon_temp,
                    proj_dim=args.supcon_proj_dim,
                    model_selection=not args.fixed_budget,
                )
            elif args.method == "cdan":
                learn = build_cdan_learner(
                    dls=dls,
                    n_fault_classes=n_fault_classes,
                    n_domains=n_domains,
                    lambda_max=args.lambda_max,
                    model_name=fold_name,
                    save_dir=save_dir,
                    wd=TRAIN_CONFIG.get("weight_decay", 1e-5),
                    patience=TRAIN_CONFIG.get("early_stopping_patience", 10),
                    backbone=args.backbone,
                    grl_gamma=args.grl_gamma,
                    use_entropy=not args.cdan_no_entropy,
                    model_selection=not args.fixed_budget,
                )
            elif args.method == "cdansupcon":
                learn = build_cdan_supcon_learner(
                    dls=dls,
                    n_fault_classes=n_fault_classes,
                    n_domains=n_domains,
                    lambda_max=args.lambda_max,
                    model_name=fold_name,
                    save_dir=save_dir,
                    wd=TRAIN_CONFIG.get("weight_decay", 1e-5),
                    patience=TRAIN_CONFIG.get("early_stopping_patience", 10),
                    backbone=args.backbone,
                    grl_gamma=args.grl_gamma,
                    use_entropy=not args.cdan_no_entropy,
                    lambda_sc=args.supcon_lambda,
                    temperature=args.supcon_temp,
                    proj_dim=args.supcon_proj_dim,
                    model_selection=not args.fixed_budget,
                )
            elif args.method == "dannsupcon":
                learn = build_dann_supcon_learner(
                    dls=dls,
                    n_fault_classes=n_fault_classes,
                    n_domains=n_domains,
                    lambda_max=args.lambda_max,
                    model_name=fold_name,
                    save_dir=save_dir,
                    wd=TRAIN_CONFIG.get("weight_decay", 1e-5),
                    patience=TRAIN_CONFIG.get("early_stopping_patience", 10),
                    backbone=args.backbone,
                    grl_gamma=args.grl_gamma,
                    lambda_sc=args.supcon_lambda,
                    temperature=args.supcon_temp,
                    proj_dim=args.supcon_proj_dim,
                    model_selection=not args.fixed_budget,
                    bt_target_ratio=args.bt_ratio,
                    bt_anneal=not args.bt_no_anneal,
                    domain_u=domain_u,
                )
            else:
                learn = build_dann_learner(
                    dls=dls,
                    n_fault_classes=n_fault_classes,
                    n_domains=n_domains,
                    lambda_max=args.lambda_max,
                    model_name=fold_name,
                    save_dir=save_dir,
                    wd=TRAIN_CONFIG.get("weight_decay", 1e-5),
                    patience=TRAIN_CONFIG.get("early_stopping_patience", 10),
                    backbone=args.backbone,
                    grl_gamma=args.grl_gamma,
                    domain_label_smoothing=args.domain_label_smoothing,
                    model_selection=not args.fixed_budget,
                    bt_target_ratio=args.bt_ratio,
                    bt_anneal=not args.bt_no_anneal,
                )

            if args.resume and ckpt_path.exists():
                print(f"  [RESUME] Checkpoint found: {ckpt_path.name} — loading for eval.")
                if not args.fp32:
                    learn = learn.to_fp16()   # BUGFIX: match fp16 input dtype during eval (autocast)
                learn.load(fold_name)
            else:
                if not args.fp32:
                    learn = learn.to_fp16()   # mixed precision: 2-3x speedup on RTX 40xx
                learn.fit_one_cycle(args.epochs, lr_max=args.lr)
                if args.fixed_budget:
                    learn.save(fold_name)   # no SaveModelCallback in this mode; persist FINAL model

            # ── Final evaluation on the held-out (UNSEEN) torque ──────────
            test_res = evaluate_on_test(
                learn=learn,
                X=X,
                y_fault=y_fault,
                recording_ids=recording_ids,
                test_idx=np.array(test_idx),
                n_fault_classes=n_fault_classes,
                batch_size=args.batch_size,
                seed=seed,
                by_var_standardize=args.by_var,
            )
            test_res["seed"] = seed
            test_res["fold"] = fold_i

            # Generalization-gap diagnostic (train vs inner-val vs held-out).
            train_probe_idx = inner_train_idx[: min(2000, len(inner_train_idx))]
            train_probe = evaluate_on_test(
                learn=learn, X=X, y_fault=y_fault, recording_ids=recording_ids,
                test_idx=train_probe_idx, n_fault_classes=n_fault_classes,
                batch_size=args.batch_size, seed=seed, by_var_standardize=args.by_var,
            )
            gap = generalization_gap(
                train_acc=train_probe["win_accuracy"],
                test_acc=test_res["win_accuracy"],
                val_acc=None,
            )
            flag = "  <<< OVERFIT?" if gap["overfit"] else ""
            print(f"    Gen-gap (win-acc) train={gap['train_acc']*100:.1f}% "
                  f"held-out={gap['test_acc']*100:.1f}% gap={gap['gap']*100:.1f}%{flag}")

            print(f"\n  [SEED {seed} FOLD {fold_i + 1}] UNSEEN-torque test metrics:")
            print(f"    Win-F1-macro : {test_res['win_f1_macro']*100:.2f}%  "
                  f"(N={test_res['n_windows']} windows)")
            print(f"    Win-Accuracy : {test_res['win_accuracy']*100:.2f}%")
            print(f"    Rec-Accuracy : {test_res['rec_accuracy']*100:.2f}%  "
                  f"(N={test_res['n_recordings']} recordings)")
            print(f"    Rec-F1-macro : {test_res['rec_f1_macro']*100:.2f}%")

            fold_results.append(test_res)
            del learn
            torch.cuda.empty_cache()

        return fold_results

    # ── Run all seeds + cross-seed summary (extracted to src/training/cross_seed) ──
    per_seed_means = run_cross_seed(seeds, run_cv_for_seed, cv_label="Leave-Torque-Out")
    print_cross_seed_summary(
        per_seed_means,
        title=f"DANN LTO-CV SUMMARY: {args.name}",
        subtitle_lines=[
            f"lambda_max={args.lambda_max}  |  seeds={seeds}",
            "Reported as mean +/- std across seeds (each seed = full LTO-CV).",
        ],
        domain_word="torque",
        also_per_seed_rec=True,
        trailing_lines=[
            "=" * 62, "",
            "  COMPARISON GUIDE:",
            "  +----------------------------------------------------------+",
            "  | Run baseline:   --lambda_max 0.0 --name ..._baseline     |",
            "  | Run full DANN:  --lambda_max 1.0 --name ..._dann         |",
            "  | Compare by Win-F1-macro (primary), mean +/- std/seeds.   |",
            "  | A backbone wins only if it clears the cross-seed std.    |",
            "  +----------------------------------------------------------+",
            "",
        ],
    )


if __name__ == "__main__":
    main()
