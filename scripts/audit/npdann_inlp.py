"""npdann_inlp.py — NP-DANN tahap 1: telusuri kurva akurasi-vs-invariansi lewat INLP.

LATAR (Sec. IV-D makalah CENIM 2026)
    Lima belas metode diaudit; pada trunk tidak satu pun menghapus >3.3% informasi domain,
    dan titik paling invarian yang pernah dicapai apa pun adalah 85.54% (z_task OFS o3/d0.5).
    Sumbu invariansi membentang 16.67%-100%; rentang 16.67%-85.54% — 83% sumbu — BELUM
    PERNAH disentuh satu eksperimen pun, karena lambda tidak mengendalikan invariansi.

METODE
    INLP (Ravfogel et al., ACL 2020, arXiv:2004.07667): latih classifier linear yang
    memprediksi domain dari fitur, proyeksikan fitur ke nullspace baris-ruang bobotnya,
    ulangi. Tiap iterasi membuang <= n_domains-1 arah. Jumlah iterasi k adalah DIAL —
    knob yang lambda tidak pernah bisa jadi.

    P_0 = I
    ulangi k kali:
        W  <- classifier linear domain pada (F @ P)          # (n_domains, d)
        B  <- basis ortonormal baris-ruang W (lewat SVD)
        P  <- (I - B^T B) @ P

PROTOKOL BEBAS KEBOCORAN
    P dipelajari HANYA dari fitur torsi latih (1-6) memakai label domain latih.
    Kepala fault dilatih ulang pada fitur latih terproyeksi, lalu dievaluasi pada torsi
    ekstrem 0 & 7 yang tak pernah dilihat. Torsi uji tidak menyentuh P maupun kepala fault.

PROBE NONLINEAR — ditambahkan 8 Agu 2026
    Keberatan yang beralasan: INLP hanya menjamin hilangnya
    keterbacaan LINEAR. Probe linear yang jatuh 96,17 -> 20,00 karena itu tidak
    membuktikan informasi beban PERGI; ia bisa saja sekadar menjadi nonlinear, dan
    proyeksi nullspace tidak menyentuh yang nonlinear sama sekali.

    Makalah (Sec. IV-D) menutup ini dengan caveat. Skrip ini menutupnya dengan pengukuran: pada
    fitur terproyeksi yang SAMA, di lipatan GroupKFold yang SAMA, sebuah MLP satu
    lapis tersembunyi diadu dengan regresi logistik.
      probe_nl jatuh ke chance juga  -> informasi domain memang pergi; caveatnya boleh
                                        dilunakkan menjadi pengukuran.
      probe_nl bertahan tinggi       -> caveat R2 TEGAK dan terkuantifikasi: yang INLP
                                        hapus hanya bagian linearnya.

    KONTROL LABEL ACAK. MLP berkapasitas lebih besar daripada probe linear, jadi
    "di atas chance" saja bukan bukti — ia bisa datang dari kapasitas, bukan dari
    sinyal. Karena itu probe yang sama dijalankan atas label domain yang DIACAK, di
    k terkecil dan k terbesar. Angka itulah chance empiris untuk probe, split, dan
    ukuran sampel ini; chance analitik 100/n_domain hanya rujukan kasar. Kontrol
    dibatasi dua titik karena biayanya sama dengan probe sungguhan di tiap titik.

KELUARAN
    Untuk tiap k: Win-F1 pada torsi ekstrem, probe domain LINEAR dan NONLINEAR
    (invariansi yang BENAR-BENAR tercapai), dan dimensi tersisa. Itulah kurvanya.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, PROCESSED_DIR
from src.preprocessing.io import load_dataset
from src.training.dann_factory import build_dann_dataloaders
from src.training.splits import prepare_fold_domains
from src.training.recordings import create_recording_ids
from src.models.dann import build_dann_model
from src.models.supcon import build_supcon_model

SEEDS = [42, 123, 2025, 7, 17, 2024, 2026, 88]
CKPT = MODELS_DIR / "motor" / "dann_lto"
SOURCES = {
    "ERM":    ("FP32_tsit_ext_baseline_tsitplus", "dann"),
    "SupCon": ("SupCon_l10_t007_tsitplus",        "supcon"),
}
K_GRID = [0, 1, 2, 3, 5, 8, 12, 16, 21]


def nullspace_proj(W: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Proyektor ortogonal ke nullspace baris-ruang W. W: (c, d) -> P: (d, d)."""
    _, s, Vt = np.linalg.svd(W, full_matrices=False)
    B = Vt[s > tol * max(1.0, s[0])]          # basis ortonormal baris-ruang
    d = W.shape[1]
    return np.eye(d) - B.T @ B


def inlp(F: np.ndarray, dom: np.ndarray, k: int, seed: int, max_iter: int = 1000):
    """k iterasi INLP. Kembalikan (P, n_dim_tersisa)."""
    d = F.shape[1]
    P = np.eye(d)
    for _ in range(k):
        clf = LogisticRegression(max_iter=max_iter, n_jobs=-1, random_state=seed)
        clf.fit(F @ P, dom)
        P = nullspace_proj(clf.coef_) @ P
    rank = int(np.linalg.matrix_rank(P, tol=1e-6))
    return P, rank


def _probe(Z: np.ndarray, dom: np.ndarray, groups: np.ndarray, seed: int,
           nonlinear: bool) -> float:
    """Probe domain pada fitur beku (GroupKFold per-rekaman).

    Normalisasi, split, dan metrik IDENTIK untuk kedua probe; yang berbeda hanya
    keluarga fungsi yang boleh dipakai. Itulah syarat supaya selisih linear-lawan-
    nonlinear dapat dibaca sebagai informasi yang tersimpan nonlinear, bukan sebagai
    artefak perbedaan protokol.
    """
    mu, sd = Z.mean(0, keepdims=True), Z.std(0, keepdims=True) + 1e-8
    Zn = (Z - mu) / sd
    accs = []
    for tr, te in GroupKFold(n_splits=3).split(Zn, dom, groups=groups):
        if nonlinear:
            c = MLPClassifier(hidden_layer_sizes=(128,), activation="relu",
                              max_iter=300, early_stopping=True, n_iter_no_change=10,
                              random_state=seed)
        else:
            c = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=seed)
        c.fit(Zn[tr], dom[tr])
        accs.append(accuracy_score(dom[te], c.predict(Zn[te])))
    return float(np.mean(accs))


def probe_domain(Z: np.ndarray, dom: np.ndarray, groups: np.ndarray, seed: int) -> float:
    """Probe domain LINEAR pada fitur beku."""
    return _probe(Z, dom, groups, seed, nonlinear=False)


def probe_domain_nl(Z: np.ndarray, dom: np.ndarray, groups: np.ndarray, seed: int) -> float:
    """Probe domain NONLINEAR (MLP 1 lapis tersembunyi) — R2-6b."""
    return _probe(Z, dom, groups, seed, nonlinear=True)


def chance_empiris(Z: np.ndarray, dom: np.ndarray, groups: np.ndarray, seed: int) -> float:
    """Probe nonlinear atas label domain DIACAK — chance untuk probe & split ini.

    Pengacakan dilakukan pada tingkat SAMPEL, bukan rekaman, karena yang dikontrol
    adalah kapasitas probe: berapa akurasi yang dapat dicapai MLP ini tanpa ada
    hubungan fitur-domain sama sekali.
    """
    rng = np.random.default_rng(seed)
    return _probe(Z, rng.permutation(dom), groups, seed, nonlinear=True)


@torch.no_grad()
def extract(model, dl, device) -> np.ndarray:
    out: list[np.ndarray] = []
    h = model.backbone.register_forward_hook(
        lambda _m, _i, o: out.append(o.detach().float().cpu().numpy()))
    try:
        for b in dl:
            model(b[0].to(device))
    finally:
        h.remove()
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ERM", choices=list(SOURCES))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--ks", default=",".join(map(str, K_GRID)))
    ap.add_argument("--out", default="results/motor/npdann_inlp.npz")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="'cpu' dipakai saat kampanye latih memakai kartunya; "
                         "ekstraksi ini kecil tetapi OOM akan membunuh kampanye itu")
    ap.add_argument("--tanpa-nonlinear", action="store_true",
                    help="lewati probe nonlinear R2-6b (perilaku sebelum 8 Agu 2026)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ks = [int(s) for s in args.ks.split(",") if s.strip()]
    pakai_nl = not args.tanpa_nonlinear
    # kontrol label acak hanya di k terkecil dan terbesar: biayanya sama dengan probe
    # sungguhan, dan yang perlu diketahui adalah chance SEBELUM dan SESUDAH proyeksi.
    ks_kontrol = {min(ks), max(ks)} if pakai_nl and ks else set()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    motor = PROCESSED_DIR / "motor"
    ds = motor / "motor_multiclass_w4096_s4096_ss_env.npz"
    X, y_fault, _ = load_dataset(ds)
    si = np.load(motor / f"{ds.stem}_split_info.npz")
    torque, reps = si["torque"], si["repetition"]
    rec = create_recording_ids(y_fault, torque, reps)

    uniq = sorted(set(int(t) for t in torque.tolist()))
    tmask = np.isin(torque, [uniq[0], uniq[-1]])
    tr_idx, te_idx = np.where(~tmask)[0], np.where(tmask)[0]
    y_dom, n_dom, _ = prepare_fold_domains(torque, tr_idx, te_idx)
    n_cls = int(np.unique(y_fault).size)
    c_in, seq_len = X.shape[1], X.shape[2]
    CH = 100.0 / n_dom

    def mk(idx):
        dls = build_dann_dataloaders(
            X=X, y_fault=y_fault, y_domain=np.zeros_like(y_fault, dtype=np.int64),
            splits=(list(idx[:8]), list(idx)), batch_size=64, augment=False, seed=42)
        try:
            dls.valid.fake_l.num_workers = 0
        except AttributeError:
            pass
        return dls.valid

    dl_tr, dl_te = mk(tr_idx), mk(te_idx)
    dom_tr = y_dom[tr_idx].astype(int)
    grp_tr = rec[tr_idx]
    yf_tr, yf_te = y_fault[tr_idx].astype(int), y_fault[te_idx].astype(int)
    tq_te = torque[te_idx].astype(int)
    lo, hi = uniq[0], uniq[-1]

    prefix, kind = SOURCES[args.source]
    print(f"device={device}  sumber={args.source}  chance={CH:.2f}%  "
          f"n_train={len(tr_idx)}  n_test={len(te_idx)}\n")

    rows = []
    for seed in seeds:
        ck = CKPT / f"{prefix}_seed{seed}_fold0.pth"
        if not ck.exists():
            print(f"[lewati] {ck.name} tidak ada")
            continue
        model = (build_dann_model("tsitplus", c_in, seq_len, n_cls, n_dom) if kind == "dann"
                 else build_supcon_model("tsitplus", c_in, seq_len, n_cls)).to(device)
        sd = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(sd if not (isinstance(sd, dict) and "model" in sd) else sd["model"],
                              strict=False)
        model.eval()
        Ftr, Fte = extract(model, dl_tr, device), extract(model, dl_te, device)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"  seed {seed}: {'k':>3} {'dim':>4} {'lin%':>7} {'nl%':>7} {'acak%':>7} "
              f"{'Win-F1':>8} {'F1 tq'+str(lo):>8} {'F1 tq'+str(hi):>8}")
        for k in ks:
            P, rank = inlp(Ftr, dom_tr, k, seed)
            Ztr, Zte = Ftr @ P, Fte @ P
            mu, sd_ = Ztr.mean(0, keepdims=True), Ztr.std(0, keepdims=True) + 1e-8
            head = LogisticRegression(max_iter=3000, n_jobs=-1, random_state=seed)
            head.fit((Ztr - mu) / sd_, yf_tr)
            pred = head.predict((Zte - mu) / sd_)
            f1 = f1_score(yf_te, pred, average="macro", labels=range(n_cls), zero_division=0) * 100
            f1lo = f1_score(yf_te[tq_te == lo], pred[tq_te == lo], average="macro",
                            labels=range(n_cls), zero_division=0) * 100
            f1hi = f1_score(yf_te[tq_te == hi], pred[tq_te == hi], average="macro",
                            labels=range(n_cls), zero_division=0) * 100
            pr = probe_domain(Ztr, dom_tr, grp_tr, seed) * 100
            pr_nl = probe_domain_nl(Ztr, dom_tr, grp_tr, seed) * 100 if pakai_nl else np.nan
            pr_ac = (chance_empiris(Ztr, dom_tr, grp_tr, seed) * 100
                     if k in ks_kontrol else np.nan)
            rows.append([seed, k, rank, pr, f1, f1lo, f1hi, pr_nl, pr_ac])
            f = lambda v: f"{v:>6.2f}%" if np.isfinite(v) else f"{'-':>7}"
            print(f"  {'':10} {k:>3} {rank:>4} {f(pr)} {f(pr_nl)} {f(pr_ac)} "
                  f"{f1:>8.2f} {f1lo:>8.2f} {f1hi:>8.2f}")
        print()

    A = np.array(rows, dtype=float)
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, data=A, ks=np.array(ks), chance=CH, source=args.source)

    print("=" * 90)
    print(f"KURVA AKURASI-vs-INVARIANSI  (sumber {args.source}, n={len(set(A[:,0]))} seed)")
    print("=" * 90)
    print(f"{'k':>3} {'dim':>5} {'probe LINEAR %':>16} {'probe NONLINEAR %':>19} "
          f"{'Win-F1':>16}")
    def sd(v: np.ndarray) -> float:
        """sd antar-seed; 0 bila hanya satu seed (uji asap), bukan nan yang mencolok."""
        v = v[np.isfinite(v)]
        return float(v.std(ddof=1)) if v.size > 1 else 0.0

    base = None
    for k in ks:
        m = A[:, 1] == k
        pr, f1, prnl = A[m, 3], A[m, 4], A[m, 7]
        if base is None:
            base = f1.mean()
        nl = (f"{np.nanmean(prnl):>12.2f} +/-{sd(prnl):>4.2f}"
              if np.isfinite(prnl).any() else f"{'-':>19}")
        print(f"{k:>3} {A[m,2].mean():>5.0f} {pr.mean():>9.2f} +/-{sd(pr):>4.2f} "
              f"{nl} {f1.mean():>9.2f} +/-{sd(f1):>4.2f}")
    # label menyebut k TERKECIL yang dijalankan, bukan "k=0" yang dikeraskan: pada run
    # satu titik (mis. --ks 21) label lama mencetak angka k=21 dengan nama k=0.
    print(f"\nchance analitik = {CH:.2f}%   |   k={min(ks)} Win-F1 = {base:.2f} (rujukan sweep)")

    ac = A[np.isfinite(A[:, 8])]
    if ac.size:
        print("\nKONTROL LABEL DIACAK (chance empiris untuk probe nonlinear ini):")
        for k in sorted(set(ac[:, 1].astype(int))):
            v = ac[ac[:, 1] == k, 8]
            print(f"  k={k:<3} {v.mean():>6.2f}% +/-{sd(v):>4.2f}   (n={len(v)} seed)")

    # --- vonis R2-6b: apakah caveat "hanya linear" tegak? ---
    kmax = max(ks)
    m = A[:, 1] == kmax
    if np.isfinite(A[m, 7]).any():
        lin_k, nl_k = A[m, 3].mean(), np.nanmean(A[m, 7])
        acak_k = np.nanmean(A[m, 8]) if np.isfinite(A[m, 8]).any() else CH
        print("\n" + "=" * 90)
        print(f"R2-6b — apakah INLP hanya menghapus bagian LINEARnya?  (di k={kmax})")
        print("=" * 90)
        print(f"  probe linear        {lin_k:>6.2f}%")
        print(f"  probe nonlinear     {nl_k:>6.2f}%")
        print(f"  chance (kontrol)    {acak_k:>6.2f}%")
        print(f"  kelebihan nonlinear atas chance = {nl_k - acak_k:+.2f} poin")
        print()
        if nl_k - acak_k > 5.0:
            print("  >> CAVEAT R2 TEGAK dan kini TERKUANTIFIKASI. Probe nonlinear tetap di")
            print(f"     atas chance sebesar {nl_k - acak_k:.2f} poin sesudah proyeksi, jadi")
            print("     informasi beban TIDAK pergi — ia menjadi nonlinear, dan proyeksi")
            print("     nullspace tidak menyentuh yang nonlinear. Makalah harus menyebut")
            print("     angka ini, bukan sekadar memberi caveat kualitatif.")
        else:
            print("  >> Probe nonlinear ikut jatuh ke chance. Keberatan 'hanya keterbacaan")
            print("     linear' TERJAWAB DENGAN PENGUKURAN untuk sumber dan k ini: MLP")
            print("     berkapasitas lebih besar pun tidak menemukan sisa informasi beban.")
        print()
        print("  BATAS: satu keluarga probe (MLP 1 lapis, 128 unit) bukan 'semua fungsi")
        print("  nonlinear'. Yang terukur adalah batas bawah informasi tersisa, bukan bukti")
        print("  ketiadaannya. Jangan menuliskannya sebagai 'terbukti tidak ada'.")

    print(f"\ntersimpan: {out}")
    print("kolom: seed, k, rank, probe_linear, win_f1, f1_lo, f1_hi, probe_nl, probe_acak")


if __name__ == "__main__":
    main()
