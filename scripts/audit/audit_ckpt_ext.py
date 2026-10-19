"""audit_ckpt_ext.py — Audit checkpoint EXT: per-torsi, per-kelas, dan domain probe.

Menjawab tiga pertanyaan yang tidak bisa dijawab dari log:

1. PER-TORSI. Metrik yang dilaporkan menggabungkan torsi 0 dan 7 jadi satu angka.
   Keduanya arah ekstrapolasi berlawanan (di bawah vs di atas rentang latih), jadi
   kemenangan di satu sisi bisa tertutup kekalahan di sisi lain.

2. PER-KELAS. r1b (satu batang patah) punya sideband paling lemah; kalau GRL
   menggosok residu fault, kelas itu yang paling dulu jatuh.

3. DOMAIN PROBE — yang paling menentukan. Log mencatat domain_acc dari
   diskriminator DARING yang dilatih bersamaan dengan backbone. Nilai di bawah
   chance bisa berarti dua hal yang sangat berbeda:
     (a) representasi memang anti-prediktif terhadap domain, atau
     (b) diskriminator sekadar tertinggal karena backbone bergerak tiap langkah
         (artefak pengukuran, bukan sifat representasi).
   Probe menyelesaikannya: bekukan backbone, latih classifier domain BARU dari nol
   di atas fitur beku. Probe yang terlatih penuh tidak mungkin di bawah chance.
     probe ~ 1/6  -> fitur benar-benar invarian-domain
     probe >> 1/6 -> GRL hanya menipu diskriminator daring; info domain masih ada

Jalur data, standardisasi, dan split identik dengan train_motor_dann.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, PROCESSED_DIR
from src.preprocessing.io import load_dataset
from src.training.dann_factory import build_dann_dataloaders
from src.training.splits import prepare_fold_domains
from src.training.recordings import create_recording_ids
from src.evaluation.metrics import recording_level_metrics
from src.models.dann import build_dann_model
from src.models.dann_supcon import build_dann_supcon_model
from src.models.supcon import build_supcon_model
from src.models.cdan import build_cdan_model, build_cdan_supcon_model
from src.models.dsn import build_dsn_model
from src.models.dsran import build_dsran_model

SEEDS = [42, 123, 2025, 7, 17, 2024, 2026, 88]
SEEDS_S12 = [1, 3, 5, 11, 13, 23, 31, 57, 99, 101, 202, 777]
CKPT_DIR = MODELS_DIR / "motor" / "dann_lto"

# label -> (prefix checkpoint, jenis model)
# Prefix boleh tuple: sweep yang sama dijalankan dalam dua tahap dengan --name berbeda
# (n=8 lalu n=12). Seed di kedua tahap tidak beririsan, jadi cukup coba tiap prefix.
METHODS = {
    # --- non-adversarial ---
    "ERM (baseline)":      (("FP32_tsit_ext_baseline_tsitplus",
                             "S20_baseline_tsitplus"),           "dann"),
    "SupCon":              (("SupCon_l10_t007_tsitplus",
                             "S20_supcon_tsitplus"),             "supcon"),
    "OFS o1/d1":           ("DSN_tsit_ext_tsitplus",             "dsn"),
    "OFS o3/d0.5":         ("DSN_o3_d05_tsitplus",               "dsn"),
    "OFS o5/d1.0":         ("DSN_o5_d10_tsitplus",               "dsn"),
    # --- adversarial (GRL) ---
    "DANN lam=1.0":        (("FP32_tsit_ext_dann_lam10_tsitplus",
                             "S20_dann_lam10_tsitplus"),         "dann"),
    "DANN lam=0.3":        ("FP32_tsit_ext_dann_lam03_tsitplus", "dann"),
    "DANN lam=0.1":        ("FP32_tsit_ext_dann_lam01_tsitplus", "dann"),
    "DANN+ELS":            ("FP32_tsit_ext_dann_els01_tsitplus", "dann"),
    "DANN+SupCon":         (("DANNSupCon_l10_sc10_tsitplus",
                             "S20_dannsupcon_tsitplus"),         "dannsupcon"),
    # BT-DANN (2026-08-02): arsitektur IDENTIK dengan DANN+SupCon — yang berbeda hanya
    # callback pelatihan (lambda dari keseimbangan gradien, bukan jadwal waktu). Karena
    # itu jenis modelnya tetap "dannsupcon". Probe ini memutuskan apakah BT memperbaiki
    # GRL atau sekadar mematikannya: lambda memudar ke ~1e-6 di epoch akhir, jadi bila
    # probe terbaca seperti SupCon (~95,96%), klaim "memperbaiki GRL" gugur.
    "BT-DANN+SupCon r0.3": (("BT_dannsupcon_r03_tsitplus",
                             "BT_dannsupcon_r03_s12_tsitplus"),  "dannsupcon"),
    # BT polos gagal menstabilkan (87,07 +/- 3,47; +0,06 dari DANN vanilla, p=0,96).
    # Disertakan sebagai kontrol: kalau probe-nya sama dengan varian SupCon, maka
    # perbedaan Win-F1 +3,28 di antara keduanya BUKAN soal penghapusan info domain.
    "BT-DANN polos r0.3":  (("BT_dann_r03_tsitplus",
                             "BT_dann_r03_s12_tsitplus"),        "dann"),
    # UJI A (2026-08-02): BT-DANN+SupCon dengan kepala domain REGRESI (feat->64->1, MSE)
    # menggantikan kepala kategoris (feat->64->6, CE). Jenis modelnya HARUS varian regresi
    # — memuat bobot [1,64] ke kepala [6,64] gagal karena bentuknya beda, dan load_state_dict
    # strict=False hanya memaafkan kunci hilang, bukan bentuk yang tidak cocok.
    # Probe tetap klasifikasi 6-arah walau diskriminatornya regresi, agar angkanya sebanding
    # dengan seluruh tabel ini.
    "UJI A: BT+SupCon regresi-idx": ("DIDX_f2s_BT_dannsupcon_tsitplus",
                                     "dannsupcon_reg"),
    # A3 (dijalankan 2026-07-31, w4096): TSStandardize(by_sample=True, by_var=True) —
    # per-sampel DAN per-kanal. Memulihkan Vib_axial (teredam ke 3,9%) dan envelope (5,7%)
    # yang dibuang standardisasi satu-skalar. Agregat 8 seed: ERM +1,32 p=0,413;
    # DANN -3,09 p=0,400 (Sec. IV-C makalah). Pecahan per-torsi dihitung oleh skrip ini.
    # CATATAN: checkpoint ini dilatih DENGAN by_var, jadi mengauditnya tanpa by_var akan
    # memberi angka tak berarti. Audit WAJIB memakai --by_var (gerbang A3-A0).
    "A3: ERM by_var":      ("BYVAR_baseline_tsitplus",              "dann"),
    "A3: DANN by_var":     ("BYVAR_dann_lam10_tsitplus",            "dann"),
    # UJI C (2026-08-02): ERM pada jendela 16384. WAJIB diaudit dengan
    #   --npz motor_multiclass_w16384_s16384_ss_env.npz
    # Angkanya TIDAK sebanding dengan baris lain di tabel ini (basis window berbeda:
    # 3.200 vs 13.200); satu-satunya pembanding sah adalah ERM w4096 pada seed sama.
    "UJI C: ERM w16384":   ("W16384_baseline_tsitplus",                "dann"),
    # KONTROL UJI C: w4096 dengan jumlah window disamakan ke w16384 (3.200). WAJIB
    #   --npz motor_multiclass_w4096_s4096_ss_env_sub3200.npz
    "KONTROL: ERM w4096-sub": ("W4096SUB_baseline_tsitplus",           "dann"),
    # UJI E: w16384 stride 4096 (tumpang tindih 75%) — resolusi tinggi + volume penuh.
    #   --npz motor_multiclass_w16384_s4096_ss_env.npz
    # Dua prefix: tahap n=8 (S8) dan perluasan n=20 (S12, ditambahkan 3 Agu). Seed di kedua
    # tahap tidak beririsan, jadi cukup coba tiap prefix — pola sama dengan BT dan S20.
    "UJI E: ERM w16384-ovl": (("W16384OVL_baseline_tsitplus",
                               "W16384OVL_baseline_s12_tsitplus"),      "dann"),
    # UJI F (2026-08-03): lever METODE dijalankan ulang di jendela w16384-ovl, agar
    # batasan Uji E no. 2 tertutup — klaim "mengalahkan semua metode" baru sah bila
    # metodenya diuji di jendela yang sama. WAJIB
    #   --npz motor_multiclass_w16384_s4096_ss_env.npz
    # Pembandingnya HANYA "UJI E: ERM w16384-ovl" pada seed yang sama; angka w4096
    # mana pun tidak sebanding (batasan Uji E no. 5).
    "UJI F: SupCon w16384-ovl":    ("W16384OVL_supcon_tsitplus",       "supcon"),
    "UJI F: BT+SupCon w16384-ovl": ("W16384OVL_BT_dannsupcon_r03_tsitplus",
                                                                      "dannsupcon"),
    # TSSEQ (7 Agu 2026): faktorial R3 diulang dengan backbone NON-TRANSFORMER
    # TSSequencerPlus (BiLSTM menggantikan self-attention; tokenisasi, LayerNorm,
    # positional encoding, dan CLS-token identik dengan TSiTPlus). Memeriksa batasan
    # backbone tunggal (Sec. IV-B dan VI makalah CENIM 2026).
    # Tiap lengan WAJIB diaudit dengan --npz-nya sendiri; menukarnya memberi angka
    # tak berarti, sama seperti peringatan pada baris Uji C di atas.
    #   --npz motor_multiclass_w4096_s4096_ss_env_sub3200.npz
    "TSSEQ: ERM w4096-sub":   ("TSSEQ_W4096SUB_baseline_tssequencerplus",  "dann"),
    #   --npz motor_multiclass_w16384_s16384_ss_env.npz
    "TSSEQ: ERM w16384":      ("TSSEQ_W16384_baseline_tssequencerplus",    "dann"),
    #   --npz motor_multiclass_w4096_s4096_ss_env.npz
    "TSSEQ: ERM w4096-penuh": ("TSSEQ_W4096FULL_baseline_tssequencerplus", "dann"),
    #   --npz motor_multiclass_w16384_s4096_ss_env.npz
    "TSSEQ: ERM w16384-ovl":  ("TSSEQ_W16384OVL_baseline_tssequencerplus", "dann"),
    # LR3E4 (8 Agu 2026): faktorial R3 diulang pada LAJU BELAJAR KEDUA, 3e-4 lawan 1e-3.
    # Seluruh tabel yang ada berjalan di satu titik hiperparameter; sapuan atas semua skrip
    # run mengonfirmasi --lr tidak pernah dilewatkan sekali pun. Dua blok di bawah mengisi
    # kolom lr kedua untuk kedua backbone, sehingga backbone x lr jadi faktorial penuh.
    # Kampanye: scripts/_run_lr_faktorial.sh dan scripts/_run_hp_faktorial.sh.
    # Aturan --npz per lengan sama persis dengan blok TSSEQ di atas.
    #   --npz motor_multiclass_w4096_s4096_ss_env_sub3200.npz
    "LR3E4: ERM w4096-sub":   ("LR3E4_W4096SUB_baseline_tsitplus",         "dann"),
    #   --npz motor_multiclass_w16384_s16384_ss_env.npz
    "LR3E4: ERM w16384":      ("LR3E4_W16384_baseline_tsitplus",           "dann"),
    #   --npz motor_multiclass_w4096_s4096_ss_env.npz
    "LR3E4: ERM w4096-penuh": ("LR3E4_W4096FULL_baseline_tsitplus",        "dann"),
    #   --npz motor_multiclass_w16384_s4096_ss_env.npz
    "LR3E4: ERM w16384-ovl":  ("LR3E4_W16384OVL_baseline_tsitplus",        "dann"),
    #   --npz motor_multiclass_w4096_s4096_ss_env_sub3200.npz
    "LR3E4TSSEQ: ERM w4096-sub":   ("LR3E4TSSEQ_w4096sub_tssequencerplus",  "dann"),
    #   --npz motor_multiclass_w16384_s16384_ss_env.npz
    "LR3E4TSSEQ: ERM w16384":      ("LR3E4TSSEQ_w16384_tssequencerplus",    "dann"),
    #   --npz motor_multiclass_w4096_s4096_ss_env.npz
    "LR3E4TSSEQ: ERM w4096-penuh": ("LR3E4TSSEQ_w4096full_tssequencerplus", "dann"),
    #   --npz motor_multiclass_w16384_s4096_ss_env.npz
    "LR3E4TSSEQ: ERM w16384-ovl":  ("LR3E4TSSEQ_w16384ovl_tssequencerplus", "dann"),
    # D-P (27 Agu 2026): desimasi 2x. Dua lengan ERM berpasangan, flag identik,
    # T & df IDENTIK (2,1558 s / 0,4639 Hz); yang beda hanya panjang sekuens 16384 vs
    # 8192. Diaudit untuk D-P3 (per-torsi). Aturan --npz per lengan seperti blok Uji C.
    #   --npz motor_multiclass_w16384_s16384_ss_env.npz
    "D-P: ERM w16384":   ("DP_armA_w16384_tsitplus",  "dann"),
    "V-P: ERM w16384":   ("VP_armA_w16384_s12_tsitplus", "dann"),
    #   --npz motor_multiclass_w16384_s16384_ss_env_dec2.npz
    "D-P: ERM dec2":     ("DP_armB_dec2_tsitplus",    "dann"),
    "V-P: ERM dec2":     ("VP_armB_dec2_s12_tsitplus",  "dann"),
    "CDAN+E":              ("CDAN_l10_ent_tsitplus",             "cdan"),
    "CDAN+SupCon":         ("CDANSupCon_l10_sc10_tsitplus",      "cdansupcon"),
    "DSRAN l1.0 o3/d0.5":  ("DSRAN_l10_o3_d05_tsitplus",         "dsran"),
    "DSRAN l0.5 o3/d0.5":  ("DSRAN_l05_o3_d05_tsitplus",         "dsran"),
    "DSRANfix o1/d1/r1":   ("DSRANfix_l10_o1d1r1_tsitplus",      "dsran"),
}

FAULT_NAMES = ["rs", "r1b", "r2b", "r3b", "r4b"]


def build_model(kind: str, c_in: int, seq_len: int, n_classes: int, n_domains: int,
                backbone: str = "tsitplus"):
    # backbone TIDAK lagi dikunci. Sebelum 7 Agu 2026 seluruh cabang menuliskan
    # "tsitplus" secara literal, sehingga checkpoint backbone lain dimuat ke
    # arsitektur yang salah, 33 kunci gagal termuat, dan hasilnya setingkat
    # kebetulan acak yang tetap tercetak sebagai angka.
    if kind == "dann":
        return build_dann_model(backbone, c_in, seq_len, n_classes, n_domains)
    if kind == "dannsupcon":
        return build_dann_supcon_model(backbone, c_in, seq_len, n_classes, n_domains)
    if kind == "dannsupcon_reg":
        return build_dann_supcon_model(backbone, c_in, seq_len, n_classes, n_domains,
                                       domain_regression=True)
    if kind == "supcon":
        return build_supcon_model(backbone, c_in, seq_len, n_classes)
    if kind == "cdan":
        return build_cdan_model(backbone, c_in, seq_len, n_classes, n_domains)
    if kind == "cdansupcon":
        return build_cdan_supcon_model(backbone, c_in, seq_len, n_classes, n_domains)
    if kind == "dsn":
        return build_dsn_model(backbone, c_in, seq_len, n_classes, n_domains)
    if kind == "dsran":
        return build_dsran_model(backbone, c_in, seq_len, n_classes, n_domains)
    raise ValueError(kind)


@torch.no_grad()
def infer(model, dl, device):
    """Jalankan model, kembalikan (probs, feats). Fitur diambil lewat hook backbone."""
    feats: list[np.ndarray] = []
    h = model.backbone.register_forward_hook(
        lambda _m, _i, o: feats.append(o.detach().float().cpu().numpy())
    )
    probs: list[np.ndarray] = []
    try:
        for batch in dl:
            xb = batch[0].to(device)
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    finally:
        h.remove()
    return np.concatenate(probs), np.concatenate(feats)


def geometri_kelas(feat: np.ndarray, y: np.ndarray, dom: np.ndarray) -> tuple[float, float]:
    """Dua rasio geometri kelas pada fitur beku — menumpang di audit yang sudah jalan.

    MENGAPA ADA DI SINI (3 Agu 2026)
    --------------------------------
    Usulan MDA (Micro-level Distribution Alignment) berdiri di atas premis: metode
    kontrastif MENGOMPRES varians dalam-kelas secara berlebihan sehingga membuang fitur
    yang masih berguna untuk generalisasi. Premis itu bisa diuji GRATIS dari fitur beku
    yang sudah diekstrak untuk probe — tidak perlu run baru.

    Keduanya RASIO, bukan jarak mentah, karena skala fitur SupCon dan ERM berbeda dan
    jarak mentah tidak sebanding antar-model.

      intra_inter  = sebaran dalam-kelas / sebaran antar-kelas
                     kecil = kelas rapat relatif terhadap pemisahannya

      dom_in_class = sebaran sentroid antar-DOMAIN di dalam kelas / sebaran antar-kelas
                     kecil = beban tidak lagi memisahkan sampel sekelas.
                     INI yang tepat menjadi sasaran MDA.

    Cara membacanya: bila SupCon menekan dom_in_class jauh di bawah ERM DAN seed dengan
    dom_in_class terkecil justru berkinerja torsi-0 lebih buruk, premis MDA didukung data
    kita. Bila tidak ada hubungan, premisnya gugur tanpa perlu membangun apa pun.
    """
    kelas = np.unique(y)
    sentroid = {int(c): feat[y == c].mean(0) for c in kelas}
    pusat = np.stack([sentroid[int(c)] for c in kelas])

    # sebaran antar-kelas: jarak rerata sentroid kelas ke pusat global
    inter = float(np.linalg.norm(pusat - pusat.mean(0, keepdims=True), axis=1).mean())
    if inter < 1e-12:
        return float("nan"), float("nan")

    intra, dom_sp = [], []
    for c in kelas:
        m = y == c
        f_c, d_c = feat[m], dom[m]
        intra.append(float(np.linalg.norm(f_c - sentroid[int(c)], axis=1).mean()))
        cd = [f_c[d_c == d].mean(0) for d in np.unique(d_c) if (d_c == d).sum() > 0]
        if len(cd) > 1:
            cd = np.stack(cd)
            dom_sp.append(float(np.linalg.norm(cd - cd.mean(0, keepdims=True), axis=1).mean()))

    return (float(np.mean(intra)) / inter,
            float(np.mean(dom_sp)) / inter if dom_sp else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--methods", default="", help="subset label, dipisah ';'")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--probe_max_iter", type=int, default=2000)
    ap.add_argument("--no_probe", action="store_true")
    ap.add_argument("--by_var", action="store_true",
                    help="A3: standardisasi per-sampel DAN per-kanal. WAJIB dinyalakan "
                         "saat mengaudit checkpoint BYVAR_* — mereka dilatih dengan setelan "
                         "ini, dan mengauditnya tanpa ia memberi angka tak berarti. Sekelas "
                         "peringatan --npz di atas.")
    ap.add_argument("--out", default="results/motor/audit_ext.npz")
    ap.add_argument("--npz", default="motor_multiclass_w4096_s4096_ss_env.npz",
                    help="UJI C: dataset yang dipakai mengaudit. WAJIB cocok dengan "
                         "jendela tempat checkpoint dilatih — mengaudit ckpt w16384 di "
                         "atas data w4096 memberi angka yang tidak berarti apa-apa. "
                         "Default = jendela 4096 (seluruh tabel yang sudah ada).")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    methods = dict(METHODS)
    if args.methods:
        want = {m.strip() for m in args.methods.split(";") if m.strip()}
        methods = {k: v for k, v in methods.items() if k in want}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    motor_dir = PROCESSED_DIR / "motor"
    ds = motor_dir / args.npz
    if not ds.exists():
        raise SystemExit(f"[ABORT] dataset audit tidak ada: {ds}")
    print(f"dataset audit = {ds.name}")
    X, y_fault, _ = load_dataset(ds)
    split_info = np.load(motor_dir / f"{ds.stem}_split_info.npz")
    torque = split_info["torque"]
    reps = split_info["repetition"]
    rec_ids = create_recording_ids(y_fault, torque, reps)

    n_classes = int(np.unique(y_fault).size)
    c_in, seq_len = X.shape[1], X.shape[2]

    # --- split EXT identik dengan train_motor_dann.py -----------------------
    uniq = sorted(set(int(t) for t in torque.tolist()))
    test_torques = {uniq[0], uniq[-1]}
    test_mask = np.isin(torque, list(test_torques))
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    y_domain, n_domains, t2i = prepare_fold_domains(torque, train_idx, test_idx)
    print(f"train torques={sorted(set(uniq) - test_torques)}  test={sorted(test_torques)}  "
          f"n_domains={n_domains}")
    print(f"X={X.shape}  n_train={len(train_idx)}  n_test={len(test_idx)}\n")

    # dataloader evaluasi — persis pola evaluate_on_test (augment OFF)
    def make_dl(idx):
        dls = build_dann_dataloaders(
            X=X, y_fault=y_fault, y_domain=np.zeros_like(y_fault, dtype=np.int64),
            splits=(list(idx[: max(1, min(8, len(idx)))]), list(idx)),
            batch_size=args.batch_size, augment=False, seed=42,
            by_var_standardize=args.by_var,   # WAJIB cocok dengan setelan latih (gerbang A3-A0)
        )
        try:
            dls.valid.fake_l.num_workers = 0
        except AttributeError:
            pass
        return dls.valid

    dl_test = make_dl(test_idx)
    dl_train = make_dl(train_idx)

    y_test = y_fault[test_idx].astype(int)
    tq_test = torque[test_idx].astype(int)
    dom_train = y_domain[train_idx].astype(int)
    grp_train = rec_ids[train_idx]

    lo, hi = uniq[0], uniq[-1]
    store: dict[str, np.ndarray] = {}

    for label, (prefixes, kind) in methods.items():
        if isinstance(prefixes, str):
            prefixes = (prefixes,)
        print("=" * 104)
        print(label)
        print("=" * 104)
        print(f"{'seed':>6} {'F1 total':>9} {f'F1 tq{lo}':>9} {f'F1 tq{hi}':>9} "
              f"{'selisih':>8} | " + " ".join(f"{n:>6}" for n in FAULT_NAMES) +
              (f" | {'probe':>7} {'chance':>7} {'intra':>7} {'dom/kls':>7}"
               if not args.no_probe else ""))
        rows = []
        for seed in seeds:
            kandidat = [CKPT_DIR / f"{p}_seed{seed}_fold0.pth" for p in prefixes]
            ck = next((c for c in kandidat if c.exists()), None)
            if ck is None:
                print(f"{seed:>6}   [checkpoint tidak ada: {kandidat[0].name}]")
                continue
            bb = next((b for b in ("tssequencerplus", "tsitplus")
                       if ck.stem.split("_seed")[0].endswith(b)), "tsitplus")
            model = build_model(kind, c_in, seq_len, n_classes, n_domains, bb).to(device)
            sd = torch.load(ck, map_location=device, weights_only=False)
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                raise SystemExit(
                    f"[ABORT] {ck.name}: {len(missing)} kunci gagal termuat ke "
                    f"backbone '{bb}' (mis. {missing[:3]}). Bobot backbone TIDAK "
                    f"termuat, jadi angka apa pun dari sini setingkat kebetulan. "
                    f"Diubah dari WARN ke ABORT 7 Agu 2026.")
            model.eval()

            probs, _ = infer(model, dl_test, device)
            pred = probs.argmax(1)

            f1_all = f1_score(y_test, pred, average="macro", labels=range(n_classes),
                              zero_division=0) * 100
            per_tq = {}
            for tq in (lo, hi):
                m = tq_test == tq
                per_tq[tq] = f1_score(y_test[m], pred[m], average="macro",
                                      labels=range(n_classes), zero_division=0) * 100
            per_cls = f1_score(y_test, pred, average=None, labels=range(n_classes),
                               zero_division=0) * 100

            probe, r_intra, r_dom = np.nan, np.nan, np.nan
            if not args.no_probe:
                _, feat_tr = infer(model, dl_train, device)
                mu, sd_ = feat_tr.mean(0, keepdims=True), feat_tr.std(0, keepdims=True) + 1e-8
                Z = (feat_tr - mu) / sd_
                accs = []
                for tr, te in GroupKFold(n_splits=3).split(Z, dom_train, groups=grp_train):
                    clf = LogisticRegression(max_iter=args.probe_max_iter, n_jobs=-1)
                    clf.fit(Z[tr], dom_train[tr])
                    accs.append(accuracy_score(dom_train[te], clf.predict(Z[te])))
                probe = float(np.mean(accs))
                # premis MDA — menumpang fitur beku yang sama, ongkos tambahan ~nol
                r_intra, r_dom = geometri_kelas(feat_tr, y_fault[train_idx].astype(int),
                                                dom_train)

            # R2-P3 — Rec-Accuracy. rec_ids sudah dibangun (baris 250) tetapi
            # tidak pernah dikeluarkan sebelumnya, sehingga kolom Rec-Accuracy kosong.
            # Kolom DITAMBAHKAN DI UJUNG supaya indeks m[8]/m[9]/m[10] di bawah tetap sah.
            rec_all = recording_level_metrics(probs, y_test, rec_ids[test_idx],
                                              n_classes)["rec_accuracy"] * 100
            rec_tq = {}
            for tq in (lo, hi):
                m_tq = tq_test == tq
                rec_tq[tq] = recording_level_metrics(
                    probs[m_tq], y_test[m_tq], rec_ids[test_idx][m_tq], n_classes
                )["rec_accuracy"] * 100

            rows.append([seed, f1_all, per_tq[lo], per_tq[hi], *per_cls, probe,
                         r_intra, r_dom, rec_all, rec_tq[lo], rec_tq[hi]])
            tail = (f" | {probe*100:>6.2f}% {100/n_domains:>6.2f}%"
                    f" {r_intra:>7.3f} {r_dom:>7.3f}") if not args.no_probe else ""
            tail += f" | {rec_all:>6.2f} {rec_tq[lo]:>6.2f} {rec_tq[hi]:>6.2f}"
            print(f"{seed:>6} {f1_all:>9.2f} {per_tq[lo]:>9.2f} {per_tq[hi]:>9.2f} "
                  f"{per_tq[hi]-per_tq[lo]:>+8.2f} | " +
                  " ".join(f"{v:>6.2f}" for v in per_cls) + tail)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

        if rows:
            A = np.array(rows, dtype=float)
            store[label] = A
            m, s = A[:, 1:].mean(0), A[:, 1:].std(0, ddof=1)
            # indeks eksplisit: m[8]=probe, m[9]=intra/inter, m[10]=domain-dalam-kelas.
            # JANGAN pakai m[-1] untuk probe — dua kolom geometri ditambahkan setelahnya.
            tail = (f" | {m[8]*100:>6.2f}% {'':>7} {m[9]:>7.3f} {m[10]:>7.3f}"
                    if not args.no_probe else "")
            tail += f" | {m[11]:>6.2f} {m[12]:>6.2f} {m[13]:>6.2f}"
            print("-" * 104)
            print(f"{'MEAN':>6} {m[0]:>9.2f} {m[1]:>9.2f} {m[2]:>9.2f} {m[2]-m[1]:>+8.2f} | " +
                  " ".join(f"{v:>6.2f}" for v in m[3:8]) + tail)
            print(f"{'SD':>6} {s[0]:>9.2f} {s[1]:>9.2f} {s[2]:>9.2f} {'':>8} | " +
                  " ".join(f"{v:>6.2f}" for v in s[3:8]))
        print()

    if store:
        out = PROJECT_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **{k.replace(" ", "_"): v for k, v in store.items()})
        print(f"tersimpan: {out}")
        print("kolom: seed, f1_total, f1_tq_lo, f1_tq_hi, f1_rs..f1_r4b, domain_probe, "
              "intra_inter, dom_in_class, rec_acc, rec_acc_lo, rec_acc_hi")
        print("       (tiga kolom rec_* ditambahkan 3 Agu 18:58 untuk R2-P3; sengaja "
              "di UJUNG agar berkas format lama tetap terbaca pada indeks yang sama)")


if __name__ == "__main__":
    main()
