"""audit_normalisasi.py — Uji N dan P6: sapuan normalisasi, dan kanal envelope kedua.

Kriteria N-P1, N-P2, P6 TERKUNCI di docstring ini, ditulis sebelum satu angka dilihat.
Tanpa GPU. Latar: normalisasi per-kanal, Sec. IV-C makalah CENIM 2026.

MENGAPA UJI INI ADA
===================
P1/P2/P4/P5 semuanya berbentuk "alternatif lawan yang sekarang", dan yang sekarang menang
tiap kali. Itu MEMATAHKAN PESAING, bukan MENEMUKAN YANG TERBAIK. Untuk normalisasi bahkan
lebih tipis lagi: dari tujuh kandidat yang masuk akal, hanya dua yang pernah disentuh, dan
satu di antaranya (by_sample=False) sudah terpasang di dann_factory.py tanpa pernah
dijalankan sekali pun.

APA YANG DIUKUR
===============
Probe linear, GroupKFold PER REKAMAN, resep identik audit_a6_resolvabilitas. Dua sumbu:
  N-P1  akurasi KELAS di torsi-0   -> yang ingin dinaikkan
  N-P2  akurasi TORSI              -> yang TIDAK boleh ikut naik
Kandidat yang menaikkan keduanya hanya menaikkan SNR umum, bukan memisahkan kelas dari domain.

Fitur = spektrum log-magnitudo per kanal. Dipilih karena tak berubah terhadap geseran waktu
(probe linear pada deret waktu mentah tidak bermakna) sekaligus tetap membawa efek penskalaan
antar-kanal, yaitu persis yang sedang diuji.

P6 — MENGAPA IA MENJAWAB APA YANG P5 TIDAK JAWAB
=================================================
P5 menguji SATU SKALAR per pita (prominensi 2sf) lewat jarak Mahalanobis. Model tidak pernah
melihat skalar itu; ia melihat seluruh bentuk gelombang envelope. P6 memberi probe spektrum
envelope PENUH dari satu pita lawan dua pita.

Validasi silang membuatnya mengendalikan diri sendiri: menambah blok fitur tak informatif
MENURUNKAN akurasi CV lewat varians, berbeda dari Mahalanobis yang hampir selalu naik saat
fitur ditambah. Karena itu P6 tidak butuh kendali umpan.

RAMBU
=====
Ini PENYARING untuk memilih kandidat GPU, bukan vonis. Probe linear buta terhadap struktur
non-linear, dan buta total terhadap dinamika optimisasi serta skala gradien -- yang untuk
normalisasi kemungkinan justru mekanisme utamanya.

Jalankan:
    ${PYTHON:-python} -u scripts/audit/audit_normalisasi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
NPZ = REPO / "data" / "processed" / "motor" / "motor_multiclass_w4096_s4096_ss_env.npz"
SPLIT = REPO / "data" / "processed" / "motor" / "motor_multiclass_w4096_s4096_ss_env_split_info.npz"

FS = 7600.0
CH_ENV = 5                       # kanal envelope = indeks terakhir (urutan pipeline)
PITA_1 = (40.0, 100.0)
PITA_2 = (110.0, 130.0)
EPS = 1e-8

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass


def garis(judul: str = "") -> None:
    print("=" * 96)
    if judul:
        print(judul)
        print("=" * 96)


# ---------------------------------------------------------------- kandidat normalisasi

def n_sekarang(X):
    """by_sample=True, by_var=False — satu mean/std per window, digabung lintas kanal."""
    mu = X.mean(axis=(1, 2), keepdims=True)
    sd = X.std(axis=(1, 2), keepdims=True)
    return (X - mu) / np.maximum(sd, EPS)


def n_byvar(X):
    """by_sample=True, by_var=True — per window DAN per kanal."""
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True)
    return (X - mu) / np.maximum(sd, EPS)


def n_global_var(X):
    """by_sample=False, by_var=True — statistik global per kanal.
    TERPASANG di dann_factory.py tetapi tidak pernah dijalankan."""
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True)
    return (X - mu) / np.maximum(sd, EPS)


def n_global_pooled(X):
    """by_sample=False, by_var=False — satu statistik global untuk semuanya."""
    return (X - X.mean()) / max(X.std(), EPS)


def n_robust(X):
    """Per window per kanal, median dan IQR. Tahan terhadap pencilan impulsif."""
    med = np.median(X, axis=2, keepdims=True)
    q1, q3 = np.percentile(X, [25, 75], axis=2, keepdims=True)
    return (X - med) / np.maximum((q3 - q1) / 1.349, EPS)


def n_log_env(X):
    """Seperti sekarang, tetapi kanal envelope dikompresi log lebih dulu.
    Envelope non-negatif dan berekor panjang; kompresi log menyetarakan rentang
    dinamisnya tanpa menyentuh lima kanal getaran."""
    Y = X.copy()
    e = Y[:, CH_ENV, :]
    Y[:, CH_ENV, :] = np.log1p(np.abs(e) / np.maximum(np.abs(e).mean(), EPS))
    return n_sekarang(Y)


def n_tanpa(X):
    """Kendali: tanpa normalisasi apa pun."""
    return X


KANDIDAT = {
    "sekarang (bs=T,bv=F)": n_sekarang,
    "by_var  (bs=T,bv=T)": n_byvar,
    "global per-kanal (bs=F,bv=T)": n_global_var,
    "global pooled (bs=F,bv=F)": n_global_pooled,
    "robust median/IQR": n_robust,
    "log pada kanal envelope": n_log_env,
    "TANPA normalisasi": n_tanpa,
}


# ---------------------------------------------------------------- fitur & probe

N_BIN = 64          # bin per kanal; 6 kanal -> 384 fitur


def _bin_log(spek: np.ndarray, n_bin: int = N_BIN) -> np.ndarray:
    """Rata-ratakan spektrum ke n_bin pita lebar-sama, lalu log.

    Tanpa binning fiturnya 6 x 2049 = 12.294 dimensi lawan ~1.650 sampel di torsi-0,
    yaitu p >> n: probe jadi lambat dan peringkatnya didominasi derau dimensi tinggi,
    bukan oleh normalisasi yang sedang diuji. Binning menjaga informasi bentuk dan
    skala antar-kanal, yang justru sumbu yang diuji.
    """
    n_f = spek.shape[-1]
    tepi = np.linspace(0, n_f, n_bin + 1).astype(int)
    out = np.stack([spek[..., tepi[i]:max(tepi[i + 1], tepi[i] + 1)].mean(axis=-1)
                    for i in range(n_bin)], axis=-1)
    return np.log(out + 1e-30)


def fitur_spektrum(X: np.ndarray) -> np.ndarray:
    """(N,C,T) -> (N, C*N_BIN) log-magnitudo terbin, tak berubah terhadap geseran waktu."""
    n = X.shape[2]
    spek = np.abs(np.fft.rfft(X.astype(np.float32) * np.hanning(n).astype(np.float32), axis=2))
    return _bin_log(spek).reshape(X.shape[0], -1).astype(np.float32)


PLAFON_PROBE = 0.95     # gerbang ruang gerak; lihat cek_ruang_gerak()


def cek_ruang_gerak(akurasi: float, nama: str) -> bool:
    """Gerbang plafon. Bila probe acuan sudah mentok, selisih antar kandidat tidak
    dapat dibaca dan sapuan HARUS dilaporkan sebagai gagal-ukur.

    Dipasang setelah versi pertama uji ini memakai kontras 5-kelas di torsi-0 dan
    mendapat 99,7-99,9% untuk SETIAP kandidat — plafon sempurna, nol ruang gerak.
    Itu kelas kesalahan yang sama dengan L-P1. A6 memakai kontras BINER
    rs vs r1b justru karena ia yang sulit (79,8%).
    """
    if not np.isfinite(akurasi):
        return False
    if akurasi > PLAFON_PROBE:
        print(f"\n  [GAGAL-UKUR] probe acuan {nama} = {akurasi:.1%} > {PLAFON_PROBE:.0%}.")
        print("  Ruang gerak habis; selisih antar kandidat TIDAK dapat dibaca.")
        return False
    return True


def probe(F: np.ndarray, lab: np.ndarray, grup: np.ndarray,
          kembalikan_lipatan: bool = False):
    """Regresi logistik, GroupKFold per rekaman, kelas diseimbangkan. Resep A6."""
    u, cnt = np.unique(lab, return_counts=True)
    if len(u) < 2 or cnt.min() < 5:
        return (np.nan, []) if kembalikan_lipatan else np.nan
    rng = np.random.default_rng(0)
    keep = np.concatenate([rng.choice(np.where(lab == k)[0], cnt.min(), replace=False)
                           for k in u])
    F, lab, grup = F[keep], lab[keep], grup[keep]
    if len(np.unique(grup)) < 5:
        return (np.nan, []) if kembalikan_lipatan else np.nan
    akur = []
    for tr, te in GroupKFold(n_splits=5).split(F, lab, groups=grup):
        # Standardisasi fitur DIPASANG DARI LIPATAN LATIH SAJA (tanpa itu lbfgs gagal
        # konvergen pada 2000 iterasi dan akurasinya tidak dapat dipercaya). Statistik
        # diambil hanya dari tr supaya tidak ada kebocoran ke te.
        mu, sd = F[tr].mean(axis=0), F[tr].std(axis=0)
        sd = np.maximum(sd, 1e-8)
        Ftr, Fte = (F[tr] - mu) / sd, (F[te] - mu) / sd
        clf = LogisticRegression(max_iter=5000)
        clf.fit(Ftr, lab[tr])
        if clf.n_iter_.max() >= 5000:
            print("      [PERHATIAN] lbfgs belum konvergen; akurasi lipatan ini rapuh.")
        akur.append(accuracy_score(lab[te], clf.predict(Fte)))
    return (float(np.mean(akur)), akur) if kembalikan_lipatan else float(np.mean(akur))


def main() -> None:
    garis("UJI N dan P6 — sapuan normalisasi + kanal envelope kedua. Kriteria N-P1, N-P2, P6 (terkunci di docstring)")

    z = np.load(NPZ)
    X = (z["X"] if "X" in z.files else z[z.files[0]]).astype(np.float32)
    y = z["y"]
    si = np.load(SPLIT)
    torque, rep = si["torque"].astype(int), si["repetition"].astype(int)
    rec = torque * 1000 + rep * 10 + y
    print(f"  X = {X.shape}   torsi = {sorted(set(torque.tolist()))}")

    # Kontras BINER rs vs r1b di torsi-0 — resep A6 yang sesungguhnya.
    # Versi pertama uji ini memakai 5 kelas dan mentok 99,7-99,9% untuk SETIAP
    # kandidat, yaitu gagal-ukur. Kontras biner inilah yang sulit (A6: 79,8%).
    m0 = (torque == 0) & np.isin(y, [0, 1])
    y_biner = (y == 1).astype(int)
    print(f"  window torsi-0, kontras BINER rs vs r1b = {m0.sum()}")
    print(f"  chance kelas biner = 50,0%   chance torsi 8-arah = 12,5%\n")

    # ------------------------------------------------------------ Uji N
    garis("N-P1 / N-P2 — akurasi probe per kandidat normalisasi")
    print(f"  {'kandidat':<32}{'KELAS @torsi-0':>16}{'TORSI (semua)':>16}{'selisih':>10}")
    print("  " + "-" * 74)
    hasil = {}
    for nama, fn in KANDIDAT.items():
        Xn = fn(X)
        F = fitur_spektrum(Xn)
        a_kelas = probe(F[m0], y_biner[m0], rec[m0])
        a_torsi = probe(F, torque, rec)
        hasil[nama] = (a_kelas, a_torsi)
        print(f"  {nama:<32}{a_kelas:>15.1%}{a_torsi:>16.1%}"
              f"{a_kelas - a_torsi:>+10.1%}", flush=True)
        del Xn, F

    base_k, base_t = hasil["sekarang (bs=T,bv=F)"]
    print()
    berdaya = cek_ruang_gerak(base_k, "kelas @torsi-0 (setelan sekarang)")
    garis("VONIS N — frontier, bukan pemenang tunggal")
    if not berdaya:
        print("  N-P1 dan N-P2 TIDAK menyimpulkan apa pun (gerbang ruang gerak).")
        print("  Angka di atas deskriptif belaka.\n")
    print(f"  setelan sekarang: kelas {base_k:.1%}, torsi {base_t:.1%}\n")
    layak = []
    for nama, (k, t) in hasil.items():
        if nama.startswith("sekarang"):
            continue
        naik_kelas = np.isfinite(k) and k > base_k
        torsi_aman = np.isfinite(t) and t <= base_t
        if naik_kelas and torsi_aman:
            layak.append(nama)
            print(f"  LAYAK GPU  {nama:<32} kelas {k:+.1%} torsi {t - base_t:+.1%}")
        elif naik_kelas:
            print(f"  SNR umum   {nama:<32} kelas {k - base_k:+.1%} TETAPI torsi {t - base_t:+.1%}")
    if not layak:
        print("  Tidak ada kandidat yang menaikkan kelas tanpa menaikkan torsi.")
    print()

    # ------------------------------------------------------------ P6
    garis("P6 — kanal envelope KEDUA (110-130 Hz), versi penuh bukan skalar")
    print("  Fitur = spektrum envelope. CV menghukum fitur tak informatif, jadi")
    print("  tidak butuh kendali umpan (beda dari P5 yang memakai Mahalanobis).\n")

    def env_pita(sig2d, lo, hi):
        sos = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="bandpass", output="sos")
        env = np.abs(hilbert(sosfiltfilt(sos, sig2d.mean(axis=1)), axis=-1))
        env = env - env.mean(axis=-1, keepdims=True)
        return _bin_log(np.abs(np.fft.rfft(env, axis=-1))).astype(np.float32)

    vib = X[m0][:, :5, :]              # lima kanal getaran, jalur pipeline (RERATA)
    F1 = env_pita(vib, *PITA_1)
    F2 = env_pita(vib, *PITA_2)
    y0, rec0 = y_biner[m0], rec[m0]

    a1, lip1 = probe(F1, y0, rec0, kembalikan_lipatan=True)
    a2, _ = probe(F2, y0, rec0, kembalikan_lipatan=True)
    a12, lip12 = probe(np.hstack([F1, F2]), y0, rec0, kembalikan_lipatan=True)
    print(f"  {'fitur':<34}{'akurasi kelas @torsi-0':>26}")
    print(f"  {'40-100 saja (pipeline)':<34}{a1:>25.1%}")
    print(f"  {'110-130 saja':<34}{a2:>25.1%}")
    print(f"  {'40-100 + 110-130':<34}{a12:>25.1%}")
    print()
    print()
    if lip1 and lip12 and len(lip1) == len(lip12):
        d = np.array(lip12) - np.array(lip1)
        print("  Selisih BERPASANGAN per lipatan (lipatan identik untuk kedua fitur):")
        print("    " + "  ".join(f"{v:+.1%}" for v in d))
        print(f"    rerata {d.mean():+.1%}   sd {d.std(ddof=1):.1%}   "
              f"lipatan positif {int((d > 0).sum())}/{len(d)}")
        konsisten = (d > 0).all()
        naik = np.isfinite(a12) and np.isfinite(a1) and a12 > a1 and konsisten
        print()
        if not konsisten and a12 > a1:
            print("  [PERHATIAN] rerata naik tetapi TIDAK di semua lipatan. Pada 5 lipatan")
            print("  dan selisih sekecil ini, itu tidak terbedakan dari derau pembagian.")
    else:
        naik = False
    print(f"  P6 {'LOLOS — pembawa kedua menambah, konsisten di semua lipatan' if naik else 'TIDAK MENYIMPULKAN — kenaikan tidak konsisten lintas lipatan'}"
          f"   ({a12 - a1:+.1%})")

    garis("RAMBU")
    print("  Ini PENYARING, bukan vonis. Probe linear buta terhadap struktur non-linear dan")
    print("  buta TOTAL terhadap dinamika optimisasi serta skala gradien, yang untuk")
    print("  normalisasi kemungkinan justru mekanisme utamanya.")
    print("  Tujuh kandidat pada satu metrik tetap sapuan kecil. 'Optimal' di luar jangkauan;")
    print("  yang dilaporkan adalah frontier dari ruang yang benar-benar disapu.")


if __name__ == "__main__":
    main()
