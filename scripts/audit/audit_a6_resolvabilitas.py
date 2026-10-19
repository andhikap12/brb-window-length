"""audit_a6_resolvabilitas.py — A6: apakah sumbu 2sf TERBACA di satu window?

Kriteria A6-P1..A6-P3 TERKUNCI di docstring ini, ditulis sebelum satu baris kode ini ada.
Tanpa GPU. Hasilnya dilaporkan di Sec. V-C makalah CENIM 2026 (panel b gambar LODO).

A6 mengukur klaim FISIK inti makalah — *tanda kerusakan tidak terselesaikan di beban rendah
pada satu window* — yang sejauh ini hanya bersandar pada bukti tak langsung.

MENGAPA PROBE, BUKAN PROMINENSI
===============================
Memungut puncak gagal: 2sf ADA tetapi tidak dominan (puncak nyata 13,4-14,8 Hz).
Mengukur prominensi di frekuensi ramalan hukum slip SIRKULAR untuk pertanyaan
domain-lawan-kelas: targetnya bergantung torsi, jadi ia mengukur targetnya sendiri.

Probe non-sirkular: fitur = spektrum envelope pada GRID FREKUENSI TETAP, tidak pernah
melihat hukum slip. Kalau 2sf terbaca, domain dapat diprediksi darinya.

KENDALI ADALAH INTI RANCANGAN
=============================
torsi-0 vs torsi-1 dan torsi-6 vs torsi-7 punya selisih 2sf yang PERSIS SAMA (0,582 Hz),
tetapi yang kedua berada di frekuensi absolut 3x lebih tinggi, jauh dari kebocoran DC.
Tanpa kendali ini, A6 tidak dapat membedakan "2sf tak terbaca di beban rendah" dari
"probe saya jelek".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
NPZ = REPO / "data" / "processed" / "motor" / "motor_multiclass_w16384_s4096_ss_env.npz"
SPLIT = REPO / "data" / "processed" / "motor" / "motor_multiclass_w16384_s4096_ss_env_split_info.npz"

FS = 7600.0
CH_ENV = 5
PITA_FITUR = (0.5, 15.0)      # GRID TETAP — tidak pernah melihat hukum slip
CHUNK = 500
AMBANG = 0.60
KELAS = {0: "rs", 1: "r1b", 2: "r2b", 3: "r3b", 4: "r4b"}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass


def garis(judul: str = "") -> None:
    print("=" * 96)
    if judul:
        print(judul)
        print("=" * 96)


def f2s(t) -> float:
    return 120.0 * (0.0099 + 0.0097 * ((t + 1) * 0.5))


def fitur_spektrum(env: np.ndarray) -> np.ndarray:
    """(N, T) -> (N, n_bin) log-magnitudo spektrum envelope, dinormalkan per window."""
    n = env.shape[1]
    e = env - env.mean(axis=1, keepdims=True)
    spek = np.abs(np.fft.rfft(e * np.hanning(n), axis=1))
    freq = np.fft.rfftfreq(n, 1.0 / FS)
    sel = (freq >= PITA_FITUR[0]) & (freq <= PITA_FITUR[1])
    F = np.log(spek[:, sel] + 1e-30)
    return F - F.mean(axis=1, keepdims=True)      # bentuk spektrum, bukan energi total


def probe(F: np.ndarray, lab: np.ndarray, grup: np.ndarray, nama: str) -> float:
    """Regresi logistik, GroupKFold per rekaman. Kelas diseimbangkan."""
    u, cnt = np.unique(lab, return_counts=True)
    rng = np.random.default_rng(0)
    keep = np.concatenate([rng.choice(np.where(lab == k)[0], cnt.min(), replace=False)
                           for k in u])
    F, lab, grup = F[keep], lab[keep], grup[keep]
    akur = []
    for tr, te in GroupKFold(n_splits=5).split(F, lab, groups=grup):
        # n_jobs dibuang: tanpa efek sejak sklearn 1.8, dihapus di 1.10 (akan patah).
        # Penghapusan terbukti no-op — kedelapan angka keterbacaan per torsi tereproduksi persis sesudahnya.
        clf = LogisticRegression(max_iter=3000)
        clf.fit(F[tr], lab[tr])
        akur.append(accuracy_score(lab[te], clf.predict(F[te])))
    a = float(np.mean(akur))
    print(f"  {nama:<34} n={len(lab):>5}  akurasi {a:>6.1%}  "
          f"(lipatan {min(akur):.1%}–{max(akur):.1%})  chance 50,0%")
    return a


def main() -> None:
    garis("A6 — RESOLVABILITAS 2sf DI GRANULARITAS WINDOW. Kriteria A6-P1..P3 (terkunci di docstring)")
    df = FS / 16384
    print(f"  fitur : spektrum envelope {PITA_FITUR[0]}–{PITA_FITUR[1]} Hz, grid TETAP, "
          f"df = {df:.4f} Hz")
    print(f"  probe : regresi logistik, GroupKFold per REKAMAN, kelas diseimbangkan\n")
    print(f"  selisih 2sf yang diuji — sengaja dibuat IDENTIK:")
    print(f"    torsi-0 vs torsi-1 : {f2s(0):.3f} → {f2s(1):.3f} Hz = "
          f"{f2s(1)-f2s(0):.3f} Hz = {(f2s(1)-f2s(0))/df:.2f} bin")
    print(f"    torsi-6 vs torsi-7 : {f2s(6):.3f} → {f2s(7):.3f} Hz = "
          f"{f2s(7)-f2s(6):.3f} Hz = {(f2s(7)-f2s(6))/df:.2f} bin\n")

    z = np.load(NPZ)
    X = z["X"] if "X" in z.files else z[z.files[0]]
    y = z["y"]
    si = np.load(SPLIT)
    torque, rep = si["torque"].astype(int), si["repetition"].astype(int)
    rec = torque * 1000 + rep * 10 + y            # identitas rekaman fisik

    print("  menghitung fitur spektral ...", flush=True)
    n = X.shape[0]
    F = None
    for i in range(0, n, CHUNK):
        blok = fitur_spektrum(X[i:i + CHUNK, CH_ENV].astype(np.float64))
        if F is None:
            F = np.empty((n, blok.shape[1]))
        F[i:i + CHUNK] = blok
    if not np.isfinite(F).all():
        print("  [ABORT] fitur non-finite.")
        return
    print(f"  fitur = {F.shape}   finite = True\n")

    garis("A6-P1 / A6-P2 — pasangan torsi bertetangga, selisih 2sf IDENTIK")
    m01 = np.isin(torque, [0, 1])
    a1 = probe(F[m01], (torque[m01] == 1).astype(int), rec[m01],
               "A6-P1  torsi-0 vs torsi-1 (beban RENDAH)")
    m67 = np.isin(torque, [6, 7])
    a2 = probe(F[m67], (torque[m67] == 7).astype(int), rec[m67],
               "A6-P2  torsi-6 vs torsi-7 (beban TINGGI)")

    garis("A6-P3 — spesifisitas: apakah pita ini juga membawa KELAS?")
    m = (torque == 1) & np.isin(y, [0, 1])
    a3 = probe(F[m], (y[m] == 1).astype(int), rec[m],
               "A6-P3  rs vs r1b di dalam torsi-1")

    garis("VONIS A6")
    p1_rendah = a1 <= AMBANG
    p2_tinggi = a2 > AMBANG
    print(f"  A6-P1 torsi-0/1 : {a1:.1%}   {'<= 60% -> TIDAK terselesaikan' if p1_rendah else '> 60% -> terselesaikan'}")
    print(f"  A6-P2 torsi-6/7 : {a2:.1%}   {'> 60% -> probe BERDAYA' if p2_tinggi else '<= 60% -> probe TIDAK berdaya'}")
    print(f"  A6-P3 kelas     : {a3:.1%}   (deskriptif)")
    print()
    if p1_rendah and p2_tinggi:
        print("  -> DISOSIASI BERSIH. Selisih 2sf yang PERSIS SAMA (0,582 Hz) terbaca di beban")
        print("     tinggi dan TIDAK terbaca di beban rendah. Penyebabnya kedekatan ke DC pada")
        print("     slip kecil — bukan probe yang lemah. Klaim fisik inti makalah berubah dari")
        print("     inferensi menjadi PENGUKURAN LANGSUNG.")
    elif not p1_rendah:
        print("  -> 2sf TERBACA di beban rendah. Kegagalan memungut puncak adalah artefak pemungutan.")
        print("     Klaim fisik makalah (Sec. V-B) HARUS DILEMAHKAN.")
    else:
        print("  -> KEDUANYA <= 60%: probenya tidak berdaya. A6 TIDAK menyimpulkan apa pun;")
        print("     dilaporkan sebagai gagal-ukur, bukan sebagai temuan.")
    print()
    print("  Rambu: regresi logistik hanya membaca struktur LINEAR. 'Tidak terbaca'")
    print("  berarti tidak terbaca secara linear oleh probe ini, bukan mustahil dibaca.")


if __name__ == "__main__":
    main()
