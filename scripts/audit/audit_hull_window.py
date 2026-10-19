"""audit_hull_window.py — R1: apakah M13 (convex hull) bertahan di granularitas window?

Kriteria R1-P0..R1-P3 TERKUNCI di docstring ini, ditulis sebelum satu angka pun dilihat.
Skrip ini hanya mengeksekusinya. Hasilnya menopang perbandingan convex hull di Sec. V-A
makalah CENIM 2026 (perbandingan itu post-hoc, dan dinyatakan begitu di sana).

MENGAPA SKRIP INI ADA
=====================
M13 — "torsi-0 53,1% di luar convex hull sumber, torsi-7 18,2%" — adalah SELURUH premis
Uji B, dan salah satu dari dua penjelasan teoretis yang tersisa. Angka itu dikutip
di lima berkas dan tidak diturunkan di satu pun; tidak ada skrip yang menghitungnya.

U3 dibatalkan persis karena premisnya berdiri di rekaman penuh (|d| = 2,18) dan
BERBALIK di tingkat window (0,32 vs 0,35). Uji B mengaugmentasi window per window, bukan
distribusi, jadi yang harus berdiri adalah pernyataan tingkat window.

APA YANG DIUKUR
===============
Deskriptor: sigma per window per kanal (std lintas waktu) — tepat besaran yang transform
Uji B manipulasi.

  AGREGAT  1 titik per torsi        hull = [min, maks] atas torsi 1-6
  WINDOW   1 titik per window       hull = [q01, q99] atas seluruh window sumber

Statistik utama sama di keduanya sehingga langsung dibandingkan:

    ekstrapolasi% = jarak keluar hull / lebar hull x 100

Jalankan:
    ${PYTHON:-python} -u scripts/audit/audit_hull_window.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
NPZ = REPO / "data" / "processed" / "motor" / "motor_multiclass_w16384_s4096_ss_env.npz"
SPLIT = REPO / "data" / "processed" / "motor" / "motor_multiclass_w16384_s4096_ss_env_split_info.npz"
OUT = REPO / "results" / "motor" / "audit_hull_window.npz"

# Urutan kanal dari meta json dataset — JANGAN diubah tanpa memeriksa ulang meta.
KANAL = ["Vib_acpe", "Vib_acpi", "Vib_axial", "Vib_base", "Vib_carc", "env_hilbert"]
KELAS = {0: "rs", 1: "r1b", 2: "r2b", 3: "r3b", 4: "r4b"}

TORSI_SUMBER = [1, 2, 3, 4, 5, 6]
TORSI_TARGET = [0, 7]

# Hanya kanal bertren linear terhadap beban yang dipakai. Nama KANAL_UJI_B berasal dari
# uji augmentasi yang tidak dirilis.
KANAL_UJI_B = ["Vib_base", "Vib_carc", "Vib_axial", "Vib_acpe"]

CHUNK = 500  # window per potongan; menjaga puncak memori ~200 MB di atas X

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass


def garis(judul: str = "") -> None:
    print("=" * 100)
    if judul:
        print(judul)
        print("=" * 100)


def sigma_per_window(X: np.ndarray) -> np.ndarray:
    """std lintas waktu, (N, C). Dipotong agar tidak mengalokasikan salinan penuh."""
    n, c = X.shape[0], X.shape[1]
    out = np.empty((n, c), np.float64)
    for i in range(0, n, CHUNK):
        out[i : i + CHUNK] = X[i : i + CHUNK].astype(np.float64).std(axis=2)
    return out


def ekstrapolasi(nilai: float, lo: float, hi: float) -> float:
    """Jarak keluar hull [lo, hi] sebagai % lebar hull. 0 bila di dalam."""
    lebar = hi - lo
    if lebar <= 0:
        return float("nan")
    if nilai < lo:
        return (lo - nilai) / lebar * 100.0
    if nilai > hi:
        return (nilai - hi) / lebar * 100.0
    return 0.0


def main() -> None:
    if not NPZ.exists():
        raise SystemExit(f"[ABORT] {NPZ.name} tidak ada.")

    garis("R1 — M13 DI GRANULARITAS WINDOW. Kriteria R1-P0..P3 (terkunci di docstring)")
    print(f"  dataset  : {NPZ.name}")
    print(f"  kriteria : R1-P0 reproduksi | R1-P1 GERBANG | R1-P2 arah | R1-P3 pelengkap\n")

    z = np.load(NPZ)
    xkey = "X" if "X" in z.files else z.files[0]
    ykey = "y" if "y" in z.files else None
    print(f"  memuat {xkey} ... (2,9 GB terkompresi)", flush=True)
    X = z[xkey]
    y = z[ykey] if ykey else None
    torque = np.load(SPLIT)["torque"].astype(int)

    C = X.shape[1]
    nama = KANAL[:C]
    print(f"  X = {X.shape}   torsi unik = {sorted(set(torque.tolist()))}\n")

    print("  menghitung sigma per window ...", flush=True)
    sw = sigma_per_window(X)
    del X, z  # 4,7 GB dibebaskan sebelum bagian LP

    src = np.isin(torque, TORSI_SUMBER)
    uniq = sorted(set(torque.tolist()))

    # ================================================================= 1. AGREGAT
    garis("1. AGREGAT — sigma_c(t) satu titik per torsi (dugaan resep M13)")
    sig_ag = np.zeros((len(uniq), C))
    for i, t in enumerate(uniq):
        sig_ag[i] = sw[torque == t].mean(axis=0)

    print(f"  {'torsi':>6} " + " ".join(f"{n:>11}" for n in nama))
    for i, t in enumerate(uniq):
        tandai = "  <- TARGET" if t in TORSI_TARGET else ""
        print(f"  {t:>6} " + " ".join(f"{v:>11.4f}" for v in sig_ag[i]) + tandai)

    idx_src = [uniq.index(t) for t in TORSI_SUMBER]
    lo_ag, hi_ag = sig_ag[idx_src].min(axis=0), sig_ag[idx_src].max(axis=0)

    print(f"\n  hull sumber (torsi 1-6), per kanal:")
    print(f"  {'kanal':>12} {'lo':>11} {'hi':>11} {'lebar':>11} "
          f"{'torsi-0 %':>11} {'torsi-7 %':>11}")
    ek_ag = {}
    for c in range(C):
        e0 = ekstrapolasi(sig_ag[uniq.index(0), c], lo_ag[c], hi_ag[c])
        e7 = ekstrapolasi(sig_ag[uniq.index(7), c], lo_ag[c], hi_ag[c])
        ek_ag[nama[c]] = (e0, e7)
        print(f"  {nama[c]:>12} {lo_ag[c]:>11.4f} {hi_ag[c]:>11.4f} "
              f"{hi_ag[c] - lo_ag[c]:>11.4f} {e0:>10.1f}% {e7:>10.1f}%")

    m_all0 = float(np.mean([ek_ag[n][0] for n in nama]))
    m_all7 = float(np.mean([ek_ag[n][1] for n in nama]))
    m_b0 = float(np.mean([ek_ag[n][0] for n in KANAL_UJI_B]))
    m_b7 = float(np.mean([ek_ag[n][1] for n in KANAL_UJI_B]))
    print(f"\n  rerata 6 kanal        : torsi-0 {m_all0:5.1f}%   torsi-7 {m_all7:5.1f}%")
    print(f"  rerata 4 kanal Uji B  : torsi-0 {m_b0:5.1f}%   torsi-7 {m_b7:5.1f}%")

    # 2sf sebagai pembanding — hukum slip terukur, linear dalam T (Nm)
    nm = np.array([(t + 1) * 0.5 for t in uniq])
    slip = 0.0099 + 0.0097 * nm
    lo_s, hi_s = slip[idx_src].min(), slip[idx_src].max()
    e0_s = ekstrapolasi(slip[uniq.index(0)], lo_s, hi_s)
    e7_s = ekstrapolasi(slip[uniq.index(7)], lo_s, hi_s)
    print(f"  sumbu 2sf (slip law)  : torsi-0 {e0_s:5.1f}%   torsi-7 {e7_s:5.1f}%   "
          f"(simetris menurut konstruksi)")

    print(f"\n  KLAIM M13             : torsi-0  53,1%   torsi-7  18,2%")
    kand = {"6 kanal": (m_all0, m_all7), "4 kanal Uji B": (m_b0, m_b7),
            "2sf": (e0_s, e7_s)}
    cocok = [k for k, (a, b) in kand.items() if abs(a - 53.1) <= 3 and abs(b - 18.2) <= 3]
    p0 = bool(cocok)
    print(f"  R1-P0 (±3 poin)       : {'LOLOS via ' + ', '.join(cocok) if p0 else 'GUGUR'}"
          f"{'' if p0 else ' — tidak ada resep yang mereproduksi 53,1/18,2'}")

    # ================================================================= 2. WINDOW
    garis("2. WINDOW — satu titik per window, hull [q01, q99] atas window sumber")
    ssrc = sw[src]
    print(f"  window sumber = {ssrc.shape[0]}   window torsi-0 = {int((torque == 0).sum())}"
          f"   window torsi-7 = {int((torque == 7).sum())}\n")

    q01, q99 = np.percentile(ssrc, 1, axis=0), np.percentile(ssrc, 99, axis=0)
    mn, mx = ssrc.min(axis=0), ssrc.max(axis=0)

    print(f"  {'kanal':>12} {'q01':>9} {'q99':>9} | {'med t0':>9} {'ekst%':>7} {'di luar':>8}"
          f" | {'med t7':>9} {'ekst%':>7} {'di luar':>8} | {'minmax t0':>10}")
    frac0, frac7, ek_w = {}, {}, {}
    for c in range(C):
        s0, s7 = sw[torque == 0, c], sw[torque == 7, c]
        f0 = float(((s0 < q01[c]) | (s0 > q99[c])).mean())
        f7 = float(((s7 < q01[c]) | (s7 > q99[c])).mean())
        f0mm = float(((s0 < mn[c]) | (s0 > mx[c])).mean())
        e0 = ekstrapolasi(float(np.median(s0)), q01[c], q99[c])
        e7 = ekstrapolasi(float(np.median(s7)), q01[c], q99[c])
        frac0[nama[c]], frac7[nama[c]], ek_w[nama[c]] = f0, f7, (e0, e7)
        bintang = " *" if nama[c] in KANAL_UJI_B else "  "
        print(f"  {nama[c]:>12}{bintang}{q01[c]:>7.3f} {q99[c]:>9.3f} | "
              f"{np.median(s0):>9.3f} {e0:>6.1f}% {f0:>7.1%} | "
              f"{np.median(s7):>9.3f} {e7:>6.1f}% {f7:>7.1%} | {f0mm:>9.1%}")
    print("  * = kanal yang akan disentuh Uji B (rambu B.5 no. 4)")

    # ---- R1-P1
    garis("R1-P1 — GERBANG UTAMA")
    lolos_kanal = [n for n in KANAL_UJI_B if frac0[n] >= 0.50]
    print(f"  ambang: >= 50% window torsi-0 di luar [q01, q99] sumber, pada >= 3 dari 4 kanal\n")
    for n in KANAL_UJI_B:
        print(f"    {n:>12}  {frac0[n]:>6.1%}   {'LOLOS' if frac0[n] >= 0.50 else 'gugur'}")
    p1 = len(lolos_kanal) >= 3
    print(f"\n  {len(lolos_kanal)}/4 kanal lolos  ->  R1-P1 {'LOLOS' if p1 else 'GUGUR'}")
    print(f"  {'U2 boleh dijalankan (gerbang terbuka)' if p1 else 'U2 DIBLOKIR — premis Uji B artefak granularitas, seperti U3'}")

    # ---- R1-P2
    garis("R1-P2 — arah asimetri (torsi-0 > torsi-7)")
    a_ag = m_b0 > m_b7
    e_w0 = float(np.mean([ek_w[n][0] for n in KANAL_UJI_B]))
    e_w7 = float(np.mean([ek_w[n][1] for n in KANAL_UJI_B]))
    f_w0 = float(np.mean([frac0[n] for n in KANAL_UJI_B]))
    f_w7 = float(np.mean([frac7[n] for n in KANAL_UJI_B]))
    print(f"  agregat  : torsi-0 {m_b0:5.1f}%  vs torsi-7 {m_b7:5.1f}%   -> {'t0 > t7' if a_ag else 'TERBALIK'}")
    print(f"  window   : torsi-0 {e_w0:5.1f}%  vs torsi-7 {e_w7:5.1f}%   -> {'t0 > t7' if e_w0 > e_w7 else 'TERBALIK'}")
    print(f"  di luar  : torsi-0 {f_w0:5.1%}  vs torsi-7 {f_w7:5.1%}   -> {'t0 > t7' if f_w0 > f_w7 else 'TERBALIK'}")
    p2 = e_w0 > e_w7
    print(f"\n  R1-P2 {'LOLOS' if p2 else 'GUGUR — B-P2 wajib ditulis ulang sebelum U2'}")

    # ---- per kelas (rambu 26.5)
    if y is not None:
        garis("Pecahan per KELAS — agar tidak ada yang tersembunyi di rerata")
        print(f"  {'kelas':>6} " + " ".join(f"{n:>11}" for n in KANAL_UJI_B))
        for k in sorted(KELAS):
            m0 = (torque == 0) & (y == k)
            if not m0.any():
                continue
            baris = []
            for n in KANAL_UJI_B:
                c = nama.index(n)
                s0 = sw[m0, c]
                baris.append(f"{((s0 < q01[c]) | (s0 > q99[c])).mean():>10.1%}")
            print(f"  {KELAS[k]:>6} " + " ".join(baris))

    # ================================================================= 3. MULTIVARIAT
    garis("R1-P3 — convex hull 6-D (pelengkap, BUKAN gerbang)")
    try:
        from scipy.optimize import linprog

        rng = np.random.default_rng(0)
        isrc = rng.choice(np.where(src)[0], size=min(1500, int(src.sum())), replace=False)
        P = sw[isrc].T  # (C, M)
        A_eq = np.vstack([P, np.ones((1, P.shape[1]))])
        cobj = np.zeros(P.shape[1])

        for t in TORSI_TARGET:
            it = np.where(torque == t)[0]
            it = rng.choice(it, size=min(150, len(it)), replace=False)
            di_luar = 0
            for i in it:
                b = np.concatenate([sw[i], [1.0]])
                r = linprog(cobj, A_eq=A_eq, b_eq=b, bounds=(0, None), method="highs")
                di_luar += int(not r.success)
            print(f"  torsi-{t}: {di_luar}/{len(it)} = {di_luar / len(it):.1%} window "
                  f"DI LUAR conv(1500 window sumber)")
        print("  (subsampel 1.500 sumber x 150 target; hull subsampel <= hull penuh, "
              "jadi angka ini batas ATAS)")
    except ImportError:
        print("  [LEWATI] scipy tidak tersedia")

    # ================================================================= vonis
    garis("VONIS R1")
    print(f"  R1-P0 reproduksi   : {'LOLOS' if p0 else 'GUGUR'}")
    print(f"  R1-P1 GERBANG      : {'LOLOS' if p1 else 'GUGUR'}")
    print(f"  R1-P2 arah         : {'LOLOS' if p2 else 'GUGUR'}")
    print()
    if p1:
        print("  -> FASE 1 (P1 implementasi Uji B + P2 sanity check) boleh dimulai.")
    else:
        print("  -> U2 DIBLOKIR. Uji B dirancang ulang atau dicoret sebelum GPU dipakai.")
    if not p0:
        print("  -> M13 diganti dengan nilai terukur di SELURUH dokumen (5 berkas).")
    if not p2:
        print("  -> B-P2 ditulis ulang; kutipan '53,1% vs 18,2%' dicabut.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, sigma_window=sw, torque=torque, sigma_agregat=sig_ag,
             q01=q01, q99=q99, hull_lo_agregat=lo_ag, hull_hi_agregat=hi_ag,
             kanal=np.array(nama))
    print(f"\n  disimpan: {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
