"""audit_r3_faktorial.py — R3: memisahkan PANJANG JENDELA dari JUMLAH WINDOW.

Kriteria R3-P1..R3-P4 TERKUNCI di docstring ini, ditulis sebelum satu angka lengan dibaca.
Hasilnya adalah Tabel II dan Sec. IV-B makalah CENIM 2026. Skrip ini hanya mengeksekusinya.

INTI PERSOALAN
==============
Klaim positif satu-satunya proyek ini (+13,16 di torsi-0) diukur di w16384-ovl. Uji C
sudah memisahkan efek data dari efek resolusi, tetapi di **F1 tingkat window** — dan window
16384 memuat 4x lebih banyak bukti PER KEPUTUSAN daripada window 4096. Sebagian dari
"+18,04 efek resolusi" karena itu mungkin bukan generalisasi yang lebih baik melainkan
tugas yang dibuat lebih mudah.

Keempat lengan mengklasifikasi 100 REKAMAN FISIK YANG SAMA (diverifikasi 19:30). Karena itu
Rec-Accuracy sebanding lintas panjang jendela; Win-F1 tidak. R3 mengulang dekomposisinya di
satuan yang sebanding.

                    w4096 (res rendah)      w16384 (res tinggi)
  ~3.200 window     W4096SUB                W16384
  ~12-13k window    FP32_tsit_ext           W16384OVL
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results" / "motor"

# nama lengan -> (berkas, kunci di dalamnya)
LENGAN = {
    "w4096-penuh": (RES / "audit_ext_family_rec.npz", "ERM_(baseline)"),
    "w4096-sub":   (RES / "r3_w4096sub.npz",  "KONTROL:_ERM_w4096-sub"),
    "w16384":      (RES / "r3_w16384.npz",    "UJI_C:_ERM_w16384"),
    "w16384-ovl":  (RES / "r3_w16384ovl.npz", "UJI_E:_ERM_w16384-ovl"),
}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass


def garis(judul: str = "") -> None:
    print("=" * 100)
    if judul:
        print(judul)
        print("=" * 100)


def muat() -> dict[str, np.ndarray]:
    out = {}
    for nama, (fp, kunci) in LENGAN.items():
        if not fp.exists():
            raise SystemExit(f"[ABORT] {fp.name} belum ada.")
        z = np.load(fp, allow_pickle=True)
        if kunci not in z.files:
            cocok = [k for k in z.files if k.replace("_", " ") == kunci.replace("_", " ")]
            if not cocok:
                raise SystemExit(f"[ABORT] kunci '{kunci}' tidak ada di {fp.name}. "
                                 f"Tersedia: {list(z.files)}")
            kunci = cocok[0]
        out[nama] = z[kunci]
    return out


def sejajarkan(D: dict[str, np.ndarray]) -> tuple[list[int], dict[str, np.ndarray]]:
    """Ambil hanya seed yang ada di KEEMPAT lengan; urutkan sama."""
    sets = [set(A[:, 0].astype(int).tolist()) for A in D.values()]
    seeds = sorted(set.intersection(*sets))
    out = {}
    for nama, A in D.items():
        idx = {int(s): i for i, s in enumerate(A[:, 0])}
        out[nama] = A[[idx[s] for s in seeds]]
    return seeds, out


def lapor(nama: str, d: np.ndarray) -> str:
    n_pos = int((d > 0).sum())
    p_t = ttest_rel(d, np.zeros_like(d)).pvalue if d.std() > 0 else 1.0
    try:
        p_w = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p_w = float("nan")
    return (f"  {nama:<34} {d.mean():>+8.2f} ± {d.std(ddof=1):>5.2f}   "
            f"{n_pos}/{len(d)}   t p={p_t:.4f}  W p={p_w:.4f}")


def main() -> None:
    D0 = muat()
    seeds, D = sejajarkan(D0)
    ncol = D["w4096-penuh"].shape[1]
    I_F1_T0, I_REC_T0 = 2, ncol - 2

    garis("R3 — FAKTORIAL 2x2: PANJANG JENDELA vs JUMLAH WINDOW. Kriteria R3-P1..P4 (terkunci di docstring)")
    print(f"  seed berpasangan di keempat lengan : {seeds}")
    print(f"  metrik utama : Rec-Acc torsi-0 (kolom {I_REC_T0}) — 100 rekaman, 50 per torsi")
    print(f"  pembanding   : Win-F1 torsi-0 (kolom {I_F1_T0}) — TIDAK sebanding lintas kolom\n")

    garis("1. Keempat sel — rerata ± sd atas seed berpasangan")
    print(f"  {'lengan':<14}{'window':>8}  {'Rec-Acc t0':>16} {'Rec-Acc tot':>16} "
          f"{'Win-F1 t0':>16}")
    n_win = {"w4096-penuh": 13200, "w4096-sub": 3200, "w16384": 3200, "w16384-ovl": 12000}
    for nama in ["w4096-sub", "w16384", "w4096-penuh", "w16384-ovl"]:
        A = D[nama]
        print(f"  {nama:<14}{n_win[nama]:>8}  "
              f"{A[:, I_REC_T0].mean():>9.2f}±{A[:, I_REC_T0].std(ddof=1):<6.2f} "
              f"{A[:, ncol-3].mean():>9.2f}±{A[:, ncol-3].std(ddof=1):<6.2f} "
              f"{A[:, I_F1_T0].mean():>9.2f}±{A[:, I_F1_T0].std(ddof=1):<6.2f}")

    r = {k: D[k][:, I_REC_T0] for k in D}
    w = {k: D[k][:, I_F1_T0] for k in D}

    # ------------------------------------------------------------------ R3-P1
    garis("R3-P1 — UTAMA. Efek panjang jendela pada satuan yang sebanding (Rec-Acc torsi-0)")
    d1 = r["w16384-ovl"] - r["w4096-penuh"]
    print(f"  {'':34} {'Δ rerata':>8}   {'menang':>7}")
    print(lapor("w16384-ovl − w4096-penuh", d1))
    p1 = int((d1 > 0).sum()) >= 6
    print(f"\n  ambang: positif pada >= 6 dari {len(d1)} seed")
    print(f"  R3-P1 {'LOLOS' if p1 else 'GUGUR'}"
          + ("" if p1 else " — klaim +13,16 tidak selamat di satuan keputusan yang sama; "
                          "wajib dinyatakan ulang sebagai efek tingkat-window"))

    # ------------------------------------------------------------------ R3-P2
    garis("R3-P2 — ATRIBUSI. Panjang jendela, atau sekadar volume data?")
    pj_lo = r["w16384"] - r["w4096-sub"]          # panjang, pada jumlah rendah
    pj_hi = r["w16384-ovl"] - r["w4096-penuh"]    # panjang, pada jumlah tinggi
    jm_lo = r["w4096-penuh"] - r["w4096-sub"]     # jumlah, pada w4096
    jm_hi = r["w16384-ovl"] - r["w16384"]         # jumlah, pada w16384
    print(lapor("PANJANG @ 3.200 window", pj_lo))
    print(lapor("PANJANG @ ~12k window", pj_hi))
    print(lapor("JUMLAH  @ w4096", jm_lo))
    print(lapor("JUMLAH  @ w16384", jm_hi))
    e_pj = 0.5 * (pj_lo.mean() + pj_hi.mean())
    e_jm = 0.5 * (jm_lo.mean() + jm_hi.mean())
    print(f"\n  efek PANJANG rerata = {e_pj:+.2f}")
    print(f"  efek JUMLAH  rerata = {e_jm:+.2f}")
    p2 = e_pj > e_jm
    print(f"  R3-P2 {'LOLOS' if p2 else 'GUGUR'}"
          + ("" if p2 else " — keunggulan w16384-ovl sebagian besar milik VOLUME DATA, "
                           "yang dapat diperoleh lebih murah dengan menurunkan stride di w4096"))

    # ------------------------------------------------------------------ R3-P3
    garis("R3-P3 — INFLASI SATUAN. Efek yang sama, dua satuan (deskriptif)")
    wpj_lo = w["w16384"] - w["w4096-sub"]
    wpj_hi = w["w16384-ovl"] - w["w4096-penuh"]
    print(f"  {'efek':<26} {'Win-F1':>10} {'Rec-Acc':>10} {'inflasi':>10}")
    for nm, dw, dr in [("PANJANG @ 3.200", wpj_lo, pj_lo),
                       ("PANJANG @ ~12k", wpj_hi, pj_hi)]:
        print(f"  {nm:<26} {dw.mean():>+10.2f} {dr.mean():>+10.2f} "
              f"{dw.mean() - dr.mean():>+10.2f}")
    print("\n  'inflasi' = berapa poin efek panjang jendela TAMPAK lebih besar bila diukur")
    print("  per window daripada per rekaman. Window 16384 memuat 4x bukti per keputusan;")
    print("  selisih ini adalah bagian yang milik satuan, bukan milik generalisasi.")
    print(f"\n  Pembanding Uji C (tingkat window, tercatat): efek RESOLUSI +18,04")
    print(f"  R3 mengukur ulang efek yang sama @3.200 window: Win-F1 {wpj_lo.mean():+.2f}")

    # ------------------------------------------------------------------ R3-P4
    garis("R3-P4 — INTERAKSI (deskriptif). Aditivitas yang Uji C asumsikan")
    inter = pj_hi - pj_lo
    print(lapor("interaksi (panjang@tinggi − panjang@rendah)", inter))
    print(f"\n  aditif bila ~0. Uji C menjumlahkan −11,05 dan +18,04 dengan asumsi ini.")

    # ------------------------------------------------------------------ vonis
    garis("VONIS R3")
    print(f"  R3-P1 UTAMA    : {'LOLOS' if p1 else 'GUGUR'}   "
          f"(Δ {d1.mean():+.2f}, {int((d1>0).sum())}/{len(d1)} seed)")
    print(f"  R3-P2 ATRIBUSI : {'LOLOS' if p2 else 'GUGUR'}   "
          f"(panjang {e_pj:+.2f} vs jumlah {e_jm:+.2f})")
    print(f"  R3-P3, R3-P4   : deskriptif, di atas")
    print()
    if p1 and p2:
        print("  -> Klaim +13,16 SELAMAT: efeknya milik panjang jendela dan bertahan di")
        print("     satuan keputusan yang sebanding. Gerbang bab hasil terbuka penuh.")
    elif p1:
        print("  -> Efek bertahan tetapi ATRIBUSINYA berubah; klaim ditulis ulang isinya.")
    else:
        print("  -> Klaim wajib dinyatakan ulang sebagai efek tingkat-window.")


if __name__ == "__main__":
    main()
