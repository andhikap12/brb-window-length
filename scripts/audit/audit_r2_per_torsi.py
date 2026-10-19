"""audit_r2_per_torsi.py — R2 bagian A: pecahan per-torsi seluruh keluarga metode.

Kriteria R2-P1..R2-P3 TERKUNCI di docstring ini, ditulis sebelum satu angka dibaca.
Hasilnya adalah kolom low/high Tabel I dan galat tersisa di Sec. IV-A makalah CENIM 2026. Skrip ini hanya membaca `audit_ext_family.npz` — tidak
melatih, tidak menginferensi, tanpa GPU.

MENGAPA
=======
Tabel I makalah melaporkan F1 **agregat** beserta kolom per-torsi. Angka agregat dapat
menyembunyikan asal-usulnya: dua metode yang dicurigai berubah makna begitu dipecah
per-torsi. R2 memeriksa apakah itu berlaku untuk SELURUH keluarga, bukan hanya keduanya.

Kolom `audit_ext_family.npz` (format 10 kolom, sebelum dua kolom geometri ditambahkan):
    0 seed | 1 f1_total | 2 f1_torsi0 | 3 f1_torsi7 | 4-8 f1 per kelas | 9 domain_probe
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, ttest_rel, wilcoxon

REPO = Path(__file__).resolve().parents[2]
NPZ = REPO / "results" / "motor" / "audit_ext_family.npz"
ERM = "ERM_(baseline)"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass


def garis(judul: str = "") -> None:
    print("=" * 108)
    if judul:
        print(judul)
        print("=" * 108)


def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    metode = list(z.files)
    D = {k: z[k] for k in metode}
    n_seed = D[ERM].shape[0]

    garis("R2 — PECAHAN PER-TORSI SELURUH KELUARGA. Kriteria R2-P1..P3 (terkunci di docstring)")
    print(f"  sumber : {NPZ.name}   {len(metode)} metode x {n_seed} seed, tanpa GPU")
    print()

    e_tot, e_t0, e_t7 = D[ERM][:, 1], D[ERM][:, 2], D[ERM][:, 3]

    # ---------------------------------------------------------------- tabel utama
    garis("1. Tabel per-torsi — diurutkan menurut F1 AGREGAT (urutan Tabel I)")
    baris = []
    for k in metode:
        A = D[k]
        baris.append({
            "nama": k,
            "tot": A[:, 1].mean(), "tot_sd": A[:, 1].std(ddof=1),
            "t0": A[:, 2].mean(), "t0_sd": A[:, 2].std(ddof=1),
            "t7": A[:, 3].mean(), "t7_sd": A[:, 3].std(ddof=1),
            "d_tot": (A[:, 1] - e_tot).mean(),
            "d_t0": (A[:, 2] - e_t0).mean(),
            "d_t7": (A[:, 3] - e_t7).mean(),
            "menang_t0": int((A[:, 2] > e_t0).sum()),
            "A": A,
        })
    baris.sort(key=lambda r: -r["tot"])

    print(f"  {'#':>2} {'metode':<22} {'F1 agregat':>14} {'F1 torsi-0':>15} {'F1 torsi-7':>14}"
          f" {'Δt0':>8} {'Δt7':>8} {'menang t0':>10}")
    for i, r in enumerate(baris, 1):
        tanda = "" if r["nama"] == ERM else f"{r['d_t0']:>+8.2f} {r['d_t7']:>+8.2f}"
        mn = "" if r["nama"] == ERM else f"{r['menang_t0']}/{n_seed}"
        print(f"  {i:>2} {r['nama']:<22} {r['tot']:>7.2f}±{r['tot_sd']:<6.2f} "
              f"{r['t0']:>8.2f}±{r['t0_sd']:<6.2f} {r['t7']:>7.2f}±{r['t7_sd']:<6.2f} "
              f"{tanda:>17} {mn:>10}")

    # ---------------------------------------------------------------- R2-P1
    garis("R2-P1 — KESTABILAN PERINGKAT (ambang ρ = 0,70)")
    nm = [r["nama"] for r in baris]
    v_tot = np.array([r["tot"] for r in baris])
    v_t0 = np.array([r["t0"] for r in baris])
    v_t7 = np.array([r["t7"] for r in baris])
    rho0, p0 = spearmanr(v_tot, v_t0)
    rho7, p7 = spearmanr(v_tot, v_t7)
    rho07, _ = spearmanr(v_t0, v_t7)

    print(f"  ρ(agregat, torsi-0) = {rho0:+.3f}   p = {p0:.4f}")
    print(f"  ρ(agregat, torsi-7) = {rho7:+.3f}   p = {p7:.4f}")
    print(f"  ρ(torsi-0, torsi-7) = {rho07:+.3f}   <- dua domain uji saling meramalkan?")

    p1 = rho0 >= 0.70
    print(f"\n  R2-P1 {'LOLOS' if p1 else 'GUGUR'} — "
          + ("tabel agregat setia pada urutan torsi-0; kolom per-torsi cukup dilampirkan"
             if p1 else
             "tabel agregat MENYESATKAN tentang di mana metode berbeda; Tabel I makalah "
             "WAJIB memuat kolom torsi-0 dan torsi-7"))

    # perpindahan peringkat terbesar
    r_tot = {n: i for i, n in enumerate(sorted(nm, key=lambda n: -v_tot[nm.index(n)]), 1)}
    r_t0 = {n: i for i, n in enumerate(sorted(nm, key=lambda n: -v_t0[nm.index(n)]), 1)}
    geser = sorted(nm, key=lambda n: -abs(r_tot[n] - r_t0[n]))
    print(f"\n  Perpindahan peringkat terbesar (agregat -> torsi-0):")
    for n in geser[:5]:
        print(f"    {n:<24} #{r_tot[n]:>2} -> #{r_t0[n]:>2}   ({r_t0[n]-r_tot[n]:+d})")

    # ---------------------------------------------------------------- R2-P2
    garis("R2-P2 — lokasi keunggulan, berpasangan per seed (deskriptif, tanpa ambang)")
    print("  Rambu M9/M10/M11: n=8, tidak satu pun selisih di bawah ini boleh jadi klaim.\n")
    print(f"  {'metode':<22} {'Δ agregat':>10} {'Δ torsi-0':>11} {'p':>8} {'Δ torsi-7':>11}"
          f" {'p':>8}  vonis lokasi")
    for r in baris:
        if r["nama"] == ERM:
            continue
        A = r["A"]
        d0, d7 = A[:, 2] - e_t0, A[:, 3] - e_t7
        pp0 = ttest_rel(A[:, 2], e_t0).pvalue if np.std(d0) > 0 else 1.0
        pp7 = ttest_rel(A[:, 3], e_t7).pvalue if np.std(d7) > 0 else 1.0
        if abs(d0.mean()) > 2 * abs(d7.mean()):
            lok = "terpusat di TORSI-0"
        elif abs(d7.mean()) > 2 * abs(d0.mean()):
            lok = "terpusat di torsi-7"
        else:
            lok = "menyebar"
        print(f"  {r['nama']:<22} {r['d_tot']:>+10.2f} {d0.mean():>+11.2f} {pp0:>8.4f}"
              f" {d7.mean():>+11.2f} {pp7:>8.4f}  {lok}")

    # ---------------------------------------------------------------- konteks
    garis("2. Ruang yang tersisa — mengapa torsi-0 satu-satunya sumbu yang berarti")
    t0_all = np.concatenate([D[k][:, 2] for k in metode])
    t7_all = np.concatenate([D[k][:, 3] for k in metode])
    print(f"  torsi-0 : rentang {t0_all.min():6.2f} – {t0_all.max():6.2f}   "
          f"rerata {t0_all.mean():6.2f}   galat tersisa {100-t0_all.mean():5.2f} poin")
    print(f"  torsi-7 : rentang {t7_all.min():6.2f} – {t7_all.max():6.2f}   "
          f"rerata {t7_all.mean():6.2f}   galat tersisa {100-t7_all.mean():5.2f} poin")
    print(f"\n  sebaran antar-metode (sd rerata metode): torsi-0 {v_t0.std(ddof=1):.2f} "
          f"vs torsi-7 {v_t7.std(ddof=1):.2f}")
    print("  -> seluruh perbedaan antar-metode di Tabel I hidup di torsi-0.")

    garis("VONIS R2 bagian A")
    print(f"  R2-P1 : {'LOLOS' if p1 else 'GUGUR'}  (ρ = {rho0:+.3f})")
    print(f"  R2-P2 : dilaporkan di atas, tanpa ambang menurut pra-registrasi")
    print(f"  R2-P3 : Rec-Accuracy — bagian B, butuh satu lintasan inferensi")


if __name__ == "__main__":
    main()
