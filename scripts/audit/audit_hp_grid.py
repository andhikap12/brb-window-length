"""audit_hp_grid.py — efek utama PANJANG di setiap sel (backbone, lr).

MENGAPA SKRIP INI ADA
=====================
Menjalankan faktorial R3 di banyak sel (backbone x lr) menciptakan bahaya yang tidak ada
saat hanya satu sel dijalankan: dengan cukup banyak sel, SELALU ada satu yang tidak
mereplikasi karena kebetulan. Kalau aturan bacanya baru disusun setelah kisinya terisi,
sel yang menyimpang bisa dibingkai sebagai "kasus khusus" atau sel yang cocok dipilih
sebagai "yang representatif". Dua-duanya taman bercabang, dan dua-duanya tak terdeteksi
dari luar.

Karena itu aturan baca dikunci DI SINI, dalam kode, sebelum sel keempat ada:

  1. Metrik utamanya SATU: efek utama PANJANG pada Rec-Acc torsi-0, yaitu rata-rata dua
     kontras marginal (panjang @ volume rendah, panjang @ volume tinggi). Konvensi ini
     disalin dari audit_r3_faktorial.py dan menghasilkan +15,88 untuk TSiTPlus @ 1e-3.
     TIDAK ADA metrik kedua yang boleh dipromosikan jadi utama belakangan.
  2. Vonisnya dibaca dari SELURUH kisi sekaligus, bukan sel per sel:
       - efek positif di SEMUA sel                      -> bertahan lintas konfigurasi
       - positif di sebagian besar, satu sel menyimpang -> bertahan, DAN sel yang menyimpang
                                                          wajib dilaporkan dengan namanya
       - tanda campur aduk tanpa pola                   -> efeknya bergantung konfigurasi;
                                                          klaim makalah (Sec. IV-B) harus dilemahkan
  3. Ambang lolos-per-sel dikunci sama dengan yang ditulis di kepala _run_lr_faktorial.sh
     pada 8 Agu 2026 sebelum satu angka pun ada: >= +10 bertahan, +3..+10 arah saja,
     < +3 tidak bertahan.
  4. Tidak ada uji signifikansi lintas sel. Delapan seed per sel dan sel yang jumlahnya
     sedikit tidak menopang koreksi perbandingan ganda apa pun yang bermakna. Kisi ini
     DESKRIPTIF, dan harus ditulis begitu.

Sel keluarga CNN, kalau nanti diisi, menjawab pertanyaan BERBEDA: BatchNorm + GAP membuat
baseline auto-invarian terhadap beban by construction, jadi selnya terkonfusi dengan
pencucian amplitudo dan tidak dapat diatribusikan ke lr. Skrip menandainya, dan angkanya
TIDAK boleh dijumlahkan ke dalam vonis butir 2 di atas.

Jalankan dari akar repositori:  ${PYTHON:-python} scripts/audit/audit_hp_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results" / "motor"

# Kolom di dalam array hasil (n_seed, 15), konvensi audit_r3_faktorial.py.
I_REC_T0 = -2
I_F1_T0 = 2

# Sel -> berkas per lengan. Nama historisnya tidak seragam, jadi dipetakan eksplisit
# alih-alih ditebak dari pola. Sel yang berkasnya belum ada akan dilewati, bukan menggagalkan.
KISI = {
    ("TSiTPlus", "1e-3"): {
        "w4096sub":  RES / "r3_w4096sub.npz",
        "w16384":    RES / "r3_w16384.npz",
        "w4096full": RES / "audit_ext_family_rec.npz",
        "w16384ovl": RES / "r3_w16384ovl.npz",
    },
    ("TSiTPlus", "3e-4"): {
        "w4096sub":  RES / "audit_lr3e4_w4096sub_s8.npz",
        "w16384":    RES / "audit_lr3e4_w16384_s8.npz",
        "w4096full": RES / "audit_lr3e4_w4096full_s8.npz",
        "w16384ovl": RES / "audit_lr3e4_w16384ovl_s8.npz",
    },
    ("TSSequencerPlus", "1e-3"): {
        "w4096sub":  RES / "audit_tsseq_w4096sub_s8.npz",
        "w16384":    RES / "audit_tsseq_w16384_s8.npz",
        "w4096full": RES / "audit_tsseq_w4096full_s8.npz",
        "w16384ovl": RES / "audit_tsseq_w16384ovl_s8.npz",
    },
    ("TSSequencerPlus", "3e-4"): {
        "w4096sub":  RES / "audit_lr3e4tsseq_w4096sub_s8.npz",
        "w16384":    RES / "audit_lr3e4tsseq_w16384_s8.npz",
        "w4096full": RES / "audit_lr3e4tsseq_w4096full_s8.npz",
        "w16384ovl": RES / "audit_lr3e4tsseq_w16384ovl_s8.npz",
    },
}

# Backbone yang selnya terkonfound BN+GAP. Dilaporkan terpisah, tidak masuk vonis.
TERKONFOUND = {"InceptionTimePlus", "ResNetPlus", "CNN1D"}

AMBANG_BERTAHAN = 10.0
AMBANG_ARAH = 3.0

# Kunci di audit_ext_family_rec.npz memuat banyak metode; lengan w4096-penuh adalah ERM.
KUNCI_KHUSUS = {"audit_ext_family_rec.npz": "ERM_(baseline)"}


def muat_lengan(fp: Path) -> np.ndarray | None:
    if not fp.exists():
        return None
    z = np.load(fp, allow_pickle=True)
    kunci = KUNCI_KHUSUS.get(fp.name)
    if kunci is None:
        if len(z.files) != 1:
            raise SystemExit(f"[ABORT] {fp.name} punya {len(z.files)} kunci "
                             f"({list(z.files)}); tambahkan ke KUNCI_KHUSUS.")
        kunci = z.files[0]
    elif kunci not in z.files:
        cocok = [k for k in z.files if k.replace("_", " ") == kunci.replace("_", " ")]
        if not cocok:
            raise SystemExit(f"[ABORT] kunci '{kunci}' tidak ada di {fp.name}.")
        kunci = cocok[0]
    return z[kunci]


def sejajarkan(lengan: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Hanya seed yang ada di KEEMPAT lengan, diurutkan sama. Tanpa ini pasangannya palsu."""
    seeds = sorted(set.intersection(*[set(A[:, 0].astype(int).tolist())
                                      for A in lengan.values()]))
    out = {}
    for nama, A in lengan.items():
        idx = {int(s): i for i, s in enumerate(A[:, 0])}
        out[nama] = A[[idx[s] for s in seeds]]
    return out, seeds


def efek_sel(lengan: dict[str, np.ndarray], kolom: int) -> dict:
    d, seeds = sejajarkan(lengan)
    r = {k: v[:, kolom] for k, v in d.items()}
    pj_lo = r["w16384"] - r["w4096sub"]
    pj_hi = r["w16384ovl"] - r["w4096full"]
    jm_lo = r["w4096full"] - r["w4096sub"]
    jm_hi = r["w16384ovl"] - r["w16384"]
    return {
        "n_seed": len(seeds),
        "panjang_lo": float(pj_lo.mean()), "panjang_hi": float(pj_hi.mean()),
        "efek_panjang": float(0.5 * (pj_lo.mean() + pj_hi.mean())),
        "efek_jumlah": float(0.5 * (jm_lo.mean() + jm_hi.mean())),
        "menang_lo": int((pj_lo > 0).sum()), "menang_hi": int((pj_hi > 0).sum()),
    }


def vonis_sel(e: float) -> str:
    if e >= AMBANG_BERTAHAN:
        return "BERTAHAN"
    if e >= AMBANG_ARAH:
        return "ARAH SAJA"
    return "TIDAK BERTAHAN"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 92)
    print("KISI HIPERPARAMETER x BACKBONE — efek utama PANJANG, Rec-Acc torsi-0")
    print("Aturan baca dikunci di docstring skrip ini sebelum sel keempat ada.")
    print("=" * 92)

    terisi, kosong = {}, []
    for (bb, lr), berkas in KISI.items():
        lengan = {k: muat_lengan(v) for k, v in berkas.items()}
        hilang = [k for k, v in lengan.items() if v is None]
        if hilang:
            kosong.append((bb, lr, hilang))
            continue
        terisi[(bb, lr)] = efek_sel(lengan, I_REC_T0)

    if not terisi:
        raise SystemExit("[ABORT] belum ada sel yang lengkap.")

    print(f"\n  {'backbone':<18}{'lr':>7}{'seed':>6}{'pjg@lo':>9}{'pjg@hi':>9}"
          f"{'EFEK PJG':>10}{'efek vol':>10}{'menang':>9}  vonis")
    for (bb, lr), e in sorted(terisi.items()):
        tanda = " *" if bb in TERKONFOUND else ""
        menang = f"{e['menang_lo']}/{e['n_seed']},{e['menang_hi']}/{e['n_seed']}"
        print(f"  {bb:<18}{lr:>7}{e['n_seed']:>6}{e['panjang_lo']:>+9.2f}"
              f"{e['panjang_hi']:>+9.2f}{e['efek_panjang']:>+10.2f}"
              f"{e['efek_jumlah']:>+10.2f}{menang:>9}"
              f"  {vonis_sel(e['efek_panjang'])}{tanda}")

    if any(bb in TERKONFOUND for bb, _ in terisi):
        print("\n  * BN+GAP: baseline auto-invarian terhadap beban by construction. Sel ini")
        print("    menjawab 'apakah efek muncul di rezim normalisasi lain', BUKAN 'apakah")
        print("    efek artefak penyetelan'. TIDAK ikut ke vonis kisi di bawah.")

    if kosong:
        print("\n  Sel belum lengkap:")
        for bb, lr, hilang in kosong:
            print(f"    {bb} @ {lr} — kurang: {', '.join(hilang)}")

    bersih = {k: v for k, v in terisi.items() if k[0] not in TERKONFOUND}
    print("\n" + "=" * 92)
    print(f"VONIS KISI atas {len(bersih)} sel tak terkonfound")
    print("=" * 92)
    if len(bersih) < 2:
        print("  Sel bersih kurang dari dua; kisi belum dapat dibaca sebagai kisi.")
        return 0

    nilai = {k: v["efek_panjang"] for k, v in bersih.items()}
    n_pos = sum(1 for v in nilai.values() if v >= AMBANG_ARAH)
    menyimpang = [k for k, v in nilai.items() if v < AMBANG_ARAH]

    if n_pos == len(nilai):
        print("  Efek panjang positif di SEMUA sel bersih -> bertahan lintas konfigurasi.")
        print(f"  Rentang: {min(nilai.values()):+.2f} sampai {max(nilai.values()):+.2f}")
    elif n_pos >= len(nilai) - 1:
        print("  Bertahan di sebagian besar sel. Sel yang menyimpang WAJIB dilaporkan")
        print("  dengan namanya, bukan dihilangkan sebagai kasus khusus:")
        for bb, lr in menyimpang:
            print(f"    {bb} @ lr {lr}: {nilai[(bb, lr)]:+.2f}")
    else:
        print("  Tanda campur aduk tanpa pola -> efeknya BERGANTUNG KONFIGURASI.")
        print("  Klaim makalah harus dilemahkan, bukan dipertahankan dengan memilih sel.")
        for (bb, lr), v in sorted(nilai.items()):
            print(f"    {bb} @ lr {lr}: {v:+.2f}")

    print("\n  Kisi ini DESKRIPTIF. Delapan seed per sel tidak menopang koreksi perbandingan")
    print("  ganda yang bermakna, jadi tidak ada nilai p lintas sel dan tidak boleh ada")
    print("  klaim signifikansi yang ditarik dari tabel di atas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
