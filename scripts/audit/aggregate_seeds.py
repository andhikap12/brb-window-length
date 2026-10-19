"""aggregate_seeds.py — gabungkan hasil per-seed lintas log dan uji beda antar metode.

Angka n=20 sebelumnya dihitung ad-hoc; skrip ini menggantikannya supaya tabel utama dapat
direproduksi dari log mentah kapan saja.

Cara kerja:
  1. Segmentasi log berdasarkan penanda tahap ("=== [k/n] NAMA ===" atau "###### NAMA ######").
  2. Kumpulkan baris "[SEED n] CV-mean over ... Win-F1=xx.xx%" di dalam tiap segmen.
  3. Gabungkan run 8-seed lama dengan run 14-seed baru per metode. Seed 42 dan 123 hadir di
     keduanya sebagai KONTROL REPRODUKSIBILITAS — penggabungan hanya sah bila keduanya cocok,
     dan skrip menolak menggabung bila tidak.
  4. Uji berpasangan terhadap ERM pada seed yang sama (paired t + Wilcoxon).

Jalankan:
    ${PYTHON:-python} -u scripts/audit/aggregate_seeds.py
"""

import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "logs"

SEED_RE = re.compile(r"\[SEED (\d+)\] CV-mean over \d+ folds: Win-F1=([\d.]+)%")
STAGE_RE = re.compile(r"^(?:=== \[\d+/\d+\] (.+?) ===|#{6,} (.+?) #{6,})\s*$", re.M)

TOLERANSI_KONTROL = 0.01  # poin Win-F1; run deterministik harus cocok persis

# (label, berkas log, nama segmen atau None untuk seluruh berkas)
SUMBER = [
    ("ERM",         "tsit_ext_8seed_FP32_battery_20260626_183546.log", "BASELINE fp32"),
    ("ERM",         "byvar_seeds_20260731_193153.log",                 "s20_erm"),
    ("DANN l=1.0",  "tsit_ext_8seed_FP32_battery_20260626_183546.log", "DANN lambda=1.0 fp32 (vanilla)"),
    ("DANN l=1.0",  "byvar_seeds_20260731_193153.log",                 "s20_dann"),
    ("SupCon",      "supcon_tsit_ext_20260627_002936.log",             None),
    ("SupCon",      "byvar_seeds_20260802_110615.log",                 "s20_supcon"),
    ("DANN+SupCon", "dannsupcon_dsranfix_ext_20260627_011843.log",     None),
    ("DANN+SupCon", "byvar_seeds_20260802_110615.log",                 "s20_dannsupcon"),
]


def baca_segmen(path: Path, nama_segmen):
    """Kembalikan {seed: win_f1} untuk satu segmen log (atau seluruh berkas bila None)."""
    if not path.exists():
        return None
    teks = path.read_text(encoding="utf-8", errors="ignore")

    if nama_segmen is None:
        potongan = teks
    else:
        penanda = [(m.start(), (m.group(1) or m.group(2)).strip()) for m in STAGE_RE.finditer(teks)]
        potongan = None
        for i, (pos, nama) in enumerate(penanda):
            if nama != nama_segmen:
                continue
            akhir = penanda[i + 1][0] if i + 1 < len(penanda) else len(teks)
            potongan = teks[pos:akhir]
            break
        if potongan is None:
            return None

    return {int(s): float(v) for s, v in SEED_RE.findall(potongan)}


def kumpulkan():
    """Gabungkan seluruh sumber per metode; kembalikan {metode: (lama, baru, gabungan)}."""
    per_metode = {}
    for label, berkas, segmen in SUMBER:
        hasil = baca_segmen(LOGS / berkas, segmen)
        if not hasil:
            print(f"  [LEWATI] {label:<13} <- {berkas} / {segmen} (belum ada / kosong)")
            continue
        slot = per_metode.setdefault(label, {"lama": {}, "baru": {}})
        kunci = "baru" if "byvar_seeds" in berkas else "lama"
        slot[kunci].update(hasil)
        print(f"  [BACA]   {label:<13} <- {segmen or berkas}  ({len(hasil)} seed)")
    return per_metode


def periksa_kontrol(label, lama, baru):
    """Seed yang muncul di kedua run harus menghasilkan angka identik."""
    bersama = sorted(set(lama) & set(baru))
    if not bersama:
        return True, "tidak ada seed kontrol"
    pesan = []
    sah = True
    for s in bersama:
        beda = abs(lama[s] - baru[s])
        tanda = "cocok" if beda < TOLERANSI_KONTROL else f"BEDA {beda:+.2f}"
        if beda >= TOLERANSI_KONTROL:
            sah = False
        pesan.append(f"seed {s}: {lama[s]:.2f}/{baru[s]:.2f} {tanda}")
    return sah, "; ".join(pesan)


def ringkas(v):
    v = np.asarray(v, float)
    return f"{v.mean():.2f} +/- {v.std(ddof=1):.2f}"


def main():
    print("=" * 78)
    print("SUMBER LOG")
    print("=" * 78)
    per_metode = kumpulkan()
    if "ERM" not in per_metode:
        sys.exit("ERM tidak ditemukan — tidak ada pembanding")

    gabungan = {}
    print()
    print("=" * 78)
    print("KONTROL REPRODUKSIBILITAS (syarat sah penggabungan)")
    print("=" * 78)
    for label, slot in per_metode.items():
        sah, pesan = periksa_kontrol(label, slot["lama"], slot["baru"])
        print(f"  {label:<13} {'OK ' if sah else 'GAGAL'} {pesan}")
        if not sah:
            print(f"  {'':13} -> penggabungan {label} DITOLAK; hanya run lama dipakai")
            gabungan[label] = dict(slot["lama"])
        else:
            g = dict(slot["lama"])
            g.update(slot["baru"])
            gabungan[label] = g

    print()
    print("=" * 78)
    print("RINGKASAN")
    print("=" * 78)
    print(f"  {'Metode':<13} {'n=8 (lama)':>16} {'n gabungan':>12} {'gabungan':>16} {'median':>8}")
    for label, g in gabungan.items():
        lama = per_metode[label]["lama"]
        v_lama = np.array([lama[k] for k in sorted(lama)]) if lama else np.array([np.nan])
        v_gab = np.array([g[k] for k in sorted(g)])
        print(
            f"  {label:<13} {ringkas(v_lama) if lama else '—':>16} "
            f"{len(v_gab):>12} {ringkas(v_gab):>16} {np.median(v_gab):>8.2f}"
        )

    print()
    print("=" * 78)
    print("UJI BERPASANGAN TERHADAP ERM (seed yang sama saja)")
    print("=" * 78)
    erm = gabungan["ERM"]
    print(f"  {'Metode':<13} {'n':>3} {'delta':>8} {'t':>7} {'p (t)':>9} {'p (Wilc)':>9} {'menang':>8}")
    for label, g in gabungan.items():
        if label == "ERM":
            continue
        bersama = sorted(set(g) & set(erm))
        if len(bersama) < 3:
            print(f"  {label:<13} {len(bersama):>3}  seed bersama terlalu sedikit")
            continue
        a = np.array([g[s] for s in bersama])
        b = np.array([erm[s] for s in bersama])
        d = a - b
        t, p_t = stats.ttest_rel(a, b)
        try:
            _, p_w = stats.wilcoxon(a, b)
        except ValueError:
            p_w = float("nan")
        print(
            f"  {label:<13} {len(bersama):>3} {d.mean():>+8.2f} {t:>7.2f} "
            f"{p_t:>9.4f} {p_w:>9.4f} {int((d > 0).sum()):>4}/{len(d)}"
        )

    print()
    print("Catatan: sd memakai ddof=1. Uji berpasangan hanya memakai seed yang hadir di")
    print("kedua metode, sehingga delta di sini bisa sedikit beda dari selisih rerata di atas.")


if __name__ == "__main__":
    main()
