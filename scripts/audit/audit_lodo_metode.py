"""audit_lodo_metode.py — LODO 8 lipatan x 5 seed untuk TIGA METODE TERKUAT lawan ERM.

Pasangan dari `scripts/_run_lodo_metode.sh`. Skrip ini hanya MEMBACA log.

    ${PYTHON:-python} -u scripts/audit/audit_lodo_metode.py

APA YANG DIUJI
==============
Separuh NEGATIF kontribusi paper — "tidak ada intervensi sisi-objektif yang terpisah
dari baseline" — sampai 8 Agu 2026 berdiri di atas SATU split domain untuk seluruh
keluarga metode (batasan split tunggal, Sec. VI makalah CENIM 2026), sementara separuh POSITIFnya sudah punya
LODO 8 lipatan. Kampanye ini memberi tiga lengan terkuat protokol yang sama dengan
separuh positif, sehingga klaim negatifnya berhenti bergantung pada satu split.

    89,25  BT+SupCon regresi-idx   LODOM_BT_didx
    87,50  SupCon                  LODOM_supcon
    86,00  DANN+SupCon             LODOM_dannsupcon

Pembandingnya ERM LODO (`LODO_baseline`) dengan seed yang SAMA (42,123,2025,7,17) dan
lipatan yang sama, jadi perbandingannya BERPASANGAN per (torsi, seed) — bukan dua
rerata yang kebetulan disandingkan.

KRITERIA LM-P1..LM-P3 — DIKUNCI 8 Agu 2026 07:40, SEBELUM hasil lengan mana pun dibaca
=======================================================================================
Keadaan pengetahuan saat kriteria ditulis, dinyatakan supaya tidak ada klaim buta yang
palsu: lima belas dari empat puluh hasil `LODOM_supcon` SUDAH terlihat (lipatan 0-2,
yaitu torsi 7, 6, 5) dan seluruhnya mentok — Rec-Acc 100,00 di kelima seed, Win-F1
98,97-100,00. Tidak satu pun hasil `LODOM_dannsupcon` atau `LODOM_BT_didx` ada saat
kriteria dikunci. Ambang di bawah TIDAK dipilih dengan melihat angka-angka itu; ia
diturunkan dari uji yang dipakai, dan penurunannya dicetak oleh kode.

LM-P1 (UTAMA, per lengan)
    Wilcoxon signed-rank dua sisi atas selisih berpasangan (metode - ERM) pada Win-F1
    torsi tertahan, 8 torsi x 5 seed = 40 pasang. Bonferroni atas TIGA lengan:
    alpha = 0,05/3 = 0,016667.
      p <  alpha  -> lengan TERPISAH dari ERM. Klaim negatif paper MELEMAH untuk
                     lengan itu dan wajib dilaporkan apa adanya.
      p >= alpha  -> lengan TIDAK TERPISAH. Klaim negatif MENGUAT: ia kini bertahan
                     di bawah 8 lipatan, bukan satu split.

LM-P2 (RAMBU DAYA — diperiksa SEBELUM p dibaca)
    Wilcoxon signed-rank membuang selisih nol. Dengan n_eff pasang tak-nol, nilai p
    dua sisi TERKECIL yang mungkin dicapai adalah 2^(1-n_eff), berapa pun datanya.
    Bila 2^(1-n_eff) > alpha, uji itu TIDAK DAPAT menolak sekalipun setiap pasang
    berpihak pada satu arah; hasilnya wajib dibaca GAGAL-UKUR, bukan "tidak terpisah".
    Ambangnya karena itu DITURUNKAN, bukan dipilih: n_eff minimum adalah bilangan bulat
    terkecil yang memenuhi 2^(1-n) <= alpha, dan kode mencetak penurunannya.

    Rambu ini bukan kehati-hatian teoretis. ERM LODO mentok di 6 dari 8 lipatan
    (audit_lodo.py, rambu plafon); bila lengan metode juga mentok di lipatan yang sama,
    pasangan di sana berselisih nol dan menguap dari uji.

LM-P3 (CAKUPAN)
    Tidak ada vonis untuk sebuah lengan sampai 8 torsi x 5 seed lengkap bagi lengan itu.
    Lengan yang belum lengkap dilaporkan deskriptif dan diberi label DITAHAN.

YANG DICABUT DAN YANG TIDAK
===========================
Ini PENCABUTAN SEBAGIAN B7. Sesudah kampanye ini B7 berbunyi: tercabut untuk ERM dan
untuk ketiga lengan terkuat; MASIH BERLAKU untuk dua belas lengan sisanya. Kode
mencetak kalimat itu di akhir supaya ia tidak pernah dilaporkan sebagai cabut penuh.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from math import log2
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_lodo import (  # noqa: E402
    LOGS,
    NAMA_RE,
    PLAFON,
    SEED_FOLD_RE,
    TORSI_RE,
    baris_log,
)

# --- konstanta pra-registrasi — JANGAN diubah tanpa mengubah docstring di atas lebih dulu ---
ERM = "LODO_baseline"
LENGAN = {                      # nama run -> label laporan
    "LODOM_supcon":      "SupCon",
    "LODOM_dannsupcon":  "DANN+SupCon",
    "LODOM_BT_didx":     "BT+SupCon regresi-idx",
}
ALPHA = 0.05 / len(LENGAN)      # Bonferroni atas tiga lengan
SEED_DIHARAP = [42, 123, 2025, 7, 17]
TORSI_DIHARAP = list(range(8))
TOLERANSI_DUPLIKAT = 0.01

CVMEAN2_RE = re.compile(
    r"\[SEED (\d+)\] CV-mean over \d+ folds: Win-F1=([\d.]+)%\s+Rec-Acc=([\d.]+)%")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass


def garis(judul: str = "") -> None:
    print("=" * 96)
    if judul:
        print(judul)
        print("=" * 96)


def n_eff_minimum(alpha: float) -> int:
    """Pasangan tak-nol paling sedikit supaya Wilcoxon dua sisi BISA mencapai alpha.

    p dua sisi terkecil yang dapat dicapai = 2^(1-n). Cari n bulat terkecil dengan
    2^(1-n) <= alpha, yaitu n >= 1 - log2(alpha).
    """
    n = 1.0 - log2(alpha)
    return int(np.ceil(n - 1e-12))


# ----------------------------------------------------------------------------------
# 1. Baca log — diindeks per (lengan, torsi, seed)
# ----------------------------------------------------------------------------------

def baca_semua() -> tuple[dict[tuple[str, int, int], tuple[float, float]], list[str]]:
    berkas = sorted(LOGS.glob("lodo_*.log"))
    if not berkas:
        sys.exit(f"[ABORT] tidak ada logs/lodo_*.log di {LOGS}")

    gabung: dict[tuple[str, int, int], tuple[float, float]] = {}
    asal: dict[tuple[str, int, int], str] = {}
    bentrok: list[str] = []
    tak_dikenal: set[str] = set()

    print(f"  membaca {len(berkas)} berkas log:")
    for p in berkas:
        n_baru = 0
        lengan: str | None = None
        seed: int | None = None
        torsi: int | None = None
        for baris in baris_log(p):
            m = NAMA_RE.search(baris)
            if m:
                lengan = str(m.group(1))
                if lengan != ERM and lengan not in LENGAN:
                    tak_dikenal.add(lengan)
                seed = torsi = None
                continue
            m = SEED_FOLD_RE.search(baris)
            if m:
                seed, torsi = int(m.group(1)), None
                continue
            m = TORSI_RE.search(baris)
            if m and seed is not None:
                torsi = int(m.group(1))
                continue
            m = CVMEAN2_RE.search(baris)
            if m:
                s = int(m.group(1))
                if torsi is None or s != seed or lengan is None:
                    seed = torsi = None
                    continue
                nilai = (float(m.group(2)), float(m.group(3)))
                kunci = (lengan, torsi, s)
                if kunci in gabung and max(abs(a - b) for a, b in zip(gabung[kunci], nilai)) > TOLERANSI_DUPLIKAT:
                    bentrok.append(
                        f"{lengan} torsi-{torsi} seed-{s}: "
                        f"{gabung[kunci][0]:.2f}/{gabung[kunci][1]:.2f} ({asal[kunci]}) vs "
                        f"{nilai[0]:.2f}/{nilai[1]:.2f} ({p.name})")
                elif kunci not in gabung:
                    gabung[kunci] = nilai
                    asal[kunci] = p.name
                    n_baru += 1
                seed = torsi = None
        print(f"    {p.name:<34} {n_baru:>3} run baru")

    if tak_dikenal:
        print(f"\n  [PERHATIAN] nama run tak dikenal ditemui dan diabaikan: "
              f"{', '.join(sorted(tak_dikenal))}")
        print("  Kalau salah satunya seharusnya ikut dianalisis, tambahkan ke LENGAN —")
        print("  jangan biarkan ia lolos diam-diam.")
    return gabung, bentrok


def susun(gabung, lengan: str) -> dict[tuple[int, int], tuple[float, float]]:
    return {(t, s): v for (ln, t, s), v in gabung.items() if ln == lengan}


# ----------------------------------------------------------------------------------
# 2. Laporan per lengan
# ----------------------------------------------------------------------------------

def cakupan(hasil: dict[tuple[int, int], tuple[float, float]]) -> tuple[list[int], dict[int, list[int]]]:
    ada = defaultdict(set)
    for (t, s) in hasil:
        ada[t].add(s)
    hilang = [t for t in TORSI_DIHARAP if t not in ada]
    kurang = {t: sorted(set(SEED_DIHARAP) - ada[t]) for t in ada
              if set(SEED_DIHARAP) - ada[t]}
    return hilang, kurang


def deskriptif(nama: str, met, erm) -> None:
    print(f"  {'torsi':>5}  {'n':>2}  {'ERM Win-F1':>11}  {nama[:14] + ' Win-F1':>21}  "
          f"{'selisih':>8}   {'kedua mentok':>12}")
    print("  " + "-" * 82)
    for t in TORSI_DIHARAP:
        pasang = [(erm[(t, s)][0], met[(t, s)][0]) for s in SEED_DIHARAP
                  if (t, s) in met and (t, s) in erm]
        if not pasang:
            continue
        e = np.array([a for a, _ in pasang])
        m = np.array([b for _, b in pasang])
        mentok = int(np.sum((e >= PLAFON) & (m >= PLAFON)))
        d = float((m - e).mean())
        print(f"  {t:>5}  {len(pasang):>2}  {e.mean():>10.2f}%  {m.mean():>20.2f}%  "
              f"{d:>+7.2f}   {mentok:>7} / {len(pasang)}")


def uji_lengan(nama_run: str, label: str, gabung, erm) -> dict | None:
    garis(f"LENGAN: {label}   ({nama_run})")
    met = susun(gabung, nama_run)
    if not met:
        print("  belum ada satu pun hasil. DITAHAN.\n")
        return None

    hilang, kurang = cakupan(met)
    print(f"  hasil tersedia: {len(met)} dari {len(TORSI_DIHARAP) * len(SEED_DIHARAP)} "
          f"pasang (8 torsi x 5 seed)")
    if hilang:
        print(f"  [PARSIAL] torsi belum ada: {hilang}")
    if kurang:
        print("  [PARSIAL] seed belum lengkap: "
              + ", ".join(f"torsi-{t}:{s}" for t, s in sorted(kurang.items())))
    print()

    deskriptif(label, met, erm)
    print()

    lengkap = not (hilang or kurang)
    if not lengkap:
        print("  >> LM-P3: DITAHAN. Vonis LM-P1 menuntut 8 torsi x 5 seed lengkap.")
        print("     Tabel di atas deskriptif dan TIDAK boleh dikutip sebagai hasil.\n")
        return None

    kunci = [(t, s) for t in TORSI_DIHARAP for s in SEED_DIHARAP]
    e = np.array([erm[k][0] for k in kunci])
    m = np.array([met[k][0] for k in kunci])
    d = m - e
    n_eff = int(np.sum(d != 0))
    n_min = n_eff_minimum(ALPHA)
    p_min = 2.0 ** (1 - n_eff) if n_eff > 0 else 1.0

    garis("LM-P2 (rambu daya) — diperiksa SEBELUM p dibaca")
    print(f"  pasang total          = {len(d)}")
    print(f"  selisih tepat nol     = {len(d) - n_eff}   (mentok di kedua lengan, "
          f"atau kebetulan identik)")
    print(f"  n_eff (tak-nol)       = {n_eff}")
    print(f"  alpha Bonferroni      = {ALPHA:.6f}  (0,05 / {len(LENGAN)} lengan)")
    print(f"  n_eff minimum         = {n_min}   diturunkan dari 2^(1-n) <= alpha")
    print(f"  p dua sisi terkecil yang dapat dicapai dengan n_eff={n_eff}: {p_min:.6f}")
    berdaya = n_eff >= n_min
    if not berdaya:
        print(f"\n  >> GAGAL-UKUR. {p_min:.4f} > {ALPHA:.4f}: uji ini tidak dapat menolak")
        print("     sekalipun SETIAP pasang berpihak ke satu arah. Hasil di bawah TIDAK")
        print("     boleh dibaca sebagai 'tidak terpisah'.")
    else:
        print("\n  [OK] uji berdaya untuk mencapai alpha.")
    print()

    garis("LM-P1 (UTAMA) — Wilcoxon signed-rank berpasangan, metode - ERM, Win-F1")
    try:
        stat, p = stats.wilcoxon(m, e, alternative="two-sided")
    except ValueError as exc:          # seluruh selisih nol
        stat, p = float("nan"), 1.0
        print(f"  [catatan scipy] {exc}")
    menang = int(np.sum(d > 0))
    kalah = int(np.sum(d < 0))
    print(f"  rerata selisih   = {d.mean():+.3f} poin Win-F1")
    print(f"  median selisih   = {np.median(d):+.3f}")
    print(f"  menang / kalah / seri = {menang} / {kalah} / {len(d) - menang - kalah}")
    print(f"  W = {stat}   p = {p:.6f}   alpha = {ALPHA:.6f}")
    print()
    if not berdaya:
        vonis = "GAGAL-UKUR"
        print("  >> VONIS: GAGAL-UKUR (rambu LM-P2 lebih dulu berlaku).")
        print("     p di atas tidak melisensikan klaim ke arah mana pun.")
    elif not np.isfinite(p):
        # Tidak terjangkau selama LM-P2 berlaku (n_eff >= 7 membuat scipy selalu
        # mengembalikan p hingga). Ditulis eksplisit karena `nan < alpha` bernilai
        # False, jadi p yang tak hingga akan LOLOS ke cabang "TIDAK TERPISAH" dan
        # terbaca sebagai vonis — persis kegagalan senyap yang harus dicegah.
        vonis = "GAGAL-UKUR"
        print(f"  >> VONIS: GAGAL-UKUR. p tidak hingga ({p}) meski n_eff={n_eff} lolos")
        print("     LM-P2. Ini kombinasi yang seharusnya mustahil; periksa datanya,")
        print("     jangan membaca hasilnya.")
    elif p < ALPHA:
        vonis = "TERPISAH"
        arah = "DI ATAS" if d.mean() > 0 else "DI BAWAH"
        print(f"  >> LM-P1: {label} TERPISAH dari ERM (p {p:.6f} < {ALPHA:.6f}), {arah} ERM.")
        print("     Klaim negatif paper MELEMAH untuk lengan ini. Laporkan apa adanya;")
        print("     lengan ini tidak boleh lagi disebut 'tidak terpisah dari baseline'.")
    else:
        vonis = "TIDAK TERPISAH"
        print(f"  >> LM-P1: {label} TIDAK TERPISAH dari ERM (p {p:.6f} >= {ALPHA:.6f}).")
        print("     Klaim negatif MENGUAT: ia bertahan di bawah 8 lipatan, bukan satu split.")
    print()

    # LM-P2 lanjutan: di mana pun dayanya berada, itu ada di lipatan yang TIDAK mentok
    tak_mentok = [i for i, k in enumerate(kunci) if not (e[i] >= PLAFON and m[i] >= PLAFON)]
    if tak_mentok and berdaya:
        dt = d[tak_mentok]
        torsi_t = sorted({kunci[i][0] for i in tak_mentok})
        print(f"  [ekor] {len(tak_mentok)} pasang tidak mentok di kedua lengan "
              f"(torsi {torsi_t}); rerata selisih di situ = {dt.mean():+.3f}")
        print("  Deskriptif, tanpa ambang: seluruh daya uji hidup di pasang ini.")
        print()
    return {"lengan": nama_run, "label": label, "p": p, "d": d.mean(),
            "n_eff": n_eff, "vonis": vonis}


# ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Analisis LODO tiga metode — kriteria LM-P1..P3")
    ap.parse_args()

    garis("LODO METODE — 3 lengan x 8 lipatan x 5 seed lawan ERM. Kriteria dikunci 8 Agu 07:40")

    gabung, bentrok = baca_semua()
    erm = susun(gabung, ERM)
    print(f"\n  ERM ({ERM}): {len(erm)} pasang")
    for nama, label in LENGAN.items():
        print(f"  {label} ({nama}): {len(susun(gabung, nama))} pasang")

    if bentrok:
        print(f"\n  [PERHATIAN] {len(bentrok)} run duplikat TIDAK COCOK antar log:")
        for b in bentrok:
            print(f"    {b}")
        print("  Konfigurasi dan seed identik seharusnya menghasilkan angka sama.")

    hilang_erm, kurang_erm = cakupan(erm)
    if hilang_erm or kurang_erm:
        sys.exit(f"\n[ABORT] pembanding ERM belum lengkap (hilang {hilang_erm}, "
                 f"kurang {kurang_erm}). Perbandingan berpasangan tidak dapat dibentuk.")
    print(f"\n  [OK] pembanding ERM lengkap: 8 torsi x 5 seed.")

    print()
    print(f"  [RAMBU] ERM LODO mentok (>{PLAFON:.0f} Win-F1) di 6 dari 8 lipatan; lihat")
    print("  audit_lodo.py rambu plafon. Daya perbandingan ini hidup di torsi-0 dan torsi-7.")
    print()

    ringkas = [uji_lengan(n, l, gabung, erm) for n, l in LENGAN.items()]
    ringkas = [r for r in ringkas if r]

    garis("RINGKASAN")
    if not ringkas:
        print("  Belum ada lengan yang lengkap. Tidak ada vonis.")
    else:
        print(f"  {'lengan':>24}  {'n_eff':>5}  {'d rerata':>9}  {'p':>9}   vonis")
        print("  " + "-" * 74)
        for r in ringkas:
            print(f"  {r['label']:>24}  {r['n_eff']:>5}  {r['d']:>+8.3f}  {r['p']:>9.6f}   "
                  f"{r['vonis']}")
    print()
    garis("BATAS YANG BERLAKU ATAS SELURUH KELUARAN DI ATAS")
    print("  - INI PENCABUTAN SEBAGIAN B7, BUKAN PENUH. Sesudah kampanye ini B7 berbunyi:")
    print("    tercabut untuk ERM dan untuk ketiga lengan terkuat; MASIH BERLAKU untuk dua")
    print("    belas lengan sisanya, yang seluruhnya masih satu split. Jangan menuliskannya")
    print("    sebagai tercabut penuh.")
    print("  - Ketiga lengan dipilih karena Rec-Acc torsi-0 TERTINGGI di Tabel I, yaitu")
    print("    titik paling rentan bagi klaim negatif. Itu memperkuat vonis TIDAK TERPISAH")
    print("    dan TIDAK melisensikan generalisasi ke lengan yang tidak diuji.")
    print("  - n = 5 seed. Tidak satu pun selisih antar-lipatan boleh jadi klaim tanpa uji.")


if __name__ == "__main__":
    main()
