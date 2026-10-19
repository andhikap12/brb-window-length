"""audit_lodo.py — analisis LEAVE-ONE-TORQUE-OUT, 8 lipatan x 5 seed, ERM.

Kriteria L-P1..L-P3 TERKUNCI di docstring ini dan hasilnya dilaporkan di Sec. V-A makalah
CENIM 2026 (panel a gambar LODO). Ditulis sebelum satu lipatan dijalankan. Skrip ini hanya MEMBACA; ia tidak boleh
menambah, melonggarkan, atau menawar satu pun ambang di sana.

Dijalankan setelah `scripts/_run_lodo.sh` selesai:
    ${PYTHON:-python} -u scripts/audit/audit_lodo.py

DUA JEBAKAN YANG SKRIP INI TANGANI SECARA EKSPLISIT
===================================================

1. **fold_idx BUKAN nomor torsi.** `GroupKFold(n_splits=8)` menyeimbangkan ukuran grup;
   ia TIDAK mengurutkan grup. Pada run 3 Agu, lipatan 0 menahan **torsi-7**, lipatan 1
   menahan torsi-6, dan seterusnya menurun. Menganggap fold_idx == torsi akan MEMBALIK
   sumbu L-P1 dan menghasilkan rho bertanda terbalik tanpa satu pun tanda kesalahan.
   Karena itu torsi yang ditahan selalu dibaca dari baris `Test torques : [t]` di log,
   tidak pernah disimpulkan dari nomor lipatan.

2. **Log ganda.** Run 3 Agu diluncurkan dua kali secara bersamaan (dua proses menulis
   penanda `.done_lodo` yang sama). Semua `logs/lodo_*.log` dibaca, hasilnya diindeks
   per (torsi, seed), dan duplikat wajib COCOK. Ketidakcocokan dilaporkan, bukan
   dirata-ratakan diam-diam — dua proses identik yang menghasilkan angka berbeda adalah
   temuan tentang determinisme, bukan derau untuk dihaluskan.

3. **Log kampanye LAIN yang namanya ikut tertangkap glob.** Ditambahkan 8 Agu 2026.
   `_run_lodo_metode.sh` menulis ke `logs/lodo_metode_*.log`, yang COCOK dengan pola
   `lodo_*.log` di atas. Log itu memuat baris `SEED n | FOLD k/8`, `Test torques : [t]`,
   dan `CV-mean` dengan bentuk yang persis sama, jadi tanpa penyaring ia akan terbaca
   sebagai ERM. Karena ERM LODO sudah lengkap 8x5, SETIAP hasil metode akan menabrak
   kunci (torsi, seed) yang sudah terisi dan muncul sebagai "ketidakcocokan duplikat" —
   yaitu empat puluh temuan determinisme PALSU pada skrip yang justru dirancang untuk
   memperlakukan ketidakcocokan sebagai temuan serius.

   Penyaringnya memakai nama run di baris `DANN TRAINING: <nama>` (ERM = `LODO_baseline`,
   metode = `LODOM_*`), BUKAN nama berkas. Nama berkas adalah kebetulan penamaan skrip
   peluncur; nama run ditulis oleh kode pelatihan dari argumen `--name`, jadi ia bertahan
   kalau suatu saat log digabung, diganti nama, atau dialihkan.

SUMBER ANGKA A6
===============
Delapan angka keterbacaan per torsi berasal dari perluasan A6-P3 ke seluruh
torsi yang dijalankan ad-hoc — `audit_a6_resolvabilitas.py` yang ada di repo hanya
menghitung TIGA probe. Supaya L-P1 dapat direproduksi utuh dari repo, skrip ini
MENGHITUNG ULANG kedelapan angka itu memakai fitur dan probe yang sama persis
(diimpor, bukan disalin), lalu mencocokkannya dengan nilai terkunci di A6_TERKUNCI. Selisih
> 0,5 poin menghentikan skrip: yang salah bisa saja transkripsinya, dan pra-registrasi
tidak boleh diam-diam bergeser.

`--a6-terkunci` melewati perhitungan ulang (~2 menit) dan memakai nilai A6_TERKUNCI apa adanya.

RAMBU PLAFON — dipaksakan oleh kode, bukan diserahkan ke pembaca
================================================================
Plafon diperiksa SEBELUM rho dilaporkan. Bila >= 6 dari 8 lipatan di atas 98, ruang
gerak korelasi habis dan L-P1 dinyatakan GAGAL-UKUR — bukan "rho rendah yang bermakna".
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_a6_resolvabilitas import (  # noqa: E402
    CH_ENV,
    CHUNK,
    NPZ as A6_NPZ,
    SPLIT as A6_SPLIT,
    fitur_spektrum,
    probe,
)

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "logs"

# --- konstanta pra-registrasi L-P1..L-P3 — JANGAN diubah tanpa mengubah kriteria di docstring lebih dulu ---
AMBANG_LP1 = 0.70          # rho Spearman minimum supaya L-P1 lolos
PLAFON = 98.0              # F1 di atas ini dianggap mentok (rambu plafon)
PLAFON_MAKS = 6            # >= sebanyak ini lipatan mentok -> L-P1 gagal-ukur
A6_TERKUNCI = {0: 79.8, 1: 91.8, 2: 95.5, 3: 93.5, 4: 90.0, 5: 95.5, 6: 94.3, 7: 91.0}
TOLERANSI_A6 = 0.5         # poin persen; di atas ini skrip berhenti
SEED_DIHARAP = [42, 123, 2025, 7, 17]
TOLERANSI_DUPLIKAT = 0.01  # poin persen

SEED_FOLD_RE = re.compile(r"SEED (\d+) \| FOLD (\d+)/(\d+)")
TORSI_RE = re.compile(r"Test torques\s*:\s*\[(\d+)\]")
CVMEAN_RE = re.compile(r"\[SEED (\d+)\] CV-mean over \d+ folds: Win-F1=([\d.]+)%")
NAMA_RE = re.compile(r"DANN TRAINING:\s*(\S+)")
LENGAN_ERM = "LODO_baseline"   # satu-satunya nama run yang skrip ini boleh baca (jebakan 3)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass


def garis(judul: str = "") -> None:
    print("=" * 96)
    if judul:
        print(judul)
        print("=" * 96)


# ----------------------------------------------------------------------------------
# 1. Baca log
# ----------------------------------------------------------------------------------

def baris_log(path: Path):
    """Log berisi bilah kemajuan ber-\\r; pecah pada \\r DAN \\n."""
    teks = path.read_text(encoding="utf-8", errors="replace")
    for baris in teks.replace("\r", "\n").split("\n"):
        yield baris


def baca_satu_log(path: Path) -> tuple[dict[tuple[int, int], float], set[str]]:
    """-> ({(torsi_ditahan, seed): win_f1}, nama_run_lain_yang_ditemui).

    Torsi dibaca dari log, tidak pernah ditebak. Hanya blok milik LENGAN_ERM yang
    dicatat; nama run lain dikembalikan supaya penyaringnya terlihat, bukan senyap.
    """
    hasil: dict[tuple[int, int], float] = {}
    ditolak: set[str] = set()
    lengan_kini: str | None = None
    seed_kini: int | None = None
    torsi_kini: int | None = None

    for baris in baris_log(path):
        m = NAMA_RE.search(baris)
        if m:
            nama = str(m.group(1))
            lengan_kini = nama
            if nama != LENGAN_ERM:
                ditolak.add(nama)
            # blok baru dimulai: apa pun yang menggantung dari blok lalu dibuang
            seed_kini = torsi_kini = None
            continue
        m = SEED_FOLD_RE.search(baris)
        if m:
            seed_kini, torsi_kini = int(m.group(1)), None
            continue
        m = TORSI_RE.search(baris)
        if m and seed_kini is not None:
            torsi_kini = int(m.group(1))
            continue
        m = CVMEAN_RE.search(baris)
        if m:
            seed = int(m.group(1))
            if torsi_kini is None or seed != seed_kini:
                # blok tak lengkap (run terpotong) — dilewati, bukan ditebak
                seed_kini = torsi_kini = None
                continue
            # lengan_kini None berarti log tanpa header sama sekali. Itu TIDAK boleh
            # diam-diam dianggap ERM: skrip berhenti, karena menebak di sini persis
            # kesalahan yang jebakan 3 ada untuk mencegah.
            if lengan_kini is None:
                sys.exit(f"[ABORT] {path.name}: blok hasil tanpa baris "
                         f"'DANN TRAINING:' di atasnya. Lengannya tidak dapat "
                         f"dipastikan, dan menebaknya akan mencampur kampanye.")
            if lengan_kini == LENGAN_ERM:
                hasil[(torsi_kini, seed)] = float(m.group(2))
            seed_kini = torsi_kini = None
    return hasil, ditolak


def kumpulkan() -> tuple[dict[tuple[int, int], float], list[str]]:
    berkas = sorted(LOGS.glob("lodo_*.log"))
    if not berkas:
        sys.exit(f"[ABORT] tidak ada logs/lodo_*.log di {LOGS}")

    gabung: dict[tuple[int, int], float] = {}
    asal: dict[tuple[int, int], str] = {}
    bentrok: list[str] = []
    lengan_lain: set[str] = set()

    print(f"  membaca {len(berkas)} berkas log (hanya lengan {LENGAN_ERM}):")
    for p in berkas:
        satu, ditolak = baca_satu_log(p)
        lengan_lain |= ditolak
        catatan = f"  [+{len(ditolak)} lengan lain disaring]" if ditolak else ""
        print(f"    {p.name:<34} {len(satu):>3} run{catatan}")
        for kunci, nilai in satu.items():
            if kunci in gabung and abs(gabung[kunci] - nilai) > TOLERANSI_DUPLIKAT:
                bentrok.append(
                    f"torsi-{kunci[0]} seed-{kunci[1]}: "
                    f"{gabung[kunci]:.2f}% ({asal[kunci]}) vs {nilai:.2f}% ({p.name})"
                )
            else:
                gabung.setdefault(kunci, nilai)
                asal.setdefault(kunci, p.name)

    if lengan_lain:
        print(f"\n  [SARING] {len(lengan_lain)} lengan bukan-ERM diabaikan: "
              f"{', '.join(sorted(lengan_lain))}")
        print("  Analisis di bawah adalah ERM MURNI. Lengan itu dianalisis terpisah oleh")
        print("  scripts/audit/audit_lodo_metode.py; menggabungkannya di sini akan")
        print("  menampilkannya sebagai ketidakcocokan duplikat yang palsu.")
    return gabung, bentrok


def susun(gabung: dict[tuple[int, int], float]) -> dict[int, dict[int, float]]:
    per_torsi: dict[int, dict[int, float]] = defaultdict(dict)
    for (torsi, seed), f1 in gabung.items():
        per_torsi[torsi][seed] = f1
    return dict(sorted(per_torsi.items()))


# ----------------------------------------------------------------------------------
# 2. Keterbacaan A6 per torsi — perluasan A6-P3, fitur & probe DIIMPOR dari A6
# ----------------------------------------------------------------------------------

def a6_per_torsi() -> dict[int, float]:
    """rs vs r1b di dalam tiap torsi. Resep identik dengan A6-P3."""
    z = np.load(A6_NPZ)
    X = z["X"] if "X" in z.files else z[z.files[0]]
    y = z["y"]
    si = np.load(A6_SPLIT)
    torque, rep = si["torque"].astype(int), si["repetition"].astype(int)
    rec = torque * 1000 + rep * 10 + y

    print("  menghitung fitur spektral (w16384, envelope 0,5-15 Hz) ...", flush=True)
    n = X.shape[0]
    F = None
    for i in range(0, n, CHUNK):
        blok = fitur_spektrum(X[i:i + CHUNK, CH_ENV].astype(np.float64))
        if F is None:
            F = np.empty((n, blok.shape[1]))
        F[i:i + CHUNK] = blok
    if not np.isfinite(F).all():
        sys.exit("  [ABORT] fitur non-finite.")

    keluar: dict[int, float] = {}
    for t in sorted(set(torque.tolist())):
        m = (torque == t) & np.isin(y, [0, 1])
        keluar[int(t)] = 100.0 * probe(F[m], (y[m] == 1).astype(int), rec[m],
                                       f"  torsi-{t}  rs vs r1b")
    return keluar


def periksa_a6(hitung: dict[int, float]) -> None:
    """Pra-registrasi tidak boleh bergeser diam-diam."""
    buruk = [
        f"torsi-{t}: hitung {hitung[t]:.1f}% vs terkunci {A6_TERKUNCI[t]:.1f}%"
        for t in sorted(A6_TERKUNCI)
        if t in hitung and abs(hitung[t] - A6_TERKUNCI[t]) > TOLERANSI_A6
    ]
    if buruk:
        print()
        print("  [ABORT] keterbacaan A6 hasil hitung ulang TIDAK COCOK dengan A6_TERKUNCI:")
        for b in buruk:
            print(f"    {b}")
        print("  L-P1 memakai angka terkunci sebagai sumbu-x. Selisih di atas "
              f"{TOLERANSI_A6} poin berarti salah satu dari keduanya keliru;")
        print("  perbaiki A6_TERKUNCI lebih dulu, jangan lanjutkan analisis.")
        sys.exit(1)
    print(f"\n  [OK] kedelapan angka cocok dengan A6_TERKUNCI dalam {TOLERANSI_A6} poin.")


# ----------------------------------------------------------------------------------
# 3. Kriteria
# ----------------------------------------------------------------------------------

def lp3(per_torsi: dict[int, dict[int, float]], a6: dict[int, float]) -> dict[int, dict]:
    """L-P3 — kurva penuh, deskriptif, tanpa ambang."""
    garis("L-P3 (deskriptif) — F1 lawan torsi yang DITAHAN")
    print(f"  {'torsi':>5}  {'n seed':>6}  {'F1 rerata':>9}  {'sd':>6}  "
          f"{'median':>7}  {'min':>6}  {'maks':>6}   {'A6':>6}")
    print("  " + "-" * 74)
    ringkas: dict[int, dict] = {}
    for t, per_seed in per_torsi.items():
        v = np.array(list(per_seed.values()))
        ringkas[t] = {
            "rerata": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "median": float(np.median(v)), "min": float(v.min()), "maks": float(v.max()),
            "n": len(v), "per_seed": per_seed,
        }
        r = ringkas[t]
        tanda = "  <<" if r["rerata"] < PLAFON else ""
        print(f"  {t:>5}  {r['n']:>6}  {r['rerata']:>8.2f}%  {r['sd']:>5.2f}  "
              f"{r['median']:>6.2f}%  {r['min']:>5.2f}%  {r['maks']:>5.2f}%   "
              f"{a6.get(t, float('nan')):>5.1f}%{tanda}")
    print()
    for t, r in ringkas.items():
        detil = ", ".join(f"{s}:{f:.2f}" for s, f in r["per_seed"].items())
        print(f"    torsi-{t} per seed: {detil}")
    return ringkas


def periksa_plafon(ringkas: dict[int, dict], lengkap: bool) -> bool:
    """Rambu plafon — diperiksa SEBELUM rho dibaca. -> True bila L-P1 masih berdaya."""
    garis("RAMBU PLAFON")
    mentok = [t for t, r in ringkas.items() if r["rerata"] > PLAFON]
    print(f"  lipatan dengan F1 rerata > {PLAFON:.0f}: {len(mentok)} dari {len(ringkas)}"
          f"   {sorted(mentok)}")
    if not lengkap:
        kurang_torsi = 8 - len(ringkas)
        sebab = (f"{kurang_torsi} torsi belum ada" if kurang_torsi
                 else "kedelapan torsi ada, tetapi seednya belum lengkap")
        print(f"\n  [SEMENTARA] ambang plafon adalah {PLAFON_MAKS} dari 8 lipatan; {sebab}.")
        print("  Rerata lipatan yang seednya belum penuh masih dapat bergerak, jadi hitungan")
        print("  di atas belum dapat memutuskan berdaya atau tidak.")
        return False
    if len(mentok) >= PLAFON_MAKS:
        print(f"\n  >> L-P1 TIDAK BERDAYA. {len(mentok)} >= {PLAFON_MAKS} lipatan mentok;")
        print("     ruang gerak korelasi habis. Rho tetap dicetak di bawah, tetapi ia")
        print("     DILAPORKAN SEBAGAI GAGAL-UKUR, bukan sebagai rho rendah yang bermakna.")
        print("     (rambu plafon: kekuatan uji ini ada di ujung bawah dan bentuk kurva.)")
        return False
    print("\n  [OK] plafon tidak menghabiskan ruang gerak; L-P1 berdaya.")
    return True


def lp1(ringkas: dict[int, dict], a6: dict[int, float], berdaya: bool) -> None:
    garis("L-P1 (UTAMA) — Spearman rho: F1 LODO per torsi lawan keterbacaan A6 per torsi")
    torsi = sorted(set(ringkas) & set(a6))
    if len(torsi) < len(A6_TERKUNCI):
        print(f"  [PARSIAL] hanya {len(torsi)} dari 8 torsi tersedia: {torsi}")
        print("  L-P1 pra-registrasi menuntut 8 titik. TIDAK ADA VONIS diberikan.")
        return

    f1 = np.array([ringkas[t]["rerata"] for t in torsi])
    kb = np.array([a6[t] for t in torsi])
    rho, p = stats.spearmanr(f1, kb)
    print(f"  n titik = {len(torsi)}   (torsi {torsi})")
    print(f"  rho = {rho:+.4f}   p = {p:.4f}   ambang = {AMBANG_LP1:+.2f}")

    # sensitivitas: median antar-seed, supaya satu seed ambruk tidak menentukan vonis
    f1_med = np.array([ringkas[t]["median"] for t in torsi])
    rho_m, p_m = stats.spearmanr(f1_med, kb)
    print(f"  sensitivitas (median antar-seed): rho = {rho_m:+.4f}   p = {p_m:.4f}")

    print()
    if not berdaya:
        print("  >> VONIS: GAGAL-UKUR (rambu plafon lebih dulu berlaku).")
        print("     Rho di atas TIDAK boleh dibaca sebagai bukti untuk atau melawan M14.")
    elif rho >= AMBANG_LP1:
        print(f"  >> L-P1 LOLOS (rho {rho:+.4f} >= {AMBANG_LP1:.2f}).")
        print("     Keterbacaan tingkat sinyal meramalkan kesulitan tingkat model di seluruh")
        print("     delapan domain, diukur TANPA melatih apa pun. M14 DITEGAKKAN.")
    else:
        print(f"  >> L-P1 GUGUR (rho {rho:+.4f} < {AMBANG_LP1:.2f}).")
        print("     Kesulitan tidak dijelaskan keterbacaan sinyal. M14 DILEMAHKAN dan")
        print("     penjelasan lain wajib dicari.")
    if berdaya and (rho >= AMBANG_LP1) != (rho_m >= AMBANG_LP1):
        print()
        print("  [PERHATIAN] vonis rerata dan vonis median BERBEDA. Verdikat yang berlaku")
        print("  adalah yang berbasis rerata (statistik yang dipakai seluruh proyek), tetapi")
        print("  perbedaan ini berarti hasilnya digerakkan oleh sedikit seed. Laporkan keduanya.")


def lp2(ringkas: dict[int, dict]) -> None:
    garis("L-P2 (uji M13) — apakah 'di luar convex hull' dengan sendirinya menghukum?")
    interior = [t for t in ringkas if 1 <= t <= 6]
    if 7 not in ringkas or len(interior) < 6:
        print(f"  [PARSIAL] butuh torsi-7 dan torsi 1-6; tersedia {sorted(ringkas)}.")
        print("  TIDAK ADA VONIS diberikan.")
        return

    f7 = ringkas[7]["rerata"]
    nilai_int = {t: ringkas[t]["rerata"] for t in interior}
    lo, hi = min(nilai_int.values()), max(nilai_int.values())
    t_lo = min(nilai_int, key=nilai_int.get)
    print(f"  F1(torsi-7)          = {f7:.2f}%")
    print(f"  rentang interior 1-6 = {lo:.2f}% (torsi-{t_lo}) .. {hi:.2f}%")
    print()
    if f7 < lo:
        print(f"  >> M13 DIPERTAHANKAN sebagai penjelasan. F1(torsi-7) {f7:.2f} < "
              f"min interior {lo:.2f}:")
        print("     ujung atas ikut terhukum, seperti yang M13 ramalkan.")
    else:
        print(f"  >> M13 DICABUT sebagai penjelasan. F1(torsi-7) {f7:.2f} berada DI DALAM")
        print(f"     rentang interior [{lo:.2f}, {hi:.2f}] — berada di luar hull tidak")
        print("     dengan sendirinya menghukum. M13 tetap berlaku sebagai DESKRIPSI keadaan")
        print("     data; yang dicabut hanya perannya sebagai SEBAB.")

    med7 = ringkas[7]["median"]
    lo_med = min(ringkas[t]["median"] for t in interior)
    print()
    print(f"  sensitivitas (median): F1(torsi-7) = {med7:.2f}%, min interior = {lo_med:.2f}%"
          f"  -> M13 {'dipertahankan' if med7 < lo_med else 'dicabut'}")
    if (f7 < lo) != (med7 < lo_med):
        print("  [PERHATIAN] rerata dan median memberi vonis BERLAWANAN. Satu seed yang ambruk")
        print("  sedang menentukan pembacaan M13; rambu plafon melarang membaca satu lipatan sebagai")
        print("  klaim. Laporkan keduanya dan jangan pilih yang lebih enak.")


# ----------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Analisis LODO — kriteria L-P1..L-P3")
    ap.add_argument("--a6-terkunci", action="store_true",
                    help="pakai nilai A6_TERKUNCI apa adanya; lewati hitung ulang (~2 mnt)")
    args = ap.parse_args()

    garis("LODO — leave-one-torque-out, 8 lipatan x 5 seed, ERM. Kriteria L-P1..P3 (terkunci di docstring)")

    gabung, bentrok = kumpulkan()
    per_torsi = susun(gabung)
    print(f"\n  total {len(gabung)} run unik, {len(per_torsi)} torsi: {sorted(per_torsi)}")

    if bentrok:
        print(f"\n  [PERHATIAN] {len(bentrok)} run duplikat TIDAK COCOK antar log:")
        for b in bentrok:
            print(f"    {b}")
        print("  Dua proses dengan seed dan konfigurasi identik seharusnya menghasilkan angka")
        print("  yang sama. Selisih di sini adalah temuan tentang determinisme, bukan derau.")

    hilang = [t for t in range(8) if t not in per_torsi]
    kurang = {t: sorted(set(SEED_DIHARAP) - set(per_torsi[t])) for t in per_torsi
              if set(SEED_DIHARAP) - set(per_torsi[t])}
    if hilang:
        print(f"\n  [PARSIAL] torsi belum ada hasilnya: {hilang}")
    if kurang:
        print(f"  [PARSIAL] seed belum lengkap: "
              + ", ".join(f"torsi-{t}:{s}" for t, s in kurang.items()))
    if hilang or kurang:
        print("\n  Analisis diteruskan sebagai DESKRIPTIF. Vonis L-P1/L-P2 ditahan sampai")
        print("  kedelapan torsi x lima seed lengkap — pra-registrasi menuntut 8 titik.")

    print()
    if args.a6_terkunci:
        garis("Keterbacaan A6 per torsi — nilai TERKUNCI (hitung ulang dilewati)")
        a6 = dict(A6_TERKUNCI)
        for t, v in sorted(a6.items()):
            print(f"  torsi-{t}  rs vs r1b  {v:.1f}%")
        print("\n  [CATATAN] tanpa hitung ulang, L-P1 bersandar pada transkripsi A6_TERKUNCI.")
    else:
        garis("Keterbacaan A6 per torsi — HITUNG ULANG (perluasan A6-P3)")
        a6 = a6_per_torsi()
        periksa_a6(a6)

    print()
    print("  [RAMBU] A6 diukur di w16384; LODO dilatih di w4096. Korelasi PERINGKAT")
    print("  tetap sah, tetapi kesesuaian NILAI antara keduanya tidak boleh diklaim, dan")
    print("  perbedaan panjang jendela ini harus disebut saat hasilnya dilaporkan.")

    print()
    ringkas = lp3(per_torsi, a6)
    print()
    lengkap = not (hilang or kurang)
    berdaya = periksa_plafon(ringkas, lengkap)
    print()
    if not lengkap:
        garis("L-P1 / L-P2 — DITAHAN")
        print("  Hasil belum lengkap (lihat [PARSIAL] di atas). Tidak ada vonis diberikan.")
    else:
        lp1(ringkas, a6, berdaya)
        print()
        lp2(ringkas)

    print()
    garis("RAMBU YANG BERLAKU ATAS SELURUH KELUARAN DI ATAS")
    print("  - PROTOKOL BERBEDA. LODO melatih di 7 torsi; extremes-out di 6. Angka di atas")
    print("    TIDAK boleh disandingkan dengan 78,64 / 99,04 sebagai jika setara.")
    print("  - n = 5 seed, bukan 8. M9/M10/M11 tetap berlaku: tidak satu pun selisih")
    print("    antar-lipatan boleh jadi klaim tanpa uji. Yang dibaca BENTUK KURVA.")
    print("  - A6 mengukur rs vs r1b (biner); LODO mengukur makro-F1 5 kelas. Korelasi")
    print("    peringkat sah; kesesuaian NILAI tidak boleh diklaim.")


if __name__ == "__main__":
    main()
