"""audit_slip.py — ukur slip SEBENARNYA per level torsi dari kanal getaran.

MENGUJI klaim Sec. V-B makalah CENIM 2026: pada torsi rendah, modulasi 2sf yang
membawa tanda-tangan BRB jatuh di bawah resolusi frekuensi jendela 4096 sampel, dan
jendelanya bahkan lebih pendek daripada satu periode modulasi. Bila benar, kesulitan di
torsi-0 bersifat FISIK (sinyalnya tidak terwakili), bukan pergeseran domain — dan itu
menjelaskan mengapa seluruh sumbu invariansi datar.

Sampai sekarang slip hanya DIASUMSIKAN sebanding torsi memakai slip nominal 0,0472.
Skrip ini mengukurnya.

Metode:
  1. Baca kanal getaran MENTAH (bukan .npz — pipeline mem-bandpass 40-100 Hz, yang
     justru membuang komponen poros ~29 Hz yang kita butuhkan).
  2. Potong ke steady-state memakai falling edge Trigger, logika identik s2_steady_state.
  3. Spektrum resolusi tinggi atas seluruh segmen steady-state (~20 s -> ~0,05 Hz).
  4. Puncak pada pita frekuensi poros -> f_r; slip = (f_sync - f_r) / f_sync.
  5. Bandingkan 2sf terukur terhadap resolusi bin dan panjang jendela pelatihan.

Jalankan:
    ${PYTHON:-python} -u scripts/audit/audit_slip.py
    ${PYTHON:-python} -u scripts/audit/audit_slip.py --reps 5 --kondisi rs,r2b
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "DatasetRaw" / "banco de dados experimental"

FS = 7600.0  # laju getaran, terkonfirmasi audit_fs_ratio.py
F_LINE = 60.0
KUTUB = 4
F_SYNC = 2 * F_LINE / KUTUB  # 30 Hz mekanis = 1800 rpm

WINDOW = 4096  # jendela pelatihan
STRIDE_DUR = WINDOW / FS  # 0,539 s
RES_BIN = FS / WINDOW  # 1,855 Hz

# Pita pencarian frekuensi poros. Sinkron 30 Hz; nominal 1715 rpm = 28,58 Hz pada beban
# penuh. Diberi margin ke bawah untuk berjaga-jaga.
PITA_POROS = (27.0, 30.2)

# Torsi nominal motor (spesifikasi IEEE DataPort: 1 hp, 4 kutub, 4,1 Nm, 1715 rpm)
T_NOM = 4.1
SLIP_NOM = (1800.0 - 1715.0) / 1800.0  # 0,04722

KANAL_UTAMA = "Vib_base"  # std terbesar (12,14) -> komponen poros paling kuat


def muat(f, kondisi, torsi, rep, kanal):
    ref = f[kondisi][torsi][kanal][rep, 0]
    return f[ref][:].flatten().astype(np.float64)


def onset_steady(f, kondisi, torsi, rep, ambang=0.5):
    """Indeks awal steady-state — falling edge Trigger pertama. Identik s2_steady_state."""
    try:
        trig = muat(f, kondisi, torsi, rep, "Trigger")
    except Exception:
        return 0
    bawah = np.where(trig < ambang)[0]
    return int(bawah[0]) if len(bawah) else 0


def puncak_poros(sinyal, fs, pita):
    """Kembalikan (f_puncak, prominensi) dari spektrum resolusi penuh di dalam pita."""
    sinyal = sinyal - sinyal.mean()
    spek = np.abs(np.fft.rfft(sinyal * np.hanning(len(sinyal))))
    freq = np.fft.rfftfreq(len(sinyal), 1.0 / fs)

    sel = (freq >= pita[0]) & (freq <= pita[1])
    if not sel.any():
        return np.nan, np.nan
    idx_lokal = int(np.argmax(spek[sel]))
    f_puncak = float(freq[sel][idx_lokal])
    amp = float(spek[sel][idx_lokal])

    # prominensi = puncak dibagi median amplitudo di sekitar pita (10-40 Hz)
    latar = (freq >= 10) & (freq <= 40)
    med = float(np.median(spek[latar])) or 1e-12
    return f_puncak, amp / med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kondisi", default="rs", help="daftar dipisah koma, mis. rs,r2b")
    ap.add_argument("--reps", type=int, default=5, help="repetisi per level torsi")
    ap.add_argument("--kanal", default=KANAL_UTAMA)
    args = ap.parse_args()

    kondisi_list = [k.strip() for k in args.kondisi.split(",")]
    hasil = {}  # torsi -> daftar (f_r, prominensi)

    for kond in kondisi_list:
        fp = RAW / f"struct_{kond}_R1.mat"
        if not fp.exists():
            print(f"  [LEWATI] {fp.name} tidak ada")
            continue
        print(f"memindai {fp.name} (kanal {args.kanal}) ...", flush=True)
        with h5py.File(fp, "r") as f:
            for torsi in sorted(f[kond].keys()):
                for rep in range(args.reps):
                    try:
                        sig = muat(f, kond, torsi, rep, args.kanal)
                        onset = onset_steady(f, kond, torsi, rep)
                        seg = sig[onset:]
                        if len(seg) < int(5 * FS):
                            continue
                        f_r, prom = puncak_poros(seg, FS, PITA_POROS)
                        hasil.setdefault(torsi, []).append((f_r, prom, len(seg) / FS))
                    except Exception as e:  # noqa: BLE001
                        print(f"    gagal {kond}/{torsi}/rep{rep}: {e}")

    if not hasil:
        raise SystemExit("tidak ada pengukuran")

    print()
    print("=" * 96)
    print(f"SLIP TERUKUR  (f_sync = {F_SYNC:.1f} Hz = 1800 rpm; kanal {args.kanal})")
    print("=" * 96)
    print(
        f"{'torsi':<10}{'Nm':>6}{'%beban':>8}{'f_r Hz':>10}{'rpm':>8}"
        f"{'slip':>9}{'2sf Hz':>9}{'periode s':>11}{'jendela/T':>11}{'prom':>7}"
    )

    ringkas = []
    for torsi in sorted(hasil):
        arr = np.array([h[0] for h in hasil[torsi]], float)
        prom = np.median([h[1] for h in hasil[torsi]])
        f_r = float(np.median(arr))
        nm = int(torsi.replace("torque", "")) / 10.0
        slip = (F_SYNC - f_r) / F_SYNC
        f2s = 2 * slip * F_LINE
        periode = 1.0 / f2s if f2s > 0 else np.inf
        print(
            f"{torsi:<10}{nm:>6.1f}{nm / T_NOM * 100:>7.0f}%{f_r:>10.3f}{f_r * 60:>8.0f}"
            f"{slip:>9.4f}{f2s:>9.2f}{periode:>11.3f}{STRIDE_DUR / periode:>11.2f}{prom:>7.1f}"
        )
        ringkas.append((torsi, nm, f_r, slip, f2s, periode))

    print()
    print("=" * 96)
    print("VONIS ATAS KLAIM SEC. V-B MAKALAH")
    print("=" * 96)
    print(f"  jendela pelatihan  = {WINDOW} sampel = {STRIDE_DUR:.3f} s")
    print(f"  resolusi bin       = {RES_BIN:.3f} Hz")
    print()

    for torsi, nm, f_r, slip, f2s, periode in ringkas:
        di_bawah_res = f2s < RES_BIN
        kurang_1_periode = STRIDE_DUR < periode
        catatan = []
        if di_bawah_res:
            catatan.append("2sf DI BAWAH resolusi bin")
        if kurang_1_periode:
            catatan.append("jendela < 1 periode")
        status = "  |  ".join(catatan) if catatan else "terselesaikan"
        print(f"  {torsi:<10} 2sf={f2s:6.2f} Hz  T={periode:6.3f} s   -> {status}")

    print()
    slip_terukur = np.array([r[3] for r in ringkas])
    nm_arr = np.array([r[1] for r in ringkas])
    slip_asumsi = SLIP_NOM * nm_arr / T_NOM
    print("  PEMBANDINGAN TERHADAP ASUMSI 'slip sebanding torsi' yang dipakai sebelum slip diukur:")
    print(f"    {'torsi':<10}{'diasumsikan':>14}{'terukur':>12}{'selisih':>12}")
    for (torsi, *_), sa, st in zip(ringkas, slip_asumsi, slip_terukur):
        print(f"    {torsi:<10}{sa:>14.4f}{st:>12.4f}{st - sa:>+12.4f}")

    if np.all(np.isfinite(slip_terukur)) and len(slip_terukur) > 2:
        r = np.corrcoef(nm_arr, slip_terukur)[0, 1]
        print(f"\n    korelasi torsi-slip terukur = {r:+.4f}"
              f"  (asumsi linear {'didukung' if r > 0.9 else 'TIDAK didukung'})")


if __name__ == "__main__":
    main()
