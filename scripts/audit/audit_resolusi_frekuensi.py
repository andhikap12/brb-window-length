"""audit_resolusi_frekuensi.py — jendela mana yang secara FISIK dapat memuat 2sf?

MENGAPA SKRIP INI ADA
=====================
Seluruh sapuan panjang jendela di proyek ini (W-P1, W-P1b, W-P1c) menanyakan
"apakah |d| naik". Tidak satu pun menanyakan yang lebih mendasar: pada panjang
jendela itu, apakah 2sf DAPAT diselesaikan sama sekali.

Itu bukan pertanyaan empiris. Ia aritmetika:

    T  = w / fs                 durasi jendela
    df = 1 / T = fs / w         lebar bin
    N  = 2sf * T = 2sf / df     berapa siklus modulasi termuat

Bila N < 1, jendela tidak memuat satu pun periode penuh modulasi. Tidak ada
metode spektral yang dapat memulihkan apa yang tidak terekam.

Skrip ini tidak memuat data dan tidak melatih apa pun. Ia mencetak batas fisik
untuk setiap kombinasi (panjang jendela, torsi) dan menandai mana di antara
tabel hasil proyek yang berdiri di atas kombinasi yang tidak terselesaikan.

EMPAT AMBANG, dinyatakan di muka
================================
R0  Batas keras, bebas metode      2sf >= 1*df      N >= 1
R1  Rayleigh, jendela persegi      2sf >= 2*df      N >= 2
R2  Lobe utama Hann                2sf >= 4*df      N >= 4
R3  Standar MCSA beban ringan      df <= 0,05 Hz

PEMBEDAAN YANG MENENTUKAN — R0 berlaku ke model, R1/R2 hanya ke metrik audit
============================================================================
R0 adalah satu-satunya ambang yang mengikat MODEL. Bila N < 1, jendela tidak
memuat satu pun periode penuh modulasi 2sf. Tidak ada metode apa pun, spektral
atau tidak, yang dapat memulihkan periodisitas yang tak terekam.

R1 dan R2 adalah batas untuk METRIK AUDIT kita, yang menghitung prominensi dari
FFT ber-jendela Hann. Model TIDAK menghitung FFT; ia melihat deret waktu lewat
backbone konvolusi/transformer dan dapat memanfaatkan varians amplitudo atau pola
temporal yang tidak menuntut keterpisahan spektral. Karena itu R1/R2 yang gagal
adalah PETUNJUK, bukan diskualifikasi, bagi angka model.

Membaca tabel ini sebagai "w4096 tidak sah" akan melebihi apa yang aritmetika ini
dukung. Yang ia dukung: pada w4096 torsi-0, model bekerja di bawah batas fisik
yang keras; pada torsi lain di w4096, model bekerja di wilayah tempat metrik
audit kita buta tetapi model belum tentu buta.

R1 adalah ambang yang dipakai `prominensi_envelope` (`f2s < 2.0*df -> NaN`).
Catat ketidakkonsistenan: fungsi itu mengalikan sinyal dengan `np.hanning`
sebelum FFT, sehingga lobe utamanya selebar 4 bin. Ambang yang konsisten dengan
jendela Hann adalah R2, bukan R1. R1 karena itu LONGGAR untuk jalur audit kita.

R3 berasal dari literatur MCSA untuk beban ringan (bin 0,01-0,05 Hz, rekaman
30-100+ detik). Ia tidak dapat dipenuhi dataset ini sama sekali: steady-state
terpendek 17,96 s memberi df terbaik 0,0557 Hz.
"""

from __future__ import annotations

import sys

FS = 7600.0
F_LINE = 60.0
SLIP = [0.0148, 0.0185, 0.0222, 0.0277, 0.0324, 0.0369, 0.0433, 0.0488]
WINDOWS = [4096, 8192, 16384, 32768, 65536, 131072]

# Panjang steady-state terpendek lintas seluruh rekaman (17,96 s; Sec. III-A makalah
# CENIM 2026), diturunkan oleh audit_steady_state.py.
N_STEADY = 136470

AMBANG_R3 = 0.05          # Hz, standar MCSA beban ringan

# Di mana tiap panjang jendela dipakai di proyek. Dipakai untuk menandai
# tabel mana yang terdampak.
PEMAKAIAN = {
    4096:   "jendela pendek: Tabel I, Sec. IV-A makalah",
    8192:   "hanya sapuan audit",
    16384:  "jendela panjang (overlap 75%): Sec. IV-B, Tabel II, LODO",
    32768:  "hanya sapuan audit",
    65536:  "hanya sapuan audit; tidak dipakai makalah",
    131072: "belum dipakai; batas fisik dataset",
}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass


def main() -> None:
    print("=" * 100)
    print("BATAS RESOLUSI FREKUENSI — aritmetika, bukan eksperimen")
    print("=" * 100)
    print(f"  fs = {FS:.0f} Hz | f_line = {F_LINE:.0f} Hz | steady-state terpendek "
          f"= {N_STEADY} sampel = {N_STEADY / FS:.2f} s")
    print(f"  df terbaik yang mungkin pada dataset ini = {FS / N_STEADY:.4f} Hz")
    print(f"  ambang R3 (literatur, beban ringan) = {AMBANG_R3} Hz  -> "
          f"{'TERPENUHI' if FS / N_STEADY <= AMBANG_R3 else 'TIDAK TERPENUHI oleh panjang rekaman'}")
    print()

    print("  2sf per torsi:")
    for it, s in enumerate(SLIP):
        print(f"    torsi-{it}  slip={s:.4f}  2sf={2 * s * F_LINE:>6.2f} Hz")
    print()

    for w in WINDOWS:
        T = w / FS
        df = FS / w
        muat_penuh = w <= N_STEADY
        print("=" * 100)
        print(f"w{w}   T = {T:.3f} s   df = {df:.4f} Hz"
              + ("" if muat_penuh else "   [TIDAK MUAT di steady-state terpendek]"))
        print(f"   dipakai: {PEMAKAIAN[w]}")
        print(f"   {'torsi':<8} {'2sf (Hz)':>9} {'N siklus':>9} {'R0 (N>=1)':>11} "
              f"{'R1 (N>=2)':>11} {'R2 (N>=4)':>11} {'R3 (df<=0,05)':>14}")
        for it, s in enumerate(SLIP):
            f2s = 2 * s * F_LINE
            N = f2s / df
            r0 = "lolos" if N >= 1.0 else "GAGAL KERAS"
            r1 = "lolos" if N >= 2.0 else "GAGAL"
            r2 = "lolos" if N >= 4.0 else "GAGAL"
            r3 = "lolos" if df <= AMBANG_R3 else "GAGAL"
            print(f"   torsi-{it:<2} {f2s:>9.2f} {N:>9.2f} {r0:>11} "
                  f"{r1:>11} {r2:>11} {r3:>14}")
        print()

    print("=" * 100)
    print("RINGKASAN — torsi terendah yang terselesaikan pada tiap jendela")
    print("=" * 100)
    print(f"   {'w':>8} {'df (Hz)':>9}  {'R1 mulai torsi':>15} {'R2 mulai torsi':>15}")
    for w in WINDOWS:
        df = FS / w
        t_r1 = next((it for it, s in enumerate(SLIP) if 2 * s * F_LINE / df >= 2.0), None)
        t_r2 = next((it for it, s in enumerate(SLIP) if 2 * s * F_LINE / df >= 4.0), None)
        print(f"   {w:>8} {df:>9.4f}  {str(t_r1) if t_r1 is not None else 'tidak ada':>15} "
              f"{str(t_r2) if t_r2 is not None else 'tidak ada':>15}")

    print()
    print("=" * 100)
    print("TABEL PROYEK YANG TERDAMPAK")
    print("=" * 100)
    for w in WINDOWS:
        df = FS / w
        gagal_r0 = [it for it, s in enumerate(SLIP) if 2 * s * F_LINE / df < 1.0]
        gagal_r1 = [it for it, s in enumerate(SLIP) if 2 * s * F_LINE / df < 2.0]
        gagal_r2 = [it for it, s in enumerate(SLIP) if 2 * s * F_LINE / df < 4.0]
        if not gagal_r2:
            continue
        print(f"   w{w}  ({PEMAKAIAN[w]})")
        if gagal_r0:
            print(f"      MENGIKAT MODEL — di bawah batas keras R0: torsi {gagal_r0}")
        if gagal_r1:
            print(f"      metrik audit buta pada R1 longgar:        torsi {gagal_r1}")
        print(f"      metrik audit buta pada R2 konsisten-Hann: torsi {gagal_r2}")
    print()
    print("  Hanya baris 'MENGIKAT MODEL' yang membatasi angka model. Dua baris lain")
    print("  membatasi metrik audit ini sendiri; lihat docstring.")
    print("=" * 100)


if __name__ == "__main__":
    main()
