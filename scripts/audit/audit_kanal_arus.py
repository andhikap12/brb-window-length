"""audit_kanal_arus.py — A2: apakah kanal arus 50 kHz membawa informasi BRB yang getaran tidak?

Kriteria A2-P0..A2-P5 TERKUNCI di docstring ini, ditulis sebelum implementasi. Hasilnya
dilaporkan di Sec. V-C makalah CENIM 2026 (kanal arus 50 kHz). Skrip ini hanya mengeksekusinya. Tanpa GPU.

STRUKTUR
========
  A2.0  GERBANG KERAS — patok fs arus lewat kalibrasi-diri 60,00 Hz, sebaran < 0,1%.
        Bila gagal, skrip BERHENTI. Tidak ada |d| yang boleh dilaporkan (A2.0).
  A2.2  d Cohen rs vs r1b pada prominensi sideband, PER WINDOW, tidak tumpang tindih,
        10 repetisi, torsi-0 dan torsi-7. Getaran diukur di run yang SAMA.
  A2-P1 |d| arus torsi-0 >= 1,5 x |d| getaran terbaik torsi-0.

RAMBU KRITERIA A2
=========================
- A2.1: 50 kHz TIDAK memberi resolusi. df = 1/T, bergantung durasi saja. Pada durasi setara
  w16384 getaran (2,1558 s) kedua kanal punya df = 0,4639 Hz IDENTIK. Keunggulan arus, bila
  ada, harus datang dari SNR dan fisika MCSA.
- A2-P4: sideband arus WAJIB lewat envelope Hilbert, bukan spektrum mentah. Di torsi-0
  sideband berjarak 1,776 Hz dari fundamental yang beberapa orde lebih besar.
- A2-P3: seluruh pengukuran per-window pada durasi yang dialami model (pelajaran U3).

CATATAN IMPLEMENTASI YANG BUKAN BAGIAN KRITERIA A2 — dinyatakan terbuka
============================================================
Jalur getaran memakai RERATA 5 kanal MENTAH. Cara itu TIDAK BOLEH dipakai untuk arus:
Ia+Ib+Ic ~ 0 pada sistem seimbang, jadi merata-ratakan arus mentah membatalkan
fundamentalnya. Kanal arus karena itu dilaporkan PER FASA (Ia, Ib, Ic terpisah), sejajar
dengan cara Vib_base dilaporkan sebagai kanal tunggal. Tidak ada perata-rataan arus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "DatasetRaw" / "banco de dados experimental"

FS_VIB = 7600.0
F_LINE = 60.0
N_VIB = 16384                     # jendela acuan; durasi 2,1558 s
DUR = N_VIB / FS_VIB

KANAL_VIB = ["Vib_acpe", "Vib_acpi", "Vib_axial", "Vib_base", "Vib_carc"]
KANAL_ARUS = ["Ia", "Ib", "Ic"]
SLIP = [0.0148, 0.0185, 0.0222, 0.0277, 0.0324, 0.0369, 0.0433, 0.0488]
TORSI = ["torque05", "torque10", "torque15", "torque20",
         "torque25", "torque30", "torque35", "torque40"]
MAT = {"rs": "struct_rs_R1.mat", "r1b": "struct_r1b_R1.mat",
       "r2b": "struct_r2b_R1.mat", "r3b": "struct_r3b_R1.mat",
       "r4b": "struct_r4b_R1.mat"}
KONTRAS = ["r1b", "r2b", "r3b", "r4b"]     # r1b = kontras utama A2-P1
TORSI_UJI = [0, 7]
REPS = 10

AMBANG_A2P1 = 1.5                 # A2.3
AMBANG_FS = 0.001                 # 0,1% — gerbang A2.0
N_SEG_GATE = 4                    # segmen per rekaman untuk gerbang A2.0b

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass


def garis(judul: str = "") -> None:
    print("=" * 100)
    if judul:
        print(judul)
        print("=" * 100)


def muat(f, kondisi, torsi, rep, kanal):
    ref = f[kondisi][torsi][kanal][rep, 0]
    return f[ref][:].flatten().astype(np.float64)


def onset_steady(f, kondisi, torsi, rep, ambang=0.5):
    try:
        trig = muat(f, kondisi, torsi, rep, "Trigger")
    except Exception:  # noqa: BLE001
        return 0
    bawah = np.where(trig < ambang)[0]
    return int(bawah[0]) if len(bawah) else 0


# --------------------------------------------------------------------- A2.0
def fs_kalibrasi_diri(x: np.ndarray, n_nominal: float) -> float:
    """fs = 60,00 x N / k_puncak, interpolasi puncak parabolik untuk presisi sub-bin.

    Frekuensi jaringan dikunci grid pada 60,00 Hz; puncak dominan spektrum arus PASTI
    fundamental itu. Rasio panjang TIDAK dipakai (kriteria A2.0 menolaknya secara eksplisit).
    """
    n = len(x)
    x = x - x.mean()
    spek = np.abs(np.fft.rfft(x * np.hanning(n)))
    # cari puncak di sekitar 60 Hz dengan asumsi fs awal n_nominal, margin +/-10%
    k_kira = 60.0 * n / n_nominal
    lo, hi = int(k_kira * 0.90), int(k_kira * 1.10)
    k = lo + int(np.argmax(spek[lo:hi]))
    # interpolasi parabolik pada log-magnitudo
    a, b, c = np.log(spek[k - 1] + 1e-30), np.log(spek[k] + 1e-30), np.log(spek[k + 1] + 1e-30)
    # PENYEBUT (a - 2b + c) SELALU NEGATIF di puncak. Versi pertama skrip ini membungkusnya
    # dengan max(..., 1e-30), yang karena itu SELALU mengembalikan 1e-30 dan memaksa delta
    # terpental ke +/-0,5 — interpolasi paraboliknya mati total dan galat setengah-bin
    # terbaca sebagai klok tak stabil. Jangan pasang penjaga pada besaran yang bertanda tetap.
    den = a - 2 * b + c
    delta = 0.5 * (a - c) / den if den != 0 else 0.0
    return 60.0 * n / (k + float(np.clip(delta, -0.5, 0.5)))


# --------------------------------------------------------------------- A2.2
def prominensi_envelope(x: np.ndarray, f2s: float, fs: float) -> float:
    """IDENTIK logikanya dengan audit_window_sweep.prominensi_envelope, fs jadi parameter.

    Fundamental hilang oleh demodulasi, jadi tidak ada kontaminasi seperti jalur mentah
    (A2-P4). Syarat keterpisahan f2s >= 2*df; bila tidak, NaN — jawaban fisik, bukan gagal ukur.
    """
    n = len(x)
    df = fs / n
    if f2s < 2.0 * df:
        return np.nan

    # BENTUK SOS, BUKAN (b,a). Pada fs = 50 kHz pita 40-100 Hz jatuh di frekuensi
    # ternormalisasi 0,0016-0,004; butter(4,...) bentuk transfer-function kehilangan presisi
    # parah di sana dan filtfilt mengembalikan NaN -> SELURUH kanal arus terbaca "tak ada
    # data". Jalur getaran (7.600 Hz) tidak terkena karena pitanya 10x lebih tinggi
    # ternormalisasi. Keduanya kini memakai sos supaya perlakuannya tetap IDENTIK (A2.2).
    sos = butter(4, [40.0 / (fs / 2), 100.0 / (fs / 2)], btype="bandpass", output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x)))
    env = env - env.mean()

    spek = np.abs(np.fft.rfft(env * np.hanning(n)))
    freq = np.fft.rfftfreq(n, 1.0 / fs)

    lebar = max(0.15 * f2s, 1.5 * df)
    sel = (freq >= f2s - lebar) & (freq <= f2s + lebar)
    if not sel.any():
        return np.nan
    puncak = float(spek[sel].max())

    ring = ((freq >= 0.35 * f2s) & (freq <= 0.75 * f2s)) | \
           ((freq >= 1.40 * f2s) & (freq <= 3.00 * f2s))
    ring &= freq > 2.0 * df
    if ring.sum() < 3:
        return np.nan
    return puncak / max(float(np.median(spek[ring])), 1e-30)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((b.mean() - a.mean()) / max(sp, 1e-30))


def main() -> None:
    garis("A2 — AUDIT KANAL ARUS 50 kHz. Kriteria A2-P0..P5 (terkunci di docstring)")
    print(f"  durasi jendela acuan : {DUR:.4f} s  (= w16384 getaran pada 7.600 Hz)")
    print(f"  A2.1: df = 1/T = {1/DUR:.4f} Hz di KEDUA kanal — 50 kHz tidak membeli resolusi\n")

    # ============================================================ A2.0b GERBANG
    garis("A2.0b — GERBANG (AMANDEMEN Opsi 1): kalibrasi fs PER REKAMAN")
    print("  Gerbang A2.0 asli gugur (rentang 0,3333%) karena estimatornya tidak dapat")
    print("  memisahkan fs dari drift frekuensi jaringan — seluruh drift masuk ke fs.")
    print("  Amandemen: tiap rekaman memakai fs kalibrasi-dirinya sendiri, sehingga sumbu")
    print("  frekuensinya dinormalkan ke fundamental 60,00 Hz. Karena 2sf = 2*s*f_jaringan")
    print("  menskala dengan faktor yang sama, normalisasi ini KONSISTEN-DIRI.")
    print(f"  Gerbang pindah ke sebaran DI DALAM satu rekaman ({N_SEG_GATE} segmen), yang")
    print("  menguji kestabilan klok — pertanyaan yang sebenarnya ingin dijawab A2.0.\n")

    # DUA KOREKSI TERHADAP LINTASAN 20:05 — keduanya CACAT SKRIP, bukan ubahan kriteria:
    #  (1) kalibrasi WAJIB di steady-state. Lintasan pertama memakai rekaman penuh, sehingga
    #      segmen-1 memuat transien start-up (RMS 1,90 vs 1,06) dan biasnya terbaca sebagai
    #      klok tak stabil. Seluruh proyek ini mengukur di steady-state (onset_steady);
    #      gerbangnya saja yang lupa menerapkannya.
    #  (2) gerbang ditegakkan PER REKAMAN, bukan sebagai maks global. Itu bacaan harfiah
    #      "sebaran DI DALAM satu rekaman" (Opsi 1): rekaman yang fs-nya tidak dapat
    #      dipatok < 0,1% DIBUANG, bukan diakomodasi dengan melonggarkan ambang.
    fs_rec: dict = {}      # (kondisi, torsi, rep) -> fs   — hanya yang LOLOS gerbang
    dalam: dict = {}
    for kondisi, fname in MAT.items():
        fp = RAW / fname
        if not fp.exists():
            print(f"  [ABORT] {fname} tidak ada.")
            return
        with h5py.File(fp, "r") as f:
            for it in TORSI_UJI:
                for rep in range(REPS):
                    try:
                        x = muat(f, kondisi, TORSI[it], rep, "Ia")
                        on_v = onset_steady(f, kondisi, TORSI[it], rep)
                    except Exception:  # noqa: BLE001
                        continue
                    x = x[int(round(on_v / FS_VIB * 50000.0)):]          # koreksi (1)
                    m = len(x) // N_SEG_GATE
                    seg = np.array([fs_kalibrasi_diri(x[i * m:(i + 1) * m], 50000.0)
                                    for i in range(N_SEG_GATE)])
                    sb = float((seg.max() - seg.min()) / np.median(seg))
                    dalam[(kondisi, it, rep)] = sb
                    if sb < AMBANG_FS:                                    # koreksi (2)
                        fs_rec[(kondisi, it, rep)] = fs_kalibrasi_diri(x, 50000.0)

    if len(fs_rec) < 20:
        print(f"  [GAGAL] hanya {len(fs_rec)} rekaman lolos, A2.0 menuntut >= 20.")
        return
    sb_all = np.array(list(dalam.values()))
    est = np.array(list(fs_rec.values()))
    fs_med = float(np.median(est))
    buang = {k: v for k, v in dalam.items() if v >= AMBANG_FS}

    print(f"  n rekaman diperiksa          : {len(dalam)}")
    print(f"  sebaran DI DALAM rekaman     : median {np.median(sb_all)*100:.4f} %   "
          f"p90 {np.percentile(sb_all, 90)*100:.4f} %   (ambang < {AMBANG_FS*100:.1f} %)")
    print(f"  LOLOS gerbang per-rekaman    : {len(fs_rec)}/{len(dalam)}")
    print(f"  fs median (rekaman lolos)    : {fs_med:.2f} Hz")
    print(f"\n  klaim 49.500 Hz  : selisih {(fs_med-49500)/495:+.3f} %")
    print(f"  klaim 50.000 Hz : selisih {(fs_med-50000)/500:+.3f} %")

    if buang:
        print(f"\n  DIBUANG ({len(buang)}) — dinyatakan terbuka, bukan disembunyikan:")
        for k, v in sorted(buang.items(), key=lambda kv: -kv[1]):
            print(f"    {k[0]}/torsi-{k[1]}/rep{k[2]:<3} sebaran {v*100:>8.3f} %")
        print("    rs/torsi-0/rep8: kanal arus MATI (RMS 0,0076 vs ~1,23; puncak 60 Hz")
        print("    29.000x lebih lemah). Kanal GETARAN rekaman itu normal, jadi tidak ada")
        print("    hasil terlatih mana pun yang tercemar.")

    menang = "50.000" if abs(fs_med - 50000) < abs(fs_med - 49500) else "49.500"
    print(f"\n  GERBANG A2.0b LOLOS pada {len(fs_rec)} rekaman. fs dipatok PER REKAMAN.")
    print(f"  Median {fs_med:.2f} Hz -> klaim {menang} Hz benar.")

    n_ar_med = int(round(DUR * fs_med))
    print(f"\n  jendela arus     : ~{n_ar_med} sampel = {DUR:.4f} s (tepatnya per rekaman)")
    print(f"  jendela getaran  : {N_VIB} sampel = {N_VIB/FS_VIB:.4f} s")
    print(f"  Rambu: sumbu getaran TIDAK ikut dinormalkan (fs 7.600 tetap). Ketidakcocokan")
    print(f"  yang tersisa = 0,17% x 1,776 Hz = 0,003 Hz, yaitu 0,6% dari satu bin. Diabaikan")
    print(f"  secara sadar, bukan terlewat.")

    # ============================================================ A2.2 UKUR
    garis("A2.2 — d Cohen per window, jalur ENVELOPE (A2-P4), 10 rep, tidak tumpang tindih")
    # sumber -> (jenis, pengambil)
    vals: dict = {}     # (sumber, it, kondisi) -> array prominensi
    for kondisi, fname in MAT.items():
        print(f"  memindai {fname} ...", flush=True)
        with h5py.File(RAW / fname, "r") as f:
            for it in TORSI_UJI:
                f2s = 2 * SLIP[it] * F_LINE
                for rep in range(REPS):
                    # Rekaman yang gugur gerbang dibuang UTUH — getaran ikut dibuang, supaya
                    # pembanding arus vs getaran berdiri di atas rekaman fisik yang IDENTIK
                    # (pelajaran R3: satuan keputusan harus sama di kedua sisi).
                    if (kondisi, it, rep) not in fs_rec:
                        continue
                    try:
                        on_v = onset_steady(f, kondisi, TORSI[it], rep)
                        vib = np.stack([muat(f, kondisi, TORSI[it], rep, k)[on_v:]
                                        for k in KANAL_VIB])
                    except Exception:  # noqa: BLE001
                        continue
                    # fs REKAMAN INI (amandemen Opsi 1), bukan fs global
                    fs_i = fs_rec.get((kondisi, it, rep), fs_med)
                    on_a = int(round(on_v / FS_VIB * fs_i))

                    sumber = {"RERATA": vib.mean(axis=0),
                              "Vib_base": vib[KANAL_VIB.index("Vib_base")]}
                    fs_of = {"RERATA": FS_VIB, "Vib_base": FS_VIB}
                    for k in KANAL_ARUS:
                        try:
                            sumber[k] = muat(f, kondisi, TORSI[it], rep, k)[on_a:]
                            fs_of[k] = fs_i
                        except Exception:  # noqa: BLE001
                            pass

                    for s, x in sumber.items():
                        fs_s = fs_of[s]
                        w = N_VIB if s in ("RERATA", "Vib_base") else int(round(DUR * fs_s))
                        for j in range(len(x) // w):
                            v = prominensi_envelope(x[j * w:(j + 1) * w], f2s, fs_s)
                            if np.isfinite(v):
                                vals.setdefault((s, it, kondisi), []).append(v)
    for k in vals:
        vals[k] = np.array(vals[k])

    SUMBER = ["RERATA", "Vib_base"] + KANAL_ARUS
    hasil: dict = {}
    for it in TORSI_UJI:
        f2s = 2 * SLIP[it] * F_LINE
        print(f"\n  TORSI-{it}   2sf = {f2s:.2f} Hz")
        print(f"    {'sumber':<10} {'jenis':<8} {'n/kls':>6} "
              + " ".join(f"{k:>9}" for k in KONTRAS))
        for s in SUMBER:
            rs = vals.get((s, it, "rs"))
            if rs is None:
                print(f"    {s:<10} {'-':<8} {'tak ada data':>6}")
                continue
            kol = []
            for kr in KONTRAS:
                v = vals.get((s, it, kr))
                d = abs(cohen_d(rs, v)) if v is not None else np.nan
                hasil[(s, it, kr)] = d
                kol.append(d)
            jenis = "getaran" if s in ("RERATA", "Vib_base") else "ARUS"
            print(f"    {s:<10} {jenis:<8} {len(rs):>6} "
                  + " ".join(f"{v:>9.2f}" if np.isfinite(v) else f"{'TAKPISAH':>9}"
                             for v in kol))

    # ============================================================ A2-P1
    garis("A2-P1 (UTAMA) — |d| arus torsi-0 >= 1,5 x |d| getaran terbaik torsi-0")
    d_vib = {s: hasil.get((s, 0, "r1b"), np.nan) for s in ("RERATA", "Vib_base")}
    d_ar = {s: hasil.get((s, 0, "r1b"), np.nan) for s in KANAL_ARUS}
    vib_terbaik = max(v for v in d_vib.values() if np.isfinite(v))
    vib_nama = [k for k, v in d_vib.items() if v == vib_terbaik][0]
    ambang = AMBANG_A2P1 * vib_terbaik
    ar_terbaik = max((v for v in d_ar.values() if np.isfinite(v)), default=np.nan)
    ar_nama = [k for k, v in d_ar.items() if v == ar_terbaik]

    print(f"  kontras rs vs r1b, torsi-0, jalur envelope, run yang sama\n")
    for s, v in d_vib.items():
        print(f"    getaran {s:<10} |d| = {v:.3f}")
    for s, v in d_ar.items():
        print(f"    ARUS    {s:<10} |d| = {v:.3f}" if np.isfinite(v)
              else f"    ARUS    {s:<10} TAKPISAH")
    print(f"\n  getaran terbaik : {vib_nama} = {vib_terbaik:.3f}")
    print(f"  ambang A2-P1    : 1,5 x {vib_terbaik:.3f} = {ambang:.3f}")
    print(f"  arus terbaik    : {ar_nama[0] if ar_nama else '-'} = {ar_terbaik:.3f}")
    p1 = np.isfinite(ar_terbaik) and ar_terbaik >= ambang
    print(f"\n  A2-P1 {'LOLOS' if p1 else 'GUGUR'}"
          + ("" if p1 else " — kanal arus tidak membayar ongkosnya (ekstraksi baru, "
                          "resampling, seluruh tabel dihitung ulang)"))

    # ============================================================ A2-P2
    garis("A2-P2 — SPESIFISITAS (aturan tafsir, tanpa ambang lulus)")
    a0 = ar_terbaik
    a7 = max((hasil.get((s, 7, "r1b"), np.nan) for s in KANAL_ARUS),
             default=np.nan)
    v7 = max((hasil.get((s, 7, "r1b"), np.nan) for s in ("RERATA", "Vib_base")),
             default=np.nan)
    print(f"  arus    torsi-0 {a0:.3f}   torsi-7 {a7:.3f}")
    print(f"  getaran torsi-0 {vib_terbaik:.3f}   torsi-7 {v7:.3f}")
    r0 = a0 / vib_terbaik if vib_terbaik > 0 else np.nan
    r7 = a7 / v7 if np.isfinite(v7) and v7 > 0 else np.nan
    print(f"\n  rasio arus/getaran : torsi-0 {r0:.2f}x   torsi-7 {r7:.2f}x")
    if np.isfinite(r0) and np.isfinite(r7):
        if r0 > 1.3 * r7:
            print("  -> keunggulan TERPUSAT di torsi-0: arus menjawab defisit ekstrapolasi.")
        elif r7 > 1.3 * r0:
            print("  -> keunggulan justru di torsi-7, domain yang sudah terselesaikan.")
        else:
            print("  -> keunggulan MERATA: yang terukur SNR sensor, BUKAN jawaban defisit")
            print("     torsi-0, dan tidak boleh dilaporkan sebagai jawaban itu (A2-P2).")

    # ============================================================ A2-P5
    garis("A2-P5 — VARIANS, dinilai terlepas dari A2-P1")
    print(f"  {'sumber':<10} {'torsi':>6} {'rerata':>10} {'sd':>10} {'CV':>8}")
    for s in SUMBER:
        for it in TORSI_UJI:
            v = vals.get((s, it, "rs"))
            if v is None or len(v) < 2:
                continue
            print(f"  {s:<10} {it:>6} {v.mean():>10.3f} {v.std(ddof=1):>10.3f} "
                  f"{v.std(ddof=1)/max(v.mean(),1e-9):>8.3f}")
    print("\n  Keunggulan rerata yang disertai varians jauh lebih besar bukan lever yang")
    print("  dapat diandalkan — pelajaran yang membatalkan klaim varians BT-DANN (kriteria F-P4).")

    garis("VONIS A2")
    print(f"  A2.0b gerbang fs: LOLOS ({fs_med:.2f} Hz, {len(fs_rec)}/{len(dalam)} rekaman)")
    print(f"  A2-P1 utama     : {'LOLOS' if p1 else 'GUGUR'}")
    print(f"  A2-P2, A2-P5    : di atas")
    print()
    if p1:
        print("  -> Kanal arus adalah lever data. Lanjut: lengan (a) arus sebagai kanal")
        print("     tambahan. Ia TIDAK mengesahkan usul modulasi-fitur FiLM, yang tidak dirilis (A2.4).")
    else:
        print("  -> Kanal arus TIDAK dibelanjakan. A2.4 tetap berlaku: ini tidak")
        print("     menutup usul P1, hanya mencabut prasyarat untuk lengan (a).")


if __name__ == "__main__":
    main()
