"""make_subsample_w4096.py — KONTROL UJI C: w4096 disubsampel ke jumlah window w16384.

MASALAH YANG DIPECAHKAN
Uji C mengganti jendela 4096 -> 16384 dan menemukan torsi-0 naik +7,00 poin sementara
torsi-7 turun -2,24. Tetapi perlakuannya mengubah DUA hal sekaligus:

    (a) resolusi frekuensi   1,855 Hz -> 0,464 Hz   (yang ingin diuji, M14)
    (b) jumlah window latih  13.200   -> 3.200      (efek samping, 4x lebih sedikit data)

Torsi-7 tidak butuh (a) sama sekali — pada 4,0 Nm jendela 4096 sudah memuat 3,16 periode
kerusakan — sehingga ia hanya menanggung rugi dari (b). Itu menjelaskan penurunannya dan
ledakan variansnya (sd 0,73 -> 4,09; uji-F p = 0,0002). Selama (a) dan (b) bergerak
bersamaan, Uji C tidak dapat dipertahankan di sidang.

YANG DIBUAT
Salinan dataset w4096 yang berisi 3.200 window — jumlah, struktur rekaman, dan cakupan
waktu yang SAMA PERSIS dengan lengan w16384 — tetapi resolusi frekuensi tetap rendah.

    w4096 penuh      13.200 window (33/rekaman)   resolusi rendah
    w4096 subsampel   3.200 window ( 8/rekaman)   resolusi rendah   <- dibuat di sini
    w16384            3.200 window ( 8/rekaman)   resolusi tinggi

Perbandingan w16384 vs w4096-subsampel mengisolasi (a) dengan (b) terkendali.

CARA MEMILIH WINDOW — berimpit, bukan sekadar merata
Tiap rekaman punya 33 window di w4096 dan 8 window di w16384. Window w16384 ke-i
mencakup sampel [i*16384, (i+1)*16384). Yang diambil adalah window w4096 di indeks
j = i*4, yaitu tepat 4096 sampel PERTAMA dari tiap blok 16384. Jadi kedelapan window
subsampel duduk di titik waktu yang sama dengan kedelapan window w16384 — bukan sekadar
tersebar merata. Ini menghapus kemungkinan bahwa selisihnya berasal dari bagian rekaman
yang berbeda.

    ${PYTHON:-python} -u scripts/make_subsample_w4096.py

Skrip ini diparameterkan (--src, --ref, --dst, --faktor); nilai default adalah kontrol
volume pada Sec. IV-B makalah CENIM 2026.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

MOTOR = PROJECT_ROOT / "data" / "processed" / "motor"


def main(SRC: str, REF: str, DST: str, FAKTOR: int) -> None:
    print("=" * 78)
    print(f"KONTROL — subsampel {SRC}")
    print(f"          agar jumlah & posisi window == {REF}")
    print("=" * 78)

    d = np.load(MOTOR / f"{SRC}.npz", allow_pickle=True)
    si = np.load(MOTOR / f"{SRC}_split_info.npz")
    X, y = d["X"], d["y"]
    tq, rep = si["torque"], si["repetition"]

    ref = np.load(MOTOR / f"{REF}_split_info.npz")
    n_ref = len(ref["torque"])

    # Rekaman = (kelas, torsi, repetisi) — definisi yang sama dengan create_recording_ids.
    key = np.stack([y, tq, rep], axis=1)
    uniq = np.unique(key, axis=0)
    print(f"  sumber   : {X.shape[0]} window, {len(uniq)} rekaman")

    pilih: list[int] = []
    per_rekaman = set()
    for k in uniq:
        idx = np.where((key == k).all(axis=1))[0]
        # idx sudah urut waktu (window dibuat berurutan oleh preprocessor).
        ambil = idx[:: FAKTOR][: n_ref // len(uniq)]
        per_rekaman.add(len(ambil))
        pilih.extend(ambil.tolist())

    pilih = np.sort(np.array(pilih))
    print(f"  diambil  : indeks j = 0, {FAKTOR}, {2*FAKTOR}, ... (awal tiap blok 16384)")
    print(f"  window/rekaman setelah subsampel: {sorted(per_rekaman)}")

    assert len(per_rekaman) == 1, f"jumlah window per rekaman tidak seragam: {per_rekaman}"
    assert len(pilih) == n_ref, f"total {len(pilih)} != acuan w16384 {n_ref}"

    Xs, ys, tqs, reps = X[pilih], y[pilih], tq[pilih], rep[pilih]

    # Verifikasi struktur benar-benar cocok dengan lengan pembanding.
    print()
    print("  VERIFIKASI terhadap w16384:")
    ok = True
    for nama, a, b in (("jumlah window", len(ys), n_ref),
                       ("jumlah rekaman", len(np.unique(np.stack([ys, tqs, reps], 1), axis=0)), 400)):
        tanda = "OK " if a == b else "BEDA"
        ok &= a == b
        print(f"    [{tanda}] {nama:16s} {a} vs {b}")
    for nama, a, b in (("window per torsi", np.bincount(tqs), np.bincount(ref["torque"])),
                       ("window per kelas", np.bincount(ys), np.bincount(np.load(MOTOR / f'{REF}.npz')["y"]))):
        sama = bool(np.array_equal(a, b))
        ok &= sama
        print(f"    [{'OK ' if sama else 'BEDA'}] {nama:16s} {a.tolist()} vs {b.tolist()}")
    if not ok:
        raise SystemExit("[ABORT] struktur tidak cocok dengan lengan w16384.")

    np.savez_compressed(MOTOR / f"{DST}.npz", X=Xs, y=ys)
    np.savez(MOTOR / f"{DST}_split_info.npz", torque=tqs, repetition=reps)
    print()
    print(f"  tersimpan: {DST}.npz  X={Xs.shape}")
    print(f"             {DST}_split_info.npz")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Subsampel dataset agar jumlah & posisi window cocok dengan acuan")
    ap.add_argument("--src", default="motor_multiclass_w4096_s4096_ss_env",
                    help="dataset yang disubsampel")
    ap.add_argument("--ref", default="motor_multiclass_w16384_s16384_ss_env",
                    help="acuan yang jumlah windownya ditiru")
    ap.add_argument("--dst", default="motor_multiclass_w4096_s4096_ss_env_sub3200",
                    help="nama keluaran")
    ap.add_argument("--faktor", type=int, default=4,
                    help="stride acuan dibagi stride sumber; menentukan langkah "
                         "pengambilan agar window subsampel BERIMPIT dengan acuan")
    a = ap.parse_args()
    main(a.src, a.ref, a.dst, a.faktor)
