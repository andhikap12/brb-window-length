"""audit_dataset.py — Audit independen ekstraksi Dataset A (.mat v7.3 / HDF5).

Rekonstruksi ulang dari .mat MENTAH memakai hanya h5py + scipy + numpy, TANPA
memanggil satu pun fungsi pipeline proyek, lalu dibandingkan dengan .npz tersimpan.
Kalau cocok sampai presisi float32, ekstraksinya sahih.

Yang diperiksa:
  A. Konsistensi metadata (jumlah rekaman, window/rekaman, label, torsi, repetisi)
  B. Rekonstruksi bit-level untuk beberapa rekaman di posisi berbeda
  C. Laju sampling (dari rasio panjang kanal vibrasi vs listrik)
  D. Skala antar-kanal (relevan untuk isu standardisasi by_var)
  E. Kebocoran: apakah ada window identik lintas-rekaman
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

RAW = Path(__file__).resolve().parents[2] / "DatasetRaw" / "banco de dados experimental"
PROC = Path(__file__).resolve().parents[2] / "data" / "processed" / "motor"
NPZ = PROC / "motor_multiclass_w4096_s4096_ss_env.npz"
SPLIT = PROC / "motor_multiclass_w4096_s4096_ss_env_split_info.npz"

MAT = {"rs": "struct_rs_R1.mat", "r1b": "struct_r1b_R1.mat", "r2b": "struct_r2b_R1.mat",
       "r3b": "struct_r3b_R1.mat", "r4b": "struct_r4b_R1.mat"}
LABEL = {"rs": 0, "r1b": 1, "r2b": 2, "r3b": 3, "r4b": 4}
TORQUES = [f"torque{t:02d}" for t in (5, 10, 15, 20, 25, 30, 35, 40)]
VIB = ["Vib_acpe", "Vib_acpi", "Vib_axial", "Vib_base", "Vib_carc"]
FS, W, S = 7600, 4096, 4096
BLO, BHI, ORDER = 40.0, 100.0, 4


def rebuild(cond: str, tq: str, rep: int):
    """Rekonstruksi independen satu rekaman -> (n_win, 6, 4096) float32."""
    with h5py.File(RAW / MAT[cond], "r") as f:
        g = f[cond][tq]
        chans = [np.array(f[g[c][rep, 0]]).ravel().astype(np.float32) for c in VIB]
        trig = np.array(f[g["Trigger"][rep, 0]]).ravel().astype(np.float32)
    n = min(c.size for c in chans)
    x = np.stack([c[:n] for c in chans], axis=0)              # (5, n)
    below = np.where(trig < 0.5)[0]
    onset = int(below[0]) if below.size else 0
    x = x[:, onset:n]                                          # trim_to_onset
    # envelope slip-band pada sinyal TER-TRIM
    mean_sig = x.mean(axis=0).astype(np.float64)
    b, a = butter(ORDER, [BLO / (FS / 2), BHI / (FS / 2)], btype="bandpass")
    env = np.abs(hilbert(filtfilt(b, a, mean_sig))).astype(np.float32)
    x = np.concatenate([x, env[None, :]], axis=0)              # (6, n')
    nw = (x.shape[1] - W) // S + 1
    return np.stack([x[:, i * S:i * S + W] for i in range(nw)]), onset, n


def main() -> None:
    X, y = None, None
    with np.load(NPZ) as z:
        keys = list(z.keys())
        X = z[keys[0]] if "X" not in z else z["X"]
        y = z["y"] if "y" in z else z[keys[1]]
    si = np.load(SPLIT)
    torque, reps = si["torque"], si["repetition"]

    print("=" * 88)
    print("A. KONSISTENSI METADATA")
    print("=" * 88)
    print(f"  X = {X.shape} {X.dtype}   y = {y.shape}, kelas {sorted(set(y.tolist()))}")
    print(f"  torque unik    = {sorted(set(torque.tolist()))}")
    print(f"  repetisi unik  = {sorted(set(reps.tolist()))}")
    rid = np.stack([y, torque, reps], axis=1)
    _, idx, cnt = np.unique(rid, axis=0, return_index=True, return_counts=True)
    print(f"  rekaman unik   = {len(idx)}  (harapan 5x8x10 = 400)")
    print(f"  window/rekaman : min={cnt.min()} max={cnt.max()} "
          f"{'SERAGAM' if cnt.min() == cnt.max() else '<< TIDAK SERAGAM'}")
    for k, nm in [(y, "kelas"), (torque, "torsi"), (reps, "repetisi")]:
        u, c = np.unique(k, return_counts=True)
        print(f"  saldo {nm:9}: {dict(zip(u.tolist(), c.tolist()))}")

    print("\n" + "=" * 88)
    print("B. REKONSTRUKSI INDEPENDEN vs .npz TERSIMPAN")
    print("=" * 88)
    conds = list(MAT.keys())
    cases = [("rs", "torque05", 0), ("rs", "torque40", 9),
             ("r2b", "torque25", 7), ("r4b", "torque15", 3)]
    ok_all = True
    for cond, tq, rep in cases:
        ci, ti = conds.index(cond), TORQUES.index(tq)
        start = ((ci * 8 + ti) * 10 + rep) * 33
        got = X[start:start + 33]
        exp, onset, nraw = rebuild(cond, tq, rep)
        n = min(len(got), len(exp))
        d = np.abs(got[:n].astype(np.float64) - exp[:n].astype(np.float64))
        rel = d.max() / (np.abs(exp[:n]).max() + 1e-12)
        ok = rel < 1e-5
        ok_all &= ok
        print(f"  {cond}/{tq}/rep{rep}  idx {start:>5}  n_raw={nraw}  onset={onset}  "
              f"n_win(rekon)={len(exp)}")
        print(f"      selisih maks = {d.max():.3e}   relatif = {rel:.3e}   "
              f"{'COCOK' if ok else '<<< TIDAK COCOK'}")
        print(f"      label npz: y={y[start]} torsi={torque[start]} rep={reps[start]}  "
              f"| harapan: y={LABEL[cond]} torsi={ti} rep={rep}  "
              f"{'OK' if (y[start] == LABEL[cond] and torque[start] == ti and reps[start] == rep) else '<<< SALAH LABEL'}")

    print("\n" + "=" * 88)
    print("C. LAJU SAMPLING (independen, dari rasio vibrasi vs listrik)")
    print("=" * 88)
    with h5py.File(RAW / MAT["rs"], "r") as f:
        g = f["rs"]["torque05"]
        nv = np.array(f[g["Vib_acpe"][0, 0]]).size
        ne = np.array(f[g["Ia"][0, 0]]).size
    print(f"  vibrasi  = {nv} sampel | listrik = {ne} sampel | rasio = {ne/nv:.4f}")
    for fe in (50000, 25000, 20000):
        print(f"    jika listrik {fe} Hz -> vibrasi = {fe*nv/ne:8.1f} Hz  "
              f"(durasi {ne/fe:.2f} s)")
    print(f"  FS yang dipakai kode = {FS} Hz -> durasi vibrasi {nv/FS:.2f} s")

    print("\n" + "=" * 88)
    print("D. SKALA ANTAR-KANAL (isu standardisasi by_var, Sec. IV-C makalah CENIM 2026)")
    print("=" * 88)
    names = VIB + ["env_hilbert"]
    sub = X[::37]
    print(f"  {'kanal':14} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'rasio std':>10}")
    stds = [float(sub[:, c].std()) for c in range(X.shape[1])]
    for c, nm in enumerate(names):
        v = sub[:, c]
        print(f"  {nm:14} {v.mean():>10.4f} {v.std():>10.4f} {v.min():>10.3f} "
              f"{v.max():>10.3f} {stds[c]/max(stds):>10.3f}")
    print(f"  -> std terbesar/terkecil = {max(stds)/min(stds):.1f}x")

    print("\n" + "=" * 88)
    print("E. KEBOCORAN: window duplikat lintas-rekaman?")
    print("=" * 88)
    h = np.array([hash(X[i, 0, ::64].tobytes()) for i in range(0, len(X), 1)])
    u, c = np.unique(h, return_counts=True)
    dup = int((c > 1).sum())
    print(f"  window diperiksa = {len(h)}   hash unik = {len(u)}   duplikat = {dup}")
    print(f"  {'TIDAK ADA DUPLIKAT' if dup == 0 else '<<< ADA DUPLIKAT — periksa!'}")

    print("\n" + "=" * 88)
    print(f"VONIS REKONSTRUKSI: {'SEMUA COCOK' if ok_all else 'ADA YANG TIDAK COCOK'}")
    print("=" * 88)


if __name__ == "__main__":
    main()
