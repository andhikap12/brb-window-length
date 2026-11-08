"""audit_fs_ratio.py — uji ulang dugaan galat laju sampling (Sec. III-A makalah CENIM 2026)."""

import sys
from pathlib import Path

import h5py
import numpy as np

RAW = Path(__file__).resolve().parents[2] / "DatasetRaw" / "banco de dados experimental"
FS_ELEC = 50_000.0  # laju kanal listrik yang diasumsikan pada penurunan semula
FS_KODE = 7_600.0  # nilai yang dipakai pipeline
FS_DUGAAN = 7_667.5  # nilai yang diusulkan penurunan semula

VIB_HINT = ("vib",)
ELEC_HINT = ("current", "corrente", "ia", "va", "voltage", "tensao")


def scan(filepath):
    """Kembalikan (kondisi, torsi, repetisi, n_getaran, n_listrik, nama_vib, nama_elec)."""
    rows = []
    with h5py.File(filepath, "r") as f:
        for cond in (k for k in f.keys() if not k.startswith("#")):
            grup_kondisi = f[cond]
            if not isinstance(grup_kondisi, h5py.Group):
                continue
            for torsi in sorted(grup_kondisi.keys()):
                grup_torsi = grup_kondisi[torsi]
                if not isinstance(grup_torsi, h5py.Group):
                    continue
                kanal = list(grup_torsi.keys())
                vib = [c for c in kanal if any(h in c.lower() for h in VIB_HINT)]
                elec = [c for c in kanal if any(h in c.lower() for h in ELEC_HINT)]
                if not vib:
                    continue
                for rep in range(grup_torsi[vib[0]].shape[0]):
                    n_vib = int(np.prod(f[grup_torsi[vib[0]][rep, 0]].shape))
                    n_elec = (
                        int(np.prod(f[grup_torsi[elec[0]][rep, 0]].shape)) if elec else None
                    )
                    rows.append(
                        (cond, torsi, rep, n_vib, n_elec, vib[0], elec[0] if elec else None)
                    )
    return rows


def main():
    baris = []
    for filepath in sorted(RAW.glob("*.mat")):
        print(f"memindai {filepath.name} ...", flush=True)
        baris += scan(filepath)

    if not baris:
        sys.exit("tidak ada rekaman terbaca")

    n_vib = np.array([r[3] for r in baris], float)
    dengan_listrik = [r for r in baris if r[4]]

    print("\n" + "=" * 78)
    print(
        f"n rekaman = {len(baris)}   "
        f"(kanal getaran = {baris[0][5]}, kanal listrik = {baris[0][6]})"
    )
    print(
        f"panjang GETARAN : min={n_vib.min():.0f}  median={np.median(n_vib):.0f}  "
        f"maks={n_vib.max():.0f}  sebaran={(n_vib.max() - n_vib.min()) / n_vib.min() * 100:.2f}%"
    )

    if not dengan_listrik:
        sys.exit("kanal listrik tidak ditemukan — rasio tak dapat dihitung")

    n_elec = np.array([r[4] for r in dengan_listrik], float)
    n_vib_pasangan = np.array([r[3] for r in dengan_listrik], float)
    print(
        f"panjang LISTRIK : min={n_elec.min():.0f}  median={np.median(n_elec):.0f}  "
        f"maks={n_elec.max():.0f}  sebaran={(n_elec.max() - n_elec.min()) / n_elec.min() * 100:.2f}%"
    )

    fs = FS_ELEC * n_vib_pasangan / n_elec
    median = float(np.median(fs))
    print(f"\n--- fs getaran TERSIRAT bila listrik = {FS_ELEC:.0f} Hz dan rentang waktu sama ---")
    print(f"  min={fs.min():8.1f}   median={median:8.1f}   maks={fs.max():8.1f}   sd={fs.std():6.1f} Hz")
    print(
        f"  rentang = {fs.max() - fs.min():.1f} Hz "
        f"({(fs.max() - fs.min()) / median * 100:.2f}% dari median)"
    )
    print(
        f"  median vs {FS_KODE:.0f} Hz : {median - FS_KODE:+.1f} Hz "
        f"({(median - FS_KODE) / FS_KODE * 100:+.2f}%)"
    )
    lebih_dekat = int((np.abs(fs - FS_KODE) < np.abs(fs - FS_DUGAAN)).sum())
    print(f"  rekaman lebih dekat ke {FS_KODE:.0f} daripada {FS_DUGAAN:.1f} : {lebih_dekat}/{len(fs)}")

    print(
        f"\n  durasi pada rekaman median : getaran {np.median(n_vib_pasangan) / FS_KODE:.3f} s "
        f"vs listrik {np.median(n_elec) / FS_ELEC:.3f} s"
    )
    print(
        "\nVONIS: rasio panjang bervariasi antar rekaman, sehingga tidak dapat menetapkan fs. "
        f"Median mereproduksi {FS_KODE:.0f} Hz."
    )


if __name__ == "__main__":
    main()
