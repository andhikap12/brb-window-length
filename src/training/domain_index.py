"""domain_index.py — UJI A: target skalar indeks domain untuk diskriminator regresi.

Rujukan: lengan LODO "BT + SupCon regresi-indeks" (Sec. VI makalah CENIM 2026).

MENGAPA SKALAR
Kepala domain kategoris memberi backbone 6! = 720 permutasi label sebagai jalan kabur
yang MEMPERTAHANKAN informasi: GRL bisa memenuhi "maksimalkan loss diskriminator"
dengan memetakan ulang fitur ke wilayah berlabel salah, bukan menghapus. Itulah M3,
dan buktinya domain_loss melampaui plafon ln(6) pada 32/32 kombinasi. Target skalar
tidak punya label untuk dipermutasi; ruang kaburnya menyusut jadi pembalikan monoton,
yang langsung terbaca sebagai korelasi(D(z), u) < 0.

CATATAN PENTING TENTANG PILIHAN VARIABEL (diukur 2026-08-02, bukan diasumsikan)
Spesifikasi menawarkan tiga nilai: `torque`, `slip`, `f2s`. Setelah normalisasi
ke [0,1], DUA DI ANTARANYA IDENTIK PERSIS:

    2sf = 2 * slip * f_line = 120 * slip   -> penskalaan murni
    (120s - 120s_min) / (120s_maks - 120s_min) == (s - s_min) / (s_maks - s_min)

Selisih terukur u_f2s vs u_slip = 1,1e-16 (presisi mesin). Jadi `f2s` dan `slip`
BUKAN dua ablasi — keduanya satu target yang sama. Keduanya tetap disediakan agar
penamaan cocok dengan dokumen protokol, tetapi menjalankan keduanya sebagai
eksperimen terpisah adalah pemborosan GPU dan TIDAK boleh dilaporkan sebagai ablasi.

`torque` benar-benar berbeda, tetapi tipis: maks |u_slip - u_torque| = 0,058 pada
enam domain latih (r = 0,9978). Slip terukur sedikit cembung terhadap torsi karena
motor punya slip dasar tanpa beban — itu mencerminkan pengukuran, bukan asumsi.

SLIP TERUKUR (scripts/audit/audit_slip.py)
Frekuensi poros dari puncak spektral Vib_base MENTAH pada segmen steady-state ~20 s
(resolusi ~0,05 Hz), prominensi 618-4.467x median latar.
"""

from __future__ import annotations

import numpy as np

F_LINE = 60.0  # Hz jala-jala; 2sf = 2 * slip * F_LINE

# indeks level torsi (0..7, sesuai split_info["torque"]) -> (Nm, slip terukur)
TORQUE_NM: dict[int, float] = {
    0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0, 6: 3.5, 7: 4.0,
}
SLIP_TERUKUR: dict[int, float] = {
    0: 0.0148, 1: 0.0185, 2: 0.0222, 3: 0.0277,
    4: 0.0324, 5: 0.0369, 6: 0.0433, 7: 0.0488,
}

MODE_SAH = ("none", "torque", "slip", "f2s")


def _nilai_mentah(mode: str, level: int) -> float:
    if mode == "torque":
        return TORQUE_NM[level]
    if mode == "slip":
        return SLIP_TERUKUR[level]
    if mode == "f2s":
        return 2.0 * SLIP_TERUKUR[level] * F_LINE
    raise ValueError(f"mode indeks domain tidak dikenal: {mode!r} (sah: {MODE_SAH})")


def build_domain_u(mode: str, train_torque_to_idx: dict[int, int]) -> np.ndarray:
    """Tabel u[indeks_domain] -> skalar di [0,1], dinormalisasi dari domain LATIH saja.

    `train_torque_to_idx` datang dari build_fold_domain_targets: level torsi -> indeks
    domain kontigu. Torsi uji tidak ada di dalamnya (dimasker -1), sehingga min/maks
    di sini otomatis hanya menyentuh statistik latih — syarat protokol.
    """
    if mode == "none":
        raise ValueError("build_domain_u dipanggil dengan mode 'none'.")

    n = len(train_torque_to_idx)
    mentah = np.zeros(n, dtype=np.float64)
    for level, idx in train_torque_to_idx.items():
        mentah[idx] = _nilai_mentah(mode, int(level))

    lo, hi = float(mentah.min()), float(mentah.max())
    if hi - lo <= 0:
        raise ValueError(f"indeks domain '{mode}' konstan pada domain latih: {mentah}")
    return ((mentah - lo) / (hi - lo)).astype(np.float32)


def ringkas_u(mode: str, u: np.ndarray, train_torque_to_idx: dict[int, int]) -> str:
    """Satu baris untuk log — agar tabel u ikut terekam di log pelatihan."""
    pasangan = sorted(train_torque_to_idx.items(), key=lambda kv: kv[1])
    isi = ", ".join(f"lv{lv}({TORQUE_NM[lv]:.1f}Nm)->{u[i]:.4f}" for lv, i in pasangan)
    catatan = " [identik dengan 'slip' setelah normalisasi]" if mode == "f2s" else ""
    return f"u[{mode}]{catatan}: {isi}"
