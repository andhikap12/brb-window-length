"""domain_index.py — UJI A: target skalar indeks domain untuk diskriminator regresi."""

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
    """Tabel u[indeks_domain] -> skalar di [0,1], dinormalisasi dari domain LATIH saja."""
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
