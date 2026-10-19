"""Preprocessing sinyal getaran.

Struktur:
  stages/    — fungsi murni per-tahap (s1..s5)
  pipelines/ — orkestrasi preprocessing (motor)
  io         — persistensi & ringkasan dataset

Tahap 6 (Normalisasi Per-Jendela) TIDAK offline — dijalankan ONLINE saat
training via TSStandardize(by_sample=True). Lihat src/training/.
"""

from src.preprocessing.io import load_dataset, save_dataset  # noqa: F401
