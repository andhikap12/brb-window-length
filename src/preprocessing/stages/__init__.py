"""Tahapan preprocessing sinyal getaran.

s1_load_raw → s2_steady_state → s3_bandpass → s4_envelope → s5_windowing.
CATATAN: tahap 6 (Normalisasi Per-Jendela) TIDAK di sini — ia dijalankan
ONLINE saat training via TSStandardize(by_sample=True). Lihat src/training/.
"""
