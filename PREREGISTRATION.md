# Pre-registered criteria and runner headers

Each experiment's pass/fail criteria were written into the header of the runner or analysis
script that applies them, before any number was read. To keep the code readable, those headers
and the longer comment blocks were later moved out of the code and into this file. The text below
is copied verbatim, in the original Indonesian and English, with nothing added or reworded.

Each block names the file it came from and its line range in that file before the move. Criterion
IDs (L-P1, R2-P3, A2-P0 and so on) are explained under "Internal labels" in the README.

## Contents

- [`scripts/_antre_lr3e4_tsseq.sh`](#scripts_antre_lr3e4_tsseqsh)
- [`scripts/_run_a3b_byvar_n38.sh`](#scripts_run_a3b_byvar_n38sh)
- [`scripts/_run_bt_dann.sh`](#scripts_run_bt_dannsh)
- [`scripts/_run_byvar_seeds.sh`](#scripts_run_byvar_seedssh)
- [`scripts/_run_cdan_ext.sh`](#scripts_run_cdan_extsh)
- [`scripts/_run_domain_index.sh`](#scripts_run_domain_indexsh)
- [`scripts/_run_hp_faktorial.sh`](#scripts_run_hp_faktorialsh)
- [`scripts/_run_inlp_nl.sh`](#scripts_run_inlp_nlsh)
- [`scripts/_run_lodo.sh`](#scripts_run_lodosh)
- [`scripts/_run_lodo_metode.sh`](#scripts_run_lodo_metodesh)
- [`scripts/_run_lr_faktorial.sh`](#scripts_run_lr_faktorialsh)
- [`scripts/_run_tsseq_faktorial.sh`](#scripts_run_tsseq_faktorialsh)
- [`scripts/_run_w16384_method.sh`](#scripts_run_w16384_methodsh)
- [`scripts/_run_w16384_ovl.sh`](#scripts_run_w16384_ovlsh)
- [`scripts/_run_w4096_sub.sh`](#scripts_run_w4096_subsh)
- [`scripts/_run_window16384.sh`](#scripts_run_window16384sh)
- [`scripts/audit/aggregate_seeds.py`](#scriptsauditaggregate_seedspy)
- [`scripts/audit/audit_a6_resolvabilitas.py`](#scriptsauditaudit_a6_resolvabilitaspy)
- [`scripts/audit/audit_ckpt_ext.py`](#scriptsauditaudit_ckpt_extpy)
- [`scripts/audit/audit_dataset.py`](#scriptsauditaudit_datasetpy)
- [`scripts/audit/audit_fs_ratio.py`](#scriptsauditaudit_fs_ratiopy)
- [`scripts/audit/audit_grad_traj.py`](#scriptsauditaudit_grad_trajpy)
- [`scripts/audit/audit_hp_grid.py`](#scriptsauditaudit_hp_gridpy)
- [`scripts/audit/audit_hull_window.py`](#scriptsauditaudit_hull_windowpy)
- [`scripts/audit/audit_kanal_arus.py`](#scriptsauditaudit_kanal_aruspy)
- [`scripts/audit/audit_lodo.py`](#scriptsauditaudit_lodopy)
- [`scripts/audit/audit_lodo_metode.py`](#scriptsauditaudit_lodo_metodepy)
- [`scripts/audit/audit_normalisasi.py`](#scriptsauditaudit_normalisasipy)
- [`scripts/audit/audit_r2_per_torsi.py`](#scriptsauditaudit_r2_per_torsipy)
- [`scripts/audit/audit_r3_faktorial.py`](#scriptsauditaudit_r3_faktorialpy)
- [`scripts/audit/audit_resolusi_frekuensi.py`](#scriptsauditaudit_resolusi_frekuensipy)
- [`scripts/audit/audit_slip.py`](#scriptsauditaudit_slippy)
- [`scripts/audit/audit_stats2.py`](#scriptsauditaudit_stats2py)
- [`scripts/audit/audit_steady_state.py`](#scriptsauditaudit_steady_statepy)
- [`scripts/audit/npdann_inlp.py`](#scriptsauditnpdann_inlppy)
- [`scripts/audit/sanity_bt_dann.py`](#scriptsauditsanity_bt_dannpy)
- [`scripts/audit/sanity_domain_index.py`](#scriptsauditsanity_domain_indexpy)
- [`scripts/make_subsample_w4096.py`](#scriptsmake_subsample_w4096py)
- [`scripts/run_preprocessing.py`](#scriptsrun_preprocessingpy)
- [`scripts/train_motor_dann.py`](#scriptstrain_motor_dannpy)
- [`src/models/backbones/tsitplus.py`](#srcmodelsbackbonestsitpluspy)
- [`src/preprocessing/stages/s2_steady_state.py`](#srcpreprocessingstagess2_steady_statepy)
- [`src/training/callbacks.py`](#srctrainingcallbackspy)

## scripts/_antre_lr3e4_tsseq.sh

Lines 2-27:

~~~~text
# _antre_lr3e4_tsseq.sh — tunggu kampanye lr TSiTPlus selesai, lalu kerjakan sisanya.
#
# Urutan, semuanya berurutan karena semuanya memakai GPU yang sama:
#   1. tunggu 4 penanda selesai milik _run_lr_faktorial.sh
#   2. audit 4 lengan TSiTPlus @ 3e-4          -> audit_lr3e4_*_s8.npz
#   3. kampanye TSSequencerPlus @ 3e-4          (~1 jam, terukur 2x lebih cepat)
#   4. audit 4 lengan TSSequencerPlus @ 3e-4   -> audit_lr3e4tsseq_*_s8.npz
#   5. cetak kisi backbone x lr yang sudah lengkap
#
# MENGAPA MENUNGGU, BUKAN JALAN BARENG
# ====================================
# Kartunya 6 GB. Dua kampanye sekaligus membuat keduanya OOM, dan yang lebih buruk,
# OOM di jam kedua meninggalkan kampanye separuh jadi yang penanda selesainya tidak
# ditulis sehingga tidak jelas mana yang sahih.
#
# MENUNGGU PENANDA, BUKAN PROSES
# ==============================
# Penanda hanya ditulis kalau lengannya BERHASIL (lihat fungsi stage di skrip kampanye).
# Jadi kalau kampanye lr gagal di tengah, jumlah penanda tidak pernah sampai 4 dan skrip
# ini berhenti dengan pesan alih-alih menumpuk kampanye baru di atas kegagalan. Itu
# perilaku yang diinginkan: jangan antre di atas hasil yang belum tentu sahih.
#
# --methods WAJIB DILEWATKAN. Tanpa itu audit_ckpt_ext.py mengulang SELURUH isi METHODS
# terhadap --npz yang diberikan, sehingga metode lain dievaluasi memakai dataset yang salah
# dan keluarannya berkunci banyak. audit_hp_grid.py menuntut satu kunci per berkas dan akan
# menolak berkas berkunci banyak; itu gerbangnya, bukan kebetulan.
~~~~

## scripts/_run_a3b_byvar_n38.sh

Lines 2-28:

~~~~text
# _run_a3b_byvar_n38.sh — A3b: by_var=True pada n=38. TAHAN-MATI.
#
# Kriteria A3b-P1..P3 TERKUNCI di komentar ini, ditulis sebelum satu seed baru dijalankan.
# Hasilnya dilaporkan di Sec. IV-C makalah CENIM 2026 (normalisasi per-kanal, 38 seed).
#
# =====================================================================================
# MENGAPA UJI INI ADA
# =====================================================================================
# A3-P1 gugur 5 dari 8 seed dengan Delta torsi-0 = +3,80 dan varians membaik 0,76x.
# Ia gugur karena KURANG BERDAYA, bukan karena tidak bekerja: analisis daya menuntut
# n ~ 38. Ini pengulangan hipotesis yang SAMA pada n yang dituntut analisis itu.
#
# Masalah yang hendak diperbaiki kini terukur langsung, pada 13.200 window:
#   Vib_base 26,8%  Vib_carc 26,5%  Vib_acpe 23,2%  Vib_acpi 20,2%
#   env_hilbert 2,0%   Vib_axial 1,3%      <- rasio terkuat/terlemah 21x
# Dan A3-P3 menemukan Vib_axial justru 6,44x lebih milik KELAS daripada torsi.
#
# =====================================================================================
# APA YANG DIJALANKAN
# =====================================================================================
# Seed lama dipakai ulang apa adanya. Yang dijalankan hanya kekurangannya:
#   lengan B (by_var)  sudah 8 seed  -> butuh 30 lagi
#   lengan A (default) sudah 20 seed -> butuh 18 lagi
# Seed 42 dan 123 sudah hadir di kedua lengan sebagai kontrol reproduksibilitas.
#
# CARA MELANJUTKAN SETELAH LAPTOP DIMATIKAN:
#   bash scripts/_run_a3b_byvar_n38.sh      # tahap yang selesai dilewati otomatis
~~~~

## scripts/_run_bt_dann.sh

Lines 2-27:

~~~~text
# _run_bt_dann.sh — BT-DANN (normalisasi gradien; Sec. IV-A dan IV-D makalah CENIM 2026).
# TAHAN-MATI: tahap yang sudah selesai dilewati. Pola sama dengan _run_byvar_seeds.sh.
#
# APA YANG DIUJI
#   lambda tidak lagi dijadwalkan menurut waktu, melainkan dihitung tiap batch agar
#   ||g_dom|| / ||g_fault|| tetap di rasio_target. Tanpa normalisasi rasio meledak
#   0,48 -> 5,06 dan 45/50 epoch dikuasai objektif adversarial (Sec. IV-D makalah).
#
# PREDIKSI TERUKUR (semuanya bisa gugur)
#   1. rasio efektif datar di 0,30 — terbaca di baris [BT-DANN-BAL ep N] tiap epoch
#   2. varians antar-seed TURUN (sd DANN+SupCon 5,80 tertinggi di seluruh tabel)
#   3. seed kolaps hilang
#   4. rerata naik ke SEKITAR ERM — bukan melampauinya (argumen langit-langit)
#
# PEMBANDING (n=20, aggregate_seeds.py 2026-08-02)
#   ERM 87,64 +/- 4,36 | DANN 87,01 +/- 4,17 | SupCon 89,85 +/- 4,62
#
# TAHAP 1-2 memberi n=8 lebih dulu (~50 menit) sebagai penapis; tahap 3-4 melengkapi
# ke n=20 memakai 12 seed yang sama dengan run perluasan.
#
# MELANJUTKAN SETELAH LAPTOP DIMATIKAN:
#   cd <repository root>
#   bash scripts/_run_bt_dann.sh
#
#   Ulang satu tahap : rm logs/.done_bt_dann/<nama_tahap>
#   Ulang semuanya   : rm -rf logs/.done_bt_dann
~~~~

## scripts/_run_byvar_seeds.sh

Lines 2-18:

~~~~text
# _run_byvar_seeds.sh — TAHAN-MATI. Aman dijalankan ulang: tahap yang sudah selesai dilewati.
#
# TAHAP 1 — uji by_var=True (prasyarat; ~1 jam)
#   Standardisasi per-sampel DAN per-kanal. Default berbagi satu mean/std lintas keenam
#   kanal, menekan Vib_axial ke 3.9% dan envelope ke 5.7% (Sec. IV-C makalah CENIM 2026).
#   Pembanding: ERM 89.40 +/- 4.18, DANN lam=1.0 86.62 +/- 3.13.
#
# TAHAP 2 — perluasan seed 8 -> 20 (~3.3 jam)
#   12 seed BARU + seed 42 & 123 sebagai kontrol reproduksibilitas. Bila ERM seed-42
#   mereproduksi 97.73, penggabungan dengan 8 seed lama sah. Daya uji 0.40 -> ~0.84.
#
# CARA MELANJUTKAN SETELAH LAPTOP DIMATIKAN:
#   cd <repository root>
#   bash scripts/_run_byvar_seeds.sh          # tahap selesai dilewati otomatis
#
#   Ulang satu tahap saja  : rm logs/.done_byvar_seeds/<nama_tahap>
#   Ulang semuanya         : rm -rf logs/.done_byvar_seeds
~~~~

## scripts/_run_cdan_ext.sh

Lines 2-3:

~~~~text
# CDAN (conditional adversarial) vs CDAN+SupCon — protokol IDENTIK dengan
# 8-seed EXT lain (bandingkan langsung ke: ERM 89.40, DANN 86.62, DANNSupCon 88.79).
~~~~

## scripts/_run_domain_index.sh

Lines 2-72:

~~~~text
# _run_domain_index.sh — UJI A: regresi indeks domain di atas BT-DANN-SupCon.
# TAHAN-MATI: tahap yang sudah selesai dilewati. Pola sama dengan _run_bt_dann.sh.
#
# Rujukan: lengan LODO "BT + SupCon regresi-indeks" (Sec. VI makalah CENIM 2026).
# PRASYARAT: scripts/audit/sanity_domain_index.py lolos 32/32 (sudah, 2026-08-02).
#
# =====================================================================================
# HIPOTESIS
# =====================================================================================
# Kepala domain KATEGORIS memberi backbone 6! = 720 permutasi label sebagai jalan kabur
# yang MEMPERTAHANKAN informasi (M3): GRL memenuhi "maksimalkan loss diskriminator"
# dengan memetakan ulang fitur ke wilayah berlabel salah, bukan menghapus. Bukti M3:
# domain_loss melampaui plafon ln(6) = 1,7918 pada 32/32 kombinasi varian x seed
# (Wilcoxon p = 0,0078) — "salah dengan percaya diri" mustahil tanpa informasi.
#
# Target SKALAR tidak punya label untuk dipermutasi. Ruang kaburnya menyusut jadi pada
# dasarnya satu: pembalikan monoton (u -> 1-u), yang terdeteksi langsung sebagai
# korelasi(D(z), u) negatif.
#
# =====================================================================================
# TEMUAN YANG MENGUBAH RANCANGAN ABLASI (diukur 2026-08-02, sebelum run)
# =====================================================================================
# Ada tiga nilai indeks yang mungkin: torque, slip, f2s. Setelah normalisasi [0,1]:
#
#     u[f2s] == u[slip]  PERSIS   (2sf = 120*slip -> penskalaan murni; beda 1,1e-16)
#     u[torque] beda maks 0,0581 dari u[slip]  (r = 0,9978)
#
# Jadi "tiga varian indeks" sesungguhnya SATU target, plus satu varian yang beda tipis.
# Menjalankan f2s dan slip sebagai dua ablasi terpisah adalah pemborosan GPU dan TIDAK
# boleh dilaporkan sebagai ablasi — hasilnya identik secara aritmetika, bukan empiris.
# Karena itu hanya `f2s` yang dijalankan (variabel kausal fisiknya, default).
# Ablasi `torque` disiapkan tetapi TIDAK otomatis — lihat TAHAP 2.
#
# =====================================================================================
# KRITERIA GUGUR — DIDAFTARKAN DI MUKA, SEBELUM ANGKA TERLIHAT
# (aturan pelaporan no.5: tidak boleh diubah)
# =====================================================================================
# Pembanding probe domain (beku, 6-arah, GroupKFold, audit_ckpt_ext.py):
#     ERM 96,16  |  DANN lambda=1,0 93,57  |  BT-DANN-SupCon 93,81
# Pembanding akurasi n=8: BT-DANN-SupCon 89,81 +/- 1,71
#
#   A-P1 (utama)     Probe turun DI BAWAH 93,57% — lebih baik daripada varian GRL mana
#                    pun sejauh ini.
#                    GUGUR bila probe >= 95%, yaitu praktis tidak ada penghapusan.
#
#   A-P2 (mekanisme) korelasi(D(z), u) tetap POSITIF sampai akhir pelatihan.
#                    GUGUR bila korelasi < -0,5 -> jalan kabur pembalikan monoton
#                    terjadi; analisis permutasi tetap berlaku tetapi perbaikan ini
#                    tidak bekerja.
#
#   A-P3 (keamanan)  Win-F1 dan F1 per-kelas TIDAK turun; kelas r1b dipantau khusus
#                    (sideband terlemah, kelas yang paling rentan bila GRL menggosok
#                    residu kerusakan).
#                    GUGUR bila r1b turun > 5 poin sementara kelas lain tetap.
#
#   A-P4 (stabilitas) MSE domain tidak divergen sepanjang 50 epoch.
#                    GUGUR bila tumbuh tak terbatas -> butuh pembatasan (mis. Huber).
#
# BILA A-P1 GUGUR SEMENTARA A-P2 LOLOS: target skalar tidak memberi diskriminator daya
# cukup untuk menekan backbone. Seluruh analisis permutasi (M3) — salah satu pilar
# diagnosis Sec. IV-D makalah — perlu ditinjau ulang, dan itu memengaruhi cara makalah ditulis.
# Itu sebabnya Uji A didahulukan meski kodenya paling kecil.
#
# ATURAN DAYA: n=8 PENAPIS, n=20 KLAIM (aturan pelaporan no.2).
#
# MELANJUTKAN SETELAH LAPTOP DIMATIKAN:
#   cd <repository root>
#   bash scripts/_run_domain_index.sh
#
#   Ulang satu tahap : rm logs/.done_domain_index/<nama_tahap>
#   Ulang semuanya   : rm -rf logs/.done_domain_index
~~~~

Lines 113-118:

~~~~text
# ---- TAHAP 2: ablasi `torque` — TIDAK otomatis ----
# u[torque] hanya beda maks 0,058 dari u[f2s] (r = 0,998), jadi ablasi ini menguji
# perbedaan yang sangat tipis. Jalankan HANYA bila TAHAP 1 memberi efek terukur pada
# A-P1; kalau f2s saja tidak menggerakkan probe, ablasi variabelnya tidak informatif.
# stage didx_torque_bt_supcon_s8 $PY scripts/train_motor_dann.py $BASE --seeds $S8 \
#     --method dannsupcon $BT $SC --domain_index torque --name DIDX_torque_BT_dannsupcon
~~~~

## scripts/_run_hp_faktorial.sh

Lines 2-40:

~~~~text
# _run_hp_faktorial.sh — faktorial 2x2 R3 pada SATU sel (backbone, lr) mana pun.
#
# Pemakaian:
#     bash scripts/_run_hp_faktorial.sh <backbone> <lr> [awalan]
# Contoh:
#     bash scripts/_run_hp_faktorial.sh tssequencerplus 3e-4
#     bash scripts/_run_hp_faktorial.sh tsitplus        3e-4 LR3E4
#
# MENGAPA TERPARAMETER
# ====================
# Rancangan hiperparameter x backbone butuh satu kampanye per sel. Menyalin
# _run_tsseq_faktorial.sh sekali per sel akan melahirkan tujuh berkas nyaris identik yang
# pasti saling menyimpang begitu satu di antaranya diperbaiki. Skrip ini menggantikan
# keluarga itu; _run_lr_faktorial.sh (tsitplus @ 3e-4, dilepas 8 Agu 00:29) adalah kasus
# khususnya yang sengaja DIBIARKAN UTUH karena ia sedang berjalan saat berkas ini ditulis.
#
# RANCANGAN YANG SEDANG DILENGKAPI
# ================================
# Sel = (backbone, lr). Isi tiap sel = faktorial 2x2 R3 (panjang jendela x jumlah window),
# 4 lengan x 8 seed. Yang sudah ada dan yang belum, per 8 Agu 2026:
#
#                        lr 1e-3                    lr 3e-4
#   TSiTPlus             ADA (r3_*.npz)             BERJALAN (_run_lr_faktorial.sh)
#   TSSequencerPlus      ADA (audit_tsseq_*.npz)    BELUM  <- sel yang melengkapi 2x2
#
# Melengkapi sel keempat membuat backbone x lr jadi faktorial PENUH atas dua backbone yang
# berbagi tokenisasi, LayerNorm, positional encoding, dan CLS-token. Itu satu-satunya
# pasangan di repo ini yang isolasi bias induktifnya bersih.
#
# MENGAPA KELUARGA CNN TIDAK ADA DI TABEL ITU
# ===========================================
# InceptionTimePlus, ResNetPlus, dan CNN1D mengembalikan BatchNorm + Global Average Pooling
# sekaligus. `src/models/backbones/tsitplus.py` sudah mencatat bahwa BN+GAP membuat baseline
# auto-invarian terhadap beban BY CONSTRUCTION, jadi selnya akan terkonfusi dengan isu
# pencucian amplitudo dan tidak dapat diatribusikan ke lr. Skrip ini TIDAK menolaknya, tetapi
# hasil dari sana menjawab pertanyaan yang BERBEDA: "apakah efeknya muncul di rezim
# normalisasi lain", bukan "apakah efeknya artefak penyetelan". Jangan campur keduanya dalam
# satu kalimat.
#
~~~~

## scripts/_run_inlp_nl.sh

Lines 2-15:

~~~~text
# _run_inlp_nl.sh — probe nonlinear setelah INLP (Sec. IV-D makalah CENIM 2026).
#
# MENGAPA CPU, BUKAN GPU
# ======================
# Ekstraksi fiturnya kecil (inferensi, batch 64), tetapi kartunya 6,4 GB dan sedang
# dipakai kampanye LODO tiga-metode yang berjalan ~7 jam. OOM di sana membuang seluruh
# kampanye itu; menunggu beberapa menit lebih lama di CPU tidak membuang apa pun.
#
# MENGAPA ENAM DARI ENAM BELAS INTI
# =================================
# Uji asap 8 Agu memakai seluruh inti dan lipatan kampanye yang sedang berjalan melar
# dari ~10 menit ke ~19 menit. Dataloader kampanye itu berebut CPU dengan sklearn di
# sini. Enam inti membuat run ini lebih lambat tetapi tidak menyandera pekerjaan yang
# jauh lebih mahal.
~~~~

## scripts/_run_lodo.sh

Lines 2-35:

~~~~text
# _run_lodo.sh — LEAVE-ONE-TORQUE-OUT, 8 lipatan x 5 seed, ERM. TAHAN-MATI.
#
# Kriteria L-P1..L-P3 TERKUNCI di komentar ini, ditulis sebelum satu lipatan dijalankan.
# Hasilnya dilaporkan di Sec. V-A makalah CENIM 2026 (panel a gambar LODO).
#
# =====================================================================================
# MENGAPA UJI INI ADA
# =====================================================================================
# Audit fondasi 3 Agu 21:55 menemukan: 16 dari 16 skrip run memakai --extremes_out.
# Leave-one-torque-out 8 lipatan SUDAH terimplementasi dan merupakan DEFAULT, tetapi
# tidak pernah dipakai sekali pun. Seluruh proyek karena itu berdiri di atas SATU split
# domain, dan tidak ada domain interior yang pernah ditahan.
#
# Akibatnya M13 (target di luar convex hull) dan M14 (sinyal tidak terselesaikan)
# meramalkan urutan defisit yang SAMA dan tidak pernah terpisahkan. Bukti awal sudah
# melawan M13: torsi-7 juga di luar hull (18,2%) tetapi F1-nya 99,04 dan keterbacaan
# sinyalnya 91,0% -- normal.
#
# =====================================================================================
# YANG DIBACA DARI HASILNYA
# =====================================================================================
#   L-P1  Spearman rho antara F1 LODO per torsi dan keterbacaan A6 per torsi (8 titik).
#         LOLOS bila rho >= 0,70 -> keterbacaan sinyal meramalkan kesulitan model,
#         diukur TANPA melatih apa pun. M14 ditegakkan.
#
#   L-P2  M13 DICABUT sebagai penjelasan bila F1(torsi-7) berada DI DALAM rentang
#         interior -- berada di luar hull tidak dengan sendirinya menghukum.
#
#   L-P3  Kurva penuh, deskriptif.
#
# RAMBU: plafon di ujung atas; protokol berbeda dari extremes-out (7 torsi
# latih, bukan 6) sehingga perbandingan lintas-protokol DESKRIPTIF saja; n=5 seed.
#
# =====================================================================================
~~~~

## scripts/_run_lodo_metode.sh

Lines 2-35:

~~~~text
# _run_lodo_metode.sh — LODO 8 lipatan untuk TIGA METODE TERKUAT, mencabut sebagian B7.
#
# MENGAPA
# =======
# Batasan split tunggal (Sec. VI makalah CENIM 2026): split domain tunggal TERCABUT untuk ERM (LODO 8 lipatan,
# 4 Agu) tetapi TETAP BERLAKU untuk keluarga metode, yang seluruhnya masih satu split.
# Artinya separuh NEGATIF kontribusi paper — "tidak ada intervensi sisi-objektif yang
# terpisah dari baseline" — berdiri di atas fondasi yang lebih tipis daripada separuh
# positifnya. Separuh positif kini punya n=20, LODO 8 lipatan, dan 4 sel hiperparameter;
# separuh negatif punya satu split.
#
# MENGAPA HANYA TIGA, BUKAN LIMA BELAS
# ====================================
# LODO memakai 8 lipatan x 5 seed, yaitu ~5x biaya satu lengan extremes-out. Terukur dari
# kampanye 8 Agu: w4096-penuh 8 seed satu split = 31 menit, jadi LODO ~2,5 jam per metode.
# Lima belas metode = ~39 jam dan itu tidak masuk akal.
#
# Yang dipilih adalah tiga metode dengan Rec-Acc torsi-0 TERTINGGI di Tabel I, yaitu titik
# paling rentan bagi klaim negatif: kalau metode yang PALING mungkin mengalahkan ERM tetap
# tidak terpisah di bawah 8 lipatan, klaim negatifnya jauh lebih kuat daripada sebelumnya.
# Menguji metode yang lemah tidak menambah apa pun, karena ia sudah kalah di satu split.
#
#   89,25  BT+SupCon regresi-idx   (dannsupcon + bt_ratio 0.3 + domain_index f2s)
#   87,50  SupCon
#   86,00  DANN+SupCon
#
# INI PENCABUTAN SEBAGIAN, BUKAN PENUH. Sesudah kampanye ini B7 harus berbunyi: tercabut
# untuk ERM dan untuk ketiga lengan terkuat; MASIH BERLAKU untuk dua belas lengan sisanya.
# Jangan menuliskannya sebagai tercabut penuh.
#
# PROTOKOL disalin utuh dari _run_lodo.sh: BASE identik dengan seluruh tabel dengan
# --extremes_out DIBUANG dan --n_folds 8 dipasang, seed S5 SAMA dengan ERM LODO supaya
# perbandingannya berpasangan per seed. Satu lipatan per pemanggilan dengan --fold_idx,
# penanda per lipatan, sehingga mati di tengah tidak membuang lipatan yang sudah selesai.
~~~~

## scripts/_run_lr_faktorial.sh

Lines 2-51:

~~~~text
# _run_lr_faktorial.sh — faktorial 2x2 R3 diulang pada LAJU BELAJAR KEDUA.
#
# MENGAPA
# =======
# Seluruh 15 lengan Tabel I, faktorial R3, dan replikasi TSSequencerPlus berjalan di
# SATU titik hiperparameter: lr 1e-3, batch 64, 50 epoch (src/config.py TRAIN_CONFIG).
# Sapuan atas seluruh skrip run mengonfirmasi --lr tidak pernah dilewatkan sekali pun;
# satu-satunya --batch_size 32 pada riset ini dipakai di lengan eksplorasi w65536, yang
# tidak dilaporkan makalah dan tidak dirilis: jendela 65.536 sampel tidak muat di kartu
# 6,4 GB, jadi itu paksaan memori dan bukan variasi terancang.
#
# Menahan hiperparameter tetap adalah KEKUATAN untuk validitas internal: itulah yang
# membuat Sec. III-G makalah boleh bilang beda antar-baris berasal dari objektif dan bukan
# dari penyetelan. Yang belum diuji adalah validitas eksternalnya. Jendela panjang punya
# urutan 4x lebih panjang dengan anggaran latih yang sama, jadi kalau 1e-3 kebetulan lebih
# cocok untuk urutan panjang, sebagian "efek durasi" sebenarnya efek penyetelan.
#
# Kampanye ini menutup lubang itu dengan bentuk yang SENGAJA dijiplak dari
# _run_tsseq_faktorial.sh: seed sama, flag sama, epoch sama, npz sama, backbone kembali ke
# TSiTPlus. Yang berubah HANYA --lr. Dengan begitu hasilnya sebanding baris-per-baris
# dengan faktorial asli dan dapat dilaporkan dengan pola kalimat yang sama.
#
# YANG DIPERTARUHKAN
# ==================
# Ini satu-satunya eksperimen dalam antrean yang BISA MEMBATALKAN KLAIM UTAMA makalah.
# Pembacaan dikunci di muka, 8 Agu 2026 sebelum satu angka pun ada, atas efek utama
# PANJANG (rata-rata w16384 dikurangi rata-rata w4096 pada kedua tingkat volume):
#
#     >= +10        bertahan; klaim bukan artefak penyetelan
#     +3 sampai +10 arah bertahan, besaran bergantung konfigurasi -> wajib masuk batasan
#     <  +3         klaim panjang TERIKAT pada 1e-3 -> temuan serius, makalah harus berubah
#
# Pembanding pada lr 1e-3: efek utama panjang +15,88 dan volume +7,62 (TSiTPlus),
# +14,75 dan +0,50 (TSSequencerPlus).
#
# CACAT DESAIN YANG DINYATAKAN DI MUKA, BUKAN DISEMBUNYIKAN
# =========================================================
# Menurunkan lr pada anggaran epoch TETAP mencampur "laju" dengan "jumlah latihan": kalau
# efeknya menyusut, lr dan kurang-latih tidak dapat dipisahkan dari kampanye ini sendiri.
# Tidak ada knob tunggal yang bersih pada anggaran epoch tetap, jadi cacat ini melekat.
# Ia harus ikut tertulis ke mana pun angka ini dibawa. Kalau hasilnya jatuh di bawah +3,
# langkah BERIKUTNYA adalah memisahkan keduanya lewat lr 3e-4 dengan epoch dinaikkan,
# bukan langsung menyimpulkan lr yang menyebabkannya.
#
# 3e-4 dipilih TURUN, bukan naik: 3e-3 berisiko divergen dan kegagalan divergen tidak
# informatif untuk pertanyaan ini.
#
# NAMA TIDAK BOLEH BENTROK. Checkpoint disimpan sebagai {name}_{backbone}_seed{s}_fold0.pth
# dan audit menurunkan backbone dari akhiran nama itu. Awalan LR3E4_ menjaga jalannya
# terpisah dari W4096SUB_baseline dkk milik lr 1e-3, yang TIDAK BOLEH tertimpa.
~~~~

## scripts/_run_tsseq_faktorial.sh

Lines 2-27:

~~~~text
# _run_tsseq_faktorial.sh — faktorial 2x2 R3 diulang dengan backbone NON-TRANSFORMER.
#
# MENGAPA
# =======
# "Backbone tunggal" adalah batasan utama paper (Sec. IV-B dan VI makalah CENIM 2026).
# Kampanye ini mengulang faktorial R3 (panjang jendela x jumlah window) dengan
# TSSequencerPlus, yaitu BiLSTM yang menggantikan self-attention, dan mempertahankan
# tokenisasi, LayerNorm, positional encoding, serta CLS-token IDENTIK dengan TSiTPlus.
# Yang diuji: apakah efek panjang jendela selamat di bias induktif lain.
#
# Alasan memilih TSSequencerPlus alih-alih CNN ada di docstring
# src/models/backbones/tssequencerplus.py. Ringkasnya: CNN akan mengembalikan
# BatchNorm + GAP sekaligus, sehingga hasilnya tidak dapat diatribusikan.
#
# BIAYA TERUKUR (RTX 4050 Laptop 6,4 GB, batch 32): TSSequencerPlus 0,014 s/step di
# T=4096 dan 0,031 s/step di T=16384, yaitu dua kali lebih cepat daripada TSiTPlus di
# jendela panjang. Taksiran seluruh kampanye ~1,5 sampai 2 jam.
#
# CATATAN JUJUR: di T=16384 backbone ini punya 3,76M parameter lawan 0,38M milik
# TSiTPlus, jadi perbandingannya TIDAK parameter-matched. Klaim yang diambil karena itu
# hanya kontras DI DALAM backbone (w16384 lawan w4096), bukan level absolut antar
# backbone.
#
# Protokol disalin utuh dari _run_w4096_sub.sh dan _run_w16384_ovl.sh: seed, flag, dan
# epoch sama, hanya --backbone yang berubah. Penanda selesai membuat skrip ini dapat
# diulang tanpa mengerjakan lengan yang sudah beres.
~~~~

## scripts/_run_w16384_method.sh

Lines 2-124:

~~~~text
# _run_w16384_method.sh — UJI F: apakah lever METODE masih menyumbang setelah lever
# DATA dipakai? SupCon dan BT-DANN+SupCon dijalankan ulang di jendela w16384-ovl.
# TAHAN-MATI.
#
# =====================================================================================
# MENGAPA UJI INI ADA
# =====================================================================================
# Uji E (2026-08-02 23:18) membuktikan satu keputusan preprocessing — jendela 16384
# dengan tumpang tindih 75% — memberi +12,51 poin torsi-0 (p = 0,0274; 7/8) dan
# +5,70 poin F1 total (p = 0,0154; 6/8) memakai ERM POLOS, tanpa domain adaptation
# sama sekali. Itu lebih jauh daripada seluruh lever metode di proyek ini, dan
# signifikan — yang tak pernah dicapai lever metode mana pun.
#
# Batasan Uji E no. 2 menyatakan secara eksplisit: "Hanya ERM yang diuji di jendela
# baru. Klaim 'mengalahkan semua metode' BELUM sah sampai metode-metode itu
# dijalankan ulang di w16384-ovl." Uji F menutup lubang itu.
#
# Pertanyaan yang dijawab: apakah keuntungan metode selama ini generalisasi sejati,
# atau tambalan atas keterbatasan jendela yang kini sudah diperbaiki?
#
#   TIDAK ADA CABANG YANG SIA-SIA
#   - Bila metode TIDAK lagi unggul di jendela yang benar: temuan jauh lebih kuat
#     daripada +3,02 yang sekarang — keuntungan metode terbukti kompensasi atas cacat
#     preprocessing, bukan sumbangan generalisasi. Ini menentukan bingkai makalah.
#   - Bila metode MASIH unggul: metode usulan justru menguat, karena berdiri di atas
#     baseline yang jauh lebih tinggi (95,10 vs 89,40 F1).
#
# =====================================================================================
# PEMBANDING — ERM w16384-ovl, 8 seed BERPASANGAN yang SAMA (audit_erm_w16384ovl_s8.npz)
# =====================================================================================
#   torsi-0  91,15 +/- 11,1   |  torsi-7  97,85 +/- 2,9
#   F1 total 95,10 +/-  4,4   |  probe    98,78%   <- TERTINGGI di seluruh proyek
#
# Angka w4096 mana pun DILARANG dipakai sebagai pembanding di sini (batasan Uji E
# no. 5: basis window berbeda, 12.000 vs 13.200, tidak sebanding).
#
# Acuan dari jendela LAMA, dicatat hanya sebagai latar — BUKAN pembanding:
#   SupCon             +2,21 vs ERM w4096 (p = 0,162; tidak signifikan, n=20)
#   BT-DANN+SupCon r0.3 +3,02 vs DANN+SupCon w4096 (p = 0,019, n=8)
#
# =====================================================================================
# KRITERIA — DIDAFTARKAN DI MUKA, SEBELUM ANGKA TERLIHAT
# Ambang tunggal tanpa zona mati (pelajaran A-P1); sumbu varians dilaporkan terpisah
# (pelajaran D-P2); daya dihitung sebelum mengusulkan perluasan (pelajaran Uji D);
# pembanding selalu dalam jendela yang sama pada seed yang sama (batasan Uji E no. 5).
# =====================================================================================
#
#   F-P1 (utama) — dinilai TERPISAH untuk masing-masing metode, satu vonis per metode.
#
#       LOLOS bila  F1 total(metode) > F1 total(ERM w16384-ovl) pada >= 6 dari 8 seed
#                   berpasangan.
#       GUGUR bila  tidak.
#
#       Ambang 6/8 identik dengan E-P1 dan CTRL-P1 — dipakai ulang justru agar tidak
#       bisa disetel setelah melihat hasil. Ambang ini membelah habis ruang hasil:
#       tidak ada keluaran yang jatuh di zona "tak berkesimpulan".
#
#       Metrik utamanya F1 TOTAL, bukan torsi-0. Alasannya: yang ditanya adalah apakah
#       lever metode menyumbang secara umum. torsi-0 adalah sumbu tempat lever DATA
#       bekerja; memakainya sebagai sumbu utama akan menilai metode di lapangan yang
#       sudah dipanen lever lain.
#
#   F-P2 (atribusi) — aturan penafsiran, didaftarkan SEBELUM hasil, tanpa ambang.
#
#       Bila F-P1 GUGUR untuk BT-DANN+SupCon, letak hilangnya keunggulan menentukan
#       penafsiran, dan penafsirannya sudah dikunci di sini:
#         - hilang TERPUSAT di torsi-0  -> keunggulan lama adalah KOMPENSASI atas
#           keterbatasan resolusi; begitu resolusi diperbaiki, tak ada lagi yang
#           ditambal. Ini yang diprediksi.
#         - hilang MERATA di kedua torsi -> efek LANGIT-LANGIT (ERM sudah 95,10;
#           ruang tersisa 4,90 poin). Bukan bukti metode tidak berguna.
#       Yang dilaporkan: Delta per-torsi, bukan vonis lulus/gagal.
#
#       Catatan ruang-kepala yang wajib disebut: pada w4096 ruang tersisa 10,60 poin,
#       di sini 4,90 poin. Delta absolut TIDAK sebanding lintas jendela. Rasio
#       Delta/ruang-tersisa dilaporkan sebagai angka pendamping — dan hanya itu, karena
#       rumusan R pernah cacat sekali (pelajaran C-P1) dan tidak dipakai sebagai ambang.
#
#   F-P3 (mekanisme — ini yang bisa MENCABUT dugaan korelasi terbalik) — probe backbone-beku.
#
#       Uji E menemukan probe naik 96,16% -> 98,78% justru pada konfigurasi yang
#       memperbaiki masalahnya, sehingga muncul dugaan bahwa keterbacaan beban
#       berkorelasi TERBALIK dengan hasil. Dugaan itu berdiri di atas SATU titik data.
#       (Sec. IV-D makalah akhirnya menguji dan menolak mediasinya.)
#       Uji F memberi titik kedua, dan bisa merobohkannya:
#         - probe TURUN dan F1 IKUT TURUN  -> klaim korelasi terbalik dapat dukungan
#           kedua yang independen.
#         - probe TURUN tetapi F1 NAIK     -> klaim korelasi terbalik GUGUR dan WAJIB
#           dicabut, berapa pun hasil F-P1.
#         - probe TIDAK turun              -> metode adversarial terbukti tidak
#           menghapus info domain di jendela ini; catat sebagai kegagalan mekanisme,
#           terpisah dari kegagalan hasil.
#       Probe WAJIB dari audit backbone-beku, tidak pernah domain_acc daring (aturan
#       pelaporan no. 1; metrik daring terbukti meleset 0,79 pada M1).
#
#   F-P4 (varians) — dilaporkan TERLEPAS dari F-P1, tidak boleh dilebur ke dalamnya.
#
#       Klaim utama BT-DANN sejak awal adalah penurunan varians, bukan kenaikan rerata
#       (sd 5,80 -> 1,71 di w4096). sd F1 ERM w16384-ovl = 4,4.
#       LOLOS-varians bila sd(metode) < sd(ERM w16384-ovl) dengan uji-F p < 0,05.
#       Vonis ini berdiri sendiri: metode boleh GUGUR di F-P1 tetapi LOLOS di F-P4,
#       dan itu tetap temuan yang dilaporkan (pelajaran D-P2 — Uji D memperbaiki
#       varians tanpa memperbaiki rerata).
#
#   DAYA — dihitung SEBELUM hasil, agar perluasan ke n=20 tidak diusulkan reaktif.
#       Bila |d| yang terukur >= 0,95, n=11 sudah berdaya 0,80 dan n=8 hampir memadai
#       (seperti Uji E, d = 0,982) -> perluasan S12 dibenarkan.
#       Bila |d| < 0,50, n=20 hanya berdaya < 0,45 -> perluasan DILARANG, seperti
#       Uji D yang sengaja dilewati (d = 0,397; daya n=20 = 0,393).
#
# PRASYARAT
#   Dataset sudah ada (dipakai Uji E):
#     data/processed/motor/motor_multiclass_w16384_s4096_ss_env.npz
#   Sanity BT-DANN sudah lolos 11/11 (2026-08-02) — arsitektur tidak berubah, hanya
#   dataset. Tidak ada implementasi adversarial BARU di sini, jadi aturan pelaporan
#   no. 4 terpenuhi tanpa sanity ulang.
#
# MELANJUTKAN SETELAH LAPTOP DIMATIKAN:
#   cd <repository root>
#   bash scripts/_run_w16384_method.sh
#
#   Ulang satu tahap : rm logs/.done_w16384method/<nama_tahap>
#   Ulang semuanya   : rm -rf logs/.done_w16384method
~~~~

## scripts/_run_w16384_ovl.sh

Lines 2-119:

~~~~text
# _run_w16384_ovl.sh — UJI E: jendela 16384 dengan stride 4096 (tumpang tindih 75%).
# Mengambil resolusi tinggi DAN volume data sekaligus. TAHAN-MATI.
#
# =====================================================================================
# MENGAPA UJI INI ADA — dekomposisi yang mendahuluinya
# =====================================================================================
# Kontrol Uji C (2026-08-02 21:13, ERM, 8 seed berpasangan, audit backbone-beku)
# memisahkan dua efek yang di Uji C saling meniadakan:
#
#   lengan              window   resolusi   torsi-0        torsi-7
#   w4096 penuh         13.200   rendah     78,64 +/- 9,0  99,05 +/- 0,7
#   w4096 subsampel      3.200   rendah     67,59 +/- 7,0  96,23 +/- 4,3
#   w16384               3.200   TINGGI     85,63 +/-14,2  96,81 +/- 4,1
#
#   efek DATA     (13.200 -> 3.200) : -11,05 poin  (p = 0,0207; 6/8)
#   efek RESOLUSI (4096 -> 16384)   : +18,04 poin  (p = 0,0164; Wilcoxon 0,0391; 6/8)
#   neto yang terlihat di Uji C     :  +7,00 poin
#
# Disosiasinya persis prediksi M14: resolusi menolong torsi-0 (+18,04) dan TIDAK
# menolong torsi-7 (+0,58; p = 0,79) yang jendela 4096-nya sudah memuat 3,16 periode.
#
# Uji ini menanyakan: bila kedua efek diambil BERSAMAAN, apakah keduanya aditif?
#
#   w16384 stride 4096 -> ~12.000 window (resolusi TINGGI + volume ~w4096 penuh)
#
# =====================================================================================
# KRITERIA — DIDAFTARKAN DI MUKA, SEBELUM ANGKA TERLIHAT
# Ambang tunggal tanpa zona mati (pelajaran A-P1); sumbu varians disebut terpisah
# (pelajaran D-P2); daya dihitung sebelum mengusulkan perluasan (pelajaran Uji D).
# =====================================================================================
#   E-P1 (utama)  Pembanding yang relevan secara praktis adalah w4096 PENUH — yaitu
#                 konfigurasi yang dipakai SELURUH baris Tabel I makalah (torsi-0 = 78,64).
#
#       LOLOS bila  torsi-0(w16384-ovl) > torsi-0(w4096-penuh) pada >= 6 dari 8 seed.
#       GUGUR bila  tidak.
#
#       Ini pembanding yang lebih keras daripada w4096-subsampel, dan sengaja dipilih:
#       klaim yang berguna adalah "lebih baik daripada yang dipakai
#       sekarang", bukan "lebih baik daripada baseline yang dilemahkan".
#
#   E-P2 (aditivitas)  Prediksi aditif: torsi-0 ~ 67,59 + 18,04 + 11,05 = 96,7.
#       Dicatat sebagai angka acuan, BUKAN ambang lulus. Efeknya diperkirakan di
#       BAWAH prediksi aditif karena tumpang tindih 75% membuat window latih
#       berkorelasi, sehingga ukuran sampel efektif < 12.000. Yang dilaporkan adalah
#       selisih terukur terhadap prediksi, bukan vonis lulus/gagal.
#
#   E-P3 (spesifisitas)  torsi-7 TIDAK boleh naik terukur. M14 memprediksi resolusi
#       tidak menolong domain yang sudah teresolusi; bila torsi-7 ikut naik sebesar
#       torsi-0, maka yang bekerja BUKAN resolusi melainkan sekadar volume/augmentasi
#       dari tumpang tindih — dan seluruh atribusi M14 harus ditinjau ulang.
#
#   E-P4 (varians)  Laporkan sd antar-seed dan uji-F terhadap w4096 penuh TERLEPAS
#       dari E-P1. sd torsi-0 di w16384 non-tumpang-tindih tinggi (14,15); perlu
#       diketahui apakah volume data menormalkannya kembali.
#
# =====================================================================================
# AMBANG E-P1 DI n=20 — DITETAPKAN 3 Agu 2026, 03:47
# Ditetapkan TANPA melihat satu pun nilai hasil S12. Koreksi ketepatan (03:48): saat
# ambang ini ditulis, checkpoint seed-1 ternyata sudah tersimpan (1 dari 12) — jadi
# pernyataan "belum ada tahap yang selesai" akan keliru. Yang benar dan yang menjadi
# dasar keabsahan pra-registrasi ini: TIDAK ADA angka hasil S12 yang pernah dibaca —
# log hanya diperiksa untuk memastikan tahap S8 dilewati dan S12 berjalan. Nilai Win-F1
# atau per-torsi S12 pertama kali akan dibaca pada audit setelah seluruh 12 seed selesai.
# =====================================================================================
# Kriteria di atas ditulis untuk n=8 dan TIDAK mendefinisikan ambangnya di n=20.
# Menetapkannya setelah melihat hasil akan membatalkan pra-registrasi (aturan
# pelaporan no. 5), jadi ditetapkan sekarang:
#
#   E-P1 di n=20:  LOLOS bila torsi-0(w16384-ovl) > torsi-0(w4096-penuh) pada
#                  >= 15 dari 20 seed berpasangan.
#
#   Dasarnya perpanjangan proporsional yang setia: 6/8 = 75% -> 15/20 = 75%.
#   TIDAK ada ambang magnitudo baru yang ditambahkan. E-P1 memang kriteria
#   KONSISTENSI; menambah gerbang besar-efek sekarang adalah persis pelanggaran yang
#   aturan no. 5 larang. Besar efek dilaporkan deskriptif, bukan sebagai vonis.
#
#   E-P2, E-P3, E-P4 berlaku apa adanya, dinilai ulang pada 20 seed gabungan.
#
# PEMBANDING SUDAH ADA DAN SEED-NYA COCOK PERSIS.
#   results/motor/audit_ext_n20.npz -> 'ERM_(baseline)', n=20, seed identik dengan
#   S8 + S12. Terukur: torsi-0 74,18 +/- 9,87 | torsi-7 98,77 | F1 87,64.
#
#   Catat: baseline w4096 TURUN dari 78,64 (n=8) ke 74,18 (n=20). Penyusutan saat
#   diperluas adalah pola nyata di proyek ini (defisit DANN -2,78 -> -0,63). Karena
#   itu selisih di n=20 bisa MELEBAR walau w16384-ovl ikut turun sedikit. Kedua
#   sub-kasus dilaporkan terpisah; tidak ada yang dipilih setelah melihat angka.
#
# YANG DIAWASI LEBIH KETAT DARIPADA E-P1: E-P3.
#   Bila di n=20 torsi-7 IKUT NAIK terukur, mekanismenya bukan resolusi melainkan
#   volume window berkorelasi, dan seluruh atribusi M14 harus ditinjau ulang. Di n=8
#   torsi-7 justru turun (-1,19), jadi arahnya aman sejauh ini. E-P1 gagal hanya
#   melemahkan satu klaim; E-P3 gagal merobohkan penjelasannya.
#
# AUDIT SETELAH TAHAP INI SELESAI (WAJIB, --npz tidak boleh lupa):
#   $PY scripts/audit/audit_ckpt_ext.py --methods 'UJI E: ERM w16384-ovl' \
#       --npz motor_multiclass_w16384_s4096_ss_env.npz \
#       --seeds 42,123,2025,7,17,2024,2026,88,1,3,5,11,13,23,31,57,99,101,202,777 \
#       --out results/motor/audit_erm_w16384ovl_n20.npz
#   Prefix S8 dan S12 keduanya sudah terdaftar di METHODS, jadi satu perintah
#   mengambil kedua tahap sekaligus.
#
# CATATAN INTEGRITAS YANG WAJIB MASUK METODOLOGI
#   1. Tumpang tindih 75% membuat window latih SALING BERKORELASI. GroupKFold
#      per-rekaman tetap mencegah kebocoran ke himpunan uji — torsi uji tidak pernah
#      menyumbang window latih — sehingga angkanya sah. Yang berkurang adalah ukuran
#      sampel EFEKTIF, bukan validitasnya.
#   2. w4096 penuh memakai stride 4096 (0% tumpang tindih). Jadi perbandingan E-P1
#      mengubah resolusi DAN tumpang tindih sekaligus. Itu sebabnya E-P3 ada: ia
#      memisahkan "resolusi" dari "sekadar lebih banyak window berkorelasi".
#
# PRASYARAT:
#   ${PYTHON:-python} -u scripts/run_preprocessing.py --motor --window 16384 --stride 4096
#
# MELANJUTKAN SETELAH LAPTOP DIMATIKAN:
#   cd <repository root>
#   bash scripts/_run_w16384_ovl.sh
#
#   Ulang : rm -rf logs/.done_w16384ovl
~~~~

Lines 163-172:

~~~~text
# ---- TAHAP 2 (U1, ditambahkan 3 Agu 2026 03:40): perluasan ke n=20 ----
# E-P1 lolos di n=8 dengan d = 0,982 -> n=20 berdaya > 0,99. Perluasan DIBENARKAN, berbeda
# dengan Uji D yang sengaja dilewati (d = 0,397; daya n=20 hanya 0,393).
#
# Kriteria TIDAK berubah: E-P1..E-P4 di kepala berkas ini berlaku apa adanya, dinilai ulang
# pada n=20 gabungan. Seed S12 tidak beririsan dengan S8, jadi penggabungannya sah.
#
# Envelope SENGAJA tetap yang lama. Klaim jendela adalah selisih w4096 vs w16384-ovl dengan
# envelope identik di kedua lengan; mengganti envelope di tengah merusak basis E-P1.
# (U3 dibatalkan 3 Agu — premisnya gugur di tingkat window.)
~~~~

## scripts/_run_w4096_sub.sh

Lines 2-67:

~~~~text
# _run_w4096_sub.sh — KONTROL UJI C: w4096 dengan jumlah window disamakan ke w16384.
# TAHAN-MATI: tahap yang sudah selesai dilewati.
#
# =====================================================================================
# MENGAPA KONTROL INI ADA
# =====================================================================================
# Uji C (2026-08-02 20:35) menemukan pola yang SEARAH dengan M14 tetapi tidak dapat
# dipertahankan, karena perlakuannya mengubah dua hal sekaligus:
#
#     (a) resolusi frekuensi   1,855 Hz -> 0,464 Hz     <- yang ingin diuji
#     (b) jumlah window latih  13.200   -> 3.200        <- efek samping
#
# Hasil Uji C (ERM, 8 seed berpasangan, audit backbone-beku):
#     torsi-0   78,64 -> 85,63   (+7,00; p = 0,21; 5/8)
#     torsi-7   99,05 -> 96,81   (-2,24; p = 0,18; 2/8)   <- sd 0,73 -> 4,09, F p = 0,0002
#     C-P1 (R0-R7 > 0 pada >=6/8): 5/8 -> GUGUR
#
# Torsi-7 tidak membutuhkan (a) sama sekali — pada 4,0 Nm jendela 4096 sudah memuat
# 3,16 periode kerusakan — sehingga ia hanya menanggung rugi dari (b). Selama keduanya
# bergerak bersamaan, +7,00 di torsi-0 tidak dapat diatribusikan ke resolusi.
#
# Lengan ini membekukan (b) dan menyisakan (a) sebagai satu-satunya yang berbeda:
#
#     w4096 penuh       13.200 window (33/rekaman)   resolusi RENDAH
#     w4096 subsampel    3.200 window ( 8/rekaman)   resolusi RENDAH   <- lengan ini
#     w16384             3.200 window ( 8/rekaman)   resolusi TINGGI
#
# Window subsampel diambil di indeks j = 0, 4, 8, ... yaitu 4096 sampel PERTAMA dari
# tiap blok 16384, sehingga letak waktunya BERIMPIT dengan window w16384 — bukan sekadar
# tersebar merata (scripts/make_subsample_w4096.py, 4/4 verifikasi struktur cocok).
#
# =====================================================================================
# KRITERIA — DIDAFTARKAN DI MUKA, SEBELUM ANGKA TERLIHAT
# Ambang tunggal tanpa zona mati (pelajaran A-P1) dan menyebut sumbunya (pelajaran D-P2).
# =====================================================================================
#   CTRL-P1 (utama)  Perbandingan yang menentukan: w16384 versus w4096-SUBSAMPEL,
#                    berpasangan pada 8 seed yang sama, pada torsi-0.
#
#       M14 DIDUKUNG bila  torsi-0(w16384) > torsi-0(w4096-sub)  pada >= 6 dari 8 seed.
#       M14 DITOLAK  bila  tidak.
#
#       Ini menghabiskan ruang hasil. Bila didukung, kenaikan torsi-0 berasal dari
#       resolusi dan bukan dari perbedaan data. Bila ditolak, +7,00 pada Uji C adalah
#       artefak jumlah window, dan klaim resolusi-jendela (Sec. IV-B makalah) turun dari "keyakinan sedang"
#       menjadi DITOLAK.
#
#   CTRL-P2 (kontrol kewarasan)  torsi-0(w4096-penuh) > torsi-0(w4096-subsampel).
#       Membuang 4/5 data harus merugikan. Bila TIDAK merugikan sama sekali, maka
#       jumlah window bukan variabel yang berpengaruh di rentang ini, dan seluruh
#       kekhawatiran confound (b) gugur dengan sendirinya — hasil Uji C boleh dibaca
#       apa adanya. Kedua kemungkinan informatif; tidak ada yang membatalkan uji.
#
#   CTRL-P3 (varians)  Laporkan sd antar-seed dan uji-F untuk ketiga lengan, TERLEPAS
#       dari CTRL-P1. Torsi-7 meledak dari 0,73 ke 4,09 di Uji C; perlu diketahui
#       apakah itu efek resolusi atau efek data.
#
# ATURAN DAYA: n=8 PENAPIS. Hitung daya sebelum mengusulkan perluasan — pelajaran Uji D,
# di mana n=20 hanya berdaya 0,393 untuk kontras d = 0,397.
#
# PRASYARAT: ${PYTHON:-python} -u scripts/make_subsample_w4096.py
#
# MELANJUTKAN SETELAH LAPTOP DIMATIKAN:
#   cd <repository root>
#   bash scripts/_run_w4096_sub.sh
#
#   Ulang : rm -rf logs/.done_w4096sub
~~~~

## scripts/_run_window16384.sh

Lines 2-93:

~~~~text
# _run_window16384.sh — UJI C: jendela 16384 menguji batas fisik (M14).
# TAHAN-MATI: tahap yang sudah selesai dilewati. Pola sama dengan _run_bt_dann.sh.
#
# Rujukan: Sec. IV-B dan V-B makalah CENIM 2026.
#
# =====================================================================================
# HIPOTESIS (M14)
# =====================================================================================
# Seluruh kesulitan terkonsentrasi di torsi terendah: F1 78,64% pada 0,5 Nm versus
# 99,04% pada 4,0 Nm. Slip diukur langsung dari kanal getaran mentah untuk kedelapan
# level beban (audit_slip.py). Pada beban terendah:
#
#   jendela  4096 = 0,539 s | resolusi 1,855 Hz | 2sf = 1,78 Hz -> DI BAWAH RESOLUSI
#                                                | jendela memuat 0,96 periode
#   jendela 16384 = 2,156 s | resolusi 0,464 Hz | 2sf = 1,78 Hz -> TERESOLUSI
#                                                | jendela memuat 3,84 periode
#
# Bila terkonfirmasi, ini menjelaskan seluruh temuan dalam satu kalimat: TIDAK ADA
# metode invariansi yang dapat menciptakan sinyal yang tidak terekam. Itu sebabnya
# sumbu invariansi datar (+0,53 poin, p = 0,218, sepanjang probe 96,17% -> 20,00%).
#
# =====================================================================================
# YANG DIBANDINGKAN — DAN MENGAPA HANYA INI
# =====================================================================================
# Jumlah window turun 4x (13.200 -> ~3.300), sehingga basis perbandingannya berubah
# total. Angka w16384 TIDAK sebanding dengan tabel n=20 mana pun. Satu-satunya
# perbandingan yang sah adalah ERM w16384 versus ERM w4096 pada 8 seed yang SAMA.
#
# Pembanding ERM w4096 (n=8, seed identik): 89,40 +/- 4,18
#   per-torsi dari audit_ckpt_ext.py: torsi-0 74,18 | torsi-7 98,77 (angka n=20;
#   audit n=8 dijalankan ulang saat membaca hasil agar benar-benar berpasangan).
#
# ONGKOS TERUKUR (smoke 2 epoch, 2026-08-02 20:12): ~2-3 detik/epoch, jadi ~2,5 menit
# per seed dan ~20 menit untuk 8 seed. Perkiraan analitik sebelumnya (4x lebih mahal,
# ~3 jam) KELIRU: ia menghitung attention O(token^2) tetapi mengabaikan bahwa jumlah
# batch per epoch ikut turun 4x dan GPU menyerap biaya attention pada token sekecil
# 256. Angka terukur yang berlaku, bukan estimasinya.
#
# CATATAN CONFOUND YANG HARUS DILAPORKAN. token_size tetap 64, sehingga jendela 4x
# lebih panjang menghasilkan 4x lebih banyak token (64 -> 256).
# token_size TIDAK dinaikkan ke 256 dengan sengaja: token 256 sampel = 33,7 ms,
# sedangkan pembawa 60 Hz berperioda 16,7 ms — tokenizer sekasar itu meng-alias
# pembawa yang justru menjadi tempat sideband kerusakan menumpang. Jadi jumlah token
# ikut berubah, dan itu bagian dari perlakuan, bukan variabel yang dikendalikan.
# Harus dinyatakan eksplisit di metodologi.
#
# =====================================================================================
# KRITERIA GUGUR — DIDAFTARKAN DI MUKA, SEBELUM ANGKA TERLIHAT
# =====================================================================================
# PELAJARAN DARI UJI D DAN UJI A: kriteria harus MEMBELAH HABIS ruang hasil. D-P2 hanya
# menyebut rerata sehingga efek nyata di varians nyaris lolos; A-P1 memberi dua ambang
# berjauhan (<93,57 berhasil, >=95 gugur) dan hasilnya jatuh di zona mati 94,02.
# Karena itu C-P1 di bawah ditulis sebagai SATU ambang tunggal tanpa celah.
#
#   C-P1 (utama, ambang tunggal — tidak ada zona mati)
#       Ukuran: reduksi galat RELATIF, bukan poin absolut.
#           R(torsi) = (F1_16384 - F1_4096) / (100 - F1_4096)
#       Alasan wajib relatif: torsi-7 pada 98,77 hanya punya 1,23 poin ruang naik,
#       torsi-0 pada 74,18 punya 25,82. Membandingkan kenaikan poin absolut di antara
#       keduanya menyesatkan secara struktural.
#
#       LOLOS  bila  R(torsi-0) - R(torsi-7) > 0  pada >= 6 dari 8 seed berpasangan.
#       GUGUR  bila  tidak. (Dua kemungkinan itu habis membelah ruang hasil.)
#
#       Bila gugur, penjelasan resolusi-jendela (M14) gugur sebagai penyebab defisit
#       torsi rendah, dan klaim resolusi-jendela makalah (Sec. V-B) harus ditulis ulang dari "keyakinan
#       sedang" menjadi ditolak.
#
#   C-P2 (deskriptif, sudah terverifikasi secara aritmetika sebelum run)
#       Torsi-0 berpindah kategori: 0,96 -> 3,84 periode per jendela; resolusi
#       1,855 -> 0,464 Hz sehingga 2sf = 1,78 Hz berpindah dari DI BAWAH resolusi
#       menjadi teresolusi. Ini bukan prediksi empiris melainkan konsekuensi
#       aritmetika; dicatat agar angkanya masuk laporan, bukan sebagai uji.
#
#   C-P3 (varians — sumbu yang C-P1 tidak minta; ditambahkan setelah pelajaran Uji D)
#       Laporkan sd antar-seed dan uji-F terhadap ERM w4096 TERLEPAS dari hasil C-P1.
#       Uji A memburuk signifikan di sumbu ini (sd 1,71 -> 4,20; p = 0,0303) dan itu
#       hanya ketahuan karena diperiksa terpisah.
#
# ATURAN DAYA: n=8 PENAPIS. Sebelum memutuskan perluasan, hitung dayanya dulu —
# pelajaran Uji D: n=20 hanya berdaya 0,393 untuk kontras d = 0,397, jadi perluasan
# tidak otomatis menyelesaikan apa pun.
#
# PRASYARAT: data/processed/motor/motor_multiclass_w16384_s16384_ss_env.npz harus ada
#   ${PYTHON:-python} -u scripts/run_preprocessing.py --motor --window 16384 --stride 16384
#
# MELANJUTKAN SETELAH LAPTOP DIMATIKAN:
#   cd <repository root>
#   bash scripts/_run_window16384.sh
#
#   Ulang satu tahap : rm logs/.done_w16384/<nama_tahap>
#   Ulang semuanya   : rm -rf logs/.done_w16384
~~~~

## scripts/audit/aggregate_seeds.py

Lines 1-16:

~~~~text
"""aggregate_seeds.py — gabungkan hasil per-seed lintas log dan uji beda antar metode.

Angka n=20 sebelumnya dihitung ad-hoc; skrip ini menggantikannya supaya tabel utama dapat
direproduksi dari log mentah kapan saja.

Cara kerja:
  1. Segmentasi log berdasarkan penanda tahap ("=== [k/n] NAMA ===" atau "###### NAMA ######").
  2. Kumpulkan baris "[SEED n] CV-mean over ... Win-F1=xx.xx%" di dalam tiap segmen.
  3. Gabungkan run 8-seed lama dengan run 14-seed baru per metode. Seed 42 dan 123 hadir di
     keduanya sebagai KONTROL REPRODUKSIBILITAS — penggabungan hanya sah bila keduanya cocok,
     dan skrip menolak menggabung bila tidak.
  4. Uji berpasangan terhadap ERM pada seed yang sama (paired t + Wilcoxon).

Jalankan:
    ${PYTHON:-python} -u scripts/audit/aggregate_seeds.py
"""
~~~~

## scripts/audit/audit_a6_resolvabilitas.py

Lines 1-24:

~~~~text
"""audit_a6_resolvabilitas.py — A6: apakah sumbu 2sf TERBACA di satu window?

Kriteria A6-P1..A6-P3 TERKUNCI di docstring ini, ditulis sebelum satu baris kode ini ada.
Tanpa GPU. Hasilnya dilaporkan di Sec. V-C makalah CENIM 2026 (panel b gambar LODO).

A6 mengukur klaim FISIK inti makalah — *tanda kerusakan tidak terselesaikan di beban rendah
pada satu window* — yang sejauh ini hanya bersandar pada bukti tak langsung.

MENGAPA PROBE, BUKAN PROMINENSI
===============================
Memungut puncak gagal: 2sf ADA tetapi tidak dominan (puncak nyata 13,4-14,8 Hz).
Mengukur prominensi di frekuensi ramalan hukum slip SIRKULAR untuk pertanyaan
domain-lawan-kelas: targetnya bergantung torsi, jadi ia mengukur targetnya sendiri.

Probe non-sirkular: fitur = spektrum envelope pada GRID FREKUENSI TETAP, tidak pernah
melihat hukum slip. Kalau 2sf terbaca, domain dapat diprediksi darinya.

KENDALI ADALAH INTI RANCANGAN
=============================
torsi-0 vs torsi-1 dan torsi-6 vs torsi-7 punya selisih 2sf yang PERSIS SAMA (0,582 Hz),
tetapi yang kedua berada di frekuensi absolut 3x lebih tinggi, jauh dari kebocoran DC.
Tanpa kendali ini, A6 tidak dapat membedakan "2sf tak terbaca di beban rendah" dari
"probe saya jelek".
"""
~~~~

## scripts/audit/audit_ckpt_ext.py

Lines 1-24:

~~~~text
"""audit_ckpt_ext.py — Audit checkpoint EXT: per-torsi, per-kelas, dan domain probe.

Menjawab tiga pertanyaan yang tidak bisa dijawab dari log:

1. PER-TORSI. Metrik yang dilaporkan menggabungkan torsi 0 dan 7 jadi satu angka.
   Keduanya arah ekstrapolasi berlawanan (di bawah vs di atas rentang latih), jadi
   kemenangan di satu sisi bisa tertutup kekalahan di sisi lain.

2. PER-KELAS. r1b (satu batang patah) punya sideband paling lemah; kalau GRL
   menggosok residu fault, kelas itu yang paling dulu jatuh.

3. DOMAIN PROBE — yang paling menentukan. Log mencatat domain_acc dari
   diskriminator DARING yang dilatih bersamaan dengan backbone. Nilai di bawah
   chance bisa berarti dua hal yang sangat berbeda:
     (a) representasi memang anti-prediktif terhadap domain, atau
     (b) diskriminator sekadar tertinggal karena backbone bergerak tiap langkah
         (artefak pengukuran, bukan sifat representasi).
   Probe menyelesaikannya: bekukan backbone, latih classifier domain BARU dari nol
   di atas fitur beku. Probe yang terlatih penuh tidak mungkin di bawah chance.
     probe ~ 1/6  -> fitur benar-benar invarian-domain
     probe >> 1/6 -> GRL hanya menipu diskriminator daring; info domain masih ada

Jalur data, standardisasi, dan split identik dengan train_motor_dann.py.
"""
~~~~

Lines 165-168:

~~~~text
    # D-P (27 Agu 2026): desimasi 2x. Dua lengan ERM berpasangan, flag identik,
    # T & df IDENTIK (2,1558 s / 0,4639 Hz); yang beda hanya panjang sekuens 16384 vs
    # 8192. Diaudit untuk D-P3 (per-torsi). Aturan --npz per lengan seperti blok Uji C.
    #   --npz motor_multiclass_w16384_s16384_ss_env.npz
~~~~

Lines 186-189:

~~~~text
    # backbone TIDAK lagi dikunci. Sebelum 7 Agu 2026 seluruh cabang menuliskan
    # "tsitplus" secara literal, sehingga checkpoint backbone lain dimuat ke
    # arsitektur yang salah, 33 kunci gagal termuat, dan hasilnya setingkat
    # kebetulan acak yang tetap tercetak sebagai angka.
~~~~

Lines 415-417:

~~~~text
            # R2-P3 — Rec-Accuracy. rec_ids sudah dibangun (baris 250) tetapi
            # tidak pernah dikeluarkan sebelumnya, sehingga kolom Rec-Accuracy kosong.
            # Kolom DITAMBAHKAN DI UJUNG supaya indeks m[8]/m[9]/m[10] di bawah tetap sah.
~~~~

## scripts/audit/audit_dataset.py

Lines 1-13:

~~~~text
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
~~~~

## scripts/audit/audit_fs_ratio.py

Lines 1-16:

~~~~text
"""audit_fs_ratio.py — uji ulang dugaan galat laju sampling (Sec. III-A makalah CENIM 2026).

Dugaan semula (2026-07-31): fs getaran = 7.667,5 Hz, bukan 7.600 Hz. Diturunkan dari rasio
panjang kanal listrik : kanal getaran pada SATU rekaman, dengan asumsi kedua kanal merekam
rentang waktu yang persis sama.

Skrip ini menguji asumsi itu pada SELURUH rekaman. Hanya metadata bentuk (`.shape`) yang
dibaca — tidak ada sampel yang dimuat — sehingga pemindaian 5 berkas 1,4 GB berjalan cepat.

Hasil 2026-08-02 (400 rekaman): panjang kanal listrik konstan 1.001.000, tetapi panjang
kanal getaran bervariasi 7,84%. fs tersirat karenanya bergerak 7.120,8–7.679,4 Hz dengan
median 7.600,4 Hz — 0,4 Hz dari 7.600. Dugaan galat DITARIK; fs = 7.600 Hz benar.

Jalankan:
    ${PYTHON:-python} -u scripts/audit/audit_fs_ratio.py
"""
~~~~

## scripts/audit/audit_grad_traj.py

Lines 1-14:

~~~~text
"""audit_grad_traj.py — Lintasan rasio gradien adversarial/klasifikasi SELAMA pelatihan.

Pengukuran statis pada checkpoint hanya memberi titik akhir. Yang menentukan adalah
bagaimana keseimbangan kedua objektif BERGESER sepanjang pelatihan:

    rasio(t) = ||d L_domain / d theta_backbone||  /  ||d L_fault / d theta_backbone||

Hipotesis dari pengukuran statis: rasio ~0.6 di awal (seimbang, sesuai maksud Ganin)
lalu meledak ke ~10 di akhir, karena L_fault jatuh ke ~0 (data latih dihafal) sementara
lambda justru DINAIKKAN oleh jadwal anneal. Kalau benar, backbone di paruh akhir
pelatihan digerakkan hampir sepenuhnya oleh objektif adversarial.

Protokol identik battery 8-seed: tsitplus, EXT, fp32, no-augment, fixed-budget, 50 epoch.
"""
~~~~

## scripts/audit/audit_hp_grid.py

Lines 1-37:

~~~~text
"""audit_hp_grid.py — efek utama PANJANG di setiap sel (backbone, lr).

MENGAPA SKRIP INI ADA
=====================
Menjalankan faktorial R3 di banyak sel (backbone x lr) menciptakan bahaya yang tidak ada
saat hanya satu sel dijalankan: dengan cukup banyak sel, SELALU ada satu yang tidak
mereplikasi karena kebetulan. Kalau aturan bacanya baru disusun setelah kisinya terisi,
sel yang menyimpang bisa dibingkai sebagai "kasus khusus" atau sel yang cocok dipilih
sebagai "yang representatif". Dua-duanya taman bercabang, dan dua-duanya tak terdeteksi
dari luar.

Karena itu aturan baca dikunci DI SINI, dalam kode, sebelum sel keempat ada:

  1. Metrik utamanya SATU: efek utama PANJANG pada Rec-Acc torsi-0, yaitu rata-rata dua
     kontras marginal (panjang @ volume rendah, panjang @ volume tinggi). Konvensi ini
     disalin dari audit_r3_faktorial.py dan menghasilkan +15,88 untuk TSiTPlus @ 1e-3.
     TIDAK ADA metrik kedua yang boleh dipromosikan jadi utama belakangan.
  2. Vonisnya dibaca dari SELURUH kisi sekaligus, bukan sel per sel:
       - efek positif di SEMUA sel                      -> bertahan lintas konfigurasi
       - positif di sebagian besar, satu sel menyimpang -> bertahan, DAN sel yang menyimpang
                                                          wajib dilaporkan dengan namanya
       - tanda campur aduk tanpa pola                   -> efeknya bergantung konfigurasi;
                                                          klaim makalah (Sec. IV-B) harus dilemahkan
  3. Ambang lolos-per-sel dikunci sama dengan yang ditulis di kepala _run_lr_faktorial.sh
     pada 8 Agu 2026 sebelum satu angka pun ada: >= +10 bertahan, +3..+10 arah saja,
     < +3 tidak bertahan.
  4. Tidak ada uji signifikansi lintas sel. Delapan seed per sel dan sel yang jumlahnya
     sedikit tidak menopang koreksi perbandingan ganda apa pun yang bermakna. Kisi ini
     DESKRIPTIF, dan harus ditulis begitu.

Sel keluarga CNN, kalau nanti diisi, menjawab pertanyaan BERBEDA: BatchNorm + GAP membuat
baseline auto-invarian terhadap beban by construction, jadi selnya terkonfusi dengan
pencucian amplitudo dan tidak dapat diatribusikan ke lr. Skrip menandainya, dan angkanya
TIDAK boleh dijumlahkan ke dalam vonis butir 2 di atas.

Jalankan dari akar repositori:  ${PYTHON:-python} scripts/audit/audit_hp_grid.py
"""
~~~~

## scripts/audit/audit_hull_window.py

Lines 1-31:

~~~~text
"""audit_hull_window.py — R1: apakah M13 (convex hull) bertahan di granularitas window?

Kriteria R1-P0..R1-P3 TERKUNCI di docstring ini, ditulis sebelum satu angka pun dilihat.
Skrip ini hanya mengeksekusinya. Hasilnya menopang perbandingan convex hull di Sec. V-A
makalah CENIM 2026 (perbandingan itu post-hoc, dan dinyatakan begitu di sana).

MENGAPA SKRIP INI ADA
=====================
M13 — "torsi-0 53,1% di luar convex hull sumber, torsi-7 18,2%" — adalah SELURUH premis
Uji B, dan salah satu dari dua penjelasan teoretis yang tersisa. Angka itu dikutip
di lima berkas dan tidak diturunkan di satu pun; tidak ada skrip yang menghitungnya.

U3 dibatalkan persis karena premisnya berdiri di rekaman penuh (|d| = 2,18) dan
BERBALIK di tingkat window (0,32 vs 0,35). Uji B mengaugmentasi window per window, bukan
distribusi, jadi yang harus berdiri adalah pernyataan tingkat window.

APA YANG DIUKUR
===============
Deskriptor: sigma per window per kanal (std lintas waktu) — tepat besaran yang transform
Uji B manipulasi.

  AGREGAT  1 titik per torsi        hull = [min, maks] atas torsi 1-6
  WINDOW   1 titik per window       hull = [q01, q99] atas seluruh window sumber

Statistik utama sama di keduanya sehingga langsung dibandingkan:

    ekstrapolasi% = jarak keluar hull / lebar hull x 100

Jalankan:
    ${PYTHON:-python} -u scripts/audit/audit_hull_window.py
"""
~~~~

## scripts/audit/audit_kanal_arus.py

Lines 1-29:

~~~~text
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
~~~~

Lines 93-97:

~~~~text
    """fs = 60,00 x N / k_puncak, interpolasi puncak parabolik untuk presisi sub-bin.

    Frekuensi jaringan dikunci grid pada 60,00 Hz; puncak dominan spektrum arus PASTI
    fundamental itu. Rasio panjang TIDAK dipakai (kriteria A2.0 menolaknya secara eksplisit).
    """
~~~~

Lines 118-122:

~~~~text
    """IDENTIK logikanya dengan audit_window_sweep.prominensi_envelope, fs jadi parameter.

    Fundamental hilang oleh demodulasi, jadi tidak ada kontaminasi seperti jalur mentah
    (A2-P4). Syarat keterpisahan f2s >= 2*df; bila tidak, NaN — jawaban fisik, bukan gagal ukur.
    """
~~~~

Lines 176-183:

~~~~text
    # DUA KOREKSI TERHADAP LINTASAN 20:05 — keduanya CACAT SKRIP, bukan ubahan kriteria:
    #  (1) kalibrasi WAJIB di steady-state. Lintasan pertama memakai rekaman penuh, sehingga
    #      segmen-1 memuat transien start-up (RMS 1,90 vs 1,06) dan biasnya terbaca sebagai
    #      klok tak stabil. Seluruh proyek ini mengukur di steady-state (onset_steady);
    #      gerbangnya saja yang lupa menerapkannya.
    #  (2) gerbang ditegakkan PER REKAMAN, bukan sebagai maks global. Itu bacaan harfiah
    #      "sebaran DI DALAM satu rekaman" (Opsi 1): rekaman yang fs-nya tidak dapat
    #      dipatok < 0,1% DIBUANG, bukan diakomodasi dengan melonggarkan ambang.
~~~~

## scripts/audit/audit_lodo.py

Lines 1-56:

~~~~text
"""audit_lodo.py — analisis LEAVE-ONE-TORQUE-OUT, 8 lipatan x 5 seed, ERM.

Kriteria L-P1..L-P3 TERKUNCI di docstring ini dan hasilnya dilaporkan di Sec. V-A makalah
CENIM 2026 (panel a gambar LODO). Ditulis sebelum satu lipatan dijalankan. Skrip ini hanya MEMBACA; ia tidak boleh
menambah, melonggarkan, atau menawar satu pun ambang di sana.

Dijalankan setelah `scripts/_run_lodo.sh` selesai:
    ${PYTHON:-python} -u scripts/audit/audit_lodo.py

DUA JEBAKAN YANG SKRIP INI TANGANI SECARA EKSPLISIT
===================================================

1. **fold_idx BUKAN nomor torsi.** `GroupKFold(n_splits=8)` menyeimbangkan ukuran grup;
   ia TIDAK mengurutkan grup. Pada run 3 Agu, lipatan 0 menahan **torsi-7**, lipatan 1
   menahan torsi-6, dan seterusnya menurun. Menganggap fold_idx == torsi akan MEMBALIK
   sumbu L-P1 dan menghasilkan rho bertanda terbalik tanpa satu pun tanda kesalahan.
   Karena itu torsi yang ditahan selalu dibaca dari baris `Test torques : [t]` di log,
   tidak pernah disimpulkan dari nomor lipatan.

2. **Log ganda.** Run 3 Agu diluncurkan dua kali secara bersamaan (dua proses menulis
   penanda `.done_lodo` yang sama). Semua `logs/lodo_*.log` dibaca, hasilnya diindeks
   per (torsi, seed), dan duplikat wajib COCOK. Ketidakcocokan dilaporkan, bukan
   dirata-ratakan diam-diam — dua proses identik yang menghasilkan angka berbeda adalah
   temuan tentang determinisme, bukan derau untuk dihaluskan.

3. **Log kampanye LAIN yang namanya ikut tertangkap glob.** Ditambahkan 8 Agu 2026.
   `_run_lodo_metode.sh` menulis ke `logs/lodo_metode_*.log`, yang COCOK dengan pola
   `lodo_*.log` di atas. Log itu memuat baris `SEED n | FOLD k/8`, `Test torques : [t]`,
   dan `CV-mean` dengan bentuk yang persis sama, jadi tanpa penyaring ia akan terbaca
   sebagai ERM. Karena ERM LODO sudah lengkap 8x5, SETIAP hasil metode akan menabrak
   kunci (torsi, seed) yang sudah terisi dan muncul sebagai "ketidakcocokan duplikat" —
   yaitu empat puluh temuan determinisme PALSU pada skrip yang justru dirancang untuk
   memperlakukan ketidakcocokan sebagai temuan serius.

   Penyaringnya memakai nama run di baris `DANN TRAINING: <nama>` (ERM = `LODO_baseline`,
   metode = `LODOM_*`), BUKAN nama berkas. Nama berkas adalah kebetulan penamaan skrip
   peluncur; nama run ditulis oleh kode pelatihan dari argumen `--name`, jadi ia bertahan
   kalau suatu saat log digabung, diganti nama, atau dialihkan.

SUMBER ANGKA A6
===============
Delapan angka keterbacaan per torsi berasal dari perluasan A6-P3 ke seluruh
torsi yang dijalankan ad-hoc — `audit_a6_resolvabilitas.py` yang ada di repo hanya
menghitung TIGA probe. Supaya L-P1 dapat direproduksi utuh dari repo, skrip ini
MENGHITUNG ULANG kedelapan angka itu memakai fitur dan probe yang sama persis
(diimpor, bukan disalin), lalu mencocokkannya dengan nilai terkunci di A6_TERKUNCI. Selisih
> 0,5 poin menghentikan skrip: yang salah bisa saja transkripsinya, dan pra-registrasi
tidak boleh diam-diam bergeser.

`--a6-terkunci` melewati perhitungan ulang (~2 menit) dan memakai nilai A6_TERKUNCI apa adanya.

RAMBU PLAFON — dipaksakan oleh kode, bukan diserahkan ke pembaca
================================================================
Plafon diperiksa SEBELUM rho dilaporkan. Bila >= 6 dari 8 lipatan di atas 98, ruang
gerak korelasi habis dan L-P1 dinyatakan GAGAL-UKUR — bukan "rho rendah yang bermakna".
"""
~~~~

Line 82:

~~~~text
# --- konstanta pra-registrasi L-P1..L-P3 — JANGAN diubah tanpa mengubah kriteria di docstring lebih dulu ---
~~~~

Lines 213-215:

~~~~text
# ----------------------------------------------------------------------------------
# 2. Keterbacaan A6 per torsi — perluasan A6-P3, fitur & probe DIIMPOR dari A6
# ----------------------------------------------------------------------------------
~~~~

Lines 264-266:

~~~~text
# ----------------------------------------------------------------------------------
# 3. Kriteria
# ----------------------------------------------------------------------------------
~~~~

## scripts/audit/audit_lodo_metode.py

Lines 1-62:

~~~~text
"""audit_lodo_metode.py — LODO 8 lipatan x 5 seed untuk TIGA METODE TERKUAT lawan ERM.

Pasangan dari `scripts/_run_lodo_metode.sh`. Skrip ini hanya MEMBACA log.

    ${PYTHON:-python} -u scripts/audit/audit_lodo_metode.py

APA YANG DIUJI
==============
Separuh NEGATIF kontribusi paper — "tidak ada intervensi sisi-objektif yang terpisah
dari baseline" — sampai 8 Agu 2026 berdiri di atas SATU split domain untuk seluruh
keluarga metode (batasan split tunggal, Sec. VI makalah CENIM 2026), sementara separuh POSITIFnya sudah punya
LODO 8 lipatan. Kampanye ini memberi tiga lengan terkuat protokol yang sama dengan
separuh positif, sehingga klaim negatifnya berhenti bergantung pada satu split.

    89,25  BT+SupCon regresi-idx   LODOM_BT_didx
    87,50  SupCon                  LODOM_supcon
    86,00  DANN+SupCon             LODOM_dannsupcon

Pembandingnya ERM LODO (`LODO_baseline`) dengan seed yang SAMA (42,123,2025,7,17) dan
lipatan yang sama, jadi perbandingannya BERPASANGAN per (torsi, seed) — bukan dua
rerata yang kebetulan disandingkan.

KRITERIA LM-P1..LM-P3 — DIKUNCI 8 Agu 2026 07:40, SEBELUM hasil lengan mana pun dibaca
=======================================================================================
Keadaan pengetahuan saat kriteria ditulis, dinyatakan supaya tidak ada klaim buta yang
palsu: lima belas dari empat puluh hasil `LODOM_supcon` SUDAH terlihat (lipatan 0-2,
yaitu torsi 7, 6, 5) dan seluruhnya mentok — Rec-Acc 100,00 di kelima seed, Win-F1
98,97-100,00. Tidak satu pun hasil `LODOM_dannsupcon` atau `LODOM_BT_didx` ada saat
kriteria dikunci. Ambang di bawah TIDAK dipilih dengan melihat angka-angka itu; ia
diturunkan dari uji yang dipakai, dan penurunannya dicetak oleh kode.

LM-P1 (UTAMA, per lengan)
    Wilcoxon signed-rank dua sisi atas selisih berpasangan (metode - ERM) pada Win-F1
    torsi tertahan, 8 torsi x 5 seed = 40 pasang. Bonferroni atas TIGA lengan:
    alpha = 0,05/3 = 0,016667.
      p <  alpha  -> lengan TERPISAH dari ERM. Klaim negatif paper MELEMAH untuk
                     lengan itu dan wajib dilaporkan apa adanya.
      p >= alpha  -> lengan TIDAK TERPISAH. Klaim negatif MENGUAT: ia kini bertahan
                     di bawah 8 lipatan, bukan satu split.

LM-P2 (RAMBU DAYA — diperiksa SEBELUM p dibaca)
    Wilcoxon signed-rank membuang selisih nol. Dengan n_eff pasang tak-nol, nilai p
    dua sisi TERKECIL yang mungkin dicapai adalah 2^(1-n_eff), berapa pun datanya.
    Bila 2^(1-n_eff) > alpha, uji itu TIDAK DAPAT menolak sekalipun setiap pasang
    berpihak pada satu arah; hasilnya wajib dibaca GAGAL-UKUR, bukan "tidak terpisah".
    Ambangnya karena itu DITURUNKAN, bukan dipilih: n_eff minimum adalah bilangan bulat
    terkecil yang memenuhi 2^(1-n) <= alpha, dan kode mencetak penurunannya.

    Rambu ini bukan kehati-hatian teoretis. ERM LODO mentok di 6 dari 8 lipatan
    (audit_lodo.py, rambu plafon); bila lengan metode juga mentok di lipatan yang sama,
    pasangan di sana berselisih nol dan menguap dari uji.

LM-P3 (CAKUPAN)
    Tidak ada vonis untuk sebuah lengan sampai 8 torsi x 5 seed lengkap bagi lengan itu.
    Lengan yang belum lengkap dilaporkan deskriptif dan diberi label DITAHAN.

YANG DICABUT DAN YANG TIDAK
===========================
Ini PENCABUTAN SEBAGIAN B7. Sesudah kampanye ini B7 berbunyi: tercabut untuk ERM dan
untuk ketiga lengan terkuat; MASIH BERLAKU untuk dua belas lengan sisanya. Kode
mencetak kalimat itu di akhir supaya ia tidak pernah dilaporkan sebagai cabut penuh.
"""
~~~~

## scripts/audit/audit_normalisasi.py

Lines 1-43:

~~~~text
"""audit_normalisasi.py — Uji N dan P6: sapuan normalisasi, dan kanal envelope kedua.

Kriteria N-P1, N-P2, P6 TERKUNCI di docstring ini, ditulis sebelum satu angka dilihat.
Tanpa GPU. Latar: normalisasi per-kanal, Sec. IV-C makalah CENIM 2026.

MENGAPA UJI INI ADA
===================
P1/P2/P4/P5 semuanya berbentuk "alternatif lawan yang sekarang", dan yang sekarang menang
tiap kali. Itu MEMATAHKAN PESAING, bukan MENEMUKAN YANG TERBAIK. Untuk normalisasi bahkan
lebih tipis lagi: dari tujuh kandidat yang masuk akal, hanya dua yang pernah disentuh, dan
satu di antaranya (by_sample=False) sudah terpasang di dann_factory.py tanpa pernah
dijalankan sekali pun.

APA YANG DIUKUR
===============
Probe linear, GroupKFold PER REKAMAN, resep identik audit_a6_resolvabilitas. Dua sumbu:
  N-P1  akurasi KELAS di torsi-0   -> yang ingin dinaikkan
  N-P2  akurasi TORSI              -> yang TIDAK boleh ikut naik
Kandidat yang menaikkan keduanya hanya menaikkan SNR umum, bukan memisahkan kelas dari domain.

Fitur = spektrum log-magnitudo per kanal. Dipilih karena tak berubah terhadap geseran waktu
(probe linear pada deret waktu mentah tidak bermakna) sekaligus tetap membawa efek penskalaan
antar-kanal, yaitu persis yang sedang diuji.

P6 — MENGAPA IA MENJAWAB APA YANG P5 TIDAK JAWAB
=================================================
P5 menguji SATU SKALAR per pita (prominensi 2sf) lewat jarak Mahalanobis. Model tidak pernah
melihat skalar itu; ia melihat seluruh bentuk gelombang envelope. P6 memberi probe spektrum
envelope PENUH dari satu pita lawan dua pita.

Validasi silang membuatnya mengendalikan diri sendiri: menambah blok fitur tak informatif
MENURUNKAN akurasi CV lewat varians, berbeda dari Mahalanobis yang hampir selalu naik saat
fitur ditambah. Karena itu P6 tidak butuh kendali umpan.

RAMBU
=====
Ini PENYARING untuk memilih kandidat GPU, bukan vonis. Probe linear buta terhadap struktur
non-linear, dan buta total terhadap dinamika optimisasi serta skala gradien -- yang untuk
normalisasi kemungkinan justru mekanisme utamanya.

Jalankan:
    ${PYTHON:-python} -u scripts/audit/audit_normalisasi.py
"""
~~~~

Lines 172-179:

~~~~text
    """Gerbang plafon. Bila probe acuan sudah mentok, selisih antar kandidat tidak
    dapat dibaca dan sapuan HARUS dilaporkan sebagai gagal-ukur.

    Dipasang setelah versi pertama uji ini memakai kontras 5-kelas di torsi-0 dan
    mendapat 99,7-99,9% untuk SETIAP kandidat — plafon sempurna, nol ruang gerak.
    Itu kelas kesalahan yang sama dengan L-P1. A6 memakai kontras BINER
    rs vs r1b justru karena ia yang sulit (79,8%).
    """
~~~~

## scripts/audit/audit_r2_per_torsi.py

Lines 1-15:

~~~~text
"""audit_r2_per_torsi.py — R2 bagian A: pecahan per-torsi seluruh keluarga metode.

Kriteria R2-P1..R2-P3 TERKUNCI di docstring ini, ditulis sebelum satu angka dibaca.
Hasilnya adalah kolom low/high Tabel I dan galat tersisa di Sec. IV-A makalah CENIM 2026. Skrip ini hanya membaca `audit_ext_family.npz` — tidak
melatih, tidak menginferensi, tanpa GPU.

MENGAPA
=======
Tabel I makalah melaporkan F1 **agregat** beserta kolom per-torsi. Angka agregat dapat
menyembunyikan asal-usulnya: dua metode yang dicurigai berubah makna begitu dipecah
per-torsi. R2 memeriksa apakah itu berlaku untuk SELURUH keluarga, bukan hanya keduanya.

Kolom `audit_ext_family.npz` (format 10 kolom, sebelum dua kolom geometri ditambahkan):
    0 seed | 1 f1_total | 2 f1_torsi0 | 3 f1_torsi7 | 4-8 f1 per kelas | 9 domain_probe
"""
~~~~

## scripts/audit/audit_r3_faktorial.py

Lines 1-21:

~~~~text
"""audit_r3_faktorial.py — R3: memisahkan PANJANG JENDELA dari JUMLAH WINDOW.

Kriteria R3-P1..R3-P4 TERKUNCI di docstring ini, ditulis sebelum satu angka lengan dibaca.
Hasilnya adalah Tabel II dan Sec. IV-B makalah CENIM 2026. Skrip ini hanya mengeksekusinya.

INTI PERSOALAN
==============
Klaim positif satu-satunya proyek ini (+13,16 di torsi-0) diukur di w16384-ovl. Uji C
sudah memisahkan efek data dari efek resolusi, tetapi di **F1 tingkat window** — dan window
16384 memuat 4x lebih banyak bukti PER KEPUTUSAN daripada window 4096. Sebagian dari
"+18,04 efek resolusi" karena itu mungkin bukan generalisasi yang lebih baik melainkan
tugas yang dibuat lebih mudah.

Keempat lengan mengklasifikasi 100 REKAMAN FISIK YANG SAMA (diverifikasi 19:30). Karena itu
Rec-Accuracy sebanding lintas panjang jendela; Win-F1 tidak. R3 mengulang dekomposisinya di
satuan yang sebanding.

                    w4096 (res rendah)      w16384 (res tinggi)
  ~3.200 window     W4096SUB                W16384
  ~12-13k window    FP32_tsit_ext           W16384OVL
"""
~~~~

## scripts/audit/audit_resolusi_frekuensi.py

Lines 1-54:

~~~~text
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
~~~~

## scripts/audit/audit_slip.py

Lines 1-23:

~~~~text
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
~~~~

## scripts/audit/audit_stats2.py

Lines 1-5:

~~~~text
"""Uji statistik atas hasil audit checkpoint (berpasangan per-seed, n=8).

Bagian A menghasilkan perbandingan probe beban beku ERM vs DANN lambda 1,0 di Sec. IV-D
makalah CENIM 2026 (96,16% -> 93,57%, t = -6,16). Membaca `audit_ext_family.npz`.
"""
~~~~

## scripts/audit/audit_steady_state.py

Lines 1-11:

~~~~text
"""audit_steady_state.py — panjang steady-state terpendek lintas 400 rekaman.

Makalah CENIM 2026 (Sec. III-A, V-D) menyatakan segmen steady-state terpendek berdurasi
17,96 s. Angka itu juga menjadi batas atas `--window` di `run_preprocessing.py`
(`N_STEADY_SHORTEST = 136470` sampel pada 7,6 kHz). Skrip ini menurunkannya dari data mentah
dengan kode ekstraksi yang sama persis dengan preprocessing (tahap s1 dan s2), tanpa
envelope dan tanpa windowing.

Jalankan dari akar repositori:
    ${PYTHON:-python} -u scripts/audit/audit_steady_state.py [--data-dir DIR]
"""
~~~~

## scripts/audit/npdann_inlp.py

Lines 1-50:

~~~~text
"""npdann_inlp.py — NP-DANN tahap 1: telusuri kurva akurasi-vs-invariansi lewat INLP.

LATAR (Sec. IV-D makalah CENIM 2026)
    Lima belas metode diaudit; pada trunk tidak satu pun menghapus >3.3% informasi domain,
    dan titik paling invarian yang pernah dicapai apa pun adalah 85.54% (z_task OFS o3/d0.5).
    Sumbu invariansi membentang 16.67%-100%; rentang 16.67%-85.54% — 83% sumbu — BELUM
    PERNAH disentuh satu eksperimen pun, karena lambda tidak mengendalikan invariansi.

METODE
    INLP (Ravfogel et al., ACL 2020, arXiv:2004.07667): latih classifier linear yang
    memprediksi domain dari fitur, proyeksikan fitur ke nullspace baris-ruang bobotnya,
    ulangi. Tiap iterasi membuang <= n_domains-1 arah. Jumlah iterasi k adalah DIAL —
    knob yang lambda tidak pernah bisa jadi.

    P_0 = I
    ulangi k kali:
        W  <- classifier linear domain pada (F @ P)          # (n_domains, d)
        B  <- basis ortonormal baris-ruang W (lewat SVD)
        P  <- (I - B^T B) @ P

PROTOKOL BEBAS KEBOCORAN
    P dipelajari HANYA dari fitur torsi latih (1-6) memakai label domain latih.
    Kepala fault dilatih ulang pada fitur latih terproyeksi, lalu dievaluasi pada torsi
    ekstrem 0 & 7 yang tak pernah dilihat. Torsi uji tidak menyentuh P maupun kepala fault.

PROBE NONLINEAR — ditambahkan 8 Agu 2026
    Keberatan yang beralasan: INLP hanya menjamin hilangnya
    keterbacaan LINEAR. Probe linear yang jatuh 96,17 -> 20,00 karena itu tidak
    membuktikan informasi beban PERGI; ia bisa saja sekadar menjadi nonlinear, dan
    proyeksi nullspace tidak menyentuh yang nonlinear sama sekali.

    Makalah (Sec. IV-D) menutup ini dengan caveat. Skrip ini menutupnya dengan pengukuran: pada
    fitur terproyeksi yang SAMA, di lipatan GroupKFold yang SAMA, sebuah MLP satu
    lapis tersembunyi diadu dengan regresi logistik.
      probe_nl jatuh ke chance juga  -> informasi domain memang pergi; caveatnya boleh
                                        dilunakkan menjadi pengukuran.
      probe_nl bertahan tinggi       -> caveat R2 TEGAK dan terkuantifikasi: yang INLP
                                        hapus hanya bagian linearnya.

    KONTROL LABEL ACAK. MLP berkapasitas lebih besar daripada probe linear, jadi
    "di atas chance" saja bukan bukti — ia bisa datang dari kapasitas, bukan dari
    sinyal. Karena itu probe yang sama dijalankan atas label domain yang DIACAK, di
    k terkecil dan k terbesar. Angka itulah chance empiris untuk probe, split, dan
    ukuran sampel ini; chance analitik 100/n_domain hanya rujukan kasar. Kontrol
    dibatasi dua titik karena biayanya sama dengan probe sungguhan di tiap titik.

KELUARAN
    Untuk tiap k: Win-F1 pada torsi ekstrem, probe domain LINEAR dan NONLINEAR
    (invariansi yang BENAR-BENAR tercapai), dan dimensi tersisa. Itulah kurvanya.
"""
~~~~

Lines 178-179:

~~~~text
    # kontrol label acak hanya di k terkecil dan terbesar: biayanya sama dengan probe
    # sungguhan, dan yang perlu diketahui adalah chance SEBELUM dan SESUDAH proyeksi.
~~~~

## scripts/audit/sanity_bt_dann.py

Lines 1-18:

~~~~text
"""sanity_bt_dann.py — verifikasi BT-DANN sebelum membakar jam GPU.

Pola yang sama dipakai sebelum menjalankan CDAN: implementasi baru
harus lolos pemeriksaan mekanis dulu, karena bug `loss_grad` terdahulu membuat 19
eksperimen mengukur suku adversarial yang inert tanpa ada yang menyadarinya.

Enam pemeriksaan:
  1. REGRESI   — bt_ratio=None harus memberi callback DANN lama, bukan varian BT.
  2. KAIT      — hook menangkap vektor fitur dan melepasnya lagi setelah fit.
  3. UKUR      — ||g_dom_mentah|| yang diprobe BEBAS dari lambda saat ini.
  4. KENDALI   — lambda_eff bergerak berlawanan arah dengan rasio mentah.
  5. KONVERGEN — rasio EFEKTIF mendekati rasio_target setelah beberapa batch.
  6. HIDUP     — gradien adversarial tetap sampai ke backbone (bukan inert).

Dijalankan di CPU dengan data sintetis agar tidak bertabrakan dengan pelatihan di GPU.

    ${PYTHON:-python} -u scripts/audit/sanity_bt_dann.py
"""
~~~~

## scripts/audit/sanity_domain_index.py

Lines 1-21:

~~~~text
"""sanity_domain_index.py — verifikasi UJI A (regresi indeks domain) sebelum GPU.

Rujukan: lengan LODO "BT + SupCon regresi-indeks" (Sec. VI makalah CENIM 2026).
Prasyarat protokol: implementasi adversarial baru WAJIB lolos
pemeriksaan mekanis sebelum membakar jam GPU.

Sebelas pemeriksaan:
  1. REGRESI   — domain_index=none tidak mengubah apa pun (kepala tetap kategoris).
  2. TABEL u   — normalisasi [0,1] HANYA dari torsi latih; torsi uji tidak menyentuhnya.
  3. IDENTIK   — 'f2s' dan 'slip' menghasilkan target yang sama persis (bukan dua ablasi).
  4. KEPALA    — DomainRegHead berbentuk feat->64->1 dan kapasitasnya sekembar DomainHead.
  5. LOSS      — loss domain benar-benar MSE terhadap u, bukan CE.
  6. NOL       — MSE = 0 tepat ketika D(z) == u (uji analitik, bukan sekadar 'kecil').
  7. GRL       — gradien MSE tetap dibalik GRL dan berskala linear terhadap lambda.
  8. KORELASI  — korelasi(D(z),u) benar: +1 pada prediksi sempurna, -1 pada pembalikan.
  9. BT SEJAJAR— probe rasio BT memakai loss yang SAMA (MSE), bukan CE yang tertinggal.
 10. LABEL     — verify_domain_labels tetap sah walau kepala hanya punya 1 keluaran.
 11. FIT       — fit sungguhan jalan, loss berhingga, telemetri korelasi terisi.

    ${PYTHON:-python} -u scripts/audit/sanity_domain_index.py
"""
~~~~

## scripts/make_subsample_w4096.py

Lines 1-37:

~~~~text
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
~~~~

## scripts/run_preprocessing.py

Lines 1-7:

~~~~text
"""
run_preprocessing.py - Entry point for preprocessing the motor dataset.

Usage:
    python scripts/run_preprocessing.py --motor
    python scripts/run_preprocessing.py --motor --window 2048 --stride 1024
"""
~~~~

## scripts/train_motor_dann.py

Lines 1-42:

~~~~text
"""
train_motor_dann.py - Load-Robust Motor Fault Classification via Domain-Adversarial DG.

Strategy: Domain Generalization via Inter-Source Adversarial Alignment
-----------------------------------------------------------------------
The GRL forces the backbone to produce representations that are invariant
across all seen torque levels. By induction, these features generalise to
the unseen held-out torque (Leave-Torque-Out fold). This is Domain Generalization
(DG), NOT Domain Adaptation — the validation torque is NEVER seen during training.

Compared baseline: same architecture without GRL (lambda_max=0)
                   = plain TSiTPlus with identical data/splits.
The baseline directly quantifies GRL's contribution to load robustness.

How to run
----------
    # Full DANN-DG (λ_max=1.0, recommended)
    python scripts/train_motor_dann.py --name motor_dann_lto_v1

    # Ablation: no adversarial signal (baseline)
    python scripts/train_motor_dann.py --lambda_max 0.0 --name motor_dann_baseline_v1

    # Weaker adversarial pressure (useful if fault accuracy degrades)
    python scripts/train_motor_dann.py --lambda_max 0.5 --name motor_dann_half_v1

Metrics
-------
  Window-level Accuracy : inflated (many windows per recording), secondary.
  Recording-level Accuracy: majority vote over all windows from the same
                             physical recording, UNSEEN TORQUES. This is the
                             paper's secondary metric (Sec. III-D); the primary
                             metric is window-level macro F1.

GRL Convergence Verification
------------------------------
Monitor [GRL-EPOCH] lines in training output. Healthy signs:
  - domain_loss approaches log(n_domains) = theoretical max entropy
  - domain_acc  approaches 1/n_domains   = random chance
  - STATUS: CONVERGING
If STATUS shows WARN (domain classifier still discriminative), the backbone
is not yet load-invariant — try increasing lambda_max or training epochs.
"""
~~~~

## src/models/backbones/tsitplus.py

Lines 1-38:

~~~~text
"""tsitplus.py — Wrapper backbone untuk tsai TSiTPlus (transformer ViT-style).

Tujuan ilmiah: backbone NON-GAP, NON-BatchNorm sebagai uji SYARAT-PERLU DANN.
CNN kita (InceptionTime/ResNet/CNN1D) memakai BatchNorm + Global Average Pooling
→ auto-invarian terhadap domain (beban) by construction → baseline sudah load-invarian
→ GRL tak punya ruang, hanya bisa overshoot (negative transfer, terbukti empiris).

TSiTPlus berbeda secara fundamental:
    - LayerNorm (bukan BatchNorm) → tak menghapus offset domain via statistik batch.
    - Positional encoding + CLS-token → MEMPERTAHANKAN informasi posisi/amplitudo/domain.
    - Fitur diambil dari CLS-token (z[:, :, 0]), BUKAN GAP atas dimensi waktu.

Hipotesis: baseline TSiTPlus akan MENGKODEKAN domain (domain_acc tinggi) dan overfit
beban → ada ruang nyata bagi DANN untuk memperbaiki generalisasi (DANN > baseline).

SeqTokenizer (token_size) memampatkan seq_len (4096) menjadi token agar attention
O(n^2) tractable. Head klasifikasi tsai dibuang (c_out dummy); peran head dimainkan
FaultHead/DomainHead di DANNModel.

BATAS JUMLAH TOKEN — ditambahkan 4 Agu 2026 untuk lengan w65536
===============================================================
token_size tetap 64 membuat jumlah token tumbuh linear terhadap seq_len:

    w4096  ->   64 token       w16384 ->  256 token       w65536 -> 1024 token

Attention O(n^2) pada 1024 token, batch 64, 8 kepala, 3 lapis, tidak muat di GPU ini.
Lebih penting daripada memori: membiarkan token tumbuh membuat perbandingan
w16384 lawan w65536 mengubah DUA hal sekaligus — rentang waktu per token DAN jumlah
token — sehingga kenaikan apa pun tidak dapat diatribusikan ke resolusi.

`token_size = max(64, seq_len // MAKS_TOKEN)` mengunci jumlah token di 256 dan
membiarkan rentang waktu per token yang tumbuh. Itu persis variabel yang R-P1 uji.

NETRAL untuk seluruh lengan yang sudah ada — diverifikasi, bukan diasumsikan:
    w4096  : 4096//256  =  16 -> max(64,16) = 64 ->  64 token  (sama seperti dulu)
    w16384 : 16384//256 =  64 -> max(64,64) = 64 -> 256 token  (sama seperti dulu)
    w65536 : 65536//256 = 256 -> max(64,256)= 256 -> 256 token (baru, dibatasi)
"""
~~~~

## src/preprocessing/stages/s2_steady_state.py

Lines 1-5:

~~~~text
"""s2_steady_state.py — Tahap 2: Deteksi Steady-State (Dataset A).

Trigger turun di bawah ambang saat motor mencapai steady-state; hanya sampel
pasca-falling-edge yang dipakai agar pelatihan tak terbias dinamika transien.
"""
~~~~

## src/training/callbacks.py

Lines 317-319:

~~~~text
        # UJI A: pada regresi tidak ada plafon ln(n) maupun chance 1/n, sehingga
        # seluruh blok "alignment %" di bawah tidak terdefinisi. Yang menggantikannya
        # sebagai tanda patologi adalah korelasi(D(z), u) — A-P2 gugur bila < -0,5.
~~~~
